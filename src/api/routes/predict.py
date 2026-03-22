"""
/predict endpoint — bridges the frontend's simple contract to the internal
scoring + planner + RAG pipeline.

Frontend sends:
    {"input_text": "Salary=35000, credit_score=520, ...", "extra_features": {...}}

Backend returns:
    {"prediction": "approved"|"rejected", "confidence": float, "shap_values": {...}, "explanation": str}
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict

from fastapi import APIRouter, HTTPException

from src.api.schemas.payload import PredictRequest, PredictResponse, UserInputFeatures
from src.planner.planning import generate_response
from src.planner.rag_bridge import get_rag_manager, make_rag_lookup
from src.planner.scoring import compute_plan_inputs

router = APIRouter()
logger = logging.getLogger(__name__)

# Feature name aliases: frontend might send slightly different names
_ALIASES = {
    "salary": "Salary",
    "occupation": "Occupation",
    "marriage_status": "Marriage_Status",
    "marital_status": "Marriage_Status",
    "coapplicant": "Coapplicant",
    "co_applicant": "Coapplicant",
    "interest_rate": "Interest_rate",
}

# Defaults for optional fields so UserInputFeatures validation passes
_DEFAULTS: Dict[str, Any] = {
    "Occupation": "Unknown",
    "Marriage_Status": "Unknown",
    "credit_grade": "CC",
    "outstanding": 0.0,
    "overdue": 0.0,
    "Coapplicant": False,
    "Interest_rate": None,
}


def _parse_input_text(text: str) -> Dict[str, Any]:
    """Parse 'key=value, key=value, ...' into a dict.

    Handles:
      - comma, semicolon, or newline as separators
      - quoted and unquoted values
      - whitespace around keys/values
    """
    features: Dict[str, Any] = {}
    # Split on comma, semicolon, or newline
    pairs = re.split(r"[,;\n]+", text)
    for pair in pairs:
        pair = pair.strip()
        if "=" not in pair:
            continue
        key, _, raw_value = pair.partition("=")
        key = key.strip()
        raw_value = raw_value.strip().strip("'\"")

        # Normalize key via alias map
        normalized_key = _ALIASES.get(key.lower(), key)

        # Try to coerce value to number or bool
        value: Any = raw_value
        if raw_value.lower() in ("true", "yes", "1"):
            value = True
        elif raw_value.lower() in ("false", "no", "0"):
            value = False
        else:
            try:
                value = float(raw_value)
                if value == int(value) and "." not in raw_value:
                    value = int(value)
            except ValueError:
                pass  # keep as string

        features[normalized_key] = value
    return features


@router.post("/predict", response_model=PredictResponse)
async def predict(payload: PredictRequest):
    """
    Frontend-facing prediction endpoint.

    1. Parses ``input_text`` into feature key-value pairs.
    2. Merges with ``extra_features`` (extra_features wins on conflict).
    3. Runs internal scoring model (risk probability + SHAP).
    4. Generates Thai-language explanation via planner + RAG.
    5. Returns the shape the frontend expects.
    """
    # ── Step 1+2: Parse and merge features ────────────────────────────────
    parsed = _parse_input_text(payload.input_text)
    if payload.extra_features:
        for k, v in payload.extra_features.items():
            normalized_key = _ALIASES.get(k.lower(), k)
            parsed[normalized_key] = v

    # Apply defaults for missing optional fields
    for key, default in _DEFAULTS.items():
        if key not in parsed:
            parsed[key] = default

    # ── Step 3: Validate features ─────────────────────────────────────────
    try:
        features = UserInputFeatures(**parsed)
    except Exception as exc:
        logger.warning("Feature validation failed: %s | parsed=%s", exc, parsed)
        raise HTTPException(
            status_code=422,
            detail=f"Could not parse features from input_text. Missing or invalid fields: {exc}",
        )

    # ── Step 4: Compute score + SHAP ──────────────────────────────────────
    try:
        user_input, shap_json, risk_prob = compute_plan_inputs(features)
    except Exception as exc:
        logger.error("Scoring failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Scoring failed: {exc}")

    approved = risk_prob < 0.50
    p_approve = round(1.0 - risk_prob, 4)

    # ── Step 5: Generate explanation via planner + RAG ─────────────────────
    explanation = ""
    try:
        manager = get_rag_manager()
        rag_lookup = make_rag_lookup(manager.query) if manager else None

        model_output = {
            "prediction": 1 if approved else 0,
            "probabilities": {"1": p_approve, "0": round(risk_prob, 4)},
        }
        plan_result = generate_response(
            user_input, model_output, shap_json, rag_lookup=rag_lookup,
        )
        explanation = plan_result.get("result_th", "")
    except Exception as exc:
        logger.warning("Planner/RAG failed (non-fatal): %s", exc)

    return PredictResponse(
        prediction="approved" if approved else "rejected",
        confidence=p_approve,
        shap_values=shap_json.get("values", {}),
        explanation=explanation,
    )
