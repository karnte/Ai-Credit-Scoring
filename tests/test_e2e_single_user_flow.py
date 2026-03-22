"""
End-to-end single-user flow test.

ONE canonical user → /predict (what the frontend actually calls) → verify
the full pipeline: input parsing → scoring → planner → RAG → response.

Also tests consistency: the same user sent via /predict and /plan/simple
must produce the same decision.

Run:
    pytest tests/test_e2e_single_user_flow.py -v
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# ---------------------------------------------------------------------------
# Load the single canonical user — this is your ONE data source
# ---------------------------------------------------------------------------

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "canonical_user.json"
CANONICAL = json.loads(FIXTURE_PATH.read_text())

FRONTEND_PAYLOAD = CANONICAL["frontend_payload"]
FEATURES = CANONICAL["features"]
EXPECTED = CANONICAL["expected"]


# ---------------------------------------------------------------------------
# Shared fixture: mock RAG so no vector index / LLM is needed
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _mock_backend():
    """Patch RAG manager so tests run without Ollama/ChromaDB/llama_index."""
    mock_manager = MagicMock()
    mock_manager.query.side_effect = lambda question, **kw: {
        "answer": f"Mock answer for: {question[:40]}",
        "sources": [{"metadata": {"title": "test-doc", "category": "general_info"}, "score": 0.85}],
        "router_label": "general_info",
        "question": question,
        "retrieved_node_count": 1,
        "validated_node_count": 1,
    }

    from src.api.main import app
    import src.api.routes.scoring as scoring_mod
    import src.api.routes.rag as rag_mod
    import src.api.routes.predict as predict_mod

    def _patched_make_rag_lookup(query_fn, use_cache=True):
        def rag_lookup(query: str) -> dict:
            try:
                result = query_fn(query)
                return {"answer": result.get("answer", ""), "sources": result.get("sources", [])}
            except Exception:
                return {"answer": "", "sources": []}
        return rag_lookup

    mock_cache = MagicMock()
    mock_cache.get.return_value = None
    mock_cache.stats.return_value = {"size": 0, "hit_rate": 0.0}
    mock_cache.clear.return_value = 0
    mock_cache_module = MagicMock(get_cache=MagicMock(return_value=mock_cache))

    saved_rag = sys.modules.get("src.rag")
    saved_rag_cache = sys.modules.get("src.rag.cache")
    if "src.rag" not in sys.modules:
        sys.modules["src.rag"] = MagicMock()
    sys.modules["src.rag.cache"] = mock_cache_module

    with patch.object(scoring_mod, "get_rag_manager", return_value=mock_manager), \
         patch.object(rag_mod, "get_rag_manager", return_value=mock_manager), \
         patch.object(predict_mod, "get_rag_manager", return_value=mock_manager), \
         patch.object(scoring_mod, "make_rag_lookup", side_effect=_patched_make_rag_lookup), \
         patch.object(rag_mod, "make_rag_lookup", side_effect=_patched_make_rag_lookup), \
         patch.object(predict_mod, "make_rag_lookup", side_effect=_patched_make_rag_lookup):
        yield {"app": app, "client": TestClient(app), "mock_manager": mock_manager}

    if saved_rag is not None:
        sys.modules["src.rag"] = saved_rag
    elif "src.rag" in sys.modules:
        del sys.modules["src.rag"]
    if saved_rag_cache is not None:
        sys.modules["src.rag.cache"] = saved_rag_cache
    elif "src.rag.cache" in sys.modules:
        del sys.modules["src.rag.cache"]


# ===================================================================
# Step 1: /predict — the ONLY endpoint the frontend calls
# ===================================================================

class TestPredictEndpoint:
    """Tests for the /predict endpoint using the canonical user fixture."""

    def test_predict_returns_expected_shape(self, _mock_backend):
        """POST /predict → response has prediction, confidence, shap_values, explanation."""
        client = _mock_backend["client"]

        resp = client.post("/api/v1/predict", json=FRONTEND_PAYLOAD)
        assert resp.status_code == 200, resp.text
        data = resp.json()

        # Shape the frontend relies on
        assert "prediction" in data
        assert "confidence" in data
        assert "shap_values" in data
        assert "explanation" in data

        assert data["prediction"] in ("approved", "rejected")
        assert 0.0 <= data["confidence"] <= 1.0
        assert isinstance(data["shap_values"], dict)
        assert isinstance(data["explanation"], str)

    def test_predict_matches_expected_decision(self, _mock_backend):
        """Canonical user (weak profile) should be rejected."""
        client = _mock_backend["client"]

        resp = client.post("/api/v1/predict", json=FRONTEND_PAYLOAD)
        data = resp.json()

        assert data["prediction"] == EXPECTED["prediction"]
        assert data["confidence"] >= EXPECTED["confidence_min"]
        assert data["confidence"] <= EXPECTED["confidence_max"]

    def test_predict_returns_shap_for_key_features(self, _mock_backend):
        """SHAP values must include the features the frontend displays."""
        client = _mock_backend["client"]

        resp = client.post("/api/v1/predict", json=FRONTEND_PAYLOAD)
        data = resp.json()

        for key in EXPECTED["required_shap_keys"]:
            assert key in data["shap_values"], f"Missing SHAP key: {key}"

    def test_predict_returns_explanation(self, _mock_backend):
        """Explanation (Thai plan) should be non-empty for rejected applicant."""
        client = _mock_backend["client"]

        resp = client.post("/api/v1/predict", json=FRONTEND_PAYLOAD)
        data = resp.json()

        if data["prediction"] == "rejected":
            assert len(data["explanation"]) > 0, "Rejected applicant should get an explanation"

    def test_predict_with_extra_features_override(self, _mock_backend):
        """extra_features can override values from input_text."""
        client = _mock_backend["client"]

        # Override Salary to a high value via extra_features
        payload = {
            "input_text": FRONTEND_PAYLOAD["input_text"],
            "extra_features": {"Salary": 150000.0, "credit_score": 780.0, "credit_grade": "AA", "outstanding": 0, "overdue": 0},
        }
        resp = client.post("/api/v1/predict", json=payload)
        assert resp.status_code == 200
        data = resp.json()

        # Strong override should flip to approved
        assert data["prediction"] == "approved"
        assert data["confidence"] > 0.5

    def test_predict_with_extra_features_partial(self, _mock_backend):
        """extra_features can add just one field without breaking the rest."""
        client = _mock_backend["client"]

        payload = {
            "input_text": FRONTEND_PAYLOAD["input_text"],
            "extra_features": {"Interest_rate": 3.5},
        }
        resp = client.post("/api/v1/predict", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["prediction"] in ("approved", "rejected")


# ===================================================================
# Step 2: Input parsing — verify structured text is parsed correctly
# ===================================================================

class TestInputParsing:
    """Verify the input_text parser handles various formats."""

    def test_parse_standard_format(self, _mock_backend):
        """Standard comma-separated key=value pairs."""
        client = _mock_backend["client"]

        payload = {
            "input_text": "Salary=50000, credit_score=700, loan_amount=1000000, loan_term=20",
            "extra_features": {},
        }
        resp = client.post("/api/v1/predict", json=payload)
        assert resp.status_code == 200

    def test_parse_semicolons(self, _mock_backend):
        """Semicolon-separated pairs."""
        client = _mock_backend["client"]

        payload = {
            "input_text": "Salary=50000; credit_score=700; loan_amount=1000000; loan_term=20",
            "extra_features": {},
        }
        resp = client.post("/api/v1/predict", json=payload)
        assert resp.status_code == 200

    def test_parse_aliases(self, _mock_backend):
        """Lowercase aliases (salary → Salary, coapplicant → Coapplicant)."""
        client = _mock_backend["client"]

        payload = {
            "input_text": "salary=50000, credit_score=700, loan_amount=1000000, loan_term=20, coapplicant=true",
            "extra_features": {},
        }
        resp = client.post("/api/v1/predict", json=payload)
        assert resp.status_code == 200

    def test_missing_required_field(self, _mock_backend):
        """Missing Salary and credit_score → 422."""
        client = _mock_backend["client"]

        payload = {
            "input_text": "loan_amount=1000000, loan_term=20",
            "extra_features": {},
        }
        resp = client.post("/api/v1/predict", json=payload)
        assert resp.status_code == 422

    def test_empty_input_text(self, _mock_backend):
        """Empty string → 422 (no features parsed)."""
        client = _mock_backend["client"]

        resp = client.post("/api/v1/predict", json={"input_text": "", "extra_features": {}})
        assert resp.status_code == 422


# ===================================================================
# Step 3: Consistency — /predict and /plan/simple agree
# ===================================================================

class TestCrossEndpointConsistency:
    """Same user data via different endpoints must produce the same decision."""

    def test_predict_vs_plan_simple_same_decision(self, _mock_backend):
        """/predict and /plan/simple with same features → same approval decision."""
        client = _mock_backend["client"]

        # /predict (frontend format)
        predict_resp = client.post("/api/v1/predict", json=FRONTEND_PAYLOAD)
        predict_data = predict_resp.json()

        # /plan/simple (internal format, same features)
        # Remove _comment key from features
        clean_features = {k: v for k, v in FEATURES.items() if k != "_comment"}
        simple_resp = client.post("/api/v1/plan/simple", json={
            "request_id": "consistency-check",
            "features": clean_features,
        })
        simple_data = simple_resp.json()

        # Decisions must match
        predict_approved = predict_data["prediction"] == "approved"
        assert predict_approved == simple_data["approved"], (
            f"/predict says {predict_data['prediction']}, "
            f"/plan/simple says approved={simple_data['approved']}"
        )

        # Confidence should match p_approve
        assert abs(predict_data["confidence"] - simple_data["p_approve"]) < 0.001, (
            f"confidence mismatch: predict={predict_data['confidence']}, "
            f"simple p_approve={simple_data['p_approve']}"
        )

    def test_predict_vs_batch_same_decision(self, _mock_backend):
        """/predict and /plan/batch with same features → same approval decision."""
        client = _mock_backend["client"]

        predict_resp = client.post("/api/v1/predict", json=FRONTEND_PAYLOAD)
        predict_data = predict_resp.json()

        clean_features = {k: v for k, v in FEATURES.items() if k != "_comment"}
        batch_resp = client.post("/api/v1/plan/batch", json={
            "batch_id": "consistency-batch",
            "include_plan": False,
            "items": [{"request_id": "batch-cmp", "features": clean_features}],
        })
        batch_item = batch_resp.json()["results"][0]

        predict_approved = predict_data["prediction"] == "approved"
        assert predict_approved == batch_item["approved"]
        assert abs(predict_data["confidence"] - batch_item["p_approve"]) < 0.001


# ===================================================================
# Step 4: Full dataflow trace through /predict
# ===================================================================

class TestPredictDataflow:
    """Verify /predict actually invokes scoring → planner → RAG."""

    def test_predict_calls_rag_for_rejected(self, _mock_backend):
        """Rejected applicant → planner queries RAG for improvement actions."""
        client = _mock_backend["client"]
        mock_manager = _mock_backend["mock_manager"]

        resp = client.post("/api/v1/predict", json=FRONTEND_PAYLOAD)
        data = resp.json()

        if data["prediction"] == "rejected":
            # RAG should have been consulted for the improvement plan
            assert mock_manager.query.call_count > 0, (
                "Rejected applicant should trigger RAG queries for improvement plan"
            )

    def test_predict_approved_still_returns_explanation(self, _mock_backend):
        """Even approved applicants get an explanation (approval checklist)."""
        client = _mock_backend["client"]

        payload = {
            "input_text": "Salary=150000, credit_score=800, credit_grade=AA, outstanding=0, overdue=0, loan_amount=1000000, loan_term=15",
            "extra_features": {},
        }
        resp = client.post("/api/v1/predict", json=payload)
        data = resp.json()

        assert data["prediction"] == "approved"
        assert isinstance(data["explanation"], str)
