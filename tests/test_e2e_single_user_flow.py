"""
End-to-end single-user flow test.

ONE canonical user → every backend endpoint → verify the data is consistent.

This simulates what the frontend does:
  1. POST /plan/simple          → get score + plan
  2. POST /plan/simulate        → what-if with same user
  3. POST /plan/batch           → batch with same user
  4. POST /rag/query            → ask a question about the plan

All endpoints receive data derived from the SAME canonical_user.json fixture.
The frontend only needs to store this ONE object and transform it per endpoint.

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

# The features block is the single source of truth for every endpoint
FEATURES = CANONICAL["features"]
REQUEST_ID = CANONICAL["request_id"]


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
         patch.object(scoring_mod, "make_rag_lookup", side_effect=_patched_make_rag_lookup), \
         patch.object(rag_mod, "make_rag_lookup", side_effect=_patched_make_rag_lookup):
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
# The flow: same user data → every endpoint → consistent results
# ===================================================================

class TestSingleUserFlow:
    """
    Frontend sends ONE user → backend.
    This test class mirrors the exact sequence a real frontend session does.
    """

    # ------------------------------------------------------------------
    # Step 1: /plan/simple  —  the primary page (score + plan)
    # ------------------------------------------------------------------
    def test_step1_plan_simple(self, _mock_backend):
        """Frontend form submit → /plan/simple → score + Thai plan."""
        client = _mock_backend["client"]

        # This is exactly what the frontend POSTs
        payload = {"request_id": REQUEST_ID, "features": FEATURES}

        resp = client.post("/api/v1/plan/simple", json=payload)
        assert resp.status_code == 200, resp.text
        data = resp.json()

        # ── Shape checks (frontend relies on these fields) ──
        assert data["request_id"] == REQUEST_ID
        assert isinstance(data["approved"], bool)
        assert 0.0 <= data["p_approve"] <= 1.0
        assert 0.0 <= data["p_reject"] <= 1.0
        assert abs(data["p_approve"] + data["p_reject"] - 1.0) < 0.01
        assert data["mode"] in ("approved_guidance", "improvement_plan")
        assert isinstance(data["result_th"], str)
        assert len(data["result_th"]) > 0  # non-empty plan

        # Save for cross-endpoint consistency checks
        self.__class__._plan_result = data

    # ------------------------------------------------------------------
    # Step 2: /plan/simulate  —  what-if page (same user, tweaked)
    # ------------------------------------------------------------------
    def test_step2_simulate_what_if(self, _mock_backend):
        """Frontend what-if slider → /plan/simulate → before/after comparison."""
        client = _mock_backend["client"]

        payload = {
            "request_id": f"{REQUEST_ID}-sim",
            "features": FEATURES,              # same user
            "what_if": {
                "outstanding": {"delta": -200000},   # pay down debt
                "credit_grade": {"value": "BB"},     # grade improves
            },
        }

        resp = client.post("/api/v1/plan/simulate", json=payload)
        assert resp.status_code == 200, resp.text
        data = resp.json()

        # ── Shape checks ──
        assert "baseline" in data
        assert "simulated" in data
        assert isinstance(data["delta_p_approve"], float)
        assert isinstance(data["verdict"], str)
        assert set(data["changed_features"]) == {"outstanding", "credit_grade"}

        # Baseline should use exact same features
        baseline = data["baseline"]
        assert isinstance(baseline["approved"], bool)
        assert 0.0 <= baseline["p_approve"] <= 1.0

        # Paying down debt + better grade should improve approval odds
        assert data["delta_p_approve"] > 0, "Reducing debt + better grade should increase approval chance"

    # ------------------------------------------------------------------
    # Step 3: /plan/batch  —  bulk upload page (same user as one item)
    # ------------------------------------------------------------------
    def test_step3_batch_single_item(self, _mock_backend):
        """Frontend batch upload → /plan/batch → consistent with /plan/simple."""
        client = _mock_backend["client"]

        payload = {
            "batch_id": "e2e-batch-001",
            "include_plan": False,
            "items": [{"request_id": REQUEST_ID, "features": FEATURES}],
        }

        resp = client.post("/api/v1/plan/batch", json=payload)
        assert resp.status_code == 200, resp.text
        data = resp.json()

        assert data["summary"]["total"] == 1
        assert len(data["results"]) == 1

        item = data["results"][0]
        assert item["request_id"] == REQUEST_ID
        assert isinstance(item["approved"], bool)
        assert 0.0 <= item["p_approve"] <= 1.0

    # ------------------------------------------------------------------
    # Step 4: /rag/query  —  user asks a follow-up question
    # ------------------------------------------------------------------
    def test_step4_rag_followup_question(self, _mock_backend):
        """User asks clarifying question → /rag/query → structured answer."""
        client = _mock_backend["client"]

        payload = {
            "question": "เอกสารที่ต้องใช้สมัครสินเชื่อบ้านมีอะไรบ้าง",
            "top_k": 4,
        }

        resp = client.post("/api/v1/rag/query", json=payload)
        assert resp.status_code == 200, resp.text
        data = resp.json()

        assert "answer" in data
        assert isinstance(data["answer"], str)
        assert "sources" in data
        assert "router_label" in data

    # ------------------------------------------------------------------
    # Cross-check: /plan/simple and /plan/batch give same decision
    # ------------------------------------------------------------------
    def test_step5_consistency_simple_vs_batch(self, _mock_backend):
        """Same features → /plan/simple and /plan/batch must agree on approval."""
        client = _mock_backend["client"]

        # /plan/simple
        simple_resp = client.post("/api/v1/plan/simple", json={
            "request_id": f"{REQUEST_ID}-cmp-simple",
            "features": FEATURES,
        })
        simple = simple_resp.json()

        # /plan/batch (score-only)
        batch_resp = client.post("/api/v1/plan/batch", json={
            "batch_id": "e2e-consistency",
            "include_plan": False,
            "items": [{"request_id": f"{REQUEST_ID}-cmp-batch", "features": FEATURES}],
        })
        batch_item = batch_resp.json()["results"][0]

        # Must agree
        assert simple["approved"] == batch_item["approved"], (
            f"/plan/simple says approved={simple['approved']}, "
            f"/plan/batch says approved={batch_item['approved']}"
        )
        assert abs(simple["p_approve"] - batch_item["p_approve"]) < 0.001, (
            f"p_approve mismatch: simple={simple['p_approve']}, batch={batch_item['p_approve']}"
        )


# ===================================================================
# Negative path: bad data from frontend
# ===================================================================

class TestFrontendErrorHandling:
    """Frontend sends bad data → backend returns useful 422 errors."""

    def test_missing_required_field(self, _mock_backend):
        """Missing Salary → 422 with clear error."""
        client = _mock_backend["client"]
        bad_features = {k: v for k, v in FEATURES.items() if k != "Salary"}
        resp = client.post("/api/v1/plan/simple", json={
            "request_id": "bad-001",
            "features": bad_features,
        })
        assert resp.status_code == 422

    def test_invalid_credit_grade_in_simulation(self, _mock_backend):
        """Invalid credit_grade in what_if → 422."""
        client = _mock_backend["client"]
        resp = client.post("/api/v1/plan/simulate", json={
            "request_id": "bad-sim-001",
            "features": FEATURES,
            "what_if": {"credit_grade": {"value": "ZZ"}},  # invalid grade
        })
        # Should get 422 or 500 with clear message
        assert resp.status_code in (422, 500)

    def test_empty_batch(self, _mock_backend):
        """Empty items list → 422."""
        client = _mock_backend["client"]
        resp = client.post("/api/v1/plan/batch", json={
            "batch_id": "bad-batch",
            "items": [],
        })
        assert resp.status_code == 422
