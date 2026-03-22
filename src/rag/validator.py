"""Post-retrieval relevance validator for Thai home-loan RAG."""

from __future__ import annotations

from typing import Any, Iterable, List

NO_ANSWER_MESSAGE = "ไม่พบข้อมูลในเอกสารที่มีอยู่"
CLOSE_ACCOUNT_CLARIFICATION_MESSAGE = (
    "หมายถึงค่าปิดสินเชื่อก่อนกำหนด (prepayment) หรือค่าปิดบัญชีเงินฝาก/บัญชีธนาคาร?"
)

HOME_DOMAIN_KEYWORDS = (
    "สินเชื่อบ้าน",
    "รีไฟแนนซ์",
    "จำนอง",
    "บ้านแลกเงิน",
    "mortgage",
    "home loan",
    "refinance",
)

GLOBAL_BLOCKLIST = (
    "เงินฝาก",
    "บัญชีเงินฝาก",
    "fx",
    "เงินตราต่างประเทศ",
    "บัตรเครดิต",
    "เดบิต",
    "โอนเงินต่างประเทศ",
    "พร้อมเพย์",
    "ndid",
    "กรมสรรพากร",
    "ภาษี",
)

EXPLICIT_ALLOW_TERMS = (
    "เงินฝาก",
    "บัตร",
    "fx",
    "ndid",
    "พร้อมเพย์",
    "กรมสรรพากร",
    "ภาษี",
)

DOC_KIND_ALLOW = {"policy", "rate_sheet", "form"}
PREPAYMENT_HINTS = ("ก่อน 5 ปี", "1% ของวงเงินกู้", "ค่าปรับ")

ROUTE_MUST_HAVE = {
    "policy_requirement": ("คุณสมบัติ", "เอกสาร", "รายได้ขั้นต่ำ", "เงื่อนไข", "eligibility", "requirement"),
    "interest_structure": ("ดอกเบี้ย", "rate", "mrr", "fixed", "floating", "%"),
    "fee_structure": ("ค่าธรรมเนียม", "ค่าปรับ", "ค่าจดจำนอง", "จดจำนอง", "ปิดสินเชื่อ", "ปิดบัญชี", "fee", "charge", "ประกันอัคคีภัย", "เบี้ยประกัน"),
    "refinance": ("รีไฟแนนซ์", "สินเชื่อบ้าน", "บ้านแลกเงิน", "mortgage", "home loan"),
    "hardship_support": ("มาตรการ", "พักชำระ", "ผ่อนไม่ไหว", "ปรับโครงสร้างหนี้", "โควิด", "น้ำท่วม", "relief", "ขยายระยะเวลา", "ขยายสัญญา", "ขยายเวลา"),
}

ROUTE_BLOCKLIST = {
    "refinance": ("ndid", "พร้อมเพย์", "กรมสรรพากร", "ภาษี"),
    "interest_structure": ("ndid", "พร้อมเพย์", "กรมสรรพากร", "ภาษี"),
    "fee_structure": ("ndid", "พร้อมเพย์", "กรมสรรพากร", "ภาษี"),
}


def _safe_score(node: Any) -> float:
    score = getattr(node, "score", None)
    if score is None:
        return float("-inf")
    try:
        return float(score)
    except (TypeError, ValueError):
        return float("-inf")


def _extract_text(node: Any) -> str:
    text = getattr(node, "text", None)
    if isinstance(text, str):
        return text

    inner_node = getattr(node, "node", None)
    if inner_node is not None:
        inner_text = getattr(inner_node, "text", None)
        if isinstance(inner_text, str):
            return inner_text
        if hasattr(inner_node, "get_content"):
            try:
                return inner_node.get_content()
            except Exception:  # pragma: no cover
                return ""
    return ""


def _extract_metadata(node: Any) -> dict:
    metadata = getattr(node, "metadata", None)
    if isinstance(metadata, dict):
        return metadata

    inner_node = getattr(node, "node", None)
    if inner_node is not None:
        inner_meta = getattr(inner_node, "metadata", None)
        if isinstance(inner_meta, dict):
            return inner_meta
    return {}


def _has_any(text: str, keywords: Iterable[str]) -> bool:
    return any(keyword in text for keyword in keywords)


def _to_search_text(node: Any) -> str:
    metadata = _extract_metadata(node)
    meta_joined = " ".join(
        str(metadata.get(key, ""))
        for key in ("title", "file_name", "category", "doc_kind", "institution", "product_type", "domain", "topic")
    )
    return f"{_extract_text(node)} {meta_joined}".lower()


def _is_cimb_loan_doc(metadata: dict) -> bool:
    institution = str(metadata.get("institution", "")).lower()
    doc_kind = str(metadata.get("doc_kind", "")).lower()
    domain = str(metadata.get("domain", "")).lower()
    product_type = str(metadata.get("product_type", "")).lower()
    category = str(metadata.get("category", "")).lower()

    is_cimb = "cimb" in institution
    doc_kind_allowed = doc_kind in DOC_KIND_ALLOW
    looks_loan = (
        "loan" in domain
        or "home_loan" in product_type
        or category in {"policy_requirement", "interest_structure", "fee_structure", "refinance", "hardship_support"}
    )
    return is_cimb and doc_kind_allowed and looks_loan


def validate_nodes(
    question: str,
    nodes: List[Any],
    router_label: str = "general_info",
) -> List[Any]:
    """
    Validate retrieved nodes to keep only domain-correct home-loan evidence.

    Keep rule:
    - Text has home-loan anchors, OR
    - Metadata says it is a CIMB loan doc (policy/rate_sheet/form).
    """
    q = (question or "").lower()
    question_has_explicit_terms = _has_any(q, EXPLICIT_ALLOW_TERMS)
    label = (router_label or "general_info").strip().lower()

    validated: List[Any] = []
    for node in nodes:
        text = _to_search_text(node)
        if not text.strip():
            continue

        if _has_any(text, GLOBAL_BLOCKLIST) and not question_has_explicit_terms:
            continue

        metadata = _extract_metadata(node)
        has_anchor = _has_any(text, HOME_DOMAIN_KEYWORDS)
        metadata_allow = _is_cimb_loan_doc(metadata)

        if not (has_anchor or metadata_allow):
            continue

        must_have = ROUTE_MUST_HAVE.get(label, ())
        if must_have and not _has_any(text, must_have):
            continue

        route_blocklist = ROUTE_BLOCKLIST.get(label, ())
        if route_blocklist and _has_any(text, route_blocklist) and not question_has_explicit_terms:
            continue

        validated.append(node)

    return sorted(validated, key=_safe_score, reverse=True)


def needs_close_account_clarification(question: str, nodes: List[Any]) -> bool:
    """Detect semantic collision: close account vs loan prepayment penalty."""
    q = (question or "").lower()
    if "ปิดบัญชี" not in q:
        return False

    for node in nodes[:3]:
        text = _to_search_text(node)
        if _has_any(text, PREPAYMENT_HINTS):
            return True
    return False
