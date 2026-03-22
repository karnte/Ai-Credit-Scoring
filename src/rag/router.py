"""Rule-first router and metadata gating helpers for Thai home-loan RAG."""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, Iterable, List, Optional

from llama_index.core.settings import Settings
from llama_index.core.vector_stores.types import (
    FilterCondition,
    MetadataFilter,
    MetadataFilters,
)

logger = logging.getLogger(__name__)

ROUTE_LABELS = (
    "policy_requirement",
    "interest_structure",
    "fee_structure",
    "refinance",
    "hardship_support",
    "general_info",
)

ROUTE_KEYWORDS: Dict[str, List[str]] = {
    "interest_structure": ["ดอกเบี้ย", "mrr", "fixed", "floating", "%", "ปีแรก"],
    "fee_structure": ["ค่าธรรมเนียม", "ค่าปรับ", "ปิดบัญชี", "ปิดก่อนกำหนด", "จดจำนอง", "ค่าใช้จ่าย", "ประกันอัคคีภัย", "เบี้ยประกัน"],
    "refinance": ["รีไฟแนนซ์", "บ้านแลกเงิน", "mortgage power", "refinance"],
    "hardship_support": ["ผ่อนไม่ไหว", "ปรับโครงสร้างหนี้", "พักชำระ", "พักชำระดอกเบี้ย", "พักชำระหนี้", "มาตรการ", "โควิด", "น้ำท่วม", "ขยายระยะเวลา", "ขยายสัญญา", "ขยายเวลา"],
    "policy_requirement": ["คุณสมบัติ", "เอกสาร", "รายได้ขั้นต่ำ", "อาชีพ", "สัญชาติ", "เงื่อนไข"],
}

ROUTE_PRIORITY = [
    "fee_structure",
    "interest_structure",
    "hardship_support",
    "refinance",
    "policy_requirement",
]

# Queries containing these terms are off-domain / adversarial — force general_info
# so the validator returns NO_ANSWER instead of leaking policy docs.
SAFETY_BLOCKLIST: List[str] = [
    "ปลอมแปลง", "ปลอม", "ทุจริต", "ฉ้อโกง", "แอบอ้าง", "โกง",
    "หลบเลี่ยง", "เลี่ยง", "ซ่อน", "ซ่อนเงิน", "ฟอกเงิน",
    "forgery", "fraud", "fake", "falsify", "launder",
]


def _contains_any(text: str, terms: Iterable[str]) -> bool:
    return any(term in text for term in terms)


def _normalize_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set)):
        return " ".join(str(item) for item in value).lower()
    return str(value).lower()


def _topic_has(metadata: Dict[str, Any], tag: str) -> bool:
    topic_tags = _normalize_value(metadata.get("topic_tags"))
    return tag in topic_tags


def _route_by_keywords(question: str) -> Optional[str]:
    text = (question or "").lower().strip()
    if not text:
        return "general_info"

    scores = {
        label: sum(1 for keyword in keywords if keyword in text)
        for label, keywords in ROUTE_KEYWORDS.items()
    }
    best = max(scores.values()) if scores else 0
    if best <= 0:
        return None

    tied = {label for label, score in scores.items() if score == best}
    for label in ROUTE_PRIORITY:
        if label in tied:
            return label
    return "general_info"


def _route_by_llm(question: str) -> Optional[str]:
    llm = getattr(Settings, "llm", None)
    if llm is None:
        return None

    prompt = f"""
Classify this user question into exactly one label:
- policy_requirement
- interest_structure
- fee_structure
- refinance
- hardship_support
- general_info

Return only the label.
Question: {question}
""".strip()

    try:
        raw = llm.complete(prompt)
        candidate = str(raw).strip().lower()
        match = re.search(
            r"(policy_requirement|interest_structure|fee_structure|refinance|hardship_support|general_info)",
            candidate,
        )
        if match:
            label = match.group(1)
            if label in ROUTE_LABELS:
                return label
    except Exception as exc:  # pragma: no cover
        logger.debug("LLM router fallback failed: %s", exc)

    return None


def route_query(question: str) -> str:
    """Route a question to one domain label using rules first, LLM optional fallback."""
    text = (question or "").lower()
    if _contains_any(text, SAFETY_BLOCKLIST):
        logger.debug("Safety blocklist matched — routing to general_info: %r", question)
        return "general_info"

    routed = _route_by_keywords(question)
    if routed:
        return routed

    routed = _route_by_llm(question)
    if routed:
        return routed

    return "general_info"


def metadata_matches_route(metadata: Dict[str, Any], router_label: str) -> bool:
    """Python-side metadata route matching used during stage-2 filtering."""
    if router_label == "general_info":
        return True

    category = _normalize_value(metadata.get("category"))
    doc_kind = _normalize_value(metadata.get("doc_kind"))
    title = _normalize_value(metadata.get("title"))
    file_name = _normalize_value(metadata.get("file_name"))
    topic = _normalize_value(metadata.get("topic"))
    joined = " ".join([title, file_name, category, topic])

    if router_label == "interest_structure":
        return (
            category == "interest_structure"
            or doc_kind == "rate_sheet"
            or topic == "interest_rate"
            or _topic_has(metadata, "interest")
            or _contains_any(joined, ("ดอกเบี้ย", "mrr", "fixed", "floating", "%"))
        )

    if router_label == "fee_structure":
        return (
            category == "fee_structure"
            or topic == "fees"
            or _topic_has(metadata, "fee")
            or _contains_any(joined, ("ค่าธรรมเนียม", "ค่าปรับ", "จดจำนอง", "ปิดสินเชื่อ", "fee", "charge"))
        )

    if router_label == "refinance":
        return (
            category == "refinance"
            or topic == "refinance"
            or _topic_has(metadata, "refinance")
            or _contains_any(joined, ("รีไฟแนนซ์", "บ้านแลกเงิน", "refinance", "mortgage power"))
        )

    if router_label == "hardship_support":
        return (
            category in {"hardship_support", "consumer_guideline", "relief"}
            or topic == "hardship_relief"
            or _topic_has(metadata, "relief")
            or _contains_any(joined, ("ผ่อนไม่ไหว", "ปรับโครงสร้างหนี้", "พักชำระ", "มาตรการ", "โควิด", "น้ำท่วม"))
        )

    if router_label == "policy_requirement":
        return (
            category == "policy_requirement"
            or (doc_kind == "policy" and _contains_any(joined, ("คุณสมบัติ", "เอกสาร", "รายได้ขั้นต่ำ", "เงื่อนไข", "eligibility")))
        )

    return True


def _eq_filter(key: str, value: str) -> MetadataFilter:
    return MetadataFilter(key=key, value=value)


def build_metadata_filters(router_label: str) -> Optional[MetadataFilters]:
    """Build best-effort metadata filters for vector stores (mainly Chroma)."""
    if router_label == "interest_structure":
        return MetadataFilters(filters=[_eq_filter("doc_kind", "rate_sheet")])

    if router_label == "policy_requirement":
        return MetadataFilters(filters=[_eq_filter("doc_kind", "policy")])

    if router_label == "fee_structure":
        return MetadataFilters(filters=[_eq_filter("category", "fee_structure")])

    if router_label == "refinance":
        return MetadataFilters(filters=[_eq_filter("category", "refinance")])

    if router_label == "hardship_support":
        return MetadataFilters(
            filters=[
                _eq_filter("category", "hardship_support"),
                _eq_filter("category", "consumer_guideline"),
                _eq_filter("category", "policy_requirement"),
            ],
            condition=FilterCondition.OR,
        )

    return None
