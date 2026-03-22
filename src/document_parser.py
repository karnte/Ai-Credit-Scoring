"""Custom document parser for structured documents with metadata."""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from llama_index.core import Document

CLEANING_VERSION = "2026-03-04-v1"

NOISE_TERMS = (
    "ติดต่อเรา",
    "สมัคร",
    "ดาวน์โหลด",
    "cookie",
    "cookies",
    "privacy",
    "นโยบายความเป็นส่วนตัว",
    "ติดตามเรา",
    "สงวนลิขสิทธิ์",
    "copyright",
    "share",
    "related",
    "โปรโมชั่น",
    "ค้นหา",
    "quicklinks",
    "see all",
    "other sites",
    "back",
    "language",
    "countries",
)

LOAN_CONTEXT_TERMS = (
    "สินเชื่อบ้าน",
    "บ้านแลกเงิน",
    "รีไฟแนนซ์",
    "จำนอง",
    "mortgage",
    "home loan",
    "refinance",
)

TABLE_SIGNAL_TERMS = ("mrr", "mlr", "mor", "ltv", "%", "ปีที่")

TOPIC_KEYWORDS = {
    "fee": ("ค่าธรรมเนียม", "fee", "charge", "ปิดสินเชื่อก่อนกำหนด", "ปิดบัญชี"),
    "interest": ("ดอกเบี้ย", "rate", "mrr", "mlr", "mor", "fixed", "floating"),
    "refinance": ("รีไฟแนนซ์", "refinance"),
    "relief": (
        "มาตรการช่วยเหลือ",
        "ช่วยเหลือ",
        "ผ่อนไม่ไหว",
        "ปรับโครงสร้างหนี้",
        "hardship",
        "relief",
        "flood",
        "covid",
    ),
    "document_requirement": (
        "เอกสาร",
        "คุณสมบัติ",
        "เงื่อนไข",
        "requirement",
        "eligibility",
        "application",
        "สมัคร",
    ),
}

CHROME_TOKENS = (
    "search",
    "quicklinks",
    "back",
    "see all",
    "other sites",
    "sitemap",
    "all rights reserved",
    "copyright",
    "ประกาศการเก็บข้อมูลคุกกี้",
    "ประกาศความเป็นส่วนตัว",
    "ติดต่อเรา",
    "สาขาธนาคาร",
    "เกี่ยวกับเรา",
    "ติดตามเรา",
    "นโยบายความเป็นส่วนตัว",
)

NAV_CHROME_KEYWORDS = (
    "search",
    "back",
    "see all",
    "quicklinks",
    "other sites",
    "sitemap",
    "ติดต่อเรา",
    "สาขาธนาคาร",
    "เกี่ยวกับเรา",
    "ผลิตภัณฑ์ธนาคาร",
    "เงินฝาก",
    "บัตร",
    "ประกัน",
    "ประกาศความเป็นส่วนตัว",
    "ประกาศการเก็บข้อมูลคุกกี้",
    "all rights reserved",
    "copyright",
    "cimb thai app",
    "cimb thai connect",
    "พร้อมเพย์",
    "ndid",
    "bizchannel@cimb",
)

DOMAIN_SIGNAL_KEYWORDS = (
    "สินเชื่อบ้าน",
    "รีไฟแนนซ์",
    "บ้านแลกเงิน",
    "จดจำนอง",
    "mortgage",
    "home loan",
    "mrr",
    "ltv",
    "อัตราดอกเบี้ย",
    "ค่าจดจำนอง",
    "มาตรการช่วยเหลือ",
    "พักชำระ",
    "ปรับโครงสร้างหนี้",
    "relief",
    "covid",
    "flood",
)

KEEP_GUARD_TERMS = (
    "mrr",
    "ltv",
    "ดอกเบี้ย",
    "ค่าธรรมเนียม",
    "เงื่อนไข",
    "มีผล",
    "ยกเว้น",
)

POSITIVE_RELEVANCE_KEYWORDS = (
    "สินเชื่อบ้าน",
    "รีไฟแนนซ์",
    "บ้านแลกเงิน",
    "จดจำนอง",
    "mortgage",
    "home loan",
    "mortgage power",
    "property loan",
    "มาตรการช่วยเหลือ",
    "ปรับโครงสร้างหนี้",
    "พักชำระ",
    "relief",
    "hardship",
    "debt restructuring",
    "loan interest",
    "อัตราดอกเบี้ยเงินให้สินเชื่อ",
)

NEGATIVE_RELEVANCE_KEYWORDS = (
    "customer profiling",
    "ndid",
    "debit card",
    "เงินฝาก",
    "fx",
    "โอนเงินต่างประเทศ",
    "บัตรเครดิต",
    "พร้อมเพย์",
)

HOME_LOAN_URL_HINTS = (
    "/home-loan/",
    "/mortgage",
    "/refinance",
    "/home-loan",
    "mortgage-power",
    "property-loan",
    "/products/loans/",
    "/documents/loan/",
    "/loan/homeloan/",
    "/loan/refinance/",
    "/loan/mortgage/",
    "/loan-interest-rates",
    "/special-relief-assistance",
    "/customer-support-measures-covid",
    "/flood-relief",
    "/relief-measures",
)

INSUFFICIENT_BODY_MESSAGE = (
    "ไม่พบเนื้อหาหลักที่เพียงพอหลังการลบเมนูเว็บไซต์ "
    "เอกสารถูกกันออกเพื่อป้องกันการดึงข้อมูลผิดโดเมน"
)

UNRELATED_SUMMARY_MESSAGE = (
    "เอกสารนี้ไม่เกี่ยวข้องกับสินเชื่อบ้าน/รีไฟแนนซ์ในชุดข้อมูลนี้ และถูกกันออกจากการทำดัชนี"
)

WEB_CHROME_QUARANTINE_SUMMARY = (
    "เอกสารนี้มีเนื้อหาเป็นเมนู/ส่วนประกอบเว็บไซต์เป็นส่วนใหญ่และไม่เกี่ยวข้องกับสินเชื่อบ้าน "
    "จึงถูกกันออกจากการทำดัชนี"
)

TOPIC_TO_CATEGORY = {
    "home_loan_policy": "policy_requirement",
    "interest_rate": "interest_structure",
    "fees": "fee_structure",
    "refinance": "refinance",
    "hardship_relief": "hardship_support",
    "unrelated_web_chrome": "unrelated",
    "unrelated": "unrelated",
}


def _normalize_line(line: str) -> str:
    line = line.replace("\u00a0", " ").strip()
    line = re.sub(r"\s+", " ", line)
    return line


def _is_placeholder_line(line: str) -> bool:
    return bool(re.fullmatch(r"[-*•●·|_\.]{1,8}", line))


def _looks_like_policy_line(line: str) -> bool:
    lower = line.lower()
    if re.search(r"\d", line):
        return True
    if "%" in line:
        return True
    return any(term in lower for term in ("สินเชื่อ", "ดอกเบี้ย", "เงื่อนไข", "mrr", "mlr", "mor"))


def _is_noise_line(line: str) -> bool:
    if not line:
        return True

    lower = line.lower()
    if any(term in lower for term in NOISE_TERMS):
        # Keep long paragraphs that may mention privacy/cookies in legitimate policy context.
        return len(line) <= 140

    if line.startswith("©"):
        return True

    if re.fullmatch(r"[A-Za-z ]{1,20}", line):
        token_count = len(line.split())
        if token_count <= 3 and not _looks_like_policy_line(line):
            return True

    return False


def _remove_repeated_short_lines(lines: List[str]) -> List[str]:
    short_keys = [
        re.sub(r"\s+", " ", line.lower())
        for line in lines
        if line and len(line) <= 60
    ]
    counts = Counter(short_keys)
    repeated = {k for k, count in counts.items() if count >= 3}

    cleaned: List[str] = []
    for line in lines:
        key = re.sub(r"\s+", " ", line.lower())
        if key in repeated and len(line) <= 60 and not _looks_like_policy_line(line):
            continue
        cleaned.append(line)
    return cleaned


def _is_table_like_line(line: str) -> bool:
    lower = line.lower()
    signal_hits = sum(1 for term in TABLE_SIGNAL_TERMS if term in lower)

    has_columns = bool(re.search(r"\s{2,}|\t|\|", line))
    has_year_row = bool(re.match(r"^(ปีที่\s*\d+|\d+\s*(ปี|yr|year))", lower))
    has_ratio = bool(re.search(r"(<=|>=|<|>)\s*\d+%|\d+\s*%", line))

    return has_year_row or has_ratio or (signal_hits >= 2 and has_columns)


def _rowify_table_line(line: str) -> Tuple[str, bool]:
    original = line
    line = re.sub(r"\t+", " | ", line)
    line = re.sub(r"\s{3,}", " | ", line)

    cells = [part.strip() for part in re.split(r"\s*\|\s*", line) if part.strip()]
    changed = False

    if len(cells) >= 3:
        head = cells[0]
        tail = " | ".join(cells[1:])

        if re.match(r"^ปีที่\s*\d+", head):
            line = f"{head}: {tail}"
            changed = True
        elif re.match(r"^ltv\s*(<=|>=|<|>)?\s*\d+%", head, flags=re.IGNORECASE):
            line = f"{head}: {tail}"
            changed = True
        else:
            line = " | ".join(cells)
            changed = original != line
    else:
        year_match = re.search(r"(ปีที่\s*\d+)", line)
        if year_match and ":" not in line:
            line = line.replace(year_match.group(1), f"{year_match.group(1)}:", 1)
            changed = True

        ltv_match = re.search(r"(LTV\s*(?:<=|>=|<|>)?\s*\d+%)", line, flags=re.IGNORECASE)
        if ltv_match and ":" not in line:
            line = line.replace(ltv_match.group(1), f"{ltv_match.group(1)}:", 1)
            changed = True

        if not changed and _is_table_like_line(line) and not line.endswith(";"):
            line = f"{line} ;"
            changed = True

    return _normalize_line(line), changed


def _convert_table_like_lines(lines: List[str]) -> Tuple[List[str], float, float]:
    if not lines:
        return [], 0.0, 0.0

    converted: List[str] = []
    table_line_count = 0
    converted_line_count = 0

    for line in lines:
        if _is_table_like_line(line):
            table_line_count += 1
            new_line, changed = _rowify_table_line(line)
            converted.append(new_line)
            if changed:
                converted_line_count += 1
        else:
            converted.append(line)

    table_likeness_score = table_line_count / max(len(lines), 1)
    row_conversion_score = converted_line_count / max(table_line_count, 1)
    return converted, table_likeness_score, row_conversion_score


def _merge_broken_lines(lines: List[str]) -> List[str]:
    merged: List[str] = []

    for line in lines:
        if not merged:
            merged.append(line)
            continue

        prev = merged[-1]
        if not prev:
            merged[-1] = line
            continue

        starts_new_block = bool(
            re.match(
                r"^(ปีที่\s*\d+|\d+\.|LTV|MRR|MLR|MOR|หมายเหตุ|เงื่อนไข|ข้อกำหนด|หัวข้อ)",
                line,
                flags=re.IGNORECASE,
            )
        )

        should_merge = (
            not starts_new_block
            and not re.search(r"[.!?:;]$", prev)
            and len(prev) < 180
            and not _is_table_like_line(prev)
            and not _is_table_like_line(line)
        )

        if should_merge:
            merged[-1] = _normalize_line(f"{prev} {line}")
        else:
            merged.append(line)

    return merged


def _contains_loan_context(text: str) -> bool:
    lower = text.lower()
    return any(term in lower for term in LOAN_CONTEXT_TERMS)


def _has_any(text: str, keywords: Tuple[str, ...]) -> bool:
    return any(keyword in text for keyword in keywords)


def _has_keep_guard(line: str) -> bool:
    lower = line.lower()
    if re.search(r"\d", line):
        return True
    if "%" in line:
        return True
    if re.search(r"\b(?:19|20|25)\d{2}\b", line):
        return True
    if re.search(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", line):
        return True
    return any(term in lower for term in KEEP_GUARD_TERMS)


def extract_main_body(text: str, title: str = "") -> Tuple[str, Dict[str, object]]:
    """
    Try to isolate article body from scraped page content.

    Heuristic order:
    1) Prefer content after the last "You're viewing:" / "ประกาศ" / repeated title.
    2) Prefer content after a nearby year line around the selected anchor.
    3) If title repeats, use the last title segment.
    4) Fallback to original text.
    """
    if not text:
        return "", {"anchor": "empty", "start_idx": 0}

    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    lower = normalized.lower()

    title_norm = _normalize_line(title) if title else ""
    title_positions: List[int] = []
    if title_norm:
        title_positions = [
            m.start()
            for m in re.finditer(re.escape(title_norm), normalized, flags=re.IGNORECASE)
        ]

    anchor_candidates: List[Tuple[str, int]] = []
    for marker in ("you're viewing:",):
        pos = lower.rfind(marker.lower())
        if pos != -1:
            start = pos + len(marker)
            tail = normalized[start:].strip()
            if len(tail) >= 120:
                anchor_candidates.append((f"after_{marker}", start))

    # Thai "ประกาศ" is common in valid content, so only treat it as an anchor
    # when it appears as an isolated line with nearby menu chrome cues.
    for match in re.finditer(r"(?im)^\s*ประกาศ\s*$", normalized):
        start = match.end()
        window_before = normalized[max(0, match.start() - 500) : match.start()].lower()
        if any(token in window_before for token in ("quicklinks", "see all", "other sites", "you're viewing", "back")):
            tail = normalized[start:].strip()
            if len(tail) >= 120:
                anchor_candidates.append(("after_ประกาศ", start))

    if len(title_positions) >= 2:
        last_title_pos = title_positions[-1]
        tail = normalized[last_title_pos:].strip()
        if len(tail) >= 120:
            anchor_candidates.append(("last_title", last_title_pos))

    if anchor_candidates:
        # "last occurrence" preference across allowed anchors.
        anchor_label, anchor_start = max(anchor_candidates, key=lambda x: x[1])

        # Year line around the anchor can indicate beginning of article section.
        near = normalized[anchor_start : anchor_start + 320]
        year_match = re.search(r"\b(?:20\d{2}|25\d{2})\b", near)
        if year_match:
            anchor_start = anchor_start + year_match.end()
            anchor_label = f"{anchor_label}_year_aligned"

        body = normalized[anchor_start:].strip()
        if body:
            return body, {"anchor": anchor_label, "start_idx": anchor_start}

    if len(title_positions) >= 2:
        start = title_positions[-1]
        body = normalized[start:].strip()
        if body:
            return body, {"anchor": "fallback_last_title", "start_idx": start}

    return normalized.strip(), {"anchor": "fallback_original", "start_idx": 0}


def _is_chrome_line(line: str) -> bool:
    if not line:
        return True

    lower = line.lower()

    if any(token in lower for token in CHROME_TOKENS):
        # Keep long policy-like paragraphs even if they mention a token.
        return len(line) <= 140 or re.fullmatch(r"[A-Za-zก-๙0-9 /&\-.]{1,80}", line) is not None

    if re.match(r"^(see all|back|search|quicklinks|other sites|sitemap)\b", lower):
        return True

    if len(line) <= 30 and re.fullmatch(r"[A-Za-z][A-Za-z0-9 /&\-.]{0,29}", line):
        return True

    if len(line) <= 28 and re.fullmatch(r"[ก-๙A-Za-z0-9 /&\-.]{1,28}", line):
        if not _looks_like_policy_line(line):
            return True

    return False


def remove_boilerplate_lines(lines: List[str]) -> List[str]:
    """
    Remove website chrome/navigation lines conservatively.

    Rules:
    - Remove short nav/footer lines with known chrome tokens.
    - Drop short lines repeated >2 times (likely menu blocks).
    - Never remove lines that look policy/numeric/date/conditions.
    """
    normalized = [_normalize_line(line) for line in lines]

    first_pass: List[str] = []
    for line in normalized:
        if not line:
            continue
        if _is_placeholder_line(line):
            continue
        if _has_keep_guard(line):
            first_pass.append(line)
            continue
        if _is_chrome_line(line):
            continue
        if _is_noise_line(line):
            continue
        first_pass.append(line)

    short_counts = Counter(
        re.sub(r"\s+", " ", line.lower())
        for line in first_pass
        if len(line) <= 60
    )
    repeated_short = {key for key, count in short_counts.items() if count > 2}

    second_pass: List[str] = []
    for line in first_pass:
        key = re.sub(r"\s+", " ", line.lower())
        if key in repeated_short and len(line) <= 60 and not _has_keep_guard(line):
            continue
        second_pass.append(line)

    deduped: List[str] = []
    for line in second_pass:
        if deduped and deduped[-1].lower() == line.lower():
            continue
        deduped.append(line)

    return deduped


def _apply_disambiguation(lines: List[str], loan_context: bool) -> List[str]:
    output: List[str] = []

    for line in lines:
        updated = line

        if loan_context and "ปิดบัญชี" in updated and "ปิดสินเชื่อก่อนกำหนด" not in updated:
            updated = updated.replace("ปิดบัญชี", "ปิดบัญชี (ปิดสินเชื่อก่อนกำหนด)")

        if loan_context and "ค่าธรรมเนียม" in updated and "ค่าธรรมเนียมสินเชื่อบ้าน" not in updated:
            updated = updated.replace("ค่าธรรมเนียม", "ค่าธรรมเนียมสินเชื่อบ้าน", 1)

        output.append(updated)

    return output


def analyze_scraped_text(text: str) -> Dict[str, float]:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [_normalize_line(line) for line in normalized.split("\n")]
    non_empty_lines = [line for line in lines if line]

    if not non_empty_lines:
        return {
            "char_count": 0,
            "line_count": 0,
            "noise_line_ratio": 0.0,
            "table_likeness_score": 0.0,
            "row_conversion_score": 0.0,
            "duplicate_boilerplate_score": 0.0,
        }

    noise_line_count = sum(1 for line in non_empty_lines if _is_noise_line(line))

    short_keys = [line.lower() for line in non_empty_lines if len(line) <= 60]
    key_counts = Counter(short_keys)
    duplicate_hits = sum(count - 1 for count in key_counts.values() if count > 1)

    _, table_likeness_score, row_conversion_score = _convert_table_like_lines(non_empty_lines)

    return {
        "char_count": len(text),
        "line_count": len(non_empty_lines),
        "noise_line_ratio": noise_line_count / max(len(non_empty_lines), 1),
        "table_likeness_score": table_likeness_score,
        "row_conversion_score": row_conversion_score,
        "duplicate_boilerplate_score": duplicate_hits / max(len(non_empty_lines), 1),
    }


def compute_chrome_noise_metrics(text: str) -> Dict[str, float]:
    """Compute chrome/noise ratio and domain signal strength on cleaned text."""
    lines = [_normalize_line(line) for line in text.splitlines()]
    nonempty = [line for line in lines if line]
    if not nonempty:
        return {
            "noise_line_ratio": 1.0,
            "noise_lines": 0,
            "total_nonempty_lines": 0,
            "content_signal": 0,
        }

    noise_lines = 0
    for line in nonempty:
        lower = line.lower()
        if any(token in lower for token in NAV_CHROME_KEYWORDS):
            noise_lines += 1

    merged = "\n".join(nonempty).lower()
    content_signal = sum(1 for token in DOMAIN_SIGNAL_KEYWORDS if token in merged)

    return {
        "noise_line_ratio": noise_lines / max(len(nonempty), 1),
        "noise_lines": noise_lines,
        "total_nonempty_lines": len(nonempty),
        "content_signal": content_signal,
    }


def _is_rate_sheet_like(lines: List[str]) -> bool:
    nonempty = [line for line in lines if line.strip()]
    if not nonempty:
        return False

    signal_hits = 0
    for line in nonempty:
        lower = line.lower()
        if re.search(r"\d+(?:\.\d+)?\s*%", line):
            signal_hits += 1
        if any(token in lower for token in ("mrr", "ltv", "ปีที่", "เฉลี่ย", "อายุสัญญา")):
            signal_hits += 1

    density = signal_hits / max(len(nonempty), 1)
    return density >= 0.30 or signal_hits >= 8


def _rate_conditions_from_line(line: str) -> List[str]:
    conditions: List[str] = []
    lower = line.lower()
    if "ทำประกัน" in line:
        if "ไม่" in line and "ไม่ทำประกัน" in line:
            conditions.append("ไม่ทำประกันชีวิต")
        else:
            conditions.append("ทำประกันชีวิต")
    ltv_match = re.search(r"(ltv\s*(?:<=|>=|<|>)?\s*\d+%?)", line, flags=re.IGNORECASE)
    if ltv_match:
        conditions.append(ltv_match.group(1).upper())
    income_match = re.search(r"(\d{2,3}(?:,\d{3})*)\s*/\s*(\d{2,3}(?:,\d{3})*)", line)
    if income_match:
        conditions.append(f"รายได้ {income_match.group(1)}/{income_match.group(2)} บาท")
    if "รายได้" in lower and "บาท" in lower and not income_match:
        conditions.append("ตามเงื่อนไขรายได้")
    return conditions


def _extract_rate_tokens(line: str) -> List[str]:
    tokens: List[str] = []
    for match in re.findall(r"MRR\s*[-+]?\s*\d+(?:\.\d+)?%?", line, flags=re.IGNORECASE):
        tokens.append(re.sub(r"\s+", "", match.upper()))
    for match in re.findall(r"\d+(?:\.\d+)?\s*%", line):
        normalized = re.sub(r"\s+", "", match)
        if normalized not in tokens:
            tokens.append(normalized)
    return tokens


def _convert_rate_sheet_lines(lines: List[str]) -> List[str]:
    """Convert table-like interest rows into readable sentence rows."""
    if not _is_rate_sheet_like(lines):
        return lines

    output: List[str] = []
    current_year_label = ""

    for raw in lines:
        line = _normalize_line(raw)
        if not line:
            continue

        year_match = re.search(r"ปีที่\s*(\d+)", line)
        if year_match:
            current_year_label = f"ปีที่ {year_match.group(1)}"

        if "เฉลี่ย 3 ปี" in line:
            rates = _extract_rate_tokens(line)
            if rates:
                output.append(f"เฉลี่ย 3 ปี: {', '.join(rates)}")
                continue

        if "ตลอดอายุสัญญา" in line:
            rates = _extract_rate_tokens(line)
            if rates:
                output.append(f"อัตราดอกเบี้ยตลอดอายุสัญญา: {', '.join(rates)}")
                continue

        if not (re.search(r"\d+(?:\.\d+)?\s*%", line) or "mrr" in line.lower()):
            continue

        rates = _extract_rate_tokens(line)
        if not rates:
            continue

        conditions = _rate_conditions_from_line(line)
        prefix_parts = []
        if current_year_label:
            prefix_parts.append(current_year_label)
        if conditions:
            prefix_parts.extend(conditions)
        prefix = " | ".join(prefix_parts) if prefix_parts else "เงื่อนไขอัตราดอกเบี้ย"
        output.append(f"{prefix}: {', '.join(rates)}")

    deduped: List[str] = []
    for line in output:
        if not deduped or deduped[-1] != line:
            deduped.append(line)

    # Keep original lines when conversion confidence is low.
    if len(deduped) < 3:
        return lines
    return deduped


def clean_scraped_text(text: str, title: str = "", apply_body_extraction: bool = True) -> str:
    """Clean web-scraped content for RAG use while preserving policy details."""
    if not text:
        return ""

    if apply_body_extraction:
        main_body, _ = extract_main_body(text, title=title)
    else:
        main_body = text
    normalized = main_body.replace("\r\n", "\n").replace("\r", "\n")
    lines = [_normalize_line(line) for line in normalized.split("\n")]

    filtered = remove_boilerplate_lines(lines)
    filtered = _remove_repeated_short_lines(filtered)
    filtered, _, _ = _convert_table_like_lines(filtered)

    loan_context = _contains_loan_context("\n".join(filtered))
    filtered = _apply_disambiguation(filtered, loan_context=loan_context)

    filtered = _merge_broken_lines(filtered)
    filtered = _convert_rate_sheet_lines(filtered)

    deduped: List[str] = []
    for line in filtered:
        if not deduped or deduped[-1] != line:
            deduped.append(line)

    cleaned = "\n".join(deduped)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    return cleaned.strip()


def extract_effective_year(text: str) -> str:
    """Extract Thai Buddhist years (e.g., 2568/2569) from title/content."""
    if not text:
        return ""

    ranges = []
    seen = set()
    ranged_years = set()
    for start, end in re.findall(r"(25\d{2})\s*[/\-]\s*(25\d{2})", text):
        value = f"{start}/{end}"
        if value not in seen:
            seen.add(value)
            ranges.append(value)
        ranged_years.update((start, end))

    singles = []
    for year in re.findall(r"\b(25\d{2})\b", text):
        if year in ranged_years:
            continue
        if year not in seen:
            seen.add(year)
            singles.append(year)

    values = ranges + singles
    return ", ".join(values)


def infer_topic_tags(title: str, content: str) -> List[str]:
    text = f"{title}\n{content}".lower()
    tags = []

    for tag, keywords in TOPIC_KEYWORDS.items():
        if any(keyword in text for keyword in keywords):
            tags.append(tag)

    return tags


def _infer_topic_from_text(title: str, content: str, url: str = "", file_name: str = "") -> str:
    title_lower = title.lower()
    url_lower = url.lower()
    file_name_lower = file_name.lower()
    body_lines = [line for line in content.splitlines() if line.strip()]
    # Focus early body where article content usually appears.
    body_focus = "\n".join(body_lines[:40]).lower()

    # Strong URL/title routing first to avoid menu-driven drift.
    relief_hints = ("relief", "flood", "covid", "ช่วยเหลือ", "พักชำระ", "ปรับโครงสร้างหนี้")
    if (
        any(token in title_lower for token in relief_hints)
        or any(token in url_lower for token in relief_hints)
        or any(token in file_name_lower for token in ("debt", "restructur", "relief", "covid", "flood"))
    ):
        return "hardship_relief"

    refinance_hints = ("รีไฟแนนซ์", "refinance", "mortgage power", "บ้านแลกเงิน")
    if (
        any(token in title_lower for token in refinance_hints)
        or any(token in url_lower for token in refinance_hints)
        or any(token in file_name_lower for token in ("refinance", "mortgage-power", "home-loan-refinance"))
    ):
        return "refinance"

    interest_hints = ("ดอกเบี้ย", "interest", "mrr", "mlr", "mor", "rate")
    if (
        any(token in title_lower for token in interest_hints)
        or "loan-interest-rates" in url_lower
        or any(token in file_name_lower for token in ("interest-rates", "generic-rate"))
    ):
        return "interest_rate"

    fee_hints = ("ค่าธรรมเนียม", "fee", "charge", "ค่าปรับ", "จดจำนอง")
    if (
        any(token in title_lower for token in fee_hints)
        or "/fees/" in url_lower
        or any(token in file_name_lower for token in ("service-fees", "close-account", "collateral-release"))
    ):
        return "fees"

    topic_rules = {
        "hardship_relief": TOPIC_KEYWORDS["relief"],
        "interest_rate": TOPIC_KEYWORDS["interest"],
        "fees": TOPIC_KEYWORDS["fee"],
        "refinance": TOPIC_KEYWORDS["refinance"],
        "home_loan_policy": (
            "สินเชื่อบ้าน",
            "home loan",
            "เงื่อนไข",
            "คุณสมบัติ",
            "เอกสาร",
            "eligibility",
            "application",
        ),
    }

    scores: Dict[str, int] = {}
    for topic, keywords in topic_rules.items():
        title_hits = sum(1 for keyword in keywords if keyword in title_lower)
        focus_hits = sum(1 for keyword in keywords if keyword in body_focus)
        scores[topic] = (title_hits * 3) + (focus_hits * 2)

    best_topic = max(scores, key=scores.get)
    if scores[best_topic] <= 0:
        return "home_loan_policy"
    return best_topic


def infer_relevance(
    cleaned_body: str,
    title: str,
    url: str,
    category: str = "",
    doc_kind: str = "",
    institution: str = "",
    file_name: str = "",
) -> Dict[str, object]:
    """Infer whether a document is home-loan relevant and explain the decision."""
    body_lower = cleaned_body.lower()
    title_lower = title.lower()
    url_lower = url.lower()
    merged_lower = f"{title_lower}\n{body_lower}\n{url_lower}"
    category_lower = category.lower()
    doc_kind_lower = doc_kind.lower()
    institution_lower = institution.lower()
    file_name_lower = file_name.lower()

    positive_body_hits = sum(1 for kw in POSITIVE_RELEVANCE_KEYWORDS if kw in body_lower)
    positive_title_hits = sum(1 for kw in POSITIVE_RELEVANCE_KEYWORDS if kw in title_lower)
    negative_hits = sum(1 for kw in NEGATIVE_RELEVANCE_KEYWORDS if kw in merged_lower)
    url_hint = any(hint in url_lower for hint in HOME_LOAN_URL_HINTS)

    category_signal = category_lower in {
        "interest_structure",
        "fee_structure",
        "refinance",
        "hardship_support",
        "policy_requirement",
        "eligibility_rule",
    }
    doc_kind_signal = doc_kind_lower in {"policy", "rate_sheet", "form"}
    institution_signal = "cimb" in institution_lower
    file_signal = any(
        token in file_name_lower
        for token in ("loan", "mortgage", "refinance", "debt", "relief", "hardship", "homeloan")
    )

    positive_hits = positive_body_hits + positive_title_hits

    positive_score = min(0.65, (positive_body_hits * 0.14) + (positive_title_hits * 0.2))
    if url_hint:
        positive_score += 0.22
    if category_signal:
        positive_score += 0.12
    if doc_kind_signal and (url_hint or positive_hits > 0):
        positive_score += 0.08
    if institution_signal and (url_hint or positive_hits > 0):
        positive_score += 0.05
    if file_signal:
        positive_score += 0.12

    negative_penalty = min(0.55, negative_hits * 0.17)

    relevance_score = max(0.0, min(1.0, 0.2 + positive_score - negative_penalty))
    has_min_signal = (positive_hits > 0) or url_hint or category_signal or file_signal
    negative_dominates = negative_hits >= max(
        3,
        positive_hits
        + (1 if url_hint else 0)
        + (1 if category_signal else 0)
        + (1 if file_signal else 0)
        + 1,
    )
    has_body = len(cleaned_body.strip()) >= 80

    min_threshold = 0.30 if file_signal else 0.35

    is_relevant = bool(
        has_body
        and has_min_signal
        and not negative_dominates
        and relevance_score >= min_threshold
    )

    if not has_body:
        reason = "insufficient_body_after_cleaning"
    elif not has_min_signal:
        reason = "no_home_loan_keywords_or_url_hints"
    elif negative_dominates and positive_hits == 0:
        reason = "negative_indicators_dominate_without_positive_signal"
    elif negative_dominates:
        reason = "negative_indicators_dominate"
    elif is_relevant:
        reason = (
            f"positive_hits={positive_hits}, url_hint={url_hint}, "
            f"category_signal={category_signal}, negative_hits={negative_hits}"
        )
    else:
        reason = "relevance_score_below_threshold"

    topic = (
        _infer_topic_from_text(title=title, content=cleaned_body, url=url, file_name=file_name)
        if is_relevant
        else "unrelated"
    )

    return {
        "is_home_loan_relevant": is_relevant,
        "relevance_score": round(float(relevance_score), 4),
        "relevance_reason": reason,
        "topic": topic,
    }


def _split_summary_units(text: str) -> List[str]:
    units: List[str] = []

    for raw_line in text.splitlines():
        line = _normalize_line(raw_line)
        if not line:
            continue
        if _is_placeholder_line(line):
            continue
        if re.search(r"[_\.]{4,}", line):
            continue
        if _is_chrome_line(line) and not _has_keep_guard(line):
            continue

        fragments = re.split(r"(?<=[.!?])\s+", line)
        for fragment in fragments:
            fragment = _normalize_line(fragment)
            if not fragment:
                continue
            if len(fragment) < 18 and not re.search(r"\d|%", fragment):
                continue
            if _is_chrome_line(fragment) and not _has_keep_guard(fragment):
                continue
            units.append(fragment)

    return units


def _summary_score(unit: str) -> float:
    lower = unit.lower()
    score = 0.0

    if any(term in lower for term in LOAN_CONTEXT_TERMS):
        score += 2.0
    if any(term in lower for term in ("ดอกเบี้ย", "ค่าธรรมเนียม", "เงื่อนไข", "mrr", "ltv", "rate", "%")):
        score += 1.5
    if re.search(r"\d", unit):
        score += 1.0
    if 30 <= len(unit) <= 220:
        score += 0.6
    if len(unit) > 260:
        score -= 0.4

    return score


def _finalize_summary_sentences(sentences: List[str]) -> List[str]:
    finalized: List[str] = []
    for sentence in sentences:
        s = _normalize_line(sentence).strip(" -")
        if not s:
            continue
        if not re.search(r"[.!?]$", s):
            s = f"{s}."
        finalized.append(s)
    return finalized


def generate_grounded_summary(cleaned_body: str) -> str:
    """Generate 3-5 extractive, grounded summary sentences from cleaned content."""
    units = _split_summary_units(cleaned_body)
    if len(units) < 2:
        return INSUFFICIENT_BODY_MESSAGE

    scored = [
        (idx, unit, _summary_score(unit))
        for idx, unit in enumerate(units)
    ]
    ranked = sorted(scored, key=lambda item: (-item[2], item[0]))

    selected: List[Tuple[int, str]] = []
    seen = set()
    for idx, unit, _ in ranked:
        key = unit.lower()
        if key in seen:
            continue
        seen.add(key)
        selected.append((idx, unit))
        if len(selected) == 5:
            break

    if len(selected) < 3:
        for idx, unit in enumerate(units):
            key = unit.lower()
            if key in seen:
                continue
            seen.add(key)
            selected.append((idx, unit))
            if len(selected) == 3:
                break

    if len(selected) < 3:
        return INSUFFICIENT_BODY_MESSAGE

    selected.sort(key=lambda item: item[0])
    sentences = _finalize_summary_sentences([item[1] for item in selected[:5]])
    if len(sentences) < 3:
        return INSUFFICIENT_BODY_MESSAGE
    return " ".join(sentences)


class StructuredDocumentParser:
    """Parse documents with metadata headers."""

    LAST_PARSE_REPORT: Dict[str, object] = {
        "total_docs": 0,
        "indexed_docs": 0,
        "quarantined_docs": 0,
        "quarantined_examples": [],
    }

    @staticmethod
    def parse_file(file_path: Path) -> Optional[Document]:
        """
        Parse a structured document file.

        Expected format:
        TITLE: ...
        SOURCE URL: ...
        INSTITUTION: ...
        PUBLICATION DATE: ...
        CATEGORY: ...
        ---
        SUMMARY (3-5 sentences relevance)
        ...
        ---
        FULL CLEANED TEXT CONTENT
        ...
        """
        try:
            content = file_path.read_text(encoding="utf-8")

            metadata: Dict[str, object] = {}

            title_match = re.search(r"TITLE:\s*(.+?)(?:\n|$)", content)
            if title_match:
                metadata["title"] = title_match.group(1).strip()

            url_match = re.search(r"SOURCE URL:\s*(.+?)(?:\n|$)", content)
            if url_match:
                metadata["source_url"] = url_match.group(1).strip()

            institution_match = re.search(r"INSTITUTION:\s*(.+?)(?:\n|$)", content)
            if institution_match:
                metadata["institution"] = institution_match.group(1).strip()

            date_match = re.search(r"PUBLICATION DATE:\s*(.+?)(?:\n|$)", content)
            if date_match:
                metadata["publication_date"] = date_match.group(1).strip()

            category_match = re.search(r"CATEGORY:\s*(.+?)(?:\n|$)", content)
            if category_match:
                metadata["category"] = category_match.group(1).strip()

            content_match = re.search(r"FULL CLEANED TEXT CONTENT\n(.*)", content, re.DOTALL)
            if content_match:
                main_content = content_match.group(1).strip()
            else:
                main_content = content

            metadata["file_name"] = file_path.name
            metadata["file_path"] = str(file_path)

            title = str(metadata.get("title", ""))
            source_url = str(metadata.get("source_url", ""))
            doc_kind = StructuredDocumentParser._infer_doc_kind(title, file_path.name)
            metadata["doc_kind"] = doc_kind

            product_type = str(metadata.get("product_type", "home_loan"))
            metadata["product_type"] = product_type
            metadata["domain"] = "loan"
            metadata["cleaning_version"] = CLEANING_VERSION

            extracted_body, extraction_meta = extract_main_body(main_content, title=title)
            cleaned_content = clean_scraped_text(
                extracted_body,
                title=title,
                apply_body_extraction=False,
            )
            if not cleaned_content:
                cleaned_content = clean_scraped_text(main_content, title=title)

            effective_year = extract_effective_year(f"{title}\n{cleaned_content}")
            metadata["effective_year"] = effective_year

            chrome_metrics = compute_chrome_noise_metrics(cleaned_content)
            noise_line_ratio = float(chrome_metrics["noise_line_ratio"])
            content_signal = int(chrome_metrics["content_signal"])
            metadata["noise_line_ratio"] = round(noise_line_ratio, 4)
            metadata["content_signal"] = content_signal

            relevance = infer_relevance(
                cleaned_content,
                title=title,
                url=source_url,
                category=str(metadata.get("category", "")),
                doc_kind=doc_kind,
                institution=str(metadata.get("institution", "")),
                file_name=file_path.name,
            )
            topic = str(relevance["topic"])
            relevance_score = float(relevance["relevance_score"])
            relevance_reason = str(relevance["relevance_reason"])
            quarantined = not bool(relevance["is_home_loan_relevant"])

            lower_title = title.lower()
            lower_body = cleaned_content.lower()
            lower_url = source_url.lower()
            strong_relevant_hint = _has_any(
                f"{lower_title} {lower_url}",
                (
                    "home-loan",
                    "สินเชื่อบ้าน",
                    "refinance",
                    "รีไฟแนนซ์",
                    "mortgage",
                    "relief",
                    "covid",
                    "flood",
                    "/documents/loan/",
                    "debt",
                ),
            )
            unrelated_title_hit = "customer profiling" in lower_title
            unrelated_nav_heavy = (
                _has_any(lower_body, ("เงินฝาก", "บัตร", "ประกัน", "ndid", "พร้อมเพย์"))
                and noise_line_ratio > 0.35
                and content_signal < 4
                and not strong_relevant_hint
            )
            chrome_dominates = noise_line_ratio > 0.40 and content_signal < 3 and not strong_relevant_hint

            if unrelated_title_hit or unrelated_nav_heavy or chrome_dominates:
                quarantined = True
                topic = "unrelated_web_chrome"
                relevance_score = min(relevance_score, 0.15)
                if chrome_dominates:
                    relevance_reason = "high_chrome_noise_ratio"
                elif unrelated_title_hit:
                    relevance_reason = "unrelated_title_topic"
                else:
                    relevance_reason = "unrelated_navigation_content"

            topic_tags = infer_topic_tags(title, cleaned_content)
            topic_to_tag = {
                "interest_rate": "interest",
                "fees": "fee",
                "refinance": "refinance",
                "hardship_relief": "relief",
                "home_loan_policy": "document_requirement",
            }
            mapped_tag = topic_to_tag.get(topic)
            if mapped_tag and mapped_tag not in topic_tags:
                topic_tags.append(mapped_tag)
            topic_tags = sorted(set(topic_tags))
            metadata["topic_tags"] = ", ".join(topic_tags)

            original_category = str(metadata.get("category", "")).strip()
            category_lower = original_category.lower()
            resolved_category = original_category
            _explicit_categories = {
                "fee_structure", "interest_structure", "refinance",
                "hardship_support", "policy_requirement",
            }
            if topic in {"unrelated", "unrelated_web_chrome"} and category_lower == "bank_policy":
                resolved_category = "unrelated"
            elif category_lower not in _explicit_categories and topic in TOPIC_TO_CATEGORY and topic != "unrelated":
                resolved_category = TOPIC_TO_CATEGORY[topic]
            elif not resolved_category and topic in TOPIC_TO_CATEGORY:
                resolved_category = TOPIC_TO_CATEGORY[topic]
            metadata["category"] = resolved_category or "uncategorized"

            if quarantined:
                if topic == "unrelated_web_chrome" or relevance_reason in {
                    "high_chrome_noise_ratio",
                    "unrelated_title_topic",
                    "unrelated_navigation_content",
                }:
                    summary_text = WEB_CHROME_QUARANTINE_SUMMARY
                else:
                    summary_text = UNRELATED_SUMMARY_MESSAGE
            else:
                summary_text = generate_grounded_summary(cleaned_content)
                if summary_text == INSUFFICIENT_BODY_MESSAGE:
                    quarantined = True
                    relevance_reason = "insufficient_body_after_cleaning"
                    relevance_score = min(relevance_score, 0.25)

            metadata["topic"] = topic
            metadata["relevance_score"] = round(float(relevance_score), 4)
            metadata["quarantined"] = bool(quarantined)
            metadata["relevance_reason"] = relevance_reason
            metadata["body_anchor"] = str(extraction_meta.get("anchor", ""))
            metadata["body_anchor_idx"] = int(extraction_meta.get("start_idx", 0))

            enhanced_text = (
                f"TITLE: {metadata.get('title', 'N/A')}\n"
                f"CLEANING_VERSION: {CLEANING_VERSION}\n"
                f"SOURCE URL: {metadata.get('source_url', 'N/A')}\n"
                f"INSTITUTION: {metadata.get('institution', 'N/A')}\n"
                f"PUBLICATION DATE: {metadata.get('publication_date', 'N/A')}\n"
                f"CATEGORY: {metadata.get('category', 'N/A')}\n"
                f"DOC KIND: {metadata.get('doc_kind', 'unknown')}\n"
                f"PRODUCT TYPE: {metadata.get('product_type', 'home_loan')}\n"
                f"DOMAIN: {metadata.get('domain', 'loan')}\n"
                f"EFFECTIVE YEAR: {metadata.get('effective_year', 'N/A')}\n"
                f"TOPIC: {metadata.get('topic', 'unrelated')}\n"
                f"TOPIC TAGS: {metadata.get('topic_tags', '')}\n"
                f"RELEVANCE SCORE: {metadata.get('relevance_score', 0.0)}\n"
                f"QUARANTINED: {metadata.get('quarantined', False)}\n"
                f"RELEVANCE REASON: {metadata.get('relevance_reason', '')}\n"
                f"---\n"
                f"SUMMARY\n"
                f"{summary_text}\n"
                f"---\n"
                f"FULL CLEANED TEXT CONTENT\n"
                f"{cleaned_content}"
            )

            return Document(
                text=enhanced_text,
                metadata=metadata,
                excluded_embed_metadata_keys=list(metadata.keys()),
                excluded_llm_metadata_keys=list(metadata.keys()),
            )

        except Exception as exc:
            print(f"Error parsing {file_path}: {exc}")
            return None

    @staticmethod
    def _infer_doc_kind(title: str, file_name: str) -> str:
        """Infer coarse document kind from title/file name."""
        text = f"{title} {file_name}".lower()

        form_keywords = ("แบบฟอร์ม", "คำขอ", "form", "consent", "download-center")
        rate_keywords = ("อัตราดอกเบี้ย", "ดอกเบี้ย", "rate", "mrr", "mlr", "mor")

        if any(keyword in text for keyword in form_keywords):
            return "form"
        if any(keyword in text for keyword in rate_keywords):
            return "rate_sheet"
        return "policy"

    @staticmethod
    def parse_directory(directory: Path, include_quarantined: bool = False) -> List[Document]:
        """Parse all .txt documents in a directory."""
        documents: List[Document] = []
        quarantined_examples: List[Dict[str, str]] = []
        total_docs = 0
        quarantined_docs = 0

        for file_path in sorted(directory.glob("*.txt")):
            doc = StructuredDocumentParser.parse_file(file_path)
            if not doc:
                continue

            total_docs += 1
            metadata = doc.metadata or {}
            is_quarantined = bool(metadata.get("quarantined", False))

            if is_quarantined:
                quarantined_docs += 1
                if len(quarantined_examples) < 20:
                    quarantined_examples.append(
                        {
                            "title": str(metadata.get("title", file_path.name)),
                            "reason": str(metadata.get("relevance_reason", "unknown")),
                        }
                    )
                if include_quarantined:
                    documents.append(doc)
                continue

            documents.append(doc)

        StructuredDocumentParser.LAST_PARSE_REPORT = {
            "total_docs": total_docs,
            "indexed_docs": len(documents),
            "quarantined_docs": quarantined_docs,
            "quarantined_examples": quarantined_examples,
        }

        return documents

    @staticmethod
    def get_last_parse_report() -> Dict[str, object]:
        return dict(StructuredDocumentParser.LAST_PARSE_REPORT)
