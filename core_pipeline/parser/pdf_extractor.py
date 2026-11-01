"""
PDF Text Extractor for Korean insurance product summaries (상품요약서).

Primary target: KB손해보험 상품요약서 — 손해보험협회 상품통일 공시기준 양식.

    가. 가입자격제한 등 상품(별/의) 특이사항
        1) 가입자격제한
        2) 상품의 특이사항
    나. 보험금 지급사유, 지급금액 및 지급제한사항
        1) (담보별) 보험금 지급사유 및 (지급)금액   -> [보장명 | 지급사유 | 지급금액] 표
        2) 보험금 지급제한 사항                       -> 면책
    다. 보험료 산출기초  (… 라. 마. 해약환급금 …)

일반보험(소멸성)은 같은 내용을 "1. / 2. / 가. / 나." 번호 체계로 쓴다.
생명보험사 양식("상품의 특이사항 / 보험가입자격요건 / 보험금 지급사유 및 지급제한")도
fallback 패턴으로 계속 지원한다.

Text extraction: pdftotext(poppler) 우선, 없으면 pypdf.
  * KB 요약서는 pypdf에서 글자 순서가 뒤섞이므로(폰트 매핑 문제) poppler 설치를 권장:
      macOS: brew install poppler
"""

import logging
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ExtractedSections:
    product_name: str = ""
    insurer: str = ""
    full_text: str = ""
    특이사항: str = ""
    가입자격: str = ""
    보장내용: str = ""       # 보험금 지급사유 및 지급금액
    면책사항: str = ""       # 보험금 지급제한 사항
    page_count: int = 0
    extraction_method: str = "pypdf"
    layout: str = "unknown"  # "kb-nonlife" | "life" | "unknown"


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------

def extract_text_from_pdf(pdf_path: Path) -> tuple[str, str]:
    """Return (text, method). pdftotext if available, else pypdf."""
    if shutil.which("pdftotext"):
        try:
            result = subprocess.run(
                ["pdftotext", "-layout", str(pdf_path), "-"],
                capture_output=True, text=True, timeout=60,
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout, "pdftotext"
        except Exception as e:  # pragma: no cover
            logger.warning("pdftotext failed: %s", e)
    return _extract_with_pypdf(pdf_path), "pypdf"


def _extract_with_pypdf(pdf_path: Path) -> str:
    from pypdf import PdfReader
    reader = PdfReader(str(pdf_path))
    parts = []
    for page in reader.pages:
        try:
            t = page.extract_text(extraction_mode="layout")
        except TypeError:  # old pypdf
            t = page.extract_text()
        if t:
            parts.append(t)
    return "\n".join(parts)


def get_page_count(pdf_path: Path) -> int:
    try:
        from pypdf import PdfReader
        return len(PdfReader(str(pdf_path)).pages)
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# Section detection
# ---------------------------------------------------------------------------

# Each entry: (section_key, [regex alternatives]).  First alternative that
# matches wins; patterns are tried in order so insurer-specific ones go first.
_SECTION_PATTERNS: list[tuple[str, list[str]]] = [
    ("가입자격", [
        r"(?:^|\n)\s*(?:1\)|가\.)\s*가입자격\s*제한",          # KB 장기/일반
        r"보험가입\s*자격요건",                                # 생보 양식
    ]),
    ("특이사항", [
        r"(?:^|\n)\s*(?:2\)|나\.)\s*상품의\s*특이사항",         # KB
        r"상품의\s*특이사항",                                   # 생보 양식
    ]),
    ("보장내용", [
        r"보험금\s*지급사유\s*,?\s*지급금액\s*및\s*지급제한\s*사항",   # KB 장기
        r"보험금\s*지급사유\s*및\s*지급제한\s*사항",                 # KB 일반 / 생보
    ]),
    ("면책사항", [
        r"(?:^|\n)\s*(?:2\)|나\.)\s*보험금\s*지급제한\s*사항",   # KB
        r"보험금을\s*지급하지\s*않는\s*사유",
        r"일반적인\s*보험금\s*지급제한\s*사유",
        r"면책사항",
    ]),
]

# Hard stop for the last section so 면책 doesn't swallow 보험료/해약환급금
_END_PATTERNS = [
    r"(?:^|\n)\s*(?:다\.|3\.)\s*보험료\s*산출",
    r"(?:^|\n)\s*[라마바]\.\s*",
    r"해약환급금에\s*관한\s*사항",
]


def _find(patterns: list[str], text: str, start: int = 0) -> int | None:
    for p in patterns:
        m = re.search(p, text[start:])
        if m:
            return start + m.start()
    return None


def extract_sections(pdf_path: Path) -> ExtractedSections:
    full_text, method = extract_text_from_pdf(pdf_path)
    page_count = get_page_count(pdf_path)
    product_name, insurer = _extract_product_name(full_text)

    # 1) locate 보장내용 first: everything else is anchored around it
    pos_cov = _find(dict(_SECTION_PATTERNS)["보장내용"], full_text)
    pos_elig = _find(dict(_SECTION_PATTERNS)["가입자격"], full_text)
    pos_spec = _find(dict(_SECTION_PATTERNS)["특이사항"], full_text)
    # 면책 must come AFTER 보장내용 (the sub-section "2) 보험금 지급제한 사항")
    pos_excl = _find(dict(_SECTION_PATTERNS)["면책사항"], full_text,
                     start=pos_cov if pos_cov is not None else 0)
    pos_end = _find(_END_PATTERNS, full_text,
                    start=pos_excl if pos_excl is not None else (pos_cov or 0))

    layout = "kb-nonlife" if re.search(r"(?:^|\n)\s*(?:가\.|1\.)\s*가입자격", full_text) else (
        "life" if re.search(r"보험가입\s*자격요건", full_text) else "unknown")

    def slice_(start, *ends, cap=20000):
        if start is None:
            return ""
        nxt = [e for e in ends if e is not None and e > start]
        stop = min(nxt) if nxt else min(len(full_text), start + cap)
        return full_text[start:stop].strip()

    sections = ExtractedSections(
        product_name=product_name,
        insurer=insurer,
        full_text=full_text,
        가입자격=slice_(pos_elig, pos_spec, pos_cov),
        특이사항=slice_(pos_spec, pos_cov),
        보장내용=slice_(pos_cov, pos_excl, pos_end, cap=60000),
        면책사항=slice_(pos_excl, pos_end, cap=12000),
        page_count=page_count,
        extraction_method=method,
        layout=layout,
    )
    return sections


def _extract_product_name(text: str) -> tuple[str, str]:
    """Return (product_name, insurer).

    KB 상품요약서: 1행에 상품명, 2행에 '상품요약서'.
    생보 양식: '보험약관 등 <상품명>의 기초서류' 문장.
    """
    head = text[:3000]
    lines = [re.sub(r"\s+", " ", l).strip() for l in head.split("\n")]
    lines = [l for l in lines if l]

    insurer = ""
    if re.search(r"KB\s*손해보험|KB손보|\bKB\b", head):
        insurer = "KB손해보험"
    else:
        m = re.search(r"([가-힣A-Za-z]{2,10}(?:생명|손해|화재|해상)보험)", head)
        insurer = m.group(1) if m else ""

    # KB: product name line directly before '상품요약서'
    for i, l in enumerate(lines[:6]):
        if l.replace(" ", "") == "상품요약서" and i > 0:
            return lines[i - 1], insurer
        if l.endswith("상품요약서") and len(l) > 6:          # e.g. "KB스마트운전자보험상품요약서"
            return l[: -len("상품요약서")].strip(), insurer

    m = re.search(r"보험약관\s*등\s*(무배당[\s\S]*?)의\s*기초서류", head)
    if m:
        return re.sub(r"\s+", " ", m.group(1)).strip(), insurer

    for l in lines[:10]:
        if re.search(r"(KB|무배당).*보험", l) and len(l) > 6:
            return l, insurer
    return "", insurer


# ---------------------------------------------------------------------------
# LLM-ready condensation
# ---------------------------------------------------------------------------

def _condense_table_text(text: str) -> str:
    """Collapse pdftotext -layout whitespace runs so the [보장명|지급사유|지급금액]
    table survives truncation; drop page-break form feeds and running footers."""
    text = text.replace("\f", "\n")
    text = re.sub(r"[ \t]{2,}", " | ", text)      # column gaps -> pipe
    text = re.sub(r"\n\s*\n+", "\n", text)
    text = re.sub(r"(?m)^\s*-\s*\d+\s*-\s*$\n?", "", text)   # "- 12 -" page numbers
    # Drop exact duplicate lines (KB tables repeat identical rows per 종/형 variant)
    seen: set[str] = set()
    kept = []
    for line in text.split("\n"):
        key = line.strip()
        if len(key) > 12 and key in seen:
            continue
        seen.add(key)
        kept.append(line)
    return "\n".join(kept).strip()


def extract_coverage_section_for_llm(pdf_path: Path, max_chars: int = 4000) -> str:
    """Condensed text for Node 1 (Haiku). Budget goes mostly to 보장내용."""
    s = extract_sections(pdf_path)
    parts = []
    if s.product_name:
        parts.append(f"[상품명] {s.product_name}")
    if s.insurer:
        parts.append(f"[보험사] {s.insurer}")

    budget = max_chars - sum(len(p) for p in parts) - 120
    cov = _condense_table_text(s.보장내용)
    exc = _condense_table_text(s.면책사항)
    spec = _condense_table_text(s.특이사항)

    exc_budget = min(len(exc), int(budget * 0.2))
    spec_budget = min(len(spec), int(budget * 0.1))
    cov_budget = max(budget - exc_budget - spec_budget, 1500)

    if spec:
        parts.append(f"[상품 특이사항]\n{spec[:spec_budget]}")
    if cov:
        parts.append(f"[보험금 지급사유 및 지급금액]\n{cov[:cov_budget]}")
    if exc:
        parts.append(f"[보험금 지급제한(면책) 사항]\n{exc[:exc_budget]}")
    return "\n\n".join(parts)
