"""
PDF Text Extractor for MetLife Product Summaries.

Extracts structured text sections from product summary PDFs.
Handles Korean text with proper section boundary detection.
"""

import logging
import re
from pathlib import Path
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ExtractedSections:
    product_name: str = ""
    full_text: str = ""
    특이사항: str = ""
    가입자격: str = ""
    보장내용: str = ""       # 보험금 지급사유 및 지급제한사항
    면책사항: str = ""
    page_count: int = 0
    extraction_method: str = "pdftotext"


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract full text from PDF.

    Tries pdftotext first (better layout), falls back to pypdf.
    """
    import shutil

    if shutil.which("pdftotext"):
        try:
            import subprocess
            result = subprocess.run(
                ["pdftotext", "-layout", str(pdf_path), "-"],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout
        except Exception as e:
            logger.warning("pdftotext failed: %s", e)

    return _extract_with_pypdf(pdf_path)


def _extract_with_pypdf(pdf_path: Path) -> str:
    """Extract text using pypdf."""
    from pypdf import PdfReader
    reader = PdfReader(str(pdf_path))
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text


def get_page_count(pdf_path: Path) -> int:
    """Get PDF page count."""
    try:
        from pypdf import PdfReader
        reader = PdfReader(str(pdf_path))
        return len(reader.pages)
    except Exception:
        return 0


def extract_sections(pdf_path: Path) -> ExtractedSections:
    """Extract and split PDF into structured sections.

    Identifies key section boundaries:
    - 상품의 특이사항
    - 보험가입자격요건
    - 보험금 지급사유 및 지급제한사항
    - 면책사항 / 보험금을 지급하지 않는 사유
    """
    full_text = extract_text_from_pdf(pdf_path)
    page_count = get_page_count(pdf_path)

    # Extract product name from first page
    product_name = _extract_product_name(full_text)

    # Section markers
    markers = [
        ("특이사항", r"상품의\s*특이사항"),
        ("가입자격", r"보험가입\s*자격요건"),
        ("보장내용", r"보험금\s*지급사유\s*및\s*지급제한"),
        ("면책사항", r"면책사항|보험금을\s*지급하지\s*않는|일반적인\s*보험금\s*지급제한\s*사유"),
    ]

    sections = {}
    positions = []

    for key, pattern in markers:
        match = re.search(pattern, full_text)
        if match:
            positions.append((match.start(), key))

    # Sort by position
    positions.sort(key=lambda x: x[0])

    # Extract text between markers
    for i, (pos, key) in enumerate(positions):
        if i + 1 < len(positions):
            next_pos = positions[i + 1][0]
            sections[key] = full_text[pos:next_pos].strip()
        else:
            # Last section: take remaining text (but cap at reasonable length)
            sections[key] = full_text[pos:pos + 10000].strip()

    return ExtractedSections(
        product_name=product_name,
        full_text=full_text,
        특이사항=sections.get("특이사항", ""),
        가입자격=sections.get("가입자격", ""),
        보장내용=sections.get("보장내용", ""),
        면책사항=sections.get("면책사항", ""),
        page_count=page_count,
    )


def _extract_product_name(text: str) -> str:
    """Extract product name from first page text."""
    # The name appears after "보험약관 등" in the first page
    first_page = text[:2000]

    # Try pattern: "보험약관 등 <product name>의 기초서류"
    match = re.search(
        r"보험약관\s*등\s*(무배당[\s\S]*?)의\s*기초서류",
        first_page,
    )
    if match:
        name = match.group(1)
        # Clean: remove line breaks, extra spaces
        name = re.sub(r"\s+", " ", name).strip()
        return name

    # Fallback: find "무배당" in first 20 lines
    for line in text.split("\n")[:20]:
        line = line.strip()
        if "무배당" in line and len(line) > 10:
            return line.strip()

    return ""


def extract_coverage_section_for_llm(pdf_path: Path, max_chars: int = 4000) -> str:
    """Extract the coverage section optimized for LLM parsing.

    Returns a condensed version of the 보장내용 section,
    trimmed to fit within token limits for Haiku.
    """
    sections = extract_sections(pdf_path)

    # Build context for LLM
    parts = []

    if sections.product_name:
        parts.append(f"[상품명] {sections.product_name}")

    if sections.특이사항:
        # Take first 1500 chars of 특이사항 (has product overview)
        parts.append(f"[상품 특이사항]\n{sections.특이사항[:1500]}")

    if sections.보장내용:
        # This is the main section - give it most of the budget
        remaining = max_chars - sum(len(p) for p in parts) - 200
        parts.append(f"[보험금 지급사유]\n{sections.보장내용[:max(remaining, 3000)]}")

    if sections.면책사항:
        parts.append(f"[면책사항]\n{sections.면책사항[:1000]}")

    return "\n\n".join(parts)
