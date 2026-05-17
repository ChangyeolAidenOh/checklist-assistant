"""
Step 6: Real MetLife PDF Parsing Test.

Tests:
  1. PDF text extraction (section detection)
  2. LLM parsing of extracted coverage sections
  3. Parse quality assessment across 4 product types

Usage:
  # Test extraction only (no API cost)
  python scripts/test_pdf_parsing.py --extract-only

  # Full test with LLM parsing
  python scripts/test_pdf_parsing.py

  # Single PDF
  python scripts/test_pdf_parsing.py --file data/product_summaries/360_comprehensive.pdf

Run: python scripts/test_pdf_parsing.py
"""

import sys
import argparse
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from core_pipeline.parser.pdf_extractor import (
    extract_sections,
    extract_coverage_section_for_llm,
)
from core_pipeline.parser.coverage_parser import CoverageParser


PDF_DIR = Path(__file__).parent.parent / "data" / "product_summaries"


def find_pdfs() -> list[Path]:
    """Find all PDF files in product_summaries directory."""
    if not PDF_DIR.exists():
        print(f"Directory not found: {PDF_DIR}")
        return []

    pdfs = sorted(PDF_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {PDF_DIR}")
        print("Place MetLife product summary PDFs there and rerun.")
    return pdfs


def test_extraction(pdf_path: Path) -> dict:
    """Test PDF text extraction and section detection."""
    print(f"\n{'='*60}")
    print(f"PDF: {pdf_path.name}")
    print(f"{'='*60}")

    sections = extract_sections(pdf_path)

    print(f"  Product name: {sections.product_name}")
    print(f"  Pages: {sections.page_count}")
    print(f"  Full text length: {len(sections.full_text):,} chars")
    print(f"  Sections detected:")
    print(f"    특이사항:  {len(sections.특이사항):>6,} chars {'✓' if sections.특이사항 else '✗'}")
    print(f"    가입자격:  {len(sections.가입자격):>6,} chars {'✓' if sections.가입자격 else '✗'}")
    print(f"    보장내용:  {len(sections.보장내용):>6,} chars {'✓' if sections.보장내용 else '✗'}")
    print(f"    면책사항:  {len(sections.면책사항):>6,} chars {'✓' if sections.면책사항 else '✗'}")

    # Extract LLM-ready text
    llm_text = extract_coverage_section_for_llm(pdf_path)
    print(f"  LLM-ready text: {len(llm_text):,} chars")

    # Show first 300 chars of coverage section
    if sections.보장내용:
        print(f"\n  Coverage section preview:")
        for line in sections.보장내용[:500].split("\n")[:10]:
            if line.strip():
                print(f"    {line.strip()[:80]}")

    return {
        "file": pdf_path.name,
        "product_name": sections.product_name,
        "pages": sections.page_count,
        "sections_found": sum([
            bool(sections.특이사항),
            bool(sections.가입자격),
            bool(sections.보장내용),
            bool(sections.면책사항),
        ]),
        "total_sections": 4,
    }


def test_llm_parsing(pdf_path: Path) -> dict:
    """Test full LLM parsing of extracted PDF text."""
    print(f"\n  --- LLM Parsing (Haiku) ---")

    llm_text = extract_coverage_section_for_llm(pdf_path)
    parser = CoverageParser()
    result = parser.parse(llm_text)

    print(f"  Parse confidence: {result.parse_confidence}")
    print(f"  Coverage items: {len(result.coverage_items)}")
    print(f"  Exclusions: {len(result.exclusions)}")

    if result.coverage_items:
        print(f"  Items:")
        for item in result.coverage_items[:8]:
            print(f"    [{item.category}] {item.item_name}")
            print(f"      -> {item.plain_description[:60]}...")
            if item.benefit_amount:
                print(f"      Amount: {item.benefit_amount}")

    if result.exclusions:
        print(f"  Exclusions:")
        for exc in result.exclusions[:3]:
            print(f"    - {exc.get('item', '')}: {exc.get('plain_description', '')[:60]}...")

    if result.unparsed_sections:
        print(f"  Unparsed: {result.unparsed_sections}")

    return {
        "confidence": result.parse_confidence,
        "items": len(result.coverage_items),
        "exclusions": len(result.exclusions),
        "categories": list(set(item.category for item in result.coverage_items)),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--extract-only", action="store_true",
                        help="Only test extraction, no LLM calls")
    parser.add_argument("--file", type=str, help="Test single PDF file")
    parser.add_argument("--top4", action="store_true",
                        help="Test only the 4 target products")
    args = parser.parse_args()

    if args.file:
        pdfs = [Path(args.file)]
    else:
        pdfs = find_pdfs()
        if args.top4:
            keywords = ["360종합", "360암", "360치매", "모두의종신"]
            filtered = []
            for pdf in pdfs:
                import unicodedata
                name = unicodedata.normalize("NFC", pdf.name)
                if any(kw in name for kw in keywords):
                    filtered.append(pdf)
            pdfs = filtered
            print(f"Filtered to {len(pdfs)} target PDFs")

    if not pdfs:
        print("\nNo PDFs to test. Place files in data/product_summaries/")
        sys.exit(1)

    extraction_results = []
    parsing_results = []

    for pdf in pdfs:
        if not pdf.exists():
            print(f"File not found: {pdf}")
            continue

        try:
            # Always test extraction
            ext_result = test_extraction(pdf)
            extraction_results.append(ext_result)

            # Optionally test LLM parsing
            if not args.extract_only:
                parse_result = test_llm_parsing(pdf)
                parsing_results.append(parse_result)
        except Exception as e:
            print(f"  ERROR: {e}")
            extraction_results.append({
                "file": pdf.name, "product_name": "ERROR",
                "pages": 0, "sections_found": 0, "total_sections": 4,
            })

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    print(f"\nExtraction Results ({len(extraction_results)} PDFs):")
    total_sections = sum(r["sections_found"] for r in extraction_results)
    max_sections = sum(r["total_sections"] for r in extraction_results)
    print(f"  Section detection rate: {total_sections}/{max_sections} "
          f"({total_sections/max_sections*100:.0f}%)")

    for r in extraction_results:
        status = "✓" if r["sections_found"] == r["total_sections"] else "△"
        print(f"  {status} {r['file']}: {r['sections_found']}/{r['total_sections']} sections, "
              f"{r['pages']} pages")

    if parsing_results:
        print(f"\nLLM Parsing Results ({len(parsing_results)} PDFs):")
        for i, r in enumerate(parsing_results):
            conf_icon = {"high": "✓", "medium": "△", "low": "✗"}.get(r["confidence"], "?")
            print(f"  {conf_icon} {extraction_results[i]['file']}: "
                  f"confidence={r['confidence']}, "
                  f"items={r['items']}, "
                  f"categories={r['categories']}")
