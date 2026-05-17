"""
ChromaDB RAG Ingestion & Test.

Ingests all 42 MetLife product summaries and tests retrieval.

Usage:
  # Ingest all PDFs
  python scripts/setup_rag.py

  # Force re-ingest
  python scripts/setup_rag.py --force

  # Ingest + test queries
  python scripts/setup_rag.py --test

Run: python scripts/setup_rag.py
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from core_pipeline.rag import CoverageRAG

PDF_DIR = Path(__file__).parent.parent / "data" / "product_summaries"


def ingest(force: bool = False):
    """Ingest all PDFs into ChromaDB."""
    print("=" * 50)
    print("ChromaDB RAG Ingestion")
    print("=" * 50)

    rag = CoverageRAG()

    # Show current state
    stats = rag.get_stats()
    print(f"Current state: {stats['total_chunks']} chunks from {stats['unique_sources']} sources")

    if force:
        print("Force mode: clearing existing data...")
        rag.clear()

    # Ingest
    result = rag.ingest_directory(PDF_DIR, force=force)
    print(f"\nIngestion results:")
    print(f"  Ingested: {result['ingested']} PDFs")
    print(f"  Skipped: {result['skipped']} (already in DB)")
    print(f"  Errors: {result['errors']}")
    print(f"  Total chunks: {result['total_chunks']}")

    # Final state
    stats = rag.get_stats()
    print(f"\nFinal state: {stats['total_chunks']} chunks from {stats['unique_sources']} sources")

    return rag


def test_queries(rag: CoverageRAG):
    """Test various RAG queries."""
    print("\n" + "=" * 50)
    print("RAG Query Tests")
    print("=" * 50)

    test_cases = [
        {
            "query": "암 진단 보험금 지급",
            "description": "General: cancer diagnosis coverage",
        },
        {
            "query": "치매 간병 생활자금",
            "description": "Specific: dementia care benefits",
        },
        {
            "query": "사망보험금 지급 조건",
            "description": "General: death benefit conditions",
        },
        {
            "query": "면책사항 보험금을 지급하지 않는 경우",
            "description": "Exclusions search",
        },
        {
            "query": "입원비 통원비 수술비",
            "description": "Medical expense coverage",
        },
    ]

    for tc in test_cases:
        print(f"\n--- {tc['description']} ---")
        print(f"Query: {tc['query']}")

        results = rag.search(tc["query"], top_k=3)

        if not results:
            print("  No results above threshold")
            continue

        for i, r in enumerate(results):
            print(f"  [{i+1}] score={r.score:.3f} | {r.source[:40]} | {r.section}")
            print(f"      {r.text[:100]}...")

    # Test product-specific search
    print(f"\n--- Product-specific: 360 치매간병보험 보장내용 ---")
    results = rag.search_for_coverage("360 치매간병")
    for i, r in enumerate(results[:3]):
        print(f"  [{i+1}] score={r.score:.3f} | {r.section}")
        print(f"      {r.text[:100]}...")

    print(f"\n--- Product-specific: 360 암보험 면책사항 ---")
    results = rag.search_for_exclusions("360 암보험")
    for i, r in enumerate(results[:3]):
        print(f"  [{i+1}] score={r.score:.3f} | {r.section}")
        print(f"      {r.text[:100]}...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Force re-ingest all")
    parser.add_argument("--test", action="store_true", help="Run test queries after ingestion")
    args = parser.parse_args()

    rag = ingest(force=args.force)

    if args.test:
        test_queries(rag)

    print("\nDone. RAG is ready for pipeline integration.")
