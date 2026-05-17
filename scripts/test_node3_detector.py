"""
Step 3: Node 3 (Gap Detector) test.

No API key needed. Uses hardcoded Node 1 + Node 2 outputs.

Tests:
  - Coverage vs interest area matching
  - Priority score calculation
  - Status detection (covered / not_covered / partial)
  - Consultation note generation

Run: python scripts/test_node3_detector.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core_pipeline.parser.coverage_parser import ParsedCoverage, CoverageItem
from core_pipeline.profiler.health_profiler import InterestVector, INTEREST_CATEGORIES
from core_pipeline.detector.gap_detector import GapDetector


def make_health_insurance_coverage() -> ParsedCoverage:
    """Simulates Node 1 output for 건강보험."""
    return ParsedCoverage(
        product_name="무배당 메트라이프 건강보험",
        insurer="메트라이프생명보험",
        product_type="건강보험",
        coverage_items=[
            CoverageItem(
                category="사망", item_name="일반사망",
                plain_description="사망 시 지급", benefit_amount="1억원",
            ),
            CoverageItem(
                category="진단", item_name="암 진단",
                plain_description="일반암 진단 시 지급", benefit_amount="3,000만원",
            ),
            CoverageItem(
                category="입원", item_name="입원 일당",
                plain_description="입원 시 1일당 지급", benefit_amount="3만원",
            ),
            CoverageItem(
                category="수술", item_name="수술비",
                plain_description="수술 시 지급", benefit_amount="50만원",
            ),
        ],
        parse_confidence="high",
        raw_json={
            "product_name": "무배당 메트라이프 건강보험",
            "coverage_items": [
                {"category": "사망", "item_name": "일반사망",
                 "plain_description": "사망 시 지급", "benefit_amount": "1억원",
                 "conditions": "", "waiting_period": "", "source_reference": ""},
                {"category": "진단", "item_name": "암 진단",
                 "plain_description": "일반암 진단 시 지급", "benefit_amount": "3,000만원",
                 "conditions": "", "waiting_period": "", "source_reference": ""},
                {"category": "입원", "item_name": "입원 일당",
                 "plain_description": "입원 시 1일당 지급", "benefit_amount": "3만원",
                 "conditions": "", "waiting_period": "", "source_reference": ""},
                {"category": "수술", "item_name": "수술비",
                 "plain_description": "수술 시 지급", "benefit_amount": "50만원",
                 "conditions": "", "waiting_period": "", "source_reference": ""},
            ],
            "exclusions": [],
            "premium_info": {},
            "parse_confidence": "high",
            "unparsed_sections": [],
        },
    )


def make_office_worker_interest() -> InterestVector:
    """Simulates Node 2 output for 30대 사무직 with 허리/스트레스 concerns."""
    scores = {
        "입원": 0.35, "외래/통원": 0.55, "수술": 0.15, "치과": 0.60,
        "정신건강": 0.60, "안과": 0.50, "산부인과": 0.45,
        "정형외과": 0.40, "피부과": 0.35, "건강검진": 0.40,
        "응급": 0.12, "재활": 0.15,
    }
    return InterestVector(
        categories=INTEREST_CATEGORIES,
        scores=[scores.get(cat, 0.0) for cat in INTEREST_CATEGORIES],
        top_interests=["정신건강", "치과", "외래/통원"],
    )


def test_gap_detection():
    """Core test: 건강보험 has 입원/수술 but no 치과/정신건강/통원."""
    print("=== Test 1: Gap detection (건강보험 vs 30대 사무직) ===")

    detector = GapDetector()
    coverage = make_health_insurance_coverage()
    interest = make_office_worker_interest()

    gap = detector.analyze(coverage, interest)

    print(f"  Total categories analyzed: {gap.total_categories}")
    print(f"  Covered: {gap.covered_count}")
    print(f"  Check needed: {gap.check_needed_count}")
    print()

    # Display all items
    for item in gap.check_items:
        icon = {"covered": "+", "not_covered": "X", "partial": "~", "unknown": "?"}
        print(f"  [{icon.get(item.status, '?')}] {item.category:8s} "
              f"priority={item.priority_score:4.1f}/10  "
              f"interest={item.interest_score:.2f}  "
              f"status={item.status}")

    print()

    # Assertions
    assert gap.total_categories > 0, "Should have analyzed some categories"
    assert gap.check_needed_count > 0, "건강보험 should have gaps"

    # 입원 should be covered
    hospitalization = next(
        (i for i in gap.check_items if i.category == "입원"), None
    )
    assert hospitalization is not None
    assert hospitalization.status == "covered", f"입원 should be covered, got {hospitalization.status}"

    # 치과 should NOT be covered (건강보험 doesn't have it)
    dental = next(
        (i for i in gap.check_items if i.category == "치과"), None
    )
    assert dental is not None
    assert dental.status == "not_covered", f"치과 should be not_covered, got {dental.status}"

    # 치과 priority should be high (high interest + not covered)
    assert dental.priority_score >= 5.0, f"치과 priority should be >= 5, got {dental.priority_score}"

    # 정신건강 should NOT be covered
    mental = next(
        (i for i in gap.check_items if i.category == "정신건강"), None
    )
    assert mental is not None
    assert mental.status == "not_covered"

    print("PASSED")
    print()


def test_consultation_notes():
    """Verify consultation notes are safe (no recommendation language)."""
    print("=== Test 2: Consultation note safety ===")

    detector = GapDetector()
    coverage = make_health_insurance_coverage()
    interest = make_office_worker_interest()
    gap = detector.analyze(coverage, interest)

    blocked_phrases = ["추천합니다", "가입하세요", "꼭 필요합니다", "이 상품이 좋습니다"]

    for item in gap.check_items:
        for phrase in blocked_phrases:
            assert phrase not in item.consultation_note, (
                f"Blocked phrase '{phrase}' found in note for {item.category}"
            )
        # Should contain safe language
        if item.status == "not_covered":
            assert "설계사" in item.consultation_note, (
                f"Not-covered note should mention 설계사: {item.consultation_note}"
            )

    print("  All notes safe - no recommendation language")
    print("PASSED")
    print()


def test_priority_ordering():
    """Verify items are sorted by priority (descending)."""
    print("=== Test 3: Priority ordering ===")

    detector = GapDetector()
    coverage = make_health_insurance_coverage()
    interest = make_office_worker_interest()
    gap = detector.analyze(coverage, interest)

    priorities = [item.priority_score for item in gap.check_items]
    assert priorities == sorted(priorities, reverse=True), (
        f"Items not sorted by priority: {priorities}"
    )

    print(f"  Priority order: {[f'{i.category}({i.priority_score})' for i in gap.check_items]}")
    print("PASSED")
    print()


if __name__ == "__main__":
    test_gap_detection()
    test_consultation_notes()
    test_priority_ordering()

    print("=" * 40)
    print("All Node 3 tests PASSED")
