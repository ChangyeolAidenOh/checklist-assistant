"""
Step 4: Node 4 (Checklist Generator) + Guardrail test.

Requires: ANTHROPIC_API_KEY in .env (Sonnet call)

Tests:
  - Sonnet API checklist generation
  - Priority grouping (high / medium / informational)
  - Guardrail: recommendation language blocked
  - Guardrail: disclaimer always present
  - Fallback: works without LLM when generation fails

Run: python scripts/test_node4_generator.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from core_pipeline.detector.gap_detector import GapAnalysis, CheckItem
from core_pipeline.generator.checklist_generator import ChecklistGenerator
from core_pipeline.guardrail.filter import GuardrailFilter
from config import RECOMMENDATION_BLOCKLIST


def make_sample_gap_analysis() -> GapAnalysis:
    """Simulates Node 3 output."""
    return GapAnalysis(
        check_items=[
            CheckItem(
                category="치과",
                priority_score=6.0,
                status="not_covered",
                plain_description="치과 진료 비용",
                consultation_note="치과 관련 보장이 현재 포함되어 있지 않습니다. 필요 여부를 설계사와 상의해보세요.",
                interest_score=0.60,
            ),
            CheckItem(
                category="정신건강",
                priority_score=6.0,
                status="not_covered",
                plain_description="정신건강 관련 진료 비용",
                consultation_note="정신건강 관련 보장이 현재 포함되어 있지 않습니다. 필요 여부를 설계사와 상의해보세요.",
                interest_score=0.60,
            ),
            CheckItem(
                category="외래/통원",
                priority_score=5.5,
                status="not_covered",
                plain_description="외래로 진료받는 경우의 비용",
                consultation_note="외래/통원 관련 보장이 현재 포함되어 있지 않습니다. 필요 여부를 설계사와 상의해보세요.",
                interest_score=0.55,
            ),
            CheckItem(
                category="입원",
                priority_score=0.7,
                status="covered",
                plain_description="입원 비용",
                consultation_note="입원 보장이 포함되어 있습니다. 보장 한도와 조건을 확인해보세요.",
                interest_score=0.35,
                coverage_detail="입원 일당: 3만원",
            ),
        ],
        covered_count=1,
        check_needed_count=3,
        total_categories=4,
    )


def test_checklist_generation():
    """Test Sonnet-powered checklist generation."""
    print("=== Test 1: Checklist generation (Sonnet) ===")

    generator = ChecklistGenerator()
    gap = make_sample_gap_analysis()
    checklist = generator.generate(gap, product_name="무배당 메트라이프 건강보험")

    print(f"  Title: {checklist.title}")
    print(f"  Summary: {checklist.summary}")
    print(f"  High priority: {len(checklist.high_priority)} items")
    print(f"  Medium priority: {len(checklist.medium_priority)} items")
    print(f"  Informational: {len(checklist.informational)} items")
    print(f"  Disclaimer: {checklist.disclaimer[:50]}...")
    print()

    # Display items
    for group_name, items in [
        ("HIGH", checklist.high_priority),
        ("MEDIUM", checklist.medium_priority),
        ("INFO", checklist.informational),
    ]:
        if items:
            print(f"  [{group_name}]")
            for item in items:
                print(f"    - {item.item}")
                print(f"      {item.description[:80]}...")
                print(f"      Q: {item.question_for_planner[:80]}...")
                print()

    # Assertions
    total_items = len(checklist.high_priority) + len(checklist.medium_priority) + len(checklist.informational)
    assert total_items > 0, "Should have at least 1 checklist item"
    assert checklist.disclaimer, "Disclaimer must be present"
    assert "설계사" in checklist.disclaimer or "상담" in checklist.disclaimer, \
        "Disclaimer should mention 설계사 or 상담"

    print("PASSED")
    print()


def test_guardrail_filter():
    """Test guardrail blocks recommendation language."""
    print("=== Test 2: Guardrail filter ===")

    gf = GuardrailFilter()

    # Should block
    dangerous_texts = [
        "이 상품을 추천합니다.",
        "치과 보험에 가입하세요.",
        "이 보험이 꼭 필요합니다.",
        "이 상품이 좋습니다. 바로 가입을 권합니다.",
    ]
    for text in dangerous_texts:
        is_safe, violations = gf.validate(text)
        assert not is_safe, f"Should block: '{text}'"
        filtered = gf.filter_text(text)
        assert "추천합니다" not in filtered and "가입하세요" not in filtered, \
            f"Blocked phrase leaked through: '{filtered}'"
        print(f"  BLOCKED: '{text[:40]}...' -> '{filtered[:40]}...'")

    # Should pass
    safe_texts = [
        "이 영역은 설계사와 상의해보세요.",
        "보장 한도를 확인해보시기 바랍니다.",
        "최종 판단은 전문 상담을 통해 이루어져야 합니다.",
    ]
    for text in safe_texts:
        is_safe, violations = gf.validate(text)
        assert is_safe, f"Should pass: '{text}', violations: {violations}"
        print(f"  SAFE: '{text[:50]}'")

    print("PASSED")
    print()


def test_fallback_checklist():
    """Test fallback works when LLM generation fails."""
    print("=== Test 3: Fallback checklist (no LLM) ===")

    generator = ChecklistGenerator()
    gap = make_sample_gap_analysis()

    # Call fallback directly
    fallback = generator._fallback_checklist(gap)

    print(f"  Title: {fallback.title}")
    print(f"  High: {len(fallback.high_priority)}, Medium: {len(fallback.medium_priority)}, Info: {len(fallback.informational)}")
    assert fallback.disclaimer, "Fallback should have disclaimer"

    total = len(fallback.high_priority) + len(fallback.medium_priority) + len(fallback.informational)
    assert total > 0, "Fallback should produce items"

    print("PASSED")
    print()


if __name__ == "__main__":
    test_guardrail_filter()    # no API needed
    test_fallback_checklist()  # no API needed
    test_checklist_generation()  # Sonnet API call

    print("=" * 40)
    print("All Node 4 tests PASSED")
    print("Cost: ~1 Sonnet call (~$0.01)")
