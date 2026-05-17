"""
Step 2: Node 1 (Coverage Parser) test.

Requires: ANTHROPIC_API_KEY in .env

Tests:
  - Haiku API connection
  - Product summary -> structured JSON parsing
  - Coverage item extraction
  - Plain language conversion
  - Parse confidence assessment

Run: python scripts/test_node1_parser.py
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Load .env
from dotenv import load_dotenv
load_dotenv()

from core_pipeline.parser.coverage_parser import CoverageParser


SAMPLE_HEALTH_INSURANCE = """
[상품명] 무배당 메트라이프 건강보험
[보험사] 메트라이프생명보험

[주요 보장 내용]
1. 일반사망: 사망 시 1억원 지급
2. 암 진단: 일반암 진단 시 3,000만원 (90일 면책)
3. 뇌혈관질환 진단: 진단 시 2,000만원
4. 심장질환 진단: 진단 시 2,000만원
5. 입원 일당: 질병/상해 입원 시 1일당 3만원 (120일 한도)
6. 수술비: 질병/상해 수술 시 건당 50만원

[면책 사항]
- 암 진단: 계약일로부터 90일 이내 진단 시 면책
- 기왕증(계약 전 알릴 의무 위반): 보장 제외 가능
- 음주운전 사고: 보장 제외

[보험료]
- 월 보험료: 45,000원 (30세, 남성 기준)
- 납입 기간: 20년
- 보장 기간: 100세까지
"""

SAMPLE_SILSON = """
[상품명] 무배당 메트라이프 실손의료보험
[보험사] 메트라이프생명보험

[주요 보장 내용]
1. 입원 의료비: 질병/상해 입원 시 5,000만원 한도 (본인부담금 20%)
2. 외래 의료비: 질병/상해 통원 시 건당 25만원 한도
3. 처방조제비: 건당 5만원 한도
4. 상급병실료 차액: 1일당 10만원, 120일 한도

[면책 사항]
- 미용/성형 목적 진료
- 건강검진 비용
- 예방접종 비용
- 치과 보철/임플란트 (별도 특약)

[보험료]
- 월 보험료: 32,000원 (30세, 남성 기준)
- 납입 기간: 15년
- 보장 기간: 갱신형 (15년 단위)
"""


def test_api_connection():
    """Verify Anthropic API key works."""
    print("=== Test 0: API connection ===")
    try:
        parser = CoverageParser()
        # Quick probe: tiny prompt
        response = parser.backend.generate(
            prompt="Say OK",
            max_tokens=10,
        )
        assert len(response) > 0
        print(f"  Haiku response: {response.strip()}")
        print("PASSED")
        print()
        return True
    except Exception as e:
        print(f"FAILED: {e}")
        print("  .env file exists and ANTHROPIC_API_KEY is set?")
        return False


def test_health_insurance():
    """Parse sample health insurance product."""
    print("=== Test 1: 건강보험 파싱 ===")
    parser = CoverageParser()
    result = parser.parse(SAMPLE_HEALTH_INSURANCE)

    print(f"  Product: {result.product_name}")
    print(f"  Insurer: {result.insurer}")
    print(f"  Type: {result.product_type}")
    print(f"  Confidence: {result.parse_confidence}")
    print(f"  Coverage items: {len(result.coverage_items)}")
    print(f"  Exclusions: {len(result.exclusions)}")

    # Assertions
    assert result.product_name, "product_name should not be empty"
    assert "메트라이프" in result.insurer, f"insurer should contain 메트라이프, got {result.insurer}"
    assert len(result.coverage_items) >= 4, f"Expected 4+ items, got {len(result.coverage_items)}"
    assert len(result.exclusions) >= 2, f"Expected 2+ exclusions, got {len(result.exclusions)}"

    # Check plain language conversion
    print("\n  Coverage items:")
    for item in result.coverage_items:
        print(f"    [{item.category}] {item.item_name}")
        print(f"      -> {item.plain_description}")
        print(f"      Amount: {item.benefit_amount}")
        assert item.plain_description, f"plain_description empty for {item.item_name}"

    print("\n  Exclusions:")
    for exc in result.exclusions:
        print(f"    - {exc.get('item', '')}: {exc.get('plain_description', '')}")

    print("\n  Premium:")
    print(f"    {result.premium_info}")

    print("PASSED")
    print()
    return result


def test_silson_insurance():
    """Parse sample 실손보험 product."""
    print("=== Test 2: 실손보험 파싱 ===")
    parser = CoverageParser()
    result = parser.parse(SAMPLE_SILSON)

    print(f"  Product: {result.product_name}")
    print(f"  Confidence: {result.parse_confidence}")
    print(f"  Coverage items: {len(result.coverage_items)}")

    assert result.product_name, "product_name should not be empty"
    assert len(result.coverage_items) >= 3, f"Expected 3+ items, got {len(result.coverage_items)}"

    # 실손 should have 통원/외래 coverage
    categories = [item.category for item in result.coverage_items]
    print(f"  Categories found: {categories}")

    print("PASSED")
    print()
    return result


def test_parse_failure_fallback():
    """Verify graceful fallback on garbage input."""
    print("=== Test 3: Fallback on invalid input ===")
    parser = CoverageParser()
    result = parser.parse("이것은 보험 문서가 아닙니다. 그냥 텍스트입니다.")

    print(f"  Confidence: {result.parse_confidence}")
    print(f"  Coverage items: {len(result.coverage_items)}")

    # Should not crash, should indicate low confidence or few items
    print("PASSED (no crash on invalid input)")
    print()


if __name__ == "__main__":
    if not test_api_connection():
        print("API connection failed. Stopping.")
        sys.exit(1)

    test_health_insurance()
    test_silson_insurance()
    test_parse_failure_fallback()

    print("=" * 40)
    print("All Node 1 tests PASSED")
    print("Note: Check plain_description quality manually")
