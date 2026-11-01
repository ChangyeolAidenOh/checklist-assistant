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
[상품명] KB 5.10.10 플러스 건강보험(세만기)(무배당)(26.09)
[보험사] KB손해보험
[상품유형] 종합건강보험 (상해·질병 보장, 세만기)

[주요 보장 내용]
□ 기본계약
1. 일반상해후유장해(20~100%): 상해사고로 20~100% 후유장해 시 가입금액 × 지급률
2. 일반상해사망: 상해사고로 사망 시 가입금액

□ 의무부가계약
3. 보험료납입면제대상보장(12대): 상해·질병 50% 이상 후유장해, 암(유사암 제외)·뇌혈관질환·허혈성심장질환·말기폐질환·말기간경화·말기신부전증·양성뇌종양·중대한재생불량성빈혈·만성당뇨합병증·중대한화상 진단 확정 시 최초 1회 가입금액 지급 후 차회 이후 보험료 납입 면제

□ 선택특약 (예시)
4. 골절진단비Ⅱ: 상해로 골절(치아파절 제외) 진단 시 특약 가입금액
5. 5대골절진단비 / 5대골절수술비: 척추·골반·대퇴골·발목·손목 골절 진단·수술 시 특약 가입금액
6. 상해수술비(종합병원): 상해로 종합병원에서 수술 시 특약 가입금액
7. 상해중환자실입원일당: 상해로 중환자실 입원 시 1일당 특약 가입금액
8. 화상진단비 / 화상수술비: 화상 진단·수술 시 특약 가입금액
9. 치아파절진단비: 상해로 치아파절 진단 시 특약 가입금액
10. 상해흉터복원수술비Ⅱ: 상해로 인한 흉터 복원수술 시 특약 가입금액

[상품 특이사항]
- 보장부분 적용이율 3.0%, 적립부분 공시이율 적용 (최저보증 연 0.2%)
- 1형(해약환급금 지급형) / 2형(해약환급금 미지급형) 선택, 납입면제 종류(1종·2종·4종)에 따라 납입면제 사유가 다름
- 암보장개시일은 계약일로부터 90일이 지난 날의 다음날

[면책 사항]
- 암 담보는 보장개시일 이전 진단 시 보장하지 않음
- 피보험자·계약자·수익자의 고의 사고, 자해, 전쟁·폭동 등 약관상 면책 사유
- 담보별 지급제한 사항은 보통약관 및 특별약관 본문에 따름

[보험료]
- 요약서 미기재 (가입설계 시 산출)
"""

SAMPLE_SILSON = """
[상품명] KB손보 실손의료비보장보험(무배당)(26.09)
[보험사] KB손해보험
[상품유형] 실손의료보험 (4세대, 갱신형)

[주요 보장 내용]
1. 상해급여 입원의료비: 상해로 입원하여 급여 치료 시 본인부담금의 80% (가입금액 1천만/3천만/5천만원 한도)
2. 상해급여 통원의료비: 상해로 통원·처방조제 시 본인부담금에서 병원규모별 1~2만원과 보장대상의료비의 20% 중 큰 금액을 차감 (가입금액별 1회당 한도 10만/15만/20만원)
3. 질병급여 입원의료비: 질병으로 입원하여 급여 치료 시 본인부담금의 80% (1천만/3천만/5천만원 한도)
4. 질병급여 통원의료비: 질병으로 통원·처방조제 시 본인부담금에서 1~2만원과 20% 중 큰 금액 차감
5. 3대 비급여 특별약관: 도수·체외충격파·증식치료(연간 50회), 비급여 주사료(연간 50회, 250만원 한도), 자기공명영상진단(300만원 한도)
6. 비중증 비급여 특별약관: 비급여 입원 치료 시 본인부담금의 50% (의원급 1회당 300만원 한도), 비급여 통원 시 5만원과 50% 중 큰 금액 차감 (연간 100일 한도)
7. 상급병실료 차액: 비급여 병실료의 50% (1일 평균 10만원 한도)

[상품 특이사항]
- 실손의료보험은 손해율에 따라 매년 보험료가 변동되며, 1년 자동갱신
- 보장내용 변경주기 최대 5년(재가입), 최대 99세까지 재가입·100세까지 보장
- 건강보험 비급여 대상 및 자기부담금은 재가입 시점 법령에 따라 변경될 수 있음

[면책 사항]
- 실손의료비를 보장하는 다수의 보험계약이 있는 경우 각 계약의 비례분담액만 지급
- 회사가 제1회 보험료를 받은 때부터 보장 개시
- 담보별 보험금을 지급하지 않는 사항은 보통약관에 따름

[보험료]
- 요약서 미기재 (가입설계 시 연령·성별·가입금액 기준으로 산출)
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
    print("=== Test 1: KB 5.10.10 플러스 건강보험 파싱 ===")
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
    assert "KB" in result.insurer, f"insurer should contain KB, got {result.insurer}"
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
    print("=== Test 2: KB손보 실손의료비보장보험 파싱 ===")
    parser = CoverageParser()
    result = parser.parse(SAMPLE_SILSON)

    print(f"  Product: {result.product_name}")
    print(f"  Confidence: {result.parse_confidence}")
    print(f"  Coverage items: {len(result.coverage_items)}")

    assert result.product_name, "product_name should not be empty"
    assert len(result.coverage_items) >= 3, f"Expected 3+ items, got {len(result.coverage_items)}"

    # 실손 should have 입원/통원 coverage
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
