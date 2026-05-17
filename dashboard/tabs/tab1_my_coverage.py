"""Tab 1: My Coverage - Product summary input & parsed coverage display."""

import json
import streamlit as st

from config import DISCLAIMER


# Sample MetLife product summaries for demo
SAMPLE_PRODUCTS = {
    "(선택) 직접 입력": None,
    "MetLife 무배당 건강보험 (샘플)": """
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
""",
    "MetLife 무배당 실손보험 (샘플)": """
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
""",
}


def render():
    st.subheader("보험 상품 요약서 입력")
    st.caption(DISCLAIMER)

    # Product selection
    selected = st.selectbox(
        "MetLife 샘플 상품 선택 또는 직접 입력",
        list(SAMPLE_PRODUCTS.keys()),
    )

    if selected == "(선택) 직접 입력":
        doc_text = st.text_area(
            "상품 요약서 텍스트를 붙여넣으세요",
            height=300,
            placeholder="보험 상품 요약서 내용을 여기에 붙여넣으세요...",
        )
    else:
        doc_text = SAMPLE_PRODUCTS[selected]
        st.text_area("상품 요약서 내용", value=doc_text, height=300, disabled=True)

    # Store in session state
    if doc_text:
        st.session_state["document_text"] = doc_text

    # Parse button
    if st.button("상품 분석 시작", type="primary", disabled=not doc_text):
        with st.spinner("상품 요약서를 분석하고 있습니다..."):
            from core_pipeline.parser.coverage_parser import CoverageParser
            parser = CoverageParser()
            result = parser.parse(doc_text)
            st.session_state["parsed_coverage"] = result.raw_json
            st.session_state["parse_confidence"] = result.parse_confidence

    # Display parsed results
    if "parsed_coverage" in st.session_state:
        parsed = st.session_state["parsed_coverage"]
        confidence = st.session_state.get("parse_confidence", "low")

        st.divider()
        st.subheader("분석 결과")

        # Confidence indicator
        conf_colors = {"high": "green", "medium": "orange", "low": "red"}
        conf_labels = {"high": "높음", "medium": "보통", "low": "낮음"}
        st.markdown(
            f"파싱 신뢰도: :{conf_colors.get(confidence, 'red')}[{conf_labels.get(confidence, '낮음')}]"
        )

        # Product info
        col1, col2 = st.columns(2)
        with col1:
            st.metric("상품명", parsed.get("product_name", "-"))
            st.metric("상품 유형", parsed.get("product_type", "-"))
        with col2:
            st.metric("보험사", parsed.get("insurer", "-"))
            premium = parsed.get("premium_info", {})
            st.metric("월 보험료", premium.get("monthly_premium", "-"))

        # Coverage items
        items = parsed.get("coverage_items", [])
        if items:
            st.subheader(f"보장 항목 ({len(items)}개)")
            for item in items:
                with st.expander(
                    f"**{item.get('category', '')}** — {item.get('item_name', '')}"
                ):
                    st.write(f"**일상 언어 설명:** {item.get('plain_description', '')}")
                    st.write(f"**보장 금액:** {item.get('benefit_amount', '')}")
                    if item.get("conditions"):
                        st.write(f"**조건:** {item['conditions']}")
                    if item.get("waiting_period"):
                        st.write(f"**대기기간:** {item['waiting_period']}")
                    st.caption(f"출처: {item.get('source_reference', '')}")

        # Exclusions
        exclusions = parsed.get("exclusions", [])
        if exclusions:
            st.subheader(f"면책 사항 ({len(exclusions)}개)")
            for exc in exclusions:
                st.warning(f"**{exc.get('item', '')}**: {exc.get('plain_description', '')}")

        # Raw JSON (collapsible)
        with st.expander("원본 JSON 데이터"):
            st.json(parsed)
