"""Tab 1: My Coverage - Product summary input & parsed coverage display."""

import json
import streamlit as st

from config import DISCLAIMER


# Sample product summaries for demo.
# 내용은 KB손해보험 공시실 상품요약서(2026.07~09 판)에서 발췌·축약한 것이며,
# 보험료는 요약서에 기재되지 않아 포함하지 않음. 실제 보장은 약관 본문 기준.
SAMPLE_PRODUCTS = {
    "(선택) 직접 입력": None,
    "KB손보 실손의료비보장보험 (샘플)": """
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
""",
    "KB 5.10.10 플러스 건강보험 (샘플)": """
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
""",
    "KB The건강한 치아보험 (샘플)": """
[상품명] KB The건강한치아보험(무배당)(26.07)
[보험사] KB손해보험
[상품유형] 치아보험 (갱신형, 전기납)

[주요 보장 내용]
1. 치주질환치료비(기본계약, 갱신형): 보장개시일 이후 치주질환으로 치료 시 가입금액
2. 치아촬영비(X-ray 및 파노라마): 상해·질병으로 X-ray 또는 파노라마 촬영 시 각 촬영 1회당 가입금액
3. 발치치료비: 단순발치 가입금액의 25%, 정교발치 50%, 완전매복발치 100%
4. 영구치상실치료비: 상해·치아관련 질병으로 영구치 발거 진단 확정 시 영구치 1개당 가입금액
5. 특정 임플란트 치조골이식술 치료비: 상해 원인은 가입금액 100%, 질병 원인은 계약 2년 미만 50% / 2년 이상 100% (영구치 1개당)
6. 영구치보철치료비: 상해·치아관련 질병으로 보철(임플란트·브릿지·틀니) 치료 시 치료항목별 가입금액

[상품 특이사항]
- 갱신형이며 갱신종료나이 80세
- 갱신 시 보험료는 갱신일 당시 피보험자 나이·위험률 기준으로 재산출
- 세액공제 대상 보장성 보험

[면책 사항]
- 치료 원인이 질병인 치아 담보는 계약일로부터 90일이 지난 날의 다음날부터 보장 (상해는 계약일부터)
- 보장개시일 이전에 발생한 치아 질환·치료 중인 치아는 보장 제외
- 미용 목적 치료는 보장하지 않음

[보험료]
- 요약서 미기재 (가입설계 시 산출)
""",
}


def render():
    st.subheader("보험 상품 요약서 입력")
    st.caption(DISCLAIMER)

    # Product selection
    selected = st.selectbox(
        "KB손해보험 샘플 상품 선택 또는 직접 입력",
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
            if result.error:
                st.error(f"LLM 파싱 호출 실패 — {result.error}")

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
