"""Tab 4: Consultation Checklist - Final checklist for planner consultation."""

import streamlit as st

from config import DISCLAIMER


def render():
    st.subheader("상담 전 체크리스트")
    st.caption(DISCLAIMER)

    if "gap_analysis" not in st.session_state:
        st.warning("먼저 Check Items 탭에서 확인 항목 분석을 완료해주세요.")
        return

    if st.button("체크리스트 생성", type="primary"):
        with st.spinner("상담 전 체크리스트를 생성하고 있습니다..."):
            from core_pipeline.detector.gap_detector import GapAnalysis, CheckItem
            from core_pipeline.generator.checklist_generator import ChecklistGenerator

            ga_data = st.session_state["gap_analysis"]
            gap = GapAnalysis(
                check_items=[CheckItem(**item) for item in ga_data["check_items"]],
                covered_count=ga_data["covered_count"],
                check_needed_count=ga_data["check_needed_count"],
                total_categories=ga_data["total_categories"],
            )

            product_name = st.session_state.get("parsed_coverage", {}).get("product_name", "")
            generator = ChecklistGenerator()
            checklist = generator.generate(gap, product_name)

            st.session_state["checklist"] = {
                "title": checklist.title,
                "summary": checklist.summary,
                "high_priority": [
                    {"item": i.item, "description": i.description,
                     "question_for_planner": i.question_for_planner}
                    for i in checklist.high_priority
                ],
                "medium_priority": [
                    {"item": i.item, "description": i.description,
                     "question_for_planner": i.question_for_planner}
                    for i in checklist.medium_priority
                ],
                "informational": [
                    {"item": i.item, "description": i.description,
                     "question_for_planner": i.question_for_planner}
                    for i in checklist.informational
                ],
                "disclaimer": checklist.disclaimer,
            }

    if "checklist" in st.session_state:
        cl = st.session_state["checklist"]
        st.divider()

        st.markdown(f"### {cl.get('title', '상담 전 체크리스트')}")
        st.write(cl.get("summary", ""))

        # High priority
        high = cl.get("high_priority", [])
        if high:
            st.markdown("#### 🔴 우선 확인 항목")
            for i, item in enumerate(high, 1):
                st.markdown(f"**{i}. {item['item']}**")
                st.write(item["description"])
                st.success(f"💬 설계사에게 물어볼 질문: {item['question_for_planner']}")

        # Medium priority
        medium = cl.get("medium_priority", [])
        if medium:
            st.markdown("#### 🟡 추가 확인 항목")
            for i, item in enumerate(medium, 1):
                st.markdown(f"**{i}. {item['item']}**")
                st.write(item["description"])
                st.info(f"💬 설계사에게 물어볼 질문: {item['question_for_planner']}")

        # Informational
        info = cl.get("informational", [])
        if info:
            st.markdown("#### 🔵 참고 사항")
            for i, item in enumerate(info, 1):
                with st.expander(f"{i}. {item['item']}"):
                    st.write(item["description"])
                    st.write(f"💬 {item['question_for_planner']}")

        # Disclaimer
        st.divider()
        st.warning(cl.get("disclaimer", DISCLAIMER))
