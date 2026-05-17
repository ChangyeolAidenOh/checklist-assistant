"""Tab 3: Check Items - Coverage vs interest area comparison + priority ranking."""

import streamlit as st
import plotly.graph_objects as go

from config import DISCLAIMER


def render():
    st.subheader("확인 필요 항목 분석")
    st.caption(DISCLAIMER)

    # Check prerequisites
    has_coverage = "parsed_coverage" in st.session_state
    has_profile = "interest_vector" in st.session_state

    if not has_coverage or not has_profile:
        missing = []
        if not has_coverage:
            missing.append("My Coverage 탭에서 상품 분석")
        if not has_profile:
            missing.append("My Profile 탭에서 프로파일 분석")
        st.warning(f"먼저 다음 단계를 완료해주세요: {', '.join(missing)}")
        return

    if st.button("확인 항목 분석 시작", type="primary"):
        with st.spinner("보장 범위와 관심 영역을 대조하고 있습니다..."):
            from core_pipeline.parser.coverage_parser import CoverageParser
            from core_pipeline.profiler.health_profiler import InterestVector, INTEREST_CATEGORIES
            from core_pipeline.detector.gap_detector import GapDetector

            parser = CoverageParser()
            parsed = parser._to_parsed_coverage(st.session_state["parsed_coverage"])

            iv_data = st.session_state["interest_vector"]
            interest = InterestVector(
                categories=INTEREST_CATEGORIES,
                scores=[iv_data.get(cat, 0.0) for cat in INTEREST_CATEGORIES],
                top_interests=sorted(iv_data, key=iv_data.get, reverse=True)[:3],
            )

            detector = GapDetector()
            gap = detector.analyze(parsed, interest)

            st.session_state["gap_analysis"] = {
                "check_items": [
                    {
                        "category": item.category,
                        "priority_score": item.priority_score,
                        "status": item.status,
                        "plain_description": item.plain_description,
                        "consultation_note": item.consultation_note,
                        "interest_score": item.interest_score,
                        "coverage_detail": item.coverage_detail,
                    }
                    for item in gap.check_items
                ],
                "covered_count": gap.covered_count,
                "check_needed_count": gap.check_needed_count,
                "total_categories": gap.total_categories,
            }

    # Display results
    if "gap_analysis" in st.session_state:
        ga = st.session_state["gap_analysis"]
        st.divider()

        # Summary metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("분석 영역", f"{ga['total_categories']}개")
        with col2:
            st.metric("보장 확인됨", f"{ga['covered_count']}개")
        with col3:
            st.metric("확인 필요", f"{ga['check_needed_count']}개")

        # Priority bar chart
        items = ga["check_items"]
        if items:
            categories = [i["category"] for i in items]
            priorities = [i["priority_score"] for i in items]
            statuses = [i["status"] for i in items]

            color_map = {
                "covered": "#2ecc71",
                "partial": "#f39c12",
                "not_covered": "#e74c3c",
                "unknown": "#95a5a6",
            }
            colors = [color_map.get(s, "#95a5a6") for s in statuses]

            fig = go.Figure(go.Bar(
                x=priorities,
                y=categories,
                orientation="h",
                marker_color=colors,
                text=[f"{p}/10" for p in priorities],
                textposition="auto",
            ))
            fig.update_layout(
                title="상담 우선순위",
                xaxis_title="우선순위 점수",
                yaxis_title="",
                height=400,
                xaxis=dict(range=[0, 10]),
            )
            st.plotly_chart(fig, use_container_width=True)

            # Legend
            st.markdown(
                "🟢 보장 확인됨 &nbsp; 🟡 일부 보장 &nbsp; 🔴 보장 미확인 &nbsp; ⚪ 확인 불가"
            )

            # Detailed items
            st.subheader("항목별 상세")
            for item in items:
                status_icon = {"covered": "✅", "partial": "⚠️", "not_covered": "❌", "unknown": "❓"}
                icon = status_icon.get(item["status"], "❓")
                with st.expander(
                    f"{icon} {item['category']} — 우선순위 {item['priority_score']}/10"
                ):
                    st.write(f"**설명:** {item['plain_description']}")
                    st.write(f"**상태:** {item['status']}")
                    st.write(f"**관심도:** {item['interest_score']:.2f}")
                    if item["coverage_detail"]:
                        st.write(f"**보장 내용:** {item['coverage_detail']}")
                    st.info(item["consultation_note"])
