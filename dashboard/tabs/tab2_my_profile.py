"""Tab 2: My Profile - Demographic profile input & interest area radar chart."""

import streamlit as st
import plotly.graph_objects as go

from core_pipeline.profiler.health_profiler import HealthProfiler, UserProfile


def render():
    st.subheader("생활·건강 프로파일 입력")
    st.caption("입력 정보는 관심 영역 분석에만 사용되며, 개인 식별 정보를 수집하지 않습니다.")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("나이", min_value=18, max_value=80, value=30)
        gender = st.selectbox("성별", ["M", "F"], format_func=lambda x: "남성" if x == "M" else "여성")
        occupation = st.selectbox(
            "직업 분류",
            ["사무직", "생산직", "자영업", "전문직", "학생", "주부", "기타"],
        )

    with col2:
        family_size = st.number_input("가족 수", min_value=1, max_value=10, value=1)
        has_children = st.checkbox("자녀 유무")
        health_concerns_text = st.text_input(
            "건강 관심사 (쉼표로 구분)",
            placeholder="예: 허리, 눈, 스트레스",
        )

    health_concerns = [c.strip() for c in health_concerns_text.split(",") if c.strip()]

    # Store profile
    profile_data = {
        "age": age,
        "gender": gender,
        "occupation_category": occupation,
        "family_size": family_size,
        "has_children": has_children,
        "health_concerns": health_concerns,
        "chronic_conditions": [],
    }
    st.session_state["user_profile"] = profile_data

    if st.button("프로파일 분석", type="primary"):
        profiler = HealthProfiler()
        profile = UserProfile(**profile_data)
        interest = profiler.generate_interest_vector(profile)

        st.session_state["interest_vector"] = interest.to_dict()
        st.session_state["top_interests"] = interest.top_interests
        st.session_state["shap_explanations"] = interest.shap_explanations

    # Display results
    if "interest_vector" in st.session_state:
        st.divider()
        st.subheader("관심 영역 분석 결과")

        iv = st.session_state["interest_vector"]
        categories = list(iv.keys())
        scores = list(iv.values())

        # Radar chart
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=scores + [scores[0]],
            theta=categories + [categories[0]],
            fill="toself",
            name="관심도",
            line_color="#1f77b4",
        ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=False,
            title="생활·건강 관심 영역 레이더 차트",
            height=500,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Top interests
        top = st.session_state.get("top_interests", [])
        if top:
            st.info(f"상위 관심 영역: **{', '.join(top)}**")

        # SHAP explanations
        shap = st.session_state.get("shap_explanations", {})
        if shap:
            model_type = shap.get("model", "rule-based")
            with st.expander("분석 근거 (SHAP-like 설명)"):
                st.caption(f"모델: {model_type}")

                if "feature_contributions" in shap:
                    # XGBoost mode: show feature importances
                    fc = shap["feature_contributions"]
                    feature_labels = {
                        "age": "연령", "gender": "성별", "occupation": "직업",
                        "family_size": "가족 수", "has_children": "자녀 유무",
                    }
                    # Show top category's feature importance
                    top_cat = st.session_state.get("top_interests", ["입원"])[0]
                    if top_cat in fc:
                        st.write(f"**{top_cat}** 관심도 주요 요인:")
                        sorted_feats = sorted(fc[top_cat].items(), key=lambda x: -x[1])
                        for feat, imp in sorted_feats:
                            label = feature_labels.get(feat, feat)
                            bar_len = int(imp * 20)
                            st.write(f"- {label}: {'█' * bar_len} {imp:.3f}")
                else:
                    # Rule-based mode
                    st.write(f"- 연령대: {shap.get('age_group', '-')}")
                    occ_effect = shap.get("occupation_effect", {})
                    if occ_effect:
                        st.write(f"- 직업 영향: {occ_effect}")

                boosts = shap.get("health_concern_boosts", [])
                if boosts:
                    st.write(f"- 건강 관심사 반영: {', '.join(boosts)}")
