"""Tab 5: Pipeline Trace - Node-by-node execution trace + cost breakdown."""

import streamlit as st
import plotly.graph_objects as go


# Approximate cost per 1K tokens (USD)
COST_TABLE = {
    "Lightweight LLM (Haiku)": {"input": 0.0008, "output": 0.004},
    "High-quality LLM (Sonnet)": {"input": 0.003, "output": 0.015},
    "XGBoost (no LLM)": {"input": 0.0, "output": 0.0},
    "Rule-based + ML": {"input": 0.0, "output": 0.0},
}


def render():
    st.subheader("파이프라인 실행 추적")

    # Full pipeline run
    has_doc = "document_text" in st.session_state
    has_profile = "user_profile" in st.session_state

    if has_doc and has_profile:
        if st.button("전체 파이프라인 실행 (4-Node)", type="primary"):
            with st.spinner("4개 노드를 순차 실행하고 있습니다..."):
                from core_pipeline.pipeline import CoverageChecklistPipeline

                pipeline = CoverageChecklistPipeline()
                result = pipeline.run(
                    document_text=st.session_state["document_text"],
                    profile_data=st.session_state["user_profile"],
                )

                st.session_state["pipeline_trace"] = result.get("trace", [])
                st.session_state["parsed_coverage"] = result.get("parsed_coverage", {})
                st.session_state["interest_vector"] = result.get("interest_vector", {})
                st.session_state["gap_analysis"] = result.get("gap_analysis", {})
                st.session_state["checklist"] = result.get("checklist", {})

    elif not has_doc:
        st.info("My Coverage 탭에서 상품 요약서를 입력해주세요.")
    elif not has_profile:
        st.info("My Profile 탭에서 프로파일을 입력해주세요.")

    # Display trace
    if "pipeline_trace" in st.session_state:
        trace = st.session_state["pipeline_trace"]
        st.divider()

        # Timeline visualization
        nodes = [t["node"] for t in trace]
        times = [t["elapsed_sec"] for t in trace]
        tiers = [t["model_tier"] for t in trace]
        statuses = [t["status"] for t in trace]

        color_map = {"success": "#2ecc71", "fallback": "#f39c12", "error": "#e74c3c"}
        colors = [color_map.get(s, "#95a5a6") for s in statuses]

        fig = go.Figure(go.Bar(
            x=times,
            y=nodes,
            orientation="h",
            marker_color=colors,
            text=[f"{t:.3f}s" for t in times],
            textposition="auto",
        ))
        fig.update_layout(
            title="Node별 실행 시간",
            xaxis_title="소요 시간 (초)",
            height=300,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Detailed trace table
        st.subheader("실행 상세")
        for t in trace:
            tier = t["model_tier"]
            cost_info = COST_TABLE.get(tier, {"input": 0, "output": 0})
            estimated_cost = (cost_info["input"] + cost_info["output"]) * 2  # rough estimate

            status_icon = {"success": "✅", "fallback": "⚠️", "error": "❌"}.get(t["status"], "❓")

            with st.expander(f"{status_icon} {t['node']} — {t['elapsed_sec']:.3f}s"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.write(f"**모델 티어:** {tier}")
                with col2:
                    st.write(f"**상태:** {t['status']}")
                with col3:
                    if estimated_cost > 0:
                        st.write(f"**예상 비용:** ~${estimated_cost:.4f}")
                    else:
                        st.write("**비용:** $0 (LLM 미사용)")

        # Cost comparison
        st.divider()
        st.subheader("비용 거버넌스: 티어 분리 vs 전수 고성능")

        sonnet_cost = COST_TABLE["High-quality LLM (Sonnet)"]
        haiku_cost = COST_TABLE["Lightweight LLM (Haiku)"]

        # Assuming ~2K tokens per node call
        tokens_per_call = 2
        all_sonnet = (sonnet_cost["input"] + sonnet_cost["output"]) * tokens_per_call * 4
        tiered = (
            (haiku_cost["input"] + haiku_cost["output"]) * tokens_per_call  # Node 1
            + 0  # Node 2 (XGBoost)
            + 0  # Node 3 (rule-based)
            + (sonnet_cost["input"] + sonnet_cost["output"]) * tokens_per_call  # Node 4
        )

        savings = (1 - tiered / all_sonnet) * 100 if all_sonnet > 0 else 0

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("전수 Sonnet (4노드)", f"${all_sonnet:.4f}/건")
        with col2:
            st.metric("티어 분리 (현재)", f"${tiered:.4f}/건")
        with col3:
            st.metric("비용 절감", f"{savings:.0f}%")
