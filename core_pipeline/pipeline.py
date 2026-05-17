"""
LangGraph Pipeline Orchestrator.

Connects 4 nodes in a StateGraph:
  Node 1 (Coverage Parser) -> Node 2 (Health Profiler) ->
  Node 3 (Gap Detector) -> Node 4 (Checklist Generator)

Pattern reused from consumer-signal-agentic-platform (Router -> Retriever -> Reporter).
"""

import logging
import time
from dataclasses import dataclass, field
from typing import TypedDict

from langgraph.graph import StateGraph, END

from core_pipeline.parser.coverage_parser import CoverageParser, ParsedCoverage
from core_pipeline.profiler.health_profiler import HealthProfiler, UserProfile, InterestVector
from core_pipeline.detector.gap_detector import GapDetector, GapAnalysis
from core_pipeline.generator.checklist_generator import ChecklistGenerator, ConsultationChecklist

logger = logging.getLogger(__name__)


class PipelineState(TypedDict, total=False):
    """Shared state across all pipeline nodes."""
    # Inputs
    document_text: str
    user_profile: dict

    # Node outputs
    parsed_coverage: dict
    interest_vector: dict
    gap_analysis: dict
    checklist: dict

    # Trace
    trace: list[dict]
    error: str | None


@dataclass
class NodeTrace:
    node_name: str
    model_tier: str
    elapsed_seconds: float
    input_size: int
    output_size: int
    status: str = "success"  # success | error | fallback


class CoverageChecklistPipeline:
    """End-to-end pipeline orchestrator."""

    def __init__(self):
        self.parser = CoverageParser()
        self.profiler = HealthProfiler()
        self.detector = GapDetector()
        self.generator = ChecklistGenerator()
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build LangGraph StateGraph."""
        workflow = StateGraph(PipelineState)

        # Add nodes
        workflow.add_node("parse_coverage", self._node_parse)
        workflow.add_node("build_profile", self._node_profile)
        workflow.add_node("detect_gaps", self._node_detect)
        workflow.add_node("generate_checklist", self._node_generate)

        # Linear flow
        workflow.set_entry_point("parse_coverage")
        workflow.add_edge("parse_coverage", "build_profile")
        workflow.add_edge("build_profile", "detect_gaps")
        workflow.add_edge("detect_gaps", "generate_checklist")
        workflow.add_edge("generate_checklist", END)

        return workflow.compile()

    def run(self, document_text: str, profile_data: dict) -> PipelineState:
        """Run the full pipeline.

        Args:
            document_text: Raw product summary text
            profile_data: User profile dict (age, gender, occupation, etc.)

        Returns:
            Final PipelineState with all node outputs
        """
        initial_state: PipelineState = {
            "document_text": document_text,
            "user_profile": profile_data,
            "trace": [],
        }

        result = self.graph.invoke(initial_state)
        return result

    def _node_parse(self, state: PipelineState) -> PipelineState:
        """Node 1: Parse coverage document."""
        start = time.time()

        parsed = self.parser.parse(state["document_text"])

        elapsed = time.time() - start
        state["parsed_coverage"] = parsed.raw_json
        state["trace"] = state.get("trace", []) + [
            {
                "node": "Coverage Parser",
                "model_tier": "Lightweight LLM (Haiku)",
                "elapsed_sec": round(elapsed, 2),
                "status": "success" if parsed.parse_confidence != "low" else "fallback",
            }
        ]
        return state

    def _node_profile(self, state: PipelineState) -> PipelineState:
        """Node 2: Build health profile."""
        start = time.time()

        profile_data = state["user_profile"]
        profile = UserProfile(
            age=profile_data.get("age", 30),
            gender=profile_data.get("gender", "M"),
            occupation_category=profile_data.get("occupation_category", "사무직"),
            family_size=profile_data.get("family_size", 1),
            has_children=profile_data.get("has_children", False),
            health_concerns=profile_data.get("health_concerns", []),
            chronic_conditions=profile_data.get("chronic_conditions", []),
        )

        interest = self.profiler.generate_interest_vector(profile)

        elapsed = time.time() - start
        state["interest_vector"] = interest.to_dict()
        state["trace"] = state.get("trace", []) + [
            {
                "node": "Health Profiler",
                "model_tier": "XGBoost (no LLM)",
                "elapsed_sec": round(elapsed, 4),
                "status": "success",
            }
        ]
        return state

    def _node_detect(self, state: PipelineState) -> PipelineState:
        """Node 3: Detect coverage gaps."""
        start = time.time()

        # Reconstruct objects from state dicts
        parsed = self.parser._to_parsed_coverage(state["parsed_coverage"])

        # Reconstruct InterestVector
        iv_data = state["interest_vector"]
        from core_pipeline.profiler.health_profiler import InterestVector, INTEREST_CATEGORIES
        interest = InterestVector(
            categories=INTEREST_CATEGORIES,
            scores=[iv_data.get(cat, 0.0) for cat in INTEREST_CATEGORIES],
            top_interests=sorted(iv_data, key=iv_data.get, reverse=True)[:3],
        )

        gap = self.detector.analyze(parsed, interest)

        elapsed = time.time() - start
        state["gap_analysis"] = {
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
        state["trace"] = state.get("trace", []) + [
            {
                "node": "Gap Detector",
                "model_tier": "Rule-based + ML",
                "elapsed_sec": round(elapsed, 4),
                "status": "success",
            }
        ]
        return state

    def _node_generate(self, state: PipelineState) -> PipelineState:
        """Node 4: Generate consultation checklist."""
        start = time.time()

        # Reconstruct GapAnalysis
        ga_data = state["gap_analysis"]
        from core_pipeline.detector.gap_detector import GapAnalysis, CheckItem
        gap = GapAnalysis(
            check_items=[CheckItem(**item) for item in ga_data["check_items"]],
            covered_count=ga_data["covered_count"],
            check_needed_count=ga_data["check_needed_count"],
            total_categories=ga_data["total_categories"],
        )

        product_name = state.get("parsed_coverage", {}).get("product_name", "")
        checklist = self.generator.generate(gap, product_name)

        elapsed = time.time() - start
        state["checklist"] = checklist.raw_json
        state["trace"] = state.get("trace", []) + [
            {
                "node": "Checklist Generator",
                "model_tier": "High-quality LLM (Sonnet)",
                "elapsed_sec": round(elapsed, 2),
                "status": "success",
            }
        ]
        return state
