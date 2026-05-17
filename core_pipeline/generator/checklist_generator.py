"""
Node 4: Checklist Generator.

Generates consultation-ready checklists from gap analysis results.
Uses high-quality LLM (Sonnet) for natural language generation.

Model tier: High-quality LLM (Sonnet) - customer/planner-facing text.
"""

import logging
from dataclasses import dataclass, field

from core_pipeline.llm.backend import get_backend
from core_pipeline.detector.gap_detector import GapAnalysis
from core_pipeline.guardrail.filter import GuardrailFilter

logger = logging.getLogger(__name__)

GENERATOR_SYSTEM_PROMPT = """\
You are a helpful insurance consultation preparation assistant for Korean consumers.

Your role: Convert gap analysis results into a clear, actionable consultation checklist.

Rules:
1. NEVER recommend specific products or suggest purchasing insurance
2. NEVER say "보장이 부족합니다" - instead say "이 영역은 상담 시 확인이 필요할 수 있습니다"
3. Always end with: "최종 판단은 반드시 보험 설계사 또는 전문 상담을 통해 이루어져야 합니다."
4. Use warm, professional Korean (존댓말)
5. Prioritize items by consultation priority score
6. For each item, provide:
   - What the item is about (plain language)
   - Current coverage status
   - What to ask the insurance planner
7. Group items: high priority (7+), medium (4-7), informational (1-4)

Output format:
{
  "checklist_title": "상담 전 확인 체크리스트",
  "summary": "전체 요약 (2-3문장)",
  "high_priority": [
    {
      "item": "항목명",
      "description": "설명",
      "question_for_planner": "설계사에게 물어볼 질문"
    }
  ],
  "medium_priority": [...],
  "informational": [...],
  "disclaimer": "최종 판단은 반드시 보험 설계사..."
}
"""


@dataclass
class ChecklistItem:
    item: str
    description: str
    question_for_planner: str
    priority_group: str  # high / medium / informational


@dataclass
class ConsultationChecklist:
    title: str
    summary: str
    high_priority: list[ChecklistItem] = field(default_factory=list)
    medium_priority: list[ChecklistItem] = field(default_factory=list)
    informational: list[ChecklistItem] = field(default_factory=list)
    disclaimer: str = ""
    raw_json: dict = field(default_factory=dict)


class ChecklistGenerator:
    """Generates consultation-ready checklists."""

    def __init__(self):
        self.backend = get_backend("generator")
        self.guardrail = GuardrailFilter()

    def generate(
        self,
        gap_analysis: GapAnalysis,
        product_name: str = "",
    ) -> ConsultationChecklist:
        """Generate a consultation checklist from gap analysis.

        Args:
            gap_analysis: Results from Node 3
            product_name: Name of the insurance product

        Returns:
            ConsultationChecklist ready for display
        """
        # Build prompt with gap analysis data
        items_text = self._format_gap_items(gap_analysis)

        prompt = (
            f"분석 대상 상품: {product_name}\n\n"
            f"확인 필요 항목 분석 결과:\n{items_text}\n\n"
            f"보장 항목 수: {gap_analysis.covered_count}개 확인됨\n"
            f"확인 필요 항목 수: {gap_analysis.check_needed_count}개\n\n"
            "위 분석 결과를 바탕으로 상담 전 체크리스트를 생성하세요."
        )

        try:
            result = self.backend.generate_structured(
                prompt=prompt,
                system=GENERATOR_SYSTEM_PROMPT,
            )
        except Exception as e:
            logger.error("Checklist generation failed: %s", e)
            return self._fallback_checklist(gap_analysis)

        # Apply guardrail filter
        result = self.guardrail.filter_output(result)

        return self._to_checklist(result)

    def _format_gap_items(self, gap: GapAnalysis) -> str:
        """Format gap analysis items for LLM prompt."""
        lines = []
        for item in gap.check_items:
            lines.append(
                f"- {item.category} (우선순위: {item.priority_score}/10, "
                f"상태: {item.status}, 관심도: {item.interest_score:.2f})\n"
                f"  설명: {item.plain_description}\n"
                f"  참고: {item.consultation_note}"
            )
        return "\n".join(lines)

    def _to_checklist(self, raw: dict) -> ConsultationChecklist:
        """Convert raw JSON to ConsultationChecklist."""
        def parse_items(items_list: list, group: str) -> list[ChecklistItem]:
            return [
                ChecklistItem(
                    item=item.get("item", ""),
                    description=item.get("description", ""),
                    question_for_planner=item.get("question_for_planner", ""),
                    priority_group=group,
                )
                for item in items_list
            ]

        return ConsultationChecklist(
            title=raw.get("checklist_title", "상담 전 확인 체크리스트"),
            summary=raw.get("summary", ""),
            high_priority=parse_items(raw.get("high_priority", []), "high"),
            medium_priority=parse_items(raw.get("medium_priority", []), "medium"),
            informational=parse_items(raw.get("informational", []), "informational"),
            disclaimer=raw.get(
                "disclaimer",
                "최종 판단은 반드시 보험 설계사 또는 전문 상담을 통해 이루어져야 합니다.",
            ),
            raw_json=raw,
        )

    def _fallback_checklist(self, gap: GapAnalysis) -> ConsultationChecklist:
        """Generate a basic checklist without LLM when generation fails."""
        high = []
        medium = []
        info = []

        for item in gap.check_items:
            cl_item = ChecklistItem(
                item=item.category,
                description=item.plain_description,
                question_for_planner=item.consultation_note,
                priority_group="",
            )
            if item.priority_score >= 7:
                cl_item.priority_group = "high"
                high.append(cl_item)
            elif item.priority_score >= 4:
                cl_item.priority_group = "medium"
                medium.append(cl_item)
            else:
                cl_item.priority_group = "informational"
                info.append(cl_item)

        return ConsultationChecklist(
            title="상담 전 확인 체크리스트",
            summary="자동 분석 결과를 기반으로 한 기본 체크리스트입니다.",
            high_priority=high,
            medium_priority=medium,
            informational=info,
            disclaimer="최종 판단은 반드시 보험 설계사 또는 전문 상담을 통해 이루어져야 합니다.",
        )
