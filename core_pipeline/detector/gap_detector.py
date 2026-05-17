"""
Node 3: Coverage Gap Detector.

Compares parsed coverage against interest-area vectors
to identify items that may need consultation review.

Model tier: Rule-based + ML hybrid (no LLM needed).

Important boundary:
  - Does NOT say "coverage is insufficient"
  - Says "this area may need review during consultation"
"""

import logging
from dataclasses import dataclass, field

from core_pipeline.parser.coverage_parser import ParsedCoverage
from core_pipeline.profiler.health_profiler import InterestVector, INTEREST_CATEGORIES

logger = logging.getLogger(__name__)

# Mapping: coverage categories -> interest categories
CATEGORY_MAPPING = {
    "사망": [],
    "입원": ["입원"],
    "수술": ["수술"],
    "통원": ["외래/통원"],
    "외래": ["외래/통원"],
    "진단": ["건강검진"],
    "치과": ["치과"],
    "정신": ["정신건강"],
    "안과": ["안과"],
    "산부인과": ["산부인과"],
    "정형": ["정형외과"],
    "피부": ["피부과"],
    "응급": ["응급"],
    "재활": ["재활"],
}


@dataclass
class CheckItem:
    category: str
    priority_score: float          # 0-10 scale
    status: str                    # "covered" | "not_covered" | "partial" | "unknown"
    plain_description: str
    consultation_note: str         # what to ask the planner
    interest_score: float          # from profiler
    coverage_detail: str = ""      # if covered, what's the detail


@dataclass
class GapAnalysis:
    check_items: list[CheckItem] = field(default_factory=list)
    covered_count: int = 0
    check_needed_count: int = 0
    total_categories: int = 0


class GapDetector:
    """Detects areas needing consultation review."""

    def __init__(self, priority_threshold: float = 0.3):
        self.priority_threshold = priority_threshold

    def analyze(
        self,
        coverage: ParsedCoverage,
        interest: InterestVector,
    ) -> GapAnalysis:
        """Compare coverage against interest vector.

        Args:
            coverage: Parsed coverage from Node 1
            interest: Interest vector from Node 2

        Returns:
            GapAnalysis with prioritized check items
        """
        # Build coverage lookup: which categories are covered
        covered_categories = self._extract_covered_categories(coverage)

        items = []
        covered_count = 0

        for cat, score in zip(interest.categories, interest.scores):
            if score < self.priority_threshold:
                continue  # low interest, skip

            status = covered_categories.get(cat, "not_covered")
            detail = covered_categories.get(f"{cat}_detail", "")

            # Priority = interest score * coverage gap weight
            gap_weight = {"covered": 0.2, "partial": 0.6, "not_covered": 1.0, "unknown": 0.8}
            priority = round(score * gap_weight.get(status, 0.5) * 10, 1)

            if status == "covered":
                covered_count += 1
                note = f"{cat} 보장이 포함되어 있습니다. 보장 한도와 조건을 확인해보세요."
            elif status == "partial":
                note = (
                    f"{cat} 관련 보장이 일부 포함되어 있으나, "
                    "보장 범위가 충분한지 설계사와 상의해보세요."
                )
            elif status == "not_covered":
                note = (
                    f"{cat} 관련 보장이 현재 포함되어 있지 않습니다. "
                    "필요 여부를 설계사와 상의해보세요."
                )
            else:
                note = (
                    f"{cat} 관련 보장 여부를 자동으로 확인할 수 없었습니다. "
                    "설계사에게 직접 문의해보세요."
                )

            items.append(CheckItem(
                category=cat,
                priority_score=priority,
                status=status,
                plain_description=self._category_description(cat),
                consultation_note=note,
                interest_score=score,
                coverage_detail=detail,
            ))

        # Sort by priority (descending)
        items.sort(key=lambda x: x.priority_score, reverse=True)

        check_needed = sum(1 for i in items if i.status in ("not_covered", "partial", "unknown"))

        return GapAnalysis(
            check_items=items,
            covered_count=covered_count,
            check_needed_count=check_needed,
            total_categories=len(items),
        )

    def _extract_covered_categories(self, coverage: ParsedCoverage) -> dict:
        """Extract which interest categories are covered."""
        result: dict[str, str] = {}

        for item in coverage.coverage_items:
            cat_key = item.category
            for prefix, mapped_cats in CATEGORY_MAPPING.items():
                if prefix in cat_key:
                    for mc in mapped_cats:
                        result[mc] = "covered"
                        result[f"{mc}_detail"] = (
                            f"{item.item_name}: {item.benefit_amount}"
                        )

        return result

    def _category_description(self, category: str) -> str:
        """Plain language description of each category."""
        descriptions = {
            "입원": "질병이나 사고로 병원에 입원하는 경우의 비용",
            "외래/통원": "병원에 입원하지 않고 외래로 진료받는 경우의 비용",
            "수술": "수술이 필요한 경우의 비용",
            "치과": "치과 진료 (충치, 임플란트, 교정 등) 비용",
            "정신건강": "정신건강 관련 진료 (상담, 치료 등) 비용",
            "안과": "눈 관련 진료 (시력 교정, 안과 질환 등) 비용",
            "산부인과": "산부인과 진료 (임신, 출산, 여성 질환 등) 비용",
            "정형외과": "뼈, 관절, 근육 관련 진료 비용",
            "피부과": "피부 관련 진료 비용",
            "건강검진": "정기 건강검진 비용",
            "응급": "응급 상황 발생 시 비용",
            "재활": "재활 치료 비용",
        }
        return descriptions.get(category, f"{category} 관련 의료비")
