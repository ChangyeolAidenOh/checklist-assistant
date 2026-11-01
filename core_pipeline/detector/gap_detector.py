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
# Mapping: coverage category / item-name keywords -> interest categories.
# Keys are matched as substrings against BOTH item.category and item.item_name,
# so 손해보험 담보명(상해후유장해, 치아파절, 골절진단비, 간병인 …)도 잡힌다.
CATEGORY_MAPPING = {
    "사망": [],
    "후유장해": ["재활", "정형외과"],
    "입원": ["입원"],
    "중환자실": ["입원", "응급"],
    "수술": ["수술"],
    "통원": ["외래/통원"],
    "외래": ["외래/통원"],
    "처방조제": ["외래/통원"],
    "진단": ["건강검진"],
    "치과": ["치과"],
    "치아": ["치과"],
    "치주": ["치과"],
    "임플란트": ["치과"],
    "정신": ["정신건강"],
    "우울": ["정신건강"],
    "안과": ["안과"],
    "백내장": ["안과"],
    "산부인과": ["산부인과"],
    "임신": ["산부인과"],
    "출산": ["산부인과"],
    "정형": ["정형외과"],
    "골절": ["정형외과", "응급"],
    "관절": ["정형외과"],
    "척추": ["정형외과"],
    "피부": ["피부과"],
    "화상": ["피부과", "응급"],
    "응급": ["응급"],
    "상해": ["응급"],
    "재해": ["응급"],
    "재활": ["재활"],
    "간병": ["재활"],
    "장기요양": ["재활"],
    "치매": ["재활", "정신건강"],
}


# 진료과 성격의 관심 영역 — 실손 포괄 담보로 "일반 진료비"는 보장될 수 있는 영역
DEPARTMENT_CATEGORIES = {"치과", "정신건강", "안과", "산부인과", "정형외과", "피부과"}

# 실손에서 흔히 문제가 되는 진료과별 확인 포인트
DEPARTMENT_CAVEATS = {
    "치과": "치과 치료(보철·임플란트·교정은 실손 보장 제외가 일반적)",
    "정신건강": "정신건강의학과 비급여 상담·치료",
    "안과": "시력교정술(라식·라섹)과 비급여 검사",
    "산부인과": "임신·출산 관련 진료와 비급여 검사",
    "정형외과": "도수치료·체외충격파 등 비급여 물리치료 한도",
    "피부과": "미용 목적 시술(보장 제외)과 질병 치료의 구분",
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
    basis: str = ""                # "관심사" | "연령·성별 의료이용 통계" — why this category is on the list
    basis_concerns: list = field(default_factory=list)  # user inputs that boosted this category


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

        # 실손의료비(또는 입원+통원 포괄 담보)는 진료과를 가리지 않고 보장한다.
        # 이 경우 진료과 카테고리는 "미포함"이 아니라 "포괄 보장 — 비급여/특정치료 범위 확인"(partial)이다.
        general_medical = self._is_general_medical(coverage, covered_categories)

        # Why is each category on the list? Invert {concern: [cats]} -> {cat: [concerns]}
        boosts = (interest.shap_explanations or {}).get("health_concern_boosts", {})
        cat_to_concerns: dict[str, list[str]] = {}
        if isinstance(boosts, dict):
            for concern, cats in boosts.items():
                for c in cats:
                    cat_to_concerns.setdefault(c, []).append(concern)
        age_group = (interest.shap_explanations or {}).get("age_group", "")

        items = []
        covered_count = 0

        for cat, score in zip(interest.categories, interest.scores):
            if score < self.priority_threshold:
                continue  # low interest, skip

            basis_concerns = cat_to_concerns.get(cat, [])
            basis = "관심사" if basis_concerns else (
                f"연령·성별 의료이용 통계{f' ({age_group})' if age_group else ''}"
            )

            status = covered_categories.get(cat, "not_covered")
            detail = covered_categories.get(f"{cat}_detail", "")
            if status == "not_covered" and general_medical and cat in DEPARTMENT_CATEGORIES:
                status = "partial"
                detail = detail or "실손 입원·통원 담보로 일반 진료비는 보장 대상 (전용 담보 없음)"

            # Priority = interest score * coverage gap weight
            gap_weight = {"covered": 0.2, "partial": 0.6, "not_covered": 1.0, "unknown": 0.8}
            priority = round(score * gap_weight.get(status, 0.5) * 10, 1)

            if status == "covered":
                covered_count += 1
                note = f"{cat} 관련 담보가 요약서에 있습니다. 보장 한도·자기부담·면책 조건을 확인해보세요."
            elif status == "partial":
                if general_medical and cat in DEPARTMENT_CATEGORIES:
                    note = (
                        f"{cat} 진료비는 실손 입원·통원 담보로 보장될 수 있으나 전용 담보는 없습니다. "
                        f"{DEPARTMENT_CAVEATS.get(cat, '비급여 항목과 보장 제외 치료')}에 대해 실손 처리 기준을 설계사에게 확인해보세요."
                    )
                else:
                    note = (
                        f"{cat} 관련 담보가 일부 있습니다. "
                        "보장 범위가 본인 상황에 충분한지 설계사와 상의해보세요."
                    )
            elif status == "not_covered":
                note = (
                    f"이 요약서에서 {cat} 관련 담보를 찾지 못했습니다. "
                    "보장 여부와 필요성을 설계사에게 확인해보세요."
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
                basis=basis,
                basis_concerns=basis_concerns,
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

    @staticmethod
    def _is_general_medical(coverage: ParsedCoverage, covered: dict) -> bool:
        """True for 실손의료비형 상품: 진료과 무관 입원+통원 포괄 보장."""
        text = f"{coverage.product_name} {coverage.product_type}"
        if "실손" in text:
            return True
        return covered.get("입원") == "covered" and covered.get("외래/통원") == "covered"

    def _extract_covered_categories(self, coverage: ParsedCoverage) -> dict:
        """Extract which interest categories are covered."""
        result: dict[str, str] = {}

        for item in coverage.coverage_items:
            # match on category label AND 담보명 (손보 요약서는 category가 거칠고 담보명이 구체적)
            cat_key = f"{item.category} {item.item_name}"
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
