"""
Node 2: Health & Lifestyle Profiler.

Generates interest-area vectors based on demographic profile.

Model strategy:
  Phase 1 (rule-based): NHIS statistical lookup tables
  Phase 2 (ML): XGBoost trained on NHIS 건강검진정보 (1M records)
  Fallback: Rule-based always available if model not trained

Model tier: XGBoost / Rule-based (no LLM needed, cost = 0).
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Medical categories aligned with insurance coverage areas
INTEREST_CATEGORIES = [
    "입원",        # Hospitalization
    "외래/통원",   # Outpatient
    "수술",        # Surgery
    "치과",        # Dental
    "정신건강",    # Mental health
    "안과",        # Ophthalmology
    "산부인과",    # OB/GYN
    "정형외과",    # Orthopedics
    "피부과",      # Dermatology
    "건강검진",    # Health screening
    "응급",        # Emergency
    "재활",        # Rehabilitation
]


# 생활/건강 관심사 키워드 → 관심 영역(복수 가능). 부분 일치로 매칭한다.
# 사용자가 적는 말은 진료과목이 아니라 생활 언어이므로 넓게 잡는다.
CONCERN_KEYWORDS: dict[str, list[str]] = {
    # 근골격 / 운동
    "허리": ["정형외과"], "목": ["정형외과"], "어깨": ["정형외과"], "무릎": ["정형외과"],
    "관절": ["정형외과"], "디스크": ["정형외과", "수술"], "척추": ["정형외과"],
    "운동": ["정형외과", "응급"], "헬스": ["정형외과"], "러닝": ["정형외과"], "달리기": ["정형외과"],
    "등산": ["정형외과", "응급"], "축구": ["정형외과", "응급"], "농구": ["정형외과", "응급"],
    "자전거": ["정형외과", "응급"], "골프": ["정형외과"], "테니스": ["정형외과"], "스키": ["정형외과", "응급"],
    "골절": ["정형외과", "응급"], "부상": ["정형외과", "응급"],
    # 눈
    "눈": ["안과"], "시력": ["안과"], "안경": ["안과"], "렌즈": ["안과"], "라식": ["안과"], "라섹": ["안과"],
    "노안": ["안과"], "백내장": ["안과"], "녹내장": ["안과"],
    # 치아
    "치아": ["치과"], "이빨": ["치과"], "잇몸": ["치과"], "충치": ["치과"], "임플란트": ["치과"],
    "교정": ["치과"], "스케일링": ["치과"], "사랑니": ["치과"],
    # 정신건강 / 수면
    "스트레스": ["정신건강"], "우울": ["정신건강"], "불안": ["정신건강"], "번아웃": ["정신건강"],
    "수면": ["정신건강"], "불면": ["정신건강"], "공황": ["정신건강"], "상담": ["정신건강"], "멘탈": ["정신건강"],
    # 피부
    "피부": ["피부과"], "여드름": ["피부과"], "아토피": ["피부과"], "탈모": ["피부과"], "두피": ["피부과"],
    # 여성 / 임신
    "임신": ["산부인과"], "출산": ["산부인과"], "난임": ["산부인과"], "생리": ["산부인과"], "자궁": ["산부인과"],
    "유방": ["산부인과", "건강검진"],
    # 식습관 / 대사 / 검진
    "식단": ["건강검진"], "식습관": ["건강검진"], "그로서리": ["건강검진"], "장보기": ["건강검진"],
    "다이어트": ["건강검진"], "체중": ["건강검진"], "비만": ["건강검진", "입원"],
    "당뇨": ["건강검진", "입원"], "혈당": ["건강검진"], "혈압": ["건강검진", "입원"],
    "고혈압": ["건강검진", "입원"], "콜레스테롤": ["건강검진"], "지방간": ["건강검진"],
    "건강검진": ["건강검진"], "검진": ["건강검진"], "음주": ["건강검진", "입원"], "술": ["건강검진"],
    "흡연": ["건강검진", "입원"], "담배": ["건강검진", "입원"],
    # 소화기 / 호흡기 / 감기 (외래)
    "소화": ["외래/통원"], "위": ["외래/통원", "건강검진"], "장": ["외래/통원"], "역류": ["외래/통원"],
    "감기": ["외래/통원"], "비염": ["외래/통원"], "알레르기": ["외래/통원"], "천식": ["외래/통원", "입원"],
    "두통": ["외래/통원"], "편두통": ["외래/통원"], "빈혈": ["외래/통원", "건강검진"],
    # 가족 / 돌봄
    "육아": ["입원", "응급"], "자녀": ["입원", "응급"], "아이": ["입원", "응급"],
    "부모": ["입원", "재활"], "간병": ["재활", "입원"], "치매": ["재활", "정신건강"],
    # 중증
    "암": ["건강검진", "입원", "수술"], "가족력": ["건강검진"], "심장": ["건강검진", "입원"],
    "뇌": ["건강검진", "입원"], "수술": ["수술", "입원"], "입원": ["입원"],
}


@dataclass
class UserProfile:
    age: int
    gender: str                    # M / F
    occupation_category: str       # 사무직 / 생산직 / 자영업 / 전문직 / 학생 / 주부 / 기타
    family_size: int = 1
    has_children: bool = False
    health_concerns: list[str] = field(default_factory=list)
    chronic_conditions: list[str] = field(default_factory=list)


@dataclass
class InterestVector:
    categories: list[str]
    scores: list[float]          # 0.0 ~ 1.0 normalized
    top_interests: list[str]     # top-3 categories
    shap_explanations: dict = field(default_factory=dict)
    model_used: str = "rule-based"  # "rule-based" or "xgboost"

    def to_dict(self) -> dict:
        return {
            cat: round(score, 3)
            for cat, score in zip(self.categories, self.scores)
        }


class HealthProfiler:
    """Generates interest-area vectors from user demographic profiles.

    Automatically uses XGBoost model if trained, otherwise falls back
    to rule-based profiling.
    """

    def __init__(self, data_dir: Path | None = None, use_ml: bool = True):
        self.data_dir = data_dir or Path(__file__).parent.parent.parent / "data"
        self.xgb_model = None
        self.shap_explainers = None
        self.feature_names = None
        self.model_source = "rule-based"

        self._load_rule_based_baseline()

        if use_ml:
            self._try_load_xgb_model()

    def _try_load_xgb_model(self) -> None:
        """Try to load trained XGBoost model."""
        try:
            from core_pipeline.profiler.xgb_trainer import MODEL_FILE
            import pickle

            if MODEL_FILE.exists():
                with open(MODEL_FILE, "rb") as f:
                    result = pickle.load(f)
                self.xgb_model = result["models"]
                self.shap_explainers = result.get("shap_explainers", {})
                self.feature_names = result["feature_names"]
                self.model_source = f"xgboost ({result.get('data_source', 'unknown')})"
                logger.info("XGBoost model loaded: %s", self.model_source)
            else:
                logger.info("No trained XGBoost model found. Using rule-based.")
        except Exception as e:
            logger.warning("Failed to load XGBoost model: %s. Using rule-based.", e)

    def _load_rule_based_baseline(self) -> None:
        """Load baseline from real HIRA data, fallback to hardcoded."""
        # Try real HIRA lookup table first
        real_csv = self.data_dir / "nhis_stats" / "real_lookup_table.csv"
        if real_csv.exists():
            self.baseline = self._load_real_baseline(real_csv)
            logger.info("Loaded REAL HIRA baseline from %s", real_csv)
        else:
            self.baseline = self._hardcoded_baseline()
            logger.info("Using hardcoded baseline (real data not found)")

        self.occupation_modifiers = {
            "사무직":  {"정신건강": +0.10, "안과": +0.10, "정형외과": -0.05},
            "생산직":  {"정형외과": +0.15, "응급": +0.10, "재활": +0.10},
            "자영업":  {"정신건강": +0.15, "건강검진": -0.10},
            "전문직":  {"정신건강": +0.10, "건강검진": +0.05},
            "학생":    {"정신건강": +0.10, "피부과": +0.10, "건강검진": -0.15},
            "주부":    {"정형외과": +0.05, "산부인과": +0.10},
            "기타":    {},
        }

        self.concern_mapping = CONCERN_KEYWORDS

    def _load_real_baseline(self, csv_path: Path) -> dict:
        """Load baseline from real HIRA lookup table CSV."""
        import pandas as pd
        df = pd.read_csv(csv_path)
        baseline = {}
        for age_group in df["age_group"].unique():
            age_data = df[df["age_group"] == age_group]
            scores = {}
            for _, row in age_data.iterrows():
                scores[row["category"]] = round(float(row["score"]), 3)
            baseline[age_group] = scores
        return baseline

    @staticmethod
    def _hardcoded_baseline() -> dict:
        """Hardcoded fallback baseline (pre-HIRA data)."""
        return {
            "20s": {"입원": 0.15, "외래/통원": 0.45, "수술": 0.08, "치과": 0.55,
                    "정신건강": 0.30, "안과": 0.20, "산부인과": 0.35, "정형외과": 0.25,
                    "피부과": 0.40, "건강검진": 0.20, "응급": 0.15, "재활": 0.10},
            "30s": {"입원": 0.25, "외래/통원": 0.55, "수술": 0.15, "치과": 0.60,
                    "정신건강": 0.35, "안과": 0.25, "산부인과": 0.45, "정형외과": 0.30,
                    "피부과": 0.35, "건강검진": 0.40, "응급": 0.12, "재활": 0.15},
            "40s": {"입원": 0.35, "외래/통원": 0.65, "수술": 0.25, "치과": 0.55,
                    "정신건강": 0.25, "안과": 0.35, "산부인과": 0.30, "정형외과": 0.40,
                    "피부과": 0.25, "건강검진": 0.60, "응급": 0.15, "재활": 0.25},
            "50s": {"입원": 0.45, "외래/통원": 0.75, "수술": 0.35, "치과": 0.50,
                    "정신건강": 0.20, "안과": 0.45, "산부인과": 0.15, "정형외과": 0.50,
                    "피부과": 0.20, "건강검진": 0.70, "응급": 0.20, "재활": 0.35},
            "60+": {"입원": 0.55, "외래/통원": 0.85, "수술": 0.40, "치과": 0.45,
                    "정신건강": 0.15, "안과": 0.55, "산부인과": 0.05, "정형외과": 0.60,
                    "피부과": 0.15, "건강검진": 0.75, "응급": 0.25, "재활": 0.45},
        }

    def _age_group(self, age: int) -> str:
        if age < 30: return "20s"
        if age < 40: return "30s"
        if age < 50: return "40s"
        if age < 60: return "50s"
        return "60+"

    def generate_interest_vector(self, profile: UserProfile) -> InterestVector:
        """Generate interest-area vector for a user profile.

        Uses XGBoost if model is trained, otherwise rule-based.
        Health concern boosts are always applied on top (both modes).
        """
        if self.xgb_model is not None:
            return self._predict_xgb(profile)
        return self._predict_rule_based(profile)

    def _predict_xgb(self, profile: UserProfile) -> InterestVector:
        """XGBoost-based prediction with SHAP explanations."""
        occupation_codes = {
            "사무직": 0, "생산직": 1, "자영업": 2,
            "전문직": 3, "학생": 4, "주부": 5, "기타": 6,
        }

        features = pd.DataFrame([{
            "age": profile.age,
            "gender": 0 if profile.gender == "M" else 1,
            "occupation": occupation_codes.get(profile.occupation_category, 6),
            "family_size": profile.family_size,
            "has_children": 1 if profile.has_children else 0,
        }])

        # Align feature columns with training
        for col in self.feature_names:
            if col not in features.columns:
                features[col] = 0
        features = features[self.feature_names]

        scores = {}
        shap_dict = {}

        for cat in INTEREST_CATEGORIES:
            if cat in self.xgb_model:
                model = self.xgb_model[cat]
                pred = float(model.predict(features)[0])
                scores[cat] = max(0.0, min(1.0, pred))

                # Feature importance (always works)
                importances = model.feature_importances_
                shap_dict[cat] = {
                    name: round(float(imp), 4)
                    for name, imp in zip(self.feature_names, importances)
                }
            else:
                scores[cat] = 0.3

        # Apply health concern boosts on top
        scores = self._apply_concern_boosts(scores, profile)
        recognized, unrecognized = self.match_concerns(profile.health_concerns)

        score_list = [scores.get(cat, 0.0) for cat in INTEREST_CATEGORIES]
        sorted_indices = np.argsort(score_list)[::-1]
        top_interests = [INTEREST_CATEGORIES[i] for i in sorted_indices[:3]]

        return InterestVector(
            categories=INTEREST_CATEGORIES,
            scores=score_list,
            top_interests=top_interests,
            shap_explanations={
                "model": self.model_source,
                "feature_contributions": shap_dict,
                "health_concern_boosts": recognized,
                "health_concern_unrecognized": unrecognized,
            },
            model_used="xgboost",
        )

    def _predict_rule_based(self, profile: UserProfile) -> InterestVector:
        """Rule-based prediction (original Phase 1 logic)."""
        age_group = self._age_group(profile.age)
        base_scores = dict(self.baseline.get(age_group, self.baseline["30s"]))

        modifiers = self.occupation_modifiers.get(profile.occupation_category, {})
        for cat, mod in modifiers.items():
            if cat in base_scores:
                base_scores[cat] = max(0.0, min(base_scores[cat] + mod, 1.0))

        if profile.gender == "M":
            base_scores["산부인과"] *= 0.1
        if profile.has_children:
            base_scores["입원"] = min(base_scores["입원"] + 0.10, 1.0)

        base_scores = self._apply_concern_boosts(base_scores, profile)
        recognized, unrecognized = self.match_concerns(profile.health_concerns)

        categories = INTEREST_CATEGORIES
        scores = [round(base_scores.get(cat, 0.0), 3) for cat in categories]

        sorted_indices = np.argsort(scores)[::-1]
        top_interests = [categories[i] for i in sorted_indices[:3]]

        return InterestVector(
            categories=categories,
            scores=scores,
            top_interests=top_interests,
            shap_explanations={
                "model": "rule-based",
                "age_group": age_group,
                "occupation_effect": modifiers,
                "health_concern_boosts": recognized,
                "health_concern_unrecognized": unrecognized,
            },
            model_used="rule-based",
        )

    def match_concerns(self, concerns: list[str]) -> tuple[dict[str, list[str]], list[str]]:
        """Return ({concern: [categories]}, [unrecognized concerns])."""
        recognized: dict[str, list[str]] = {}
        unknown: list[str] = []
        for concern in concerns:
            c = concern.strip()
            if not c:
                continue
            cats: list[str] = []
            for keyword, mapped in self.concern_mapping.items():
                if keyword in c:
                    cats.extend(mapped)
            cats = list(dict.fromkeys(cats))  # dedupe, keep order
            if cats:
                recognized[c] = cats
            else:
                unknown.append(c)
        return recognized, unknown

    def _apply_concern_boosts(self, scores: dict, profile: UserProfile) -> dict:
        """Apply health concern keyword boosts (+0.15 per matched category, capped at 1.0)."""
        recognized, _ = self.match_concerns(profile.health_concerns)
        for cats in recognized.values():
            for cat in cats:
                if cat in scores:
                    scores[cat] = min(scores[cat] + 0.15, 1.0)
        return scores
