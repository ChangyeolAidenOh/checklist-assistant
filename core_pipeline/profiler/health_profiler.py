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

        self.concern_mapping = {
            "허리": "정형외과", "목": "정형외과", "관절": "정형외과",
            "눈": "안과", "시력": "안과",
            "치아": "치과", "잇몸": "치과",
            "스트레스": "정신건강", "우울": "정신건강", "불안": "정신건강", "수면": "정신건강",
            "피부": "피부과", "여드름": "피부과",
            "임신": "산부인과", "출산": "산부인과",
        }

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
                "health_concern_boosts": [
                    c for c in profile.health_concerns
                    if any(k in c for k in self.concern_mapping)
                ],
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
                "health_concern_boosts": [
                    c for c in profile.health_concerns
                    if any(k in c for k in self.concern_mapping)
                ],
            },
            model_used="rule-based",
        )

    def _apply_concern_boosts(self, scores: dict, profile: UserProfile) -> dict:
        """Apply health concern keyword boosts."""
        for concern in profile.health_concerns:
            for keyword, cat in self.concern_mapping.items():
                if keyword in concern and cat in scores:
                    scores[cat] = min(scores[cat] + 0.15, 1.0)
        return scores
