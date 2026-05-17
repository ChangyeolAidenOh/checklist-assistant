"""
XGBoost Risk Profiler Model.

Trains a model to predict medical category interest scores
from user demographic features.

Training data:
  - Primary: NHIS 건강검진정보 (1M records) via train_with_real_data.py
  - Fallback: Synthetic data derived from NHIS aggregate statistics

Note: For real data training, use scripts/train_with_real_data.py instead.
This module provides synthetic fallback and model loading utilities.
"""

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import xgboost as xgb

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

from core_pipeline.profiler.nhis_processor import (
    load_nhis_data,
    build_lookup_table,
    DEPARTMENT_TO_CATEGORY,
    AGE_GROUPS,
)
from core_pipeline.profiler.health_profiler import INTEREST_CATEGORIES

logger = logging.getLogger(__name__)

MODEL_PATH = Path(__file__).parent.parent.parent / "models"
MODEL_FILE = MODEL_PATH / "xgb_profiler.pkl"
SHAP_EXPLAINER_FILE = MODEL_PATH / "shap_explainer.pkl"


def _age_to_group(age: int) -> str:
    if age < 10: return "0-9"
    if age < 20: return "10-19"
    if age < 30: return "20-29"
    if age < 40: return "30-39"
    if age < 50: return "40-49"
    if age < 60: return "50-59"
    if age < 70: return "60-69"
    return "70+"


def generate_synthetic_training_data(
    nhis_data_dir: Path,
    n_samples: int = 10000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate synthetic training data from NHIS patterns.

    Creates individual-level samples from aggregate statistics.

    Returns:
        (features_df, targets_df)
        features: [age, gender_encoded, occupation_encoded, family_size, has_children]
        targets: [interest score per category]
    """
    np.random.seed(42)

    nhis_df = load_nhis_data(nhis_data_dir)
    lookup = build_lookup_table(nhis_df)

    occupation_codes = {
        "사무직": 0, "생산직": 1, "자영업": 2,
        "전문직": 3, "학생": 4, "주부": 5, "기타": 6,
    }

    # Occupation -> category modifier patterns
    occ_patterns = {
        0: {"정신건강": 0.10, "안과": 0.10, "정형외과": -0.05},  # 사무직
        1: {"정형외과": 0.15, "응급": 0.10, "재활": 0.10},       # 생산직
        2: {"정신건강": 0.15, "건강검진": -0.10},                 # 자영업
        3: {"정신건강": 0.10, "건강검진": 0.05},                  # 전문직
        4: {"정신건강": 0.10, "피부과": 0.10, "건강검진": -0.15}, # 학생
        5: {"정형외과": 0.05, "산부인과": 0.10},                  # 주부
        6: {},                                                     # 기타
    }

    features = []
    targets = []

    for _ in range(n_samples):
        age = np.random.randint(18, 75)
        gender = np.random.choice([0, 1])  # 0=M, 1=F
        occ = np.random.choice(list(occupation_codes.values()))
        family_size = np.random.choice([1, 2, 3, 4, 5], p=[0.2, 0.25, 0.3, 0.15, 0.1])
        has_children = 1 if family_size >= 3 and np.random.random() > 0.3 else 0

        age_group = _age_to_group(age)
        base_scores = lookup.get(age_group, lookup.get("30-39", {}))

        # Apply occupation modifiers
        occ_mod = occ_patterns.get(occ, {})
        scores = {}
        for cat in INTEREST_CATEGORIES:
            base = base_scores.get(cat, 0.3)
            mod = occ_mod.get(cat, 0.0)
            noise = np.random.normal(0, 0.05)

            # Gender effect
            if cat == "산부인과" and gender == 0:  # Male
                base *= 0.05
            if cat == "산부인과" and gender == 1 and age < 45:
                base *= 1.3

            # Family effect
            if has_children and cat in ("입원", "외래/통원"):
                base += 0.05

            scores[cat] = max(0.0, min(1.0, base + mod + noise))

        features.append([age, gender, occ, family_size, has_children])
        targets.append([scores.get(cat, 0.0) for cat in INTEREST_CATEGORIES])

    feature_cols = ["age", "gender", "occupation", "family_size", "has_children"]
    features_df = pd.DataFrame(features, columns=feature_cols)
    targets_df = pd.DataFrame(targets, columns=INTEREST_CATEGORIES)

    return features_df, targets_df


def train_model(
    data_dir: Path,
    force_retrain: bool = False,
) -> dict:
    """Train XGBoost multi-output model with synthetic data.

    For real data training, use scripts/train_with_real_data.py instead.

    Returns:
        {"models": {category: xgb_model}, "feature_names": [...], "metrics": {...}}
    """
    MODEL_PATH.mkdir(exist_ok=True)

    if MODEL_FILE.exists() and not force_retrain:
        logger.info("Loading cached model from %s", MODEL_FILE)
        with open(MODEL_FILE, "rb") as f:
            return pickle.load(f)

    logger.info("Using synthetic training data")
    nhis_dir = data_dir / "nhis_stats" if (data_dir / "nhis_stats").exists() else data_dir
    features, targets = generate_synthetic_training_data(nhis_dir)
    data_source = "synthetic"

    features = features.fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(
        features, targets, test_size=0.2, random_state=42,
    )

    models = {}
    metrics = {}

    for cat in INTEREST_CATEGORIES:
        model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0,
        )

        model.fit(
            X_train, y_train[cat],
            eval_set=[(X_test, y_test[cat])],
            verbose=False,
        )

        preds = model.predict(X_test)
        mse = mean_squared_error(y_test[cat], preds)
        models[cat] = model
        metrics[cat] = round(mse, 4)

    logger.info("Training complete. Data source: %s", data_source)
    logger.info("MSE per category: %s", metrics)

    # SHAP: Don't cache explainers (serialization issues with newer xgboost)
    # SHAP values will be computed at prediction time in health_profiler.py
    shap_explainers = {}

    result = {
        "models": models,
        "shap_explainers": shap_explainers,
        "feature_names": list(features.columns),
        "metrics": metrics,
        "data_source": data_source,
    }

    # Cache
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(result, f)
    logger.info("Model saved to %s", MODEL_FILE)

    return result
