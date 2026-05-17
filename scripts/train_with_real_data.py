"""
Train XGBoost with REAL Korean public health data.

Data sources:
  1. HIRA 상병별 진료 통계 (277K rows) → baseline lookup table
  2. NHIS 건강검진정보 (1M rows) → XGBoost training

Usage:
  python scripts/train_with_real_data.py

Expects:
  data/nhis_stats/hira_disease_stats.csv  (cp949 encoded)
  data/nhis_stats/nhis_health_checkup.csv (cp949 encoded)
"""

import sys
import logging
import pickle
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(name)s - %(message)s")
logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb

from core_pipeline.profiler.real_data_processor import (
    process_hira_disease_stats,
    build_real_lookup_table,
    process_health_checkup,
)
from core_pipeline.profiler.health_profiler import HealthProfiler, UserProfile, INTEREST_CATEGORIES

DATA_DIR = Path(__file__).parent.parent / "data" / "nhis_stats"
MODEL_DIR = Path(__file__).parent.parent / "models"


def step1_process_hira():
    """Process HIRA disease statistics into lookup table."""
    print("=" * 60)
    print("Step 1: HIRA Disease Statistics → Baseline Lookup")
    print("=" * 60)

    hira_path = DATA_DIR / "hira_disease_stats.csv"
    if not hira_path.exists():
        print(f"  NOT FOUND: {hira_path}")
        print("  Place the HIRA CSV file there and rerun.")
        return None

    lookup = build_real_lookup_table(hira_path)

    print("\n  Real data lookup table:")
    for age_key in ["20s", "30s", "40s", "50s", "60+"]:
        scores = lookup.get(age_key, {})
        top3 = sorted(scores.items(), key=lambda x: -x[1])[:3]
        top3_str = ", ".join(f"{k}={v:.3f}" for k, v in top3)
        print(f"    {age_key}: {top3_str}")

    # Save as CSV for reference
    rows = []
    for age_key, scores in lookup.items():
        for cat, score in scores.items():
            rows.append({"age_group": age_key, "category": cat, "score": score})
    pd.DataFrame(rows).to_csv(DATA_DIR / "real_lookup_table.csv", index=False)
    print(f"\n  Saved to {DATA_DIR / 'real_lookup_table.csv'}")

    return lookup


def step2_train_xgboost():
    """Train XGBoost with real health checkup data."""
    print("\n" + "=" * 60)
    print("Step 2: NHIS Health Checkup → XGBoost Training")
    print("=" * 60)

    checkup_path = DATA_DIR / "nhis_health_checkup.csv"
    if not checkup_path.exists():
        print(f"  NOT FOUND: {checkup_path}")
        print("  Place the NHIS health checkup CSV there and rerun.")
        return None

    # Process data
    start = time.time()
    features, targets = process_health_checkup(checkup_path)
    print(f"  Data processed in {time.time() - start:.1f}s")
    print(f"  Features: {features.shape}")
    print(f"  Targets: {targets.shape}")

    # Handle NaN/inf
    features = features.replace([np.inf, -np.inf], np.nan).fillna(0)
    targets = targets.replace([np.inf, -np.inf], np.nan).fillna(0.3)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        features, targets, test_size=0.2, random_state=42,
    )
    print(f"  Train: {X_train.shape[0]:,}, Test: {X_test.shape[0]:,}")

    # Train per-category models
    models = {}
    metrics = {}
    start = time.time()

    for cat in INTEREST_CATEGORIES:
        model = xgb.XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0,
            n_jobs=-1,
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

    elapsed = time.time() - start
    print(f"\n  Training completed in {elapsed:.1f}s")
    print(f"  MSE per category:")
    for cat, mse in sorted(metrics.items(), key=lambda x: x[1]):
        print(f"    {cat:12s}: {mse:.4f}")

    # Save model
    MODEL_DIR.mkdir(exist_ok=True)
    result = {
        "models": models,
        "shap_explainers": {},
        "feature_names": list(features.columns),
        "metrics": metrics,
        "data_source": "nhis_health_checkup_2024 (1M records)",
    }

    model_path = MODEL_DIR / "xgb_profiler.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(result, f)
    print(f"\n  Model saved to {model_path}")

    return result


def step3_test_predictions(lookup: dict | None = None):
    """Test predictions with the real-data-trained model."""
    print("\n" + "=" * 60)
    print("Step 3: Prediction Tests")
    print("=" * 60)

    profiler = HealthProfiler(use_ml=True)
    print(f"  Model: {profiler.model_source}")

    test_profiles = [
        ("30세 남성 사무직", UserProfile(age=30, gender="M", occupation_category="사무직")),
        ("35세 여성 사무직 (허리/스트레스)", UserProfile(
            age=35, gender="F", occupation_category="사무직",
            family_size=3, has_children=True,
            health_concerns=["허리 통증", "스트레스"],
        )),
        ("55세 남성 생산직", UserProfile(age=55, gender="M", occupation_category="생산직")),
        ("28세 여성 학생", UserProfile(age=28, gender="F", occupation_category="학생",
                                       health_concerns=["눈 피로", "피부"])),
    ]

    for desc, profile in test_profiles:
        iv = profiler.generate_interest_vector(profile)
        print(f"\n  [{desc}] model={iv.model_used}")
        print(f"    Top 3: {iv.top_interests}")
        top5 = sorted(iv.to_dict().items(), key=lambda x: -x[1])[:5]
        for cat, score in top5:
            print(f"      {cat}: {score:.3f}")

    # Compare with HIRA lookup if available
    if lookup:
        print("\n  HIRA Lookup vs XGBoost (30s average):")
        iv = profiler.generate_interest_vector(
            UserProfile(age=35, gender="M", occupation_category="기타")
        )
        hira_30s = lookup.get("30s", {})
        print(f"    {'Category':12s}  {'HIRA':>8s}  {'XGBoost':>8s}")
        for cat in INTEREST_CATEGORIES:
            h = hira_30s.get(cat, 0)
            x = iv.to_dict().get(cat, 0)
            print(f"    {cat:12s}  {h:8.3f}  {x:8.3f}")


if __name__ == "__main__":
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)

    lookup = step1_process_hira()
    result = step2_train_xgboost()

    if result:
        step3_test_predictions(lookup)
        print("\n" + "=" * 60)
        print("REAL DATA TRAINING COMPLETE")
        print(f"Data source: NHIS 건강검진정보 2024 (1,000,000 records)")
        print(f"Model: models/xgb_profiler.pkl")
        print("Dashboard will auto-detect on next Streamlit run.")
    else:
        print("\nMissing data files. See instructions above.")
