"""
Train XGBoost profiler model (synthetic data fallback).

For real data training, use: python scripts/train_with_real_data.py

Usage:
  python scripts/train_xgb_profiler.py
  python scripts/train_xgb_profiler.py --force

Run: python scripts/train_xgb_profiler.py
"""

import sys
import argparse
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(name)s - %(message)s")

from core_pipeline.profiler.xgb_trainer import train_model, generate_synthetic_training_data
from core_pipeline.profiler.nhis_processor import load_nhis_data, build_lookup_table
from core_pipeline.profiler.health_profiler import HealthProfiler, UserProfile, INTEREST_CATEGORIES

DATA_DIR = Path(__file__).parent.parent / "data"


def step1_generate_nhis():
    """Step 1: Generate/load NHIS data and build lookup tables."""
    print("=" * 50)
    print("Step 1: NHIS Data Processing")
    print("=" * 50)

    nhis_dir = DATA_DIR / "nhis_stats"
    nhis_dir.mkdir(exist_ok=True)

    df = load_nhis_data(nhis_dir)
    print(f"  Records: {len(df)}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Departments: {df['department'].nunique()}")
    print(f"  Age groups: {df['age_group'].nunique()}")

    lookup = build_lookup_table(df)
    print(f"\n  Lookup table (sample - 30-39):")
    for cat, score in sorted(lookup.get("30-39", {}).items(), key=lambda x: -x[1]):
        print(f"    {cat}: {score}")

    return df


def step2_train_model(force: bool = False):
    """Step 2: Train XGBoost model."""
    print("\n" + "=" * 50)
    print("Step 2: XGBoost Model Training")
    print("=" * 50)

    result = train_model(DATA_DIR, force_retrain=force)

    print(f"  Data source: {result['data_source']}")
    print(f"  Features: {result['feature_names']}")
    print(f"  Categories: {len(result['models'])}")
    print(f"  SHAP explainers: {len(result.get('shap_explainers', {}))}")
    print(f"\n  MSE per category:")
    for cat, mse in sorted(result["metrics"].items(), key=lambda x: x[1]):
        print(f"    {cat:12s}: {mse:.4f}")

    return result


def step3_test_predictions():
    """Step 3: Test predictions with XGBoost model."""
    print("\n" + "=" * 50)
    print("Step 3: Prediction Tests (XGBoost)")
    print("=" * 50)

    profiler = HealthProfiler(data_dir=DATA_DIR, use_ml=True)
    print(f"  Model used: {profiler.model_source}")

    test_profiles = [
        UserProfile(age=32, gender="M", occupation_category="사무직"),
        UserProfile(age=35, gender="F", occupation_category="사무직",
                    family_size=3, has_children=True,
                    health_concerns=["허리 통증", "스트레스"]),
        UserProfile(age=55, gender="M", occupation_category="생산직"),
    ]

    for profile in test_profiles:
        iv = profiler.generate_interest_vector(profile)
        desc = f"{profile.age}세 {profile.gender} {profile.occupation_category}"
        print(f"\n  [{desc}] model={iv.model_used}")
        print(f"    Top 3: {iv.top_interests}")
        top5 = sorted(iv.to_dict().items(), key=lambda x: -x[1])[:5]
        for cat, score in top5:
            print(f"      {cat}: {score:.3f}")


def step4_compare_rule_vs_xgb():
    """Step 4: Compare rule-based vs XGBoost predictions."""
    print("\n" + "=" * 50)
    print("Step 4: Rule-based vs XGBoost Comparison")
    print("=" * 50)

    profiler_rule = HealthProfiler(data_dir=DATA_DIR, use_ml=False)
    profiler_xgb = HealthProfiler(data_dir=DATA_DIR, use_ml=True)

    profile = UserProfile(
        age=35, gender="F", occupation_category="사무직",
        family_size=3, has_children=True,
        health_concerns=["허리 통증", "스트레스", "눈 피로"],
    )

    iv_rule = profiler_rule.generate_interest_vector(profile)
    iv_xgb = profiler_xgb.generate_interest_vector(profile)

    print(f"\n  {'Category':12s}  {'Rule-based':>10s}  {'XGBoost':>10s}  {'Delta':>8s}")
    print(f"  {'-'*12}  {'-'*10}  {'-'*10}  {'-'*8}")

    for cat in INTEREST_CATEGORIES:
        r_score = iv_rule.to_dict().get(cat, 0)
        x_score = iv_xgb.to_dict().get(cat, 0)
        delta = x_score - r_score
        marker = " *" if abs(delta) > 0.1 else ""
        print(f"  {cat:12s}  {r_score:10.3f}  {x_score:10.3f}  {delta:+8.3f}{marker}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Force retrain")
    args = parser.parse_args()

    (DATA_DIR / "nhis_stats").mkdir(parents=True, exist_ok=True)
    Path(Path(__file__).parent.parent / "models").mkdir(exist_ok=True)

    step1_generate_nhis()
    step2_train_model(force=args.force)
    step3_test_predictions()
    step4_compare_rule_vs_xgb()

    print("\n" + "=" * 50)
    print("Training complete.")
    print("Model saved to: models/xgb_profiler.pkl")
    print("Dashboard will auto-detect the model on next run.")
