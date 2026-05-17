"""
NHIS Data Processor (DEPRECATED).

Superseded by real_data_processor.py which uses actual HIRA/NHIS public data.
Kept for synthetic data fallback when real data is not available.

Original purpose: Generate synthetic NHIS-style data for prototyping.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Category mapping: HIRA 진료과목 -> our interest categories
DEPARTMENT_TO_CATEGORY = {
    "내과": ["입원", "건강검진"],
    "외과": ["수술"],
    "정형외과": ["정형외과", "재활"],
    "신경외과": ["수술", "입원"],
    "흉부외과": ["수술", "입원"],
    "성형외과": [],
    "산부인과": ["산부인과"],
    "소아청소년과": [],
    "안과": ["안과"],
    "이비인후과": ["외래/통원"],
    "피부과": ["피부과"],
    "비뇨의학과": ["외래/통원"],
    "정신건강의학과": ["정신건강"],
    "재활의학과": ["재활"],
    "치과": ["치과"],
    "가정의학과": ["건강검진", "외래/통원"],
    "응급의학과": ["응급"],
}

AGE_GROUPS = ["0-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70+"]
GENDERS = ["M", "F"]


def generate_synthetic_nhis_data() -> pd.DataFrame:
    """Generate synthetic NHIS-style aggregate data.

    Based on published HIRA statistics patterns:
    - 20-30s: high outpatient, dental, mental health
    - 40-50s: rising hospitalization, surgery, health screening
    - 60+: highest overall utilization

    Returns:
        DataFrame with columns:
        [age_group, gender, department, patient_count, visit_count, cost_per_capita]
    """
    np.random.seed(42)
    records = []

    # Base utilization rates by department and age group (per 1000 population)
    # Source pattern: HIRA 2023 진료비통계지표
    base_rates = {
        "내과":         [120, 150, 250, 350, 500, 650, 800, 900],
        "외과":         [ 30,  40,  50,  70, 100, 120, 130, 110],
        "정형외과":     [ 20,  60,  80, 100, 150, 200, 250, 220],
        "산부인과":     [  5,  10,  50, 120,  80,  40,  20,  10],
        "안과":         [ 80,  70,  60,  80, 120, 170, 220, 250],
        "이비인후과":   [200, 150, 100, 100, 110, 100,  90,  80],
        "피부과":       [ 80, 120, 110,  90,  80,  70,  60,  50],
        "정신건강의학과":[  5,  30,  80, 100,  90,  70,  50,  40],
        "재활의학과":   [ 10,  20,  40,  60, 100, 140, 180, 200],
        "치과":         [100, 150, 180, 200, 190, 170, 150, 120],
        "가정의학과":   [ 30,  40,  80, 150, 250, 300, 320, 280],
        "응급의학과":   [ 40,  50,  50,  50,  60,  70,  90, 120],
    }

    # Gender modifiers
    gender_mod = {
        "산부인과": {"M": 0.02, "F": 1.0},
        "정신건강의학과": {"M": 0.8, "F": 1.2},
        "정형외과": {"M": 1.1, "F": 0.9},
        "피부과": {"M": 0.7, "F": 1.3},
    }

    for dept, rates in base_rates.items():
        for i, age_group in enumerate(AGE_GROUPS):
            for gender in GENDERS:
                g_mod = gender_mod.get(dept, {"M": 1.0, "F": 1.0}).get(gender, 1.0)
                rate = rates[i] * g_mod

                # Add noise
                rate = max(0, rate + np.random.normal(0, rate * 0.05))

                # Cost per visit (만원 단위)
                base_cost = 5 + (i * 2)  # increases with age
                if dept in ("외과", "정형외과", "산부인과"):
                    base_cost *= 2.5
                cost = base_cost + np.random.normal(0, 1)

                records.append({
                    "age_group": age_group,
                    "gender": gender,
                    "department": dept,
                    "patient_count_per_1000": round(rate, 1),
                    "visit_count_per_1000": round(rate * np.random.uniform(1.5, 3.0), 1),
                    "cost_per_capita_만원": round(max(0, cost), 2),
                })

    return pd.DataFrame(records)


def load_nhis_data(data_dir: Path) -> pd.DataFrame:
    """Load NHIS data from CSV or generate synthetic.

    Looks for 'nhis_utilization.csv' in data_dir.
    Falls back to synthetic data if not found.
    """
    csv_path = data_dir / "nhis_utilization.csv"

    if csv_path.exists():
        logger.info("Loading real NHIS data from %s", csv_path)
        return pd.read_csv(csv_path)

    logger.info("Real NHIS data not found. Using synthetic data.")
    df = generate_synthetic_nhis_data()

    # Save synthetic for reference
    df.to_csv(data_dir / "nhis_utilization_synthetic.csv", index=False)
    return df


def build_lookup_table(df: pd.DataFrame) -> dict:
    """Convert NHIS DataFrame into profiler-compatible lookup table.

    Returns:
        {age_group: {category: normalized_score}}
    """
    # Map departments to our interest categories
    category_scores = {}

    for age_group in AGE_GROUPS:
        age_data = df[df["age_group"] == age_group]
        scores = {}

        for _, row in age_data.iterrows():
            dept = row["department"]
            rate = row["patient_count_per_1000"]

            mapped_cats = DEPARTMENT_TO_CATEGORY.get(dept, [])
            for cat in mapped_cats:
                if cat not in scores:
                    scores[cat] = []
                scores[cat].append(rate)

        # Average and normalize to 0-1 range
        max_rate = df["patient_count_per_1000"].max()
        normalized = {}
        for cat, rates in scores.items():
            avg_rate = np.mean(rates)
            normalized[cat] = round(min(avg_rate / max_rate, 1.0), 3)

        category_scores[age_group] = normalized

    return category_scores


def build_gender_lookup(df: pd.DataFrame) -> dict:
    """Build gender-specific modifiers from NHIS data.

    Returns:
        {category: {M: modifier, F: modifier}}
    """
    modifiers = {}

    for dept, cats in DEPARTMENT_TO_CATEGORY.items():
        if not cats:
            continue

        m_data = df[(df["department"] == dept) & (df["gender"] == "M")]
        f_data = df[(df["department"] == dept) & (df["gender"] == "F")]

        if m_data.empty or f_data.empty:
            continue

        m_avg = m_data["patient_count_per_1000"].mean()
        f_avg = f_data["patient_count_per_1000"].mean()
        overall = (m_avg + f_avg) / 2

        if overall == 0:
            continue

        for cat in cats:
            modifiers[cat] = {
                "M": round(m_avg / overall, 3),
                "F": round(f_avg / overall, 3),
            }

    return modifiers
