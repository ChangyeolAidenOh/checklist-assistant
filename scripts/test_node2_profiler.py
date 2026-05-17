"""
Step 1: Node 2 (Health Profiler) standalone test.

No API key needed. Verifies:
  - UserProfile creation
  - Interest vector generation
  - Top-3 interest ranking
  - Occupation/age/concern modifiers working correctly

Run: python scripts/test_node2_profiler.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core_pipeline.profiler.health_profiler import HealthProfiler, UserProfile


def test_basic_profile():
    """30s office worker, basic profile."""
    profiler = HealthProfiler()
    profile = UserProfile(
        age=32,
        gender="M",
        occupation_category="사무직",
        family_size=1,
        has_children=False,
        health_concerns=[],
    )
    result = profiler.generate_interest_vector(profile)

    print("=== Test 1: 30대 남성 사무직 (기본) ===")
    print(f"Top 3: {result.top_interests}")
    print(f"Scores: {result.to_dict()}")
    print()

    assert len(result.categories) == 12
    assert len(result.scores) == 12
    assert all(0.0 <= s <= 1.0 for s in result.scores)
    # Male should have very low 산부인과
    assert result.to_dict()["산부인과"] < 0.1
    print("PASSED")
    print()


def test_health_concerns():
    """30s female with health concerns."""
    profiler = HealthProfiler()
    profile = UserProfile(
        age=35,
        gender="F",
        occupation_category="사무직",
        family_size=3,
        has_children=True,
        health_concerns=["허리 통증", "스트레스", "눈 피로"],
    )
    result = profiler.generate_interest_vector(profile)

    print("=== Test 2: 35세 여성 사무직 (허리/스트레스/눈) ===")
    print(f"Top 3: {result.top_interests}")
    print(f"Scores: {result.to_dict()}")
    print(f"SHAP: {result.shap_explanations}")
    print()

    scores = result.to_dict()
    # Health concern boosts should work
    assert scores["정형외과"] > 0.3, f"정형외과 should be boosted, got {scores['정형외과']}"
    assert scores["정신건강"] > 0.3, f"정신건강 should be boosted, got {scores['정신건강']}"
    assert scores["안과"] > 0.3, f"안과 should be boosted, got {scores['안과']}"
    print("PASSED")
    print()


def test_age_groups():
    """Verify different age groups produce different vectors."""
    profiler = HealthProfiler()
    results = {}
    for age in [25, 35, 45, 55, 65]:
        profile = UserProfile(
            age=age,
            gender="M",
            occupation_category="사무직",
        )
        iv = profiler.generate_interest_vector(profile)
        results[age] = iv.to_dict()

    print("=== Test 3: Age group comparison ===")
    for age, scores in results.items():
        top3 = sorted(scores, key=scores.get, reverse=True)[:3]
        print(f"  Age {age}: top 3 = {top3}")

    # Older should have higher 입원 score
    assert results[65]["입원"] > results[25]["입원"]
    # Younger should have higher 피부과
    assert results[25]["피부과"] > results[65]["피부과"]
    print("PASSED")
    print()


def test_occupation_modifiers():
    """Verify occupation modifiers apply correctly."""
    profiler = HealthProfiler()

    office = profiler.generate_interest_vector(
        UserProfile(age=35, gender="M", occupation_category="사무직")
    )
    factory = profiler.generate_interest_vector(
        UserProfile(age=35, gender="M", occupation_category="생산직")
    )

    print("=== Test 4: Occupation comparison (사무직 vs 생산직) ===")
    print(f"  사무직 정신건강: {office.to_dict()['정신건강']}")
    print(f"  생산직 정신건강: {factory.to_dict()['정신건강']}")
    print(f"  사무직 정형외과: {office.to_dict()['정형외과']}")
    print(f"  생산직 정형외과: {factory.to_dict()['정형외과']}")

    # Office workers should have higher mental health concern
    assert office.to_dict()["정신건강"] > factory.to_dict()["정신건강"]
    # Factory workers should have higher orthopedic concern
    assert factory.to_dict()["정형외과"] > office.to_dict()["정형외과"]
    print("PASSED")
    print()


if __name__ == "__main__":
    test_basic_profile()
    test_health_concerns()
    test_age_groups()
    test_occupation_modifiers()
    print("=" * 40)
    print("All Node 2 tests PASSED")
