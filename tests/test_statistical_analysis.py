import numpy as np
import pandas as pd
import pytest

from study_code.statistical_analysis import equal_cooling, paired_fixed_dose


def test_paired_analysis_uses_only_adaptive_and_hotspot():
    table = pd.DataFrame(
        {
            "city_id": [1, 1, 2, 2, 3, 3],
            "strategy": ["adaptive", "hotspot"] * 3,
            "cooling_C": [6.0, 4.0, 5.0, 4.0, 7.0, 5.0],
        }
    )
    result = paired_fixed_dose(table, replicates=200, seed=5)
    assert result["n_pairs"] == 3
    assert result["adaptive_win_count"] == 3
    assert result["adaptive_to_hotspot_ratio"]["estimate"] > 1.0


def test_equal_cooling_returns_lower_adaptive_requirement():
    rows = []
    for city in range(8):
        for dose in (5.0, 10.0, 20.0):
            rows.append((city, "adaptive", dose, 11.0 * (1 - np.exp(-0.11 * dose))))
            rows.append((city, "hotspot", dose, 9.0 * (1 - np.exp(-0.07 * dose))))
    table = pd.DataFrame(rows, columns=["city_id", "strategy", "dose_pp", "cooling_C"])
    result = equal_cooling(table, hotspot_target_dose_pp=20.0)
    assert result["adaptive_dose_pp"] < 20.0
    assert 0.0 < result["equal_cooling_reduction_fraction"] < 1.0


def test_ties_are_not_counted_as_adaptive_wins():
    table = pd.DataFrame(
        {
            "city_id": [1, 1, 2, 2],
            "strategy": ["adaptive", "hotspot"] * 2,
            "cooling_C": [4.0, 4.0, 5.0, 4.0],
        }
    )
    result = paired_fixed_dose(table, replicates=100, seed=3)
    assert result["adaptive_win_count"] == 1
    assert result["adaptive_win_fraction"] == 0.5


def test_string_city_ids_are_preserved():
    table = pd.DataFrame(
        {
            "city_id": ["CHI", "CHI", "IND", "IND"],
            "strategy": ["adaptive", "hotspot"] * 2,
            "cooling_C": [5.0, 4.0, 6.0, 4.5],
        }
    )
    result = paired_fixed_dose(table, replicates=20)
    assert result["city_ids"] == ["CHI", "IND"]


def test_missing_strategy_has_clear_error():
    table = pd.DataFrame(
        {"city_id": [1, 2], "strategy": ["adaptive", "adaptive"], "cooling_C": [5, 6]}
    )
    with pytest.raises(ValueError, match="missing strategies.*hotspot"):
        paired_fixed_dose(table, replicates=20)


def test_duplicate_city_strategy_is_rejected():
    table = pd.DataFrame(
        {
            "city_id": [1, 1, 1],
            "strategy": ["adaptive", "adaptive", "hotspot"],
            "cooling_C": [5.0, 5.1, 4.0],
        }
    )
    with pytest.raises(ValueError, match="duplicate city/strategy"):
        paired_fixed_dose(table, replicates=20)


def test_equal_cooling_rejects_extrapolated_target_dose():
    rows = []
    for strategy in ("adaptive", "hotspot"):
        for dose in (5.0, 10.0, 20.0):
            rows.append((strategy, dose, 8.0 * (1.0 - np.exp(-0.1 * dose))))
    table = pd.DataFrame(rows, columns=["strategy", "dose_pp", "cooling_C"])
    with pytest.raises(ValueError, match="outside the measured dose range"):
        equal_cooling(table, hotspot_target_dose_pp=25.0)


def test_equal_cooling_rejects_extrapolated_adaptive_inversion():
    rows = []
    for dose in (5.0, 10.0, 20.0):
        rows.append(("adaptive", dose, 20.0 * (1.0 - np.exp(-0.5 * dose))))
        rows.append(("hotspot", dose, 5.0 * (1.0 - np.exp(-0.05 * dose))))
    table = pd.DataFrame(rows, columns=["strategy", "dose_pp", "cooling_C"])
    with pytest.raises(ValueError, match="inferred Adaptive.*outside the measured"):
        equal_cooling(table, hotspot_target_dose_pp=20.0)
