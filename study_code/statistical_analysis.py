"""Paired Adaptive-versus-Hotspot statistics and equal-cooling analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


def block_bootstrap_median(
    values: np.ndarray,
    blocks: np.ndarray,
    replicates: int = 50_000,
    seed: int = 42,
) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    blocks = np.asarray(blocks)
    if not isinstance(replicates, int) or isinstance(replicates, bool) or replicates < 1:
        raise ValueError("replicates must be a positive integer")
    valid = np.isfinite(values) & pd.notna(blocks)
    values, blocks = values[valid], blocks[valid]
    unique = np.unique(blocks)
    if not len(unique):
        raise ValueError("no valid geographic blocks")
    members = [np.flatnonzero(blocks == block) for block in unique]
    rng = np.random.default_rng(seed)
    result = np.empty(replicates, dtype=float)
    for index in range(replicates):
        sampled = rng.integers(0, len(unique), len(unique))
        result[index] = np.median(values[np.concatenate([members[item] for item in sampled])])
    low, high = np.percentile(result, (2.5, 97.5))
    return {"estimate": float(np.median(values)), "ci95_low": float(low), "ci95_high": float(high)}


def paired_fixed_dose(
    table: pd.DataFrame,
    blocks: pd.Series | None = None,
    replicates: int = 50_000,
    seed: int = 42,
) -> dict:
    table = table.rename(
        columns={"city": "city_id", "arm": "strategy", "domain_cooling_C": "cooling_C"}
    )
    required = {"city_id", "strategy", "cooling_C"}
    if missing := required - set(table.columns):
        raise ValueError(f"missing columns: {sorted(missing)}")
    aliases = {
        "adaptive": "adaptive",
        "svf_gate": "adaptive",
        "shade opportunity": "adaptive",
        "hotspot": "hotspot",
        "hotspot_add": "hotspot",
    }
    frame = table.copy()
    if {"target_pixels", "n_added_core"}.issubset(frame.columns):
        exact = pd.to_numeric(frame["n_added_core"], errors="coerce").eq(
            pd.to_numeric(frame["target_pixels"], errors="coerce")
        )
        frame = frame[exact]
    if "constrained" in frame.columns:
        constrained = frame["constrained"].astype(str).str.lower().isin({"true", "1", "yes"})
        frame = frame[~constrained]
    frame["strategy"] = frame["strategy"].astype(str).str.strip().str.lower().map(aliases)
    frame["cooling_C"] = pd.to_numeric(frame["cooling_C"], errors="coerce")
    frame = frame.dropna(subset=["city_id", "strategy", "cooling_C"])
    duplicates = frame.duplicated(["city_id", "strategy"], keep=False)
    if duplicates.any():
        duplicate_pairs = (
            frame.loc[duplicates, ["city_id", "strategy"]]
            .drop_duplicates()
            .to_dict("records")
        )
        raise ValueError(f"duplicate city/strategy observations: {duplicate_pairs}")
    available = set(frame["strategy"])
    missing_strategies = {"adaptive", "hotspot"} - available
    if missing_strategies:
        raise ValueError(f"missing strategies: {sorted(missing_strategies)}")
    wide = frame.pivot(index="city_id", columns="strategy", values="cooling_C")
    wide = wide.dropna(subset=["adaptive", "hotspot"])
    if wide.empty:
        raise ValueError("no paired Adaptive/Hotspot observations")
    adaptive = wide["adaptive"].to_numpy(float)
    hotspot = wide["hotspot"].to_numpy(float)
    if np.any(hotspot <= 0) or np.any(adaptive <= 0):
        raise ValueError("cooling magnitudes must be positive")
    if blocks is None:
        block_values = np.arange(len(wide))
    else:
        aligned_blocks = blocks.reindex(wide.index)
        if aligned_blocks.isna().any():
            missing_ids = wide.index[aligned_blocks.isna()].tolist()
            raise ValueError(f"missing geographic blocks for city IDs: {missing_ids}")
        block_values = aligned_blocks.to_numpy()
    metrics = {
        "adaptive_cooling_C": adaptive,
        "hotspot_cooling_C": hotspot,
        "adaptive_to_hotspot_ratio": adaptive / hotspot,
        "equal_cooling_reduction_fraction": 1.0 - hotspot / adaptive,
        "adaptive_minus_hotspot_C": adaptive - hotspot,
    }
    city_ids = [value.item() if isinstance(value, np.generic) else value for value in wide.index]
    if any(not isinstance(value, (str, int, float, bool)) for value in city_ids):
        raise ValueError("city IDs must be JSON-compatible scalar values")
    output = {"n_pairs": len(wide), "city_ids": city_ids}
    for offset, (name, values) in enumerate(metrics.items()):
        output[name] = block_bootstrap_median(values, block_values, replicates, seed + offset)
    output["adaptive_win_count"] = int(np.sum(adaptive > hotspot))
    output["adaptive_win_fraction"] = float(np.mean(adaptive > hotspot))
    return output


def _saturating(dose: np.ndarray, maximum: float, rate: float) -> np.ndarray:
    return maximum * (1.0 - np.exp(-rate * dose))


def equal_cooling(dose_table: pd.DataFrame, hotspot_target_dose_pp: float = 20.0) -> dict:
    if not np.isfinite(hotspot_target_dose_pp) or hotspot_target_dose_pp <= 0:
        raise ValueError("hotspot_target_dose_pp must be finite and positive")
    dose_table = dose_table.copy()
    if "cooling_C" not in dose_table and "dTmrt_peak_mean" in dose_table:
        dose_table["cooling_C"] = -pd.to_numeric(
            dose_table["dTmrt_peak_mean"], errors="coerce"
        )
    required = {"strategy", "dose_pp", "cooling_C"}
    if missing := required - set(dose_table.columns):
        raise ValueError(f"missing columns: {sorted(missing)}")
    aliases = {"adaptive": "adaptive", "svf_gate": "adaptive", "hotspot": "hotspot"}
    frame = dose_table.copy()
    frame["strategy"] = frame["strategy"].astype(str).str.strip().str.lower().map(aliases)
    frame["dose_pp"] = pd.to_numeric(frame["dose_pp"], errors="coerce")
    frame["cooling_C"] = pd.to_numeric(frame["cooling_C"], errors="coerce")
    frame = frame.dropna(subset=["strategy", "dose_pp", "cooling_C"])
    if (frame["dose_pp"] <= 0).any():
        raise ValueError("measured doses must be positive")
    missing_strategies = {"adaptive", "hotspot"} - set(frame["strategy"])
    if missing_strategies:
        raise ValueError(f"missing strategies: {sorted(missing_strategies)}")
    hotspot_doses = frame.loc[frame["strategy"] == "hotspot", "dose_pp"]
    if not hotspot_doses.min() <= hotspot_target_dose_pp <= hotspot_doses.max():
        raise ValueError("hotspot target dose lies outside the measured dose range")
    medians = frame.groupby(["strategy", "dose_pp"])["cooling_C"].median()
    fits = {}
    for strategy in ("adaptive", "hotspot"):
        curve = medians.loc[strategy].sort_index()
        x = np.r_[0.0, curve.index.to_numpy(float)]
        y = np.r_[0.0, curve.to_numpy(float)]
        if len(x) < 4:
            raise ValueError(f"{strategy} requires at least three measured doses")
        if np.any(~np.isfinite(y)) or np.any(y < 0.0):
            raise ValueError(f"{strategy} cooling values must be finite and non-negative")
        parameters, _ = curve_fit(
            _saturating, x, y, p0=(max(y) * 1.2, 0.05),
            bounds=(0.0, np.inf), maxfev=20_000,
        )
        if (
            np.any(~np.isfinite(parameters))
            or parameters[0] <= 0.0
            or parameters[1] <= np.finfo(float).eps
        ):
            raise RuntimeError(f"{strategy} saturating-curve fit is degenerate")
        fits[strategy] = parameters
    target = float(_saturating(hotspot_target_dose_pp, *fits["hotspot"]))
    maximum, rate = fits["adaptive"]
    if target >= maximum:
        raise ValueError("hotspot target lies above the fitted Adaptive maximum")
    adaptive_dose = float(-np.log(1.0 - target / maximum) / rate)
    if not np.isfinite(adaptive_dose) or adaptive_dose <= 0:
        raise RuntimeError("equal-cooling inversion did not produce a positive finite dose")
    adaptive_doses = frame.loc[frame["strategy"] == "adaptive", "dose_pp"]
    adaptive_min = float(adaptive_doses.min())
    adaptive_max = float(adaptive_doses.max())
    if not adaptive_min <= adaptive_dose <= adaptive_max:
        raise ValueError(
            "inferred Adaptive equal-cooling dose lies outside the measured "
            f"Adaptive dose range [{adaptive_min:g}, {adaptive_max:g}] pp"
        )
    return {
        "hotspot_target_dose_pp": hotspot_target_dose_pp,
        "target_cooling_C": target,
        "adaptive_dose_pp": adaptive_dose,
        "adaptive_requirement_fraction": adaptive_dose / hotspot_target_dose_pp,
        "equal_cooling_reduction_fraction": 1.0 - adaptive_dose / hotspot_target_dose_pp,
        "fit_parameters": {
            name: {"maximum_C": float(value[0]), "rate_per_pp": float(value[1])}
            for name, value in fits.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    fixed = subparsers.add_parser("paired")
    fixed.add_argument("table", type=Path)
    fixed.add_argument("--blocks", type=Path)
    fixed.add_argument("--replicates", type=int, default=50_000)
    fixed.add_argument("--output", type=Path, required=True)
    dose = subparsers.add_parser("equal-cooling")
    dose.add_argument("table", type=Path)
    dose.add_argument("--adaptive-5pp", type=Path)
    dose.add_argument("--hotspot-5pp", type=Path)
    dose.add_argument("--target-dose-pp", type=float, default=20.0)
    dose.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    table = pd.read_csv(args.table)
    if args.command == "paired":
        blocks = None
        if args.blocks:
            block_table = pd.read_csv(args.blocks)
            block_table = block_table.rename(columns={"city": "city_id"})
            blocks = block_table.set_index("city_id")["geographic_block"]
        result = paired_fixed_dose(table, blocks, args.replicates)
    else:
        additions = []
        for strategy, path in (
            ("adaptive", args.adaptive_5pp),
            ("hotspot", args.hotspot_5pp),
        ):
            if path:
                extra = pd.read_csv(path)
                extra["strategy"] = strategy
                extra["dose_pp"] = 5.0
                additions.append(extra)
        if additions:
            table = pd.concat([table, *additions], ignore_index=True, sort=False)
        result = equal_cooling(table, args.target_dose_pp)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
