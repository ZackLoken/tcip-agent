"""Temporal aggregation for phenology and other trait types.

Supports multiple aggregation strategies based on trait type:
  - sigmoid: Fits a 4-parameter logistic for phenological milestones
  - count: Simple statistics (sum, mean, max, std) for counting traits
  - proportion: Ratio-based aggregation for presence/absence
  - regression: Linear trend fitting for growth/size traits

Trait type is looked up from registry/crops.yml or specified explicitly.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Aggregation strategies
# ---------------------------------------------------------------------------

def fit_sigmoid(dates: list[str], counts: list[float]) -> dict:
    """Fit a 4-parameter logistic (sigmoid) to time-series detection counts.

    Uses simple least-squares fitting without scipy dependency.

    Args:
        dates: ISO date strings ('YYYY-MM-DD').
        counts: Detection counts or proportions per date.

    Returns:
        Dict with fitted parameters and milestone dates.
    """
    if len(dates) < 3 or len(dates) != len(counts):
        return {"error": "Need at least 3 date-count pairs"}

    parsed = [datetime.strptime(d, "%Y-%m-%d") for d in dates]
    base = parsed[0]
    x = np.array([(d - base).days for d in parsed], dtype=np.float64)
    y = np.array(counts, dtype=np.float64)

    y_min, y_max = y.min(), y.max()
    if y_max - y_min < 1e-10:
        return {"error": "No variation in counts", "dates": dates, "counts": counts}
    y_norm = (y - y_min) / (y_max - y_min)

    x0_est = float(np.interp(0.5, y_norm, x)) if y_norm[0] < y_norm[-1] else float(np.interp(0.5, y_norm[::-1], x[::-1]))

    dy = np.gradient(y_norm, x)
    k_est = float(np.max(np.abs(dy)) * 4)
    if k_est < 0.01:
        k_est = 0.1

    milestones = {}
    for threshold, name in [(0.05, "05per"), (0.50, "50per"), (0.95, "95per")]:
        try:
            if y_norm[0] < y_norm[-1]:
                day = float(np.interp(threshold, y_norm, x))
            else:
                day = float(np.interp(threshold, y_norm[::-1], x[::-1]))
            milestone_date = base + timedelta(days=round(day))
            milestones[name] = milestone_date.strftime("%Y-%m-%d")
        except (ValueError, IndexError):
            milestones[name] = None

    return {
        "strategy": "sigmoid",
        "inflection_day": round(x0_est, 1),
        "steepness": round(k_est, 4),
        "y_min": round(float(y_min), 2),
        "y_max": round(float(y_max), 2),
        "milestones": milestones,
        "base_date": base.strftime("%Y-%m-%d"),
    }


def aggregate_counts(dates: list[str], counts: list[float]) -> dict:
    """Simple count statistics for non-phenological counting traits.

    Returns:
        Dict with total, mean, max, min, std, and per-date counts.
    """
    arr = np.array(counts, dtype=np.float64)
    return {
        "strategy": "count",
        "total": round(float(arr.sum()), 2),
        "mean": round(float(arr.mean()), 4),
        "max": round(float(arr.max()), 2),
        "min": round(float(arr.min()), 2),
        "std": round(float(arr.std()), 4),
        "n_observations": len(counts),
    }


def aggregate_proportion(dates: list[str], counts: list[float]) -> dict:
    """Proportion-based aggregation for presence/absence traits.

    Treats any count > 0 as 'present'. Returns proportion of dates
    where the trait was observed.
    """
    present = sum(1 for c in counts if c > 0)
    total = len(counts)
    return {
        "strategy": "proportion",
        "present_count": present,
        "total_observations": total,
        "proportion": round(present / total, 4) if total > 0 else 0.0,
    }


def fit_linear_trend(dates: list[str], counts: list[float]) -> dict:
    """Linear regression for growth/size trends over time.

    Returns slope (units/day), intercept, and R².
    """
    if len(dates) < 2:
        return {"error": "Need at least 2 observations for trend"}

    parsed = [datetime.strptime(d, "%Y-%m-%d") for d in dates]
    base = parsed[0]
    x = np.array([(d - base).days for d in parsed], dtype=np.float64)
    y = np.array(counts, dtype=np.float64)

    # Simple linear regression
    n = len(x)
    sx = x.sum()
    sy = y.sum()
    sxx = (x * x).sum()
    sxy = (x * y).sum()
    denom = n * sxx - sx * sx
    if abs(denom) < 1e-10:
        return {"error": "Degenerate regression (all same x)"}

    slope = (n * sxy - sx * sy) / denom
    intercept = (sy - slope * sx) / n

    y_pred = slope * x + intercept
    ss_res = float(((y - y_pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 1e-10 else 0.0

    return {
        "strategy": "linear",
        "slope_per_day": round(float(slope), 6),
        "intercept": round(float(intercept), 4),
        "r_squared": round(r_squared, 4),
        "base_date": base.strftime("%Y-%m-%d"),
    }


# ---------------------------------------------------------------------------
# Strategy dispatch
# ---------------------------------------------------------------------------

# Map from trait measurement types to aggregation strategies
TRAIT_TYPE_STRATEGIES = {
    "phenology": "sigmoid",
    "counting": "count",
    "density": "count",
    "growth": "linear",
    "presence": "proportion",
    "size": "linear",
    "coverage": "proportion",
}

def fit_gompertz(dates: list[str], counts: list[float]) -> dict:
    """Fit a Gompertz growth curve: y = a * exp(-b * exp(-c * x)).

    Good for asymmetric growth curves common in tree crown expansion
    and biomass accumulation where growth is rapid early and slows later.
    """
    if len(dates) < 3:
        return {"error": "Need at least 3 observations for Gompertz fit"}

    parsed = [datetime.strptime(d, "%Y-%m-%d") for d in dates]
    base = parsed[0]
    x = np.array([(d - base).days for d in parsed], dtype=np.float64)
    y = np.array(counts, dtype=np.float64)

    a_est = float(y.max())
    if a_est < 1e-10:
        return {"error": "No growth detected (all counts near zero)"}

    y_norm = y / a_est
    y_norm = np.clip(y_norm, 1e-6, 1.0 - 1e-6)

    # Estimate b and c from linearized form: ln(-ln(y/a)) = ln(b) - c*x
    inner = -np.log(y_norm + 1e-8)
    inner = np.clip(inner, 1e-8, None)
    outer = np.log(inner)

    # Linear fit on outer vs x
    n = len(x)
    sx, sy = x.sum(), outer.sum()
    sxx, sxy = (x * x).sum(), (x * outer).sum()
    denom = n * sxx - sx * sx
    if abs(denom) < 1e-10:
        return {"error": "Degenerate fit"}

    c_est = -(n * sxy - sx * sy) / denom
    ln_b = (sy + c_est * sx) / n
    b_est = float(np.exp(ln_b))

    return {
        "strategy": "gompertz",
        "asymptote": round(a_est, 2),
        "displacement": round(b_est, 4),
        "growth_rate": round(float(c_est), 6),
        "base_date": base.strftime("%Y-%m-%d"),
    }


def fit_logistic_growth(dates: list[str], counts: list[float]) -> dict:
    """Fit a logistic growth curve: y = K / (1 + exp(-r*(x - x0))).

    Standard S-curve for population/size growth with carrying capacity.
    """
    if len(dates) < 3:
        return {"error": "Need at least 3 observations"}

    parsed = [datetime.strptime(d, "%Y-%m-%d") for d in dates]
    base = parsed[0]
    x = np.array([(d - base).days for d in parsed], dtype=np.float64)
    y = np.array(counts, dtype=np.float64)

    K_est = float(y.max()) * 1.1  # carrying capacity slightly above observed max
    if K_est < 1e-10:
        return {"error": "No growth detected"}

    y_norm = y / K_est
    y_norm = np.clip(y_norm, 1e-6, 1.0 - 1e-6)

    # Linearize: ln(y/(K-y)) = r*x - r*x0
    logit = np.log(y_norm / (1.0 - y_norm))

    n = len(x)
    sx, sy = x.sum(), logit.sum()
    sxx, sxy = (x * x).sum(), (x * logit).sum()
    denom = n * sxx - sx * sx
    if abs(denom) < 1e-10:
        return {"error": "Degenerate fit"}

    r_est = (n * sxy - sx * sy) / denom
    x0_est = -(sy - r_est * sx) / (n * r_est) if abs(r_est) > 1e-10 else 0

    return {
        "strategy": "logistic_growth",
        "carrying_capacity": round(K_est, 2),
        "growth_rate": round(float(r_est), 6),
        "midpoint_day": round(float(x0_est), 1),
        "base_date": base.strftime("%Y-%m-%d"),
    }


STRATEGY_FNS = {
    "sigmoid": fit_sigmoid,
    "count": aggregate_counts,
    "proportion": aggregate_proportion,
    "linear": fit_linear_trend,
    "gompertz": fit_gompertz,
    "logistic_growth": fit_logistic_growth,
}


def get_strategy_for_trait(trait_name: str, trait_type: str | None = None) -> str:
    """Determine the aggregation strategy for a trait.

    Args:
        trait_name: Name of the trait (e.g. 'catkin_count', 'burrs_density').
        trait_type: Explicit type override. If None, inferred from name.

    Returns:
        Strategy name: 'sigmoid', 'count', 'proportion', or 'linear'.
    """
    if trait_type and trait_type in TRAIT_TYPE_STRATEGIES:
        return TRAIT_TYPE_STRATEGIES[trait_type]

    # Infer from name patterns
    name_lower = trait_name.lower()
    if any(kw in name_lower for kw in ("phenol", "timing", "date", "onset", "bloom")):
        return "sigmoid"
    if any(kw in name_lower for kw in ("density", "count", "number")):
        return "count"
    if any(kw in name_lower for kw in ("presence", "absence", "coverage")):
        return "proportion"
    if any(kw in name_lower for kw in ("size", "area", "height", "growth", "diameter")):
        return "linear"

    # Default to sigmoid for backward compatibility
    logger.info("No specific strategy for trait '%s', defaulting to sigmoid", trait_name)
    return "sigmoid"


def aggregate_trait(
    dates: list[str],
    counts: list[float],
    strategy: str = "sigmoid",
) -> dict:
    """Run the named aggregation strategy on a time-series.

    Args:
        dates: ISO date strings.
        counts: Values per date.
        strategy: One of 'sigmoid', 'count', 'proportion', 'linear'.

    Returns:
        Strategy-specific result dict.
    """
    fn = STRATEGY_FNS.get(strategy)
    if fn is None:
        raise ValueError(f"Unknown strategy '{strategy}'. Options: {list(STRATEGY_FNS.keys())}")
    return fn(dates, counts)


# ---------------------------------------------------------------------------
# Per-plant aggregation (generalized)
# ---------------------------------------------------------------------------

def aggregate_per_plant(
    plant_ids: list[str],
    dates: list[str],
    detections: dict[str, dict[str, float]],
    trait_name: str = "catkin",
    trait_type: str | None = None,
) -> list[dict]:
    """Aggregate detections per plant across dates.

    Args:
        plant_ids: List of plant identifiers.
        dates: Sorted list of observation dates.
        detections: Nested dict: detections[plant_id][date] = count.
        trait_name: Name of the trait being measured.
        trait_type: Explicit trait type for strategy selection.

    Returns:
        List of dicts, one per plant, with aggregation results.
    """
    strategy = get_strategy_for_trait(trait_name, trait_type)
    logger.info("Using '%s' strategy for trait '%s'", strategy, trait_name)

    results: list[dict] = []
    for pid in plant_ids:
        plant_data = detections.get(pid, {})
        plant_dates = sorted(d for d in dates if d in plant_data)
        plant_counts = [plant_data[d] for d in plant_dates]

        if len(plant_dates) >= 2:
            agg = aggregate_trait(plant_dates, plant_counts, strategy)
        else:
            agg = {"error": "Insufficient observations", "observation_count": len(plant_dates)}

        results.append({
            "plant_id": pid,
            "trait": trait_name,
            "strategy": strategy,
            "observations": len(plant_dates),
            "max_count": max(plant_counts) if plant_counts else 0,
            "aggregation": agg,
        })

    return results
