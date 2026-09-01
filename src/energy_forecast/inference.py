"""Significance tests for the rolling-origin benchmark.

Per-origin metrics are the sampling unit wherever possible: hours inside one
24-hour window share weather and demand level, so treating them as independent
would overstate confidence. The Diebold-Mariano test works on hourly losses but
corrects its variance for that serial dependence instead.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def _paired(a, b) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.ndim != 1 or a.shape != b.shape:
        raise ValueError("expected two 1-D arrays of equal length")
    return a, b


def paired_model_comparison(
    errors_a, errors_b, n_boot: int = 10_000, seed: int = 42
) -> dict[str, float]:
    """Wilcoxon signed-rank plus a paired-bootstrap 95% CI on mean(a - b).

    Negative values mean model A errs less than model B.
    """
    a, b = _paired(errors_a, errors_b)
    diff = a - b
    rng = np.random.default_rng(seed)
    samples = diff[rng.integers(0, len(diff), size=(n_boot, len(diff)))].mean(axis=1)
    lower, upper = np.quantile(samples, [0.025, 0.975])
    p = 1.0 if np.allclose(diff, 0.0) else float(stats.wilcoxon(a, b).pvalue)
    return {
        "mean_diff": float(diff.mean()),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "wilcoxon_p": p,
        "n": int(len(diff)),
    }


def diebold_mariano(loss_a, loss_b, h: int = 24) -> dict[str, float]:
    """Diebold-Mariano on per-hour loss differentials, HLN small-sample corrected.

    ``h`` is the forecast horizon: the long-run variance sums h-1 autocovariance
    lags because errors inside one h-step window are serially dependent.
    """
    a, b = _paired(loss_a, loss_b)
    d = a - b
    n = len(d)
    if n <= 2 * h:
        raise ValueError(f"need more than {2 * h} loss pairs, got {n}")
    if np.allclose(d, 0.0):
        return {"dm_stat": 0.0, "p_value": 1.0, "n": n}
    centered = d - d.mean()
    gamma = np.array([centered[k:] @ centered[: n - k] for k in range(h)]) / n
    long_run = gamma[0] + 2.0 * gamma[1:].sum()
    if long_run <= 0:  # the truncated kernel can go negative on short series
        long_run = gamma[0]
    if long_run <= 0:  # a differential with no variance at all: the gap is deterministic
        return {"dm_stat": float(np.sign(d.mean()) * np.inf), "p_value": 0.0, "n": n}
    dm = d.mean() / np.sqrt(long_run / n)
    hln = dm * np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    return {
        "dm_stat": float(hln),
        "p_value": float(2 * stats.t.sf(abs(hln), df=n - 1)),
        "n": n,
    }


def slice_difference_test(
    values, mask, n_boot: int = 10_000, n_perm: int = 5_000, seed: int = 42
) -> dict[str, float]:
    """Mean difference between the masked slice and the rest, with uncertainty.

    Bootstrap 95% CI (resampling within each group), permutation p-value, and
    Cohen's d. Feed per-day values, not per-hour, when days are the natural unit.
    """
    data = np.asarray(values, dtype=float)
    flags = np.asarray(mask, dtype=bool)
    if data.shape != flags.shape or data.ndim != 1:
        raise ValueError("values and mask must be 1-D arrays of equal length")
    inside, outside = data[flags], data[~flags]
    if len(inside) == 0 or len(outside) == 0:
        raise ValueError("both sides of the mask need at least one value")
    observed = float(inside.mean() - outside.mean())
    rng = np.random.default_rng(seed)
    boot = (
        inside[rng.integers(0, len(inside), size=(n_boot, len(inside)))].mean(axis=1)
        - outside[rng.integers(0, len(outside), size=(n_boot, len(outside)))].mean(axis=1)
    )
    lower, upper = np.quantile(boot, [0.025, 0.975])
    exceed = 0
    for _ in range(n_perm):
        shuffled = rng.permutation(flags)
        exceed += abs(data[shuffled].mean() - data[~shuffled].mean()) >= abs(observed)
    pooled_var = (
        (len(inside) - 1) * inside.var(ddof=1) + (len(outside) - 1) * outside.var(ddof=1)
    ) / (len(inside) + len(outside) - 2)
    return {
        "mean_diff": observed,
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "permutation_p": float((exceed + 1) / (n_perm + 1)),
        "cohens_d": float(observed / np.sqrt(pooled_var)) if pooled_var > 0 else 0.0,
        "n_slice": int(len(inside)),
        "n_rest": int(len(outside)),
    }


def coverage_test(
    per_origin_coverage, nominal: float = 0.95, n_boot: int = 10_000, seed: int = 42
) -> dict[str, float]:
    """Is observed interval coverage consistent with the nominal rate?

    Per-origin coverage rates are the sampling unit -- a raw binomial test over
    hours would be overconfident because hours inside one origin are correlated.
    Bootstrap 95% CI on the mean rate plus a one-sample t-test against ``nominal``.
    """
    rates = np.asarray(per_origin_coverage, dtype=float)
    if rates.ndim != 1 or len(rates) < 2:
        raise ValueError("need at least two per-origin coverage rates")
    rng = np.random.default_rng(seed)
    samples = rates[rng.integers(0, len(rates), size=(n_boot, len(rates)))].mean(axis=1)
    lower, upper = np.quantile(samples, [0.025, 0.975])
    t = stats.ttest_1samp(rates, nominal)
    return {
        "mean_coverage": float(rates.mean()),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "nominal": float(nominal),
        "p_value": float(t.pvalue),
        "n_origins": int(len(rates)),
    }
