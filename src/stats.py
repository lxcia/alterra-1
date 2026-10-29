"""Small, dependency-light statistics helpers."""
from __future__ import annotations

import numpy as np
from scipy.stats import binomtest


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """95% Wilson score interval for k successes out of n."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def fmt_prop(k: int, n: int) -> str:
    lo, hi = wilson(k, n)
    return f"{k}/{n} = {k / n:.0%} [95% CI {lo:.0%}-{hi:.0%}]" if n else "0/0"


def mcnemar_exact(b: int, c: int) -> float:
    """Exact two-sided McNemar p-value from the discordant counts b (0->1) and c (1->0)."""
    n = b + c
    if n == 0:
        return 1.0
    return float(binomtest(b, n, 0.5).pvalue)


def bootstrap_ci(values_by_patient: dict, stat=np.mean, n_boot: int = 10000, seed: int = 0):
    """Patient-level bootstrap CI. values_by_patient: {patient: array of per-observation values}."""
    rng = np.random.default_rng(seed)
    keys = list(values_by_patient)
    arrs = [np.asarray(values_by_patient[k], float) for k in keys]
    point = stat(np.concatenate(arrs))
    boots = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(keys), len(keys))
        boots.append(stat(np.concatenate([arrs[i] for i in pick])))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def cluster_bootstrap_prop(success, cluster, n_boot: int = 10000, seed: int = 0):
    """Proportion of successes with a 95% CI from a bootstrap over clusters (patients).

    Use when a patient contributes several reads (e.g. several prompt variants or sequences), so
    reads are not independent. Returns (point, lo, hi, n_reads, n_clusters)."""
    s = np.asarray(success, float)
    c = np.asarray(cluster)
    keys = np.unique(c)
    groups = [s[c == k] for k in keys]
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(groups), len(groups))
        allv = np.concatenate([groups[i] for i in pick])
        boots.append(allv.mean())
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(s.mean()), float(lo), float(hi), len(s), len(keys)
