import math

import numpy as np


def modified_rs(values, q=20):
    data = np.asarray(values, dtype=float)
    if len(data) < max(64, q + 2):
        return None
    centered = data - data.mean()
    variance = float(np.dot(centered, centered) / len(data))
    for lag in range(1, q + 1):
        gamma = float(np.dot(centered[lag:], centered[:-lag]) / len(data))
        variance += 2 * (1 - lag / (q + 1)) * gamma
    if variance <= 1e-28:
        return None
    cumulative = np.concatenate(([0.0], np.cumsum(centered)))
    statistic = float(np.ptp(cumulative) / math.sqrt(variance))
    return math.log(statistic) / math.log(len(data)) if statistic > 0 else None


def paired_bootstrap(differences, seed=123, repeats=3000):
    data = np.asarray([d for d in differences if d is not None and math.isfinite(d)])
    if len(data) < 3:
        return {"n": len(data), "mean": float(data.mean()) if len(data) else None,
                "ci_low": None, "ci_high": None}
    generator = np.random.default_rng(seed)
    means = generator.choice(data, size=(repeats, len(data)), replace=True).mean(axis=1)
    return {"n": len(data), "mean": float(data.mean()),
            "ci_low": float(np.quantile(means, 0.025)),
            "ci_high": float(np.quantile(means, 0.975))}
