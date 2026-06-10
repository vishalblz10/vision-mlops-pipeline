"""Data-drift detection: PSI and two-sample KS tests over feature distributions.

A reference window (what the model was trained/validated on) is compared with
a current production window. Population Stability Index is the primary signal
(rule of thumb: < 0.1 stable, 0.1-0.25 moderate shift, > 0.25 significant);
the Kolmogorov-Smirnov test provides a second, non-parametric check.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy import stats

PSI_THRESHOLD = 0.25
KS_PVALUE_THRESHOLD = 0.05


@dataclass(frozen=True)
class FeatureDrift:
    feature: str
    psi: float
    ks_statistic: float
    ks_pvalue: float
    drifted: bool


@dataclass(frozen=True)
class DriftReport:
    features: tuple[FeatureDrift, ...]
    drifted: bool

    def to_dict(self) -> dict:
        return {
            "drifted": self.drifted,
            "features": [asdict(f) for f in self.features],
        }


def population_stability_index(reference, current, bins: int = 10) -> float:
    """PSI between a reference and a current sample of a numeric feature."""
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    if len(reference) == 0 or len(current) == 0:
        raise ValueError("reference and current samples must be non-empty")

    edges = np.unique(np.quantile(reference, np.linspace(0.0, 1.0, bins + 1)))
    if len(edges) < 2:  # constant reference feature: spread a hair around it
        edges = np.array([edges[0] - 1e-6, edges[0] + 1e-6])
    edges[0], edges[-1] = -np.inf, np.inf

    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)

    eps = 1e-6
    ref_pct = np.maximum(ref_counts / len(reference), eps)
    cur_pct = np.maximum(cur_counts / len(current), eps)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def detect_drift(
    reference: dict[str, np.ndarray],
    current: dict[str, np.ndarray],
    psi_threshold: float = PSI_THRESHOLD,
    ks_pvalue_threshold: float = KS_PVALUE_THRESHOLD,
) -> DriftReport:
    """Compare feature distributions; a feature drifts when PSI exceeds the
    threshold or the KS test rejects the same-distribution hypothesis."""
    results = []
    for name in sorted(set(reference) & set(current)):
        psi = population_stability_index(reference[name], current[name])
        ks = stats.ks_2samp(reference[name], current[name])
        drifted = psi > psi_threshold or ks.pvalue < ks_pvalue_threshold
        results.append(
            FeatureDrift(
                feature=name,
                psi=round(psi, 6),
                ks_statistic=round(float(ks.statistic), 6),
                ks_pvalue=float(ks.pvalue),
                drifted=drifted,
            )
        )
    features = tuple(results)
    return DriftReport(features=features, drifted=any(f.drifted for f in features))
