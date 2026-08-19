"""
OOD rejection layer for the hybrid method: reject query points that look
significantly more anomalous than the reference distribution before
applying (structure-aware) weighted conformal prediction to the survivors.

Uses a simple, standard KNN-distance anomaly score (mean distance to k
nearest reference training points) plus the same conformal p-value
construction the source paper uses (Eq. 3 in their supplementary S1.1/S2):
p(x) = (1 + #{calibration anomaly scores >= x's score}) / (n_cal + 1).
Points with p(x) < alpha_o are rejected as likely out-of-distribution.

This is a deliberately simple OOD score (vs. the source paper's dedicated
autoencoder), chosen to keep this a fast, targeted follow-up experiment
rather than a new sub-project -- documented as a scope simplification.
"""
import numpy as np
from sklearn.neighbors import NearestNeighbors


def knn_anomaly_score(X_ref: np.ndarray, X_query: np.ndarray, k: int = 10) -> np.ndarray:
    """Mean distance to k nearest neighbors in X_ref, for each point in X_query."""
    k = min(k, len(X_ref) - 1)
    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(X_ref)
    dists, _ = nn.kneighbors(X_query)
    return dists.mean(axis=1)


def conformal_pvalues(cal_scores: np.ndarray, test_scores: np.ndarray) -> np.ndarray:
    """Standard conformal p-value for each test score against calibration
    anomaly scores: high test score (very anomalous) -> low p-value."""
    n_cal = len(cal_scores)
    sorted_cal = np.sort(cal_scores)
    # count of cal_scores >= s, via searchsorted on sorted ascending array
    counts_geq = n_cal - np.searchsorted(sorted_cal, test_scores, side="left")
    return (1 + counts_geq) / (n_cal + 1)


def reject_mask(pvalues: np.ndarray, alpha_o: float) -> np.ndarray:
    """True = reject (flagged as OOD / non-conforming), False = keep."""
    return pvalues < alpha_o
