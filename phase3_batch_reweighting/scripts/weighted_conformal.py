"""
Standalone split conformal prediction with five weighting schemes, built
independently of the Phase 1/2 ConformalSCAnnotator framework (which has no
support for custom reweighting -- this is Phase 3's actual methodological
contribution, so it needs full control over the weighting math).

Implements Tibshirani et al. (2019) weighted conformal prediction, using the
APS (adaptive prediction sets) non-conformity score for consistency with
Phases 1-2. Five weighting schemes are compared:

  1. naive                        -- uniform weights (standard split
                                      conformal, ignores shift)
  2. domain_clf_naive             -- generic weighted CP baseline in its
                                      ORIGINAL unfixed form: one domain
                                      classifier, fixed C=1.0, scored
                                      in-sample, no weight truncation. Kept
                                      only for before/after comparison.
  3. domain_clf                   -- generic weighted CP: a black-box binary
                                      domain classifier (calibration=0 vs
                                      query=1) estimates a per-cell density
                                      ratio, blind to any known structure.
                                      STRENGTHENED (2026-08-08) with K-fold
                                      cross-fitting, CV-tuned regularization,
                                      and weight truncation, so instability
                                      seen in earlier results reflects a
                                      genuine property of structure-blind
                                      density-ratio estimation, not a fixable
                                      implementation weakness.
  4. structure_aware_centroid     -- batch-structure-aware HEURISTIC: one
                                      weight per source batch, via inverse
                                      centroid distance to the query. No
                                      theorem behind it -- kept only as a
                                      baseline against method 5.
  5. structure_aware_propensity   -- batch-structure-aware, theoretically
                                      grounded via generalized propensity
                                      scores (Imbens 2000): a genuine
                                      density-ratio estimator derived from
                                      an explicitly stated mixture
                                      assumption, not an arbitrary distance
                                      metric. See its docstring for the
                                      full derivation.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold


def aps_scores_calibration(probs: np.ndarray, true_idx: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Randomized APS non-conformity score (Romano et al. 2020): cumulative
    probability of classes ranked strictly above the true class, plus a
    uniform-random fraction of the true class's own probability mass.
    Deterministic APS (cumsum only, no randomization) is known to be
    systematically over-conservative -- this was the bug in the first pass."""
    n = len(true_idx)
    order = np.argsort(-probs, axis=1)  # descending
    ranks = np.argsort(order, axis=1)  # rank of each class per row
    sorted_probs = np.take_along_axis(probs, order, axis=1)
    cumsum = np.cumsum(sorted_probs, axis=1)

    true_rank = ranks[np.arange(n), true_idx]
    prev_cumsum = np.zeros(n)
    has_prev = true_rank > 0
    prev_cumsum[has_prev] = cumsum[np.arange(n)[has_prev], true_rank[has_prev] - 1]
    true_prob = sorted_probs[np.arange(n), true_rank]

    U = rng.uniform(size=n)
    return prev_cumsum + U * true_prob


def aps_prediction_set(probs_row: np.ndarray, tau: float, rng: np.random.Generator) -> list:
    """Randomized APS prediction set: classes ranked above the cutoff are
    always included; the boundary class is included with a probability
    proportional to how much of its mass is needed to reach tau exactly.
    tau=np.inf includes everything (maximally uninformative -- severe shift)."""
    order = np.argsort(-probs_row)
    sorted_probs = probs_row[order]
    cumsum = np.cumsum(sorted_probs)
    if tau == np.inf:
        return order.tolist()

    idx = int(np.searchsorted(cumsum, tau))
    if idx >= len(order):
        return order.tolist()

    prev_cum = cumsum[idx - 1] if idx > 0 else 0.0
    denom = sorted_probs[idx]
    include_prob = 1.0 if denom <= 0 else np.clip((tau - prev_cum) / denom, 0.0, 1.0)

    if rng.uniform() < include_prob:
        return order[: idx + 1].tolist()
    return order[:idx].tolist()


def weighted_quantile_threshold(
    cal_scores: np.ndarray, cal_weights: np.ndarray, test_weight: float, alpha: float
) -> float:
    """Tibshirani et al. (2019) weighted conformal quantile for a single
    test point with its own weight. Returns np.inf if even all calibration
    mass doesn't reach 1-alpha (signals severe shift for that point)."""
    order = np.argsort(cal_scores)
    sorted_scores = cal_scores[order]
    sorted_weights = cal_weights[order]
    total = sorted_weights.sum() + test_weight
    normalized = sorted_weights / total
    cumsum = np.cumsum(normalized)
    idx = np.searchsorted(cumsum, 1 - alpha)
    if idx >= len(sorted_scores):
        return np.inf
    return sorted_scores[idx]


def naive_weights(n_cal: int, n_test: int):
    """Uniform weights -- standard (unweighted) split conformal prediction."""
    return np.ones(n_cal), np.ones(n_test)


def domain_classifier_weights_naive(X_cal: np.ndarray, X_test: np.ndarray):
    """Generic weighted CP, ORIGINAL (unfixed) version: black-box
    density-ratio estimate via a single domain classifier (calibration=0,
    query=1) fit on ALL the data and then evaluated IN-SAMPLE on those same
    points, with an arbitrary fixed C=1.0 and no weight truncation.
    w(x) = p(query|x) / p(cal|x).

    Kept only as a reference point to quantify how much of the "unstable
    domain_clf" result in earlier Phase 3 runs was a fixable implementation
    weakness vs. an inherent property of black-box density-ratio weighting.
    See domain_classifier_weights() below for the fixed version used in all
    reported comparisons.
    """
    X_all = np.vstack([X_cal, X_test])
    y_all = np.concatenate([np.zeros(len(X_cal)), np.ones(len(X_test))])
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(X_all, y_all)
    p_cal = clf.predict_proba(X_cal)[:, 1]
    p_test = clf.predict_proba(X_test)[:, 1]
    eps = 1e-6
    w_cal = np.clip(p_cal, eps, 1 - eps) / np.clip(1 - p_cal, eps, 1 - eps)
    w_test = np.clip(p_test, eps, 1 - eps) / np.clip(1 - p_test, eps, 1 - eps)
    return w_cal, w_test


def domain_classifier_weights(
    X_cal: np.ndarray, X_test: np.ndarray, n_folds: int = 5, clip_percentiles=(1, 99), seed: int = 0
):
    """Generic weighted CP baseline, STRENGTHENED so the comparison against
    the structure-aware methods is fair. Three known failure modes of the
    naive version (domain_classifier_weights_naive above) are fixed:

    1. In-sample propensity bias: fitting one classifier on ALL points and
       then scoring those SAME points systematically pushes training-set
       predictions toward 0/1 (the classifier partially memorizes which
       points it was trained on), which is exactly what produces the
       extreme, unstable density-ratio weights seen in earlier Phase 3
       results. Fixed via K-fold CROSS-FITTING (Chernozhukov et al. 2018,
       standard in doubly-robust/propensity-score estimation): each point's
       probability is predicted by a classifier that never saw that point
       during training.
    2. Arbitrary, untuned regularization: C=1.0 was a fixed guess. Fixed via
       LogisticRegressionCV, which selects C by internal cross-validation
       within each cross-fitting training fold.
    3. Unbounded weights: raw odds ratios from a near-separable domain
       classifier can be enormous for a handful of points, dominating the
       weighted quantile. Fixed via WEIGHT TRUNCATION at the [1st, 99th]
       percentile of the pooled weight distribution (Cole & Hernan 2008),
       a standard stabilization technique for inverse-probability weights.

    Still a legitimately "generic, structure-blind" baseline -- none of
    these fixes give it access to batch identity, so it remains a fair
    point of comparison against the structure-aware methods, just no longer
    a strawman.
    """
    X_all = np.vstack([X_cal, X_test])
    y_all = np.concatenate([np.zeros(len(X_cal)), np.ones(len(X_test))])
    n = len(X_all)

    p_all = np.zeros(n)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for train_idx, held_idx in skf.split(X_all, y_all):
        clf = LogisticRegressionCV(
            Cs=[0.01, 0.1, 1.0, 10.0], cv=3, max_iter=2000, random_state=seed
        )
        clf.fit(X_all[train_idx], y_all[train_idx])
        p_all[held_idx] = clf.predict_proba(X_all[held_idx])[:, 1]

    eps = 1e-6
    p_all = np.clip(p_all, eps, 1 - eps)
    w_all = p_all / (1 - p_all)

    lo, hi = np.percentile(w_all, clip_percentiles)
    w_all = np.clip(w_all, lo, hi)

    n_cal = len(X_cal)
    return w_all[:n_cal], w_all[n_cal:]


def structure_aware_centroid_weights(
    X_cal: np.ndarray, batch_ids_cal: np.ndarray, X_test: np.ndarray, batch_ids_test: np.ndarray
):
    """Batch-structure-aware weighting via inverse centroid distance.

    HONESTLY LABELED AS A HEURISTIC: this has no theorem behind it.
    Tibshirani et al.'s (2019) weighted-conformal coverage guarantee
    requires w(x) to be (an estimate of) the true covariate-shift density
    ratio dQ/dP(x); raw inverse Euclidean distance between batch centroids
    does not estimate that quantity in any formal sense. It is kept here
    purely as a fast, interpretable baseline to compare against the
    theoretically-grounded estimator below
    (structure_aware_propensity_weights), not as a claimed-valid method.
    """
    test_centroid = X_test.mean(axis=0)  # query is (usually) a single batch

    unique_cal_batches = np.unique(batch_ids_cal)
    batch_weight = {}
    for b in unique_cal_batches:
        mask = batch_ids_cal == b
        centroid_b = X_cal[mask].mean(axis=0)
        dist = np.linalg.norm(centroid_b - test_centroid)
        batch_weight[b] = 1.0 / (dist + 1e-6)

    w_cal = np.array([batch_weight[b] for b in batch_ids_cal])
    # query cells all share the query batch's own weight relative to itself
    # (distance to its own centroid = 0) -> use the mean calibration weight
    # as a neutral reference scale for the test-side weight.
    w_test = np.full(len(X_test), np.mean(list(batch_weight.values())))
    return w_cal, w_test


def structure_aware_propensity_weights(
    X_cal: np.ndarray, batch_ids_cal: np.ndarray, X_test: np.ndarray, batch_ids_test: np.ndarray
):
    """Batch-structure-aware weighting via GENERALIZED PROPENSITY SCORES
    (Imbens, 2000) -- a real density-ratio estimator, unlike the centroid
    heuristic above, derived from an explicitly stated assumption rather
    than an arbitrary distance metric.

    NOVELTY CAVEAT (added 2026-08-08, after a literature check prompted by
    an honest NCS-worthiness re-assessment): this is NOT a new theoretical
    contribution. "Mixture of known reference groups" covariate shift for
    weighted conformal prediction is an active area with prior art,
    including Bhattacharyya & Barber, "Group-Weighted Conformal Prediction"
    (arXiv:2401.17452, Jan 2024), which formalizes the group-mixture-shift
    setting this function targets, and adjacent 2024-2025 multi-source /
    domain-shift-aware conformal prediction work that also uses a
    domain classifier to estimate per-point group-membership probabilities
    and combines them with an estimated target-mixture to form weights --
    essentially the same construction derived independently here without a
    prior search of this literature. The one respect in which this function
    is NOT a direct restatement of GWCP: GWCP assumes group membership is
    KNOWN for every point (including test points) and the target mixture
    proportions are GIVEN; this function instead ESTIMATES both (via a
    classifier trained on reference batches, and the query's mean predicted
    membership) because in the single-cell setting a held-out dataset/study
    is not literally one of the reference batches and its batch-mixture
    composition is not known in advance. That is a genuine (if incremental)
    extension of the same idea to a harder, more realistic estimation
    setting -- but the core theoretical framework is established prior art,
    not a new method, and should be cited as such rather than presented as
    a novel theoretical contribution. See PHASE3_SUMMARY.md's "Novelty
    reassessment" section for the full honest accounting.

    Derivation. Let P = sum_k pi_k P_k be the calibration-generating
    mixture over K known reference batches (pi_k = empirical batch
    proportion). Fit ONE multinomial classifier g(x) -> P(batch=k | x)
    using ONLY the labeled reference-batch calibration data (a well-posed
    K-class problem -- no query data needed for training, which is what
    avoids domain_classifier_weights' instability: that method collapses
    all batch heterogeneity into a single binary decision boundary, which
    is easy to overfit; this one is forced to make a genuine K-way
    discrimination using each batch's own cluster structure).

    By Bayes' rule, P(x | batch=k) is proportional to g(x)_k / pi_k.
    ASSUME (explicitly, not hidden) the query distribution Q is a mixture
    of the reference batches' distributions, Q = sum_j lambda_j P_j, with
    mixture weights lambda_j estimated as the query set's average
    predicted membership in batch j: lambda_j = mean_{x in query} g(x)_j.
    Then the covariate-shift density ratio has a closed form:

        w(x) = dQ/dP(x) = sum_j (lambda_j / pi_j) * g(x)_j

    applied identically to calibration AND test points -- no per-point
    batch-of-origin bookkeeping needed, unlike the centroid heuristic.
    If the query truly resembles one specific reference batch closely,
    lambda concentrates on that batch and w(x) reduces to (approximately)
    g(x)_j / pi_j for that batch alone -- the multi-batch generalization
    of the single-batch case worked out in the accompanying derivation.
    """
    unique_batches = np.unique(batch_ids_cal)
    K = len(unique_batches)
    batch_to_class = {b: i for i, b in enumerate(unique_batches)}
    y_cal_class = np.array([batch_to_class[b] for b in batch_ids_cal])

    pi = np.array([np.mean(y_cal_class == k) for k in range(K)])
    pi = np.clip(pi, 1e-6, None)

    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(X_cal, y_cal_class)
    # predict_proba columns follow clf.classes_ order, which for integer
    # labels 0..K-1 fit via LogisticRegression is sorted ascending -> matches
    # batch_to_class's index order directly.
    g_cal = clf.predict_proba(X_cal)  # (n_cal, K)
    g_test = clf.predict_proba(X_test)  # (n_test, K)

    lam = g_test.mean(axis=0)  # (K,) estimated query mixture weights
    lam = lam / lam.sum()

    coef = lam / pi  # (K,)
    w_cal = g_cal @ coef
    w_test = g_test @ coef
    return w_cal, w_test


def evaluate_coverage(
    cal_scores: np.ndarray,
    cal_weights: np.ndarray,
    test_probs: np.ndarray,
    test_weights: np.ndarray,
    test_true_idx: np.ndarray,
    alpha: float,
    rng: np.random.Generator,
):
    """Build a per-test-point weighted prediction set and report empirical
    coverage + average set size."""
    n_test = test_probs.shape[0]
    covered = np.zeros(n_test, dtype=bool)
    set_sizes = np.zeros(n_test, dtype=int)

    for i in range(n_test):
        tau = weighted_quantile_threshold(cal_scores, cal_weights, test_weights[i], alpha)
        pred_set = aps_prediction_set(test_probs[i], tau, rng)
        set_sizes[i] = len(pred_set)
        covered[i] = test_true_idx[i] in pred_set

    return float(covered.mean()), float(set_sizes.mean())


def evaluate_coverage_percell(
    cal_scores: np.ndarray,
    cal_weights: np.ndarray,
    test_probs: np.ndarray,
    test_weights: np.ndarray,
    test_true_idx: np.ndarray,
    alpha: float,
    rng: np.random.Generator,
):
    """Identical computation to evaluate_coverage, but returns the raw
    per-cell covered/set_size arrays rather than only their means. Added for
    peer-review response: bootstrap confidence intervals on coverage require
    the per-cell indicator, not just the aggregate rate. Purely additive --
    evaluate_coverage itself is unchanged so no other script's behavior
    changes."""
    n_test = test_probs.shape[0]
    covered = np.zeros(n_test, dtype=bool)
    set_sizes = np.zeros(n_test, dtype=int)

    for i in range(n_test):
        tau = weighted_quantile_threshold(cal_scores, cal_weights, test_weights[i], alpha)
        pred_set = aps_prediction_set(test_probs[i], tau, rng)
        set_sizes[i] = len(pred_set)
        covered[i] = test_true_idx[i] in pred_set

    return covered, set_sizes
