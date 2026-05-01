import numpy as np
from statistics import NormalDist


# =========================
# Flatten weights
# =========================

def flatten_weights(weight_dict):

    return np.concatenate([
        v.detach().cpu().numpy().flatten()
        for v in weight_dict.values()
    ])


# =========================
# Compute mean (μ)
# =========================

def compute_mean(weights_list):

    return np.mean(weights_list, axis=0)


# =========================
# Compute covariance (Cov)
# =========================

def compute_covariance(weights_list, mu):

    centered = weights_list - mu

    cov = np.dot(centered.T, centered) / len(weights_list)

    # regularization — tăng lên 1e-4 để ổn định hơn với delta vector cao chiều
    cov += np.eye(cov.shape[0]) * 1e-4

    return cov


# =========================
# Mahalanobis Distance
# =========================

def mahalanobis_distance(w, mu, cov_inv):

    diff = w - mu
    return np.sqrt(diff.T @ cov_inv @ diff)


def diagonal_mahalanobis_distance(w, mu, var, eps=1e-8):

    diff = w - mu
    # Mahalanobis với hiệp phương sai đường chéo: sqrt(sum((diff^2)/(var+eps)))
    return np.sqrt(np.sum((diff * diff) / (var + eps)))


# =========================
# Compute all MD scores
# =========================

def compute_md_scores(weights_list):

    if len(weights_list) == 0:
        return np.array([], dtype=np.float32)

    W = np.array(weights_list, dtype=np.float32)

    if W.ndim != 2:
        raise ValueError("weights_list must be a 2D array-like [num_clients, num_features].")

    n_samples, n_features = W.shape

    # Edge case: chỉ 1 client — không thể đo độ lệch khỏi nhóm
    if n_samples == 1:
        return np.array([0.0], dtype=np.float32)

    mu = compute_mean(W)

    # Full covariance có độ phức tạp O(d^2); chỉ dùng khi d nhỏ.
    use_full_cov = n_features <= 2048 and n_features <= max(64, 4 * n_samples)

    md_scores = []

    if use_full_cov:
        cov = compute_covariance(W, mu)
        cov_inv = np.linalg.inv(cov)

        for w in W:
            md = mahalanobis_distance(w, mu, cov_inv)
            md_scores.append(md)
    else:
        # Ổn định bộ nhớ cho vector update rất lớn.
        var = np.var(W, axis=0, dtype=np.float32)
        for w in W:
            md = diagonal_mahalanobis_distance(w, mu, var)
            md_scores.append(md)

    return np.array(md_scores, dtype=np.float32)


def _normalize_vector_dim(vec, target_dim):

    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    if arr.size == target_dim:
        return arr

    out = np.zeros((target_dim,), dtype=np.float32)
    take = min(target_dim, arr.size)
    out[:take] = arr[:take]
    return out


def _build_memory_matrix(all_client_updates, fallback_updates=None):

    if all_client_updates is None or len(all_client_updates) == 0:
        if fallback_updates is None or len(fallback_updates) == 0:
            return np.zeros((0, 0), dtype=np.float32)
        W = np.asarray(fallback_updates, dtype=np.float32)
        if W.ndim != 2:
            raise ValueError("fallback_updates must be a 2D array-like [num_clients, num_features].")
        return W

    # infer dimension from memory bank first, fallback to selected updates if needed
    dim = None
    for u in all_client_updates:
        if u is not None:
            dim = np.asarray(u, dtype=np.float32).reshape(-1).size
            break

    if dim is None and fallback_updates is not None and len(fallback_updates) > 0:
        dim = np.asarray(fallback_updates[0], dtype=np.float32).reshape(-1).size

    if dim is None:
        return np.zeros((0, 0), dtype=np.float32)

    mat = np.zeros((len(all_client_updates), dim), dtype=np.float32)
    for i, u in enumerate(all_client_updates):
        if u is None:
            continue
        mat[i] = _normalize_vector_dim(u, dim)

    return mat


def compute_md_scores_selected_from_memory(all_client_updates,
                                           selected_ids,
                                           fallback_selected_updates=None):

    if selected_ids is None or len(selected_ids) == 0:
        return np.array([], dtype=np.float32)

    selected_ids = [int(cid) for cid in selected_ids]
    W_all = _build_memory_matrix(all_client_updates, fallback_updates=fallback_selected_updates)

    if W_all.size == 0:
        return np.zeros((len(selected_ids),), dtype=np.float32)

    if W_all.ndim != 2:
        raise ValueError("all_client_updates memory matrix must be 2D.")

    n_samples, n_features = W_all.shape
    if n_samples == 1:
        return np.zeros((len(selected_ids),), dtype=np.float32)

    # μ = mean(all_client_updates)
    mu = compute_mean(W_all)

    # Cov = covariance(all_client_updates) with numerical stability
    use_full_cov = n_features <= 2048 and n_features <= max(64, 4 * n_samples)

    md_scores = []
    if use_full_cov:
        cov = compute_covariance(W_all, mu)  # includes +I*1e-4
        cov_inv = np.linalg.inv(cov)

        for cid in selected_ids:
            if cid < 0 or cid >= n_samples:
                md_scores.append(0.0)
                continue
            md_scores.append(mahalanobis_distance(W_all[cid], mu, cov_inv))
    else:
        # fallback đường chéo cho chiều lớn
        var = np.var(W_all, axis=0, dtype=np.float32)
        for cid in selected_ids:
            if cid < 0 or cid >= n_samples:
                md_scores.append(0.0)
                continue
            md_scores.append(diagonal_mahalanobis_distance(W_all[cid], mu, var))

    return np.asarray(md_scores, dtype=np.float32)


# =========================
# Attacker Probability (Att_ip)
# =========================

def compute_attacker_prob(client_history, round_idx):

    """
    client_history[i] = số lần client i bị nghi là attacker
    """

    att_ip = []

    for count in client_history:
        val = 1 + count / max(1, round_idx)
        att_ip.append(val)

    return np.array(att_ip)


# =========================
# Final malicious score
# =========================

def compute_malicious_scores(weights_list,
                             client_history,
                             round_idx,
                             all_client_updates=None,
                             selected_ids=None):
    # New FedDRLPD interpretation:
    # - μ, Cov tính trên toàn bộ memory bank all_client_updates
    # - MD chỉ tính cho selected_ids round hiện tại
    # Backward-compatible fallback: nếu không có memory bank thì dùng cách cũ.
    if all_client_updates is not None and selected_ids is not None:
        md_scores = compute_md_scores_selected_from_memory(
            all_client_updates,
            selected_ids,
            fallback_selected_updates=weights_list,
        )
    else:
        md_scores = compute_md_scores(weights_list)

    att_ip = compute_attacker_prob(client_history, round_idx)

    malicious_scores = att_ip * md_scores

    return malicious_scores, md_scores, att_ip


# =========================
# Update attacker history
# =========================

def _chi2_sqrt_bound(prob, df):

    # Wilson-Hilferty approximation for chi-square inverse CDF.
    df = max(1.0, float(df))
    prob = float(np.clip(prob, 1e-6, 1.0 - 1e-6))
    z = NormalDist().inv_cdf(prob)
    term = 1.0 - (2.0 / (9.0 * df)) + z * np.sqrt(2.0 / (9.0 * df))
    chi2_q = df * (term ** 3)
    return float(np.sqrt(max(0.0, chi2_q)))

def update_attacker_history(md_scores,
                            client_history,
                            threshold="mean",
                            threshold_mode=None,
                            n_features=None,
                            absolute_threshold=None):

    scores = np.asarray(md_scores, dtype=np.float32).reshape(-1)
    updated = list(client_history)

    if scores.size == 0:
        return updated

    mode = threshold_mode
    if mode is None and isinstance(threshold, str):
        mode = threshold.lower()

    if mode in ("mean", None):
        th = float(np.mean(scores))
    elif mode == "median":
        th = float(np.median(scores))
    elif mode in ("median_x1.5", "median_1.5"):
        th = float(np.median(scores) * 1.5)
    elif mode == "chi2_95":
        df = int(n_features) if n_features is not None else max(1, scores.size)
        th = _chi2_sqrt_bound(0.95, df)
    elif mode == "chi2_99":
        df = int(n_features) if n_features is not None else max(1, scores.size)
        th = _chi2_sqrt_bound(0.99, df)
    elif mode == "absolute":
        if absolute_threshold is not None:
            th = float(absolute_threshold)
        elif not isinstance(threshold, str):
            th = float(threshold)
        else:
            th = float(np.mean(scores))
    else:
        # Backward-compatible numeric threshold.
        if isinstance(threshold, str):
            th = float(np.mean(scores))
        else:
            th = float(threshold)

    for i in range(len(scores)):
        if float(scores[i]) > th:
            updated[i] += 1

    return updated