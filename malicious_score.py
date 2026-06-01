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

    n = len(weights_list)
    cov = np.dot(centered.T, centered) / max(1, n - 1)   # unbiased estimator (N-1)

    # regularization — ổn định số học cho trường hợp singular
    cov += np.eye(cov.shape[0]) * 1e-4

    return cov


# =========================
# Mahalanobis Distance
# =========================

def mahalanobis_distance(w, mu, cov_inv):

    diff = w - mu
    val = diff.T @ cov_inv @ diff
    # Clamp to avoid sqrt of negative due to numerical noise
    return np.sqrt(max(0.0, float(val)))


# =========================
# Compute all MD scores
# =========================

def compute_md_scores(weights_list):
    """Tính MD cho tất cả vectors trong weights_list.

    Luôn dùng full covariance + pseudo-inverse.
    Input phải là PCA-reduced vectors (≤100 dims).
    """
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
    cov = compute_covariance(W, mu)
    cov_inv = np.linalg.pinv(cov)   # pseudo-inverse: handle singular

    md_scores = []
    for w in W:
        md = mahalanobis_distance(w, mu, cov_inv)
        md_scores.append(md)

    return np.array(md_scores, dtype=np.float32)


# =========================
# Memory bank helpers
# =========================

def _build_memory_matrix_valid(all_client_updates):
    """Build matrix chỉ từ các rows KHÁC None trong memory bank.

    Returns
    -------
    W_valid : ndarray [n_valid, dim]
        Ma trận chỉ chứa clients đã có update thật.
    valid_ids : list[int]
        Mapping: valid_ids[local_idx] = global client_id.
    """
    if all_client_updates is None or len(all_client_updates) == 0:
        return np.zeros((0, 0), dtype=np.float32), []

    # Tìm chiều từ entry đầu tiên khác None
    dim = None
    for u in all_client_updates:
        if u is not None:
            dim = np.asarray(u, dtype=np.float32).reshape(-1).size
            break

    if dim is None:
        return np.zeros((0, 0), dtype=np.float32), []

    # Chỉ lấy rows có dữ liệu thật
    valid_rows = []
    valid_ids = []
    for i, u in enumerate(all_client_updates):
        if u is not None:
            arr = np.asarray(u, dtype=np.float32).reshape(-1)
            if arr.size == dim:
                valid_rows.append(arr)
            else:
                # pad/truncate nếu dim mismatch
                row = np.zeros(dim, dtype=np.float32)
                take = min(dim, arr.size)
                row[:take] = arr[:take]
                valid_rows.append(row)
            valid_ids.append(i)

    if len(valid_rows) == 0:
        return np.zeros((0, 0), dtype=np.float32), []

    return np.stack(valid_rows).astype(np.float32), valid_ids


def compute_md_scores_selected_from_memory(all_client_updates,
                                           selected_ids,
                                           fallback_selected_updates=None):
    """Tính MD cho selected_ids, μ/Cov tính trên toàn bộ valid memory bank.

    Pipeline: PCA vectors in memory → full cov → MD per selected client.
    Chỉ dùng rows khác None để tính μ, Cov (fix Bug B2).
    """
    if selected_ids is None or len(selected_ids) == 0:
        return np.array([], dtype=np.float32)

    selected_ids = [int(cid) for cid in selected_ids]

    W_valid, valid_ids = _build_memory_matrix_valid(all_client_updates)

    if W_valid.size == 0 or len(valid_ids) < 2:
        return np.zeros((len(selected_ids),), dtype=np.float32)

    n_valid, n_features = W_valid.shape

    # μ, Cov tính trên TẤT CẢ valid clients (không chỉ selected)
    mu = compute_mean(W_valid)
    cov = compute_covariance(W_valid, mu)
    cov_inv = np.linalg.pinv(cov)   # pseudo-inverse: handle singular

    # Build lookup: global_id → vector
    id_to_idx = {gid: idx for idx, gid in enumerate(valid_ids)}

    md_scores = []
    for cid in selected_ids:
        if cid in id_to_idx:
            w = W_valid[id_to_idx[cid]]
            md_scores.append(mahalanobis_distance(w, mu, cov_inv))
        else:
            # Client chưa có trong memory bank → MD = 0
            md_scores.append(0.0)

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
    # Pipeline PCA-reduced:
    # - μ, Cov tính trên toàn bộ valid entries trong memory bank (PCA space)
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