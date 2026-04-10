import numpy as np


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
                             round_idx):

    md_scores = compute_md_scores(weights_list)

    att_ip = compute_attacker_prob(client_history, round_idx)

    malicious_scores = att_ip * md_scores

    return malicious_scores, md_scores, att_ip


# =========================
# Update attacker history
# =========================

def update_attacker_history(md_scores,
                            client_history,
                            threshold="mean"):

    if threshold == "mean":
        th = np.mean(md_scores)
    elif threshold == "median":
        th = np.median(md_scores)
    else:
        th = threshold

    for i in range(len(md_scores)):
        if md_scores[i] > th:
            client_history[i] += 1

    return client_history