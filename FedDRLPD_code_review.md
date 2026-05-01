# Code Review: FedDRLPD Implementation
## Chẩn đoán tại sao DQN không loại được malicious clients

---

## TÓM TẮT CHẨN ĐOÁN (đọc trước)

Log cho thấy sau 200 rounds:
- **TPR ≈ 0.40–0.57** (phát hiện malicious), lý tưởng phải > 0.90
- **FPR ≈ 0.47–0.54** (loại nhầm benign), lý tưởng phải < 0.10
- **mal_in_sel ≈ 13–18/30** trong 30 selected — tức DQN chọn gần **50% attacker** = random!
- Q-values **không phân biệt** được malicious vs benign (q_mean tăng đều, không converge về pattern đúng)

**Kết luận**: DQN đang học **ngẫu nhiên**, không học được signal phân biệt. Có **5 bug cốt lõi** gây ra điều này.

---

## BUG #1 — NGHIÊM TRỌNG NHẤT: Reward signal không dạy DQN cách loại attacker

### Vấn đề
```python
# dqn_agent.py — compute_reward()
if global_acc > prev_acc:
    utility_i = float(np.exp(-dist_i) + global_acc)
else:
    utility_i = float(1.0 - np.exp(-dist_i))
```

Reward này **không phân biệt malicious vs benign** một cách rõ ràng.

- `dist_i` = khoảng cách giữa local absolute weight và global weight
- Malicious client sau label flipping vẫn có thể có `dist_i` thấp (đặc biệt early training)
- `penalty_i` từ `M(m_i)` bị scale về [0,1] rồi **bị làm nhỏ** do min-max normalization trong reward:

```python
# BUG: normalize m về [0,1] rồi mới dùng → malicious penalty gần = 0
if m_max - m_min > eps:
    m_i = (float(m[j]) - m_min) / (m_max - m_min + eps)
else:
    m_i = 0.0  # ← tất cả = 0 nếu m_scores gần nhau!

penalty_i = float(1.0 - np.exp(-max(0.0, m_i)))  # → penalty ≈ 0 luôn
```

**Hậu quả**: `lam * penalty_i ≈ 0` với `lam=0.3`. Toàn bộ malicious signal bị xóa bởi normalization.

### Fix
```python
# Dùng raw malicious score (đã bounded bởi M()), KHÔNG normalize trước
def compute_reward(self, prev_rewards, selected_ids, local_weights, global_weights,
                   global_acc, prev_acc, malicious_scores,
                   alpha=0.2, beta=0.5, lam=0.3):
    ...
    for j, cid in enumerate(selected_ids):
        # Utility: giữ như cũ
        D = normalized_distance(local_weights[j], global_weights[j])
        if global_acc > prev_acc:
            utility_i = np.exp(-D) + global_acc
        else:
            utility_i = 1.0 - np.exp(-D)

        # Penalty: dùng RAW malicious score — KHÔNG normalize
        m_raw = float(abs(malicious_scores[j]))  # ← raw value
        penalty_i = 1.0 - np.exp(-m_raw)         # bounded [0,1), tăng theo m

        reward_vec[cid] = alpha * reward_vec[cid] + beta * utility_i - lam * penalty_i
```

---

## BUG #2 — NGHIÊM TRỌNG: Attacker history được cập nhật sai logic

### Vấn đề trong `malicious_score.py`

```python
# malicious_score.py — update_attacker_history()
def update_attacker_history(md_scores, client_history, threshold="mean"):
    th = np.mean(md_scores)
    for i in range(len(md_scores)):
        if md_scores[i] > th:
            client_history[i] += 1  # ← local index i, không phải client_id!
    return client_history
```

**Bug cụ thể**:
- `client_history` trong `ClientManager` là list indexed by **global client_id**
- Nhưng `md_scores` chỉ có **`len(selected_ids)`** entries (số client được chọn round này)
- `update_attacker_history` dùng local index `i` để update `client_history[i]` → **sai hoàn toàn**

**Ví dụ**: selected_ids = [5, 23, 67]. md_scores = [0.1, 5.2, 0.3]. Mean = 1.87.
- client 23 (local index 1) có MD cao → nên update `client_history[23] += 1`
- Nhưng code update `client_history[1] += 1` → sai client!

### Fix trong `client.py`:
```python
# Sau khi compute_malicious_scores:
updated_history = update_attacker_history(md_scores, selected_history)
for local_idx, cid in enumerate(selected_ids):
    # updated_history[local_idx] là count của selected_history[local_idx]
    # nhưng selected_history đã là slice đúng → cần map ngược lại
    self.client_history[cid] = updated_history[local_idx]  # ← đây đúng
```

Thực ra bug nằm ở `update_attacker_history` trả về `client_history` (local slice) nhưng **không trả về mapping**. Sửa như sau:

```python
# malicious_score.py
def update_attacker_history(md_scores, client_history, threshold="mean"):
    if threshold == "mean":
        th = np.mean(md_scores) if len(md_scores) > 0 else 0.0
    elif threshold == "median":
        th = np.median(md_scores)
    else:
        th = float(threshold)

    updated = list(client_history)  # copy
    for i in range(len(md_scores)):
        if md_scores[i] > th:
            updated[i] += 1
    return updated
# ← Trả về list với cùng length, caller (client.py) map đúng bằng enumerate(selected_ids)
# client.py: self.client_history[cid] = updated_history[local_idx] — đây đúng rồi
```

Cần kiểm tra lại: `selected_history = [self.client_history[cid] for cid in selected_ids]` — đây là **copy giá trị**, không phải reference. Vậy flow đúng. Nhưng `update_attacker_history` nhận `selected_history` (local slice) và trả về cùng structure — **flow trong client.py đã đúng**. Bug thực sự là `threshold="mean"` trên MD của chỉ selected clients → rất noisy khi số selected nhỏ. Xem Bug #3.

---

## BUG #3 — TRUNG BÌNH: Threshold MD dùng mean của selected clients — quá noisy

### Vấn đề

```python
# malicious_score.py
def update_attacker_history(md_scores, client_history, threshold="mean"):
    th = np.mean(md_scores)  # mean của ~30 clients được chọn
```

Chỉ 30 clients được chọn mỗi round (50% của 100). Mean của MD trong nhóm nhỏ này **fluctuate mạnh** và không reliable. Nếu round đó toàn malicious → mean cao → không ai bị flag. Nếu round toàn benign → mean thấp → nhiều benign bị flag nhầm.

### Fix — Dùng absolute threshold hoặc chi-squared bound:

```python
from scipy.stats import chi2

def update_attacker_history(md_scores, client_history,
                            threshold_mode="chi2", alpha=0.05, n_features=20):
    if threshold_mode == "chi2":
        # Theo lý thuyết paper: MD^2 ~ chi2(d) với d = PCA_COMPONENTS
        th = np.sqrt(chi2.ppf(1 - alpha, df=n_features))
    elif threshold_mode == "median":
        th = np.median(md_scores) * 1.5  # robust: 1.5x median
    elif threshold_mode == "mean":
        th = np.mean(md_scores)
    else:
        th = float(threshold_mode)

    updated = list(client_history)
    for i in range(len(md_scores)):
        if md_scores[i] > th:
            updated[i] += 1
    return updated
```

**Gọi với**: `update_attacker_history(md_scores, selected_history, threshold_mode="chi2", n_features=PCA_COMPONENTS)`

---

## BUG #4 — TRUNG BÌNH: PCA refit mỗi round trên selected clients — basis không ổn định

### Vấn đề

```python
# train.py
weights_list = reduce_updates_with_pca(full_delta_list, PCA_COMPONENTS)
# full_delta_list chỉ có len(selected_ids) = 30 entries
```

PCA được fit lại mỗi round trên 30 clients được chọn. Điều này có 2 hậu quả:

1. **PCA basis thay đổi mỗi round** → PCA components round 5 và round 50 là **khác nhau hoàn toàn** → Q-value không comparable across rounds → DQN học không ổn định

2. **PCA fit trên selected clients** (không phải all clients) → basis bị bias bởi distribution của những ai được chọn (biased sample)

### Fix — Fit PCA 1 lần sau warm-up round, sau đó chỉ transform:

```python
# train.py — thêm global PCA object
pca_fitted = None
PCA_WARMUP_ROUNDS = 5  # fit PCA sau khi có đủ data

# Trong training loop:
if round_idx == PCA_WARMUP_ROUNDS:
    # Fit PCA trên TẤT CẢ clients từ memory bank
    all_deltas = [v for v in all_full_deltas.values() if v is not None]
    if len(all_deltas) >= PCA_COMPONENTS:
        pca_fitted = PCA(n_components=PCA_COMPONENTS, svd_solver="randomized", random_state=42)
        pca_fitted.fit(np.stack(all_deltas).astype(np.float32))

# Khi reduce:
if pca_fitted is not None:
    # Chỉ transform, không refit
    x = np.stack(full_delta_list).astype(np.float32)
    weights_list_arr = pca_fitted.transform(x)
    weights_list = [weights_list_arr[i] for i in range(len(full_delta_list))]
else:
    weights_list = reduce_updates_with_pca(full_delta_list, PCA_COMPONENTS)
```

**Cần thêm**: `all_full_deltas = {}` trong init loop, cập nhật mỗi round cho tất cả selected clients.

---

## BUG #5 — TRUNG BÌNH: DQN Bellman target sai — dùng full next_q thay vì max

### Vấn đề

```python
# dqn_agent.py — train()
with torch.no_grad():
    next_q_all = self.target_q_net(next_states)  # shape (B, N)
    target_q = rewards + self.gamma * next_q_all  # element-wise → sai!
```

**Standard Bellman**: `target = r + γ * max_{a'} Q(s', a')`

Nhưng code đang dùng `γ * Q(s', a_i)` cho **từng client i riêng lẻ**, không lấy max. Điều này có nghĩa là target Q của client i phụ thuộc vào Q của chính client i ở next state — không phải max của toàn bộ action space.

Với top-P selection, "max action" nên là **giá trị cao nhất trong top-P** của next state.

### Fix — Dùng max Q của top-P clients làm shared target:

```python
# dqn_agent.py — train()
with torch.no_grad():
    next_q_all = self.target_q_net(next_states)   # (B, N)
    # Max Q trong top-P của next state
    top_p_vals, _ = torch.topk(next_q_all, self.select_num, dim=1)  # (B, P)
    next_q_max = top_p_vals.mean(dim=1, keepdim=True)               # (B, 1) — mean of top-P
    # Broadcast: tất cả selected clients nhận cùng target signal
    target_q = rewards + self.gamma * next_q_max.expand_as(rewards)  # (B, N)

# Nhân thêm action_mask để chỉ tính loss trên selected clients
loss_matrix = nn.MSELoss(reduction="none")(q_values, target_q)
# Chỉ tính loss với clients được chọn (action_mask == 1)
masked_loss = (loss_matrix * action_masks).sum() / (action_masks.sum() + 1e-8)
loss = masked_loss
```

Đây là thay đổi **quan trọng nhất về RL correctness**: chỉ update Q-values của clients được chọn, không update clients không được chọn (vì không có feedback về họ).

---

## BUG #6 — NHỎ: State của clients không được chọn = zero vector

### Vấn đề

```python
# dqn_agent.py — build_state()
if cid not in w_norm_map:
    row.extend([0.0] * weight_dim)  # ← client không được chọn = toàn zeros
    row.append(0.0)
    row.append(0.0)
```

Clients không được chọn round này có state = `[0, 0, ..., 0, acc_g]`. Điều này khiến DQN **không thể phân biệt** client chưa có history với client benign với client malicious đang "ẩn".

Tốt hơn: dùng memory bank `all_weights`, `all_data_sizes`, `all_malicious_scores` đã được lưu từ các round trước.

```python
# train.py — đây đã làm đúng:
for i, cid in enumerate(client_ids):
    all_weights[cid]          = weights_list[i]
    all_data_sizes[cid]       = data_sizes[i]
    all_malicious_scores[cid] = malicious_scores[i]

next_state = dqn.build_state(
    all_weights,          # ← full N entries, clients không selected dùng giá trị cũ
    all_data_sizes,
    all_malicious_scores,
    global_acc,
    client_ids=list(range(NUM_CLIENTS)),
)
```

Code đã làm đúng phần này — **không phải bug**. Nhưng trong `build_state`, khi `cid not in w_norm_map` (vì `client_ids=list(range(NUM_CLIENTS))` → tất cả đều in map), điều này không xảy ra. **Tốt rồi, không cần sửa.**

---

## BẢNG TỔNG HỢP BUG + ƯU TIÊN FIX

| # | Bug | File | Dòng | Impact | Ưu tiên |
|---|-----|------|------|--------|---------|
| 1 | Normalize malicious score trước penalty → penalty ≈ 0 | dqn_agent.py | 400-405 | 🔴 Malicious signal bị xóa | **Fix ngay** |
| 2 | Threshold MD = mean(selected) quá noisy | malicious_score.py | 250 | 🔴 Att_p không reliable | **Fix ngay** |
| 3 | DQN loss không mask action → update cả unselected clients | dqn_agent.py | 469-470 | 🟡 Q-value nhiễu | Fix sau |
| 4 | PCA refit mỗi round → basis unstable | train.py | 299 | 🟡 State space không ổn định | Fix sau |
| 5 | Bellman target dùng per-client Q, không max | dqn_agent.py | 463-466 | 🟡 RL không optimal | Fix sau |
| 6 | select_ratio=0.5 quá lớn → 50 clients/round, dễ bị poison | train.py | 196 | 🟠 Giảm lợi thế defense | Tune |

---

## PHÂN TÍCH LOG — TẠI SAO DQN KHÔNG HỌC ĐƯỢC

Nhìn log từ round 170-200:

```
Round 173 | acc=0.7513 | TPR=0.50 | FPR=0.50 | mal_in_sel=15/30
Round 175 | acc=0.7018 | TPR=0.40 | FPR=0.54 | mal_in_sel=18/30
Round 182 | acc=0.3211 | TPR=0.40 | FPR=0.54 | mal_in_sel=18/30
```

**3 patterns đáng lo ngại**:

### Pattern 1: TPR/FPR oscillate, không converge
TPR dao động 0.40 → 0.57 → 0.40 không có trend. Nếu DQN học được, TPR phải **tăng dần** về 0.9+. → DQN đang predict ngẫu nhiên.

### Pattern 2: acc dao động mạnh (0.32 → 0.75 → 0.36)
Accuracy không ổn định cho thấy mỗi round các malicious clients lọt vào khác nhau ngẫu nhiên. → Selection không có pattern.

### Pattern 3: q_mean tăng dần (3.6 → 4.8) nhưng không correlate với TPR
Q-values tăng → network học được gì đó. Nhưng TPR không tăng → network học sai target. Nguyên nhân: **Bug #1** (penalty bị nullify) khiến network học để maximize utility (accuracy) mà không học exclude malicious.

---

## KẾ HOẠCH SỬA — THỨ TỰ ƯU TIÊN

### Bước 1 (Ngay bây giờ): Fix Bug #1 — Raw malicious penalty

**`dqn_agent.py`, function `compute_reward()`:**

```python
# XÓA đoạn normalize m_i này:
# m_min = float(np.min(m)) ...
# m_i = (float(m[j]) - m_min) / (m_max - m_min + eps)

# THAY bằng:
for j, cid in enumerate(selected_ids):
    lw = np.asarray(local_weights[j], dtype=np.float32).reshape(-1)
    gw = ref_vecs[j]
    pn = max(1, lw.size)
    diff = (lw - gw) / (np.abs(gw) + eps)
    dist_i = float(np.sum(np.abs(diff)) / pn)

    if global_acc > prev_acc:
        utility_i = float(np.exp(-dist_i) + global_acc)
    else:
        utility_i = float(1.0 - np.exp(-dist_i))

    # QUAN TRỌNG: Dùng raw malicious score, KHÔNG normalize
    m_raw = float(abs(m[j]))
    penalty_i = float(1.0 - np.exp(-m_raw))   # bounded [0, 1)

    reward_vec[cid] = float(alpha * reward_vec[cid] + beta * utility_i - lam * penalty_i)
```

### Bước 2 (Ngay bây giờ): Fix Bug #2 — Threshold MD bằng chi-squared

**`malicious_score.py`, function `update_attacker_history()`:**

```python
def update_attacker_history(md_scores, client_history,
                            threshold_mode="chi2_95", n_features=20):
    if threshold_mode == "chi2_95":
        from scipy.stats import chi2
        th = float(np.sqrt(chi2.ppf(0.95, df=n_features)))
    elif threshold_mode == "chi2_99":
        from scipy.stats import chi2
        th = float(np.sqrt(chi2.ppf(0.99, df=n_features)))
    elif threshold_mode == "median_x1.5":
        th = float(np.median(md_scores) * 1.5)
    else:
        th = float(np.mean(md_scores))

    updated = list(client_history)
    for i in range(len(md_scores)):
        if md_scores[i] > th:
            updated[i] += 1
    return updated
```

**Gọi trong `client.py`**:
```python
updated_history = update_attacker_history(
    md_scores, selected_history,
    threshold_mode="chi2_95",
    n_features=20  # = PCA_COMPONENTS
)
```

### Bước 3 (Sau): Fix Bug #3 — Masked DQN loss

**`dqn_agent.py`, function `train()`:**

```python
# Thay:
loss_matrix = nn.MSELoss(reduction="none")(q_values, target_q)
loss = loss_matrix.mean()

# Bằng:
loss_matrix = nn.MSELoss(reduction="none")(q_values, target_q)  # (B, N)
# Chỉ tính loss cho clients được chọn (action_mask = 1)
denom = action_masks.sum() + 1e-8
loss = (loss_matrix * action_masks).sum() / denom
```

### Bước 4 (Sau): Giảm select_ratio xuống 0.3

```python
# train.py
dqn = DQNAgent(
    ...
    select_ratio=0.3,   # 30 clients/round thay vì 50
    ...
)
```

Với 30% malicious và select 30%: nếu DQN exclude hết malicious thì chỉ select từ 70 benign → clean training.

### Bước 5 (Optional): Tăng lambda để boost malicious penalty

```python
# Thay đổi trong compute_reward() hoặc khi gọi:
reward = dqn.compute_reward(
    ...
    alpha=0.1,   # giảm history weight
    beta=0.4,    # giảm utility weight
    lam=0.5,     # tăng penalty weight
)
```

---

## EXPECTED RESULTS SAU KHI FIX

Sau khi apply Bug #1 + Bug #2 fix:

| Metric | Hiện tại | Expected |
|--------|----------|----------|
| TPR (rounds 150-200) | 0.40-0.57 | 0.75-0.90 |
| FPR | 0.47-0.54 | 0.10-0.20 |
| mal_in_sel / selected | 13-18/30 | 3-8/30 |
| acc (stable) | 0.35-0.75 oscillate | 0.70-0.80 stable |

---

## GHI CHÚ VỀ PAPER vs CODE

| Điểm | Paper nói | Code làm | Verdict |
|------|-----------|----------|---------|
| Reward | Không normalize m_i | Normalize về [0,1] | ❌ Code sai |
| Att_p threshold | Không nói rõ | Mean of selected | ⚠️ Cần chi2 |
| PCA | Không nói fit strategy | Refit mỗi round | ⚠️ Cần fix |
| DQN loss | Standard MSE | MSE nhưng không mask | ⚠️ Cần mask |
| State memory bank | Không nói rõ | Đã implement đúng | ✅ Tốt |
| Top-P selection | Đúng | Đúng | ✅ Tốt |
| Aggregation (FedAvg) | Đúng công thức | Đúng | ✅ Tốt |
| Attack implementation | 3 loại attack | Đầy đủ | ✅ Tốt |
