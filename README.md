# FedDRLPD — Áp dụng kỹ thuật học tăng cường phát hiện tấn công nhiễm độc mô hình trong môi trường học liên kết

## Giới thiệu

**FedDRLPD** (Federated Deep Reinforcement Learning-based Poisoning Defense) là đồ án triển khai phương pháp phòng thủ chống tấn công nhiễm độc mô hình (model poisoning) trong Federated Learning, sử dụng **Deep Q-Network (DQN)** để tự động nhận diện và loại bỏ các client độc hại.

Trong môi trường Federated Learning (FL), nhiều client cùng huấn luyện một mô hình chung mà không cần chia sẻ dữ liệu riêng tư. Tuy nhiên, kẻ tấn công có thể giả mạo cập nhật mô hình (model update) để phá hoại mô hình toàn cục. FedDRLPD giải quyết vấn đề này bằng cách:

1. **Tính điểm độc hại (Malicious Score)** cho mỗi client dựa trên Mahalanobis Distance và lịch sử hành vi
2. **Sử dụng DQN** để học chính sách chọn client tối ưu — ưu tiên client lành tính, loại bỏ client độc hại
3. **Tổng hợp an toàn** chỉ từ các client được chọn

## Kiến trúc hệ thống

```
┌─────────────────────────────────────────────────────────────┐
│                        Server (train.py)                    │
│                                                             │
│  ┌──────────┐   ┌──────────────┐   ┌──────────────────┐    │
│  │  Global   │   │  DQN Agent   │   │  Aggregation     │    │
│  │  Model    │◄──│  (Selection) │──►│  (FedAvg trên    │    │
│  │  (CNN)    │   │              │   │   selected only) │    │
│  └──────────┘   └──────────────┘   └──────────────────┘    │
│       │              ▲                      ▲               │
│       │         state, reward          weight updates       │
│       ▼              │                      │               │
│  ┌─────────────────────────────────────────────────────┐    │
│  │           Client Manager (client.py)                │    │
│  │  ┌─────┐ ┌─────┐ ┌─────┐        ┌─────┐           │    │
│  │  │ C₁  │ │ C₂  │ │ C₃  │  ...   │ Cₙ  │           │    │
│  │  │ [B] │ │ [M] │ │ [B] │        │ [B] │           │    │
│  │  └─────┘ └─────┘ └─────┘        └─────┘           │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

## Cấu trúc thư mục

```
FedDRLPD_v2/
├── train.py              # Vòng lặp huấn luyện chính (FL + DQN)
├── client.py             # Client lành tính, client độc hại, ClientManager
├── server.py             # Server tổng hợp mô hình (aggregation)
├── model.py              # Kiến trúc CNN (3 Conv + 2 FC)
├── dataset.py            # Tải và phân chia dữ liệu (IID / Non-IID Dirichlet)
├── dqn_agent.py          # DQN Agent (Q-Network, Replay Buffer, reward)
├── malicious_score.py    # Mahalanobis Distance, Att_p, malicious score
├── requirement.txt       # Các thư viện Python cần thiết
├── trained_models/       # Checkpoint đã lưu
└── runs/                 # TensorBoard logs
```

## Các loại tấn công hỗ trợ

| Tấn công | Mô tả | Tham số CLI |
|----------|-------|-------------|
| **Label Flipping** | Đảo nhãn: `y' = (y+1) mod 10` | `--attack-type label_flipping` |
| **Backdoor** | Chèn trigger 3×3 pixel + nhãn mục tiêu = 0 | `--attack-type backdoor` |
| **Noise** | Thêm nhiễu Gaussian `N(0, 0.1²)` vào weight update | `--attack-type noise` |
| **Noise + Label Flipping** | Label flipping + thêm nhiễu vào weight delta | `--attack-type noise_label_flipping` |
| **Noise + Backdoor** | Backdoor + thêm nhiễu vào weight delta | `--attack-type noise_backdoor` |

## Cài đặt

### Yêu cầu

- Python ≥ 3.10
- CUDA (khuyến nghị, hỗ trợ chạy trên CPU)

### Cài đặt thư viện

```bash
pip install -r requirement.txt
```

## Cách chạy

### Huấn luyện FedDRLPD (mặc định)

```bash
# Chạy không giới hạn round (Ctrl+C để dừng)
python train.py --attack-type noise

# Chạy 200 rounds
python train.py --rounds 200 --attack-type noise

# Tùy chỉnh đầy đủ
python train.py \
    --rounds 500 \
    --attack-type label_flipping \
    --num-clients 100 \
    --malicious-ratio 0.3 \
    --local-epoch 2 \
    --batch-size 16
```

### Chạy FedAvg (baseline, không có DQN)

```bash
# FedAvg không tấn công — baseline sạch
python train.py --mode fedavg --malicious-ratio 0 --rounds 100

# FedAvg có tấn công — để thấy baseline bị poisoning
python train.py --mode fedavg --malicious-ratio 0.3 --attack-type noise --rounds 100
```

### Tiếp tục huấn luyện từ checkpoint

```bash
python train.py --checkpoint ./trained_models/<run_folder>/last_checkpoint.pt --rounds 100
```

### Theo dõi kết quả trên TensorBoard

```bash
tensorboard --logdir runs
```

Sau đó mở trình duyệt tại `http://localhost:6006` để theo dõi:
- `FL/global_accuracy` — Độ chính xác mô hình toàn cục
- `DQN/loss` — Loss của Q-Network
- `Defense/TPR` — Tỷ lệ phát hiện đúng client độc hại
- `Defense/FPR` — Tỷ lệ loại nhầm client lành tính

## Tham số dòng lệnh

| Tham số | Mặc định | Mô tả |
|---------|----------|-------|
| `--rounds`, `-r` | `0` | Số round FL (0 = chạy cho đến khi Ctrl+C) |
| `--checkpoint`, `-c` | `None` | Đường dẫn file checkpoint để tiếp tục huấn luyện |
| `--save-dir`, `-s` | `trained_models` | Thư mục lưu checkpoint |
| `--save-every` | `10` | Lưu checkpoint mỗi N round |
| `--attack-type` | `label_flipping` | Loại tấn công |
| `--num-clients`, `-n` | `100` | Số lượng client |
| `--malicious-ratio` | `0.3` | Tỷ lệ client độc hại (0.0 – 1.0) |
| `--local-epoch` | `2` | Số epoch local training mỗi round |
| `--batch-size` | `16` | Batch size cho local training |
| `--mode` | `feddrlpd` | Chế độ: `feddrlpd` (có DQN) hoặc `fedavg` (baseline) |

## Cấu hình mặc định trong code

| Tham số | Giá trị | Mô tả |
|---------|---------|-------|
| Dataset | CIFAR-10 | Bộ dữ liệu huấn luyện |
| Data distribution | Non-IID, Dirichlet α=0.5 | Phân phối dữ liệu giữa các client |
| PCA Components | 50 | Số chiều PCA giảm cho DQN state |
| DQN γ (discount) | 0.95 | Hệ số chiết khấu reward |
| DQN ε decay | 0.98 | Tốc độ giảm exploration |
| DQN ε min | 0.02 | Mức exploration tối thiểu |
| Replay Buffer | 300 | Kích thước bộ nhớ replay |
| DQN Warmup | 50 steps | Số bước trước khi bắt đầu train DQN |
| Target Network Update | Mỗi 10 steps | Tần suất đồng bộ target network |
| Reward weights (α, β, λ) | 0.2, 0.5, 0.3 | Trọng số: momentum, utility, penalty |

## Đọc log

Mỗi round, hệ thống in log per-client:

```
C04 [M]  rw=-0.1050 m=2068.522 q=-0.1459
C06 [B]* rw=+0.2294 m=88.470  q=+0.0312
```

| Ký hiệu | Ý nghĩa |
|----------|---------|
| `[M]` | Client độc hại (Malicious) |
| `[B]` | Client lành tính (Benign) |
| `*` | Được DQN chọn trong round này |
| `rw` | Reward nhận được |
| `m` | Malicious score (Att_p × MD) |
| `q` | Q-value từ DQN (cao = ưu tiên chọn) |

## Tài liệu tham khảo

- **Paper gốc**: FedDRLPD — *Federated Deep Reinforcement Learning-based Poisoning Defense* (Knowledge-Based Systems, Vol. 339, 2026)
