import torch, os

files = [
    "trained_models/best_checkpoint.pt",
    "trained_models/last_checkpoint_prev.pt",
    "trained_models/last_checkpoint.pt",
]

for f in files:
    if not os.path.isfile(f):
        print(f"NOT FOUND: {f}")
        continue
    size_mb = os.path.getsize(f) / 1024 / 1024
    try:
        ckpt = torch.load(f, map_location="cpu", weights_only=False)
        r = ckpt.get("round_idx", "?")
        a = ckpt.get("best_acc", 0)
        print(f"OK       | round={r:>4} | best_acc={a:.4f} | size={size_mb:.1f}MB | {f}")
    except Exception as e:
        print(f"CORRUPT  | size={size_mb:.1f}MB | {f} | {str(e)[:60]}")
