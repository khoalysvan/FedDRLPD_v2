"""
Vẽ Confusion Matrix từ file final_metrics.csv do train.py tạo ra.

Cách dùng:
    python plot_confusion_matrix.py --csv trained_models/<run_name>/final_metrics.csv
    python plot_confusion_matrix.py --csv trained_models/<run_name>/final_metrics.csv --out results/
"""

import csv
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt


# =========================
# Đọc CSV
# =========================

def load_metrics(csv_path):
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "round":        int(row["round"]),
                "attack_type":  row["attack_type"],
                "tp":           int(row["tp"]),
                "tn":           int(row["tn"]),
                "fp":           int(row["fp"]),
                "fn":           int(row["fn"]),
                "tpr":          float(row["tpr"]),
                "fpr":          float(row["fpr"]),
                "accuracy":     float(row["accuracy"]),
                "loss":         float(row["loss"]),
                "num_malicious": int(row["num_malicious"]),
                "num_benign":   int(row["num_benign"]),
            })
    return rows


# =========================
# Vẽ một Confusion Matrix
# =========================

def plot_cm(row, out_dir, show=False):
    tp = row["tp"]
    tn = row["tn"]
    fp = row["fp"]
    fn = row["fn"]
    attack = row["attack_type"]
    r = row["round"]

    cm = np.array([[tp, fn],
                   [fp, tn]])

    labels = [[f"TP\n{tp}", f"FN\n{fn}"],
              [f"FP\n{fp}", f"TN\n{tn}"]]

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Excluded\n(Detected)", "Selected\n(Missed)"], fontsize=11)
    ax.set_yticklabels(["Malicious", "Benign"], fontsize=11)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)

    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            ax.text(j, i, labels[i][j],
                    ha="center", va="center", fontsize=14,
                    color="white" if cm[i, j] > thresh else "black",
                    fontweight="bold")

    tpr_v = tp / max(1, tp + fn)
    fpr_v = fp / max(1, fp + tn)
    prec  = tp / max(1, tp + fp)
    f1    = 2 * prec * tpr_v / max(1e-9, prec + tpr_v)
    acc_d = (tp + tn) / max(1, tp + tn + fp + fn)

    title = (
        f"Confusion Matrix — {attack}  (round {r})\n"
        f"TPR={tpr_v:.3f}  FPR={fpr_v:.3f}  Precision={prec:.3f}  "
        f"F1={f1:.3f}  DetAcc={acc_d:.3f}"
    )
    ax.set_title(title, fontsize=9)
    fig.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"cm_{attack}_r{r}.png")
    fig.savefig(out_path, dpi=150)
    print(f"[PLOT] Saved → {out_path}")

    if show:
        plt.show()
    plt.close(fig)

    return out_path


# =========================
# Main
# =========================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True,
                        help="Path to final_metrics.csv")
    parser.add_argument("--out", default=None,
                        help="Output directory (default: same as CSV dir)")
    parser.add_argument("--show", action="store_true",
                        help="Show plot interactively")
    args = parser.parse_args()

    out_dir = args.out or os.path.dirname(os.path.abspath(args.csv))

    rows = load_metrics(args.csv)
    print(f"[INFO] Loaded {len(rows)} row(s) from {args.csv}")

    for row in rows:
        print(f"  Round {row['round']} | {row['attack_type']} | "
              f"TP={row['tp']} TN={row['tn']} FP={row['fp']} FN={row['fn']} | "
              f"TPR={row['tpr']} FPR={row['fpr']} Acc={row['accuracy']}")
        plot_cm(row, out_dir, show=args.show)

    print("\nDone.")
