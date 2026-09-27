"""
evaluation.py
=============
Buoc 4: Danh gia mo hinh CNN tren Test Set hoac file CSV ngoai
- Ho tro 2 che do:
    1. Mac dinh: Load best_model.pth + images_22x22.npz (tap Test goc)
    2. File CSV ngoai: --csv <duong_dan_file.csv> [--label Class]
- Tinh day du: Accuracy, Precision, Recall, F1 (macro + weighted), ROC-AUC
- Ve Confusion Matrix dep bang seaborn
- Ve Per-class F1 bar chart
- Ve ROC Curve (One-vs-Rest) cho tung lop
- Luu tat ca bieu do, bao cao metrics_report.txt va predictions.csv vao thu muc evaluation/

Su dung:
    python evaluation.py
    python evaluation.py --csv your_test_data.csv
    python evaluation.py --csv your_test_data.csv --label Class --out evaluation_custom
"""

import argparse
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")   # Headless: khong can cua so GUI
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    roc_curve, auc,
)
from sklearn.preprocessing import label_binarize, MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib

import warnings
warnings.filterwarnings("ignore")

# ── Import tu project ────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(BASE_DIR))
from cnn_model import MalwareCNN, N_CLASSES, encode_label
from feature_to_image import get_semantic_ordering, IMAGE_SIZE, N_PIXELS

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG MAC DINH
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_NPZ    = BASE_DIR / "images_22x22.npz"
DEFAULT_MODEL  = BASE_DIR / "best_model.pth"
DEFAULT_CSV    = BASE_DIR / "471_extracted features.csv"
SCALER_PATH    = BASE_DIR / "scaler.joblib"
DEFAULT_OUT    = BASE_DIR / "evaluation"

# CICMalDroid2020 label names (goc 1-5 -> index 0-4)
CLASS_NAMES = ["Adware", "Banking", "SMS Malware", "Riskware", "Benign"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DARK_BG    = "#0d1117"
PANEL_BG   = "#161b22"
ACCENT     = "#00ff88"
TEXT_COLOR = "#e6edf3"
PALETTE    = ["#ff6b6b", "#ffa94d", "#69db7c", "#74c0fc", "#cc5de8"]

# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD MODEL
# ─────────────────────────────────────────────────────────────────────────────

def load_model(model_path: Path):
    if not model_path.exists():
        raise FileNotFoundError(f"Khong tim thay file checkpoint mo hinh: {model_path}")
    model = MalwareCNN(n_classes=N_CLASSES).to(DEVICE)
    ckpt  = torch.load(model_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    epoch    = ckpt.get("epoch", "?")
    best_acc = ckpt.get("best_val_acc", ckpt.get("best_acc", 0))
    print(f"[MODEL] Loaded epoch={epoch}  best_val_acc={best_acc*100:.2f}%  |  Device: {DEVICE}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# 2. LOAD DATA: TU NPZ HOAC CSV
# ─────────────────────────────────────────────────────────────────────────────

def load_test_data_npz(npz_path: Path):
    if not npz_path.exists():
        raise FileNotFoundError(f"Khong tim thay file data: {npz_path}")
    data    = np.load(npz_path)
    X_test  = torch.tensor(data["X_test"], dtype=torch.float32)
    y_test  = encode_label(data["y_test"]).numpy()   # 0-based int
    print(f"[DATA]  X_test={X_test.shape}  classes={np.unique(y_test)}")
    return X_test, y_test, None


def get_train_feature_names() -> list:
    """Doc danh sach 470 dac trung goc tu file CSV goc."""
    if not DEFAULT_CSV.exists():
        raise FileNotFoundError(f"Can file '{DEFAULT_CSV.name}' de lay danh sach 470 features chuan.")
    df_head = pd.read_csv(DEFAULT_CSV, nrows=0)
    features = [c for c in df_head.columns if c != "Class"]
    return features


def get_train_scaler() -> MinMaxScaler:
    """Load hoac tao moi MinMaxScaler fit tren 70% tap train goc."""
    if SCALER_PATH.exists():
        return joblib.load(SCALER_PATH)

    print("[SCALER] Dang tinh toan va khoi tao MinMaxScaler tu tap huan luyen goc...")
    df = pd.read_csv(DEFAULT_CSV)
    features = [c for c in df.columns if c != "Class"]
    X = df[features].values
    y = df["Class"].values

    X_temp, _, y_temp, _ = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    X_train, _, _, _ = train_test_split(
        X_temp, y_temp, test_size=0.15/0.85, random_state=42, stratify=y_temp
    )
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    joblib.dump(scaler, SCALER_PATH)
    print(f"[SCALER] Da luu scaler vao: {SCALER_PATH}")
    return scaler


def parse_labels(raw_labels) -> np.ndarray:
    """
    Chuyen doi nhan ve dang 0-4 tuong ung CLASS_NAMES:
    0: Adware, 1: Banking, 2: SMS Malware, 3: Riskware, 4: Benign
    """
    name_to_idx = {
        "adware": 0,
        "banking": 1,
        "banking malware": 1,
        "sms": 2,
        "sms malware": 2,
        "smsmalware": 2,
        "riskware": 3,
        "benign": 4,
        "goodware": 4,
    }
    encoded = []
    for val in raw_labels:
        # Kiem tra neu la so
        try:
            num = int(float(val))
            if 1 <= num <= 5:       # Goc 1-5
                encoded.append(num - 1)
                continue
            elif 0 <= num <= 4:     # Da la 0-4
                encoded.append(num)
                continue
        except (ValueError, TypeError):
            pass

        # Kiem tra neu la chuoi
        s = str(val).strip().lower()
        if s in name_to_idx:
            encoded.append(name_to_idx[s])
        else:
            encoded.append(-1)

    arr = np.array(encoded, dtype=int)
    if (arr == -1).any():
        print(f"[WARNING] Co {np.sum(arr == -1)} mau co nhan khong hop le!")
    return arr


def load_test_data_csv(csv_path, label_col: str = "Class"):
    """
    Doc file CSV bat ky (cung dataset), align cot, chuan hoa MinMax,
    tao anh 22x22 tensor va tra ve cung nhan (neu co).
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Khong tim thay file CSV: {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"[CSV INPUT] Doc file: {csv_path}")
    print(f"            So dong: {len(df)}  |  So cot: {len(df.columns)}")

    # 1. Tim va trich xuat cot nhan
    y_true = None
    target_col = None
    candidates = [label_col, "Class", "class", "Label", "label", "target"]
    for c in candidates:
        if c and c in df.columns:
            target_col = c
            break

    if target_col:
        print(f"[CSV INPUT] Phat hien cot nhan: '{target_col}'")
        raw_labels = df[target_col].values
        df_feat = df.drop(columns=[target_col])
        y_parsed = parse_labels(raw_labels)
        if not (y_parsed == -1).all():
            y_true = y_parsed
    else:
        print(f"[CSV INPUT] Khong phat hien cot nhan (chi thuc hien du doan, bo qua tinh metrics).")
        df_feat = df

    # 2. Align 470 features goc
    train_features = get_train_feature_names()
    overlap = [c for c in train_features if c in df_feat.columns]
    missing = [c for c in train_features if c not in df_feat.columns]

    print(f"[ALIGN] So features goc : {len(train_features)}")
    print(f"        Khop            : {len(overlap)} / {len(train_features)}")
    print(f"        Thieu (dien 0)  : {len(missing)}")
    if 0 < len(missing) <= 10:
        print(f"        Danh sach thieu : {missing}")

    df_aligned = pd.DataFrame(0.0, index=df.index, columns=train_features)
    for col in overlap:
        df_aligned[col] = pd.to_numeric(df_feat[col], errors="coerce").fillna(0.0)

    X_raw = df_aligned.values.astype(np.float32)

    # 3. Chuan hoa MinMax
    scaler = get_train_scaler()
    X_scaled = scaler.transform(X_raw)

    # 4. Sap xep theo thu tu Semantic
    ordered_indices, _ = get_semantic_ordering(train_features)
    X_reordered = X_scaled[:, ordered_indices]

    # 5. Padding zeros thanh 484 pixel va reshape thanh (N, 1, 22, 22)
    N = len(X_reordered)
    padded = np.zeros((N, N_PIXELS), dtype=np.float32)
    padded[:, :len(train_features)] = X_reordered
    X_img = padded.reshape(N, 1, IMAGE_SIZE[0], IMAGE_SIZE[1])
    X_tensor = torch.tensor(X_img, dtype=torch.float32)

    print(f"[PROCESS] Da chuyen doi thanh anh tensor: {X_tensor.shape}")
    return X_tensor, y_true, df


# ─────────────────────────────────────────────────────────────────────────────
# 3. PREDICT BATCH
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict(model, X_test, batch_size=128):
    all_probs, all_preds = [], []
    for i in range(0, len(X_test), batch_size):
        batch  = X_test[i:i+batch_size].to(DEVICE)
        logits = model(batch)
        probs  = torch.softmax(logits, dim=1)
        preds  = logits.argmax(dim=1)
        all_probs.append(probs.cpu().numpy())
        all_preds.append(preds.cpu().numpy())
    return np.concatenate(all_probs), np.concatenate(all_preds)


# ─────────────────────────────────────────────────────────────────────────────
# 4. METRICS REPORT
# ─────────────────────────────────────────────────────────────────────────────

def print_metrics(y_true, y_pred, y_prob):
    print("\n" + "="*65)
    print("  DANH GIA MO HINH TREN TAP DANH GIA (CICMalDroid2020)")
    print("="*65)
    print(f"  Accuracy          : {accuracy_score(y_true, y_pred)*100:.4f}%")
    print(f"  Macro Precision   : {precision_score(y_true, y_pred, average='macro',    zero_division=0):.4f}")
    print(f"  Macro Recall      : {recall_score(y_true,    y_pred, average='macro',    zero_division=0):.4f}")
    print(f"  Macro F1-Score    : {f1_score(y_true,        y_pred, average='macro',    zero_division=0):.4f}")
    print(f"  Weighted F1-Score : {f1_score(y_true,        y_pred, average='weighted', zero_division=0):.4f}")

    # Per-class ROC-AUC
    y_bin = label_binarize(y_true, classes=list(range(N_CLASSES)))
    aucs  = []
    valid_aucs = []
    for i in range(N_CLASSES):
        if len(np.unique(y_bin[:, i])) > 1:
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
            score = auc(fpr, tpr)
            aucs.append(score)
            valid_aucs.append(score)
        else:
            aucs.append(np.nan)

    macro_auc = np.mean(valid_aucs) if len(valid_aucs) > 0 else 0.0
    print(f"  Macro ROC-AUC     : {macro_auc:.4f}")
    print()
    print(classification_report(y_true, y_pred, labels=list(range(N_CLASSES)),
                                target_names=CLASS_NAMES, digits=4, zero_division=0))
    return aucs


# ─────────────────────────────────────────────────────────────────────────────
# 5. VISUALIZATION FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, out_dir: Path):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(N_CLASSES)))
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_pct = np.zeros_like(cm, dtype=float)
    np.divide(cm * 100.0, row_sums, out=cm_pct, where=row_sums != 0)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.patch.set_facecolor(DARK_BG)
    fig.suptitle("Confusion Matrix — MalwareCNN",
                 color=TEXT_COLOR, fontsize=14, fontweight="bold", y=1.01)

    for ax, data, fmt, title in [
        (axes[0], cm,     "d",    "Count"),
        (axes[1], cm_pct, ".1f",  "Normalized (%)"),
    ]:
        sns.heatmap(
            data,
            annot=True, fmt=fmt,
            xticklabels=CLASS_NAMES,
            yticklabels=CLASS_NAMES,
            cmap="YlOrRd",
            linewidths=0.5,
            linecolor="#333",
            ax=ax,
            annot_kws={"size": 11, "color": "black"},
        )
        ax.set_title(title, color=TEXT_COLOR, fontsize=12, fontweight="bold")
        ax.set_xlabel("Predicted", color=TEXT_COLOR, fontsize=11)
        ax.set_ylabel("True Label", color=TEXT_COLOR, fontsize=11)
        ax.set_facecolor(PANEL_BG)
        ax.tick_params(colors=TEXT_COLOR, labelsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")

    plt.tight_layout()
    out = out_dir / "confusion_matrix.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"[SAVE] {out}")


def plot_per_class_metrics(y_true, y_pred, out_dir: Path):
    prec   = precision_score(y_true, y_pred, labels=list(range(N_CLASSES)), average=None, zero_division=0)
    recall = recall_score(y_true,    y_pred, labels=list(range(N_CLASSES)), average=None, zero_division=0)
    f1     = f1_score(y_true,        y_pred, labels=list(range(N_CLASSES)), average=None, zero_division=0)

    x     = np.arange(N_CLASSES)
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(PANEL_BG)

    bars_p = ax.bar(x - width,  prec,   width, label="Precision", color="#74c0fc", alpha=0.9)
    bars_r = ax.bar(x,          recall, width, label="Recall",    color="#69db7c", alpha=0.9)
    bars_f = ax.bar(x + width,  f1,     width, label="F1-Score",  color="#ffa94d", alpha=0.9)

    for bars in [bars_p, bars_r, bars_f]:
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.3f}",
                        xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 4), textcoords="offset points",
                        ha="center", va="bottom",
                        color=TEXT_COLOR, fontsize=8.5)

    ax.set_title("Per-class Precision / Recall / F1-Score",
                 color=TEXT_COLOR, fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES, color=TEXT_COLOR, fontsize=10)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score", color=TEXT_COLOR)
    ax.tick_params(colors=TEXT_COLOR)
    ax.legend(facecolor="#21262d", labelcolor=TEXT_COLOR, fontsize=10)
    ax.axhline(0.8, color="#ff6b6b", ls="--", lw=1, alpha=0.6, label="0.8 target")

    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")
    ax.yaxis.grid(True, color="#30363d", alpha=0.5)
    ax.set_axisbelow(True)

    plt.tight_layout()
    out = out_dir / "per_class_metrics.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"[SAVE] {out}")


def plot_roc_curves(y_true, y_prob, out_dir: Path):
    y_bin = label_binarize(y_true, classes=list(range(N_CLASSES)))

    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(PANEL_BG)

    colors = PALETTE
    for i, (name, color) in enumerate(zip(CLASS_NAMES, colors)):
        if len(np.unique(y_bin[:, i])) > 1:
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
            roc_auc     = auc(fpr, tpr)
            label_str   = f"{name}  (AUC={roc_auc:.3f})"
        else:
            fpr, tpr = [0, 1], [0, 0]
            label_str = f"{name}  (N/A)"

        ax.plot(fpr, tpr, color=color, lw=2, label=label_str)

    ax.plot([0, 1], [0, 1], "w--", lw=1, alpha=0.4, label="Random")
    ax.set_title("ROC Curves — One-vs-Rest (OvR)",
                 color=TEXT_COLOR, fontsize=13, fontweight="bold")
    ax.set_xlabel("False Positive Rate", color=TEXT_COLOR, fontsize=11)
    ax.set_ylabel("True Positive Rate",  color=TEXT_COLOR, fontsize=11)
    ax.tick_params(colors=TEXT_COLOR)
    ax.legend(facecolor="#21262d", labelcolor=TEXT_COLOR, fontsize=10, loc="lower right")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")
    ax.yaxis.grid(True, color="#30363d", alpha=0.4)
    ax.xaxis.grid(True, color="#30363d", alpha=0.4)

    plt.tight_layout()
    out = out_dir / "roc_curves.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"[SAVE] {out}")


def plot_training_curve(out_dir: Path):
    hist_path = BASE_DIR / "training_history.npz"
    if not hist_path.exists():
        return
    h  = np.load(hist_path)
    ep = range(1, len(h["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor(DARK_BG)
    fig.suptitle("Training History — MalwareCNN", color=TEXT_COLOR,
                 fontsize=14, fontweight="bold")

    configs = [
        (axes[0], "train_loss", "test_loss",  "Loss (CrossEntropy)", "Loss"),
        (axes[1], "train_acc",  "test_acc",   "Accuracy",            "Accuracy (%)"),
    ]
    for ax, tr_key, va_key, title, ylabel in configs:
        tr = h[tr_key]
        va = h[va_key]
        if "acc" in tr_key:
            tr = [v * 100 for v in tr]
            va = [v * 100 for v in va]

        ax.plot(ep, tr, color=ACCENT,    lw=2, label="Train")
        ax.plot(ep, va, color="#ff6644", lw=2, ls="--", label="Validation")
        ax.set_title(title, color=TEXT_COLOR, fontsize=12, fontweight="bold")
        ax.set_xlabel("Epoch", color=TEXT_COLOR)
        ax.set_ylabel(ylabel, color=TEXT_COLOR)
        ax.set_facecolor(PANEL_BG)
        ax.tick_params(colors=TEXT_COLOR)
        ax.legend(facecolor="#21262d", labelcolor=TEXT_COLOR)
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")
        ax.yaxis.grid(True, color="#30363d", alpha=0.4)

    plt.tight_layout()
    out = out_dir / "training_history.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"[SAVE] {out}")


def export_predictions_csv(y_prob, y_pred, y_true, raw_df: pd.DataFrame, out_dir: Path):
    """Luu ket qua du doan chi tiet ra file predictions.csv."""
    res_df = pd.DataFrame()
    res_df["Sample_Index"]   = range(len(y_pred))
    res_df["Predicted_Class"] = y_pred
    res_df["Predicted_Name"]  = [CLASS_NAMES[p] for p in y_pred]
    res_df["Confidence_Pct"]  = [float(y_prob[i, y_pred[i]] * 100) for i in range(len(y_pred))]

    # Xac suat tung lop
    for i, cname in enumerate(CLASS_NAMES):
        res_df[f"Prob_{cname}"] = y_prob[:, i]

    if y_true is not None:
        res_df["True_Class"] = y_true
        res_df["True_Name"]  = [CLASS_NAMES[t] if 0 <= t < N_CLASSES else "Unknown" for t in y_true]
        res_df["Is_Correct"] = (y_pred == y_true)

    out_file = out_dir / "predictions.csv"
    res_df.to_csv(out_file, index=False, encoding="utf-8")
    print(f"[SAVE] {out_file} ({len(res_df)} rows)")


def format_malware_benign_summary(y_pred):
    """Tao bang tong hop Ma doc va Ma an toan gon gang, truc quan."""
    total = len(y_pred)
    counts = {i: 0 for i in range(N_CLASSES)}
    for p in y_pred:
        counts[p] += 1
    n_malware = sum(counts[i] for i in range(4))
    n_benign  = counts[4]

    type_map  = {0: "Ma doc", 1: "Ma doc", 2: "Ma doc", 3: "Ma doc", 4: "Ma an toan"}
    emoji_map = {0: "[!]",    1: "[!]",    2: "[!]",    3: "[!]",    4: "[OK]"}

    lines = [
        "=" * 60,
        "           KET QUA PHAN TICH AN TOAN & MA DOC",
        "=" * 60,
        f" Tong so mau phan tich : {total:,} mau",
        f" Tong so MA DOC        : {n_malware:,} mau ({n_malware/total*100:5.1f}%)",
        f" Tong so MA AN TOAN    : {n_benign:,} mau ({n_benign/total*100:5.1f}%)",
        "-" * 60,
        " CHI TIET TUNG LOAI DU DOAN:",
    ]
    for i in range(5):
        c_name  = CLASS_NAMES[i]
        c_type  = type_map[i]
        c_emoji = emoji_map[i]
        c_count = counts[i]
        c_pct   = (c_count / total) * 100
        lines.append(f"  {c_emoji} {c_name:<18} ({c_type}): {c_count:>5} mau ({c_pct:5.1f}%)")
    lines.append("=" * 60)
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# 6. MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="IE105 — Evaluation Script for MalwareCNN")
    parser.add_argument("--csv", type=str, default=None,
                        help="Duong dan den file CSV can danh gia.")
    parser.add_argument("--label", type=str, default="Class",
                        help="Ten cot nhan trong file CSV (mac dinh: 'Class').")
    parser.add_argument("--npz", type=str, default=None,
                        help="Duong dan den file .npz chua X_test, y_test.")
    parser.add_argument("--model", type=str, default=str(DEFAULT_MODEL),
                        help="Duong dan file checkpoint model .pth.")
    parser.add_argument("--out", type=str, default=None,
                        help="Thu muc luu ket qua danh gia.")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Batch size cho du doan.")
    args = parser.parse_args()

    # Thiet lap thu muc output (luon gom ve duy nhat thu muc evaluation/)
    out_dir = Path(args.out) if args.out else DEFAULT_OUT
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*65}")
    print(f"  IE105 — EVALUATION SCRIPT")
    print(f"  Device : {DEVICE}")
    print(f"  Output : {out_dir}")
    print(f"{'='*65}")

    model = load_model(Path(args.model))

    # Load du lieu
    raw_df = None
    if args.csv:
        X_test, y_true, raw_df = load_test_data_csv(Path(args.csv), label_col=args.label)
    else:
        npz_path = Path(args.npz) if args.npz else DEFAULT_NPZ
        X_test, y_true, raw_df = load_test_data_npz(npz_path)

    # Du doan
    print(f"\n[PREDICT] Dang du doan tren {len(X_test)} mau...")
    y_prob, y_pred = predict(model, X_test, batch_size=args.batch_size)

    # In bang tom tat ma doc va an toan
    summary_str = format_malware_benign_summary(y_pred)
    print("\n" + summary_str)

    # Xuat file predictions.csv
    export_predictions_csv(y_prob, y_pred, y_true, raw_df, out_dir)

    # Neu co nhan that -> Danh gia metrics & ve bieu do
    if y_true is not None:
        aucs = print_metrics(y_true, y_pred, y_prob)

        print("\n[PLOT] Ve Confusion Matrix...")
        plot_confusion_matrix(y_true, y_pred, out_dir)

        print("[PLOT] Ve Per-class Metrics...")
        plot_per_class_metrics(y_true, y_pred, out_dir)

        print("[PLOT] Ve ROC Curves...")
        plot_roc_curves(y_true, y_prob, out_dir)

        if not args.csv:
            print("[PLOT] Ve Training History...")
            plot_training_curve(out_dir)

        # Luu metrics ra file txt kem bang summary o dau
        report = classification_report(y_true, y_pred, labels=list(range(N_CLASSES)),
                                       target_names=CLASS_NAMES, digits=4, zero_division=0)
        valid_aucs = [a for a in aucs if not np.isnan(a)]
        macro_auc = np.mean(valid_aucs) if valid_aucs else 0.0

        metrics_txt = out_dir / "metrics_report.txt"
        with open(metrics_txt, "w", encoding="utf-8") as f:
            f.write(summary_str + "\n\n")
            f.write("IE105 — Chi tiet Metrics danh gia:\n")
            f.write("=" * 65 + "\n")
            f.write(f"Source Data       : {args.csv if args.csv else DEFAULT_NPZ.name}\n")
            f.write(f"Total Samples     : {len(y_true)}\n")
            f.write(f"Accuracy          : {accuracy_score(y_true, y_pred)*100:.4f}%\n")
            f.write(f"Macro Precision   : {precision_score(y_true, y_pred, average='macro',    zero_division=0):.4f}\n")
            f.write(f"Macro Recall      : {recall_score(y_true,    y_pred, average='macro',    zero_division=0):.4f}\n")
            f.write(f"Macro F1-Score    : {f1_score(y_true,        y_pred, average='macro',    zero_division=0):.4f}\n")
            f.write(f"Weighted F1-Score : {f1_score(y_true,        y_pred, average='weighted', zero_division=0):.4f}\n")
            f.write(f"Macro ROC-AUC     : {macro_auc:.4f}\n\n")
            f.write(report)
        print(f"[SAVE] {metrics_txt}")
    else:
        print("\n[NOTE] File CSV khong co nhan nen khong the tinh Accuracy hay ve Confusion Matrix.")
        print(f"       Ket qua du doan chi tiet da duoc luu vao: {out_dir / 'predictions.csv'}")

    print(f"\n[DONE] Hoan tat! Tat ca ket qua luu tai: {out_dir}")


if __name__ == "__main__":
    main()
