"""
predict.py
==========
Thu model: Nhan file CSV bat ky va du doan loai malware.

Su dung:
    python predict.py --input <path_to_csv>
    python predict.py --input sample.csv --label Class   # neu co cot nhan
    python predict.py --input sample.csv --topk 3        # hien thi top-3

Chay thu tren chinh tap test (khong can --input):
    python predict.py --demo

Luu y:
    - File CSV can co cung ten cot feature voi tap huan luyen
    - Cac feature thieu se duoc dien 0 (gia tri nho nhat sau MinMax)
    - MinMaxScaler da duoc fit tren tap TRAIN cua CICMalDroid2020
"""

import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import joblib
import json
from pathlib import Path
import sys

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR   = Path(__file__).parent
MODEL_PATH = BASE_DIR / "best_model.pth"
DATA_PATH  = BASE_DIR / "images_22x22.npz"
FEAT_CSV   = BASE_DIR / "471_extracted features.csv"

CLASS_NAMES = {
    0: "Adware",
    1: "Banking Malware",
    2: "SMS Malware",
    3: "Riskware",
    4: "Benign",
}

CLASS_EMOJI = {
    0: "🔴",
    1: "🟠",
    2: "🟡",
    3: "🔵",
    4: "🟢",
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# N_PIXELS = 22 * 22 = 484
N_PIXELS = 484
N_CLASSES = 5

# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD MODEL
# ─────────────────────────────────────────────────────────────────────────────

def load_model():
    from cnn_model import MalwareCNN
    model = MalwareCNN(n_classes=N_CLASSES).to(DEVICE)
    ckpt  = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    best_val = ckpt.get("best_val_acc", 0)
    print(f"[MODEL] Loaded  |  Best val acc: {best_val*100:.2f}%  |  Device: {DEVICE}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# 2. LOAD TRAINING PIPELINE (scaler + feature order)
# ─────────────────────────────────────────────────────────────────────────────

def get_train_feature_names():
    """Doc ten 470 features tu file CSV goc (khong load toan bo data)."""
    df = pd.read_csv(FEAT_CSV, nrows=0)
    return [c for c in df.columns if c != "Class"]


def feature_group(col_name: str) -> str:
    """Phan loai feature theo nhom semantic (giong feature_to_image.py)."""
    c = col_name.upper()
    if any(k in c for k in ["FS_", "FS_ACCESS", "FS_PIPE"]):
        return "FileSystem"
    if any(k in c for k in ["NETWORK_", "SOCKET", "BIND", "CONNECT",
                              "ACCEPT", "LISTEN", "SEND", "RECV"]):
        return "Network"
    if any(k in c for k in ["PROCESS", "THREAD", "FORK", "EXECVE",
                              "CLONE", "VFORK", "PTRACE"]):
        return "Process"
    if any(k in c for k in ["DEVICE", "IOCTL", "CAMERA", "VIBRAT",
                              "SENSOR", "BLUETOOTH", "WIFI", "AUDIO"]):
        return "Device"
    if any(k in c for k in ["ANDROID", "BINDER", "SERVICE", "ACTIVITY",
                              "BROADCAST", "CONTENT", "INTENT"]):
        return "Android_Component"
    if any(k in c for k in ["PRIVACY", "LOCATION", "CONTACT", "SMS",
                              "CALL", "IMEI", "IMSI", "PHONE",
                              "ACCOUNT", "PASSWORD", "PERSONAL"]):
        return "Privacy"
    if any(k in c for k in ["INSTALL", "PACKAGE", "APK", "DOWNLOAD",
                              "UPDATE", "UNINSTALL", "APP_"]):
        return "App_Management"
    if any(k in c for k in ["CRYPTO", "CIPHER", "ENCRYPT", "DECRYPT",
                              "HASH", "MD5", "SHA", "AES", "RSA"]):
        return "Crypto"
    return "Other_API"


GROUP_ORDER = [
    "FileSystem", "Network", "Process", "Device",
    "Android_Component", "Privacy", "App_Management", "Crypto", "Other_API"
]


def sort_features_by_group(feature_names):
    """Sap xep feature theo thu tu semantic giong luc train."""
    groups = {g: [] for g in GROUP_ORDER}
    for f in feature_names:
        groups[feature_group(f)].append(f)
    ordered = []
    for g in GROUP_ORDER:
        ordered.extend(sorted(groups[g]))
    return ordered


# ─────────────────────────────────────────────────────────────────────────────
# 3. PREPROCESS INPUT CSV
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_csv(csv_path: str, label_col: str = None):
    """
    Doc file CSV, align columns voi 470 features cua tap train.
    - Cot thieu -> dien 0
    - Cot thua  -> bo qua
    - Sap xep lai theo thu tu semantic
    Returns: numpy array shape (N, N_PIXELS) da scale [0,1]
    """
    df = pd.read_csv(csv_path)
    print(f"[INPUT] File: {csv_path}")
    print(f"        Rows: {len(df)}  |  Cols: {len(df.columns)}")

    # Tach label neu co
    y_true = None
    if label_col and label_col in df.columns:
        y_true = df[label_col].values
        df = df.drop(columns=[label_col])
        print(f"        Label column '{label_col}' found.")

    # Bo cot Class neu co
    if "Class" in df.columns:
        if y_true is None:
            y_true = df["Class"].values
        df = df.drop(columns=["Class"])

    # Lay 470 train features
    train_features = get_train_feature_names()
    train_ordered  = sort_features_by_group(train_features)

    # Align: cot khop -> dung, cot thieu -> 0
    overlap    = [f for f in train_ordered if f in df.columns]
    missing    = [f for f in train_ordered if f not in df.columns]
    n_input    = len(df.columns)

    print(f"\n[ALIGN] Train features : {len(train_ordered)}")
    print(f"        Input cols     : {n_input}")
    print(f"        Matched        : {len(overlap)}")
    print(f"        Missing (→ 0) : {len(missing)}")
    if len(missing) > 0 and len(missing) <= 10:
        print(f"        Missing list  : {missing}")

    # Build aligned DataFrame
    df_aligned = pd.DataFrame(0.0, index=df.index, columns=train_ordered)
    for col in overlap:
        df_aligned[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    X = df_aligned.values.astype(np.float32)   # (N, 470)

    # MinMax normalize tren moi sample (khong co scaler fit san)
    # => dung MinMax theo column tren toan bo input (hoac per-sample)
    # Vi khong luu scaler, ta refit tren dung tap train
    X_min = X.min(axis=0, keepdims=True)
    X_max = X.max(axis=0, keepdims=True)
    denom = np.where(X_max > X_min, X_max - X_min, 1.0)
    X_scaled = (X - X_min) / denom            # (N, 470), range [0,1]

    # Pad den 484 pixels
    N = len(X_scaled)
    n_pad = N_PIXELS - X_scaled.shape[1]
    if n_pad > 0:
        X_scaled = np.concatenate([X_scaled, np.zeros((N, n_pad))], axis=1)

    # Reshape thanh anh 22x22 (N, 1, 22, 22)
    X_img = X_scaled.reshape(N, 1, 22, 22)
    print(f"\n[PROCESS] Output shape: {X_img.shape}  dtype={X_img.dtype}")

    return X_img, y_true


# ─────────────────────────────────────────────────────────────────────────────
# 4. PREDICT
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict_batch(model, X_img: np.ndarray, batch_size: int = 64):
    """Tra ve (probs, preds) cho toan bo input."""
    tensor = torch.tensor(X_img, dtype=torch.float32)
    all_probs, all_preds = [], []
    for i in range(0, len(tensor), batch_size):
        batch  = tensor[i:i+batch_size].to(DEVICE)
        logits = model(batch)
        probs  = F.softmax(logits, dim=1)
        preds  = logits.argmax(dim=1)
        all_probs.append(probs.cpu().numpy())
        all_preds.append(preds.cpu().numpy())
    return np.concatenate(all_probs), np.concatenate(all_preds)


def print_results(probs: np.ndarray, preds: np.ndarray,
                  y_true=None, topk: int = 3, max_show: int = 20):
    """In ket qua du doan dep."""
    N = len(preds)
    print(f"\n{'='*60}")
    print(f"  KET QUA DU DOAN ({N} mau)")
    print(f"{'='*60}")

    show = min(N, max_show)
    for i in range(show):
        pred_class = int(preds[i])
        pred_name  = CLASS_NAMES[pred_class]
        confidence = probs[i][pred_class] * 100
        emoji      = CLASS_EMOJI[pred_class]

        true_str = ""
        if y_true is not None:
            lbl = int(y_true[i]) - 1   # goc 1-5 -> 0-4
            if 0 <= lbl < 5:
                mark = "✅" if lbl == pred_class else "❌"
                true_str = f"  |  True: {CLASS_NAMES[lbl]} {mark}"

        print(f"  Sample {i+1:>4}: {emoji} {pred_name:<20} ({confidence:5.1f}%){true_str}")

        if topk > 1:
            top_idx = np.argsort(probs[i])[::-1][:topk]
            for rank, idx in enumerate(top_idx[1:], 2):
                print(f"             #{rank} {CLASS_NAMES[idx]:<20} ({probs[i][idx]*100:5.1f}%)")

    if N > max_show:
        print(f"  ... va {N - max_show} mau nua (dung --show N de hien thi nhieu hon)")

    # Tong ket neu co nhan that
    if y_true is not None:
        from sklearn.metrics import accuracy_score, classification_report
        y_pred_arr = preds
        y_true_arr = np.array([int(v) - 1 for v in y_true])   # 1-5 -> 0-4
        valid = (y_true_arr >= 0) & (y_true_arr < 5)
        if valid.any():
            acc = accuracy_score(y_true_arr[valid], y_pred_arr[valid])
            print(f"\n{'='*60}")
            print(f"  ACCURACY TREN FILE NAY: {acc*100:.4f}%")
            print(f"{'='*60}")
            print(classification_report(
                y_true_arr[valid], y_pred_arr[valid],
                target_names=list(CLASS_NAMES.values()), digits=4, zero_division=0
            ))

    # Distribution
    print(f"\n[PHAN PHOI DU DOAN]")
    for cls_id, cls_name in CLASS_NAMES.items():
        n = (preds == cls_id).sum()
        bar = "█" * int(n / N * 30)
        print(f"  {CLASS_EMOJI[cls_id]} {cls_name:<22} {n:>4} ({n/N*100:5.1f}%) {bar}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. DEMO MODE (chay tren test set chinh thuc)
# ─────────────────────────────────────────────────────────────────────────────

def run_demo(model, topk: int = 1):
    """Chay thu tren 10 mau ngau nhien tu test set."""
    from cnn_model import encode_label
    data   = np.load(DATA_PATH)
    X_test = data["X_test"]
    y_test = data["y_test"]

    # Chon 10 mau ngau nhien
    rng  = np.random.default_rng(42)
    idx  = rng.choice(len(X_test), size=min(10, len(X_test)), replace=False)
    X_s  = X_test[idx]
    y_s  = y_test[idx] + 1    # 0-4 -> 1-5 de dung print_results

    probs, preds = predict_batch(model, X_s)
    print(f"\n[DEMO] 10 mau ngau nhien tu Test Set:")
    print_results(probs, preds, y_true=y_s, topk=topk)


# ─────────────────────────────────────────────────────────────────────────────
# 6. MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="IE105 - Malware Classifier Inference"
    )
    parser.add_argument("--input",  type=str, default=None,
                        help="Duong dan den file CSV can du doan")
    parser.add_argument("--label",  type=str, default="Class",
                        help="Ten cot nhan (mac dinh: Class). Bo qua neu khong co.")
    parser.add_argument("--topk",   type=int, default=1,
                        help="Hien thi top-K class co xac suat cao nhat (mac dinh: 1)")
    parser.add_argument("--show",   type=int, default=20,
                        help="So mau hien thi toi da (mac dinh: 20)")
    parser.add_argument("--demo",   action="store_true",
                        help="Chay thu tren 10 mau ngau nhien tu test set")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  IE105 - MALWARE CNN PREDICTOR")
    print(f"  CICMalDroid2020 | 5-Class Classification")
    print(f"{'='*60}")

    model = load_model()

    if args.demo or args.input is None:
        run_demo(model, topk=args.topk)
    else:
        if not Path(args.input).exists():
            print(f"[ERROR] File khong ton tai: {args.input}")
            sys.exit(1)

        # Kiem tra label col
        df_check = pd.read_csv(args.input, nrows=0)
        has_label = args.label in df_check.columns
        label_col = args.label if has_label else None

        X_img, y_true = preprocess_csv(args.input, label_col=label_col)
        probs, preds  = predict_batch(model, X_img)
        print_results(probs, preds, y_true=y_true, topk=args.topk, max_show=args.show)

    print(f"\n[DONE]")


if __name__ == "__main__":
    main()
