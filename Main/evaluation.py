"""
evaluation.py
=============
Đồ án IE105 - Khai phá dữ liệu & Ứng dụng
Đánh giá toàn diện hiệu năng của mô hình MalwareCNN trên tập Test hoặc file dữ liệu thực tế.

Mục đích & Chức năng:
1. Hỗ trợ 2 chế độ đánh giá linh hoạt:
   - Mặc định: Nạp checkpoint tốt nhất (`best_model.pth`) và kiểm thử trực tiếp trên tập Test
     chính thức từ file `images_22x22.npz`.
   - File CSV bên ngoài: Nhận file dữ liệu mới (`--csv file.csv`), tự động chuyển đổi sang ảnh 22x22
     bằng `scaler.joblib` và tiến hành chấm điểm.
2. Đo lường đầy đủ các chỉ số định lượng:
   - Accuracy, Macro Precision, Macro Recall, Macro F1-Score, Weighted F1-Score.
   - ROC-AUC theo chiến lược One-vs-Rest (OvR) cho từng lớp và tính trung bình Macro ROC-AUC.
3. Xuất các đồ thị đánh giá trực quan (lưu tại thư mục `evaluation/`):
   - `confusion_matrix.png`    : Ma trận nhầm lẫn dạng số lượng và tỷ lệ phần trăm chuẩn hóa.
   - `per_class_metrics.png`    : Biểu đồ cột so sánh Precision, Recall và F1 của từng dòng mã độc.
   - `roc_curves.png`           : Đường cong ROC kèm chỉ số diện tích dưới đường cong (AUC) từng lớp.
   - `training_history.png`     : Đồ thị diễn biến Loss và Accuracy qua các epoch huấn luyện.
   - `metrics_report.txt`       : Báo cáo tổng hợp số liệu dạng văn bản để đưa vào bài báo cáo.
   - `predictions.csv`          : Bảng chi tiết kết quả dự đoán kèm xác suất của từng mẫu.
"""

import argparse
from pathlib import Path
import sys
import warnings
warnings.filterwarnings("ignore")

# Cấu hình UTF-8 cho console Windows để in tiếng Việt có dấu không bị lỗi charmap
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import joblib
import matplotlib
matplotlib.use("Agg")  # Chạy chế độ headless để không phụ thuộc vào giao diện đồ họa GUI
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, label_binarize
import torch
import torch.nn as nn

# Nạp các module phụ trợ từ thư mục dự án
BASE_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(BASE_DIR))
from cnn_model import MalwareCNN, N_CLASSES, encode_label
from feature_to_image import IMAGE_SIZE, N_PIXELS, get_semantic_ordering

# ==============================================================================
# 1. CẤU HÌNH MẶC ĐỊNH & THIẾT KẾ ĐỒ HỌA
# ==============================================================================

DEFAULT_NPZ   = BASE_DIR / "images_22x22.npz"
DEFAULT_MODEL = BASE_DIR / "best_model.pth"
DEFAULT_CSV   = BASE_DIR / "471_extracted features.csv"
SCALER_PATH   = BASE_DIR / "scaler.joblib"
DEFAULT_OUT   = BASE_DIR / "evaluation"

CLASS_NAMES = ["Adware", "Banking", "SMS Malware", "Riskware", "Benign"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Bảng màu giao diện tối (Dark mode) giúp đồ thị hiển thị sắc nét và hiện đại
DARK_BG    = "#0d1117"
PANEL_BG   = "#161b22"
ACCENT     = "#00ff88"
TEXT_COLOR = "#e6edf3"
PALETTE    = ["#ff6b6b", "#ffa94d", "#69db7c", "#74c0fc", "#cc5de8"]


# ==============================================================================
# 2. NẠP MÔ HÌNH VÀ DỮ LIỆU ĐÁNH GIÁ
# ==============================================================================

def load_model(model_path: Path):
    """Nạp trọng số mô hình từ file checkpoint đã lưu."""
    if not model_path.exists():
        raise FileNotFoundError(f"Không tìm thấy checkpoint mô hình tại: {model_path}")
    model = MalwareCNN(n_classes=N_CLASSES).to(DEVICE)
    ckpt  = torch.load(model_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    epoch    = ckpt.get("epoch", "?")
    best_acc = ckpt.get("best_val_acc", ckpt.get("best_acc", 0))
    print(f"[MÔ HÌNH] Đã nạp thành công mô hình từ epoch={epoch} (Best Val Acc: {best_acc*100:.2f}%)")
    return model


def load_test_data_npz(npz_path: Path):
    """Nạp tập dữ liệu kiểm thử gốc đã được xử lý và lưu trong file .npz."""
    if not npz_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu: {npz_path}")
    data    = np.load(npz_path)
    X_test  = torch.tensor(data["X_test"], dtype=torch.float32)
    y_test  = encode_label(data["y_test"]).numpy()
    print(f"[DỮ LIỆU] Nạp từ tệp NPZ: X_test shape = {X_test.shape}, Số lớp = {len(np.unique(y_test))}")
    return X_test, y_test, None


def get_train_feature_names() -> list:
    """Lấy danh sách 470 tên đặc trưng chuẩn từ file dữ liệu gốc."""
    if not DEFAULT_CSV.exists():
        raise FileNotFoundError(f"Cần có file '{DEFAULT_CSV.name}' để lấy danh sách cột chuẩn.")
    df_head = pd.read_csv(DEFAULT_CSV, nrows=0)
    return [c for c in df_head.columns if c != "Class"]


def get_train_scaler() -> MinMaxScaler:
    """Nạp scaler đã lưu hoặc tạo mới nếu chưa có."""
    if SCALER_PATH.exists():
        return joblib.load(SCALER_PATH)

    print("[BỘ CHUẨN HÓA] Không thấy scaler.joblib, đang fit lại trên 70% tập train gốc...")
    df = pd.read_csv(DEFAULT_CSV)
    features = [c for c in df.columns if c != "Class"]
    X = df[features].values
    y = df["Class"].values

    X_temp, _, y_temp, _ = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
    X_train, _, _, _ = train_test_split(X_temp, y_temp, test_size=0.15/0.85, random_state=42, stratify=y_temp)

    scaler = MinMaxScaler()
    scaler.fit(X_train)
    joblib.dump(scaler, SCALER_PATH)
    print(f"[BỘ CHUẨN HÓA] Đã lưu scaler mới tại: {SCALER_PATH}")
    return scaler


def parse_labels(raw_labels) -> np.ndarray:
    """Ánh xạ nhãn dạng văn bản hoặc số nguyên về chỉ số lớp 0-4."""
    name_to_idx = {
        "adware": 0, "banking": 1, "banking malware": 1,
        "sms": 2, "sms malware": 2, "smsmalware": 2,
        "riskware": 3, "benign": 4, "goodware": 4,
    }
    encoded = []
    for val in raw_labels:
        try:
            num = int(float(val))
            if 1 <= num <= 5:
                encoded.append(num - 1)
                continue
            elif 0 <= num <= 4:
                encoded.append(num)
                continue
        except (ValueError, TypeError):
            pass

        s = str(val).strip().lower()
        if s in name_to_idx:
            encoded.append(name_to_idx[s])
        else:
            encoded.append(-1)

    arr = np.array(encoded, dtype=int)
    if (arr == -1).any():
        print(f"[CẢNH BÁO] Có {np.sum(arr == -1)} mẫu mang nhãn không hợp lệ.")
    return arr


def load_test_data_csv(csv_path: Path, label_col: str = "Class"):
    """Đọc và tiền xử lý file CSV mới thành định dạng tensor ảnh 22x22 tương thích với CNN."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file CSV: {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"[DỮ LIỆU CSV] Đọc file: {csv_path.name} | {len(df):,} dòng x {len(df.columns)} cột")

    # Tách cột nhãn thực tế nếu có
    y_true = None
    target_col = None
    candidates = [label_col, "Class", "class", "Label", "label", "target"]
    for c in candidates:
        if c and c in df.columns:
            target_col = c
            break

    if target_col:
        print(f"             Phát hiện cột nhãn: '{target_col}'")
        raw_labels = df[target_col].values
        df_feat = df.drop(columns=[target_col])
        y_parsed = parse_labels(raw_labels)
        if not (y_parsed == -1).all():
            y_true = y_parsed
    else:
        print(f"             Không có cột nhãn (chỉ dự đoán, bỏ qua tính điểm metrics)")
        df_feat = df

    # Căn chỉnh 470 đặc trưng chuẩn
    train_features = get_train_feature_names()
    overlap = [c for c in train_features if c in df_feat.columns]
    missing = [c for c in train_features if c not in df_feat.columns]

    print(f"[CĂN CHỈNH ĐẶC TRƯNG] Khớp {len(overlap)}/470 cột | Bù 0 cho {len(missing)} cột thiếu")

    df_aligned = pd.DataFrame(0.0, index=df.index, columns=train_features)
    for col in overlap:
        df_aligned[col] = pd.to_numeric(df_feat[col], errors="coerce").fillna(0.0)

    X_raw = df_aligned.values.astype(np.float32)

    # Chuẩn hóa giá trị
    scaler = get_train_scaler()
    X_scaled = scaler.transform(X_raw)

    # Sắp xếp theo nhóm ngữ nghĩa
    ordered_indices, _ = get_semantic_ordering(train_features)
    X_reordered = X_scaled[:, ordered_indices]

    # Zero-padding 14 điểm ảnh cuối và reshape về ảnh 22x22
    N = len(X_reordered)
    padded = np.zeros((N, N_PIXELS), dtype=np.float32)
    padded[:, :len(train_features)] = X_reordered
    X_img = padded.reshape(N, 1, IMAGE_SIZE[0], IMAGE_SIZE[1])
    X_tensor = torch.tensor(X_img, dtype=torch.float32)

    return X_tensor, y_true, df


# ==============================================================================
# 3. DỰ ĐOÁN & ĐO LƯỜNG CHỈ SỐ (INFERENCE & METRICS)
# ==============================================================================

@torch.no_grad()
def predict(model, X_test, batch_size: int = 128):
    """Dự đoán xác suất và nhãn phân loại theo từng mini-batch."""
    all_probs, all_preds = [], []
    for i in range(0, len(X_test), batch_size):
        batch  = X_test[i:i+batch_size].to(DEVICE)
        logits = model(batch)
        probs  = torch.softmax(logits, dim=1)
        preds  = logits.argmax(dim=1)
        all_probs.append(probs.cpu().numpy())
        all_preds.append(preds.cpu().numpy())
    return np.concatenate(all_probs), np.concatenate(all_preds)


def print_metrics(y_true, y_pred, y_prob):
    """In bảng đánh giá các chỉ số định lượng của mô hình trên tập kiểm định."""
    print("\n" + "=" * 65)
    print("  KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH MALWARE CNN (CICMalDroid2020)")
    print("=" * 65)
    print(f"  Độ chính xác tổng thể (Accuracy) : {accuracy_score(y_true, y_pred)*100:.4f}%")
    print(f"  Độ chính xác vĩ mô (Macro Precision) : {precision_score(y_true, y_pred, average='macro', zero_division=0):.4f}")
    print(f"  Độ thu hồi vĩ mô (Macro Recall)       : {recall_score(y_true, y_pred, average='macro', zero_division=0):.4f}")
    print(f"  Điểm F1 vĩ mô (Macro F1-Score)        : {f1_score(y_true, y_pred, average='macro', zero_division=0):.4f}")
    print(f"  Điểm F1 gia quyền (Weighted F1-Score)  : {f1_score(y_true, y_pred, average='weighted', zero_division=0):.4f}")

    # Tính chỉ số diện tích dưới đường cong ROC (ROC-AUC) theo One-vs-Rest
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
    print(f"  Diện tích ROC-AUC vĩ mô (Macro ROC-AUC): {macro_auc:.4f}")
    print("\n[BÁO CÁO PHÂN LOẠI CHI TIẾT TỪNG LỚP]")
    print(classification_report(y_true, y_pred, labels=list(range(N_CLASSES)),
                                target_names=CLASS_NAMES, digits=4, zero_division=0))
    return aucs


# ==============================================================================
# 4. TRỰC QUAN HÓA KẾT QUẢ ĐÁNH GIÁ (PLOTTING)
# ==============================================================================

def plot_confusion_matrix(y_true, y_pred, out_dir: Path):
    """Vẽ ma trận nhầm lẫn (Confusion Matrix) dạng đếm số lượng và chuẩn hóa tỷ lệ %."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(N_CLASSES)))
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_pct = np.zeros_like(cm, dtype=float)
    np.divide(cm * 100.0, row_sums, out=cm_pct, where=row_sums != 0)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.patch.set_facecolor(DARK_BG)
    fig.suptitle("Ma trận Nhầm lẫn (Confusion Matrix) — MalwareCNN",
                 color=TEXT_COLOR, fontsize=14, fontweight="bold", y=1.01)

    for ax, data, fmt, title in [
        (axes[0], cm,     "d",    "Số lượng mẫu (Count)"),
        (axes[1], cm_pct, ".1f",  "Tỷ lệ chuẩn hóa theo lớp thật (%)"),
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
        ax.set_xlabel("Nhãn dự đoán (Predicted)", color=TEXT_COLOR, fontsize=11)
        ax.set_ylabel("Nhãn thực tế (True Label)", color=TEXT_COLOR, fontsize=11)
        ax.set_facecolor(PANEL_BG)
        ax.tick_params(colors=TEXT_COLOR, labelsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")

    plt.tight_layout()
    out = out_dir / "confusion_matrix.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"[ĐÃ LƯU] Biểu đồ ma trận nhầm lẫn: {out.name}")


def plot_per_class_metrics(y_true, y_pred, out_dir: Path):
    """Vẽ biểu đồ cột so sánh Precision, Recall và F1-Score của từng dòng mã độc."""
    prec   = precision_score(y_true, y_pred, labels=list(range(N_CLASSES)), average=None, zero_division=0)
    recall = recall_score(y_true,    y_pred, labels=list(range(N_CLASSES)), average=None, zero_division=0)
    f1     = f1_score(y_true,        y_pred, labels=list(range(N_CLASSES)), average=None, zero_division=0)

    x     = np.arange(N_CLASSES)
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(PANEL_BG)

    bars_p = ax.bar(x - width,  prec,   width, label="Độ chính xác (Precision)", color="#74c0fc", alpha=0.9)
    bars_r = ax.bar(x,          recall, width, label="Độ thu hồi (Recall)",      color="#69db7c", alpha=0.9)
    bars_f = ax.bar(x + width,  f1,     width, label="Điểm F1-Score",             color="#ffa94d", alpha=0.9)

    for bars in [bars_p, bars_r, bars_f]:
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.3f}",
                        xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 4), textcoords="offset points",
                        ha="center", va="bottom",
                        color=TEXT_COLOR, fontsize=8.5)

    ax.set_title("So sánh Precision / Recall / F1-Score theo từng lớp",
                 color=TEXT_COLOR, fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES, color=TEXT_COLOR, fontsize=10)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Điểm đánh giá [0, 1]", color=TEXT_COLOR)
    ax.tick_params(colors=TEXT_COLOR)
    ax.legend(facecolor="#21262d", labelcolor=TEXT_COLOR, fontsize=10)
    ax.axhline(0.8, color="#ff6b6b", ls="--", lw=1, alpha=0.6, label="Ngưỡng mục tiêu 0.8")

    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")
    ax.yaxis.grid(True, color="#30363d", alpha=0.5)
    ax.set_axisbelow(True)

    plt.tight_layout()
    out = out_dir / "per_class_metrics.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"[ĐÃ LƯU] Biểu đồ chỉ số từng lớp: {out.name}")


def plot_roc_curves(y_true, y_prob, out_dir: Path):
    """Vẽ đường cong ROC (Receiver Operating Characteristic) theo phương pháp One-vs-Rest."""
    y_bin = label_binarize(y_true, classes=list(range(N_CLASSES)))

    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(PANEL_BG)

    for i, (name, color) in enumerate(zip(CLASS_NAMES, PALETTE)):
        if len(np.unique(y_bin[:, i])) > 1:
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
            roc_auc     = auc(fpr, tpr)
            label_str   = f"{name} (AUC = {roc_auc:.3f})"
        else:
            fpr, tpr = [0, 1], [0, 0]
            label_str = f"{name} (Không đủ mẫu)"

        ax.plot(fpr, tpr, color=color, lw=2, label=label_str)

    ax.plot([0, 1], [0, 1], "w--", lw=1, alpha=0.4, label="Đoán ngẫu nhiên (AUC = 0.5)")
    ax.set_title("Đường cong ROC — Chiến lược One-vs-Rest (OvR)",
                 color=TEXT_COLOR, fontsize=13, fontweight="bold")
    ax.set_xlabel("Tỷ lệ dương tính giả (False Positive Rate)", color=TEXT_COLOR, fontsize=11)
    ax.set_ylabel("Tỷ lệ dương tính thật (True Positive Rate)",  color=TEXT_COLOR, fontsize=11)
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
    print(f"[ĐÃ LƯU] Biểu đồ đường cong ROC: {out.name}")


def plot_training_curve(out_dir: Path):
    """Vẽ đồ thị diễn biến huấn luyện từ file lịch sử training_history.npz."""
    hist_path = BASE_DIR / "training_history.npz"
    if not hist_path.exists():
        return
    h  = np.load(hist_path)
    ep = range(1, len(h["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor(DARK_BG)
    fig.suptitle("Lịch sử Huấn luyện — MalwareCNN", color=TEXT_COLOR,
                 fontsize=14, fontweight="bold")

    val_loss_key = "val_loss" if "val_loss" in h else "test_loss"
    val_acc_key  = "val_acc" if "val_acc" in h else "test_acc"

    configs = [
        (axes[0], "train_loss", val_loss_key,  "Hàm mất mát (CrossEntropy Loss)", "Mất mát"),
        (axes[1], "train_acc",  val_acc_key,   "Độ chính xác (Accuracy)",         "Độ chính xác (%)"),
    ]
    for ax, tr_key, va_key, title, ylabel in configs:
        tr = h[tr_key]
        va = h[va_key]
        if "acc" in tr_key:
            tr = [v * 100 for v in tr]
            va = [v * 100 for v in va]

        ax.plot(ep, tr, color=ACCENT,    lw=2, label="Huấn luyện (Train)")
        ax.plot(ep, va, color="#ff6644", lw=2, ls="--", label="Kiểm định (Val)")
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
    print(f"[ĐÃ LƯU] Biểu đồ lịch sử huấn luyện: {out.name}")


def export_predictions_csv(y_prob, y_pred, y_true, raw_df: pd.DataFrame, out_dir: Path):
    """Xuất file predictions.csv chi tiết từng mẫu gồm nhãn dự đoán, xác suất và nhãn thật."""
    res_df = pd.DataFrame()
    res_df["Sample_Index"]    = range(len(y_pred))
    res_df["Predicted_Class"] = y_pred
    res_df["Predicted_Name"]  = [CLASS_NAMES[p] for p in y_pred]
    res_df["Confidence_Pct"]  = [float(y_prob[i, y_pred[i]] * 100) for i in range(len(y_pred))]

    for i, cname in enumerate(CLASS_NAMES):
        res_df[f"Prob_{cname}"] = y_prob[:, i]

    if y_true is not None:
        res_df["True_Class"] = y_true
        res_df["True_Name"]  = [CLASS_NAMES[t] if 0 <= t < N_CLASSES else "Unknown" for t in y_true]
        res_df["Is_Correct"] = (y_pred == y_true)

    out_file = out_dir / "predictions.csv"
    res_df.to_csv(out_file, index=False, encoding="utf-8")
    print(f"[ĐÃ XUẤT FILE] Bảng dự đoán chi tiết: {out_file.name} ({len(res_df):,} dòng)")


def format_malware_benign_summary(y_pred):
    """Tổng hợp tỷ lệ phân chia giữa mã độc nói chung và mã an toàn."""
    total = len(y_pred)
    counts = {i: 0 for i in range(N_CLASSES)}
    for p in y_pred:
        counts[p] += 1
    n_malware = sum(counts[i] for i in range(4))
    n_benign  = counts[4]

    type_map  = {0: "Mã độc", 1: "Mã độc", 2: "Mã độc", 3: "Mã độc", 4: "Mã an toàn"}
    emoji_map = {0: "[!]",    1: "[!]",    2: "[!]",    3: "[!]",    4: "[OK]"}

    lines = [
        "=" * 60,
        "           KẾT QUẢ PHÂN TÍCH AN TOÀN & MÃ ĐỘC",
        "=" * 60,
        f" Tổng số mẫu phân tích : {total:,} mẫu",
        f" Tổng số MÃ ĐỘC        : {n_malware:,} mẫu ({n_malware/total*100:5.1f}%)",
        f" Tổng số MÃ AN TOÀN    : {n_benign:,} mẫu ({n_benign/total*100:5.1f}%)",
        "-" * 60,
        " CHI TIẾT TỪNG LOẠI DỰ ĐOÁN:",
    ]
    for i in range(5):
        c_name  = CLASS_NAMES[i]
        c_type  = type_map[i]
        c_emoji = emoji_map[i]
        c_count = counts[i]
        c_pct   = (c_count / total) * 100
        lines.append(f"  {c_emoji} {c_name:<18} ({c_type}): {c_count:>5} mẫu ({c_pct:5.1f}%)")
    lines.append("=" * 60)
    return "\n".join(lines)


# ==============================================================================
# 5. ĐIỂM VÀO THỰC THI CHÍNH (MAIN FUNCTION)
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="IE105 — Đánh giá toàn diện mô hình MalwareCNN")
    parser.add_argument("--csv", type=str, default=None,
                        help="Đường dẫn đến file CSV cần đánh giá.")
    parser.add_argument("--label", type=str, default="Class",
                        help="Tên cột nhãn trong file CSV (mặc định: 'Class').")
    parser.add_argument("--npz", type=str, default=None,
                        help="Đường dẫn đến file .npz chứa X_test, y_test.")
    parser.add_argument("--model", type=str, default=str(DEFAULT_MODEL),
                        help="Đường dẫn file checkpoint mô hình .pth.")
    parser.add_argument("--out", type=str, default=None,
                        help="Thư mục lưu trữ kết quả và biểu đồ đánh giá.")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Kích thước batch khi suy luận.")
    args = parser.parse_args()

    out_dir = Path(args.out) if args.out else DEFAULT_OUT
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*65}")
    print(f"  IE105 — QUY TRÌNH ĐÁNH GIÁ MÔ HÌNH (EVALUATION)")
    print(f"  Thiết bị tính toán : {DEVICE}")
    print(f"  Thư mục kết xuất   : {out_dir}")
    print(f"{'='*65}")

    model = load_model(Path(args.model))

    # Nạp dữ liệu kiểm thử
    raw_df = None
    if args.csv:
        X_test, y_true, raw_df = load_test_data_csv(Path(args.csv), label_col=args.label)
    else:
        npz_path = Path(args.npz) if args.npz else DEFAULT_NPZ
        X_test, y_true, raw_df = load_test_data_npz(npz_path)

    # Thực hiện dự đoán
    print(f"\n[DỰ ĐOÁN] Đang tính toán trên {len(X_test):,} mẫu...")
    y_prob, y_pred = predict(model, X_test, batch_size=args.batch_size)

    # In tóm tắt tổng quan
    summary_str = format_malware_benign_summary(y_pred)
    print("\n" + summary_str)

    # Xuất file kết quả chi tiết
    export_predictions_csv(y_prob, y_pred, y_true, raw_df, out_dir)

    # Nếu dữ liệu có nhãn gốc thì tiến hành chấm điểm và kết xuất đồ thị
    if y_true is not None:
        aucs = print_metrics(y_true, y_pred, y_prob)

        print("\n[VẼ BIỂU ĐỒ] Đang kết xuất Ma trận nhầm lẫn...")
        plot_confusion_matrix(y_true, y_pred, out_dir)

        print("[VẼ BIỂU ĐỒ] Đang kết xuất Biểu đồ chỉ số từng lớp...")
        plot_per_class_metrics(y_true, y_pred, out_dir)

        print("[VẼ BIỂU ĐỒ] Đang kết xuất Đường cong ROC...")
        plot_roc_curves(y_true, y_prob, out_dir)

        if not args.csv:
            print("[VẼ BIỂU ĐỒ] Đang kết xuất Lịch sử huấn luyện...")
            plot_training_curve(out_dir)

        report = classification_report(y_true, y_pred, labels=list(range(N_CLASSES)),
                                       target_names=CLASS_NAMES, digits=4, zero_division=0)
        valid_aucs = [a for a in aucs if not np.isnan(a)]
        macro_auc = np.mean(valid_aucs) if valid_aucs else 0.0

        metrics_txt = out_dir / "metrics_report.txt"
        with open(metrics_txt, "w", encoding="utf-8") as f:
            f.write(summary_str + "\n\n")
            f.write("IE105 — Chi tiết Metrics đánh giá:\n")
            f.write("=" * 65 + "\n")
            f.write(f"Nguồn dữ liệu     : {args.csv if args.csv else DEFAULT_NPZ.name}\n")
            f.write(f"Tổng số mẫu       : {len(y_true):,}\n")
            f.write(f"Accuracy          : {accuracy_score(y_true, y_pred)*100:.4f}%\n")
            f.write(f"Macro Precision   : {precision_score(y_true, y_pred, average='macro',    zero_division=0):.4f}\n")
            f.write(f"Macro Recall      : {recall_score(y_true,    y_pred, average='macro',    zero_division=0):.4f}\n")
            f.write(f"Macro F1-Score    : {f1_score(y_true,        y_pred, average='macro',    zero_division=0):.4f}\n")
            f.write(f"Weighted F1-Score : {f1_score(y_true,        y_pred, average='weighted', zero_division=0):.4f}\n")
            f.write(f"Macro ROC-AUC     : {macro_auc:.4f}\n\n")
            f.write(report)
        print(f"[ĐÃ LƯU BÁO CÁO] Tệp báo cáo số liệu: {metrics_txt.name}")
    else:
        print("\n[LƯU Ý] File CSV không chứa nhãn thực tế nên không thể tính Accuracy hoặc vẽ Confusion Matrix.")
        print(f"        Kết quả dự đoán xác suất từng mẫu đã được lưu vào: {out_dir / 'predictions.csv'}")

    print(f"\n[HOÀN TẤT] Toàn bộ báo cáo và đồ thị đã được lưu tại: {out_dir}")


if __name__ == "__main__":
    main()
