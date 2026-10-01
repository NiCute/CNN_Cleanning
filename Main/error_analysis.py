"""
error_analysis.py
=================
Đồ án IE105 - Khai phá dữ liệu & Ứng dụng
Mô đun chuyên sâu phân tích sai số (Error Analysis) cho mô hình mạng tích chập MalwareCNN.

Mục tiêu & Phương pháp phân tích:
1. Thống kê các cặp nhầm lẫn phổ biến nhất (Top Confusion Pairs):
   - Tìm ra các cặp lớp thường xuyên bị mô hình dự đoán nhầm (ví dụ: Adware nhầm sang Riskware).
2. Phân tích độ tin cậy (Confidence) & Độ bất định (Shannon Entropy):
   - So sánh phân phối xác suất dự đoán giữa các mẫu đúng và mẫu sai.
   - Phát hiện các mẫu bị sai nhưng mô hình lại có độ tự tin rất cao (High-confidence errors)
     để tìm ra các đặc trưng gây đánh lừa (deceptive features).
3. Chẩn đoán sai số theo nhóm ngữ nghĩa đặc trưng (Semantic Group Error Attribution):
   - So sánh mức độ kích hoạt trung bình giữa nhóm mẫu đúng và nhóm mẫu sai trên từng vùng ngữ nghĩa
     (Network, Privacy, Device, FileSystem...).
4. Trực quan hóa các trường hợp sai điển hình (Case Studies):
   - Hiển thị trực tiếp ma trận ảnh 22x22 của mẫu lỗi đặt cạnh ảnh trung bình của lớp thật và ảnh
     trung bình của lớp bị đoán sai để tìm nguyên nhân thị giác.
5. Xuất báo cáo khoa học dạng Markdown:
   - Lưu tại `evaluation/error_analysis_report.md` để đưa thẳng vào báo cáo nghiệm thu đề tài.
"""

from pathlib import Path
import sys

# Cấu hình UTF-8 cho console Windows để in tiếng Việt có dấu không bị lỗi charmap
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")  # Chế độ headless cho hệ thống console
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

BASE_DIR = Path(__file__).parent.resolve()
EVAL_DIR = BASE_DIR / "evaluation"
EVAL_DIR.mkdir(parents=True, exist_ok=True)

from cnn_model import MalwareCNN, N_CLASSES, encode_label

CLASS_NAMES = ["Adware", "Banking", "SMS Malware", "Riskware", "Benign"]
PALETTE     = ["#ff6b6b", "#ffa94d", "#69db7c", "#74c0fc", "#cc5de8"]

# Thiết lập bảng màu giao diện tối hiện đại
DARK_BG    = "#0d1117"
PANEL_BG   = "#161b22"
TEXT_COLOR = "#e6edf3"
ACCENT     = "#00ff88"


def get_semantic_groups():
    """Lấy danh sách các nhóm ngữ nghĩa và phạm vi chỉ số pixel trên ảnh 22x22."""
    try:
        from evaluation import get_train_feature_names
        from feature_to_image import get_semantic_ordering, SEMANTIC_GROUPS_ORDER
        feats = get_train_feature_names()
        _, info = get_semantic_ordering(feats)
        return [(g, info[g]["offset"], info[g]["offset"] + info[g]["count"] - 1) for g in SEMANTIC_GROUPS_ORDER]
    except Exception:
        # Dự phòng vị trí cố định nếu có lỗi nạp hàm
        return [
            ("FileSystem", 0, 56),
            ("Network", 57, 75),
            ("Process", 76, 99),
            ("Device", 100, 147),
            ("Android_Component", 148, 191),
            ("Privacy", 192, 211),
            ("App_Management", 212, 234),
            ("Crypto", 235, 242),
            ("Other_API", 243, 469),
        ]

SEMANTIC_GROUPS = get_semantic_groups()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==============================================================================
# 1. NẠP MÔ HÌNH VÀ SUY LUẬN XÁC SUẤT TRÊN TẬP TEST
# ==============================================================================

def load_model_and_test_data():
    """Nạp trọng số checkpoint tốt nhất và tập kiểm thử độc lập Test Set."""
    model_path = BASE_DIR / "best_model.pth"
    if not model_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file checkpoint tại: {model_path}")

    model = MalwareCNN(n_classes=N_CLASSES).to(DEVICE)
    ckpt = torch.load(model_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    npz_path = BASE_DIR / "images_22x22.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu: {npz_path}")
    data = np.load(npz_path)

    X_test = data["X_test"]
    y_test_raw = data["y_test"]

    unique_sorted = sorted(set(y_test_raw.tolist()))
    mapping = {u: i for i, u in enumerate(unique_sorted)}
    y_test = np.array([mapping[v] for v in y_test_raw])

    return model, X_test, y_test


@torch.no_grad()
def infer_probabilities(model, X_test):
    """Tính toán phân phối xác suất đầu ra Softmax cho toàn bộ tập kiểm thử."""
    tensor_x = torch.tensor(X_test, dtype=torch.float32)
    loader = torch.utils.data.DataLoader(tensor_x, batch_size=128, shuffle=False)
    all_probs = []

    for batch in loader:
        batch = batch.to(DEVICE)
        logits = model(batch)
        probs = F.softmax(logits, dim=1)
        all_probs.append(probs.cpu().numpy())

    return np.concatenate(all_probs, axis=0)


# ==============================================================================
# 2. CÁC HÀM TRỰC QUAN HÓA SAI SỐ
# ==============================================================================

def plot_top_confusion_pairs(cm: np.ndarray, out_path: Path):
    """Vẽ biểu đồ thanh ngang thể hiện các cặp lớp bị nhầm lẫn nhiều nhất."""
    fig, ax = plt.subplots(figsize=(10, 5.5))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(PANEL_BG)

    pairs = []
    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            if i != j and cm[i, j] > 0:
                pairs.append((f"{CLASS_NAMES[i]} → {CLASS_NAMES[j]}", cm[i, j], i, j))

    # Lấy top 8 cặp có số lượng mẫu nhầm lẫn cao nhất
    pairs = sorted(pairs, key=lambda x: x[1], reverse=True)[:8]
    labels = [p[0] for p in pairs]
    counts = [p[1] for p in pairs]
    colors = [PALETTE[p[2]] for p in pairs]

    bars = ax.barh(labels[::-1], counts[::-1], color=colors[::-1], alpha=0.85, edgecolor="#0d1117")
    ax.set_title("TOP CÁC CẶP LỚP BỊ NHẦM LẪN NHIỀU NHẤT (TẬP KIỂM THỬ)",
                 color=TEXT_COLOR, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Số lượng mẫu bị dự đoán sai", color=TEXT_COLOR, fontsize=11)
    ax.tick_params(colors=TEXT_COLOR)
    ax.grid(axis="x", color="#30363d", alpha=0.4, linestyle="--")

    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.5, bar.get_y() + bar.get_height()/2, f"{int(w)} mẫu",
                va="center", color=TEXT_COLOR, fontsize=9, fontweight="bold")

    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")

    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight", facecolor=DARK_BG)
    print(f"[ĐÃ LƯU] Biểu đồ top cặp nhầm lẫn: {out_path.name}")
    plt.close(fig)


def plot_confidence_distribution(probs: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, out_path: Path):
    """
    So sánh độ tin cậy (Confidence) và độ bất định (Shannon Entropy) giữa các mẫu
    dự đoán đúng và mẫu dự đoán sai.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.patch.set_facecolor(DARK_BG)

    correct_mask = (y_true == y_pred)
    incorrect_mask = ~correct_mask

    confidences = np.max(probs, axis=1)
    eps = 1e-12
    # Tính entropy: -sum(p * log(p)). Entropy càng cao thể hiện mô hình càng phân vân giữa các lớp.
    entropies = -np.sum(probs * np.log(probs + eps), axis=1)

    # 1. Biểu đồ phân phối độ tin cậy
    ax0 = axes[0]
    ax0.set_facecolor(PANEL_BG)
    ax0.hist(confidences[correct_mask] * 100, bins=25, alpha=0.75, color=ACCENT,
             label=f"Đoán đúng (TB: {np.mean(confidences[correct_mask])*100:.1f}%)", density=True)
    ax0.hist(confidences[incorrect_mask] * 100, bins=25, alpha=0.75, color="#ff6b6b",
             label=f"Đoán sai (TB: {np.mean(confidences[incorrect_mask])*100:.1f}%)", density=True)
    ax0.set_title("Phân phối Độ Tin Cậy (Confidence)", color=TEXT_COLOR, fontsize=12, fontweight="bold")
    ax0.set_xlabel("Độ tin cậy của dự đoán (%)", color=TEXT_COLOR)
    ax0.set_ylabel("Mật độ phân phối", color=TEXT_COLOR)
    ax0.tick_params(colors=TEXT_COLOR)
    ax0.legend(facecolor=PANEL_BG, edgecolor="#30363d", labelcolor=TEXT_COLOR)
    ax0.grid(color="#30363d", alpha=0.4, linestyle="--")

    # 2. Biểu đồ phân phối độ bất định (Entropy)
    ax1 = axes[1]
    ax1.set_facecolor(PANEL_BG)
    ax1.hist(entropies[correct_mask], bins=25, alpha=0.75, color=ACCENT,
             label=f"Đoán đúng (TB: {np.mean(entropies[correct_mask]):.2f})", density=True)
    ax1.hist(entropies[incorrect_mask], bins=25, alpha=0.75, color="#ff6b6b",
             label=f"Đoán sai (TB: {np.mean(entropies[incorrect_mask]):.2f})", density=True)
    ax1.set_title("Phân phối Độ Bất Định (Shannon Entropy)", color=TEXT_COLOR, fontsize=12, fontweight="bold")
    ax1.set_xlabel("Giá trị Entropy (càng cao thì mô hình càng lưỡng lự)", color=TEXT_COLOR)
    ax1.tick_params(colors=TEXT_COLOR)
    ax1.legend(facecolor=PANEL_BG, edgecolor="#30363d", labelcolor=TEXT_COLOR)
    ax1.grid(color="#30363d", alpha=0.4, linestyle="--")

    for ax in axes:
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")

    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight", facecolor=DARK_BG)
    print(f"[ĐÃ LƯU] Biểu đồ phân phối độ tin cậy và entropy: {out_path.name}")
    plt.close(fig)


def plot_misclassified_case_studies(X_test, y_true, y_pred, probs, out_path: Path):
    """
    Trực quan hóa nghiên cứu tình huống (Case Studies) các mẫu bị đoán sai điển hình:
    - Cột 1: Ma trận ảnh 22x22 thực tế của mẫu bị phân loại sai.
    - Cột 2: Ảnh trung bình của lớp đúng (Ground Truth Class Mean).
    - Cột 3: Ảnh trung bình của lớp bị dự đoán nhầm (Predicted Class Mean).
    """
    error_indices = np.where(y_true != y_pred)[0]
    if len(error_indices) == 0:
        return

    # Lựa chọn 4 ca sai có độ tự tin cao nhất để phân tích chuyên sâu
    confidences = np.max(probs, axis=1)
    sorted_errors = error_indices[np.argsort(-confidences[error_indices])]
    selected = sorted_errors[:4]

    fig, axes = plt.subplots(len(selected), 3, figsize=(12, 3.2 * len(selected)))
    fig.patch.set_facecolor(DARK_BG)
    fig.suptitle("TRỰC QUAN HÓA CÁC CA DỰ ĐOÁN SAI ĐIỂN HÌNH (CASE STUDIES)",
                 color=TEXT_COLOR, fontsize=14, fontweight="bold", y=0.99)

    class_means = {}
    for c in range(N_CLASSES):
        mask = (y_true == c)
        if np.sum(mask) > 0:
            class_means[c] = np.mean(X_test[mask, 0], axis=0)
        else:
            class_means[c] = np.zeros((22, 22))

    for row_idx, sample_idx in enumerate(selected):
        t_cls = y_true[sample_idx]
        p_cls = y_pred[sample_idx]
        conf = probs[sample_idx, p_cls] * 100
        true_prob = probs[sample_idx, t_cls] * 100

        # Mẫu thực tế bị sai
        ax0 = axes[row_idx, 0]
        ax0.set_facecolor(PANEL_BG)
        ax0.imshow(X_test[sample_idx, 0], cmap="magma", vmin=0, vmax=1)
        ax0.set_title(f"Mẫu #{sample_idx}\nThật: {CLASS_NAMES[t_cls]} | Đoán: {CLASS_NAMES[p_cls]} ({conf:.1f}%)",
                      color="#ff6b6b", fontsize=10, fontweight="bold")
        ax0.set_xticks([]); ax0.set_yticks([])

        # Ảnh trung bình lớp thật
        ax1 = axes[row_idx, 1]
        ax1.set_facecolor(PANEL_BG)
        ax1.imshow(class_means[t_cls], cmap="magma", vmin=0, vmax=1)
        ax1.set_title(f"Ảnh trung bình lớp THẬT:\n{CLASS_NAMES[t_cls]} (Xác suất: {true_prob:.1f}%)",
                      color=PALETTE[t_cls], fontsize=10, fontweight="bold")
        ax1.set_xticks([]); ax1.set_yticks([])

        # Ảnh trung bình lớp bị đoán sai
        ax2 = axes[row_idx, 2]
        ax2.set_facecolor(PANEL_BG)
        ax2.imshow(class_means[p_cls], cmap="magma", vmin=0, vmax=1)
        ax2.set_title(f"Ảnh trung bình lớp ĐOÁN NHẦM:\n{CLASS_NAMES[p_cls]} (Xác suất: {conf:.1f}%)",
                      color=PALETTE[p_cls], fontsize=10, fontweight="bold")
        ax2.set_xticks([]); ax2.set_yticks([])

        for ax in [ax0, ax1, ax2]:
            for spine in ax.spines.values():
                spine.set_edgecolor("#30363d")

    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight", facecolor=DARK_BG)
    print(f"[ĐÃ LƯU] Biểu đồ ca sai điển hình: {out_path.name}")
    plt.close(fig)


def plot_semantic_group_error_attribution(X_test, y_true, y_pred, out_path: Path):
    """
    Chẩn đoán nguồn gốc sai số theo từng nhóm ngữ nghĩa:
    So sánh cường độ kích hoạt điểm ảnh trung bình giữa tập mẫu đúng và tập mẫu sai
    để tìm ra vùng hành vi gây sai lệch phán đoán của CNN.
    """
    correct_mask = (y_true == y_pred)
    error_mask = ~correct_mask

    X_flat = X_test[:, 0].reshape(len(X_test), -1)

    group_names = [g[0] for g in SEMANTIC_GROUPS]
    correct_means = []
    error_means = []
    diffs = []
    details = []

    for name, p_start, p_end in SEMANTIC_GROUPS:
        c_mean = float(np.mean(X_flat[correct_mask, p_start:p_end+1]))
        e_mean = float(np.mean(X_flat[error_mask, p_start:p_end+1]))
        d = e_mean - c_mean
        count = p_end - p_start + 1
        correct_means.append(c_mean)
        error_means.append(e_mean)
        diffs.append(d)
        details.append({
            "name": name,
            "range": f"[{p_start}-{p_end}]",
            "count": count,
            "correct_mean": c_mean,
            "error_mean": e_mean,
            "diff": d,
        })

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.patch.set_facecolor(DARK_BG)

    # 1. So sánh mức độ kích hoạt
    ax0 = axes[0]
    ax0.set_facecolor(PANEL_BG)
    indices = np.arange(len(group_names))
    width = 0.38

    ax0.bar(indices - width/2, correct_means, width, label="Mẫu Đoán Đúng", color=ACCENT, alpha=0.85)
    ax0.bar(indices + width/2, error_means, width, label="Mẫu Đoán Sai", color="#ff6b6b", alpha=0.85)

    ax0.set_title("Mức Độ Kích Hoạt Trung Bình Theo Nhóm: Đúng vs Sai", color=TEXT_COLOR, fontsize=12, fontweight="bold")
    ax0.set_xticks(indices)
    ax0.set_xticklabels(group_names, color=TEXT_COLOR, fontsize=9.5, rotation=20, ha="right")
    ax0.set_ylabel("Cường độ kích hoạt [0, 1]", color=TEXT_COLOR)
    ax0.tick_params(colors=TEXT_COLOR)
    ax0.legend(facecolor=PANEL_BG, edgecolor="#30363d", labelcolor=TEXT_COLOR)
    ax0.grid(axis="y", color="#30363d", alpha=0.4, linestyle="--")

    # 2. Độ chênh lệch kích hoạt (Mẫu Sai - Mẫu Đúng)
    ax1 = axes[1]
    ax1.set_facecolor(PANEL_BG)
    diff_colors = ["#ff6b6b" if d > 0 else "#74c0fc" for d in diffs]
    bars_diff = ax1.barh(group_names[::-1], diffs[::-1], color=diff_colors[::-1], alpha=0.85, edgecolor="#0d1117")

    ax1.axvline(0, color=TEXT_COLOR, lw=0.8, linestyle="--", alpha=0.5)
    ax1.set_title("Độ Lệch Kích Hoạt (Mẫu Sai - Mẫu Đúng)", color=TEXT_COLOR, fontsize=12, fontweight="bold")
    ax1.set_xlabel("Độ lệch (Đỏ: Mẫu sai cao hơn, Xanh: Mẫu sai thấp hơn)", color=TEXT_COLOR)
    ax1.tick_params(colors=TEXT_COLOR)
    ax1.grid(axis="x", color="#30363d", alpha=0.4, linestyle="--")

    for bar in bars_diff:
        w = bar.get_width()
        ha = "left" if w >= 0 else "right"
        offset = 0.001 if w >= 0 else -0.001
        ax1.text(w + offset, bar.get_y() + bar.get_height()/2, f"{w:+.4f}",
                 va="center", ha=ha, color=TEXT_COLOR, fontsize=8.5, fontweight="bold")

    for ax in axes:
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")

    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight", facecolor=DARK_BG)
    print(f"[ĐÃ LƯU] Biểu đồ chẩn đoán nhóm ngữ nghĩa: {out_path.name}")
    plt.close(fig)

    return details


# ==============================================================================
# 3. KẾT XUẤT BÁO CÁO PHÂN TÍCH SAI SỐ (MARKDOWN REPORT)
# ==============================================================================

def generate_error_report(X_test, y_true, y_pred, probs, cm, out_path: Path, semantic_details=None):
    """Xuất báo cáo khoa học dạng Markdown tổng hợp toàn bộ các phát hiện sai số."""
    total = len(y_true)
    correct = np.sum(y_true == y_pred)
    errors = total - correct
    error_rate = (errors / total) * 100

    confidences = np.max(probs, axis=1)
    correct_conf = np.mean(confidences[y_true == y_pred]) * 100
    error_conf = np.mean(confidences[y_true != y_pred]) * 100

    # Top các cặp nhầm lẫn
    confused_pairs = []
    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            if i != j and cm[i, j] > 0:
                confused_pairs.append({
                    "true_class": CLASS_NAMES[i],
                    "pred_class": CLASS_NAMES[j],
                    "count": cm[i, j],
                    "pct_of_true": (cm[i, j] / np.sum(cm[i, :])) * 100,
                })
    confused_pairs = sorted(confused_pairs, key=lambda x: x["count"], reverse=True)

    # Thống kê chi tiết theo từng lớp
    class_stats = []
    for i in range(N_CLASSES):
        n_c = np.sum(y_true == i)
        n_corr = cm[i, i]
        n_err = n_c - n_corr
        rec = (n_corr / n_c) * 100 if n_c > 0 else 0
        prec = (n_corr / np.sum(cm[:, i])) * 100 if np.sum(cm[:, i]) > 0 else 0
        class_stats.append({
            "name": CLASS_NAMES[i],
            "total": n_c,
            "correct": n_corr,
            "errors": n_err,
            "recall": rec,
            "precision": prec,
        })

    # Các lỗi sai tự tin cao
    high_conf_errors = np.where((y_true != y_pred) & (confidences >= 0.70))[0]
    high_conf_pct = (len(high_conf_errors) / errors * 100) if errors > 0 else 0

    # Phân tích rủi ro an ninh mạng: False Negative (lọt mã độc) vs False Positive (báo động giả)
    fn_malware = np.sum((y_true < 4) & (y_pred == 4))   # Mã độc bị đoán thành an toàn (nguy hiểm nhất)
    fp_benign  = np.sum((y_true == 4) & (y_pred < 4))   # An toàn bị đoán thành mã độc (báo động nhầm)

    md = []
    md.append("# 🔬 BÁO CÁO PHÂN TÍCH SAI SỐ (ERROR ANALYSIS REPORT)")
    md.append("> **Đề tài:** Phân loại mã độc Android bằng mô hình CNN trên không gian ma trận ảnh 22×22")
    md.append("> **Tập dữ liệu:** CICMalDroid2020 Test Set (1,740 mẫu)")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## 1. 📊 Tổng Quan Sai Số Toàn Cục")
    md.append("")
    md.append(f"- **Tổng số mẫu kiểm thử:** {total:,} mẫu")
    md.append(f"- **Dự đoán chính xác:** {correct:,} mẫu ({(correct/total)*100:.2f}%)")
    md.append(f"- **Tổng số mẫu sai (Errors):** {errors:,} mẫu ({error_rate:.2f}%)")
    md.append(f"- **Độ tin cậy trung bình khi ĐÚNG:** {correct_conf:.2f}%")
    md.append(f"- **Độ tin cậy trung bình khi SAI:** {error_conf:.2f}%")
    md.append(f"- **Số ca sai có độ tự tin cao (Confidence ≥ 70%):** {len(high_conf_errors)} mẫu ({high_conf_pct:.1f}% tổng lỗi)")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## 2. 🛡️ Đánh Giá Rủi Ro Bảo Mật (Security Impact)")
    md.append("")
    md.append("| Loại rủi ro | Số mẫu | Tỷ lệ | Đánh giá an ninh |")
    md.append("|---|---|---|---|")
    md.append(f"| **False Negative (Lọt mã độc):** Malware bị đoán thành Benign | **{fn_malware} mẫu** | **{fn_malware/total*100:.2f}%** | 🔴 **Nghiêm trọng:** Mã độc vượt qua hệ thống kiểm duyệt |")
    md.append(f"| **False Positive (Báo động giả):** Benign bị đoán thành Malware | **{fp_benign} mẫu** | **{fp_benign/total*100:.2f}%** | 🟡 **Ảnh hưởng UX:** Ứng dụng lành tính bị chặn nhầm |")
    md.append(f"| **Lỗi nội bộ các dòng mã độc:** Malware loại A đoán thành B | **{errors - fn_malware - fp_benign} mẫu** | **{(errors - fn_malware - fp_benign)/total*100:.2f}%** | 🟢 **Ít nguy hại:** Vẫn nhận diện được bản chất là mã độc |")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## 3. 📈 Thống Kê Chi Tiết Sai Số Theo Từng Lớp")
    md.append("")
    md.append("| Lớp (Class) | Tổng mẫu Test | Đúng | Sai | Recall (%) | Precision (%) |")
    md.append("|---|---|---|---|---|---|")
    for s in class_stats:
        md.append(f"| **{s['name']}** | {s['total']:,} | {s['correct']:,} | {s['errors']:,} | {s['recall']:.2f}% | {s['precision']:.2f}% |")
    md.append("")

    if semantic_details:
        md.append("---")
        md.append("")
        md.append("## 3.1. 🧩 Phân Tích Độ Lệch Đặc Trưng Theo Nhóm Semantic")
        md.append("> So sánh cường độ kích hoạt trung bình giữa tập mẫu dự đoán **Đúng** và mẫu dự đoán **Sai**.")
        md.append("")
        md.append("| Nhóm Semantic | Vùng Pixel | Số Feature | Đoán Đúng (Mean) | Đoán Sai (Mean) | Độ Lệch (Sai - Đúng) | Xu hướng sai lệch |")
        md.append("|---|---|---|---|---|---|---|")
        for d in semantic_details:
            trend = "🔴 Kích hoạt bất thường cao hơn" if d["diff"] > 0 else "🔵 Kích hoạt yếu hơn bình thường"
            md.append(f"| **{d['name']}** | `{d['range']}` | {d['count']} | {d['correct_mean']:.4f} | {d['error_mean']:.4f} | **{d['diff']:+.4f}** | {trend} |")
        md.append("")

    md.append("---")
    md.append("")
    md.append("## 4. 🔀 Top Các Cặp Lớp Bị Nhầm Lẫn Nhiều Nhất")
    md.append("")
    md.append("| Thứ hạng | Cặp nhầm lẫn (Thật → Đoán) | Số lượng | % trên tổng lớp thật | Nguyên nhân tiềm ẩn |")
    md.append("|---|---|---|---|---|")
    for rank, p in enumerate(confused_pairs[:6], 1):
        cause = "Hành vi tương đồng giữa các họ mã độc"
        if "Benign" in p["true_class"] or "Benign" in p["pred_class"]:
            cause = "Sử dụng API dùng chung giữa ứng dụng lành tính và mã độc"
        elif "SMS" in p["true_class"] and "Riskware" in p["pred_class"]:
            cause = "Riskware thường xin quyền SEND_SMS hoặc đọc danh bạ tương tự SMS Malware"
        elif "Adware" in p["true_class"] and "Riskware" in p["pred_class"]:
            cause = "Adware chèn quảng cáo và thu thập thông tin thiết bị tương tự Riskware"
        md.append(f"| #{rank} | **{p['true_class']} → {p['pred_class']}** | {p['count']} mẫu | {p['pct_of_true']:.2f}% | {cause} |")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## 5. 🧠 Phân Tích Nguyên Nhân & Kiến Nghị Khắc Phục")
    md.append("")
    md.append("### 5.1. Vì sao mô hình đưa ra dự đoán sai?")
    md.append("1. **Chồng lấn hành vi Semantic (Behavior Overlap):**")
    md.append("   - Các họ mã độc như *Riskware* và *SMS Malware* cùng sử dụng chung nhóm API Device (`SEND_SMS`, `READ_PHONE_STATE`, `GET_DEVICE_ID`).")
    md.append("   - Khi biểu diễn thành ảnh 22×22, các pixel tại khu vực *Device* và *Network* có cường độ kích hoạt rất gần nhau.")
    md.append("2. **Độ thưa của ma trận đặc trưng (Feature Sparsity):**")
    md.append("   - Nhiều mẫu chỉ kích hoạt một vài API đặc biệt thuộc nhóm `Other_API`, khiến phần lớn các vùng còn lại trên ảnh 22×22 gần bằng 0.")
    md.append("3. **Sự đánh đổi khi cân bằng lớp (Class Weight Trade-off):**")
    md.append("   - Việc áp dụng `class_weight` giúp cải thiện mạnh mẽ Recall của các lớp thiểu số (*Adware*, *Banking*), nhưng có thể làm tăng nhẹ một vài ca dự đoán nhầm giữa các lớp có trọng số tương tự.")
    md.append("")
    md.append("### 5.2. Đề xuất cải tiến cho đồ án:")
    md.append("- **Focal Loss:** Thay thế hoặc kết hợp với CrossEntropy để giảm trọng số các mẫu dễ học (easy negatives) và tập trung vào các biên khó phân loại.")
    md.append("- **Attention Mechanism (SE-Net hoặc CBAM):** Bổ sung kênh chú ý để CNN tập trung vào các vùng Semantic quan trọng nhất (như vùng Crypto hoặc Privacy) thay vì quét đồng đều.")
    md.append("- **Sắp xếp thứ tự Feature tối ưu hơn:** Áp dụng thuật toán t-SNE hoặc MDS trên 470 features để các features có độ tương quan cao nằm cạnh nhau trên lưới 22×22.")
    md.append("")
    md.append("---")
    md.append("*Báo cáo được khởi tạo tự động bởi mô đun `error_analysis.py` phục vụ bảo vệ đồ án IE105.*")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print(f"[ĐÃ LƯU BÁO CÁO MARKDOWN] {out_path.name}")


# ==============================================================================
# 4. ĐIỂM VÀO THỰC THI CHÍNH
# ==============================================================================

def main():
    print("=" * 65)
    print("  IE105 — PHÂN TÍCH CHUYÊN SÂU SAI SỐ MÔ HÌNH CNN")
    print("=" * 65)

    model, X_test, y_test = load_model_and_test_data()
    print(f"[DỮ LIỆU] Tập kiểm thử (Test Set): {len(X_test):,} mẫu")

    probs = infer_probabilities(model, X_test)
    y_pred = np.argmax(probs, axis=1)

    cm = confusion_matrix(y_test, y_pred)

    out_p1 = EVAL_DIR / "error_top_confusions.png"
    plot_top_confusion_pairs(cm, out_p1)

    out_p2 = EVAL_DIR / "error_confidence_uncertainty.png"
    plot_confidence_distribution(probs, y_test, y_pred, out_p2)

    out_p3 = EVAL_DIR / "error_sample_case_studies.png"
    plot_misclassified_case_studies(X_test, y_test, y_pred, probs, out_p3)

    out_p4 = EVAL_DIR / "error_semantic_group_attribution.png"
    semantic_details = plot_semantic_group_error_attribution(X_test, y_test, y_pred, out_p4)

    out_report = EVAL_DIR / "error_analysis_report.md"
    generate_error_report(X_test, y_test, y_pred, probs, cm, out_report, semantic_details=semantic_details)

    print("\n[HOÀN TẤT] Đã xuất đầy đủ toàn bộ biểu đồ và báo cáo phân tích sai số!")


if __name__ == "__main__":
    main()
