"""
visualize_samples.py
====================
Đồ án IE105 - Khai phá dữ liệu & Ứng dụng
Mô đun trực quan hóa không gian đặc trưng dạng ảnh 22x22 và phân tích bản đồ nhiệt ngữ nghĩa.

Mục tiêu & Phương pháp:
1. So sánh trực quan mẫu thực tế và ảnh trung bình của 5 lớp:
   - Kết xuất lưới ảnh 3 hàng x 5 cột:
       + Hàng 1: Mẫu thực tế #1
       + Hàng 2: Mẫu thực tế #2
       + Hàng 3: Ảnh trung bình (Mean Map) đại diện cho từng lớp
2. Bản đồ bố trí ngữ nghĩa (Semantic Layout) & Bản đồ khác biệt (Difference Map):
   - Thể hiện vị trí 9 nhóm ngữ nghĩa trên lưới 22x22.
   - Tính toán ma trận chênh lệch: Difference Map = Mean(Malware) - Mean(Benign).
   - Vùng màu đỏ thể hiện các API/hành vi mà mã độc kích hoạt mạnh hơn hẳn ứng dụng an toàn.
3. Biểu đồ cột so sánh mức độ kích hoạt trung bình theo từng nhóm ngữ nghĩa:
   - Định lượng rõ ràng nhóm hành vi nào là "chìa khóa" giúp phân biệt từng dòng mã độc
     (ví dụ: SMS Malware kích hoạt vượt trội ở nhóm Device/SMS, Banking kích hoạt mạnh ở Network/Privacy).

Các file biểu đồ xuất ra:
- `malware_vs_benign_22x22.png`
- `semantic_layout_and_difference_map.png`
- `semantic_group_activation_comparison.png`
"""

from pathlib import Path
import shutil
import sys

# Cấu hình UTF-8 cho console Windows để in tiếng Việt có dấu không bị lỗi charmap
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import matplotlib
matplotlib.use("Agg")  # Chế độ headless cho console
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).parent.resolve()
EVAL_DIR = BASE_DIR / "evaluation"
EVAL_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ["Adware", "Banking", "SMS Malware", "Riskware", "Benign"]
PALETTE     = ["#ff6b6b", "#ffa94d", "#69db7c", "#74c0fc", "#cc5de8"]

# Bảng màu giao diện tối hiện đại
DARK_BG    = "#0d1117"
PANEL_BG   = "#161b22"
TEXT_COLOR = "#e6edf3"
ACCENT     = "#00ff88"


def get_semantic_groups():
    """Lấy danh sách các nhóm ngữ nghĩa kèm màu sắc nhận diện và phạm vi chỉ số pixel trên ảnh 22x22."""
    colors = {
        "FileSystem": "#ff7b72",
        "Network": "#79c0ff",
        "Process": "#d2a8ff",
        "Device": "#ffa657",
        "Android_Component": "#7ee787",
        "Privacy": "#ff7b72",
        "App_Management": "#a5d6ff",
        "Crypto": "#f2cc60",
        "Other_API": "#8b949e",
    }
    try:
        from evaluation import get_train_feature_names
        from feature_to_image import get_semantic_ordering, SEMANTIC_GROUPS_ORDER
        feats = get_train_feature_names()
        _, info = get_semantic_ordering(feats)
        groups = []
        for g in SEMANTIC_GROUPS_ORDER:
            start = info[g]["offset"]
            end = start + info[g]["count"] - 1
            groups.append((g, start, end, colors.get(g, "#8b949e")))
        groups.append(("Padding_Zeros", 470, 483, "#30363d"))
        return groups
    except Exception:
        return [
            ("FileSystem", 0, 56, "#ff7b72"),
            ("Network", 57, 75, "#79c0ff"),
            ("Process", 76, 99, "#d2a8ff"),
            ("Device", 100, 147, "#ffa657"),
            ("Android_Component", 148, 191, "#7ee787"),
            ("Privacy", 192, 211, "#ff7b72"),
            ("App_Management", 212, 234, "#a5d6ff"),
            ("Crypto", 235, 242, "#f2cc60"),
            ("Other_API", 243, 469, "#8b949e"),
            ("Padding_Zeros", 470, 483, "#30363d"),
        ]

SEMANTIC_GROUPS = get_semantic_groups()


# ==============================================================================
# 1. NẠP DỮ LIỆU ĐỂ TÍNH TOÁN ẢNH TRUNG BÌNH
# ==============================================================================

def load_data():
    """Gộp tập Train và Val để có tập mẫu dồi dào nhằm tính toán ảnh trung bình (Class Mean Map) ổn định nhất."""
    npz_path = BASE_DIR / "images_22x22.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {npz_path}")
    data = np.load(npz_path)

    X = np.concatenate([data["X_train"], data["X_val"]], axis=0)
    y_raw = np.concatenate([data["y_train"], data["y_val"]], axis=0)

    unique = sorted(set(y_raw.tolist()))
    mapping = {u: i for i, u in enumerate(unique)}
    y = np.array([mapping[v] for v in y_raw])

    return X, y


# ==============================================================================
# 2. VẼ LƯỚI ẢNH MẪU VÀ ẢNH TRUNG BÌNH (MALWARE VS BENIGN GRID)
# ==============================================================================

def plot_malware_vs_benign_grid(X, y, out_paths):
    """
    Vẽ lưới ảnh 22x22 gồm 3 hàng cho 5 lớp:
    - Hàng 1: Mẫu thực tế #1
    - Hàng 2: Mẫu thực tế #2
    - Hàng 3: Ảnh trung bình (Mean Map) của toàn bộ lớp
    """
    fig, axes = plt.subplots(3, 5, figsize=(16, 10))
    fig.patch.set_facecolor(DARK_BG)
    fig.suptitle("TRỰC QUAN HÓA MA TRẬN ẢNH 22x22: MALWARE VS BENIGN (CICMalDroid2020)",
                 color=TEXT_COLOR, fontsize=15, fontweight="bold", y=0.98)

    row_titles = ["Mẫu thực tế #1", "Mẫu thực tế #2", "Ảnh trung bình lớp"]

    for c_idx in range(5):
        c_name = CLASS_NAMES[c_idx]
        mask = (y == c_idx)
        X_c = X[mask]

        s1 = X_c[0, 0]
        s2 = X_c[1, 0] if len(X_c) > 1 else X_c[0, 0]
        mean_img = np.mean(X_c[:, 0], axis=0)

        imgs = [s1, s2, mean_img]

        for r_idx in range(3):
            ax = axes[r_idx, c_idx]
            ax.set_facecolor(PANEL_BG)

            im = ax.imshow(imgs[r_idx], cmap="magma", vmin=0.0, vmax=1.0)
            ax.set_xticks([])
            ax.set_yticks([])

            # Đổi màu viền theo màu nhận diện của từng lớp
            for spine in ax.spines.values():
                spine.set_edgecolor(PALETTE[c_idx])
                spine.set_linewidth(1.5)

            if r_idx == 0:
                ax.set_title(f"{c_name}\n({np.sum(mask):,} mẫu)",
                             color=PALETTE[c_idx], fontsize=12, fontweight="bold", pad=8)
            if c_idx == 0:
                ax.set_ylabel(row_titles[r_idx], color=TEXT_COLOR,
                              fontsize=11, fontweight="bold", labelpad=8)

    # Thêm thang màu chuẩn hóa ở dưới đáy biểu đồ
    cbar_ax = fig.add_axes([0.25, 0.04, 0.5, 0.02])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation="horizontal")
    cbar.set_label("Mức xám điểm ảnh chuẩn hóa [0, 1] (MinMaxScaler)", color=TEXT_COLOR, fontsize=10)
    cbar.ax.tick_params(colors=TEXT_COLOR, labelsize=9)

    plt.subplots_adjust(left=0.08, right=0.95, top=0.88, bottom=0.10, wspace=0.15, hspace=0.25)

    for p in out_paths:
        plt.savefig(p, dpi=160, bbox_inches="tight", facecolor=DARK_BG)
        print(f"[ĐÃ LƯU] Biểu đồ lưới mẫu: {p.name}")
    plt.close(fig)


# ==============================================================================
# 3. BẢN ĐỒ BỐ TRÍ NGỮ NGHĨA VÀ BẢN ĐỒ KHÁC BIỆT (DIFFERENCE MAP)
# ==============================================================================

def plot_semantic_layout_and_difference(X, y, out_paths):
    """
    Kết xuất bộ 4 ảnh phân tích:
    1. Bản đồ bố trí 9 nhóm ngữ nghĩa trên lưới 22x22.
    2. Ảnh trung bình của dòng mã an toàn (Benign Mean).
    3. Ảnh trung bình tổng hợp của tất cả các dòng mã độc (Malware Mean).
    4. Bản đồ khác biệt (Difference Map = Malware Mean - Benign Mean).
    """
    fig, axes = plt.subplots(1, 4, figsize=(20, 5.5))
    fig.patch.set_facecolor(DARK_BG)
    fig.suptitle("PHÂN TÍCH KHU VỰC NGỮ NGHĨA VÀ BẢN ĐỒ KHÁC BIỆT (DIFFERENCE MAP)",
                 color=TEXT_COLOR, fontsize=15, fontweight="bold", y=0.98)

    # 1. Bản đồ layout các nhóm ngữ nghĩa
    layout_grid = np.zeros((22, 22))
    for group_idx, (g_name, p_start, p_end, _) in enumerate(SEMANTIC_GROUPS):
        for p in range(p_start, p_end + 1):
            r = p // 22
            c = p % 22
            layout_grid[r, c] = group_idx

    ax0 = axes[0]
    ax0.set_facecolor(PANEL_BG)
    cmap_custom = plt.get_cmap("tab10", len(SEMANTIC_GROUPS))
    ax0.imshow(layout_grid, cmap=cmap_custom, vmin=0, vmax=len(SEMANTIC_GROUPS)-1)
    ax0.set_title("1. Bố trí Nhóm Semantic (22x22)", color=TEXT_COLOR, fontsize=12, fontweight="bold")
    ax0.set_xticks([])
    ax0.set_yticks([])

    patches = [
        mpatches.Patch(color=cmap_custom(i), label=f"{name}")
        for i, (name, _, _, _) in enumerate(SEMANTIC_GROUPS)
    ]
    ax0.legend(handles=patches, loc="upper center", bbox_to_anchor=(0.5, -0.05),
               ncol=2, fontsize=7.5, facecolor=PANEL_BG, edgecolor="#30363d",
               labelcolor=TEXT_COLOR)

    # 2. Ảnh trung bình Benign (Mã an toàn)
    benign_mask = (y == 4)
    benign_mean = np.mean(X[benign_mask, 0], axis=0)

    ax1 = axes[1]
    ax1.set_facecolor(PANEL_BG)
    im1 = ax1.imshow(benign_mean, cmap="viridis", vmin=0.0, vmax=0.8)
    ax1.set_title("2. Benign (Mã an toàn)\nẢnh trung bình", color="#cc5de8", fontsize=12, fontweight="bold")
    ax1.set_xticks([])
    ax1.set_yticks([])
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04).ax.tick_params(colors=TEXT_COLOR)

    # 3. Ảnh trung bình Malware (Tổng hợp cả 4 dòng mã độc: Adware, Banking, SMS, Riskware)
    malware_mask = (y < 4)
    malware_mean = np.mean(X[malware_mask, 0], axis=0)

    ax2 = axes[2]
    ax2.set_facecolor(PANEL_BG)
    im2 = ax2.imshow(malware_mean, cmap="viridis", vmin=0.0, vmax=0.8)
    ax2.set_title("3. Malware (Tổng hợp Mã độc)\nẢnh trung bình", color="#ff6b6b", fontsize=12, fontweight="bold")
    ax2.set_xticks([])
    ax2.set_yticks([])
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04).ax.tick_params(colors=TEXT_COLOR)

    # 4. Bản đồ khác biệt: Malware Mean - Benign Mean
    diff = malware_mean - benign_mean
    ax3 = axes[3]
    ax3.set_facecolor(PANEL_BG)
    max_abs = max(abs(diff.min()), abs(diff.max()))
    im3 = ax3.imshow(diff, cmap="coolwarm", vmin=-max_abs, vmax=max_abs)
    ax3.set_title("4. Bản đồ khác biệt\n(Malware - Benign)", color=ACCENT, fontsize=12, fontweight="bold")
    ax3.set_xticks([])
    ax3.set_yticks([])
    cbar3 = plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    cbar3.set_label("Độ lệch: Đỏ (+) = Mã độc cao hơn | Xanh (-) = An toàn cao hơn",
                    color=TEXT_COLOR, fontsize=8)
    cbar3.ax.tick_params(colors=TEXT_COLOR)

    for ax in axes:
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")

    plt.tight_layout()
    for p in out_paths:
        plt.savefig(p, dpi=160, bbox_inches="tight", facecolor=DARK_BG)
        print(f"[ĐÃ LƯU] Bản đồ khác biệt: {p.name}")
    plt.close(fig)


# ==============================================================================
# 4. SO SÁNH CƯỜNG ĐỘ KÍCH HOẠT THEO NHÓM NGỮ NGHĨA (BAR CHART)
# ==============================================================================

def plot_semantic_group_activation(X, y, out_paths):
    """Vẽ biểu đồ cột so sánh cường độ kích hoạt trung bình của từng nhóm ngữ nghĩa giữa 5 lớp."""
    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(PANEL_BG)

    groups_to_eval = SEMANTIC_GROUPS[:-1]   # Bỏ qua nhóm padding số 0 ở cuối
    group_names = [g[0] for g in groups_to_eval]

    n_groups = len(groups_to_eval)
    bar_width = 0.15
    indices = np.arange(n_groups)

    for c_idx in range(5):
        c_name = CLASS_NAMES[c_idx]
        mask = (y == c_idx)
        X_c = X[mask, 0]
        X_c = X_c.reshape(len(X_c), -1)

        means = []
        for _, p_start, p_end, _ in groups_to_eval:
            group_vals = X_c[:, p_start:p_end+1]
            means.append(np.mean(group_vals))

        pos = indices + (c_idx - 2) * bar_width
        ax.bar(pos, means, width=bar_width, label=c_name, color=PALETTE[c_idx],
               alpha=0.9, edgecolor="#0d1117", lw=0.8)

    ax.set_title("MỨC ĐỘ KÍCH HOẠT ĐẶC TRƯNG THEO TỪNG NHÓM NGỮ NGHĨA (5 LỚP)",
                 color=TEXT_COLOR, fontsize=14, fontweight="bold", pad=15)
    ax.set_xticks(indices)
    ax.set_xticklabels(group_names, color=TEXT_COLOR, fontsize=10, rotation=18)
    ax.set_ylabel("Cường độ trung bình [0, 1]", color=TEXT_COLOR, fontsize=11)
    ax.tick_params(colors=TEXT_COLOR)
    ax.grid(axis="y", color="#30363d", alpha=0.5, linestyle="--")

    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")

    ax.legend(facecolor=PANEL_BG, edgecolor="#30363d", labelcolor=TEXT_COLOR,
              fontsize=10, loc="upper right")

    plt.tight_layout()
    for p in out_paths:
        plt.savefig(p, dpi=160, bbox_inches="tight", facecolor=DARK_BG)
        print(f"[ĐÃ LƯU] Biểu đồ kích hoạt ngữ nghĩa: {p.name}")
    plt.close(fig)


# ==============================================================================
# 5. ĐIỂM VÀO THỰC THI CHÍNH
# ==============================================================================

def main():
    print("=" * 65)
    print("  IE105 — TRỰC QUAN HÓA ĐẶC TRƯNG ẢNH 22x22 VÀ BẢN ĐỒ NHIỆT")
    print("=" * 65)

    X, y = load_data()
    print(f"[DỮ LIỆU] Tổng số mẫu: {len(X):,} | Kích thước ảnh: {X.shape[1:]}")

    # 1. Lưới so sánh 5 lớp
    out1 = [
        BASE_DIR / "malware_vs_benign_22x22.png",
        EVAL_DIR / "malware_vs_benign_22x22.png",
    ]
    plot_malware_vs_benign_grid(X, y, out1)

    # 2. Bản đồ vị trí Semantic và Difference Map
    out2 = [
        BASE_DIR / "semantic_layout_and_difference_map.png",
        EVAL_DIR / "semantic_layout_and_difference_map.png",
    ]
    plot_semantic_layout_and_difference(X, y, out2)

    # 3. So sánh cường độ kích hoạt theo nhóm
    out3 = [
        BASE_DIR / "semantic_group_activation_comparison.png",
        EVAL_DIR / "semantic_group_activation_comparison.png",
    ]
    plot_semantic_group_activation(X, y, out3)

    print("\n[HOÀN TẤT] Đã xuất đầy đủ 3 bộ biểu đồ trực quan hóa chất lượng cao!")


if __name__ == "__main__":
    main()
