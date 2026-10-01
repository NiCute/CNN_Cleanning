"""
feature_to_image.py
===================
Đồ án IE105 - Khai phá dữ liệu & Ứng dụng
Quy trình chuyển đổi vector đặc trưng (470 features) thành ảnh grayscale 22x22 cho mạng CNN.

Phương pháp luận & Thiết kế:
1. Sắp xếp theo nhóm ngữ nghĩa (Semantic Grouping):
   - Mạng CNN phát huy tối đa sức mạnh khi dữ liệu có tính tương quan không gian cục bộ (Spatial Locality).
   - Nếu giữ nguyên thứ tự ngẫu nhiên trong file CSV, kernel tích chập (3x3) sẽ quét qua các đặc trưng
     rời rạc không liên quan.
   - Nhóm phân tích và gom 470 đặc trưng thành 9 nhóm hành vi chính:
       + FileSystem        : Các thao tác truy cập tệp tin, phân quyền, đường ống hệ thống...
       + Network           : Kết nối mạng, Socket, HTTP, DNS, truyền nhận dữ liệu qua mạng...
       + Process           : Quản lý luồng, tạo tiến trình, kill tiến trình, anti-debug...
       + Device            : Tương tác phần cứng (Camera, Bluetooth, Wifi, Pin, Rung, Cảm biến)...
       + Android_Component : Intent, Broadcast Receiver, Service, Content Provider...
       + Privacy           : Quyền hạn nhạy cảm, định vị GPS, đọc danh bạ, lấy mã IMEI/IMSI...
       + App_Management    : Cài đặt ứng dụng, nạp file DEX động (DexClassLoader), Reflection...
       + Crypto            : Mã hóa dữ liệu, giải mã, băm mật mã (MD5, SHA, AES, RSA, SSL)...
       + Other_API         : Các API và lời gọi hệ thống bổ trợ khác.
   - Việc gom nhóm giúp các đặc trưng cùng bản chất nằm liền kề nhau trên ma trận 2D.

2. Kích thước ảnh & Đệm số 0 (Zero-Padding):
   - Kích thước ảnh vuông nhỏ nhất bao phủ được 470 đặc trưng là 22 x 22 = 484 pixels.
   - 14 pixel còn lại ở cuối ma trận được gán giá trị 0 (Zero-padding).

3. Chuẩn hóa dữ liệu (Normalization):
   - Sử dụng MinMaxScaler để đưa giá trị mọi đặc trưng về đoạn [0, 1] tương ứng với mức xám của điểm ảnh.
   - Scaler chỉ được fit trên tập Huấn luyện (Train), sau đó lưu thành `scaler.joblib` để tái sử dụng
     cho tập Val, Test và khi dự đoán mẫu thực tế, tránh rò rỉ dữ liệu (Data Leakage).
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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import joblib

# ==============================================================================
# 1. ĐỊNH NGHĨA QUY TẮC PHÂN NHÓM NGỮ NGHĨA (SEMANTIC GROUPING)
# ==============================================================================

# Thứ tự bố trí các nhóm ngữ nghĩa trên ma trận ảnh từ trên xuống dưới
SEMANTIC_GROUPS_ORDER = [
    "FileSystem",
    "Network",
    "Process",
    "Device",
    "Android_Component",
    "Privacy",
    "App_Management",
    "Crypto",
    "Other_API",
]

# Từ khóa định danh cho từng nhóm hành vi dựa trên tên lời gọi API / System Call / Quyền
SEMANTIC_KEYWORDS = {
    "FileSystem": [
        "FS_ACCESS", "FS_PIPE", "CREATE_FOLDER", "FTRUNCATE",
        "INOTIFY", "EPOLL", "CHMOD", "MMAP", "_LLSEEK", "FDATASYNC",
        "FSTAT", "FSTATFS", "FSYNC", "FLOCK", "FCNTL", "CHDIR",
        "CHOWN", "FACCESSAT", "FCHOWN", "ACCESS",
    ],
    "Network": [
        "NETWORK", "SOCKET", "HTTP", "URL", "INET", "DNS",
        "CONNECT", "BIND", "ACCEPT", "DHCP", "MOBILE_IFACE",
        "GETDATASTATE", "GETMOBILEDATA",
    ],
    "Privacy": [
        "PERMISSION", "ACCESS_PERSONAL", "ACCOUNT", "CONTACT",
        "LOCATION", "CAMERA", "MIC", "LINE1NUMBER", "ICCSERIAL",
        "IMEI", "IMSI", "GEOC",
    ],
    "Crypto": [
        "CRYPTO", "CIPHER", "ENCRYPT", "HASH", "SIGN",
        "KEY", "SSL", "TLS", "SECRET",
    ],
    "Process": [
        "PROCESS", "THREAD", "EXECUTE", "FORK", "SPAWN",
        "SCHED", "CLOCK", "EXIT", "KILL", "TERMINATE",
        "ANTI_DEBUG", "CLONE", "EXECVE", "CAPSET", "FUTEX",
        "EVENTFD", "CREATE_THREAD", "CREATE_PROCESS",
    ],
    "Device": [
        "DEVICE", "PHONE", "SMS", "CALL", "AUDIO", "DISPLAY",
        "SENSOR", "BATTERY", "GPS", "WIFI", "BLUETOOTH",
        "VIBRAT", "FLASHLIGHT", "CELLINFO", "NEIGHBORINGCELL",
        "ALTER_PHONE",
    ],
    "Android_Component": [
        "INTENT", "BROADCAST", "RECEIVER", "SERVICE", "ACTIVITY",
        "CONTENT", "PROVIDER", "REGISTER", "UNREGIST",
        "NOTIFICATION", "TOAST", "SYNC", "PANEL",
    ],
    "App_Management": [
        "PACKAGE", "INSTALL", "APP", "APK", "CLASSLOAD",
        "REFLECT", "PKGUSAGE", "COMPONENT_ENABLED",
        "GETCOMPONENT",
    ],
}

IMAGE_SIZE = (22, 22)
N_PIXELS = IMAGE_SIZE[0] * IMAGE_SIZE[1]  # 484 pixels


def classify_feature(col_name: str) -> str:
    """Xác định nhóm ngữ nghĩa của một đặc trưng dựa vào từ khóa trong tên cột."""
    upper = col_name.upper()
    for group, keywords in SEMANTIC_KEYWORDS.items():
        if any(kw in upper for kw in keywords):
            return group
    return "Other_API"


def get_semantic_ordering(feature_names: list):
    """
    Sắp xếp lại thứ tự chỉ số của các đặc trưng sao cho các đặc trưng cùng nhóm
    sẽ được xếp đứng cạnh nhau liên tục.
    """
    groups = {g: [] for g in SEMANTIC_GROUPS_ORDER}
    for idx, name in enumerate(feature_names):
        g = classify_feature(name)
        groups[g].append(idx)

    ordered_indices = []
    group_info = {}
    offset = 0
    for g in SEMANTIC_GROUPS_ORDER:
        idxs = groups[g]
        group_info[g] = {"offset": offset, "count": len(idxs), "indices": idxs}
        ordered_indices.extend(idxs)
        offset += len(idxs)

    return ordered_indices, group_info


# ==============================================================================
# 2. ĐƯỜNG ỐNG BIẾN ĐỔI (FEATURE TO IMAGE PIPELINE)
# ==============================================================================

class FeatureToImagePipeline:
    """
    Đóng gói quy trình biến đổi từ vector 470 chiều thành tensor ảnh (N, 1, 22, 22).
    - fit_transform: Chuẩn hóa và tạo ảnh cho tập Train (đồng thời fit MinMaxScaler).
    - transform: Sử dụng MinMaxScaler đã fit để biến đổi tập Validation / Test.
    """

    def __init__(self, feature_names: list):
        self.feature_names = feature_names
        self.n_features = len(feature_names)
        self.n_pad = N_PIXELS - self.n_features
        if self.n_features > N_PIXELS:
            raise ValueError(
                f"Số lượng đặc trưng ({self.n_features}) vượt quá số pixel ({N_PIXELS}=22x22). "
                f"Vui lòng tăng kích thước ảnh IMAGE_SIZE."
            )
        self.scaler = MinMaxScaler()
        self.ordered_indices, self.group_info = get_semantic_ordering(feature_names)
        self.is_fitted = False

        print("=" * 55)
        print("Thống kê số lượng đặc trưng theo từng nhóm ngữ nghĩa:")
        print("=" * 55)
        total = 0
        for g in SEMANTIC_GROUPS_ORDER:
            info = self.group_info[g]
            bar = "#" * min(info["count"], 40)
            print(f"  {g:<20} {info['count']:>3} đặc trưng  [{bar}]")
            total += info["count"]
        print(f"  {'TỔNG CỘNG':<20} {total:>3} đặc trưng")
        print(f"  {'ĐỆM SỐ 0 (PADDING)':<20}  {self.n_pad:>2} pixel")
        print(f"  {'KÍCH THƯỚC MA TRẬN':<20} {IMAGE_SIZE[0]}x{IMAGE_SIZE[1]} = {N_PIXELS} pixel")
        print("=" * 55)

    def _normalize_and_reorder(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        """Chuẩn hóa dữ liệu về [0, 1] và hoán đổi vị trí các cột theo thứ tự ngữ nghĩa."""
        if fit:
            X_norm = self.scaler.fit_transform(X)
            self.is_fitted = True
        else:
            if not self.is_fitted:
                raise RuntimeError("Cần gọi fit_transform() trên tập Train trước khi transform()!")
            X_norm = self.scaler.transform(X)

        # Hoán đổi cột theo thứ tự đã gom nhóm
        X_reordered = X_norm[:, self.ordered_indices]
        return X_reordered

    def _to_image_tensor(self, X_reordered: np.ndarray) -> np.ndarray:
        """Đệm thêm 14 pixel số 0 và reshape ma trận thành dạng (N, 1, 22, 22)."""
        N = len(X_reordered)
        padded = np.zeros((N, N_PIXELS), dtype=np.float32)
        padded[:, :self.n_features] = X_reordered
        # Định dạng kênh đầu tiên (channel-first) phù hợp với PyTorch CNN
        images = padded.reshape(N, 1, IMAGE_SIZE[0], IMAGE_SIZE[1])
        return images

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Chỉ gọi trên tập Huấn luyện (Train set)."""
        X_reordered = self._normalize_and_reorder(X, fit=True)
        return self._to_image_tensor(X_reordered)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Gọi trên tập Validation, Test hoặc mẫu thực tế mới."""
        X_reordered = self._normalize_and_reorder(X, fit=False)
        return self._to_image_tensor(X_reordered)

    def get_group_layout(self) -> dict:
        """Trả về tọa độ điểm ảnh và phạm vi hàng của từng nhóm ngữ nghĩa trên ảnh 22x22."""
        layout = {}
        for g in SEMANTIC_GROUPS_ORDER:
            info = self.group_info[g]
            start_pixel = info["offset"]
            end_pixel   = info["offset"] + info["count"] - 1
            start_row   = start_pixel // IMAGE_SIZE[1]
            end_row     = end_pixel   // IMAGE_SIZE[1]
            layout[g] = {
                "count": info["count"],
                "pixel_start": start_pixel,
                "pixel_end": end_pixel,
                "row_start": start_row,
                "row_end": end_row,
            }
        return layout


# ==============================================================================
# 3. TRỰC QUAN HÓA MẪU ẢNH (SAMPLE VISUALIZATION)
# ==============================================================================

def visualize_sample(image_tensor: np.ndarray, label: int, group_info: dict, save_path: str = None):
    """
    Vẽ ảnh 22x22 của 1 mẫu cụ thể kèm theo ranh giới tô màu của các nhóm đặc trưng
    và phân phối mức xám điểm ảnh.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.colors import LinearSegmentedColormap
    except ImportError:
        print("[LƯU Ý] Cần cài đặt matplotlib để hiển thị ảnh: pip install matplotlib")
        return

    img = image_tensor.squeeze()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor("#0d1117")

    # Bản đồ nhiệt điểm ảnh
    ax = axes[0]
    cmap = LinearSegmentedColormap.from_list("malware", ["#0d1117", "#00ff88"], N=256)
    im = ax.imshow(img, cmap=cmap, vmin=0, vmax=1, aspect="equal")
    label_str = "Mã độc (Malware)" if label == 1 else "An toàn (Benign)"
    ax.set_title(f"Ma trận ảnh 22x22\nNhãn mẫu: {label_str}",
                 fontsize=12, fontweight="bold", color="white")
    plt.colorbar(im, ax=ax, fraction=0.046)

    # Tô màu phân biệt vùng của từng nhóm ngữ nghĩa
    colors = ["#ff4444", "#ff8800", "#ffdd00", "#44ff44",
              "#00ffff", "#4488ff", "#cc44ff", "#ff44cc", "#aaaaaa"]
    patches = []
    for i, (g, info) in enumerate(group_info.items()):
        count = info["count"]
        if count == 0:
            continue
        offset = info["offset"]
        end    = offset + count
        for px in range(offset, min(end, N_PIXELS)):
            r, c = divmod(px, IMAGE_SIZE[1])
            rect = mpatches.FancyBboxPatch(
                (c - 0.5, r - 0.5), 1, 1,
                boxstyle="square,pad=0",
                linewidth=0,
                facecolor=colors[i % len(colors)],
                alpha=0.20,
            )
            ax.add_patch(rect)
        patches.append(mpatches.Patch(
            color=colors[i % len(colors)], alpha=0.8,
            label=f"{g} ({count})")
        )
    ax.legend(handles=patches, loc="lower right",
              fontsize=6.5, framealpha=0.85, ncol=1,
              facecolor="#1a1a2e", labelcolor="white")

    # Biểu đồ tần suất mức xám
    ax2 = axes[1]
    ax2.hist(img.flatten(), bins=30, color="#00ff88",
             edgecolor="#0d1117", alpha=0.85)
    ax2.set_title("Phân phối mức xám điểm ảnh (sau MinMax)", fontsize=12,
                  fontweight="bold", color="white")
    ax2.set_xlabel("Giá trị điểm ảnh [0, 1]", color="white")
    ax2.set_ylabel("Số lượng pixel", color="white")

    for a in axes:
        a.set_facecolor("#1a1a2e")
        a.tick_params(colors="white")
        for spine in a.spines.values():
            spine.set_edgecolor("#333355")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"[ĐÃ LƯU] Ảnh mẫu trực quan tại: {save_path}")
    plt.close(fig)


# ==============================================================================
# 4. THỰC THI CHUYỂN ĐỔI BỘ DỮ LIỆU GỐC (DATASET TRANSFORMATION SCRIPT)
# ==============================================================================

if __name__ == "__main__":
    BASE_DIR = Path(__file__).parent.resolve()
    DATA_PATH = BASE_DIR / "471_extracted features.csv"

    print(f"\n[KHỞI TẠO] Đang đọc tệp dữ liệu: {DATA_PATH.name}")
    df = pd.read_csv(DATA_PATH)
    print(f"  Kích thước bảng dữ liệu: {df.shape[0]} dòng x {df.shape[1]} cột")

    feature_cols = df.columns[:-1].tolist()   # 470 cột đặc trưng
    label_col    = df.columns[-1]             # Cột 'Class'

    X = df[feature_cols].values.astype(np.float32)
    y = df[label_col].values

    print("\n[PHÂN PHỐI NHÃN GỐC]")
    unique, counts = np.unique(y, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"  Lớp {u}: {c:>5} mẫu ({c/len(y)*100:5.2f}%)")

    # Phân chia dữ liệu theo tỷ lệ chuẩn 70% Train - 15% Validation - 15% Test có bảo toàn tỷ lệ nhãn (Stratified)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.15/0.85, random_state=42, stratify=y_temp
    )
    print(f"\n[PHÂN CHIA TẬP DỮ LIỆU 70 / 15 / 15]")
    print(f"  - Huấn luyện (Train) : {len(X_train)} mẫu")
    print(f"  - Kiểm định  (Val)   : {len(X_val)} mẫu")
    print(f"  - Kiểm thử   (Test)  : {len(X_test)} mẫu")

    # Khởi tạo pipeline chuyển đổi sang ảnh 22x22
    pipeline = FeatureToImagePipeline(feature_cols)

    # Thực hiện biến đổi
    print("\n[CHUYỂN ĐỔI] Đang xử lý tập Train...")
    X_train_img = pipeline.fit_transform(X_train)

    # Lưu lại scaler đã fit để dùng khi suy luận
    scaler_path = BASE_DIR / "scaler.joblib"
    joblib.dump(pipeline.scaler, scaler_path)
    print(f"[LƯU TRỮ] Đã lưu bộ chuẩn hóa MinMaxScaler tại: {scaler_path.name}")

    print("[CHUYỂN ĐỔI] Đang xử lý tập Validation...")
    X_val_img = pipeline.transform(X_val)

    print("[CHUYỂN ĐỔI] Đang xử lý tập Test...")
    X_test_img = pipeline.transform(X_test)

    print("\n[KẾT QUẢ XUẤT RA]")
    print(f"  Kích thước tập Train: {X_train_img.shape}")
    print(f"  Kích thước tập Val  : {X_val_img.shape}")
    print(f"  Kích thước tập Test : {X_test_img.shape}")
    print(f"  Khoảng giá trị pixel: [{X_train_img.min():.3f}, {X_train_img.max():.3f}]")

    print("\n[BỐ TRÍ CÁC NHÓM NGỮ NGHĨA TRÊN ẢNH 22x22]")
    layout = pipeline.get_group_layout()
    for g, info in layout.items():
        if info["count"] > 0:
            print(f"  {g:<20} pixel [{info['pixel_start']:>3} - {info['pixel_end']:>3}]"
                  f"  -> Hàng [{info['row_start']:>2} - {info['row_end']:>2}]")

    # Lưu toàn bộ dữ liệu ảnh nén dưới dạng .npz
    out_path = BASE_DIR / "images_22x22.npz"
    np.savez_compressed(
        out_path,
        X_train=X_train_img, y_train=y_train,
        X_val=X_val_img,     y_val=y_val,
        X_test=X_test_img,   y_test=y_test,
    )
    print(f"\n[LƯU FILE NÉN] Đã lưu dữ liệu ảnh hoàn tất: {out_path.name}")

    # Vẽ mẫu 2 ảnh đại diện đầu tiên để kiểm tra trực quan
    print("\n[TRỰC QUAN HÓA] Đang kết xuất ảnh mẫu kiểm tra...")
    group_info_for_vis = {
        g: {"offset": pipeline.group_info[g]["offset"],
            "count":  pipeline.group_info[g]["count"]}
        for g in SEMANTIC_GROUPS_ORDER
    }
    unique_labels = np.unique(y_train)
    for target_label in unique_labels[:2]:
        idx = np.where(y_train == target_label)[0][0]
        is_malware = str(target_label).lower() not in ["benign", "0", "goodware", "5"]
        visualize_sample(
            X_train_img[idx],
            label=1 if is_malware else 0,
            group_info=group_info_for_vis,
            save_path=str(BASE_DIR / f"sample_{target_label}.png"),
        )
