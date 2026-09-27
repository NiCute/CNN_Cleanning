"""
feature_to_image.py
===================
Pipeline chuyen doi vector 470 features (CICMalDroid2020)
thanh anh grayscale 22x22 de dua vao CNN.

Chien luoc:
  - Kich thuoc anh  : 22x22 = 484 pixel (pad 14 zeros o cuoi)
  - Thu tu feature  : nhom theo semantic (FileSystem, Network, Privacy, ...)
  - Padding          : zero-padding 14 pixel cuoi
  - Chuan hoa       : MinMaxScaler [0, 1] fit tren train set
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# 1. DINH NGHIA NHOM SEMANTIC
# ─────────────────────────────────────────────────────────────────────────────

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
N_PIXELS   = IMAGE_SIZE[0] * IMAGE_SIZE[1]   # 484 (22x22)
# N_FEATURES va N_PAD duoc tinh dong trong FeatureToImagePipeline.__init__


def classify_feature(col_name: str) -> str:
    upper = col_name.upper()
    for group, keywords in SEMANTIC_KEYWORDS.items():
        if any(kw in upper for kw in keywords):
            return group
    return "Other_API"


def get_semantic_ordering(feature_names):
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


# ─────────────────────────────────────────────────────────────────────────────
# 2. PIPELINE CHINH
# ─────────────────────────────────────────────────────────────────────────────

class FeatureToImagePipeline:
    """
    Pipeline: vector 470 features -> anh grayscale 22x22.

    Cac buoc:
      1. fit_transform (chi goi tren train set)
         a. Chuan hoa MinMax [0, 1] - fit tren train
         b. Sap xep lai theo semantic ordering
         c. Pad zeros + reshape -> (N, 1, 22, 22)
      2. transform (goi tren val/test set)
    """

    def __init__(self, feature_names):
        self.feature_names = feature_names
        self.n_features = len(feature_names)   # tu dong lay so feature thuc te
        self.n_pad      = N_PIXELS - self.n_features
        if self.n_features > N_PIXELS:
            raise ValueError(
                f"So feature ({self.n_features}) > so pixel ({N_PIXELS}=22x22). "
                f"Hay tang IMAGE_SIZE."
            )
        self.scaler = MinMaxScaler()
        self.ordered_indices, self.group_info = get_semantic_ordering(feature_names)
        self.is_fitted = False

        print("=" * 55)
        print("Semantic Groups (thu tu trong anh 22x22):")
        print("=" * 55)
        total = 0
        for g in SEMANTIC_GROUPS_ORDER:
            info = self.group_info[g]
            bar = "#" * min(info["count"], 40)
            print(f"  {g:<20} {info['count']:>3} features  [{bar}]")
            total += info["count"]
        print(f"  {'TOTAL':<20} {total:>3} features")
        print(f"  {'PADDING (zeros)':<20}  {self.n_pad:>2} pixels")
        print(f"  {'IMAGE SIZE':<20} {IMAGE_SIZE[0]}x{IMAGE_SIZE[1]} = {N_PIXELS} pixels")
        print("=" * 55)

    def _normalize_and_reorder(self, X, fit):
        if fit:
            X_norm = self.scaler.fit_transform(X)
            self.is_fitted = True
        else:
            if not self.is_fitted:
                raise RuntimeError("Phai goi fit_transform() truoc transform()!")
            X_norm = self.scaler.transform(X)
        X_reordered = X_norm[:, self.ordered_indices]
        return X_reordered

    def _to_image_tensor(self, X_reordered):
        N = len(X_reordered)
        padded = np.zeros((N, N_PIXELS), dtype=np.float32)
        padded[:, :self.n_features] = X_reordered   # chi dien dung so feature thuc te
        # (N, 1, H, W) - channel-first cho PyTorch
        images = padded.reshape(N, 1, IMAGE_SIZE[0], IMAGE_SIZE[1])
        return images

    def fit_transform(self, X):
        """CHI goi tren TRAIN SET."""
        X_reordered = self._normalize_and_reorder(X, fit=True)
        return self._to_image_tensor(X_reordered)

    def transform(self, X):
        """Goi tren VAL / TEST SET."""
        X_reordered = self._normalize_and_reorder(X, fit=False)
        return self._to_image_tensor(X_reordered)

    def get_group_layout(self):
        layout = {}
        for g in SEMANTIC_GROUPS_ORDER:
            info = self.group_info[g]
            start_pixel = info["offset"]
            end_pixel   = info["offset"] + info["count"] - 1
            start_row   = start_pixel // IMAGE_SIZE[1]
            end_row     = end_pixel   // IMAGE_SIZE[1]
            layout[g] = {
                "count"      : info["count"],
                "pixel_start": start_pixel,
                "pixel_end"  : end_pixel,
                "row_start"  : start_row,
                "row_end"    : end_row,
            }
        return layout


# ─────────────────────────────────────────────────────────────────────────────
# 3. HAM TIEN ICH: VISUALIZE
# ─────────────────────────────────────────────────────────────────────────────

def visualize_sample(image_tensor, label, group_info, save_path=None):
    """
    Hien thi 1 anh 22x22 kem duong phan cach cac nhom semantic.
    image_tensor: shape (1, 22, 22) hoac (22, 22)
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.colors import LinearSegmentedColormap
    except ImportError:
        print("Can cai matplotlib: pip install matplotlib")
        return

    img = image_tensor.squeeze()   # -> (22, 22)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor("#0d1117")

    # -- Anh 22x22 --
    ax = axes[0]
    cmap = LinearSegmentedColormap.from_list("malware", ["#0d1117", "#00ff88"], N=256)
    im = ax.imshow(img, cmap=cmap, vmin=0, vmax=1, aspect="equal")
    label_str = "Malware" if label == 1 else "Benign"
    ax.set_title(f"Feature -> Image 22x22\nLabel: {label_str}",
                 fontsize=12, fontweight="bold", color="white")
    plt.colorbar(im, ax=ax, fraction=0.046)

    # Ve vung mau theo nhom
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

    # -- Pixel value distribution --
    ax2 = axes[1]
    ax2.hist(img.flatten(), bins=30, color="#00ff88",
             edgecolor="#0d1117", alpha=0.85)
    ax2.set_title("Phan phoi gia tri pixel", fontsize=12,
                  fontweight="bold", color="white")
    ax2.set_xlabel("Gia tri (sau MinMax normalize)", color="white")
    ax2.set_ylabel("So pixel", color="white")

    for a in axes:
        a.set_facecolor("#1a1a2e")
        a.tick_params(colors="white")
        for spine in a.spines.values():
            spine.set_edgecolor("#333355")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"Da luu: {save_path}")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# 4. MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    DATA_PATH = Path("d:/IE105/Main/471_extracted features.csv")

    print(f"\nDoc du lieu: {DATA_PATH.name}")
    df = pd.read_csv(DATA_PATH)
    print(f"Shape: {df.shape}")

    feature_cols = df.columns[:-1].tolist()   # 470 features
    label_col    = df.columns[-1]             # 'Class'

    X = df[feature_cols].values.astype(np.float32)
    y = df[label_col].values

    print(f"\nLabel distribution:")
    unique, counts = np.unique(y, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"  {u}: {c} samples")

    # Train / Val / Test split: 70 / 15 / 15
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.15/0.85, random_state=42, stratify=y_temp
    )
    print(f"\nSplit 70/15/15: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

    # Khoi tao pipeline
    pipeline = FeatureToImagePipeline(feature_cols)

    # Chuyen doi
    print("\nChuyen doi train set...")
    X_train_img = pipeline.fit_transform(X_train)   # fit TREN TRAIN

    print("Chuyen doi val set...")
    X_val_img = pipeline.transform(X_val)

    print("Chuyen doi test set...")
    X_test_img = pipeline.transform(X_test)

    # Ket qua
    print(f"\nKET QUA:")
    print(f"  X_train_img shape : {X_train_img.shape}")
    print(f"  X_val_img   shape : {X_val_img.shape}")
    print(f"  X_test_img  shape : {X_test_img.shape}")
    print(f"  dtype             : {X_train_img.dtype}")
    print(f"  pixel range       : [{X_train_img.min():.3f}, {X_train_img.max():.3f}]")

    # Layout nhom trong anh
    print("\nLayout cac nhom trong anh 22x22:")
    layout = pipeline.get_group_layout()
    for g, info in layout.items():
        if info["count"] > 0:
            print(f"  {g:<20} pixel [{info['pixel_start']:>3}-{info['pixel_end']:>3}]"
                  f"  row [{info['row_start']:>2}-{info['row_end']:>2}]")

    # Luu ket qua
    out_path = Path("d:/IE105/Main/images_22x22.npz")
    np.savez_compressed(
        out_path,
        X_train=X_train_img, y_train=y_train,
        X_val=X_val_img,     y_val=y_val,
        X_test=X_test_img,   y_test=y_test,
    )
    print(f"\nDa luu: {out_path}")
    print(f"Load lai: data = np.load('{out_path}')")

    # Visualize 1 mau
    print("\nVisualize mau...")
    group_info_for_vis = {
        g: {"offset": pipeline.group_info[g]["offset"],
            "count":  pipeline.group_info[g]["count"]}
        for g in SEMANTIC_GROUPS_ORDER
    }
    unique_labels = np.unique(y_train)
    for target_label in unique_labels[:2]:
        idx = np.where(y_train == target_label)[0][0]
        is_malware = str(target_label).lower() not in ["benign", "0", "goodware"]
        visualize_sample(
            X_train_img[idx],
            label=1 if is_malware else 0,
            group_info=group_info_for_vis,
            save_path=f"d:/IE105/Main/sample_{target_label}.png",
        )
