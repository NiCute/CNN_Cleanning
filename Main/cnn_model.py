"""
cnn_model.py
============
Đồ án IE105 - Khai phá dữ liệu & Ứng dụng
Mô hình CNN phát hiện và phân loại mã độc Android từ bộ dữ liệu CICMalDroid2020.

Ý tưởng chính:
- Đầu vào: Ảnh grayscale 22x22 (được chuyển đổi từ 470 đặc trưng hành vi và zero-padding 14 pixel cuối).
- Đầu ra: Phân loại 5 lớp (1: Adware, 2: Banking, 3: SMS Malware, 4: Riskware, 5: Benign).

Thiết kế kiến trúc mạng MalwareCNN:
- Vì kích thước ảnh 22x22 tương đối nhỏ, mạng được thiết kế tinh gọn gồm 3 khối tích chập (Conv Blocks):
    + Block 1: Conv2d(1 -> 32, kernel 3x3) + BatchNorm + ReLU (kích thước sau khối: 20x20)
    + Block 2: Conv2d(32 -> 64, kernel 3x3) + BatchNorm + ReLU + MaxPool(2x2) + Dropout2d (còn 9x9)
    + Block 3: Conv2d(64 -> 128, kernel 3x3, pad=1) + BatchNorm + ReLU + Dropout2d (giữ nguyên 9x9)
- Thay vì làm phẳng (Flatten) ngay lập tức tạo ra quá nhiều tham số gây overfit, nhóm sử dụng
  Global Average Pooling (GAP) để nén về vector 128 chiều, sau đó đưa qua tầng Fully Connected
  512 chiều (Dropout 0.5) rồi dự đoán phân phối xác suất cho 5 lớp.
"""

import time
from pathlib import Path
import sys

# Cấu hình UTF-8 cho console Windows để in tiếng Việt có dấu không bị lỗi charmap
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ==============================================================================
# 1. CẤU HÌNH THỰC NGHIỆM (HYPERPARAMETERS & ENVIRONMENT)
# ==============================================================================

# Bộ dữ liệu CICMalDroid2020 gốc có nhãn từ 1 đến 5, khi huấn luyện ánh xạ về 0 đến 4
N_CLASSES = 5
CLASS_NAMES = ["Adware", "Banking", "SMS Malware", "Riskware", "Benign"]

BASE_DIR = Path(__file__).parent.resolve()

CFG = {
    "data_path": str(BASE_DIR / "images_22x22.npz"),
    "save_dir": str(BASE_DIR),
    "batch_size": 64,
    "epochs": 50,
    "lr": 1e-3,
    "weight_decay": 1e-4,  # L2 regularization giúp kiểm soát trọng số mô hình
    "patience": 10,        # Dừng sớm nếu sau 10 epoch validation loss không giảm
    "seed": 42,
    # Trên hệ điều hành Windows, đặt num_workers = 0 để tránh lỗi spawn subprocess của PyTorch
    "num_workers": 0,
}

# Tự động ưu tiên GPU CUDA nếu máy có card rời, ngược lại dùng CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int = 42):
    """Cố định seed ngẫu nhiên để kết quả huấn luyện có thể tái lập chính xác."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ==============================================================================
# 2. ĐỊNH NGHĨA KIẾN TRÚC MẠNG TÍCH CHẬP (MALWARE CNN)
# ==============================================================================

class MalwareCNN(nn.Module):
    """
    Mô hình tích chập phân loại ảnh ma trận đặc trưng kích thước 22x22 (1 kênh).
    - Input: Tensor kích thước (Batch_Size, 1, 22, 22)
    - Output: Logits chưa qua Softmax (Batch_Size, 5) để khớp với nn.CrossEntropyLoss
    """

    def __init__(self, n_classes: int = N_CLASSES, dropout: float = 0.25, dropout_fc: float = 0.5):
        super().__init__()

        # Khối 1: Trích xuất các đặc trưng cơ bản giữa các API/hành vi lân cận trong cùng nhóm ngữ nghĩa.
        # Kích thước qua khối: (N, 1, 22, 22) -> (N, 32, 20, 20)
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # Khối 2: Mở rộng số kênh lên 64 để học các tổ hợp đặc trưng phức tạp hơn.
        # MaxPool2d(2, 2) giảm kích thước từ 18x18 xuống 9x9 để mở rộng receptive field.
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(dropout),
        )

        # Khối 3: Tăng kênh lên 128 với padding=1 để bảo toàn kích thước không gian 9x9.
        # Vì ảnh đã tương đối nhỏ (9x9), việc không pooling tiếp giúp tránh mất mát vị trí của các nhóm API.
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
        )

        # Global Average Pooling: Gom trung bình mỗi feature map 9x9 thành 1 giá trị vô hướng.
        # Giúp giảm từ 128 * 9 * 9 = 10,368 tham số xuống còn 128, chống overfitting rất hiệu quả.
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Tầng phân loại (Classifier)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_fc),
            nn.Linear(512, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.gap(x)
        return self.classifier(x)


# ==============================================================================
# 3. NẠP DỮ LIỆU VÀ XỬ LÝ MẤT CÂN BẰNG LỚP (DATA LOADING & CLASS BALANCING)
# ==============================================================================

def encode_label(y: np.ndarray) -> torch.Tensor:
    """
    Chuyển nhãn về dạng số nguyên liên tục bắt đầu từ 0 để phù hợp với CrossEntropyLoss.
    Hỗ trợ cả trường hợp nhãn là số (1-5) hoặc chuỗi tên lớp.
    """
    unique = sorted(set(y.tolist()))
    mapping = {u: i for i, u in enumerate(unique)}
    if y.dtype.kind in ("U", "S", "O"):
        encoded = [mapping[v] for v in y]
    else:
        encoded = [mapping[int(v)] for v in y]
    return torch.tensor(encoded, dtype=torch.long)


def load_data(cfg: dict):
    """
    Đọc dữ liệu ảnh 22x22 đã chuẩn bị từ file .npz, tính toán trọng số cân bằng lớp
    và khởi tạo các DataLoader tương ứng cho Train / Validation / Test.
    """
    print(f"\n[DỮ LIỆU] Đang nạp tệp: {cfg['data_path']}")
    data = np.load(cfg["data_path"])

    X_train = torch.tensor(data["X_train"], dtype=torch.float32)
    X_val   = torch.tensor(data["X_val"],   dtype=torch.float32)
    X_test  = torch.tensor(data["X_test"],  dtype=torch.float32)

    # Ánh xạ nhãn sang định dạng 0-4
    y_train_raw = data["y_train"]
    unique_sorted = sorted(set(y_train_raw.tolist()))
    mapping_ref = {u: i for i, u in enumerate(unique_sorted)}
    y_train_t = encode_label(y_train_raw)
    y_val_t   = encode_label(data["y_val"])
    y_test_t  = encode_label(data["y_test"])

    n_total = len(y_train_t) + len(y_val_t) + len(y_test_t)
    print(f"  Tỷ lệ phân chia tập dữ liệu (70 / 15 / 15):")
    print(f"    - Tập Huấn luyện (Train) : {len(y_train_t):>5} mẫu ({len(y_train_t)/n_total*100:.1f}%)")
    print(f"    - Tập Kiểm định  (Val)   : {len(y_val_t):>5} mẫu ({len(y_val_t)/n_total*100:.1f}%)")
    print(f"    - Tập Kiểm thử   (Test)  : {len(y_test_t):>5} mẫu ({len(y_test_t)/n_total*100:.1f}%)")

    unique_lbl, counts = torch.unique(y_train_t, return_counts=True)
    print("\n  Phân phối các lớp trong tập Train:")
    for lbl, cnt in zip(unique_lbl.tolist(), counts.tolist()):
        c_name = CLASS_NAMES[lbl] if lbl < len(CLASS_NAMES) else f"Lớp_{lbl}"
        print(f"    - Lớp {lbl} ({c_name:<11}): {cnt:>4} mẫu ({cnt/len(y_train_t)*100:5.2f}%)")

    # Xử lý mất cân bằng lớp bằng trọng số nghịch đảo tần suất (Inverse Class Frequency):
    # Trọng số của lớp c = Tổng số mẫu / (Số lớp * Số mẫu của lớp c)
    class_weights = len(y_train_t) / (len(unique_lbl) * counts.float())
    print("\n  Trọng số điều chỉnh hàm mất mát (Class Weights):")
    for lbl, w in zip(unique_lbl.tolist(), class_weights.tolist()):
        c_name = CLASS_NAMES[lbl] if lbl < len(CLASS_NAMES) else f"Lớp_{lbl}"
        print(f"    - Lớp {lbl} ({c_name:<11}): {w:.4f}")

    train_ds = TensorDataset(X_train, y_train_t)
    val_ds   = TensorDataset(X_val,   y_val_t)
    test_ds  = TensorDataset(X_test,  y_test_t)

    # Khởi tạo DataLoader
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        pin_memory=(DEVICE.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["batch_size"] * 2,
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=(DEVICE.type == "cuda"),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg["batch_size"] * 2,
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=(DEVICE.type == "cuda"),
    )
    return train_loader, val_loader, test_loader, mapping_ref, class_weights


# ==============================================================================
# 4. CƠ CHẾ DỪNG SỚM & CÁC HÀM PHỤ TRỢ HUẤN LUYỆN
# ==============================================================================

class EarlyStopping:
    """
    Cơ chế dừng sớm (Early Stopping) theo dõi validation loss.
    Nếu sau một số epoch (patience) mà loss không cải thiện thì dừng huấn luyện
    để tránh mô hình học vẹt (overfit).
    """

    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.should_stop = False

    def step(self, val_loss: float):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Thực hiện một lượt lan truyền xuôi và lan truyền ngược qua toàn bộ tập Train."""
    model.train()
    total_loss = correct = total = 0
    for X_b, y_b in loader:
        X_b, y_b = X_b.to(device), y_b.to(device)

        # Xóa gradient tích lũy của batch trước
        optimizer.zero_grad()
        logits = model(X_b)
        loss = criterion(logits, y_b)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(y_b)
        preds = logits.argmax(dim=1)
        correct += (preds == y_b).sum().item()
        total += len(y_b)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Đánh giá mô hình ở chế độ kiểm thử (không cập nhật trọng số và tắt dropout)."""
    model.eval()
    total_loss = correct = total = 0
    all_preds, all_labels = [], []

    for X_b, y_b in loader:
        X_b, y_b = X_b.to(device), y_b.to(device)
        logits = model(X_b)
        loss = criterion(logits, y_b)
        preds = logits.argmax(dim=1)

        total_loss += loss.item() * len(y_b)
        correct += (preds == y_b).sum().item()
        total += len(y_b)
        all_preds.append(preds.cpu().numpy())
        all_labels.append(y_b.cpu().numpy())

    return (
        total_loss / total,
        correct / total,
        np.concatenate(all_preds),
        np.concatenate(all_labels),
    )


# ==============================================================================
# 5. VẼ ĐƯỜNG CONG HỌC TẬP (LEARNING CURVES)
# ==============================================================================

def plot_history(history: dict, save_dir: str):
    """Xuất đồ thị biểu diễn Loss và Accuracy qua các epoch để quan sát hiện tượng học của mạng."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor("#0d1117")
    ep = range(1, len(history["train_loss"]) + 1)

    val_loss_key = "val_loss" if "val_loss" in history else "test_loss"
    val_acc_key  = "val_acc" if "val_acc" in history else "test_acc"

    configs = [
        (axes[0], ("train_loss", val_loss_key), "Mất mát (Loss)", "Đường cong Mất mát (CrossEntropy Loss)"),
        (axes[1], ("train_acc",  val_acc_key),  "Độ chính xác (%)", "Đường cong Độ chính xác (Accuracy %)"),
    ]

    for ax, (tr_key, va_key), ylabel, title in configs:
        train_vals = history[tr_key]
        val_vals   = history[va_key]
        if "acc" in tr_key:
            train_vals = [v * 100 for v in train_vals]
            val_vals   = [v * 100 for v in val_vals]

        ax.plot(ep, train_vals, color="#00ff88", lw=2, label="Huấn luyện (Train)")
        ax.plot(ep, val_vals,   color="#ff6644", lw=2, ls="--", label="Kiểm định (Validation)")
        ax.set_title(title, color="white", fontsize=13, fontweight="bold")
        ax.set_xlabel("Epoch", color="white")
        ax.set_ylabel(ylabel, color="white")
        ax.set_facecolor("#1a1a2e")
        ax.tick_params(colors="white")
        ax.legend(facecolor="#1a1a2e", labelcolor="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333355")
        if "acc" in tr_key:
            ax.set_ylim([50, 101])

    plt.tight_layout()
    out = Path(save_dir) / "training_curve.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"[ĐỒ THỊ] Đã lưu biểu đồ huấn luyện tại: {out}")
    plt.close(fig)


# ==============================================================================
# 6. QUY TRÌNH HUẤN LUYỆN CHÍNH (MAIN PIPELINE)
# ==============================================================================

def main():
    set_seed(CFG["seed"])

    print("=" * 60)
    print(f"  Thiết bị xử lý : {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"  Tên GPU        : {torch.cuda.get_device_name(0)}")
    print("=" * 60)

    train_loader, val_loader, test_loader, label_mapping, class_weights = load_data(CFG)
    class_names = [
        CLASS_NAMES[i] if i < len(CLASS_NAMES) else f"Lớp_{k}"
        for i, k in enumerate(sorted(label_mapping.keys()))
    ]

    # Khởi tạo mô hình CNN
    model = MalwareCNN(n_classes=N_CLASSES).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[MÔ HÌNH] MalwareCNN (5 lớp)  |  Tổng số tham số học được: {n_params:,}")

    # Áp dụng hàm mất mát CrossEntropyLoss với trọng số cân bằng lớp
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))

    # Tối ưu AdamW kết hợp Cosine Annealing để giảm dần learning rate theo từng chu kỳ
    optimizer = optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG["epochs"])
    stopper   = EarlyStopping(patience=CFG["patience"])

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_acc = 0.0
    save_path = Path(CFG["save_dir"]) / "best_model.pth"

    print(f"\n[BẮT ĐẦU HUẤN LUYỆN] Tối đa {CFG['epochs']} epoch, batch={CFG['batch_size']}, lr={CFG['lr']}\n")

    for epoch in range(1, CFG["epochs"] + 1):
        t0 = time.time()

        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        va_loss, va_acc, _, _ = evaluate(model, val_loader, criterion, DEVICE)

        # Lưu ý phương pháp luận: Tuyệt đối không kiểm thử trên Test set trong vòng lặp epoch
        # để tránh rò rỉ dữ liệu (data leakage) và đảm bảo tính khách quan của thực nghiệm.
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)

        marker = ""
        # Chỉ lưu checkpoint tốt nhất dựa trên độ chính xác của tập Validation
        if va_acc > best_acc:
            best_acc = va_acc
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_val_acc": best_acc,
                "cfg": CFG,
                "class_weights": class_weights,
            }, save_path)
            marker = "  << Điểm kiểm tra tốt nhất (BEST)"

        print(
            f"Epoch {epoch:>3}/{CFG['epochs']}  "
            f"| Train loss={tr_loss:.4f} acc={tr_acc*100:.2f}%"
            f"  | Val loss={va_loss:.4f} acc={va_acc*100:.2f}%"
            f"  | {time.time()-t0:.1f}s{marker}"
        )

        # Kiểm tra điều kiện dừng sớm
        stopper.step(va_loss)
        if stopper.should_stop:
            print(f"\n[DỪNG SỚM] Kích hoạt tại epoch {epoch} (patience={CFG['patience']}) để tránh overfit.")
            break

    # ==============================================================================
    # 7. ĐÁNH GIÁ CUỐI CÙNG TRÊN TẬP KIỂM THỬ ĐỘC LẬP (FINAL TEST SET EVALUATION)
    # ==============================================================================
    print(f"\n{'='*60}")
    print(f"[ĐÁNH GIÁ] Đang nạp lại checkpoint tốt nhất: {save_path.name}")
    ckpt = torch.load(save_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(ckpt["model_state"])

    print("\n--- KẾT QUẢ TRÊN TẬP VALIDATION ---")
    va_loss, va_acc, _, _ = evaluate(model, val_loader, criterion, DEVICE)
    print(f"  Validation Loss     : {va_loss:.4f}")
    print(f"  Validation Accuracy : {va_acc*100:.4f}%")

    print("\n--- KẾT QUẢ CHÍNH THỨC TRÊN TẬP TEST (Đánh giá độc lập 1 lần) ---")
    te_loss, te_acc, all_preds, all_labels = evaluate(model, test_loader, criterion, DEVICE)
    print(f"  Test Loss         : {te_loss:.4f}")
    print(f"  Test Accuracy     : {accuracy_score(all_labels, all_preds)*100:.4f}%")
    print(f"  Macro Precision   : {precision_score(all_labels, all_preds, average='macro', zero_division=0):.4f}")
    print(f"  Macro Recall      : {recall_score(all_labels, all_preds, average='macro', zero_division=0):.4f}")
    print(f"  Macro F1-Score    : {f1_score(all_labels, all_preds, average='macro', zero_division=0):.4f}")
    print(f"  Weighted F1-Score : {f1_score(all_labels, all_preds, average='weighted', zero_division=0):.4f}")

    print("\n[BẢNG BÁO CÁO PHÂN LOẠI CHI TIẾT]")
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))

    cm = confusion_matrix(all_labels, all_preds)
    print("[MA TRẬN NHẦM LẪN (CONFUSION MATRIX)]")
    header = "  " + "".join(f"{c:>14}" for c in class_names)
    print(header)
    for i, row in enumerate(cm):
        row_str = "".join(f"{v:>14}" for v in row)
        print(f"  {class_names[i]:<14}{row_str}")

    # Lưu lại lịch sử huấn luyện để vẽ đồ thị so sánh
    hist_path = Path(CFG["save_dir"]) / "training_history.npz"
    np.savez(hist_path, **history)
    print(f"\n[LƯU TRỮ] Lịch sử huấn luyện đã lưu: {hist_path.name}")
    print(f"[LƯU TRỮ] Mô hình tốt nhất đã lưu  : {save_path.name}")
    print(f"[HOÀN TẤT] Best Val Accuracy = {best_acc*100:.4f}% | Final Test Accuracy = {te_acc*100:.4f}%")

    plot_history(history, CFG["save_dir"])


if __name__ == "__main__":
    main()
