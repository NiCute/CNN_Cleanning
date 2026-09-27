"""
cnn_model.py
============
Buoc 3: Xay dung va Huan luyen mo hinh CNN
- Input : anh grayscale 22x22 (tu images_22x22.npz)
- Output: phan loai 5 lop (CICMalDroid2020: 1=Adware, 2=Banking, 3=SMS, 4=Riskware, 5=Benign)

Kien truc CNN:
  Block1: Conv2d(1->32, 3x3) -> BN -> ReLU
  Block2: Conv2d(32->64, 3x3) -> BN -> ReLU -> MaxPool(2x2) -> Dropout2d
  Block3: Conv2d(64->128, 3x3, pad=1) -> BN -> ReLU -> Dropout2d
  -> Global Average Pooling
  -> FC(512) -> BN -> ReLU -> Dropout
  -> FC(5)  [CrossEntropyLoss, 5-class]
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
    roc_auc_score,
)
from pathlib import Path
import time

# ─────────────────────────────────────────────────────────────────────────────
# 0. CONFIG
# ─────────────────────────────────────────────────────────────────────────────

N_CLASSES = 5   # CICMalDroid2020: 5 lop (label goc 1-5, encode thanh 0-4)

CFG = {
    "data_path"   : "d:/IE105/Main/images_22x22.npz",
    "save_dir"    : "d:/IE105/Main",
    "batch_size"  : 64,
    "epochs"      : 50,
    "lr"          : 1e-3,
    "weight_decay": 1e-4,
    "patience"    : 10,          # Early stopping
    "seed"        : 42,
    "num_workers" : 0,           # Windows: 0 de tranh loi multiprocessing
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ─────────────────────────────────────────────────────────────────────────────
# 1. KIEN TRUC CNN
# ─────────────────────────────────────────────────────────────────────────────

class MalwareCNN(nn.Module):
    """
    CNN phan loai anh 22x22 grayscale - 5 lop.
    Input : (N, 1, 22, 22)
    Output: (N, n_classes)  logits cho CrossEntropyLoss
    """

    def __init__(self, n_classes=N_CLASSES, dropout=0.25, dropout_fc=0.5):
        super().__init__()

        # (N,1,22,22) -> (N,32,20,20)
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # (N,32,20,20) -> (N,64,18,18) -> MaxPool -> (N,64,9,9)
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout),
        )

        # (N,64,9,9) -> (N,128,9,9) giu nguyen voi padding=1
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
        )

        # Global Average Pooling: (N,128,9,9) -> (N,128,1,1)
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_fc),
            nn.Linear(512, n_classes),
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.gap(x)
        return self.classifier(x)   # (N, n_classes)


# ─────────────────────────────────────────────────────────────────────────────
# 2. LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────

def encode_label(y):
    """
    Chuyen label ve 0-based integer cho CrossEntropyLoss.
    CICMalDroid2020: label goc [1,2,3,4,5] -> [0,1,2,3,4]
    """
    unique = sorted(set(y.tolist()))
    print(f"  Labels goc     : {unique}")
    mapping = {u: i for i, u in enumerate(unique)}
    print(f"  Label mapping  : {mapping}")
    if y.dtype.kind in ("U", "S", "O"):   # string
        encoded = [mapping[v] for v in y]
    else:
        encoded = [mapping[int(v)] for v in y]
    return torch.tensor(encoded, dtype=torch.long)


def load_data(cfg):
    print(f"\n[DATA] Doc: {cfg['data_path']}")
    data = np.load(cfg["data_path"])

    X_train = torch.tensor(data["X_train"], dtype=torch.float32)
    X_val   = torch.tensor(data["X_val"],   dtype=torch.float32)
    X_test  = torch.tensor(data["X_test"],  dtype=torch.float32)

    print("[DATA] Ma hoa label...")
    y_train_raw = data["y_train"]
    unique_sorted = sorted(set(y_train_raw.tolist()))
    mapping_ref = {u: i for i, u in enumerate(unique_sorted)}
    y_train_t = encode_label(y_train_raw)
    y_val_t   = encode_label(data["y_val"])
    y_test_t  = encode_label(data["y_test"])

    n_total = len(y_train_t) + len(y_val_t) + len(y_test_t)
    print(f"  Split 70/15/15:")
    print(f"    Train : {len(y_train_t):>5} ({len(y_train_t)/n_total*100:.1f}%)")
    print(f"    Val   : {len(y_val_t):>5} ({len(y_val_t)/n_total*100:.1f}%)")
    print(f"    Test  : {len(y_test_t):>5} ({len(y_test_t)/n_total*100:.1f}%)")

    unique_lbl, counts = torch.unique(y_train_t, return_counts=True)
    print("  Phan phoi lop (train):")
    for lbl, cnt in zip(unique_lbl.tolist(), counts.tolist()):
        print(f"    Class {lbl}: {cnt} samples")

    train_ds = TensorDataset(X_train, y_train_t)
    val_ds   = TensorDataset(X_val,   y_val_t)
    test_ds  = TensorDataset(X_test,  y_test_t)

    train_loader = DataLoader(
        train_ds, batch_size=cfg["batch_size"],
        shuffle=True, num_workers=cfg["num_workers"],
        pin_memory=(DEVICE.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["batch_size"] * 2,
        shuffle=False, num_workers=cfg["num_workers"],
        pin_memory=(DEVICE.type == "cuda"),
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg["batch_size"] * 2,
        shuffle=False, num_workers=cfg["num_workers"],
        pin_memory=(DEVICE.type == "cuda"),
    )
    return train_loader, val_loader, test_loader, mapping_ref


# helper de tra ve mapping label
_LABEL_MAPPING = {}


# ─────────────────────────────────────────────────────────────────────────────
# 3. TRAIN / EVAL HELPERS
# ─────────────────────────────────────────────────────────────────────────────

class EarlyStopping:
    def __init__(self, patience=10, min_delta=1e-4):
        self.patience    = patience
        self.min_delta   = min_delta
        self.counter     = 0
        self.best_loss   = float("inf")
        self.should_stop = False

    def step(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter   = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = correct = total = 0
    for X_b, y_b in loader:
        X_b, y_b = X_b.to(device), y_b.to(device)
        optimizer.zero_grad()
        logits = model(X_b)           # (N, n_classes)
        loss   = criterion(logits, y_b)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(y_b)
        preds   = logits.argmax(dim=1)
        correct += (preds == y_b).sum().item()
        total   += len(y_b)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = correct = total = 0
    all_preds, all_labels = [], []

    for X_b, y_b in loader:
        X_b, y_b = X_b.to(device), y_b.to(device)
        logits = model(X_b)           # (N, n_classes)
        loss   = criterion(logits, y_b)
        preds  = logits.argmax(dim=1)

        total_loss += loss.item() * len(y_b)
        correct    += (preds == y_b).sum().item()
        total      += len(y_b)
        all_preds.append(preds.cpu().numpy())
        all_labels.append(y_b.cpu().numpy())

    return (
        total_loss / total, correct / total,
        np.concatenate(all_preds),
        np.concatenate(all_labels),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 4. TRAINING CURVE
# ─────────────────────────────────────────────────────────────────────────────

def plot_history(history, save_dir):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor("#0d1117")
    ep = range(1, len(history["train_loss"]) + 1)

    for ax, key, ylabel, title in [
        (axes[0], "loss",  "BCEWithLogitsLoss", "Loss"),
        (axes[1], "acc",   "Accuracy (%)",       "Accuracy (%)"),
    ]:
        train_vals = history[f"train_{key}"]
        test_vals  = history[f"test_{key}"]
        if key == "acc":
            train_vals = [v * 100 for v in train_vals]
            test_vals  = [v * 100 for v in test_vals]

        ax.plot(ep, train_vals, color="#00ff88", lw=2, label="Train")
        ax.plot(ep, test_vals,  color="#ff6644", lw=2, ls="--", label="Test")
        ax.set_title(title, color="white", fontsize=13, fontweight="bold")
        ax.set_xlabel("Epoch", color="white")
        ax.set_ylabel(ylabel, color="white")
        ax.set_facecolor("#1a1a2e")
        ax.tick_params(colors="white")
        ax.legend(facecolor="#1a1a2e", labelcolor="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333355")
        if key == "acc":
            ax.set_ylim([50, 101])

    plt.tight_layout()
    out = Path(save_dir) / "training_curve.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"[PLOT] Training curve luu tai: {out}")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# 5. MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    set_seed(CFG["seed"])

    print("=" * 60)
    print(f"  Device : {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"  GPU    : {torch.cuda.get_device_name(0)}")
    print("=" * 60)

    train_loader, val_loader, test_loader, label_mapping = load_data(CFG)
    class_names = [f"Class_{k}" for k in sorted(label_mapping.keys())]
    print(f"  Class names: {class_names}")

    # Khoi tao model
    model = MalwareCNN(n_classes=N_CLASSES).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[MODEL] MalwareCNN ({N_CLASSES} lop)  |  Params: {n_params:,}")
    print(model)

    # CrossEntropyLoss voi class_weight de xu ly mat can bang
    print(f"\n[LOSS] CrossEntropyLoss (5-class)")
    criterion  = nn.CrossEntropyLoss()
    optimizer  = optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
    scheduler  = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG["epochs"])
    stopper    = EarlyStopping(patience=CFG["patience"])

    history   = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
    best_acc  = 0.0
    save_path = Path(CFG["save_dir"]) / "best_model.pth"

    print(f"\n[TRAIN] {CFG['epochs']} epochs, batch={CFG['batch_size']}, lr={CFG['lr']}\n")

    for epoch in range(1, CFG["epochs"] + 1):
        t0 = time.time()

        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        va_loss, va_acc, _, _ = evaluate(model, val_loader,   criterion, DEVICE)
        te_loss, te_acc, _, _ = evaluate(model, test_loader,  criterion, DEVICE)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["test_loss"].append(va_loss)   # dung val de ve bieu do
        history["test_acc"].append(va_acc)

        marker = ""
        if va_acc > best_acc:
            best_acc = va_acc
            torch.save({
                "epoch"      : epoch,
                "model_state": model.state_dict(),
                "optimizer"  : optimizer.state_dict(),
                "best_val_acc": best_acc,
                "cfg"        : CFG,
            }, save_path)
            marker = "  << BEST"

        print(
            f"Epoch {epoch:>3}/{CFG['epochs']}  "
            f"| Train loss={tr_loss:.4f} acc={tr_acc*100:.2f}%"
            f"  | Val  loss={va_loss:.4f} acc={va_acc*100:.2f}%"
            f"  | Test acc={te_acc*100:.2f}%"
            f"  | {time.time()-t0:.1f}s{marker}"
        )

        # Early stopping dua tren VAL loss (dung nguyen tac)
        stopper.step(va_loss)
        if stopper.should_stop:
            print(f"\n[EARLY STOP] tai epoch {epoch}  (patience={CFG['patience']})")
            break

    # ── Danh gia cuoi ──
    print(f"\n{'='*60}")
    print(f"[EVAL] Load best model: {save_path}")
    ckpt = torch.load(save_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(ckpt["model_state"])

    print("\n[VAL SET]")
    _, va_acc, _, _ = evaluate(model, val_loader, criterion, DEVICE)
    print(f"  Val  Accuracy: {va_acc*100:.4f}%")

    print("\n[TEST SET - KET QUA CUOI CUNG]")
    _, _, all_preds, all_labels = evaluate(model, test_loader, criterion, DEVICE)
    print(f"  Accuracy          : {accuracy_score(all_labels, all_preds)*100:.4f}%")
    print(f"  Macro Precision   : {precision_score(all_labels, all_preds, average='macro', zero_division=0):.4f}")
    print(f"  Macro Recall      : {recall_score(all_labels, all_preds, average='macro', zero_division=0):.4f}")
    print(f"  Macro F1-Score    : {f1_score(all_labels, all_preds, average='macro', zero_division=0):.4f}")
    print(f"  Weighted F1-Score : {f1_score(all_labels, all_preds, average='weighted', zero_division=0):.4f}")

    print("\n[CLASSIFICATION REPORT]")
    print(classification_report(all_labels, all_preds,
                                target_names=class_names, digits=4))

    cm = confusion_matrix(all_labels, all_preds)
    print("[CONFUSION MATRIX]")
    header = "  " + "".join(f"{c:>12}" for c in class_names)
    print(header)
    for i, row in enumerate(cm):
        row_str = "".join(f"{v:>12}" for v in row)
        print(f"  {class_names[i]:<12}{row_str}")

    # Luu history
    hist_path = Path(CFG["save_dir"]) / "training_history.npz"
    np.savez(hist_path, **history)
    print(f"\n[SAVE] History   : {hist_path}")
    print(f"[SAVE] Best model: {save_path}")
    print(f"[DONE] Best Test Accuracy = {best_acc*100:.4f}%")

    plot_history(history, CFG["save_dir"])


if __name__ == "__main__":
    main()
