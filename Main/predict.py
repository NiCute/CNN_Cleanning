"""
predict.py
==========
Đồ án IE105 - Khai phá dữ liệu & Ứng dụng
Kịch bản thử nghiệm và dự đoán phân loại mã độc Android từ file dữ liệu CSV bất kỳ.

Cách sử dụng trong thực tế:
1. Dự đoán file CSV mới (tự động nhận diện cột đặc trưng, thiếu cột tự bù 0):
     python predict.py --input <duong_dan_file.csv>

2. Nếu file CSV đã có sẵn cột nhãn (để tính Accuracy và kiểm tra sai số):
     python predict.py --input sample.csv --label Class

3. Hiển thị Top-K xác suất cao nhất cho mỗi mẫu:
     python predict.py --input sample.csv --topk 3

4. Chạy chế độ Demo nhanh (lấy ngẫu nhiên 10 mẫu từ tập Test chính thức để thử nghiệm):
     python predict.py --demo

Nguyên tắc tiền xử lý khi suy luận (Inference Pipeline):
- Chuẩn hóa: Sử dụng đúng bộ MinMaxScaler đã lưu từ trước (scaler.joblib) để bảo đảm
  thang đo pixel đồng nhất với lúc huấn luyện. Tuyệt đối không tự fit lại scaler trên file mới.
- Khớp cột (Feature Alignment): Tự động khớp 470 cột đặc trưng theo chuẩn tập Train.
- Tái cấu trúc không gian: Sắp xếp các cột theo 9 nhóm ngữ nghĩa rồi đệm 14 số 0 thành ảnh 22x22.
"""

import argparse
import json
from pathlib import Path
import sys

# Cấu hình UTF-8 cho console Windows để in tiếng Việt có dấu không bị lỗi charmap
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

# Nhúng thư mục hiện tại để nạp các hàm bổ trợ
BASE_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(BASE_DIR))

try:
    from feature_to_image import get_semantic_ordering
    _USE_FEATURE_TO_IMAGE = True
except ImportError:
    _USE_FEATURE_TO_IMAGE = False

# ==============================================================================
# 1. CẤU HÌNH ĐƯỜNG DẪN VÀ THÔNG SỐ LỚP
# ==============================================================================

MODEL_PATH  = BASE_DIR / "best_model.pth"
DATA_PATH   = BASE_DIR / "images_22x22.npz"
FEAT_CSV    = BASE_DIR / "471_extracted features.csv"
SCALER_PATH = BASE_DIR / "scaler.joblib"

CLASS_NAMES = {
    0: "Adware",
    1: "Banking Malware",
    2: "SMS Malware",
    3: "Riskware",
    4: "Benign (An toàn)",
}

# Biểu tượng cảnh báo trực quan trong màn hình console
CLASS_EMOJI = {
    0: "[! Cảnh báo]",
    1: "[! Nguy hiểm]",
    2: "[! Nguy hiểm]",
    3: "[! Rủi ro]",
    4: "[OK An toàn]",
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_PIXELS = 484   # 22 x 22
N_CLASSES = 5


# ==============================================================================
# 2. NẠP MÔ HÌNH ĐÃ HUẤN LUYỆN (MODEL LOADING)
# ==============================================================================

def load_model():
    """Nạp trọng số tốt nhất (Best Checkpoint) vào kiến trúc MalwareCNN."""
    from cnn_model import MalwareCNN

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy file checkpoint mô hình tại: {MODEL_PATH}")

    model = MalwareCNN(n_classes=N_CLASSES).to(DEVICE)
    ckpt  = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    best_val = ckpt.get("best_val_acc", 0)
    print(f"[MÔ HÌNH] Đã nạp thành công checkpoint: {MODEL_PATH.name}")
    print(f"         Độ chính xác Validation tốt nhất : {best_val*100:.2f}%")
    print(f"         Thiết bị tính toán                : {DEVICE}")
    return model


# ==============================================================================
# 3. TIỀN XỬ LÝ DỮ LIỆU ĐẦU VÀO TỪ FILE CSV
# ==============================================================================

def get_train_feature_names() -> list:
    """Lấy danh sách 470 tên đặc trưng chuẩn bằng cách đọc dòng tiêu đề của file CSV gốc."""
    df = pd.read_csv(FEAT_CSV, nrows=0)
    return [c for c in df.columns if c != "Class"]


def preprocess_csv(csv_path: str, label_col: str = None):
    """
    Tiền xử lý file CSV người dùng đưa vào:
    1. Đọc dữ liệu và tách riêng cột nhãn thực tế (nếu có).
    2. Căn chỉnh khớp 470 cột đặc trưng chuẩn (cột thiếu tự điền 0, cột dư tự loại).
    3. Chuẩn hóa giá trị đặc trưng bằng MinMaxScaler đã fit trên tập Train.
    4. Sắp xếp lại thứ tự cột theo các nhóm ngữ nghĩa (Semantic Ordering).
    5. Đệm thêm 14 giá trị 0 ở cuối và chuyển thành ma trận ảnh 22x22.
    """
    df = pd.read_csv(csv_path)
    print(f"\n[TIỀN XỬ LÝ] Đang đọc file đầu vào: {csv_path}")
    print(f"             Số dòng: {len(df):,} | Số cột: {len(df.columns)}")

    # Trích xuất nhãn thực tế nếu có để đối chiếu độ chính xác
    y_true = None
    if label_col and label_col in df.columns:
        y_true = df[label_col].values
        df = df.drop(columns=[label_col])
        print(f"             Đã tìm thấy cột nhãn thực tế: '{label_col}'")

    if "Class" in df.columns:
        if y_true is None:
            y_true = df["Class"].values
        df = df.drop(columns=["Class"])

    # Lấy 470 đặc trưng chuẩn
    train_features = get_train_feature_names()

    # Đối chiếu mức độ trùng khớp giữa file đầu vào và bộ đặc trưng huấn luyện
    overlap = [f for f in train_features if f in df.columns]
    missing = [f for f in train_features if f not in df.columns]
    n_input = len(df.columns)

    print(f"\n[ĐỐI CHIẾU CỘT]")
    print(f"  - Số đặc trưng cần thiết : {len(train_features)}")
    print(f"  - Số cột trong file đưa vào : {n_input}")
    print(f"  - Số cột trùng khớp      : {len(overlap)}")
    print(f"  - Số cột thiếu (tự điền 0) : {len(missing)}")
    if 0 < len(missing) <= 10:
        print(f"    Chi tiết cột thiếu: {missing}")

    # Tạo DataFrame căn chỉnh theo đúng 470 cột chuẩn
    df_aligned = pd.DataFrame(0.0, index=df.index, columns=train_features)
    for col in overlap:
        df_aligned[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    X_raw = df_aligned.values.astype(np.float32)

    # Chuẩn hóa bằng scaler.joblib đã fit từ tập huấn luyện
    if SCALER_PATH.exists():
        scaler = joblib.load(SCALER_PATH)
        X_scaled = scaler.transform(X_raw)
        print(f"[CHUẨN HÓA] Đã áp dụng MinMaxScaler từ: {SCALER_PATH.name}")
    else:
        print(f"[CẢNH BÁO] Không tìm thấy {SCALER_PATH.name}, dùng chuẩn hóa MinMax cục bộ theo cột.")
        X_min = X_raw.min(axis=0, keepdims=True)
        X_max = X_raw.max(axis=0, keepdims=True)
        denom = np.where(X_max > X_min, X_max - X_min, 1.0)
        X_scaled = (X_raw - X_min) / denom

    # Tái sắp xếp cột theo thứ tự các nhóm ngữ nghĩa
    if _USE_FEATURE_TO_IMAGE:
        ordered_indices, _ = get_semantic_ordering(train_features)
        X_reordered = X_scaled[:, ordered_indices]
        print(f"[SẮP XẾP] Đã xếp theo 9 nhóm ngữ nghĩa hành vi (Semantic Ordering)")
    else:
        X_reordered = X_scaled
        print(f"[CẢNH BÁO] Giữ nguyên thứ tự cột gốc do không nạp được feature_to_image.py")

    # Đệm số 0 cho đủ 484 pixels (22 x 22)
    N = len(X_reordered)
    n_features = X_reordered.shape[1]
    padded = np.zeros((N, N_PIXELS), dtype=np.float32)
    padded[:, :n_features] = X_reordered

    # Reshape thành tensor ảnh 22x22 với 1 kênh màu (N, 1, 22, 22)
    X_img = padded.reshape(N, 1, 22, 22)
    print(f"[HOÀN TẤT BIẾN ĐỔI] Kích thước tensor ảnh đầu ra: {X_img.shape}")

    return X_img, y_true


# ==============================================================================
# 4. HÀM DỰ ĐOÁN VÀ KẾT XUẤT BÁO CÁO (PREDICTION & REPORTING)
# ==============================================================================

@torch.no_grad()
def predict_batch(model, X_img: np.ndarray, batch_size: int = 64):
    """Thực hiện dự đoán theo từng mini-batch và tính toán xác suất Softmax cho từng lớp."""
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


def parse_labels(raw_labels) -> np.ndarray:
    """Chuyển đổi linh hoạt các định dạng nhãn (chuỗi văn bản hoặc số 1-5 / 0-4) về dạng chuẩn 0-4."""
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
    return np.array(encoded, dtype=int)


def print_results(probs: np.ndarray, preds: np.ndarray,
                  y_true=None, topk: int = 1, max_show: int = 20):
    """In kết quả phân loại chi tiết và bảng thống kê tổng quát ra màn hình console."""
    N = len(preds)
    print(f"\n{'='*65}")
    print(f"             BẢNG KẾT QUẢ DỰ ĐOÁN ({N:,} MẪU)")
    print(f"{'='*65}")

    y_true_arr = parse_labels(y_true) if y_true is not None else None

    show = min(N, max_show)
    for i in range(show):
        pred_class = int(preds[i])
        pred_name  = CLASS_NAMES[pred_class]
        confidence = probs[i][pred_class] * 100
        emoji      = CLASS_EMOJI[pred_class]

        true_str = ""
        if y_true_arr is not None:
            lbl = y_true_arr[i]
            if 0 <= lbl < N_CLASSES:
                mark = "[ĐÚNG]" if lbl == pred_class else "[SAI]"
                true_str = f"  |  Nhãn gốc: {CLASS_NAMES[lbl]} {mark}"

        print(f"  Mẫu {i+1:>4}: {emoji:<13} {pred_name:<20} ({confidence:5.1f}% tin cậy){true_str}")

        # In top-k các phán đoán có khả năng cao tiếp theo nếu người dùng yêu cầu
        if topk > 1:
            top_idx = np.argsort(probs[i])[::-1][:topk]
            for rank, idx in enumerate(top_idx[1:], 2):
                print(f"                #{rank} {CLASS_NAMES[idx]:<20} ({probs[i][idx]*100:5.1f}%)")

    if N > max_show:
        print(f"  ... và còn {N - max_show:,} mẫu khác (sử dụng cờ --show N để xem thêm)")

    # Đánh giá độ chính xác nếu có nhãn gốc
    if y_true_arr is not None:
        from sklearn.metrics import accuracy_score, classification_report
        y_pred_arr = preds
        valid = (y_true_arr >= 0) & (y_true_arr < N_CLASSES)
        if valid.any():
            acc = accuracy_score(y_true_arr[valid], y_pred_arr[valid])
            print(f"\n{'='*65}")
            print(f"  ĐỘ CHÍNH XÁC (ACCURACY) TRÊN TẬP DỮ LIỆU NÀY: {acc*100:.4f}%")
            print(f"{'='*65}")
            print(classification_report(
                y_true_arr[valid], y_pred_arr[valid],
                labels=list(CLASS_NAMES.keys()),
                target_names=list(CLASS_NAMES.values()),
                digits=4,
                zero_division=0
            ))

    # Bảng phân phối tổng thể các dự đoán
    print(f"\n[PHÂN PHỐI DỰ ĐOÁN]")
    for cls_id, cls_name in CLASS_NAMES.items():
        n = (preds == cls_id).sum()
        bar = "#" * int(n / N * 25)
        print(f"  {CLASS_EMOJI[cls_id]:<13} {cls_name:<20} : {n:>5} mẫu ({n/N*100:5.1f}%) [{bar}]")


# ==============================================================================
# 5. CHẾ ĐỘ THỬ NGHIỆM NHANH (DEMO MODE)
# ==============================================================================

def run_demo(model, topk: int = 1):
    """Chọn ngẫu nhiên 10 mẫu từ tập kiểm thử độc lập (Test set) để chạy thử nghiệm nhanh."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu test: {DATA_PATH}")

    data = np.load(DATA_PATH)
    X_test = data["X_test"]
    y_test = data["y_test"]

    rng = np.random.default_rng(42)
    idx = rng.choice(len(X_test), size=min(10, len(X_test)), replace=False)
    X_s = X_test[idx]
    y_s = y_test[idx]

    probs, preds = predict_batch(model, X_s)
    print(f"\n[DEMO] Đang kiểm tra trên 10 mẫu ngẫu nhiên từ tập Test chính thức:")
    print_results(probs, preds, y_true=y_s, topk=topk)


# ==============================================================================
# 6. ĐIỂM VÀO CHÍNH (MAIN FUNCTION)
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="IE105 — Công cụ dự đoán và phân loại mã độc Android bằng mô hình CNN"
    )
    parser.add_argument("--input", "--csv", dest="input", type=str, default=None,
                        help="Đường dẫn đến file CSV cần phân tích.")
    parser.add_argument("--label", type=str, default="Class",
                        help="Tên cột nhãn thực tế nếu có trong file (mặc định: 'Class').")
    parser.add_argument("--topk", type=int, default=1,
                        help="Hiển thị top-K lớp có xác suất dự đoán cao nhất (mặc định: 1).")
    parser.add_argument("--show", type=int, default=20,
                        help="Số lượng mẫu hiển thị chi tiết tối đa ra màn hình (mặc định: 20).")
    parser.add_argument("--demo", action="store_true",
                        help="Chạy thử nghiệm nhanh trên 10 mẫu ngẫu nhiên từ tập Test.")
    args = parser.parse_args()

    print(f"\n{'='*65}")
    print(f"  HỆ THỐNG PHÂN LOẠI MÃ ĐỘC ANDROID — IE105")
    print(f"  Bộ dữ liệu: CICMalDroid2020 | Phân loại đa lớp (5 lớp)")
    print(f"{'='*65}")

    model = load_model()

    if args.demo or args.input is None:
        run_demo(model, topk=args.topk)
    else:
        input_path = Path(args.input)
        if not input_path.exists() and (BASE_DIR / args.input).exists():
            input_path = BASE_DIR / args.input
        elif not input_path.exists():
            print(f"[LỖI] Không tìm thấy file dữ liệu: {args.input}")
            sys.exit(1)

        df_check = pd.read_csv(input_path, nrows=0)
        has_label = args.label in df_check.columns
        label_col = args.label if has_label else None

        X_img, y_true = preprocess_csv(str(input_path), label_col=label_col)
        probs, preds  = predict_batch(model, X_img)
        print_results(probs, preds, y_true=y_true, topk=args.topk, max_show=args.show)

    print(f"\n[HOÀN TẤT] Quá trình dự đoán đã kết thúc.")


if __name__ == "__main__":
    main()
