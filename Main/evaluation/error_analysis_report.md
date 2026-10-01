# 🔬 BÁO CÁO PHÂN TÍCH SAI SỐ (ERROR ANALYSIS REPORT)
> **Đề tài:** Phân loại mã độc Android bằng mô hình CNN trên không gian ảnh 22×22
> **Tập dữ liệu:** CICMalDroid2020 Test Set (1,740 mẫu)

---

## 1. 📊 Tổng Quan Sai Số Toàn Cục

- **Tổng số mẫu kiểm thử:** 1,740 mẫu
- **Dự đoán chính xác:** 1,500 mẫu (86.21%)
- **Tổng số mẫu sai (Errors):** 240 mẫu (13.79%)
- **Độ tin cậy trung bình khi ĐÚNG:** 91.40%
- **Độ tin cậy trung bình khi SAI:** 64.40%
- **Số ca sai có độ tự tin cao (Confidence ≥ 70%):** 108 mẫu (45.0% tổng lỗi)

---

## 2. 🛡️ Đánh Giá Rủi Ro Bảo Mật (Security Impact)

| Loại rủi ro | Số mẫu | Tỷ lệ | Đánh giá an ninh |
|---|---|---|---|
| **False Negative (Lọt mã độc):** Malware bị đoán thành Benign | **31 mẫu** | **1.78%** | 🔴 **Nghiêm trọng:** Mã độc vượt qua hệ thống kiểm duyệt |
| **False Positive (Báo động giả):** Benign bị đoán thành Malware | **36 mẫu** | **2.07%** | 🟡 **Ảnh hưởng UX:** Ứng dụng lành tính bị chặn nhầm |
| **Lỗi nội bộ các dòng mã độc:** Malware loại A đoán thành B | **173 mẫu** | **9.94%** | 🟢 **Ít nguy hại:** Vẫn nhận diện được bản chất là mã độc |

---

## 3. 📈 Thống Kê Chi Tiết Sai Số Theo Từng Lớp

| Lớp (Class) | Tổng mẫu Test | Đúng | Sai | Recall (%) | Precision (%) |
|---|---|---|---|---|---|
| **Adware** | 188 | 157 | 31 | 83.51% | 63.31% |
| **Banking** | 315 | 256 | 59 | 81.27% | 89.82% |
| **SMS Malware** | 586 | 563 | 23 | 96.08% | 88.80% |
| **Riskware** | 382 | 291 | 91 | 76.18% | 94.17% |
| **Benign** | 269 | 233 | 36 | 86.62% | 88.26% |

---

## 4. 🔀 Top Các Cặp Lớp Bị Nhầm Lẫn Nhiều Nhất

| Thứ hạng | Cặp nhầm lẫn (Thật → Đoán) | Số lượng | % trên tổng lớp thật | Nguyên nhân tiềm ẩn |
|---|---|---|---|---|
| #1 | **Riskware → Adware** | 43 mẫu | 11.26% | Hành vi tương đồng giữa các họ mã độc |
| #2 | **Banking → Adware** | 23 mẫu | 7.30% | Hành vi tương đồng giữa các họ mã độc |
| #3 | **Riskware → SMS Malware** | 22 mẫu | 5.76% | Hành vi tương đồng giữa các họ mã độc |
| #4 | **Adware → SMS Malware** | 21 mẫu | 11.17% | Hành vi tương đồng giữa các họ mã độc |
| #5 | **Banking → SMS Malware** | 17 mẫu | 5.40% | Hành vi tương đồng giữa các họ mã độc |
| #6 | **Riskware → Benign** | 17 mẫu | 4.45% | Sử dụng API dùng chung giữa ứng dụng lành tính và mã độc |

---

## 5. 🧠 Phân Tích Nguyên Nhân & Kiến Nghị Khắc Phục

### 5.1. Vì sao mô hình đưa ra dự đoán sai?
1. **Chồng lấn hành vi Semantic (Behavior Overlap):**
   - Các họ mã độc như *Riskware* và *SMS Malware* cùng sử dụng chung nhóm API Device (`SEND_SMS`, `READ_PHONE_STATE`, `GET_DEVICE_ID`).
   - Khi biểu diễn thành ảnh 22×22, các pixel tại khu vực *Device* và *Network* có cường độ kích hoạt rất gần nhau.
2. **Độ thưa của ma trận đặc trưng (Feature Sparsity):**
   - Nhiều mẫu chỉ kích hoạt một vài API đặc biệt thuộc nhóm `Other_API`, khiến phần lớn các vùng còn lại trên ảnh 22×22 gần bằng 0.
3. **Sự đánh đổi khi cân bằng lớp (Class Weight Trade-off):**
   - Việc áp dụng `class_weight` giúp cải thiện mạnh mẽ Recall của các lớp thiểu số (*Adware*, *Banking*), nhưng có thể làm tăng nhẹ một vài ca dự đoán nhầm giữa các lớp có trọng số tương tự.

### 5.2. Đề xuất cải tiến cho đồ án:
- **Focal Loss:** Thay thế hoặc kết hợp với CrossEntropy để giảm trọng số các mẫu dễ học (easy negatives) và tập trung vào các biên khó phân loại.
- **Attention Mechanism (SE-Net hoặc CBAM):** Bổ sung kênh chú ý để CNN tập trung vào các vùng Semantic quan trọng nhất (như vùng Crypto hoặc Privacy) thay vì quét đồng đều.
- **Sắp xếp thứ tự Feature tối ưu hơn:** Áp dụng thuật toán t-SNE hoặc MDS trên 470 features để các features có độ tương quan cao nằm cạnh nhau trên lưới 22×22.

---
*Báo cáo được khởi tạo tự động bởi mô đun `error_analysis.py` phục vụ bảo vệ đồ án IE105.*