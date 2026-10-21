# Pipeline dự báo quỹ đạo bão

## 1. Dự án muốn làm gì?

**Dùng lịch sử di chuyển của một cơn bão để dự báo vị trí tâm bão trong 48 giờ tiếp theo.** Phạm vi là bão ở Tây Thái Bình Dương, gồm khu vực Philippines, Nhật Bản và Biển Đông.

Ví dụ: tại 00:00 hôm nay, ta đã biết bão ở đâu trong những giờ vừa qua. Mô hình nhận lịch sử đó và đoán vị trí lúc 06:00, 12:00, …, đến 00:00 hai ngày sau — tổng cộng **8 vị trí, cách nhau 6 giờ**.

Dự án dùng dữ liệu bão trong quá khứ để học và kiểm tra khả năng dự báo. Đầu vào hiện tại chỉ có vị trí và thông tin chuyển động tính từ vị trí; đầu ra là quỹ đạo tâm bão, không phải sức gió hay lượng mưa.

Ba câu hỏi thí nghiệm chính:

- RNN, GRU hay LSTM dự báo tốt hơn? Đây là ba loại mô hình học từ chuỗi dữ liệu theo thời gian.
- Đoán cả 48 giờ một lần hay đoán từng bước 6 giờ tốt hơn?
- Thêm tốc độ và hướng di chuyển có giúp ích hơn chỉ dùng vị trí không?

## 2. Toàn bộ luồng xử lý

**Pipeline** là chuỗi bước biến dữ liệu gốc thành dự báo và số liệu đánh giá:

```text
Lịch sử vị trí bão (.nc)
    → Lọc và tách các đoạn liên tục, cách nhau 6 giờ
    → Chia train / validation / test theo năm của bão
    → Tạo mẫu: lịch sử đầu vào + 8 vị trí tương lai làm đáp án
    → Chuẩn hóa đầu vào, lưu dữ liệu đã xử lý (.npz)
    → Huấn luyện mô hình, giữ bản tốt nhất trên validation (.pt)
    → Dự báo trên test, so với vị trí thật → sai số tính bằng km
```

Khi huấn luyện, mô hình được dùng đáp án để điều chỉnh cách dự báo. Khi kiểm tra, mô hình chỉ thấy lịch sử; các vị trí tương lai được giữ lại để chấm điểm.

## 3. Dữ liệu được chuẩn bị thế nào?

### Giữ các quan trắc phù hợp

Dữ liệu gốc nằm ở `data/IBTrACS.WP.v04r01.nc`, lấy từ bộ dữ liệu lịch sử bão **NOAA IBTrACS**. Một quan trắc là một lần ghi nhận bão: mã bão (`SID`), thời gian, vĩ độ (`LAT`) và kinh độ (`LON`). Các dòng cùng mã bão tạo thành lịch sử của cơn bão đó.

Pipeline giữ dữ liệu năm **1980–2024**, thuộc vùng Tây Thái Bình Dương (`WP`), có loại quỹ đạo chính (`main`), tại các mốc **00:00, 06:00, 12:00, 18:00 UTC**. Sau đó, bỏ vị trí không hợp lệ, sắp xếp theo thời gian và loại dòng trùng.

Việc lọc vùng biển áp dụng cho **từng quan trắc**: nếu bão đi ra ngoài WP, phần ở ngoài bị bỏ. Nếu hai quan trắc còn lại không cách nhau đúng 6 giờ, chuỗi bị tách thành các **đoạn liên tục** (*segment*). Không nối tắt qua khoảng thiếu dữ liệu.

### Chia dữ liệu để học và kiểm tra

| Tập | Năm của bão | Vai trò | Số mẫu hiện có |
|---|---|---|---:|
| Train | 1980–2016 | Cho mô hình học từ ví dụ | 27.719 |
| Validation | 2017–2020 | Chọn phiên bản mô hình tốt nhất trong lúc học | 2.499 |
| Test | 2021–2024 | Đánh giá mô hình sau khi đã chọn xong | 2.207 |

**Một cơn bão chỉ thuộc một tập.** Nhờ vậy, mô hình không được học trước một đoạn của chính cơn bão đang dùng để kiểm tra. Test không dùng để chọn mô hình.

### Cắt mỗi đoạn thành các mẫu dự báo

Một **mẫu** (*window*, hay cửa sổ) gồm **14 vị trí liên tiếp**: 6 vị trí lịch sử và 8 vị trí tương lai làm đáp án. Gọi thời điểm bắt đầu dự báo là `t`:

| Phần | Các mốc thời gian | Dùng để làm gì? |
|---|---|---|
| 1 vị trí phụ | `t−30h` | Tính chuyển động cho hàng đầu vào đầu tiên |
| 5 vị trí đưa vào mô hình | `t−24h, t−18h, t−12h, t−6h, t` | Lịch sử để dự báo |
| 8 vị trí tương lai | `t+6h, t+12h, …, t+48h` | Đáp án để học hoặc chấm điểm |

**Vì sao cần 6 vị trí lịch sử nhưng chỉ đưa vào 5 hàng?** Muốn biết bão di chuyển thế nào ở mốc `t−24h`, cần so vị trí lúc đó với vị trí trước nó (`t−30h`). Vị trí phụ chỉ phục vụ phép tính này.

Cửa sổ trượt mỗi lần một mốc 6 giờ để tạo mẫu tiếp theo. Ví dụ, đoạn có 16 vị trí tạo được 3 mẫu dài 14 vị trí. Đoạn ngắn hơn 14 vị trí không tạo được mẫu; một mẫu luôn nằm trong cùng một đoạn của cùng một cơn bão.

### Mỗi hàng đầu vào chứa gì?

Một **đặc trưng** (*feature*) là một con số mô tả bão tại một mốc thời gian. Dự án thử ba cách chọn đặc trưng:

| Cấu hình | Thông tin tại mỗi mốc | Số đặc trưng |
|---|---|---:|
| A | Vĩ độ, kinh độ | 2 |
| B | Vĩ độ, kinh độ, tốc độ, hướng biểu diễn bằng sin/cos | 5 |
| **C — cấu hình chính** | Vĩ độ, kinh độ, vận tốc theo hướng đông và hướng bắc | 4 |

Ví dụ, một hàng C trước chuẩn hóa là `[15.5, 131.0, 18, 9]`: bão ở 15,5°B, 131°Đ, di chuyển với thành phần vận tốc 18 km/h về đông và 9 km/h về bắc. Giá trị âm tương ứng với hướng tây hoặc nam.

Tốc độ và hướng được **tự tính từ hai vị trí liên tiếp đã biết**, không lấy cột chuyển động có sẵn trong file. Cách này bảo đảm đầu vào chỉ dùng thông tin quá khứ. Hướng ở cấu hình B dùng sin/cos để thể hiện đúng việc 359° và 1° gần nhau.

Trước khi học, các đặc trưng được **chuẩn hóa**: trừ trung bình rồi chia độ lệch chuẩn để các cột có thang đo tương đương. Những giá trị này chỉ tính trên train, sau đó dùng nguyên bộ cho validation và test.

Với cấu hình C, ký hiệu đầu vào `[N, 5, 4]` nghĩa là **N mẫu × 5 mốc thời gian × 4 đặc trưng**. Đáp án tọa độ có dạng `[N, 8, 2]`: 8 vị trí, mỗi vị trí gồm vĩ độ và kinh độ.

## 4. Mô hình học và dự báo ra sao?

Dự án ghép **3 loại mô hình (RNN, GRU, LSTM)** với **2 cách dự báo**, tạo thành 6 cấu hình so sánh. Cả ba loại đều đọc lịch sử theo thứ tự thời gian để học quy luật di chuyển.

| Cách dự báo | Cách hoạt động | Điểm cần hiểu |
|---|---|---|
| **Direct** | Nhận lịch sử → đoán cả 8 vị trí trong một lần | Mỗi vị trí tương lai được xác định so với vị trí cuối đã biết |
| **Recursive** | Đoán vị trí sau 6h → đưa dự báo đó vào lịch sử → đoán tiếp, lặp 8 lần | Sai số của bước trước có thể ảnh hưởng bước sau |

Mô hình thực tế xuất **độ dịch chuyển theo hướng đông/bắc, đơn vị km**. Code chuyển độ dịch chuyển thành tọa độ trên bề mặt Trái Đất để vẽ quỹ đạo và chấm điểm. Direct đo độ dịch chuyển từ điểm cuối đã biết; recursive đo từng bước từ điểm ngay trước đó.

Trong lúc huấn luyện:

- **Direct** học đoán cả 8 bước từ một mẫu. **Recursive** học đoán một bước từ lịch sử thật; mỗi cửa sổ tạo ra 8 mẫu học một bước.
- Hàm **loss** đo mức dự báo sai để mô hình điều chỉnh tham số. Dự án dùng Huber loss trên độ dịch chuyển tính bằng km.
- Sau mỗi lượt học toàn bộ dữ liệu (*epoch*), mô hình được kiểm tra trên validation. Recursive lúc này phải tự dự báo đủ 8 bước, giống lúc test.
- Bản có sai số ADE thấp nhất trên validation được lưu thành **`best.pt`** (*checkpoint*: file lưu trạng thái mô hình). Mặc định, nếu 15 epoch liên tiếp không cải thiện thì dừng học.

Hai cách dự báo dùng cùng các trường hợp validation/test để so sánh. Recursive có nhiều mẫu học một bước hơn, nên số mẫu train và thời gian học cũng cần được xét khi đọc kết quả.

## 5. Đọc kết quả thế nào?

Ở mỗi mốc tương lai, code đo khoảng cách giữa **vị trí dự báo và vị trí thật** trên bề mặt Trái Đất bằng công thức Haversine. Tất cả chỉ số dưới đây đều tính bằng **km; càng thấp càng tốt**.

| Chỉ số | Trả lời câu hỏi nào? |
|---|---|
| **MTE tại +6h, +12h, +24h, +48h** | Tại mốc đó, dự báo lệch vị trí thật trung bình bao nhiêu km? |
| **ADE** | Trung bình trên **toàn bộ 8 mốc** và tất cả mẫu, dự báo lệch bao nhiêu km? |
| **FDE** | Riêng vị trí cuối ở +48h lệch trung bình bao nhiêu km? Bằng MTE tại +48h. |

Ví dụ, **FDE = 480 km** nghĩa là vị trí dự báo sau 48 giờ cách vị trí thật trung bình 480 km trên tập đang đánh giá.

Kết quả hiện có trên test: 6 cấu hình chính đạt **ADE khoảng 245–250 km**, **FDE khoảng 479–505 km**. Đây là kết quả thí nghiệm với seed 0 (giá trị khởi tạo ngẫu nhiên cố định); chưa đủ để khẳng định một mô hình luôn tốt nhất. Bảng số liệu và phân tích nằm trong [báo cáo thí nghiệm](../experiments/report.md).

## 6. Chạy pipeline

Chạy lệnh từ **thư mục gốc repo**. Xem [README — cài đặt môi trường](../README.md#1-cài-đặt-môi-trường) nếu chưa cài Python và các thư viện cần thiết.

### Chạy một mô hình từ đầu đến cuối

```bash
conda activate machine-learning

# 1. Dữ liệu gốc → dữ liệu đã xử lý
python scripts/preprocess.py --config configs/data.yaml

# 2. Huấn luyện LSTM với cách dự báo direct, mặc định dùng đặc trưng C
python scripts/train.py --model lstm --strategy direct

# 3. Chấm mô hình tốt nhất trên test, in chỉ số ra terminal
python scripts/evaluate.py --checkpoint experiments/checkpoints/lstm_direct_C_s0/best.pt --split test
```

Chỉ cần chạy lại preprocess khi thay dữ liệu hoặc cấu hình xử lý. Nếu đã có dữ liệu đã xử lý và checkpoint tương ứng, có thể chạy thẳng bước đánh giá. Thêm `--out experiments/results/lstm_direct_test.json` vào lệnh đánh giá nếu muốn lưu chỉ số thành file.

### Chạy các thí nghiệm so sánh

Sau khi đã preprocess, chọn các lệnh phù hợp:

```bash
# Huấn luyện và đánh giá đủ 6 cấu hình RNN/GRU/LSTM × direct/recursive
python scripts/run_experiments.py --experiment main

# So sánh đặc trưng A/B/C trên cùng mô hình LSTM-direct (ablation)
python scripts/run_experiments.py --experiment ablation

# Phân tích sai số theo nhóm chuyển động: đi thẳng, đổi hướng, khó phân loại
# Cần có kết quả của main trước
python scripts/run_experiments.py --experiment movement
```

Các lệnh `run_experiments.py` cập nhật bảng CSV, hình và báo cáo từ kết quả đang có. `ablation` là thí nghiệm thay riêng nhóm đặc trưng để xem thông tin nào giúp dự báo tốt hơn.

## 7. Tìm dữ liệu, kết quả và code ở đâu?

| Đường dẫn | Nội dung |
|---|---|
| `configs/data.yaml` | Quy tắc lọc dữ liệu, năm chia tập |
| `configs/training.yaml` | Thiết lập huấn luyện và thí nghiệm |
| `data/processed/` | `train.npz`, `validation.npz`, `test.npz` chứa các mảng đầu vào/đáp án; `scalers.joblib` lưu bộ chuẩn hóa |
| `experiments/checkpoints/` | Mô hình đã học, cấu hình và lịch sử huấn luyện của từng lần chạy |
| `experiments/results/`, `experiments/figures/` | Bảng chỉ số và hình do các thí nghiệm sinh ra |
| `experiments/report.md` | Báo cáo tổng hợp kết quả |
| `src/data/`, `src/models/`, `src/training/`, `src/evaluation/` | Code tương ứng với chuẩn bị dữ liệu, mô hình, huấn luyện và đánh giá |

Đọc sâu hơn khi cần: [tiền xử lý](preprocessing_report.md) · [mô hình](models.md) · [cách tính chỉ số](evaluation.md).
