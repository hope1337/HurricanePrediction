# Hướng dẫn toàn pipeline: từ data thô tới dự báo quỹ đạo bão

Tài liệu này giải thích toàn bộ pipeline dự báo quỹ đạo bão Tây Thái Bình Dương,
từ bước đọc dữ liệu tới lúc ra con số đánh giá — viết cho người mới tiếp cận,
mỗi khái niệm đều kèm ví dụ số cụ thể.

**Thứ tự chạy (bắt buộc theo đúng thứ tự):**

```bash
# Chặng 1 — tiền xử lý (1 lần duy nhất): .nc -> .npz
python scripts/preprocess.py --config configs/data.yaml

# Chặng 2 — train (mỗi model 1 lần): .npz -> best.pt
python scripts/train.py --model lstm --strategy direct
# hoặc chạy cả cụm 6 models:
python scripts/run_experiments.py --experiment main

# Chặng 3 — đánh giá: best.pt -> số liệu + hình + báo cáo
python scripts/evaluate.py --checkpoint experiments/checkpoints/lstm_direct_C_s0/best.pt --split test
python scripts/run_experiments.py --experiment ablation
python scripts/run_experiments.py --experiment movement
```

---

## 1. Dữ liệu gốc là gì?

File `data/IBTrACS.WP.v04r01.nc` (9 MB) chứa **249,084 dòng quan trắc của 4,255 cơn bão**
từ năm 1884 tới 2026. Mỗi dòng là 1 lần ghi nhận vị trí bão. Các cột quan trọng:

| Cột | Ý nghĩa | Ví dụ |
|---|---|---|
| `SID` | **Mã định danh bão** — mỗi bão đúng 1 mã duy nhất, ví dụ `1980076N06148` (bão năm 1980). Đây là "chìa khóa" của pipeline: gom dòng cùng SID = tái hiện 1 cơn bão; chia train/val/test theo SID nên 1 cơn không bao giờ lọt vào 2 tập | `1980076N06148` |
| `SEASON` | Năm bão bắt đầu | 1980 |
| `TIME` / `ISO_TIME` | Thời điểm quan trắc (giờ UTC) | 1980-08-01 06:00 |
| `LAT` / `LON` | Vĩ độ / kinh độ (độ). Kinh độ theo chuẩn file: **0–360° Đông** | 15.5 / 131.0 |
| `BASIN` | Vùng biển chứa tâm bão **tại đúng thời điểm dòng đó** (xem giải thích dưới): WP / EP / NI | WP |
| `TRACK_TYPE` | Loại track: `main` (chính) hay spur-merge/split/other, PROVISIONAL... | main |
| `STORM_SPEED` / `STORM_DIR` | Tốc độ/hướng **có sẵn trong file — nhưng KHÔNG dùng** (xem mục 3) | — |

### 1.1. Basin nghĩa là gì? (vùng biển của từng quan trắc)

Thái Bình Dương rộng nên người ta chia thành vùng để quản lý: **WP** = Tây Thái Bình
Dương (quanh Philippines, Nhật, Biển Đông), **EP** = Đông Thái Bình Dương (gần châu Mỹ),
**NI** = Bắc Ấn Độ Dương. Cột `BASIN` trả lời câu hỏi: **"lúc ghi dòng này, tâm bão đang
nằm ở vùng biển nào?"** — và vì bão di chuyển, đáp án có thể đổi giữa chừng trong cùng
1 cơn bão.

Ví dụ 1 cơn bão đi từ Philippines sang phía châu Mỹ (kinh độ tăng dần qua 180°):

| giờ UTC | LAT | LON | BASIN | hiểu thế nào |
|---|---|---|---|---|
| 00:00 | 15.0 | 150.0 | WP | còn ở phía tây |
| 06:00 | 15.5 | 165.0 | WP | vẫn phía tây |
| 12:00 | 16.0 | 178.0 | WP | sắp qua đường đổi ngày |
| 18:00 | 16.2 | 185.0 (= −175°) | EP | đã qua phía đông → đổi nhãn thành EP |

Vì nhãn này **gắn theo từng dòng, không gắn theo cả cơn bão**, nên bước lọc F2 phải
lọc theo từng dòng: giữ dòng WP, bỏ dòng EP/NI. Hệ quả: cơn bão trên bị cắt làm đôi —
phần WP giữ lại thành 1 khúc riêng, phần EP bỏ đi. Không có chuyện "cả cơn bão thuộc WP".

## 2. Load data (`scripts/eda/io.py`)

File `.nc` lưu dạng lưới 2D `(bão × 360 ô thời gian)`, mỗi bão chỉ dùng `numobs` ô đầu.
Hàm `load_ibtracs` chỉ lấy đúng các ô thật (bỏ padding), chuyển fill value `-9999`
thành NaN, parse giờ UTC, trả về 1 bảng long-format 249,084 dòng. File gốc **không bao
giờ bị ghi đè** — pipeline chỉ đọc.

## 3. Filter — giữ lại dòng nào, bỏ dòng nào (F1–F7, `src/data/preprocessing.py`)

Hình dung dữ liệu thô là 1 đống 249,084 dòng lộn xộn của 4,255 cơn bão từ năm 1884.
Filter là 7 cái rây lọc nối tiếp nhau: dòng nào không đạt chuẩn ở rây nào thì rớt
ngay ở rây đó. Dưới đây là 1 cơn bão giả định đi qua cả 7 cái rây (dòng nào bị loại
ghi rõ lý do):

| obs | giờ UTC | LAT | LON | BASIN | TRACK | qua rây nào? |
|---|---|---|---|---|---|---|
| 0 | 01/08 03:00 | 15.0 | 130.0 | WP | main | ❌ rớt F4 (giờ 03:00, không phải mốc chuẩn) |
| 1 | 01/08 06:00 | 15.5 | 131.0 | WP | main | ✅ giữ |
| 2 | 01/08 12:00 | 16.1 | 132.2 | WP | main | ✅ giữ |
| 3 | 01/08 18:00 | 16.8 | 150.0 | EP | main | ❌ rớt F2 (đã trôi sang biển EP) |
| 4 | 02/08 00:00 | 17.0 | 133.0 | WP | spur-merge | ❌ rớt F3 (đoạn rẽ nhánh, không phải main) |
| 5 | 02/08 06:00 | 17.5 | 134.0 | WP | main | ✅ giữ |
| 6 | 02/08 18:00 | 18.0 | 135.0 | WP | main | ✅ giữ (nhưng mốc 12:00 bị thiếu → đứt khúc ở F7) |

Giải thích từng rây bằng lời thường:

- **F1 season (năm bão)**: đề bài chỉ làm bão 1980–2024. Bão năm 1975 hay 2026? Bỏ hết.
  Còn 104,927 dòng / 1,471 bão.
- **F2 basin (vùng biển)**: chỉ giữ dòng nào đang ở WP (xem mục 1.1). 1 cơn bão có thể
  vừa có dòng WP vừa có dòng EP — dòng nào ở EP thì bỏ dòng đó, không bỏ cả cơn.
  Còn 100,970 dòng.
- **F3 track chính**: chỉ giữ dòng `main`. Dòng spur (đoạn rẽ nhánh, nhập/tách khỏi
  track khác) và PROVISIONAL (số liệu tạm, chưa kiểm chứng) → bỏ vì không đáng tin.
  Còn 100,386 dòng / 1,446 bão.
- **F4 giờ chuẩn**: chỉ giữ mốc 00:00, 06:00, 12:00, 18:00 (phút = 00). Dữ liệu gốc có cả
  mốc lẻ 03:00, 09:00... (đo thêm lúc đổ bộ) → bỏ để mọi bước thời gian đều đúng 6 tiếng.
  Còn 50,674 dòng. Đây là bước rớt nhiều nhất (gần một nửa) — là chủ ý, không phải lỗi.
- **F5 vị trí hợp lệ**: LAT phải trong [−90, 90], LON trong [0, 360], không thiếu giờ.
  Thực tế không rớt dòng nào (vị trí file này đầy đủ).
- **F6 sort + dedup**: xếp mỗi bão theo giờ tăng dần; nếu trùng giờ thì giữ dòng đầu
  (thực tế không có dòng trùng — quy tắc này chỉ để chắc chắn deterministic).
- **F7 cắt segment**: trong ví dụ trên, obs 6 cách obs 5 tới 12 tiếng (thiếu mốc 12:00)
  → không được nối 2 điểm này lại với nhau (ở giữa bão đi đâu không ai biết) mà phải
  **cắt đứt** thành 2 khúc riêng: khúc {obs 1, obs 2} và khúc {obs 5, obs 6}.
  Toàn bộ data được 1,451 khúc ("segment") như vậy; khúc nào ngắn hơn 14 obs thì không
  đủ làm 1 bài tập nên không sinh window (xem mục 5).

## 4. Feature chuyển động — tính v, theta, v_E, v_N thế nào?

**Ta chỉ có vị trí + thời gian. Tốc độ và hướng phải TỰ TÍNH từ 2 điểm liên tiếp**
(không dùng cột có sẵn vì nó "nhìn trước tương lai": có giá trị ngay cả ở quan trắc
đầu tiên, nơi chưa có quá khứ để tính — đã chứng minh ở EDA).

Ví dụ thật: p0 = (15.0°N, 130.0°E) lúc 00:00, p1 = (15.5°N, 131.0°E) lúc 06:00.

- **Bước 1 — khoảng cách `d`**: công thức Geod trên ellipsoid WGS84 (mục 9).
  Ở đây d ≈ 120.9 km. (Tuyệt đối không lấy `131.0 − 130.0 = 1.0 độ` làm khoảng cách —
  độ kinh tuyến co lại gần cực, 1° ở xích đạo ≈ 111 km nhưng ở vĩ độ 60 chỉ còn ~55 km.)
- **Bước 2 — tốc độ `v`** = d / thời gian = 120.9 / 6 ≈ **20.2 km/h**.
- **Bước 3 — hướng `theta`** (bearing): góc đo **từ hướng Bắc, xoay theo chiều kim
  đồng hồ**. Bắc = 0°, Đông = 90°, Nam = 180°, Tây = 270°. Đi về đông-bắc như ví dụ
  thì theta ≈ 62.6°.
- **Bước 4 — tách thành 2 thành phần** (theta là hướng, v là tốc độ, đúng như bạn nghĩ):
  - `v_E = v · sin(theta)` = 20.2 × sin(62.6°) ≈ **17.9 km/h** (phần chạy về Đông)
  - `v_N = v · cos(theta)` = 20.2 × cos(62.6°) ≈ **9.3 km/h** (phần chạy về Bắc)
  - Kiểm tra ngược: √(17.9² + 9.3²) ≈ 20.2 ✓

Vì sao phải tách? Model không hiểu "62.6°" nghĩa là gì, nhưng 2 con số
(đông bao nhiêu, bắc bao nhiêu) thì nó cộng trừ được. Hàng input tương ứng là
`[15.5, 131.0, 17.9, 9.3]` — đây chính là **config C**.

**3 config feature là 3 cách đóng gói cùng 1 mốc thời gian** (lấy mốc obs 1 trong
ví dụ mục 4: vị trí (15.5°N, 131.0°E), tốc độ 20.2 km/h, hướng 62.6°):

| config | hàng input trông thế nào | nghĩa từng số |
|---|---|---|
| **A** (position only, 2 số) | `[15.5, 131.0]` | vĩ độ, kinh độ — model chỉ biết "đang ở đâu" |
| **B** (+speed/dir, 5 số) | `[15.5, 131.0, 20.2, 0.887, 0.461]` | + tốc độ (km/h) + hướng mã hóa sin/cos (0.887 = sin 62.6°, 0.461 = cos 62.6°). Không đưa 62.6° thô vì 359° và 1° thực ra gần nhau mà số thô thì xa nhau |
| **C (chính)** (+velocity, 4 số) | `[15.5, 131.0, 17.9, 9.3]` | + vận tốc đông/tây (17.9) và bắc/nam (9.3) km/h |

### 4.1. Thí nghiệm ablation là gì? (so sánh 3 config)

Câu hỏi cần trả lời: "cho model thêm thông tin chuyển động (B, C) có đoán chính xác
hơn chỉ cho vị trí (A) không?" Cách làm rất trực tiếp, không có gì bí ẩn:

1. Lấy **đúng 1 kiến trúc** (LSTM) + **đúng 1 chiến lược** (direct) + **đúng cùng
   2,207 bài test** (cùng sample_id — code check cái này, lệch là báo lỗi).
2. Train 3 lần y hệt nhau, chỉ khác input: lần 1 cho ăn config A, lần 2 cho ăn B, lần 3 cho ăn C.
3. So điểm ADE trên test.

Kết quả thật:

| config | ADE (km) | FDE +48h (km) | đọc thế nào |
|---|---|---|---|
| A (vị trí) | 244.1 | 472.8 | tốt nhất ở setup này |
| C (velocity) | 250.2 | 487.4 | kém hơn A ~6 km |
| B (speed+dir) | 253.0 | 492.1 | kém nhất |

Kết luận **đúng phạm vi thí nghiệm**: với LSTM-direct + seed 0, thêm motion không giúp
— có thể vì 5 vị trí liên tiếp đã ngầm chứa thông tin chuyển động, model tự suy ra được.
Đây là quan sát, không phải chân lý: đổi seed, đổi kiến trúc, kết quả có thể khác.

**Chuẩn hóa**: mỗi cột trừ mean rồi chia std — mean/std này tính **chỉ từ tập train**
(ví dụ LAT mean 18.36, std 8.43 → 15.5 thành (15.5−18.36)/8.43 ≈ −0.34). Val/test dùng
y nguyên bộ số này, không tính lại (tính lại là leak). Lưu ở `scalers.joblib`, có hàm
`invert` để đổi ngược về đơn vị vật lý.

## 5. Windowing — cắt chuỗi thành từng "bài tập" (ví dụ đầy đủ)

Sau filter, mỗi segment là 1 chuỗi obs liên tiếp cách nhau đúng 6h. Lấy 1 segment
cụ thể dài **16 obs** (đánh số 0–15, giờ giấc 00:00, 06:00, 12:00... nối tiếp nhau).
Model không ăn cả chuỗi 16 này mà ăn từng **cửa sổ dài 14 obs**, trượt dần từ đầu
tới cuối. Với segment 16 obs ta được đúng **3 cửa sổ** (16 − 13 = 3):

| cửa sổ | obs dùng | 6 vị trí lịch sử | 5 hàng input cho model | 8 target (đáp án giấu) |
|---|---|---|---|---|
| k=0 | obs 0–13 | obs 0–5 | obs 1–5 | obs 6–13 |
| k=1 | obs 1–14 | obs 1–6 | obs 2–6 | obs 7–14 |
| k=2 | obs 2–15 | obs 2–7 | obs 3–7 | obs 8–15 |
| k=3 | ❌ (obs 3–16 vượt quá segment) | — | — | — |

Đọc bảng trên theo hàng k=0: cho model xem 5 hàng input (obs 1–5, tức quá khứ tới
thời điểm gốc t = giờ obs 5), bắt đoán 8 vị trí tiếp theo (obs 6–13 = t+6h ... t+48h),
rồi so với đáp án thật. Trượt sang k=1: mọi thứ dịch phải 1 obs, giờ gốc t cũng dịch
theo. Cứ thế tới khi chạm cuối segment.

Vẽ trên trục thời gian (mỗi `•` là 1 obs, cửa sổ k=0):

```text
obs:    0   1   2   3   4   5   6   7   8       13  14  15
        •───•───•───•───•───•───•───•───•── ... ──•   •   •
        ├─ lịch sử (6) ─┤
            ├── input (5 hàng) ──┤
                            ├──── đáp án 8 bước ────┤
```

Vì sao cần **6 vị trí lịch sử mà chỉ 5 hàng input**? Hàng input đầu tiên (obs 1) cần
vận tốc tính từ cặp obs 0 → obs 1, nên obs 0 ở mốc t−30 chỉ làm "bàn đạp tính velocity",
không đưa vào model. Nhớ: **input của model luôn là 5 hàng [5, D], còn đáp án luôn là
8 vị trí**.

Công thức tổng quát: segment dài L obs → cho **L − 13 windows** (20 obs → 7 windows;
13 obs → 0 window vì vừa đủ... thực ra cần L ≥ 14 mới có window đầu tiên).
Cộng toàn bộ data: train 27,719 / val 2,499 / test 2,207 windows.

Quy tắc sắt: window không bao giờ vắt qua 2 bão, qua chỗ mất mốc giờ, qua biên basin,
hay qua ranh train/val/test — vì mỗi window nằm gọn trong 1 segment, mà segment thì
thuộc đúng 1 bão, 1 split.

**Riêng lúc train model recursive** (học đoán từng bước 6h một): từ mỗi window 14-obs
ở trên, code suy ra thêm 8 mẫu one-step — mỗi mẫu gồm 6 obs lịch sử → 5 hàng input +
1 target là bước 6h ngay sau đó. Lấy window k=0 (obs 0–13) làm ví dụ:

| mẫu con | 6 obs lịch sử | 5 hàng input | target |
|---|---|---|---|
| j=0 | obs 0–5 | obs 1–5 | obs 5 → obs 6 (bước t → t+6h) |
| j=1 | obs 1–6 | obs 2–6 | obs 6 → obs 7 |
| ... | ... | ... | ... |
| j=7 | obs 7–12 | obs 8–12 | obs 12 → obs 13 |

Mẫu j=0 tái dùng y nguyên feature đã tính; j>0 tính lại velocity từ vị trí (vẫn chỉ
dùng quá khứ — causal). Kết quả: recursive có 221,752 mẫu train, gấp 8× direct
(27,719) — con số này được log và ghi báo cáo, không giấu. Lúc validation/test thì
recursive vẫn rollout trên đúng window 14-obs như direct để so công bằng.

## 6. File output của preprocess (`.npz`)

Mỗi split 1 file (`train.npz` ~5 MB). Mở bằng `np.load(...).files` sẽ thấy:
- `X_main [N,5,4]`, `X_pos [N,5,2]`, `X_spd [N,5,5]` — input 3 config, đã chuẩn hóa.
  **[5 = 5 mốc thời gian, 4/2/5 = số feature]** — số 4 là số đặc trưng, không phải số bước.
- `Y [N,8,2]` — đáp án lat/lon gốc (chuẩn 0–360°E, chưa chuẩn hóa, để chấm Haversine).
- `Y_direct [N,8,2]` — đáp án dạng km đông/bắc so với điểm gốc (cho model direct học).
- `Y_step [N,8,2]` — đáp án từng bước 6h (cho model recursive học).
- `sample_id` (dạng `1980076N06148__19800610T0000__k3` = bão nào + giờ dự báo + window thứ mấy),
  `sid`, `season`, `last_latlon` (điểm gốc), `hist_latlon` (6 vị trí lịch sử).

`.npz` là binary nén của NumPy: `np.load` 1 dòng là có tensor train ngay, gọn hơn CSV
rất nhiều khi lưu ma trận 3 chiều, lại chốt cứng 1 bộ data cho mọi bước sau (reproducible).

## 7. Model — 6 cấu hình ở `src/models/`

- `recurrent.py` — `RecurrentBackbone`: 1 class chung cho RNN/GRU/LSTM (đọc 5 hàng theo
  thứ tự thời gian, nén thành 1 vector; 1 chiều — không dùng bidirectional vì chiều ngược
  sẽ "đọc trước tương lai"; RNN dùng hàm tanh).
- `heads.py` — 2 "đầu ra": one-step (trả 1 bước `[e, n]` km) và direct (trả 8 bước `[8,2]` km).
  Cả 2 đều là 1 layer linear, giống hệt nhau giữa các backbone để so sánh công bằng.
- `forecasting.py` — cách dùng 2 đầu trên.
- `factory.py` — `build_model("lstm", "direct", input_size=4, hidden_size=64, ...)`
  dựng 1 trong 6 combo. Tham số: rnn ~13k, gru ~39k, lstm ~52k (direct hơn recursive
  đúng 910 = chênh lệch kích cỡ head). `input_size` phải khớp config (A→2, B→5, C→4),
  lệch là code báo lỗi ngay.

**2 chiến lược lúc dự báo** (cùng 1 bài: 5 hàng quá khứ):
- **Direct**: nhìn quá khứ, phun ra luôn cả 8 bước 1 lần (mỗi bước là cặp km đông/bắc
  so với điểm hiện tại). Nhanh, các horizon độc lập.
- **Recursive (autoregressive)**: đoán bước 1 (t+6h) → convert cặp km thành tọa độ
  (công thức Geod xuôi) → tính lại vận tốc từ điểm vừa đoán → ghép vào cuối cửa sổ,
  đẩy hàng cũ nhất ra → đoán bước 2... lặp 8 lần tới t+48h. Giống đi trong sương mù:
  đi 1 đoạn, nhìn la bàn, đi tiếp. **Autoregressive vẫn cần windowing**: windowing là
  cách cắt data thành bài tập (bắt buộc với mọi model), còn autoregressive chỉ mô tả
  cách sinh đáp án lúc inference (tự ăn lại output của mình). Lúc train, recursive vẫn
  học trên window (5 input + 1 target, teacher forcing cho xem đáp án bước đó).

## 8. Output của model và decode (ví dụ số cụ thể)

Giả sử 1 bài test có điểm gốc (vị trí quan trắc cuối cùng model được thấy) là
(20.0°N, 135.0°E). Model direct đoán xong trả về `[B,8,2]` km — lấy batch B=1,
2 bước đầu ra ví dụ là:

```text
out[0, 0] = [ +45.2, +30.1 ]   # bước t+6h:  đi 45.2 km về đông, 30.1 km về bắc
out[0, 1] = [ +90.5, +55.3 ]   # bước t+12h: đi 90.5 km về đông, 55.3 km về bắc (so với GỐC, không phải so với bước 1)
```

**Đây chưa phải tọa độ** — mới chỉ là "đi bao xa theo hướng nào". Hàm `decode_direct`
dùng công thức Geod xuôi (mục 9) để đổi từng cặp km thành lat/lon, gốc là điểm
(20.0°N, 135.0°E):

```text
bước t+6h:  (20.0°N, 135.0°E) + (45.2 km đông, 30.1 km bắc) → (20.27°N, 135.43°E)
bước t+12h: (20.0°N, 135.0°E) + (90.5 km đông, 55.3 km bắc) → (20.50°N, 135.86°E)
```

Kinh độ output theo chuẩn file 0–360°E, khớp định dạng đáp án `Y` nên so trực tiếp được.

Với model **recursive**, `rollout()` làm đúng quy trình trên nhưng lặp 8 lần và **tự
động**: đoán cặp km bước 1 → decode ra tọa độ → tính lại vận tốc từ điểm vừa đoán →
ghép vào cuối cửa sổ (đẩy hàng cũ nhất ra) → đoán bước 2... Kết quả cuối cũng là
`[B,8,2]` lat/lon, vẽ được ngay không cần decode thêm.

Tóm lại 3 dạng số liệu không bao giờ lẫn vào nhau:
1. **Input model** `[B,5,D]` — đã chuẩn hóa (mean 0), không phải đơn vị vật lý.
2. **Output thô** `[B,8,2]` — km đông/bắc (direct) — đem chấm loss Huber lúc train.
3. **Tọa độ decode** `[B,8,2]` — (lat, lon) độ — đem chấm metric Haversine và vẽ hình.

## 9. Công thức Geod (tính feature + decode)

Geod = `pyproj.Geod`: giải bài toán trắc địa trên **ellipsoid WGS84** (Trái Đất dẹt
2 đầu: bán trục lớn a = 6,378,137 m, độ dẹt f = 1/298.257). 2 bài toán ngược nhau:

- **Inverse** (biết 2 điểm → khoảng cách + hướng): input p0=(15.0°N,130.0°E),
  p1=(15.5°N,131.0°E) → output d ≈ 120.9 km, azimuth α ≈ 62.6°.
- **Forward** (biết điểm + hướng + quãng đường → điểm đến): từ (20.0°N,135.0°E) đi
  hướng 90° xa 100 km → tọa độ mới. Dùng trong rollout recursive và decode direct.

Ý tưởng toán: đưa về "vĩ độ rút gọn" β = atan((1−f)·tan φ) rồi giải lặp Karney tới
hội tụ (thư viện lo phần này). Phiên bản mặt cầu tương đương:

- Ngược (2 điểm → khoảng cách, cũng là công thức Haversine dùng chấm điểm):
  `a = sin²(Δφ/2) + cosφ₁·cosφ₂·sin²(Δλ/2)`, `d = 2R·asin(√a)` với R = 6371.0088 km.
- Xuôi (điểm + hướng + quãng đường → điểm đến), với δ = d/R:
  `φ₂ = asin(sin φ₁·cos δ + cos φ₁·sin δ·cos α)`,
  `λ₂ = λ₁ + atan2(sin α·sin δ·cos φ₁, cos δ − sin φ₁·sin φ₂)`.
  Geod làm đúng việc này nhưng trên ellipsoid (độ cong thay đổi theo vĩ độ) nên chính
  xác hơn vài phần nghìn — quan trọng khi cộng dồn 8 bước rollout.

Cặp `(e,n)` chỉ là chiếu lên mặt phẳng tiếp
xúc tại điểm gốc: `e = d·sinα`, `n = d·cosα` (ngược lại `d = √(e²+n²)`, `α = atan2(e,n)`).

## 10. Metric — tính thế nào (mọi thứ ra km)

Với mỗi bài, tính Haversine(điểm đoán, điểm thật) ở từng bước → 8 con số. Ví dụ 3 bài
(chỉ viết 4 horizon báo cáo):

| window | +6h | +12h | +24h | +48h |
|---|---|---|---|---|
| S1-bài1 | 40 | 86 | 196 | 493 |
| S1-bài2 | 36 | 79 | 190 | 480 |
| S2-bài1 | 45 | 95 | 210 | 470 |

- **MTE(+6h)** = trung bình cột 6h = (40+36+45)/3 ≈ 40.3 km. Tương tự +12h/+24h/+48h.
- **ADE** = trung bình toàn bảng = 2,420/12 ≈ 201.7 km ("điểm tổng kết" cả trajectory).
- **FDE** = trung bình cột cuối = 481 km — chính là MTE(+48h) theo định nghĩa.
- **Median/P90**: xếp cột +48h [470, 480, 493] → median = 480 (đứng giữa, không bị vài
  bài sai khủng kéo lên như mean), P90 = ngưỡng mà 90% bài nằm dưới.
- **Theo storm**: bão S1 có 2 bài → ADE(S1) = 200; S2 → 205. Quy ước chính là trung bình
  theo window, bảng per-storm đi kèm để bão dài không áp đảo.
- Mọi con số đều kèm `n` (bao nhiêu mẫu đóng góp); ô invalid phải khai báo mask rõ ràng —
  không được lặng lẽ bỏ qua làm thay đổi tập mẫu giữa các model.

Kết quả thật trên test: recursive MTE +6h ~38 km vs direct ~45 km; +48h ~480–505 km;
ADE ~245–250 km (baseline "đứng yên" là 538 km — model giảm hơn một nửa).

## 11. Train / test chạy thế nào

**Train 1 model** (`scripts/train.py` → `src/training/trainer.py`):
mỗi epoch cho model xem hết 27,719 bài train theo batch 64; đoán sai bao nhiêu km thì
phạt bấy nhiêu theo **loss Huber** (δ = 50 km: sai số nhỏ phạt kiểu bình phương, sai số
khổng lồ phạt kiểu tuyến tính cho đỡ nổ gradient). Hết epoch đo ADE trên val
(recursive thì rollout thật 8 bước, không cho xem đáp án). Giữ checkpoint nào val ADE
thấp nhất (`best.pt`); 15 epoch liền không cải thiện thì dừng (early stopping). Kèm
ReduceLROnPlateau + clip grad 1.0. Mỗi run lưu `best.pt/last.pt/history.json/config.json`
vào `experiments/checkpoints/<arch>_<strategy>_<feat>_s<seed>/`.

**Test** (`scripts/evaluate.py`, hoặc `run_experiments.py` làm hộ cho cả 6 models):
load đúng `best.pt` → đoán 2,207 bài test (bão 2021–2024, chưa từng thấy) → tính
MTE/ADE/FDE → ra CSV + 6 hình + `experiments/report.md`. Test chỉ chạy 1 lần sau khi
đã chốt model — không dùng test để chọn model.

## 12. Code mẫu cho người tiếp theo (copy-paste chạy được từ root repo)

```python
import torch, joblib
from src.data.dataset import HurricaneTrackDataset
from src.models.factory import build_model
from src.training.trainer import load_checkpoint
from src.models.forecasting import decode_direct

# 1. DATA: load 1 batch (cần chặng preprocess xong)
ds = HurricaneTrackDataset('data/processed/test.npz', config='C',
                           target='direct', return_origin=True)
batch = [ds[i] for i in range(8)]
X = torch.stack([b['X'] for b in batch])          # [8,5,4] input chuẩn hóa
Y = torch.stack([b['y'] for b in batch])          # [8,8,2] đáp án (km)
origins = torch.stack([b['origin_latlon'] for b in batch]).numpy()  # [8,2] điểm gốc

# 2a. DỰNG MODEL MỚI để train
model = build_model('lstm', 'direct', input_size=4)   # 52240 params
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.HuberLoss(delta=50.0)
model.train(); loss = loss_fn(model(X), Y); loss.backward(); opt.step()

# 2b. LOAD MODEL ĐÃ TRAIN để visualize (cần chặng train xong)
model, ckpt = load_checkpoint(
    'experiments/checkpoints/lstm_direct_C_s0/best.pt', device='cpu')
model.eval()

# 3. OUTPUT để vẽ hình
with torch.no_grad():
    out = model(X)                                # direct: [8,8,2] km đông/bắc
geo = decode_direct(out.numpy(), origins)         # [8,8,2] (lat, lon) -> vẽ ngay
# recursive thì thay model(X) bằng model.rollout(X) -> ra thẳng lat/lon
```

---

## 13. FAQ nhanh (từ các câu đã hỏi)

- **Autoregressive có cần windowing không?** Có, bắt buộc. Windowing = cách cắt data
  thành bài tập (mọi model đều cần); autoregressive chỉ mô tả cách inference
  (tự ăn lại output). Lúc train recursive vẫn học trên window 5 input + 1 target.
- **Input 4 là gì, sao có chỗ input 2?** `[5, 4]` = 5 mốc thời gian × 4 đặc trưng
  (config C). Config A có 2 đặc trưng `[LAT, LON]`, B có 5. `input_size` lúc dựng model
  phải khớp config.
- **48h lấy ở đâu?** Bước thứ 8 = 8 × 6h. Đáp án là quan trắc thật đã giấu, xong lôi ra chấm.
- **Train/test có dùng chung 14 obs không?** Không. Cách cắt window giống hệt nhau,
  nhưng train = bão cũ (học), test = bão mới 2021–2024 (chấm).
- **FDE vs MTE(+48h)?** Bằng nhau theo định nghĩa (kiểm chứng chênh ~1e-12 do thứ tự
  cộng số thực).
- **File `.npz` là gì?** Binary nén của NumPy: 1 file/split chứa sẵn tensor + metadata,
  `np.load` 1 dòng là train được ngay.
