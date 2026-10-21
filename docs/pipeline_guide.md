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
| `BASIN` | Biển đang hoạt động, **tính theo từng dòng**: WP / EP / NI | WP |
| `TRACK_TYPE` | Loại track: `main` (chính) hay spur-merge/split/other, PROVISIONAL... | main |
| `STORM_SPEED` / `STORM_DIR` | Tốc độ/hướng **có sẵn trong file — nhưng KHÔNG dùng** (xem mục 3) | — |

## 2. Load data (`scripts/eda/io.py`)

File `.nc` lưu dạng lưới 2D `(bão × 360 ô thời gian)`, mỗi bão chỉ dùng `numobs` ô đầu.
Hàm `load_ibtracs` chỉ lấy đúng các ô thật (bỏ padding), chuyển fill value `-9999`
thành NaN, parse giờ UTC, trả về 1 bảng long-format 249,084 dòng. File gốc **không bao
giờ bị ghi đè** — pipeline chỉ đọc.

## 3. Filter — lọc từng bước (F1–F7, `src/data/preprocessing.py`)

Ví dụ 1 cơn bão trước khi lọc:

| obs | giờ UTC | LAT | LON | BASIN | TRACK_TYPE |
|---|---|---|---|---|---|
| 0 | 01/08 00:00 | 15.0 | 130.0 | WP | main |
| 1 | 01/08 06:00 | 15.5 | 131.0 | WP | main |
| 2 | 01/08 12:00 | 16.1 | 132.2 | WP | main |
| 3 | 01/08 18:00 | 16.8 | 133.0 | EP | main |

| Bước | Quy tắc | Còn lại | Vì sao |
|---|---|---|---|
| F1 season | Giữ bão bắt đầu 1980–2024 | 104,927 dòng / 1,471 bão | Khung đề bài |
| F2 basin | Giữ dòng `BASIN == 'WP'` (dòng số 3 ở trên bị bỏ) | 100,970 | Chỉ dự báo ở WP; bão trôi sang biển khác thì cắt khúc tại đó |
| F3 track | Giữ `TRACK_TYPE == 'main'` | 100,386 / 1,446 bão | Spur/PROVISIONAL là nhánh rẽ hoặc số liệu tạm, không đáng tin |
| F4 giờ chuẩn | Giữ mốc 00/06/12/18 UTC | 50,674 | Chuẩn hóa nhịp 6 tiếng |
| F5 hợp lệ | LAT ∈ [−90,90], LON ∈ [0,360], đủ giờ giấc | 50,674 | (thực tế không rớt dòng nào) |
| F6 sort+dedup | Xếp theo (SID, giờ); trùng giờ giữ dòng đầu | 0 dòng trùng | Quy tắc deterministic |
| F7 segment | Cắt khúc ở chỗ nhảy giờ ≠ 6h (ví dụ thiếu mốc 06:00) | 1,451 segments | Không nội suy qua khoảng trống; window không bao giờ vắt qua chỗ đứt |

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

**3 config feature** (cùng 1 mốc thời gian, chỉ khác cách đóng gói):
- **A** `[15.5, 131.0]` — chỉ vị trí (2 số).
- **B** `[15.5, 131.0, 20.2, 0.887, 0.461]` — vị trí + tốc độ + hướng mã hóa sin/cos
  (vì 359° và 1° thực ra gần nhau; đưa số độ thô vào model sẽ hiểu nhầm là xa nhau).
- **C (chính)** `[LAT, LON, v_E, v_N]` (4 số).

**Chuẩn hóa**: mỗi cột trừ mean rồi chia std — mean/std này tính **chỉ từ tập train**
(ví dụ LAT mean 18.36, std 8.43 → 15.5 thành (15.5−18.36)/8.43 ≈ −0.34). Val/test dùng
y nguyên bộ số này, không tính lại (tính lại là leak). Lưu ở `scalers.joblib`, có hàm
`invert` để đổi ngược về đơn vị vật lý.

## 5. Windowing — cắt chuỗi thành từng "bài tập"

Sau filter, mỗi segment là chuỗi dài (ví dụ 20 obs liên tiếp cách nhau đúng 6h).
Model không ăn cả chuỗi mà ăn từng **cửa sổ trượt dài 14 obs**. Xét cửa sổ bắt đầu
tại obs k (lấy k = 0 cho dễ):

| obs | 0 | 1 | 2 | 3 | 4 | 5 | 6 | … | 13 |
|---|---|---|---|---|---|---|---|---|---|
| vai trò | bàn đạp t−30 | input t−24 | input t−18 | input t−12 | input t−6 | input t (gốc) | target t+6 | … | target t+48 |

- **6 vị trí lịch sử** (obs 0–5) nhưng chỉ **5 hàng input** (obs 1–5): hàng input đầu
  tiên (obs 1) cần vận tốc tính từ obs 0 → obs 1, nên obs 0 ở mốc t−30 chỉ làm "bàn đạp",
  không đưa vào model. **Đó là lý do cần 6 vị trí mà output chỉ 5 hàng.**
- **8 target** (obs 6–13): vị trí thật ở tương lai, lúc train/inference bị giấu, xong mới
  lôi ra chấm.
- Trượt k → 1, 2, 3... Segment dài L obs cho ra **L − 13 windows** (20 obs → 7 windows).
  Cộng lại: train 27,719 / val 2,499 / test 2,207 windows.
- Quy tắc sắt: window không bao giờ vắt qua 2 bão, qua chỗ mất mốc giờ, qua biên basin,
  hay qua ranh train/val/test — vì mỗi window nằm gọn trong 1 segment.

**Riêng lúc train model recursive**: mỗi window 14-obs được suy ra thêm 8 mẫu one-step
(gồm 6 obs lịch sử → 5 hàng input + 1 target là bước 6h ngay sau đó, bắt đầu tại obs
0, 1, ..., 7) → 221,752 mẫu (gấp 8× direct). Lúc validation/test thì recursive vẫn
rollout trên đúng window 14-obs như direct để so công bằng.

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

## 8. Output của model và decode

- Direct output `[B,8,2]` km (cột 0 = đông, cột 1 = bắc, so với điểm gốc) — **chưa phải
  tọa độ**, phải qua `decode_direct(out, origins)` mới thành `(lat, lon)` chuẩn 0–360°E.
- Recursive `rollout()` trả thẳng `[B,8,2]` lat/lon, vẽ được ngay.
- Ba biểu diễn không bao giờ trộn lẫn: input chuẩn hóa / displacement km / tọa độ địa lý.

## 9. Công thức Geod (tính feature + decode)

Geod = `pyproj.Geod`: giải bài toán trắc địa trên **ellipsoid WGS84** (Trái Đất dẹt
2 đầu: bán trục lớn a = 6,378,137 m, độ dẹt f = 1/298.257). 2 bài toán ngược nhau:

- **Inverse** (biết 2 điểm → khoảng cách + hướng): input p0=(15.0°N,130.0°E),
  p1=(15.5°N,131.0°E) → output d ≈ 120.9 km, azimuth α ≈ 62.6°.
- **Forward** (biết điểm + hướng + quãng đường → điểm đến): từ (20.0°N,135.0°E) đi
  hướng 90° xa 100 km → tọa độ mới. Dùng trong rollout recursive và decode direct.

Ý tưởng toán: đưa về "vĩ độ rút gọn" β = atan((1−f)·tan φ) rồi giải lặp Karney tới
hội tụ (thư viện lo phần này). Phiên bản mặt cầu tương đương (cũng là công thức
Haversine dùng chấm điểm): `a = sin²(Δφ/2) + cosφ₁·cosφ₂·sin²(Δλ/2)`,
`d = 2R·asin(√a)` với R = 6371.0088 km. Cặp `(e,n)` chỉ là chiếu lên mặt phẳng tiếp
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
