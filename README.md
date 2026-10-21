# Hurricane-Prediction — Dự báo quỹ đạo bão Tây Thái Bình Dương

Đồ án Data Science: dự báo vị trí tâm bão 8 bước (6h → 48h) từ dữ liệu
NOAA IBTrACS v04r01, dùng RNN / GRU / LSTM với 2 chiến lược direct và recursive.

Tài liệu chi tiết (tiếng Việt, giải thích từng khái niệm kèm ví dụ số):
**[`docs/pipeline_guide.md`](docs/pipeline_guide.md)** — đọc file này trước.

## 1. Cài đặt môi trường

**Quan trọng: mọi lệnh dưới đây đều chạy trong môi trường `machine-learning`.**
Mở terminal mới là phải activate lại (terminal không nhớ env giữa các lần mở):

```bash
conda activate machine-learning
cd "C:\Users\manh\Documents\Studying\master\Introduction to Data Science\Hurricane-Prediction"
```

Kiểm tra đã đúng env chưa (phải in ra đường dẫn có `envs\machine-learning`):

```bash
python -c "import sys; print(sys.executable)"
```

Cài dependencies (chỉ cần 1 lần, đã cài sẵn trên máy này):

```bash
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu   # bản CPU
```

## 2. Chuẩn bị data (~25 MB, up git được)

| File | Dung lượng | Lấy ở đâu |
|---|---|---|
| `data/IBTrACS.WP.v04r01.nc` | 8.6 MB | Đã có sẵn trong repo. Tải lại tại [NOAA IBTrACS v04r01](https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access/netcdf/) nếu thiếu, đặt vào `data/` |
| `data/processed/` | ~16 MB | **Tự sinh** bằng chặng 1 dưới đây (không cần tải) |
| `experiments/checkpoints/` | ~7.4 MB | **Tự sinh** khi train (đã kèm sẵn `best.pt` để eval/demo không cần train lại) |

Tổng repo < 50 MB — dưới xa giới hạn 100 MB/file của GitHub nên commit toàn bộ được
(trừ `__pycache__`, đã có `.gitignore`).

## 3. Chạy pipeline (đúng thứ tự)

```bash
conda activate machine-learning

# Chặng 1 — tiền xử lý: .nc -> .npz (1 lần, ~1 phút)
python scripts/preprocess.py --config configs/data.yaml

# Chặng 2 — train (mỗi model 1 lần, CPU ~2–5 phút/con direct, ~5–15 phút/con recursive)
python scripts/train.py --model lstm --strategy direct
# hoặc chạy cả cụm 6 models (~20–40 phút):
python scripts/run_experiments.py --experiment main

# Chặng 3 — đánh giá + thí nghiệm
python scripts/evaluate.py --checkpoint experiments/checkpoints/lstm_direct_C_s0/best.pt --split test
python scripts/run_experiments.py --experiment ablation    # so feature A/B/C
python scripts/run_experiments.py --experiment movement   # straight vs turning
```

Kết quả: `data/processed/`, `experiments/checkpoints/`, `experiments/results/*.csv`,
`experiments/figures/*.png`, `experiments/report.md`.

## 4. Dùng code cơ bản (load data → model → output)

```python
import torch
from src.data.dataset import HurricaneTrackDataset
from src.models.factory import build_model
from src.training.trainer import load_checkpoint
from src.models.forecasting import decode_direct

# 1. Load 1 batch (cần data/processed/ từ chặng 1)
ds = HurricaneTrackDataset('data/processed/test.npz', config='C',
                           target='direct', return_origin=True)
X = torch.stack([ds[i]['X'] for i in range(8)])                 # [8,5,4] input
origins = torch.stack([ds[i]['origin_latlon'] for i in range(8)]).numpy()  # [8,2]

# 2a. Dựng model mới để train: config C -> input_size=4 (A->2, B->5)
model = build_model('lstm', 'direct', input_size=4)            # 52240 params

# 2b. Load model đã train xong (cần chặng 2)
model, ckpt = load_checkpoint(
    'experiments/checkpoints/lstm_direct_C_s0/best.pt', device='cpu')
model.eval()

# 3. Output: direct -> [8,8,2] km, decode ra lat/lon rồi vẽ
with torch.no_grad():
    geo = decode_direct(model(X).numpy(), origins)             # [8,8,2] (lat, lon)
# recursive: model.rollout(X) -> thẳng [8,8,2] lat/lon, không cần decode
```

## 5. Kiểm thử + cấu trúc thư mục

```bash
python -m pytest tests/ -q        # 101 tests, phải pass hết
```

```text
Hurricane-Prediction/
├── requirements.txt          # dependencies (cài 1 lần)
├── configs/                  # data.yaml, models.yaml, training.yaml
├── data/                     # .nc gốc + processed/ (.npz + scalers.joblib)
├── scripts/                  # preprocess.py, train.py, evaluate.py,
│                             # run_experiments.py, demo_models.py (eda/, quickstart mẫu)
├── src/
│   ├── data/                 # io, preprocessing, features, windowing, scaling,
│   │                         # dataset (HurricaneTrackDataset), geometry
│   ├── models/               # recurrent, heads, forecasting, factory
│   ├── evaluation/           # metrics (Haversine/MTE/ADE/FDE), evaluator, analysis
│   └── training/             # data, trainer (Huber+early stopping), inference, seed
├── tests/                    # test_preprocessing/models/metrics/training
├── EDA/                      # figures + tables + report.md (Task 1)
├── docs/                     # pipeline_guide.md (tiếng Việt), preprocessing_report.md,
│                             # evaluation.md, models.md
└── experiments/              # checkpoints/, results/*.csv, figures/01-06, report.md
```

## 6. Kết quả chính (test 2021–2024, N=2,207)

ADE ~245–250 km, FDE (+48h) ~479–505 km cho cả 6 models (baseline "đứng yên":
ADE 538 km). Recursive tốt hơn ở +6h (~38 vs ~45 km), direct nhỉnh hơn ở +48h.
Chi tiết: [`experiments/report.md`](experiments/report.md).
