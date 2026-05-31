# YÊU CẦU THỰC HIỆN CODE PROJECT CTR PREDICTION TRÊN AVAZU DATASET

## 0. Mục tiêu của file này

File này là đặc tả yêu cầu để đưa vào AI Code nhằm sinh source code cho project Deep Learning dự đoán Click-Through Rate trên dataset **Avazu Click-Through Rate Prediction**.

Project cần được viết theo dạng **module hóa**, chạy được trên **Kaggle Notebook** với phần cứng:

- Platform: Kaggle
- Framework: Pytorch
- GPU: NVIDIA Tesla T4 x2
- VRAM: 16GB/GPU
- RAM hệ thống: giới hạn theo Kaggle
- Dataset chính: Avazu Click-Through Rate Prediction
- File dữ liệu chính: `train.gz`
- Vấn đề chính cần giải quyết: file `.gz` khoảng 1GB, khi giải nén/lưu DataFrame toàn bộ vào RAM có thể làm Kaggle stop session.

---

# 1. Bối cảnh nghiên cứu

Click-through rate prediction là bài toán dự đoán xác suất người dùng click vào quảng cáo. Đây là bài toán **binary classification** với nhãn:

```text
click = 1 nếu người dùng click
click = 0 nếu người dùng không click
```

Dataset Avazu có đặc trưng:

- Dữ liệu quảng cáo mobile.
- Khoảng 40 triệu bản ghi.
- Khoảng 23 feature fields.
- Phần lớn là feature categorical, sparse, high-cardinality.
- Phù hợp với các mô hình CTR như:
  - Logistic Regression
  - Factorization Machine
  - DeepFM
  - xDeepFM
  - AutoInt
  - NAM
  - NAFI
  - KD-NAFI

Mục tiêu code là xây dựng một pipeline có thể:

1. Đọc dữ liệu `.gz` lớn an toàn trên Kaggle.
2. Tiền xử lý dữ liệu theo batch/chunk.
3. Convert dữ liệu sang format nhẹ hơn.
4. Train mô hình Deep Learning dạng module.
5. Đánh giá bằng AUC và Logloss.
6. Hỗ trợ baseline và mô hình chính theo hướng NAM + FIN.
7. Dễ mở rộng sang KD-NAFI.

---

# 2. Dataset sử dụng

## 2.1 Dataset chính

Tên dataset:

```text
Avazu Click-Through Rate Prediction
```

File chính:

```text
train.gz
```

Các cột thường gặp:

```text
id
click
hour
C1
banner_pos
site_id
site_domain
site_category
app_id
app_domain
app_category
device_id
device_ip
device_model
device_type
device_conn_type
C14
C15
C16
C17
C18
C19
C20
C21
```

## 2.2 Cột target

```text
click
```

## 2.3 Cột cần loại bỏ hoặc xử lý đặc biệt

```text
id
```

Yêu cầu:

- Không đưa `id` vào mô hình theo mặc định.
- Có thể dùng `id` để debug hoặc xuất prediction.
- Cột `hour` cần tách thêm:
  - day
  - hour_of_day
  - weekday nếu có thể parse.

---

# 3. Vấn đề kỹ thuật cần giải quyết

## 3.1 Không được load toàn bộ file `.gz` vào RAM

Không được dùng trực tiếp:

```python
pd.read_csv("train.gz")
```

vì có thể gây:

- Full RAM.
- Kaggle stop session.
- Kernel chết khi preprocessing.
- Không thể train ổn định.

## 3.2 Bắt buộc đọc dữ liệu theo chunk

Yêu cầu bắt buộc:

```python
pd.read_csv(
    path,
    compression="gzip",
    chunksize=...
)
```

Gợi ý chunksize:

```text
100_000 đến 500_000 dòng/chunk
```

Tùy RAM thực tế.

## 3.3 Bắt buộc tạo pipeline convert trung gian

Cần có bước convert từ `.gz` sang format dễ đọc hơn:

Ưu tiên:

```text
Parquet
```

Hoặc:

```text
Feather
```

Yêu cầu:

- Đọc từng chunk từ `.gz`.
- Tiền xử lý tối thiểu từng chunk.
- Ghi từng partition `.parquet`.
- Không giữ toàn bộ dataset trong RAM.
- Sau đó train từ các file parquet partition.

Ví dụ output:

```text
data/processed/train_part_000.parquet
data/processed/train_part_001.parquet
data/processed/train_part_002.parquet
...
```

---

# 4. Chiến lược chống tràn RAM trên Kaggle

## 4.1 Dtype optimization

Khi đọc chunk, cần ép kiểu:

```python
dtype = {
    "click": "int8",
    "C1": "int16",
    "banner_pos": "int8",
    "device_type": "int8",
    "device_conn_type": "int8",
    "C14": "int32",
    "C15": "int16",
    "C16": "int16",
    "C17": "int16",
    "C18": "int8",
    "C19": "int16",
    "C20": "int32",
    "C21": "int16"
}
```

Các cột categorical dạng string nên xử lý bằng:

- Hashing trick.
- Label encoding incremental.
- Frequency encoding.
- Category mapping lưu ra disk.

Không nên giữ object dtype quá lâu.

## 4.2 Không one-hot full dataset

Không được dùng:

```python
pd.get_dummies(full_dataframe)
```

Thay vào đó dùng:

- Embedding layer.
- Hash bucket.
- Label encoding.
- Frequency threshold.
- Rare category handling.

## 4.3 Hashing trick cho categorical feature

Yêu cầu hỗ trợ chế độ hash:

```text
feature_value -> hash(feature_value) % num_buckets
```

Gợi ý bucket:

```yaml
hash_buckets:
  site_id: 100000
  site_domain: 100000
  site_category: 100
  app_id: 100000
  app_domain: 100000
  app_category: 100
  device_id: 500000
  device_ip: 1000000
  device_model: 200000
  default: 100000
```

Ưu điểm:

- Không cần fit toàn bộ vocabulary vào RAM.
- Chạy được streaming.
- Phù hợp high-cardinality categorical features.

Nhược điểm:

- Có collision.
- Cần seed cố định để reproducible.

## 4.4 Chế độ sample/debug

Project cần có các mode:

```text
debug
small
full
```

Trong đó:

```yaml
debug:
  max_rows: 200000
  epochs: 1

small:
  max_rows: 3000000
  epochs: 2

full:
  max_rows: null
  epochs: 3-5
```

Mục tiêu:

- Test code nhanh.
- Không crash Kaggle.
- Có thể scale dần.

---

# 5. Cấu trúc project bắt buộc

Project cần viết dạng module, không viết toàn bộ trong một notebook duy nhất.

Cấu trúc thư mục yêu cầu:
Chỉ cần code ra model NAFI
```text
ctr-avazu-nafi/
│
├── README.md
├── requirements.txt
├── config/
│   ├── default.yaml
│   ├── kaggle_t4x2.yaml
│   └── debug.yaml
│
├── notebooks/
│   ├── 01_prepare_data.ipynb
│   ├── 02_train_baseline.ipynb
│   ├── 03_train_nafi.ipynb
│   └── 04_evaluate.ipynb
│
├── src/
│   ├── __init__.py
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── schema.py
│   │   ├── chunk_reader.py
│   │   ├── preprocess.py
│   │   ├── parquet_converter.py
│   │   ├── dataset.py
│   │   └── dataloader.py
│   │
│   ├── features/
│   │   ├── __init__.py
│   │   ├── hashing.py
│   │   ├── encoding.py
│   │   └── feature_map.py
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── embedding.py
│   │   ├── mlp.py
│   │   ├── lr.py
│   │   ├── fm.py
│   │   ├── deepfm.py
│   │   ├── autoint.py
│   │   ├── nam.py
│   │   ├── fin.py
│   │   ├── nafi.py
│   │   └── kd_nafi.py
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py
│   │   ├── losses.py
│   │   ├── metrics.py
│   │   ├── callbacks.py
│   │   └── checkpoint.py
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── evaluate.py
│   │   ├── plots.py
│   │   └── report.py
│   │
│   └── utils/
│       ├── __init__.py
│       ├── config.py
│       ├── logger.py
│       ├── seed.py
│       ├── memory.py
│       └── kaggle.py
│
├── scripts/
│   ├── prepare_avazu.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   └── run_kaggle.sh
│
├── outputs/
│   ├── checkpoints/
│   ├── logs/
│   ├── metrics/
│   ├── figures/
│   └── submissions/
│
└── tests/
    ├── test_chunk_reader.py
    ├── test_hashing.py
    ├── test_dataset.py
    └── test_models.py
```

---

# 6. File config yêu cầu

## 6.1 `config/kaggle_t4x2.yaml`

```yaml
project:
  name: ctr-avazu-nafi
  seed: 42
  mode: full

environment:
  platform: kaggle
  accelerator: gpu
  gpu: "T4 x2"
  precision: fp16
  num_workers: 2
  pin_memory: true

paths:
  raw_train_gz: /kaggle/input/avazu-ctr-prediction/train.gz
  processed_dir: /kaggle/working/data/processed
  output_dir: /kaggle/working/outputs

data:
  dataset: avazu
  target_col: click
  id_col: id
  compression: gzip
  chunksize: 250000
  train_ratio: 0.8
  valid_ratio: 0.1
  test_ratio: 0.1
  convert_to_parquet: true
  parquet_rows_per_file: 1000000
  drop_id: true
  parse_hour: true
  use_hashing: true

features:
  categorical_cols:
    - hour
    - C1
    - banner_pos
    - site_id
    - site_domain
    - site_category
    - app_id
    - app_domain
    - app_category
    - device_id
    - device_ip
    - device_model
    - device_type
    - device_conn_type
    - C14
    - C15
    - C16
    - C17
    - C18
    - C19
    - C20
    - C21
  derived_cols:
    - day
    - hour_of_day
  hash_bucket_default: 100000
  hash_buckets:
    device_ip: 1000000
    device_id: 500000
    device_model: 200000
    site_id: 100000
    site_domain: 100000
    app_id: 100000
    app_domain: 100000
    site_category: 100
    app_category: 100

model:
  name: nafi
  embedding_dim: 16
  hidden_units: [128, 64, 32]
  dropout: 0.2
  l2_reg: 0.00001

nam:
  enabled: true
  hidden_units: [32, 16]
  activation: exu
  dropout: 0.1

fin:
  enabled: true
  num_heads: 4
  num_layers: 2
  attention_dropout: 0.1
  use_residual: true

training:
  batch_size: 2048
  epochs: 5
  optimizer: adam
  learning_rate: 0.001
  loss: bce_with_logits
  gradient_accumulation_steps: 1
  mixed_precision: true
  max_grad_norm: 5.0
  early_stopping:
    enabled: true
    monitor: valid_auc
    patience: 2
    mode: max

evaluation:
  metrics:
    - auc
    - logloss
  save_predictions: true
  save_plots: true

checkpoint:
  save_best_only: true
  monitor: valid_auc
  mode: max
```

---

# 7. Yêu cầu data pipeline

## 7.1 Module `chunk_reader.py`

Cần implement class:

```python
class GzipChunkReader:
    def __init__(self, path, chunksize, dtype_map=None):
        ...

    def __iter__(self):
        ...
```

Yêu cầu:

- Đọc file `.gz` theo chunk.
- Không load full RAM.
- Có logging số dòng đã đọc.
- Có try/except để báo lỗi rõ.
- Có memory usage report sau mỗi chunk.

## 7.2 Module `preprocess.py`

Cần implement:

```python
def optimize_dtypes(df):
    ...

def parse_hour_column(df):
    ...

def clean_chunk(df, config):
    ...
```

Yêu cầu:

- Ép dtype.
- Tách `hour` thành `day` và `hour_of_day`.
- Fill missing value bằng `"__MISSING__"`.
- Convert categorical sang string trước khi hash nếu cần.
- Drop `id` nếu config yêu cầu.

## 7.3 Module `hashing.py`

Cần implement:

```python
def stable_hash(value: str, num_buckets: int, seed: int = 42) -> int:
    ...

def hash_categorical_column(series, num_buckets, seed=42):
    ...

def hash_features(df, feature_cols, hash_buckets, default_bucket):
    ...
```

Yêu cầu:

- Hash phải ổn định giữa các lần chạy.
- Không dùng Python `hash()` mặc định vì không reproducible.
- Nên dùng `hashlib.md5` hoặc `xxhash`.

## 7.4 Module `parquet_converter.py`

Cần implement:

```python
def convert_gz_to_parquet(raw_path, output_dir, config):
    ...
```

Yêu cầu:

- Đọc chunk.
- Preprocess chunk.
- Hash categorical.
- Ghi parquet partition.
- Không concat full dataset.
- Có resume nếu một số part đã tồn tại.
- Có log số dòng, RAM, thời gian.

---

# 8. Yêu cầu PyTorch Dataset và DataLoader

## 8.1 Dataset đọc nhiều parquet file

Cần implement:

```python
class AvazuParquetDataset(torch.utils.data.IterableDataset):
    def __init__(self, parquet_files, feature_cols, target_col):
        ...

    def __iter__(self):
        ...
```

Yêu cầu:

- Đọc từng parquet file.
- Yield batch hoặc row tensor.
- Không load toàn bộ dataset vào RAM.
- Shuffle ở mức file và batch nếu có thể.
- Convert feature sang `torch.long`.
- Convert label sang `torch.float32`.

## 8.2 Collate function

```python
def ctr_collate_fn(batch):
    ...
```

Output:

```python
{
  "x": LongTensor[batch_size, num_fields],
  "y": FloatTensor[batch_size]
}
```

---

# 9. Yêu cầu mô hình

## 9.1 Model baseline bắt buộc

Chỉ cần model NAFI:

```text
NAFI
```


## 9.2 Embedding layer chung

File:

```text
src/models/embedding.py
```

Cần class:

```python
class FeatureEmbedding(nn.Module):
    def __init__(self, field_dims, embedding_dim):
        ...

    def forward(self, x):
        ...
```

Input:

```text
x shape = [batch_size, num_fields]
```

Output:

```text
embeddings shape = [batch_size, num_fields, embedding_dim]
```

## 9.3 NAM module

File:

```text
src/models/nam.py
```

Yêu cầu:

- Mỗi feature field có một small neural network riêng.
- Output của các feature network được cộng lại.
- Có thể dùng MLP nhỏ thay ExU nếu cần đơn giản hóa.
- Trả về:
  - logit NAM
  - optional feature contribution cho interpretability.

Pseudo:

```python
nam_logit = sum(f_i(embedding_i) for i in fields)
```

## 9.4 FIN module

File:

```text
src/models/fin.py
```

Yêu cầu:

- Dùng Multi-Head Self-Attention.
- Có residual connection.
- Có layer norm nếu cần.
- Có pooling hoặc flatten.
- Output logit FIN.

Pseudo:

```python
attn_out = multihead_attention(embeddings)
res = relu(attn_out + residual_projection(embeddings))
fin_logit = mlp(flatten(res))
```

## 9.5 NAFI model

File:

```text
src/models/nafi.py
```

Yêu cầu:

```python
class NAFI(nn.Module):
    def __init__(...):
        ...

    def forward(self, x):
        ...
```

Công thức mong muốn:

```text
NAFI(x) = NAM(x) + FIN(x)
prediction = sigmoid(NAFI(x))
```

Trong training nên trả về logits, không sigmoid trực tiếp, để dùng:

```python
BCEWithLogitsLoss
```

Output forward:

```python
{
  "logits": logits,
  "nam_logits": nam_logits,
  "fin_logits": fin_logits,
  "attention_weights": optional,
  "feature_contributions": optional
}
```



Config:

```yaml
distillation:
  enabled: false
  temperature: 4.0
  alpha: 0.5
  teacher_models:
    - deepfm
    - autoint
```

---

# 10. Yêu cầu training pipeline

## 10.1 Trainer

File:

```text
src/training/trainer.py
```

Cần class:

```python
class CTRTrainer:
    def __init__(self, model, train_loader, valid_loader, config):
        ...

    def train_one_epoch(self):
        ...

    def validate(self):
        ...

    def fit(self):
        ...
```

Yêu cầu:

- Mixed precision bằng `torch.cuda.amp`.
- Gradient clipping.
- Early stopping theo validation AUC.
- Save best checkpoint.
- Log train loss, valid AUC, valid Logloss.
- Không giữ toàn bộ prediction trong RAM nếu quá lớn.

## 10.2 Multi-GPU trên Kaggle T4 x2

Yêu cầu:

- Detect số GPU:

```python
torch.cuda.device_count()
```

- Nếu có 2 GPU thì dùng:

```python
torch.nn.DataParallel(model)
```

hoặc hỗ trợ optional DistributedDataParallel.

Lưu ý:

- VRAM của 2 GPU không cộng thành một GPU 32GB.
- Batch size cần phù hợp từng GPU.
- Nếu lỗi CUDA OOM thì giảm batch size hoặc bật gradient accumulation.

## 10.3 Mixed precision

Bắt buộc hỗ trợ:

```python
torch.cuda.amp.autocast()
torch.cuda.amp.GradScaler()
```

Mục tiêu:

- Giảm VRAM.
- Tăng tốc training trên T4.

---

# 11. Yêu cầu metrics

File:

```text
src/training/metrics.py
```

Cần implement:

```python
def compute_auc(y_true, y_pred):
    ...

def compute_logloss(y_true, y_pred):
    ...
```

Metrics chính:

```text
AUC
Logloss
```

Yêu cầu:

- AUC càng cao càng tốt.
- Logloss càng thấp càng tốt.
- Lưu ý với CTR: cải thiện AUC khoảng 0.001 đã có ý nghĩa.

---

# 12. Yêu cầu visualization

File:

```text
src/evaluation/plots.py
```

Cần xuất:

```text
outputs/figures/loss_curve.png
outputs/figures/auc_curve.png
outputs/figures/logloss_curve.png
outputs/figures/feature_importance.png
outputs/figures/attention_heatmap.png
```

Với NAFI cần hỗ trợ:

- Feature contribution từ NAM.
- Attention weights từ FIN.
- Heatmap feature interaction.

---

# 13. Yêu cầu scripts CLI

## 13.1 Chuẩn bị dữ liệu

Command:

```bash
python scripts/prepare_avazu.py --config config/kaggle_t4x2.yaml
```

Nhiệm vụ:

- Đọc `train.gz`.
- Convert sang parquet partitions.
- Sinh metadata:
  - feature columns
  - field dims
  - train/valid/test split
  - number of rows.

## 13.2 Train model

Command:

```bash
python scripts/train.py --config config/kaggle_t4x2.yaml --model nafi
```

Tham số:

```
--model nafi
```

## 13.3 Evaluate

```bash
python scripts/evaluate.py \
  --config config/kaggle_t4x2.yaml \
  --checkpoint outputs/checkpoints/best_model.pt
```

## 13.4 Predict submission

```bash
python scripts/predict.py \
  --config config/kaggle_t4x2.yaml \
  --checkpoint outputs/checkpoints/best_model.pt \
  --output outputs/submissions/submission.csv
```

---

# 14. Yêu cầu logging

Cần log ra console và file:

```text
outputs/logs/train.log
outputs/logs/prepare_data.log
```
Log đầu tiên là số lượng tham số

Mỗi log nên có:

- Time.
- Step.
- Chunk index.
- Row count.
- RAM usage.
- GPU memory.
- Loss.
- AUC.
- Logloss.

Yêu cầu có hàm:

```python
log_memory_usage()
log_gpu_usage()
```

---

# 15. Yêu cầu checkpoint

Checkpoint lưu:

```text
outputs/checkpoints/best_model.pt
outputs/checkpoints/last_model.pt
```

Nội dung checkpoint:

```python
{
  "model_state_dict": ...,
  "optimizer_state_dict": ...,
  "epoch": ...,
  "best_auc": ...,
  "config": ...
}
```

Nếu dùng DataParallel cần xử lý prefix:

```text
module.
```

khi load lại model.

---

# 16. Yêu cầu reproducibility

Bắt buộc có:

```python
def seed_everything(seed):
    ...
```

Set:

```python
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
```

Config seed mặc định:

```yaml
seed: 42
```

---

# 17. Yêu cầu chống lỗi thường gặp trên Kaggle

## 17.1 Lỗi RAM khi đọc `.gz`

Giải pháp bắt buộc:

- Đọc chunk.
- Không concat full.
- Ghi parquet.
- Dùng dtype nhỏ.
- Hash categorical.

## 17.2 Lỗi CUDA OOM

Giải pháp:

- Giảm batch size.
- Bật mixed precision.
- Dùng gradient accumulation.
- Giảm embedding_dim từ 16 xuống 8.
- Giảm num_heads.
- Giảm hidden_units.

## 17.3 Lỗi quá nhiều category

Giải pháp:

- Hashing trick.
- Rare category.
- Không tạo vocab full nếu không đủ RAM.

## 17.4 Lỗi notebook timeout

Giải pháp:

- Chia thành nhiều notebook/script.
- Lưu intermediate parquet.
- Save checkpoint mỗi epoch.
- Có resume training.

---

# 18. Yêu cầu README

README cần có:

```text
1. Giới thiệu project
2. Dataset
3. Cấu trúc source code
4. Cách chạy trên Kaggle
5. Cách prepare data
6. Cách train baseline
7. Cách train NAFI
8. Cách evaluate
9. Kết quả mong đợi
10. Các lỗi thường gặp
```

---

# 19. Kết quả mong đợi

## 19.1 Kết quả tối thiểu

Project chạy thành công ở mode debug:

```text
200,000 dòng
1 epoch
không crash RAM
xuất được AUC và Logloss
```

## 19.2 Kết quả trung bình

Project chạy được mode small:

```text
3,000,000 dòng
2 epoch
train được DeepFM hoặc NAFI
có checkpoint
có metrics
```

## 19.3 Kết quả đầy đủ

Project chạy được mode full:

```text
toàn bộ Avazu train.gz
train/valid/test = 8/1/1
batch_size khoảng 2048
embedding_dim = 16
AUC và Logloss được ghi lại
```

---

# 20. Tiêu chí nghiệm thu code

Code được xem là đạt yêu cầu nếu:

- Không load full `.gz` vào RAM.
- Có chunk reader.
- Có parquet converter.
- Có dataset/dataloader streaming hoặc partition-based.
- Có module models riêng.
- Có ít nhất 3 model: LR/FM, DeepFM hoặc AutoInt, NAFI.
- Có training loop PyTorch.
- Có mixed precision.
- Có hỗ trợ T4 x2.
- Có AUC và Logloss.
- Có checkpoint.
- Có README.
- Có config YAML.
- Chạy được trên Kaggle Notebook.

---

# 21. Prompt yêu cầu AI Code sinh project

Dùng prompt sau để đưa vào AI Code:

```text
Hãy tạo một project Python/PyTorch dạng module cho bài toán Avazu Click-Through Rate Prediction trên Kaggle.

Yêu cầu chính:
- Dataset là Avazu train.gz, file nén gzip khoảng 1GB.
- Không được dùng pd.read_csv toàn bộ file vào RAM.
- Bắt buộc đọc dữ liệu theo chunksize.
- Bắt buộc convert dữ liệu sang nhiều file parquet partition.
- Bắt buộc tối ưu dtype.
- Bắt buộc xử lý categorical high-cardinality bằng hashing trick ổn định.
- Project chạy trên Kaggle GPU T4 x2, mỗi GPU 16GB.
- Có hỗ trợ mixed precision bằng torch.cuda.amp.
- Có hỗ trợ DataParallel nếu phát hiện 2 GPU.
- Code phải module hóa, không viết tất cả trong một notebook.

Cấu trúc project cần có:
- src/data
- src/features
- src/models
- src/training
- src/evaluation
- src/utils
- scripts
- config
- notebooks
- outputs
- tests

Model cần implement:
1. LR baseline
2. FM baseline
3. DeepFM
4. AutoInt
5. NAM
6. NAFI gồm NAM branch + FIN branch

NAFI yêu cầu:
- Embedding layer cho toàn bộ categorical fields.
- NAM branch học contribution riêng từng feature.
- FIN branch dùng multi-head self-attention + residual connection để học feature interaction.
- Output cuối là nam_logits + fin_logits.
- Training dùng BCEWithLogitsLoss.

Data pipeline yêu cầu:
- GzipChunkReader đọc train.gz theo chunk.
- preprocess chunk: optimize dtype, parse hour, drop id, fill missing.
- hash categorical features bằng hashlib/md5 hoặc xxhash.
- ghi parquet partitions.
- AvazuParquetDataset đọc từng parquet partition, không load full dataset.
- DataLoader trả về x LongTensor và y FloatTensor.

Training yêu cầu:
- Trainer class.
- train_one_epoch.
- validate.
- fit.
- AUC và Logloss.
- early stopping theo valid_auc.
- checkpoint best_model.pt và last_model.pt.
- logging RAM/GPU usage.
- config YAML.

CLI scripts:
- scripts/prepare_avazu.py
- scripts/train.py
- scripts/evaluate.py
- scripts/predict.py

README cần hướng dẫn chạy trên Kaggle:
1. prepare data
2. train model
3. evaluate
4. predict submission

Hãy sinh code đầy đủ, sạch, có type hints, logging, docstring, và có xử lý lỗi RAM/OOM.
```

---

# 22. Ghi chú thiết kế quan trọng

Không ưu tiên đạt điểm leaderboard ngay từ đầu. Ưu tiên thứ tự:

```text
1. Không crash Kaggle
2. Pipeline chạy được end-to-end
3. Baseline chạy được
4. NAFI chạy được
5. Có metrics đúng
6. Tối ưu dần AUC/Logloss
7. Thêm KD-NAFI sau
```

Với Kaggle, nên bắt đầu bằng:

```text
mode: debug
```

sau đó tăng lên:

```text
mode: small
```

cuối cùng mới chạy:

```text
mode: full
```
# 23. Tạo thêm script để test model pass