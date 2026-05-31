# Guideline Reimplement Paper NAFI/KD-NAFI

Nguồn chính: `datalab-output-NAM_FIN_CTR.pdf.md` - paper “Accurate and interpretable CTR prediction via distilled neural additive feature interaction network”.

Mục tiêu của file này là tạo một checklist thực tế để tái lập paper trong project hiện tại. Project đã train được NAFI end-to-end, nhưng kết quả hiện tại chưa đạt paper, nên guideline tập trung vào các điểm có khả năng tạo gap.

## 1. Target Cần Hướng Tới

Theo Table 2/Table 4 của paper trên Avazu:

| Model | AUC | Logloss |
|---|---:|---:|
| NAM | 0.7600 | 0.3850 |
| FIN / AutoInt-like | 0.7775 | 0.3831 |
| NAFI | 0.7801 | 0.3751-0.3797 |
| KD-NAFI | 0.7806 | 0.3752 |

Kết quả project hiện tại gần nhất:

```text
NAFI valid_auc    ~= 0.7611
NAFI valid_logloss ~= 0.3908
```

Gap chính:

```text
AUC gap    ~= -0.019
Logloss gap ~= +0.015 đến +0.016
```

Đây là gap lớn, thường không chỉ do model mà còn do preprocessing, split, feature encoding, hyperparameter search hoặc target evaluation setup.

## 2. Kiến Trúc Paper Cần Bám

Paper mô tả NAFI gồm 5 phần:

```text
feature input
feature embedding
NAM layer
FIN layer
output layer
```

Công thức tổng:

```text
NAFI(x) = sigmoid(NAM(x) + FIN(x))
```

Trong training nên dùng logits:

```text
loss = LogLoss(sigmoid(nam_logits + fin_logits), y)
```

Project hiện tại đã đúng điểm này:

```text
logits = nam_logits + fin_logits
loss = BCEWithLogitsLoss(logits, y)
```

## 3. NAM Branch

Paper dùng NAM với ExU:

```text
h(x) = f(exp(w) * (x - b))
```

Checklist:

- [x] Mỗi feature field có một feature network riêng.
- [x] Output từng field là contribution scalar.
- [x] Tổng contribution thành `nam_logits`.
- [x] Đã thêm `activation: exu`.
- [ ] Cần verify ExU implementation có đủ gần NAM gốc không.
- [ ] Cần train NAM độc lập để so với paper NAM Avazu AUC 0.7600, Logloss 0.3850.

Việc `valid_nam_auc` trong NAFI thấp không chứng minh NAM yếu, vì NAM branch trong NAFI được train với loss tổng:

```text
loss = BCE(nam_logits + fin_logits, y)
```

Muốn so với paper, cần train model NAM riêng:

```bash
python -m scripts.train \
  --config config/kaggle_t4x2.yaml \
  --model nam \
  --processed-dir <processed_dir>
```

## 4. FIN Branch

Paper mô tả FIN như feature interaction layer:

```text
multi-head self-attention
dot-product attention
concatenate heads
linear transform
residual connection
relu(attn_out + W_res * e_m)
stack multiple layers
```

Checklist:

- [x] Project có `MultiheadAttention`.
- [x] Có residual connection.
- [x] Có LayerNorm.
- [x] Có flatten + MLP output.
- [ ] Cần verify có `W_res` projection đúng như Eq. (7), hiện tại residual đang là cộng trực tiếp `out + residual`.
- [ ] Cần thử FIN-only model độc lập, không chỉ `fin_logits` tách từ NAFI.
- [ ] Cần tune `num_heads`, `num_layers`, `attention_dropout`.

Paper không công bố rõ:

```text
num_heads
num_layers
attention dropout
FIN hidden size
```

Do đó phải grid search.

## 5. Data Split

Paper nói Criteo và Avazu chia:

```text
train/valid/test = 8/1/1
```

Nhưng paper không nói rõ:

```text
random split
time-based split
hash split
seed
```

Project hiện tại split streaming theo:

```text
global_index % 10000
```

Checklist:

- [x] Tỉ lệ 8/1/1 đúng.
- [x] Không load full dataset.
- [ ] Cần thử thêm split strategy khác:
  - stable hash theo `id`
  - time-based theo `hour`
  - random streaming bằng deterministic hash

Split có thể ảnh hưởng mạnh tới Avazu vì dữ liệu có thời gian.

## 6. Feature Encoding

Paper không mô tả chi tiết encoding Avazu:

```text
hashing?
label encoding?
frequency vocab?
rare category?
```

Project hiện tại dùng hashing trick:

```text
value -> md5(seed:value) % num_buckets
```

Đây là an toàn RAM nhưng có collision. Paper có thể dùng label encoding/full vocabulary, nên kết quả có thể cao hơn.

Checklist:

- [x] Hashing ổn định.
- [x] Bucket config đã chỉnh theo EDA Config B.
- [ ] Cần chạy lại prepare sau khi đổi bucket.
- [ ] Cần thử `vocab + min_frequency/top_k` nếu muốn bám paper hơn.
- [ ] Cần so sánh hashing vs vocab trên small/full.

Thứ tự nên chạy:

```text
1. hashing Config B
2. vocab cho low/mid-cardinality fields
3. frequency vocab cho high-cardinality fields
```

## 7. Hyperparameter Search Theo Paper

Paper công bố:

```text
optimizer: Adam
objective: Logloss / Cross entropy
embedding_dim k: 16
learning_rate search: {0.0001, 0.001, 0.01}
DNN hidden: 3 fully-connected layers, 32 neurons/layer
batch_size Avazu: 2048
dropout search: 0.1 -> 0.9
L2 search: {1e-5, 1e-4, 1e-3}
```

Project hiện tại:

```text
embedding_dim = 16
batch_size = 2048
loss = logloss alias -> BCEWithLogitsLoss
learning_rate = 0.001 fixed
dropout fixed
l2_reg fixed
```

Gap:

```text
Project chưa có automated tuning.
```

Minimum tuning cần làm:

| Run | LR | Dropout | L2 | Heads | Layers |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.001 | 0.1 | 1e-5 | 4 | 2 |
| 2 | 0.0001 | 0.1 | 1e-5 | 4 | 2 |
| 3 | 0.001 | 0.3 | 1e-5 | 4 | 2 |
| 4 | 0.001 | 0.5 | 1e-5 | 4 | 2 |
| 5 | 0.001 | 0.1 | 1e-4 | 4 | 2 |
| 6 | 0.001 | 0.1 | 1e-5 | 2 | 2 |
| 7 | 0.001 | 0.1 | 1e-5 | 4 | 1 |

Mục tiêu chọn theo:

```text
valid_auc max
valid_logloss min
```

Sau đó chỉ báo cáo test metric của best config.

## 8. Baseline Bắt Buộc Cần Chạy Lại

Để biết gap nằm ở model hay data pipeline, cần chạy:

```text
LR
FM
DeepFM
AutoInt/FIN
NAM
NAFI
```

Nếu tất cả baseline đều thấp hơn paper nhiều, nguyên nhân chính nằm ở:

```text
preprocessing
encoding
split
training setup
```

Nếu baseline gần paper nhưng NAFI thấp, nguyên nhân nằm ở:

```text
NAM/FIN architecture
fusion
ExU
hyperparameter
```

## 9. Diagnostics Từ Kết Quả Hiện Tại

Log hiện tại cho thấy:

```text
Full NAFI AUC ~= 0.7611
NAM-only branch inside NAFI AUC ~= 0.5669
FIN-only branch inside NAFI AUC ~= 0.7590
```

Không nên kết luận NAM độc lập yếu từ số `valid_nam_auc`, vì branch này không train độc lập.

Điểm đáng chú ý:

```text
FIN ranking mạnh
NAM giúp calibration/logloss
Full NAFI chỉ hơn FIN một chút về AUC
```

Việc này gợi ý cần kiểm tra:

- NAM ExU có học đúng không.
- NAM hidden size có quá nhỏ không.
- FIN branch có dominate quá mạnh không.
- Fusion cộng thẳng có làm NAM bị residual hóa không.
- Có cần auxiliary loss cho NAM/FIN không.

Experiment nên thử:

```text
loss = BCE(full_logits, y)
     + alpha * BCE(nam_logits, y)
     + beta  * BCE(fin_logits, y)
```

Gợi ý ban đầu:

```yaml
aux_loss:
  enabled: true
  nam_weight: 0.1
  fin_weight: 0.1
```

Nhưng đây là biến thể, không phải paper gốc.

## 10. Checklist Reimplementation Theo Thứ Tự

### Phase 1: Reproduce NAFI Basic

- [ ] Reset về paper-like config:
  - embedding_dim 16
  - batch_size 2048
  - Adam
  - Logloss
  - hidden `[32, 32, 32]`
  - ExU for NAM
- [ ] Prepare data lại sau khi đổi hash buckets.
- [ ] Train LR/FM để sanity check.
- [ ] Train NAM độc lập.
- [ ] Train AutoInt/FIN độc lập.
- [ ] Train NAFI.
- [ ] Evaluate valid và test.

### Phase 2: Tune

- [ ] Grid search LR `{1e-4, 1e-3, 1e-2}`.
- [ ] Grid search dropout `{0.1, 0.3, 0.5}` trước, chưa cần tới 0.9.
- [ ] Grid search L2 `{1e-5, 1e-4, 1e-3}`.
- [ ] Tune FIN heads/layers.
- [ ] Tune NAM hidden.

### Phase 3: Encoding Gap

- [ ] So hashing Config B vs old large bucket config.
- [ ] Thử vocab mode cho low-cardinality fields.
- [ ] Thử frequency vocab cho high-cardinality fields.
- [ ] Kiểm tra collision rate trên `device_ip`, `device_id`.

### Phase 4: Paper Ablation

Recreate Table 5:

| Variant | Meaning |
|---|---|
| NAM | without FIN |
| FIN | without NAM |
| NAFI | NAM + FIN |
| KD-NAFI | NAFI + distillation |

Không dùng branch metric trong NAFI để thay thế NAM/FIN độc lập.

### Phase 5: KD-NAFI

Paper chọn teacher ensemble:

```text
DeepFM + xDeepFM + AutoInt
```

Project hiện chưa có xDeepFM đầy đủ. Có thể thay thế giai đoạn đầu:

```text
DeepFM + AutoInt + NAFI/FIN
```

KD loss:

```text
loss = alpha * BCE(student_logits, y)
     + (1 - alpha) * KD(student_logits/T, teacher_logits/T)
```

Cần tune:

```text
temperature: 2, 4, 8
alpha: 0.3, 0.5, 0.7
```

## 11. Những Điểm Có Thể Làm Kết Quả Chưa Đạt Paper

Ưu tiên kiểm tra theo thứ tự:

1. Encoding khác paper: hashing collision vs label encoding/vocab.
2. Split khác paper.
3. Chưa tune LR/dropout/L2.
4. FIN architecture chưa đúng Eq. (7) vì thiếu `W_res` projection.
5. NAM ExU chưa đúng NAM gốc hoặc hidden quá nhỏ.
6. Paper có thể dùng full vocabulary hoặc preprocessing khác nhưng không công bố.
7. Paper không công bố seed và best hyperparameter cụ thể.
8. Evaluation đang dùng validation split khác test split trong paper.

## 12. Command Gợi Ý

Train NAM độc lập:

```bash
python -m scripts.train \
  --config config/kaggle_t4x2.yaml \
  --model nam \
  --processed-dir <processed_dir>
```

Train NAFI:

```bash
python -m scripts.train \
  --config config/kaggle_t4x2.yaml \
  --model nafi \
  --processed-dir <processed_dir>
```

Evaluate test:

```bash
python -m scripts.evaluate \
  --config config/kaggle_t4x2.yaml \
  --checkpoint <checkpoint> \
  --split test \
  --processed-dir <processed_dir>
```

Tạo submission:

```bash
python -m scripts.predict_competition \
  --config config/kaggle_t4x2.yaml \
  --checkpoint <checkpoint> \
  --processed-dir <processed_dir> \
  --output /kaggle/working/outputs/submissions/submission.csv
```

## 13. Tiêu Chí Thành Công

Trước mắt không cần nhảy ngay tới paper target. Nên đặt milestone:

```text
Milestone 1: LR/FM gần Table 2 hơn.
Milestone 2: NAM độc lập đạt quanh 0.75-0.76 AUC.
Milestone 3: FIN độc lập đạt quanh 0.77 AUC.
Milestone 4: NAFI vượt FIN ít nhất 0.001 AUC hoặc giảm logloss rõ.
Milestone 5: NAFI tiến gần 0.78 AUC.
Milestone 6: KD-NAFI cải thiện thêm so với NAFI.
```

Nếu LR/FM cũng thấp xa paper, không nên tiếp tục sửa NAFI trước; hãy quay lại data split/encoding/preprocess.

