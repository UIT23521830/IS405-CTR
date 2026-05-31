# Review idea: Thay NAM bằng KAN trong NAFI cho Avazu CTR

## Tóm tắt nhanh

Ý tưởng thay nhánh NAM bằng KAN là **khả thi về mặt kiến trúc** và đáng thử như một hướng nghiên cứu phụ cho project Avazu CTR. Điểm cắm tự nhiên là thay từng `MLP(embedding_i -> scalar)` trong `NAMBranch` bằng một mạng KAN nhỏ:

```text
NAFI hiện tại:
  logits = NAM(embeddings) + FIN(embeddings)
  NAM = sum_i MLP_i(e_i)

Đề xuất:
  logits = KANAdditive(embeddings) + FIN(embeddings)
  KANAdditive = sum_i KAN_i(e_i)
```

Tuy nhiên, đây **không nên là hướng thay thế mặc định ngay từ đầu**. Với Avazu, đầu vào chủ yếu là categorical high-cardinality đã qua hashing và embedding. KAN mạnh nhất ở học hàm liên tục, trực quan hóa hàm và mô hình nhỏ; còn embedding categorical là latent vector, không có ý nghĩa trực tiếp như biến liên tục gốc. Vì vậy lợi ích về interpretability có thể giảm đáng kể.

Khuyến nghị: triển khai như một biến thể thử nghiệm `kafi` hoặc `kan_nafi`, chạy A/B trên debug/small trước khi đưa vào full.

## Bối cảnh từ project hiện tại

Trong code hiện tại, `NAFI` gồm:

- `FeatureEmbedding`: ánh xạ mỗi categorical field thành vector embedding.
- `NAMBranch`: mỗi feature field có một MLP nhỏ riêng, trả về contribution scalar.
- `FINBranch`: multi-head self-attention học feature interaction.
- Output cuối:

```text
logits = nam_logits + fin_logits
```

Điểm quan trọng: NAM trong project đang không nhận raw categorical value, mà nhận `embedding_i` có shape:

```text
[batch_size, embedding_dim]
```

Vì vậy nếu thay bằng KAN, KAN sẽ học hàm:

```text
f_i: R^embedding_dim -> R
```

chứ không trực tiếp học:

```text
f_i(category_id) -> R
```

Điều này làm ý tưởng khả thi về engineering, nhưng cần thận trọng khi diễn giải.

## KAN là gì và vì sao có vẻ phù hợp?

Paper KAN đề xuất Kolmogorov-Arnold Networks như một lựa chọn thay MLP. Khác với MLP đặt activation cố định ở node/neuron, KAN đặt activation học được trên edge/weight, thường tham số hóa bằng spline. Tác giả báo cáo KAN có tiềm năng tốt về accuracy và interpretability, nhất là với mô hình nhỏ và bài toán có cấu trúc hàm rõ ràng.

Nguồn chính:

- KAN paper, arXiv `2404.19756`: https://arxiv.org/abs/2404.19756
- Repo chính `pykan`: https://github.com/KindXiaoming/pykan
- Repo `efficient-kan`: https://github.com/Blealtan/efficient-kan

Điểm làm KAN hấp dẫn với NAFI:

1. NAM vốn là additive model, mỗi field có một function riêng. KAN cũng nhấn mạnh học các hàm đơn/nhỏ có thể quan sát được.
2. Nhánh NAM hiện tại dùng MLP nhỏ. KAN có thể là drop-in replacement cho các MLP này.
3. NAFI cần interpretability qua feature contribution. KAN có thể tăng khả năng phân tích shape/function của từng contribution, đặc biệt với `hour_of_day`, `day`, `weekday`, hoặc numeric categorical đã hash/bucket nhỏ.
4. Nếu KAN đạt cùng accuracy với ít hidden width hơn, nhánh additive có thể gọn hơn MLP.

## Điểm lợi tiềm năng

### 1. Nhánh additive mạnh hơn MLP nhỏ

Trong `NAMBranch`, mỗi field network đang là MLP rất nhỏ, ví dụ `[32, 16]`. Với embedding_dim 8 hoặc 16, KAN có thể học nonlinear mapping từ embedding sang contribution linh hoạt hơn bằng spline basis. Nếu MLP hiện tại underfit ở nhánh additive, KAN có khả năng cải thiện `nam_logits`.

### 2. Interpretability tốt hơn ở mức function

KAN có thể cho phép quan sát dạng hàm học được trên từng input dimension. Điều này hợp với mục tiêu của NAFI: vừa dự đoán CTR vừa giải thích contribution.

Nhưng cần ghi rõ: với Avazu, input dimension của KAN là embedding dimension, không phải feature gốc. Vì vậy interpretability trực tiếp theo nghĩa “category A làm CTR tăng” vẫn phải đi qua bước phân tích embedding/category, không tự nhiên như với feature numeric.

### 3. Có thể bổ sung regularization/sparsity

KAN thường đi cùng ý tưởng sparsification/pruning để giữ mô hình nhỏ và dễ xem. Nếu áp dụng thận trọng, có thể giúp nhánh additive bớt nhiễu, nhất là khi FIN đã đảm nhiệm phần interaction.

### 4. Tạo biến thể nghiên cứu có câu chuyện rõ

Tên biến thể có thể là:

```text
KAFI = Kolmogorov Additive Feature Interaction
KAN-NAFI = NAFI with KAN additive branch
```

Luận điểm nghiên cứu:

```text
Thay NAM branch bằng KAN branch để tăng năng lực học nonlinear contribution theo field,
trong khi giữ FIN branch để học interaction bậc cao.
```

Đây là một hướng nâng cấp gọn, dễ viết experiment hơn so với thay toàn bộ DeepFM/AutoInt.

## Rủi ro và hạn chế

### 1. KAN chưa được chứng minh rõ trên CTR sparse categorical quy mô Avazu

Tài liệu KAN ban đầu tập trung nhiều vào data fitting, PDE/science tasks, symbolic/interpretability và các mô hình nhỏ. Repo `pykan` cũng ghi chú rằng bài toán trong paper thường nhỏ hơn typical ML tasks; với task lớn nên dùng GPU, và KAN chưa hẳn là plug-in dùng ngay cho mọi bài toán ML.

Vì Avazu có khoảng 40 triệu dòng, nhiều categorical field, hashing collision và embedding lookup, cần benchmark thực tế. Không nên giả định KAN sẽ thắng MLP.

### 2. Chi phí tính toán có thể tăng

KAN spline layer có thể tốn memory/compute hơn Linear+ReLU MLP nếu implement không tối ưu. `efficient-kan` chỉ ra vấn đề memory của implementation gốc: cách tính ban đầu có thể expand tensor trung gian theo shape gần như:

```text
[batch_size, out_features, in_features]
```

Với CTR batch size 1024-2048 và nhiều field, overhead này có thể làm Kaggle T4 dễ OOM hoặc chậm.

### 3. Interpretability bị suy yếu bởi hashing + embedding

Avazu categorical được xử lý bằng:

```text
category -> stable hash bucket -> embedding vector -> KAN/MLP
```

KAN sẽ giải thích hàm trên embedding vector, không phải trên category gốc. Ngoài ra hashing collision làm nhiều category có thể cùng bucket. Do đó KAN không tự động giải quyết vấn đề interpretability category-level.

### 4. Nguy cơ overfit

KAN có grid/spline basis. Nếu grid quá lớn hoặc mỗi field dùng KAN riêng quá rộng, model có thể overfit nhanh ở debug/small. CTR thường có tín hiệu yếu; AUC tăng 0.001 đã có ý nghĩa, nhưng variance giữa run cũng cần kiểm soát seed/split.

## Thiết kế khả thi nhất

Không thay toàn bộ NAFI. Chỉ thay `NAMBranch`.

### Variant A: Per-field KAN branch

```text
embedding_i -> KAN_i -> contribution_i
sum_i contribution_i -> kan_logits
logits = kan_logits + fin_logits
```

Ưu điểm:

- Gần nhất với NAM.
- Giữ feature contribution theo field.
- Dễ so sánh với NAMBranch hiện tại.

Nhược điểm:

- Có `num_fields` mạng KAN riêng, chi phí tăng.

### Variant B: Shared KAN branch theo field

```text
embedding_i + field_embedding_i -> shared_KAN -> contribution_i
```

Ưu điểm:

- Ít parameter hơn.
- Có thể scale tốt hơn.

Nhược điểm:

- Interpretability theo field kém hơn.
- Cần thêm field embedding hoặc field id.

### Variant C: Chỉ dùng KAN cho derived/time fields

```text
hour_of_day/day/weekday -> KAN
categorical high-cardinality -> NAM MLP hoặc linear
```

Ưu điểm:

- KAN được dùng ở nơi có ý nghĩa hàm liên tục/rời rạc nhỏ.
- Rẻ hơn và dễ giải thích hơn.

Nhược điểm:

- Ít khác biệt so với NAFI gốc.

Khuyến nghị ban đầu: **Variant A**, nhưng cấu hình rất nhỏ.

## Cấu hình thử nghiệm đề xuất

Debug:

```yaml
model:
  name: kan_nafi
  embedding_dim: 8

kan:
  enabled: true
  grid_size: 3
  spline_order: 3
  hidden_units: []
  dropout: 0.0
  l1_reg: 0.0
```

Small:

```yaml
model:
  name: kan_nafi
  embedding_dim: 8

kan:
  grid_size: 3
  spline_order: 3
  hidden_units: [8]
  dropout: 0.05
  l1_reg: 1.0e-6
```

Không nên bắt đầu với grid lớn hoặc embedding_dim 16 + KAN rộng. Mục tiêu đầu tiên là kiểm tra pipeline không OOM và AUC/logloss có cạnh tranh với NAM.

## Experiment cần chạy

Chạy cùng seed, cùng parquet split:

| Model | Debug | Small | Metric chính | Ghi chú |
|---|---:|---:|---|---|
| NAM-only | yes | yes | AUC/logloss | baseline additive |
| NAFI | yes | yes | AUC/logloss | baseline chính |
| KAN-only additive | yes | optional | AUC/logloss | xem KAN branch tự đứng được không |
| KAN-NAFI | yes | yes | AUC/logloss | ý tưởng chính |
| DeepFM | yes | yes | AUC/logloss | baseline mạnh |

Theo tiêu chí trong yêu cầu project, nếu `KAN-NAFI` tăng AUC khoảng `0.001` so với `NAFI` ở small/full và không tăng quá nhiều thời gian/VRAM, đó đã là tín hiệu đáng theo tiếp.

## Kết luận

Ý tưởng **có tính khả thi kỹ thuật cao** vì `NAMBranch` hiện tại là module độc lập và có interface rất sạch: nhận embeddings, trả logits + contributions. KAN có thể thay MLP per-field mà không phá data pipeline, FIN branch, trainer, metrics hay checkpoint.

Điểm lợi lớn nhất không chắc là leaderboard ngay, mà là:

- tạo một nhánh additive giàu nonlinear hơn,
- giữ được cấu trúc explainable theo field,
- mở ra biến thể nghiên cứu mới `KAN-NAFI/KAFI`,
- có thể tốt hơn NAM khi MLP nhỏ bị underfit.

Nhưng cần xem đây là experiment, không phải nâng cấp chắc thắng. Với Avazu, rủi ro chính là chi phí KAN và interpretability bị giới hạn bởi hashing/embedding. Hướng tốt nhất là triển khai KAN branch nhỏ, dùng efficient implementation, chạy A/B từ debug đến small rồi mới quyết định có chạy full hay không.

## Nguồn tham khảo

- Liu et al., “KAN: Kolmogorov-Arnold Networks”, arXiv:2404.19756, submitted 2024, revised 2025: https://arxiv.org/abs/2404.19756
- Official `pykan` repository: https://github.com/KindXiaoming/pykan
- `efficient-kan` memory-efficient PyTorch implementation: https://github.com/Blealtan/efficient-kan

