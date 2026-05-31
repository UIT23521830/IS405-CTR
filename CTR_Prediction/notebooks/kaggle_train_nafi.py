"""
Kaggle Notebook: CTR Prediction with NAFI (Vocab Encoding + Time-based Split)
==============================================================================
Copy nội dung này vào các cell trên Kaggle.
Yêu cầu: 21 parquet files đã tồn tại tại PARQUET_DIR
GPU: T4 x2

Cấu trúc:
  Cell 1: Setup & Config
  Cell 2: Pass 1 - Fit Vocab (đọc tất cả parquet, đếm frequency)
  Cell 3: Pass 2 - Encode + Time-based Split + Lưu encoded parquet
  Cell 4: Dataset & DataLoader
  Cell 5: Model NAFI
  Cell 6: Train & Evaluate (AUC + Logloss)
"""

# ===========================================================================
# CELL 1: SETUP & CONFIG
# ===========================================================================

import os, json, time, gc, warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, IterableDataset
from sklearn.metrics import roc_auc_score, log_loss

warnings.filterwarnings("ignore")

# ----- Paths -----
PARQUET_DIR   = Path("/kaggle/working/avazu_parquet")   # 21 file train_part_*.parquet
ENCODED_DIR   = Path("/kaggle/working/avazu_encoded")   # output sau khi encode
VOCAB_PATH    = ENCODED_DIR / "vocab.json"
META_PATH     = ENCODED_DIR / "meta.json"
ENCODED_DIR.mkdir(parents=True, exist_ok=True)
for split in ["train", "valid", "test"]:
    (ENCODED_DIR / split).mkdir(exist_ok=True)

# ----- Hyperparameters -----
SEED          = 42
MIN_FREQ      = 4            # category xuất hiện < MIN_FREQ lần → <UNK>=0
EMB_DIM       = 16           # đúng như paper
NAM_HIDDEN    = [32, 32, 32] # đúng như paper (3 layers x 32)
NAM_ACTIVATION= "exu"        # ExU như paper
FIN_HEADS     = 4
FIN_LAYERS    = 2
DROPOUT       = 0.1
L2_REG        = 1e-5
LR            = 1e-3
BATCH_SIZE    = 2048
EPOCHS        = 3
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"

# ----- Feature columns -----
# hour sẽ được parse thành day, hour_of_day, weekday rồi bỏ raw hour
RAW_CAT_COLS = [
    "C1", "banner_pos",
    "site_id", "site_domain", "site_category",
    "app_id", "app_domain", "app_category",
    "device_id", "device_ip", "device_model",
    "device_type", "device_conn_type",
    "C14", "C15", "C16", "C17", "C18", "C19", "C20", "C21",
]
DERIVED_COLS  = ["day", "hour_of_day", "weekday"]  # từ cột hour
FEATURE_COLS  = DERIVED_COLS + RAW_CAT_COLS        # tổng 24 fields
TARGET_COL    = "click"
DROP_COLS     = ["id"]

torch.manual_seed(SEED)
np.random.seed(SEED)
print(f"Device: {DEVICE}")
print(f"Parquet files: {sorted(PARQUET_DIR.glob('*.parquet'))[:3]} ...")


# ===========================================================================
# CELL 2: PASS 1 - FIT VOCAB (đếm frequency)
# ===========================================================================

def parse_hour(df):
    """YYMMDDHH -> day (int), hour_of_day (int), weekday (int)"""
    hour_str = df["hour"].astype(str).str.zfill(8)
    parsed   = pd.to_datetime(hour_str, format="%y%m%d%H", errors="coerce")
    df["day"]          = parsed.dt.day.fillna(0).astype("int32")
    df["hour_of_day"]  = parsed.dt.hour.fillna(0).astype("int32")
    df["weekday"]      = parsed.dt.weekday.fillna(0).astype("int32")
    return df


def fit_vocab(parquet_files, raw_cat_cols, min_freq=4):
    """Pass 1: đọc toàn bộ parquet, đếm frequency mỗi field."""
    if VOCAB_PATH.exists():
        print("Vocab đã tồn tại, load từ file...")
        with open(VOCAB_PATH) as f:
            return json.load(f)

    freq = {col: defaultdict(int) for col in raw_cat_cols}
    hour_vals = []  # để tính time split sau

    print("Pass 1: Đếm frequency...")
    for i, fpath in enumerate(parquet_files):
        df = pd.read_parquet(fpath, columns=raw_cat_cols + ["hour"])
        # hour để ghi nhận range
        hour_vals.extend(df["hour"].unique().tolist())
        for col in raw_cat_cols:
            col_freq = df[col].astype(str).value_counts()
            for val, cnt in col_freq.items():
                freq[col][val] += int(cnt)
        del df; gc.collect()
        if (i+1) % 5 == 0:
            print(f"  Đã đọc {i+1}/{len(parquet_files)} files...")

    # Build vocab: chỉ giữ giá trị >= min_freq, index bắt đầu từ 1 (0 = UNK)
    vocab = {}
    for col in raw_cat_cols:
        mapping = {"<UNK>": 0}
        idx = 1
        for val, cnt in sorted(freq[col].items()):
            if cnt >= min_freq:
                mapping[val] = idx
                idx += 1
        vocab[col] = mapping
        print(f"  {col}: {len(mapping)-1} unique (min_freq={min_freq})")

    with open(VOCAB_PATH, "w") as f:
        json.dump(vocab, f)
    print(f"\nVocab saved to {VOCAB_PATH}")
    return vocab


parquet_files = sorted(PARQUET_DIR.glob("train_part_*.parquet"))
print(f"Tìm thấy {len(parquet_files)} parquet files")

vocab = fit_vocab(parquet_files, RAW_CAT_COLS, min_freq=MIN_FREQ)
FIELD_DIMS = (
    [32, 24, 7]  # day(max 31+1), hour_of_day(24), weekday(7)
    + [len(vocab[col]) + 1 for col in RAW_CAT_COLS]  # +1 cho safety
)
print(f"\nField dims: {FIELD_DIMS}")
print(f"Tổng số fields: {len(FIELD_DIMS)}")


# ===========================================================================
# CELL 3: PASS 2 - ENCODE + TIME-BASED SPLIT + LƯU
# ===========================================================================

def get_time_split_thresholds(parquet_files):
    """Đọc min/max hour để chia time split 8:1:1 theo thời gian."""
    hours = []
    for fpath in parquet_files:
        df = pd.read_parquet(fpath, columns=["hour"])
        hours.extend(df["hour"].unique().tolist())
    hours = sorted(set(hours))
    n = len(hours)
    # last 10% -> test, next 10% -> valid, rest -> train
    test_thresh  = hours[int(n * 0.9)]
    valid_thresh = hours[int(n * 0.8)]
    print(f"Hour range: {hours[0]} -> {hours[-1]}")
    print(f"Valid từ hour >= {valid_thresh}, Test từ hour >= {test_thresh}")
    return valid_thresh, test_thresh


def encode_and_split(parquet_files, vocab, raw_cat_cols, derived_cols, feature_cols, target_col):
    """Pass 2: encode vocab + chia train/valid/test theo time."""
    # Check nếu đã encode rồi
    meta_path_check = ENCODED_DIR / "meta.json"
    if meta_path_check.exists():
        print("Encoded data đã tồn tại, skip...")
        with open(meta_path_check) as f:
            return json.load(f)

    valid_thresh, test_thresh = get_time_split_thresholds(parquet_files)
    counters = {"train": 0, "valid": 0, "test": 0}
    file_idx = {"train": 0, "valid": 0, "test": 0}

    print("\nPass 2: Encode + Split...")
    for i, fpath in enumerate(parquet_files):
        df = pd.read_parquet(fpath)

        # Drop unnecessary columns
        for col in DROP_COLS:
            if col in df.columns:
                df.drop(columns=[col], inplace=True)

        # Parse hour -> derived features
        df = parse_hour(df)

        # Encode derived integer cols (clip to safe range)
        df["day"]         = df["day"].clip(0, 31).astype("int32")
        df["hour_of_day"] = df["hour_of_day"].clip(0, 23).astype("int32")
        df["weekday"]     = df["weekday"].clip(0, 6).astype("int32")

        # Encode categorical cols via vocab
        for col in raw_cat_cols:
            mapping = vocab[col]
            df[col] = df[col].astype(str).map(mapping).fillna(0).astype("int32")

        # Target
        df[target_col] = df[target_col].astype("int8")

        # Time-based split
        for split, mask in [
            ("test",  df["hour"] >= test_thresh),
            ("valid", (df["hour"] >= valid_thresh) & (df["hour"] < test_thresh)),
            ("train", df["hour"] < valid_thresh),
        ]:
            sub = df[mask][feature_cols + [target_col]].copy()
            if len(sub) == 0:
                continue
            out_path = ENCODED_DIR / split / f"{split}_part_{file_idx[split]:05d}.parquet"
            sub.to_parquet(out_path, index=False)
            counters[split] += len(sub)
            file_idx[split] += 1

        del df; gc.collect()
        if (i+1) % 5 == 0:
            print(f"  Đã xử lý {i+1}/{len(parquet_files)} files | {counters}")

    meta = {
        "feature_cols": feature_cols,
        "field_dims":   FIELD_DIMS,
        "target_col":   target_col,
        "split_counts": counters,
        "valid_thresh": int(valid_thresh),
        "test_thresh":  int(test_thresh),
    }
    with open(ENCODED_DIR / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nEncode xong! Counts: {counters}")
    return meta


meta = encode_and_split(parquet_files, vocab, RAW_CAT_COLS, DERIVED_COLS, FEATURE_COLS, TARGET_COL)
print(json.dumps(meta["split_counts"], indent=2))


# ===========================================================================
# CELL 4: DATASET & DATALOADER
# ===========================================================================

class AvazuParquetDataset(IterableDataset):
    """Đọc từng parquet, yield (x, y) không load full RAM."""

    def __init__(self, split_dir: Path, feature_cols, target_col, shuffle_files=True):
        self.files       = sorted(split_dir.glob("*.parquet"))
        self.feature_cols = feature_cols
        self.target_col  = target_col
        self.shuffle_files = shuffle_files

    def __iter__(self):
        files = list(self.files)
        if self.shuffle_files:
            import random; random.shuffle(files)
        for fpath in files:
            df = pd.read_parquet(fpath, columns=self.feature_cols + [self.target_col])
            df = df.sample(frac=1).reset_index(drop=True)  # shuffle within file
            x  = torch.from_numpy(df[self.feature_cols].values.astype(np.int64))
            y  = torch.from_numpy(df[self.target_col].values.astype(np.float32))
            for xi, yi in zip(x, y):
                yield xi, yi


def make_loader(split, shuffle=True, batch_size=BATCH_SIZE):
    ds = AvazuParquetDataset(
        ENCODED_DIR / split,
        FEATURE_COLS, TARGET_COL,
        shuffle_files=shuffle
    )
    return DataLoader(ds, batch_size=batch_size, num_workers=2, pin_memory=True)


# ===========================================================================
# CELL 5: MODEL (giữ nguyên kiến trúc, inline vào notebook để dễ chạy)
# ===========================================================================

class ReLUN(nn.Module):
    def __init__(self, max_value=1.0):
        super().__init__(); self.max_value = max_value
    def forward(self, x): return torch.clamp(torch.relu(x), max=self.max_value)


class ExULayer(nn.Module):
    """h(x) = ReLUN((x - b) @ exp(W)) — NAM paper"""
    def __init__(self, in_dim, out_dim, max_value=1.0, clip=10.0):
        super().__init__()
        self.W    = nn.Parameter(torch.empty(in_dim, out_dim).normal_(0, 0.5))
        self.b    = nn.Parameter(torch.empty(in_dim).normal_(0, 0.5))
        self.act  = ReLUN(max_value)
        self.clip = clip
    def forward(self, x):
        W = torch.exp(torch.clamp(self.W, -self.clip, self.clip))
        return self.act((x - self.b) @ W)


def make_mlp(in_dim, hidden, out_dim=1, dropout=0.1, activation="relu"):
    layers = []
    prev = in_dim
    use_exu = activation == "exu"
    for h in hidden:
        if use_exu:
            layers.append(ExULayer(prev, h))
        else:
            layers += [nn.Linear(prev, h), nn.ReLU()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


class FeatureEmbedding(nn.Module):
    def __init__(self, field_dims, emb_dim):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(dim, emb_dim) for dim in field_dims])
        for e in self.embs: nn.init.xavier_uniform_(e.weight)
    def forward(self, x):
        return torch.stack([
            self.embs[i](torch.remainder(x[:, i], self.embs[i].num_embeddings))
            for i in range(len(self.embs))
        ], dim=1)  # [B, F, D]


class NAMBranch(nn.Module):
    def __init__(self, num_fields, emb_dim, hidden, dropout, activation):
        super().__init__()
        self.nets = nn.ModuleList([
            make_mlp(emb_dim, hidden, 1, dropout, activation)
            for _ in range(num_fields)
        ])
    def forward(self, emb):  # emb: [B, F, D]
        contribs = torch.cat([net(emb[:, i, :]) for i, net in enumerate(self.nets)], dim=1)
        return contribs.sum(dim=1), contribs  # logits [B], contributions [B, F]


class FINBranch(nn.Module):
    def __init__(self, num_fields, emb_dim, num_heads, num_layers, dropout):
        super().__init__()
        self.attns = nn.ModuleList([
            nn.MultiheadAttention(emb_dim, num_heads, dropout=dropout, batch_first=True)
            for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(emb_dim) for _ in range(num_layers)])
        self.act   = nn.ReLU()
        self.out   = make_mlp(num_fields * emb_dim, [64, 32], 1, dropout, "relu")

    def forward(self, emb):  # emb: [B, F, D]
        out = emb
        attn_w = None
        for attn, norm in zip(self.attns, self.norms):
            res = out
            out, attn_w = attn(out, out, out, need_weights=True, average_attn_weights=False)
            out = norm(self.act(out + res))
        logits = self.out(out.flatten(1)).squeeze(-1)
        return logits, attn_w


class NAFI(nn.Module):
    def __init__(self, field_dims, emb_dim=16, nam_hidden=None, fin_heads=4,
                 fin_layers=2, dropout=0.1, nam_activation="exu"):
        super().__init__()
        nam_hidden = nam_hidden or [32, 32, 32]
        self.emb = FeatureEmbedding(field_dims, emb_dim)
        self.nam = NAMBranch(len(field_dims), emb_dim, nam_hidden, dropout, nam_activation)
        self.fin = FINBranch(len(field_dims), emb_dim, fin_heads, fin_layers, dropout)

    def forward(self, x):
        emb = self.emb(x)
        nam_logits, contribs = self.nam(emb)
        fin_logits, attn_w   = self.fin(emb)
        logits = nam_logits + fin_logits
        return {
            "logits": logits,
            "nam_logits": nam_logits,
            "fin_logits": fin_logits,
            "feature_contributions": contribs,
            "attention_weights": attn_w,
        }


model = NAFI(
    field_dims    = FIELD_DIMS,
    emb_dim       = EMB_DIM,
    nam_hidden    = NAM_HIDDEN,
    fin_heads     = FIN_HEADS,
    fin_layers    = FIN_LAYERS,
    dropout       = DROPOUT,
    nam_activation= NAM_ACTIVATION,
).to(DEVICE)

# DataParallel nếu có 2 GPU
if torch.cuda.device_count() > 1:
    print(f"Dùng {torch.cuda.device_count()} GPUs")
    model = nn.DataParallel(model)

total_params = sum(p.numel() for p in model.parameters())
print(f"Tổng parameters: {total_params:,}")


# ===========================================================================
# CELL 6: TRAIN & EVALUATE
# ===========================================================================

criterion  = nn.BCEWithLogitsLoss()
optimizer  = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=L2_REG)
scaler     = torch.cuda.amp.GradScaler(enabled=(DEVICE == "cuda"))


def evaluate(model, loader, desc="Eval"):
    model.eval()
    all_labels, all_probs = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
                out = model(x)
                logits = out["logits"] if isinstance(out, dict) else out
            probs  = torch.sigmoid(logits).cpu().numpy()
            labels = y.cpu().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(labels.tolist())
    auc     = roc_auc_score(all_labels, all_probs)
    logloss = log_loss(all_labels, all_probs)
    print(f"  {desc}: AUC={auc:.4f}  Logloss={logloss:.4f}")
    return auc, logloss


best_auc  = 0
best_epoch = 0
results   = []

print("="*60)
print(f"Bắt đầu train NAFI | {EPOCHS} epochs | Device={DEVICE}")
print("="*60)

for epoch in range(1, EPOCHS + 1):
    model.train()
    train_loader = make_loader("train", shuffle=True)
    total_loss   = 0.0
    n_batches    = 0
    t0           = time.time()

    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
            out    = model(x)
            logits = out["logits"] if isinstance(out, dict) else out
            loss   = criterion(logits, y)

        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        n_batches  += 1

        if batch_idx % 1000 == 0 and batch_idx > 0:
            print(f"  Epoch {epoch} | batch {batch_idx} | avg_loss={total_loss/n_batches:.4f} | "
                  f"elapsed={time.time()-t0:.0f}s")

    avg_loss = total_loss / max(n_batches, 1)
    print(f"\nEpoch {epoch} | Train Loss={avg_loss:.4f} | time={time.time()-t0:.0f}s")

    # Evaluate valid
    valid_loader = make_loader("valid", shuffle=False)
    val_auc, val_logloss = evaluate(model, valid_loader, "Valid")

    results.append({
        "epoch": epoch,
        "train_loss": round(avg_loss, 4),
        "valid_auc": round(val_auc, 4),
        "valid_logloss": round(val_logloss, 4),
    })

    if val_auc > best_auc:
        best_auc   = val_auc
        best_epoch = epoch
        torch.save(model.state_dict(), "/kaggle/working/best_nafi.pt")
        print(f"  ✅ New best AUC={best_auc:.4f} saved!")

    del train_loader, valid_loader
    gc.collect()
    torch.cuda.empty_cache()

# ---- Final test evaluation ----
print("\n" + "="*60)
print("TEST SET EVALUATION (Best checkpoint)")
print("="*60)
model.load_state_dict(torch.load("/kaggle/working/best_nafi.pt"))
test_loader = make_loader("test", shuffle=False)
test_auc, test_logloss = evaluate(model, test_loader, "Test")

# ---- Kết quả so sánh với paper ----
print("\n" + "="*60)
print("📊 SO SÁNH KẾT QUẢ VỚI PAPER GỐC")
print("="*60)
print(f"{'Model':<20} {'AUC':>8} {'Logloss':>10}")
print("-"*40)
print(f"{'Paper: NAM':<20} {'0.7600':>8} {'0.3850':>10}")
print(f"{'Paper: FIN':<20} {'0.7775':>8} {'0.3831':>10}")
print(f"{'Paper: NAFI':<20} {'0.7801':>8} {'0.3751':>10}")
print(f"{'Paper: KD-NAFI':<20} {'0.7806':>8} {'0.3752':>10}")
print("-"*40)
print(f"{'Ours: NAFI':<20} {test_auc:>8.4f} {test_logloss:>10.4f}")
print("="*60)

print("\nTraining history:")
for r in results:
    print(f"  Epoch {r['epoch']}: loss={r['train_loss']:.4f}, "
          f"val_auc={r['valid_auc']:.4f}, val_logloss={r['valid_logloss']:.4f}")
print(f"\nBest epoch: {best_epoch} | Best valid AUC: {best_auc:.4f}")
