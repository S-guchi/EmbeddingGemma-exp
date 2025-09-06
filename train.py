import os,random
import pandas as pd
import numpy as np
import torch
from huggingface_hub import login
from sentence_transformers import SentenceTransformer, losses, InputExample
from torch.utils.data import DataLoader

if "HUGGINGFACE_HUB_TOKEN" in os.environ:
    try:
        login(token=os.environ["HUGGINGFACE_HUB_TOKEN"])
    except Exception:
        pass

MODEL_ID = "google/embeddinggemma-300m"
CSV_PATH = "pairs.csv"
OUTPUT_DIR = "ft-embeddinggemma"

SEED = 42

random.seed(SEED); np.random.seed(SEED)

USE_AMP = torch.cuda.is_available()
DEVICE  = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

# ---- 居酒屋ドリンク予測データ読み込み ----
# データ少ないから全部学習データにする
train_df = pd.read_csv(CSV_PATH).drop_duplicates()

train_samples = [
    InputExample(texts=[r.query.strip(), r.item.strip()])
    for r in train_df.itertuples(index=False)
]
# train_samples = [
#     InputExample(texts=["唐揚げ", "ハイボール"]),
#     InputExample(texts=["枝豆", "ビール"]),
#     ...
# ]

EPOCHS = 2
BASE_BATCH_SIZE = 4
# MAX_SEQ_LEN = 64
# ---- Model ----
model = SentenceTransformer(MODEL_ID, device=DEVICE)
# model.max_seq_length = MAX_SEQ_LEN

# ---- DataLoader ----
BATCH_SIZE = min(BASE_BATCH_SIZE, max(1, len(train_samples)))
train_loader = DataLoader(
    train_samples,
    shuffle=True,
    batch_size=BATCH_SIZE,
)

# ---- Loss ----
# バッチ内の正解以外をNegativeとして扱う
# アンカー：「唐揚げ」
# ポジティブ：「ハイボール」
# ネガティブ：「枝豆」「冷奴」「レモンサワー」など
train_loss = losses.MultipleNegativesRankingLoss(model)


steps_per_epoch = max(1, len(train_loader))
WARMUP_RATIO = 0.1
warmup_steps = int(steps_per_epoch * EPOCHS * WARMUP_RATIO)

# ---- 学習 ----
model.fit(
    train_objectives=[(train_loader, train_loss)],
    epochs=EPOCHS,
    warmup_steps=warmup_steps, # 学習率を徐々に上げていくスケジュール設定らしい
    output_path=OUTPUT_DIR,
)

print(f"✅ saved to: {OUTPUT_DIR}")
