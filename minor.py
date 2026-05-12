# -*- coding: utf-8 -*-
"""
Multimodal Stock Market Prediction
ResNet18 (price charts) + FinBERT (news headlines) → MLP → Bullish/Bearish

Usage:
    python minor.py                        # full pipeline: train + evaluate + predict
    python minor.py --skip-train           # skip training, load saved model
    python minor.py --skip-baselines       # skip baseline comparison (faster)
    python minor.py --skip-train --skip-baselines  # load model + eval + t-SNE only
    python minor.py --predict-only         # live prediction only (model must exist)

Place your dataset CSV at:  data/sp500_headlines_2008_2024.csv
All outputs go to:          charts/  models/  results/
"""

import os
import sys
import argparse
import pickle
import warnings
warnings.filterwarnings('ignore')

import subprocess

# ─────────────────────────────────────────────
# 0.  INSTALL DEPENDENCIES (first run only)
# ─────────────────────────────────────────────
def install_deps():
    pkgs = [
        "pandas", "numpy", "matplotlib", "Pillow",
        "torch", "torchvision", "transformers",
        "scikit-learn", "seaborn", "yfinance", "flask", "flask-cors"
    ]
    subprocess.run([sys.executable, "-m", "pip", "install", "--quiet"] + pkgs, check=True)

# ─────────────────────────────────────────────
# 1.  CONFIGURATION  (edit paths here)
# ─────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))   # folder containing minor.py

DATA_PATH  = os.path.join(BASE_DIR, "data", "sp500_headlines_2008_2024.csv")
CHART_DIR  = os.path.join(BASE_DIR, "charts")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
RESULT_DIR = os.path.join(BASE_DIR, "results")

# Data
WINDOW_SIZE = 10
IMG_SIZE    = 224   # ↑ from 64 — matches ResNet18's native resolution
TEST_SPLIT  = 0.1
VAL_SPLIT   = 0.1
RANDOM_SEED = 42

# Model
CNN_EMBED_DIM  = 512
TEXT_EMBED_DIM = 768
TECH_DIM       = 3                # momentum, volatility, RSI
FUSION_HIDDEN1 = 256
FUSION_HIDDEN2 = 128
DROPOUT_RATE   = 0.5              # ↑ from 0.3 — stronger regularization

# Training
EPOCHS              = 50
BATCH_SIZE          = 32
LR                  = 1e-4
LR_ENCODER          = 1e-5
WEIGHT_DECAY        = 1e-3        # L2 regularization on MLP weights
MAX_LEN             = 128
EARLY_STOP_PATIENCE = 10

FINBERT_MODEL = "ProsusAI/finbert"

import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ─────────────────────────────────────────────
# 2.  DATA LOADING & CLEANING
# ─────────────────────────────────────────────
def load_and_clean(path):
    import pandas as pd
    import numpy as np

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"\n[ERROR] Dataset not found at: {path}\n"
            f"  → Create the folder 'data/' next to minor.py and place\n"
            f"    'sp500_headlines_2008_2024.csv' inside it.\n"
            f"  → Download from Kaggle: https://www.kaggle.com/datasets\n"
        )

    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")

    # Flexible column rename — handles both raw and already-renamed CSVs
    rename = {}
    if "Title" in df.columns:
        rename["Title"] = "Headline"
    if "CP" in df.columns:
        rename["CP"] = "Close"
    if rename:
        df = df.rename(columns=rename)

    required = {"Date", "Close", "Headline"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Dataset is missing columns: {missing}. "
            f"Available: {df.columns.tolist()}"
        )

    df = df.dropna(subset=["Close", "Headline"])
    df = df.drop_duplicates(subset=["Date", "Close", "Headline"])

    print(f"[Data] Loaded {len(df)} rows  "
          f"({df['Date'].min().date()} → {df['Date'].max().date()})")

    # Aggregate multiple headlines per day
    news_df  = (df.groupby("Date")["Headline"]
                  .apply(lambda x: " ".join(x.astype(str)))
                  .reset_index()
                  .rename(columns={"Headline": "AggHeadline"}))
    price_df = df.groupby("Date")["Close"].last().reset_index()
    merged   = (pd.merge(price_df, news_df, on="Date", how="inner")
                  .sort_values("Date")
                  .reset_index(drop=True))

    print(f"[Data] After aggregation: {len(merged)} trading days")
    return merged


# ─────────────────────────────────────────────
# 3.  SLIDING WINDOWS & LABELS
# ─────────────────────────────────────────────
def create_windows_and_labels(df, window_size=WINDOW_SIZE):
    import numpy as np

    prices = df["Close"].values.astype("float32")
    texts  = df["AggHeadline"].values
    dates  = df["Date"].values

    windows, labels, news_texts, window_dates = [], [], [], []

    for i in range(len(df) - window_size):
        window     = prices[i : i + window_size]
        next_price = prices[i + window_size - 1 + 1] if (i + window_size) < len(prices) else prices[-1]
        last_price = prices[i + window_size - 1]

        label = 1 if next_price > last_price else 0
        windows.append(window)
        labels.append(label)
        news_texts.append(texts[i + window_size - 1])
        window_dates.append(dates[i + window_size - 1])

    import numpy as np
    windows = np.array(windows)
    labels  = np.array(labels)
    print(f"[Windows] {len(windows)} samples | "
          f"Bullish: {labels.sum()} | Bearish: {(1-labels).sum()}")
    return windows, labels, np.array(news_texts), np.array(window_dates)


def temporal_split(windows, labels, news_texts,
                   test_split=TEST_SPLIT, val_split=VAL_SPLIT):
    n          = len(windows)
    test_size  = int(n * test_split)
    val_size   = int((n - test_size) * val_split)
    train_size = n - test_size - val_size

    splits = {
        "train": (windows[:train_size],
                  labels[:train_size],
                  news_texts[:train_size]),
        "val":   (windows[train_size : train_size + val_size],
                  labels[train_size : train_size + val_size],
                  news_texts[train_size : train_size + val_size]),
        "test":  (windows[-test_size:],
                  labels[-test_size:],
                  news_texts[-test_size:]),
    }
    for name, (w, l, _) in splits.items():
        print(f"[Split] {name:5s}: {len(w):5d} samples | "
              f"Bullish: {l.sum()} | Bearish: {(1-l).sum()}")
    return splits


# ─────────────────────────────────────────────
# 4.  CHART GENERATION
# ─────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — must be before pyplot import
import matplotlib.pyplot as plt
from PIL import Image


def price_window_to_image(prices, idx, split="train", label=0):
    """Convert a 1-D price array to a color-coded 224×224 line-chart PNG."""
    import numpy as np

    folder = os.path.join(CHART_DIR, split, str(label))
    os.makedirs(folder, exist_ok=True)
    fpath  = os.path.join(folder, f"{idx}.png")

    if os.path.exists(fpath):
        return fpath

    fig, ax = plt.subplots(figsize=(2.24, 2.24), dpi=100)   # → 224×224 px
    p_min, p_max = prices.min(), prices.max()
    norm = (prices - p_min) / (p_max - p_min + 1e-8)

    # Color by direction: green = up, red = down
    color = "#27ae60" if norm[-1] >= norm[0] else "#e74c3c"
    x     = range(len(norm))

    ax.plot(norm, color=color, linewidth=2.5, zorder=2)
    ax.fill_between(x, norm, alpha=0.18, color=color, zorder=1)
    ax.set_facecolor("#f5f5f5")
    ax.set_xlim(0, len(prices) - 1)
    ax.set_ylim(-0.05, 1.05)
    ax.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(fpath, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return fpath


def generate_all_charts(windows, labels, split="train", verbose=True):
    paths = []
    n     = len(windows)
    for i, (prices, label) in enumerate(zip(windows, labels)):
        paths.append(price_window_to_image(prices, i, split=split, label=label))
        if verbose and (i + 1) % 500 == 0:
            print(f"  [{split}] {i+1}/{n} charts generated...")
    if verbose:
        print(f"[Charts] {split} — {n} images saved to {CHART_DIR}/{split}/")
    return paths


def load_image_as_tensor(path, img_size=IMG_SIZE):
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        # No Grayscale — keep the green/red color signal
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return transform(Image.open(path).convert("RGB"))


# ─────────────────────────────────────────────
# 4b. TECHNICAL FEATURES  (momentum, volatility, RSI)
# ─────────────────────────────────────────────
def compute_tech_features(window):
    """Return a 3-D float32 vector: [momentum, volatility, RSI/100]."""
    import numpy as np
    prices  = np.array(window, dtype="float32")
    returns = np.diff(prices) / (prices[:-1] + 1e-8)

    momentum   = float(prices[-1] / (prices[-5] + 1e-8) - 1) if len(prices) >= 5 else 0.0
    volatility = float(returns.std())

    gains  = returns[returns > 0].sum()
    losses = -returns[returns < 0].sum()
    rsi    = float(100.0 - 100.0 / (1.0 + gains / (losses + 1e-8)))

    return np.array([momentum, volatility, rsi / 100.0], dtype="float32")


# ─────────────────────────────────────────────
# 5.  PYTORCH DATASET
# ─────────────────────────────────────────────
from torch.utils.data import Dataset

class MarketDataset(Dataset):
    def __init__(self, chart_paths, news_texts, labels, windows,
                 tokenizer_name=FINBERT_MODEL, max_len=MAX_LEN):
        from transformers import AutoTokenizer
        self.chart_paths = chart_paths
        self.news_texts  = news_texts
        self.labels      = labels
        self.windows     = windows          # raw price windows for tech features
        self.max_len     = max_len
        print(f"[Dataset] Loading tokenizer: {tokenizer_name}")
        self.tokenizer   = AutoTokenizer.from_pretrained(tokenizer_name)
        print(f"[Dataset] Ready — {len(labels)} samples.")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        chart_img  = load_image_as_tensor(self.chart_paths[idx])
        tech_feats = torch.tensor(compute_tech_features(self.windows[idx]),
                                  dtype=torch.float32)
        enc = self.tokenizer(
            str(self.news_texts[idx]),
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return (
            chart_img,
            enc["input_ids"].squeeze(0),
            enc["attention_mask"].squeeze(0),
            tech_feats,
            torch.tensor(self.labels[idx], dtype=torch.float32),
        )


# ─────────────────────────────────────────────
# 6.  MODEL ARCHITECTURE
# ─────────────────────────────────────────────
import torch.nn as nn
from torchvision import models
from transformers import AutoModel


class CNNEncoder(nn.Module):
    """ResNet18 backbone — fully frozen, outputs 512-D feature vector.
    With only ~2800 samples, fine-tuning causes severe overfitting."""
    def __init__(self):
        super().__init__()
        backbone     = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.encoder = nn.Sequential(*list(backbone.children())[:-1])
        for p in self.encoder.parameters():
            p.requires_grad = False   # fully frozen

    def forward(self, x):
        feat = self.encoder(x)
        return feat.view(feat.size(0), -1)   # (B, 512)


class TextEncoder(nn.Module):
    """FinBERT — fully frozen, outputs 768-D [CLS] sentiment vector."""
    def __init__(self):
        super().__init__()
        self.bert = AutoModel.from_pretrained(FINBERT_MODEL)
        for p in self.bert.parameters():
            p.requires_grad = False   # fully frozen

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state[:, 0, :]   # (B, 768)


class MultimodalFusionModel(nn.Module):
    """price chart + news + tech indicators → MLP → logit.
    Only the MLP trains. BatchNorm + high dropout fight overfitting."""
    def __init__(self):
        super().__init__()
        self.cnn_encoder  = CNNEncoder()
        self.text_encoder = TextEncoder()
        # 512 (CNN) + 768 (BERT) + 3 (tech) = 1283
        joint_dim = CNN_EMBED_DIM + TEXT_EMBED_DIM + TECH_DIM

        self.fusion_mlp = nn.Sequential(
            nn.Linear(joint_dim,      FUSION_HIDDEN1),
            nn.BatchNorm1d(FUSION_HIDDEN1),           # stabilizes training
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(FUSION_HIDDEN1, FUSION_HIDDEN2),
            nn.BatchNorm1d(FUSION_HIDDEN2),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(FUSION_HIDDEN2, 1),
            # No Sigmoid — BCEWithLogitsLoss handles it
        )

    def forward(self, img, input_ids, attn_mask, tech):
        with torch.no_grad():                         # encoders always frozen
            Ep = self.cnn_encoder(img)
            En = self.text_encoder(input_ids, attn_mask)
        E_joint = torch.cat([Ep, En, tech], dim=1)   # (B, 1283)
        return self.fusion_mlp(E_joint).squeeze(1)    # raw logit

    def predict_proba(self, img, input_ids, attn_mask, tech):
        return torch.sigmoid(self.forward(img, input_ids, attn_mask, tech))

    def get_embeddings(self, img, input_ids, attn_mask, tech):
        with torch.no_grad():
            Ep = self.cnn_encoder(img)
            En = self.text_encoder(input_ids, attn_mask)
        return torch.cat([Ep, En, tech], dim=1)


# ─────────────────────────────────────────────
# 7.  TRAINING
# ─────────────────────────────────────────────
from torch.utils.data import DataLoader


def train_model(model, train_ds, val_ds):
    import numpy as np
    os.makedirs(MODEL_DIR, exist_ok=True)
    best_path = os.path.join(MODEL_DIR, "best_model.pt")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=(DEVICE == "cuda"))
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0, pin_memory=(DEVICE == "cuda"))

    # ── Weighted loss to handle class imbalance ──
    n_bull     = int(np.array(train_ds.labels).sum())
    n_bear     = len(train_ds.labels) - n_bull
    pos_weight = torch.tensor([n_bear / (n_bull + 1e-8)], device=DEVICE)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    print(f"[Training] Class weights — Bearish/Bullish ratio: {pos_weight.item():.3f}")

    # ── Only train the MLP — encoders are fully frozen ──
    optimizer = torch.optim.Adam(
        model.fusion_mlp.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY   # L2 regularization
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=3, factor=0.5)

    best_val_loss    = float("inf")
    patience_counter = 0
    history          = {"train_loss": [], "val_loss": [],
                        "train_acc":  [], "val_acc":  []}

    print(f"\n[Training] {EPOCHS} epochs on {DEVICE}\n{'='*55}")

    for epoch in range(1, EPOCHS + 1):
        # ── train ──
        model.train()
        t_loss, t_correct = 0.0, 0
        for img, ids, mask, tech, lbl in train_loader:
            img, ids, mask, tech, lbl = (img.to(DEVICE), ids.to(DEVICE),
                                         mask.to(DEVICE), tech.to(DEVICE),
                                         lbl.to(DEVICE))
            optimizer.zero_grad()
            logit = model(img, ids, mask, tech)
            loss  = criterion(logit, lbl)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            t_loss    += loss.item() * len(lbl)
            prob       = torch.sigmoid(logit)
            t_correct += ((prob >= 0.5).float() == lbl).sum().item()

        t_loss /= len(train_ds)
        t_acc   = t_correct / len(train_ds)

        # ── validate ──
        model.eval()
        v_loss, v_correct = 0.0, 0
        with torch.no_grad():
            for img, ids, mask, tech, lbl in val_loader:
                img, ids, mask, tech, lbl = (img.to(DEVICE), ids.to(DEVICE),
                                             mask.to(DEVICE), tech.to(DEVICE),
                                             lbl.to(DEVICE))
                logit  = model(img, ids, mask, tech)
                loss   = criterion(logit, lbl)
                v_loss += loss.item() * len(lbl)
                prob    = torch.sigmoid(logit)
                v_correct += ((prob >= 0.5).float() == lbl).sum().item()

        v_loss /= len(val_ds)
        v_acc   = v_correct / len(val_ds)
        scheduler.step(v_loss)

        history["train_loss"].append(t_loss)
        history["val_loss"].append(v_loss)
        history["train_acc"].append(t_acc)
        history["val_acc"].append(v_acc)

        if v_loss < best_val_loss:
            best_val_loss    = v_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_path)
        else:
            patience_counter += 1

        print(f"Epoch {epoch:3d}/{EPOCHS} | "
              f"Train Loss: {t_loss:.4f}  Acc: {t_acc:.4f} | "
              f"Val Loss: {v_loss:.4f}  Acc: {v_acc:.4f}")

        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"\n[Early Stop] No improvement for {EARLY_STOP_PATIENCE} epochs.")
            break

    print(f"\n[Training] Done. Best Val Loss: {best_val_loss:.4f}")
    print(f"[Training] Weights saved → {best_path}")

    # Reload best weights
    model.load_state_dict(torch.load(best_path, map_location=DEVICE,
                                     weights_only=True))

    # Save history
    hist_path = os.path.join(MODEL_DIR, "training_history.pkl")
    with open(hist_path, "wb") as f:
        pickle.dump(history, f)

    import pandas as pd
    pd.DataFrame(history).to_csv(
        os.path.join(RESULT_DIR, "training_history.csv"), index=False)

    return history


# ─────────────────────────────────────────────
# 8.  EVALUATION
# ─────────────────────────────────────────────
def evaluate_model(model, test_ds, history):
    import numpy as np
    from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                 f1_score, confusion_matrix, roc_auc_score, roc_curve)
    from sklearn.manifold import TSNE
    import seaborn as sns

    os.makedirs(RESULT_DIR, exist_ok=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=0)
    model.eval()

    all_preds, all_probs, all_labels, all_embeds = [], [], [], []
    with torch.no_grad():
        for img, ids, mask, tech, lbl in test_loader:
            img, ids, mask, tech = (img.to(DEVICE), ids.to(DEVICE),
                                    mask.to(DEVICE), tech.to(DEVICE))
            probs  = model.predict_proba(img, ids, mask, tech).cpu().numpy()
            embeds = model.get_embeddings(img, ids, mask, tech).cpu().numpy()
            preds  = (probs >= 0.5).astype(int)
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(lbl.numpy().astype(int))
            all_embeds.extend(embeds)

    all_probs  = np.array(all_probs)
    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_embeds = np.array(all_embeds)

    print("\n" + "="*50)
    print(f"  Accuracy  : {accuracy_score(all_labels, all_preds):.4f}")
    print(f"  Precision : {precision_score(all_labels, all_preds):.4f}")
    print(f"  Recall    : {recall_score(all_labels, all_preds):.4f}")
    print(f"  F1-Score  : {f1_score(all_labels, all_preds):.4f}")
    print(f"  AUC-ROC   : {roc_auc_score(all_labels, all_probs):.4f}")
    print("="*50)

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(5, 4))
    plt.style.use("default")
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
                xticklabels=["Bearish", "Bullish"],
                yticklabels=["Bearish", "Bullish"])
    plt.title("Confusion Matrix")
    plt.ylabel("Actual"); plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, "confusion_matrix.png"), dpi=150)
    plt.close()
    print(f"[Saved] {RESULT_DIR}/confusion_matrix.png")

    # ROC curve
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    auc_val     = roc_auc_score(all_labels, all_probs)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, color="#0097a7", lw=2, label=f"AUC = {auc_val:.3f}")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC Curve"); plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, "roc_curve.png"), dpi=150)
    plt.close()
    print(f"[Saved] {RESULT_DIR}/roc_curve.png")

    # Training curves
    if history:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
        ep = range(1, len(history["train_loss"]) + 1)
        ax1.plot(ep, history["train_loss"], label="Train", color="#e74c3c", lw=2)
        ax1.plot(ep, history["val_loss"],   label="Val",   color="#e67e22", lw=2, ls="--")
        ax1.set_title("Loss"); ax1.set_xlabel("Epoch"); ax1.legend()
        ax2.plot(ep, history["train_acc"], label="Train", color="#27ae60", lw=2)
        ax2.plot(ep, history["val_acc"],   label="Val",   color="#0097a7", lw=2, ls="--")
        ax2.set_title("Accuracy"); ax2.set_xlabel("Epoch"); ax2.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(RESULT_DIR, "training_curves.png"), dpi=150)
        plt.close()
        print(f"[Saved] {RESULT_DIR}/training_curves.png")

    # t-SNE
    print("[t-SNE] Running... (may take a moment)")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
    # Use only CNN (512-D) + BERT (768-D) = 1280-D, drop the 3 tech feature dims
    embeds_no_tech = all_embeds[:, :CNN_EMBED_DIM + TEXT_EMBED_DIM]   # (N, 1280)
    z2d  = tsne.fit_transform(embeds_no_tech)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.patch.set_facecolor("white")

    class_cfg = [
        (0, "#e74c3c", "o", "Bearish"),
        (1, "#27ae60", "^", "Bullish"),
    ]

    # Left: scatter plot with convex hull outlines
    ax1 = axes[0]
    ax1.set_facecolor("#f8f9fa")
    for cls, color, marker, name in class_cfg:
        m = all_labels == cls
        pts = z2d[m]
        ax1.scatter(pts[:, 0], pts[:, 1], c=color, s=20, alpha=0.5,
                    label=f"{name} (n={m.sum()})", edgecolors="none", marker=marker)
        # Convex hull outline
        if len(pts) >= 4:
            try:
                from scipy.spatial import ConvexHull
                hull = ConvexHull(pts)
                hull_pts = np.append(hull.vertices, hull.vertices[0])
                ax1.plot(pts[hull_pts, 0], pts[hull_pts, 1],
                         color=color, lw=1.5, alpha=0.6, linestyle="--")
            except Exception:
                pass

    ax1.set_title("Scatter + Convex Hulls", fontsize=12,
                  fontweight="bold", pad=10)
    ax1.set_xlabel("Dimension 1", fontsize=10)
    ax1.set_ylabel("Dimension 2", fontsize=10)
    ax1.legend(fontsize=10, framealpha=0.9)
    ax1.grid(True, alpha=0.3, linewidth=0.5)
    for sp in ax1.spines.values(): sp.set_edgecolor("#dee2e6")

    # Right: density/KDE overlay — shows where each class concentrates
    ax2 = axes[1]
    ax2.set_facecolor("#f8f9fa")
    try:
        from scipy.stats import gaussian_kde
        for cls, color, marker, name in class_cfg:
            m   = all_labels == cls
            pts = z2d[m]
            ax2.scatter(pts[:, 0], pts[:, 1], c=color, s=15, alpha=0.3,
                        edgecolors="none", marker=marker)
            # KDE contour
            if len(pts) >= 10:
                kde  = gaussian_kde(pts.T, bw_method=0.35)
                xmin, xmax = z2d[:, 0].min() - 2, z2d[:, 0].max() + 2
                ymin, ymax = z2d[:, 1].min() - 2, z2d[:, 1].max() + 2
                xx, yy = np.mgrid[xmin:xmax:80j, ymin:ymax:80j]
                zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
                ax2.contour(xx, yy, zz, levels=5, colors=[color],
                            alpha=0.7, linewidths=1.2)
    except Exception:
        # Fallback: plain scatter if scipy not available
        for cls, color, marker, name in class_cfg:
            m = all_labels == cls
            ax2.scatter(z2d[m, 0], z2d[m, 1], c=color, s=20, alpha=0.5,
                        edgecolors="none", marker=marker, label=name)

    ax2.set_title("Density Contours", fontsize=12,
                  fontweight="bold", pad=10)
    ax2.set_xlabel("Dimension 1", fontsize=10)
    ax2.set_ylabel("Dimension 2", fontsize=10)
    ax2.grid(True, alpha=0.3, linewidth=0.5)
    for sp in ax2.spines.values(): sp.set_edgecolor("#dee2e6")

    # Shared annotation
    acc_str = f"Model Acc: {accuracy_score(all_labels, all_preds)*100:.1f}%  |  AUC: {roc_auc_score(all_labels, all_probs):.3f}"
    fig.text(0.5, -0.02, acc_str, ha="center", fontsize=11,
             bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#dee2e6"))

    # Legend patches
    from matplotlib.patches import Patch
    legend_els = [Patch(facecolor=c, label=n) for _, c, _, n in class_cfg]
    fig.legend(handles=legend_els, loc="upper center", ncol=2,
               fontsize=10, framealpha=0.9, bbox_to_anchor=(0.5, 1.02))

    plt.suptitle("t-SNE — Joint Embedding Space",
                 fontsize=14, fontweight="bold", y=1.06)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, "tsne_clusters.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Saved] {RESULT_DIR}/tsne_clusters.png")

    print(f"\n[Evaluation] Complete. Results in {RESULT_DIR}/")
    return all_preds, all_probs, all_labels


# ─────────────────────────────────────────────
# 9.  BASELINE COMPARISON
# ─────────────────────────────────────────────
def run_baselines(splits, train_ds, test_ds, all_preds, all_labels, all_probs):
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import (accuracy_score, precision_score,
                                 recall_score, f1_score, roc_auc_score)
    from torch.utils.data import TensorDataset

    train_w, train_l, _ = splits["train"]
    test_w,  test_l,  _ = splits["test"]

    def get_metrics(name, y_true, y_pred, y_prob=None):
        auc = round(roc_auc_score(y_true, y_prob) * 100, 2) \
              if y_prob is not None else None
        m = {
            "Model":     name,
            "Accuracy":  round(accuracy_score(y_true, y_pred)  * 100, 2),
            "Precision": round(precision_score(y_true, y_pred, zero_division=0) * 100, 2),
            "Recall":    round(recall_score(y_true, y_pred,    zero_division=0) * 100, 2),
            "F1-Score":  round(f1_score(y_true, y_pred,        zero_division=0) * 100, 2),
            "AUC-ROC":   auc if auc is not None else 0.0,
        }
        auc_str = f"  AUC={m['AUC-ROC']}%" if auc is not None else ""
        print(f"  {name:<25}  Acc={m['Accuracy']}%  F1={m['F1-Score']}%{auc_str}")
        return m

    def price_features(windows):
        """20 rich technical features per window — gives LR/SVM real signal."""
        feats = []
        for w in windows:
            w = np.array(w, dtype="float32")
            returns = np.diff(w) / (w[:-1] + 1e-8)

            # Trend
            net_return   = (w[-1] - w[0]) / (w[0] + 1e-8)
            slope        = np.polyfit(np.arange(len(w)), w, 1)[0] / (w.mean() + 1e-8)
            # Momentum
            mom3  = (w[-1] - w[-4]) / (w[-4] + 1e-8) if len(w) >= 4 else 0.0
            mom5  = (w[-1] - w[-6]) / (w[-6] + 1e-8) if len(w) >= 6 else 0.0
            # Volatility
            vol   = returns.std()
            vol3  = returns[-3:].std() if len(returns) >= 3 else vol
            # RSI
            gains  = returns[returns > 0].sum()
            losses = -returns[returns < 0].sum()
            rsi    = 100.0 - 100.0 / (1.0 + gains / (losses + 1e-8))
            # Moving average crossover
            ma5  = w[-5:].mean()  if len(w) >= 5  else w.mean()
            ma10 = w.mean()
            ma_cross = (ma5 - ma10) / (ma10 + 1e-8)
            # Price position
            p_min, p_max = w.min(), w.max()
            price_pos = (w[-1] - p_min) / (p_max - p_min + 1e-8)
            # Up/down day counts
            up_days   = (returns > 0).sum() / len(returns)
            down_days = (returns < 0).sum() / len(returns)
            # Consecutive direction
            last3_up  = float(all(returns[-3:] > 0)) if len(returns) >= 3 else 0.0
            last3_dn  = float(all(returns[-3:] < 0)) if len(returns) >= 3 else 0.0
            # Range features
            high_low_range = (p_max - p_min) / (p_min + 1e-8)
            last_ret  = float(returns[-1]) if len(returns) > 0 else 0.0
            mean_ret  = float(returns.mean())
            skew_ret  = float(((returns - returns.mean())**3).mean() /
                              (returns.std()**3 + 1e-8))

            feats.append([
                net_return, slope, mom3, mom5, vol, vol3, rsi/100.0,
                ma_cross, price_pos, up_days, down_days,
                last3_up, last3_dn, high_low_range,
                last_ret, mean_ret, skew_ret,
                w.mean(), w.std(), float(w[-1])
            ])
        return np.array(feats, dtype="float32")

    results = []
    print("\n" + "="*55)
    print("  MODEL COMPARISON")
    print("="*55)

    # 1. Logistic Regression
    print("\n[1] Logistic Regression...")
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(price_features(train_w))
    X_test  = scaler.transform(price_features(test_w))
    lr_clf  = LogisticRegression(max_iter=2000, random_state=42,
                                  class_weight="balanced", C=0.1)
    lr_clf.fit(X_train, train_l)
    lr_probs = lr_clf.predict_proba(X_test)[:, 1]
    results.append(get_metrics("Logistic Regression", test_l,
                               lr_clf.predict(X_test), lr_probs))

    # 2. SVM
    print("\n[2] SVM...")
    svm_clf = SVC(kernel="rbf", random_state=42, class_weight="balanced",
                  C=1.0, gamma="scale", probability=True)   # probability=True for AUC
    svm_clf.fit(X_train, train_l)
    svm_probs = svm_clf.predict_proba(X_test)[:, 1]
    results.append(get_metrics("SVM", test_l, svm_clf.predict(X_test), svm_probs))

    # 3. BiLSTM
    print("\n[3] BiLSTM (price only)...")

    class BiLSTMModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(1, 64, num_layers=2, batch_first=True,
                                bidirectional=True, dropout=0.3)
            self.head = nn.Sequential(
                nn.Linear(128, 32), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(32, 1)   # raw logit — use BCEWithLogitsLoss
            )
        def forward(self, x):
            out, _ = self.lstm(x)
            return self.head(out[:, -1, :]).squeeze(1)

    # Normalize each window to [0,1] so LSTM sees returns, not raw prices
    def normalize_windows(w):
        mn = w.min(axis=1, keepdims=True)
        mx = w.max(axis=1, keepdims=True)
        return (w - mn) / (mx - mn + 1e-8)

    X_tr = torch.tensor(normalize_windows(train_w), dtype=torch.float32).unsqueeze(-1)
    y_tr = torch.tensor(train_l, dtype=torch.float32)
    X_te = torch.tensor(normalize_windows(test_w),  dtype=torch.float32).unsqueeze(-1)

    bilstm   = BiLSTMModel().to(DEVICE)
    loader   = DataLoader(TensorDataset(X_tr, y_tr), batch_size=BATCH_SIZE,
                          shuffle=True)   # shuffle=True for proper training
    opt      = torch.optim.Adam(bilstm.parameters(), lr=1e-3, weight_decay=1e-4)
    n_bull_b = int(y_tr.sum().item())
    n_bear_b = len(y_tr) - n_bull_b
    pw_b     = torch.tensor([n_bear_b / (n_bull_b + 1e-8)], device=DEVICE)
    crit_b   = nn.BCEWithLogitsLoss(pos_weight=pw_b)
    for _ in range(20):   # 20 epochs with proper loss
        bilstm.train()
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            crit_b(bilstm(xb), yb).backward()
            torch.nn.utils.clip_grad_norm_(bilstm.parameters(), 1.0)
            opt.step()
    bilstm.eval()
    with torch.no_grad():
        bilstm_probs = torch.sigmoid(bilstm(X_te.to(DEVICE))).cpu().numpy()
        bilstm_preds = (bilstm_probs >= 0.5).astype(int)
    results.append(get_metrics("BiLSTM (Price Only)", test_l, bilstm_preds, bilstm_probs))

    # 4. CNN Only
    print("\n[4] CNN Only (charts)...")

    class CNNOnlyModel(nn.Module):
        def __init__(self):
            super().__init__()
            bb = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            self.enc  = nn.Sequential(*list(bb.children())[:-1])
            for p in self.enc.parameters(): p.requires_grad = False
            self.head = nn.Sequential(
                nn.Linear(512, 64), nn.BatchNorm1d(64), nn.ReLU(),
                nn.Dropout(0.4), nn.Linear(64, 1)   # raw logit
            )
        def forward(self, img, *_):
            with torch.no_grad():
                f = self.enc(img)
            return self.head(f.view(f.size(0), -1)).squeeze(1)

    cnn_only  = CNNOnlyModel().to(DEVICE)
    tr_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    te_loader = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    opt_c     = torch.optim.Adam(
        filter(lambda p: p.requires_grad, cnn_only.parameters()),
        lr=1e-3, weight_decay=1e-4
    )
    n_bull_c  = int(sum(lbl.item() for _, _, _, _, lbl in train_ds))
    n_bear_c  = len(train_ds) - n_bull_c
    pw_c      = torch.tensor([n_bear_c / (n_bull_c + 1e-8)], device=DEVICE)
    crit_c    = nn.BCEWithLogitsLoss(pos_weight=pw_c)
    for ep in range(15):
        cnn_only.train()
        for img, ids, mask, tech, lbl in tr_loader:
            img, lbl = img.to(DEVICE), lbl.to(DEVICE)
            opt_c.zero_grad()
            crit_c(cnn_only(img), lbl).backward()
            torch.nn.utils.clip_grad_norm_(cnn_only.parameters(), 1.0)
            opt_c.step()
    cnn_only.eval()
    cnn_preds = []
    cnn_probs = []
    with torch.no_grad():
        for img, ids, mask, tech, _ in te_loader:
            p = torch.sigmoid(cnn_only(img.to(DEVICE))).cpu().numpy()
            cnn_probs.extend(p)
            cnn_preds.extend((p >= 0.5).astype(int))
    results.append(get_metrics("CNN Only (Charts)", test_l,
                               np.array(cnn_preds), np.array(cnn_probs)))

    # 5. Multimodal (ours)
    print("\n[5] Multimodal (Ours)...")
    results.append(get_metrics("Multimodal (Ours)", all_labels, all_preds, all_probs))

    # Table + chart
    df_r = pd.DataFrame(results).set_index("Model")
    print("\n" + "="*55)
    print("  FINAL RESULTS")
    print("="*55)
    print(df_r.to_string())
    os.makedirs(RESULT_DIR, exist_ok=True)
    df_r.to_csv(os.path.join(RESULT_DIR, "comparison_results.csv"))

    # ── Comparison chart — clean, accurate, readable ──
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.patch.set_facecolor("white")

    model_names  = list(df_r.index)
    n_models     = len(model_names)
    bar_colors   = ["#5b8dee", "#e74c3c", "#f39c12", "#9b59b6", "#27ae60"][:n_models]
    random_base  = (test_l.sum() / len(test_l)) * 100   # majority-class baseline

    # Left: Accuracy bar chart
    ax1 = axes[0]
    ax1.set_facecolor("#f8f9fa")
    bars = ax1.bar(model_names, df_r["Accuracy"], color=bar_colors,
                   alpha=0.88, edgecolor="white", linewidth=1.2, zorder=3)
    # Random baseline line
    ax1.axhline(random_base, color="#e74c3c", linestyle="--", linewidth=1.5,
                label=f"Majority-class baseline ({random_base:.1f}%)", zorder=4)
    # Value labels on bars
    for bar, val in zip(bars, df_r["Accuracy"]):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.5,
                 f"{val:.1f}%",
                 ha="center", va="bottom", fontsize=10, fontweight="bold",
                 color="#333333")
    ax1.set_title("Accuracy Comparison", fontsize=13, fontweight="bold", pad=10)
    ax1.set_ylabel("Accuracy (%)", fontsize=11)
    ax1.set_ylim(40, min(100, df_r["Accuracy"].max() + 10))
    ax1.set_xticklabels(model_names, rotation=20, ha="right", fontsize=9)
    ax1.legend(fontsize=9, loc="lower right")
    ax1.grid(axis="y", color="#dee2e6", lw=0.6, zorder=0)
    for sp in ax1.spines.values(): sp.set_edgecolor("#dee2e6")

    # Right: All metrics grouped bar chart
    ax2 = axes[1]
    ax2.set_facecolor("#f8f9fa")
    x       = np.arange(n_models)
    w       = 0.15
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "AUC-ROC"]
    mcols   = ["#5b8dee", "#e67e22", "#9b59b6", "#27ae60", "#e74c3c"]
    for i, (met, mc) in enumerate(zip(metrics, mcols)):
        b = ax2.bar(x + i * w, df_r[met], w, color=mc,
                    alpha=0.85, label=met, zorder=3)
        for bar in b:
            ax2.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.4,
                     f"{bar.get_height():.0f}",
                     ha="center", va="bottom", fontsize=7, color=mc, fontweight="bold")
    ax2.set_title("All Metrics Comparison", fontsize=13, fontweight="bold", pad=10)
    ax2.set_ylabel("Score (%)", fontsize=11)
    ax2.set_ylim(0, 110)
    ax2.set_xticks(x + w * 2)
    ax2.set_xticklabels(model_names, rotation=20, ha="right", fontsize=9)
    ax2.legend(fontsize=9, loc="upper right")
    ax2.grid(axis="y", color="#dee2e6", lw=0.6, zorder=0)
    for sp in ax2.spines.values(): sp.set_edgecolor("#dee2e6")

    plt.suptitle("MarketMind — Model Comparison", fontsize=15,
                 fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, "model_comparison_chart.png"),
                dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"[Saved] {RESULT_DIR}/model_comparison_chart.png")


# ─────────────────────────────────────────────
# 10. LIVE PREDICTION
# ─────────────────────────────────────────────
def predict_tomorrow(model, tokenizer, prices, headline,
                     img_size=IMG_SIZE, max_len=MAX_LEN):
    """Run one inference: 10 prices + headline → Bullish/Bearish + probability."""
    import numpy as np
    from torchvision import transforms

    if len(prices) < 10:
        return "Error: need 10 prices.", 0.0, 0.0
    if not str(headline).strip():
        headline = "No significant news today."

    model.eval()
    try:
        prices_arr = np.array(prices, dtype="float32")
        p_min, p_max = prices_arr.min(), prices_arr.max()
        norm  = (prices_arr - p_min) / (p_max - p_min + 1e-8)
        color = "#27ae60" if norm[-1] >= norm[0] else "#e74c3c"

        # Build color-coded chart image in memory
        fig, ax = plt.subplots(figsize=(2.24, 2.24), dpi=100)
        ax.plot(norm, color=color, linewidth=2.5, zorder=2)
        ax.fill_between(range(len(norm)), norm, alpha=0.18, color=color, zorder=1)
        ax.set_facecolor("#f5f5f5")
        ax.axis("off")
        fig.canvas.draw()
        img_arr = np.array(fig.canvas.renderer.buffer_rgba())
        img_pil = Image.fromarray(img_arr).convert("RGB")
        plt.close(fig)

        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            # No Grayscale — keep color signal
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        img_tensor = transform(img_pil).unsqueeze(0).to(DEVICE)

        # Technical features
        tech_tensor = torch.tensor(
            compute_tech_features(prices_arr), dtype=torch.float32
        ).unsqueeze(0).to(DEVICE)

        enc = tokenizer(str(headline).strip(), padding="max_length",
                        truncation=True, max_length=max_len, return_tensors="pt")
        input_ids = enc["input_ids"].to(DEVICE)
        attn_mask = enc["attention_mask"].to(DEVICE)

        with torch.no_grad():
            prob = model.predict_proba(img_tensor, input_ids,
                                       attn_mask, tech_tensor).item()

        up_pct   = prob * 100
        down_pct = (1 - prob) * 100
        label    = "Bullish 🟢" if prob >= 0.5 else "Bearish 🔴"
        return label, up_pct, down_pct

    except Exception as e:
        return f"Inference Error: {e}", 0.0, 0.0


def run_live_prediction(model):
    import yfinance as yf
    from transformers import AutoTokenizer
    from datetime import datetime

    os.makedirs(RESULT_DIR, exist_ok=True)
    print("\n" + "="*50)
    print(f"  LIVE PREDICTION SYSTEM  ({datetime.now().strftime('%Y-%m-%d')})")
    print("="*50)

    # Fetch prices
    print("Fetching last 10 trading days for S&P 500...")
    try:
        data = yf.download("^GSPC", period="1mo", progress=False)
        if data.empty:
            raise ValueError("Yahoo Finance returned empty data.")
        prices = data["Close"].tail(10).squeeze().tolist()
        if len(prices) < 10:
            raise ValueError(f"Only {len(prices)} days available.")
        print(f"✅ Latest close: ${prices[-1]:.2f}")
    except Exception as e:
        print(f"❌ Price fetch failed: {e}")
        return

    # Get headline
    print("\nTIP: Paste a single headline or short summary.")
    headline = input("📰 Today's top financial headline: ").strip()

    tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL)
    label, up_pct, down_pct = predict_tomorrow(model, tokenizer, prices, headline)

    print("\n" + "="*55)
    print("  PREDICTION RESULTS")
    print("="*55)
    print(f"  News      : {headline[:75]}{'...' if len(headline)>75 else ''}")
    print(f"  Prices    : ${prices[0]:.2f} → ${prices[-1]:.2f}")
    print(f"  Prediction: {label}")
    print(f"  UP   prob : {up_pct:.2f}%")
    print(f"  DOWN prob : {down_pct:.2f}%")
    print("="*55)

    # Log result
    from datetime import datetime
    log_path = os.path.join(RESULT_DIR, "live_predictions_log.txt")
    with open(log_path, "a") as f:
        f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
                f"{headline[:50]} | ${prices[0]:.2f}→${prices[-1]:.2f} | "
                f"{label} (Up:{up_pct:.1f}%)\n")
    print(f"[Saved] {log_path}")


# ─────────────────────────────────────────────
# 11. FLASK API  (optional — run separately)
# ─────────────────────────────────────────────
def start_flask_server():
    """
    Starts the REST API on http://localhost:5000
    Endpoints:
        GET  /live-data   → last 10 S&P 500 closes
        POST /predict     → { prices: [...], headline: "..." } → prediction
    """
    try:
        from flask import Flask, jsonify, request
        from flask_cors import CORS
        import yfinance as yf
        from transformers import AutoTokenizer
        import io
    except ImportError:
        print("[Flask] Install flask and flask-cors: pip install flask flask-cors")
        return

    flask_app = Flask(__name__)
    CORS(flask_app)

    # Load model for Flask
    best_path = os.path.join(MODEL_DIR, "best_model.pt")
    flask_model = MultimodalFusionModel().to(DEVICE)
    try:
        flask_model.load_state_dict(
            torch.load(best_path, map_location=DEVICE, weights_only=True))
        flask_model.eval()
        print(f"[Flask] Model loaded from {best_path}")
    except Exception as e:
        print(f"[Flask] WARNING: Could not load model ({e}). Running in demo mode.")
        flask_model = None

    flask_tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL)

    @flask_app.route("/live-data", methods=["GET"])
    def get_live_data():
        try:
            hist   = yf.Ticker("^GSPC").history(period="1mo")
            prices = hist["Close"].tail(10).values.flatten().tolist()
            return jsonify({"status": "success", "prices": prices}), 200
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500

    @flask_app.route("/predict", methods=["POST"])
    def predict():
        data     = request.json or {}
        prices   = data.get("prices", [])
        headline = data.get("headline", "")

        if len(prices) != 10 or not headline:
            return jsonify({"error": "Need 10 prices and a headline."}), 400

        if flask_model is None:
            import random
            trend     = prices[-1] - prices[0]
            mock_prob = min(0.98, max(0.02, (0.6 if trend > 0 else 0.3)
                                     + random.random() * 0.2 - 0.1))
            return jsonify({
                "prediction": "Bullish" if mock_prob >= 0.5 else "Bearish",
                "probability": mock_prob,
                "confidence": "Demo"
            })

        label, up_pct, _ = predict_tomorrow(
            flask_model, flask_tokenizer, prices, headline)
        prob = up_pct / 100
        return jsonify({
            "prediction": "Bullish" if prob >= 0.5 else "Bearish",
            "probability": prob,
            "confidence": "High" if prob >= 0.75 or prob <= 0.25 else "Medium"
        }), 200

    print("[Flask] Starting server on http://localhost:5000  (Ctrl+C to stop)")
    flask_app.run(host="0.0.0.0", port=5000, debug=False)


# ─────────────────────────────────────────────
# 12. MAIN ENTRY POINT
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Multimodal Stock Prediction — ResNet18 + FinBERT")
    parser.add_argument("--skip-train",    action="store_true",
                        help="Skip training; load existing model weights")
    parser.add_argument("--skip-baselines", action="store_true",
                        help="Skip baseline model comparison (saves time)")
    parser.add_argument("--predict-only",  action="store_true",
                        help="Run live prediction only (model must already exist)")
    parser.add_argument("--flask",         action="store_true",
                        help="Start the Flask REST API server")
    args = parser.parse_args()

    # Create output directories
    for d in [CHART_DIR, MODEL_DIR, RESULT_DIR,
              os.path.join(BASE_DIR, "data")]:
        os.makedirs(d, exist_ok=True)

    print(f"[Config] Device : {DEVICE}")
    print(f"[Config] Data   : {DATA_PATH}")
    print(f"[Config] Charts : {CHART_DIR}")
    print(f"[Config] Models : {MODEL_DIR}")
    print(f"[Config] Results: {RESULT_DIR}")

    # ── Flask-only mode ──────────────────────────────────
    if args.flask:
        start_flask_server()
        return

    # ── Predict-only mode ────────────────────────────────
    if args.predict_only:
        best_path = os.path.join(MODEL_DIR, "best_model.pt")
        if not os.path.exists(best_path):
            print(f"[ERROR] No trained model found at {best_path}.")
            print("  Run without --predict-only first to train the model.")
            sys.exit(1)
        model = MultimodalFusionModel().to(DEVICE)
        model.load_state_dict(torch.load(best_path, map_location=DEVICE,
                                         weights_only=True))
        run_live_prediction(model)
        return

    # ── Full pipeline ────────────────────────────────────
    # Step 1: Load data
    merged = load_and_clean(DATA_PATH)

    # Step 2: Windows + splits
    windows, labels, news_texts, dates = create_windows_and_labels(merged)
    splits = temporal_split(windows, labels, news_texts)

    train_w, train_l, train_n = splits["train"]
    val_w,   val_l,   val_n   = splits["val"]
    test_w,  test_l,  test_n  = splits["test"]

    # Save splits
    splits_path = os.path.join(MODEL_DIR, "data_splits.pkl")
    with open(splits_path, "wb") as f:
        pickle.dump(splits, f)
    print(f"[Saved] Splits → {splits_path}")

    # Step 3: Generate charts
    print("\n[Step 3] Generating price charts (224×224, color-coded)...")
    # Only clear and regenerate if charts don't exist yet
    charts_exist = all(
        os.path.exists(os.path.join(CHART_DIR, s))
        for s in ["train", "val", "test"]
    )
    if not charts_exist:
        import shutil
        for split_name in ["train", "val", "test"]:
            old_dir = os.path.join(CHART_DIR, split_name)
            if os.path.exists(old_dir):
                shutil.rmtree(old_dir)
                print(f"  [Charts] Cleared old charts: {old_dir}")
    else:
        print("  [Charts] Reusing existing 224×224 charts.")
    train_paths = generate_all_charts(train_w, train_l, split="train")
    val_paths   = generate_all_charts(val_w,   val_l,   split="val")
    test_paths  = generate_all_charts(test_w,  test_l,  split="test")

    # Step 4: Build datasets
    print("\n[Step 4] Building datasets...")
    train_ds = MarketDataset(train_paths, train_n, train_l, train_w)
    val_ds   = MarketDataset(val_paths,   val_n,   val_l,   val_w)
    test_ds  = MarketDataset(test_paths,  test_n,  test_l,  test_w)

    # Step 5: Build model
    print(f"\n[Step 5] Building model on {DEVICE}...")
    model = MultimodalFusionModel().to(DEVICE)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] Trainable params: {trainable:,}")

    # Step 6: Train (or load)
    best_path = os.path.join(MODEL_DIR, "best_model.pt")
    history   = None

    if args.skip_train and os.path.exists(best_path):
        print(f"\n[Step 6] Loading existing weights from {best_path}")
        model.load_state_dict(torch.load(best_path, map_location=DEVICE,
                                         weights_only=True))
        hist_pkl = os.path.join(MODEL_DIR, "training_history.pkl")
        if os.path.exists(hist_pkl):
            with open(hist_pkl, "rb") as f:
                history = pickle.load(f)
    else:
        print("\n[Step 6] Training...")
        os.makedirs(RESULT_DIR, exist_ok=True)
        history = train_model(model, train_ds, val_ds)

    # Step 7: Evaluate
    print("\n[Step 7] Evaluating on test set...")
    all_preds, all_probs, all_labels = evaluate_model(model, test_ds, history)

    # Step 8: Baseline comparison
    if args.skip_baselines:
        print("\n[Step 8] Baseline comparison skipped (--skip-baselines).")
    else:
        print("\n[Step 8] Running baseline comparison...")
        run_baselines(splits, train_ds, test_ds, all_preds, all_labels, all_probs)

    # Step 9: Live prediction (skipped in pipeline — use app.py for the web UI)
    print("\n[Step 9] Training complete. Run  py app.py  to start the web UI.")
    print(f"         Open http://localhost:5000 in your browser.")


if __name__ == "__main__":
    main()
