# -*- coding: utf-8 -*-
"""
app.py  —  Flask backend for the MarketMind Prediction UI
Run:  py app.py
API:
    GET  /api/live-prices   → last 10 S&P 500 closes + dates
    POST /api/predict       → { prices, headline } → prediction
    GET  /api/model-status  → whether trained weights exist
    GET  /api/history       → training history (if exists)
"""

import os
import io
import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR  = os.path.join(BASE_DIR, "models")
RESULT_DIR = os.path.join(BASE_DIR, "results")
FRONTEND   = os.path.join(BASE_DIR, "frontend")

MODEL_PATH   = os.path.join(MODEL_DIR, "best_model.pt")
HISTORY_PATH = os.path.join(MODEL_DIR, "training_history.pkl")

# ── Model config — must match minor.py exactly ────────────────────────────────
FINBERT_MODEL  = "ProsusAI/finbert"
CNN_EMBED_DIM  = 512
TEXT_EMBED_DIM = 768
TECH_DIM       = 3        # momentum, volatility, RSI
FUSION_HIDDEN1 = 256
FUSION_HIDDEN2 = 128
DROPOUT_RATE   = 0.5
IMG_SIZE       = 224      # updated from 64
MAX_LEN        = 128      # updated from 64

# ── Lazy-loaded globals ───────────────────────────────────────────────────────
_model     = None
_tokenizer = None
_device    = None


def get_device():
    global _device
    if _device is None:
        import torch
        _device = "cuda" if torch.cuda.is_available() else "cpu"
    return _device


def compute_tech_features(window):
    """3-D float32 vector: [momentum, volatility, RSI/100]."""
    prices  = np.array(window, dtype="float32")
    returns = np.diff(prices) / (prices[:-1] + 1e-8)
    momentum   = float(prices[-1] / (prices[-5] + 1e-8) - 1) if len(prices) >= 5 else 0.0
    volatility = float(returns.std())
    gains  = returns[returns > 0].sum()
    losses = -returns[returns < 0].sum()
    rsi    = float(100.0 - 100.0 / (1.0 + gains / (losses + 1e-8)))
    return np.array([momentum, volatility, rsi / 100.0], dtype="float32")


def load_model():
    """Load model + tokenizer once, cache globally."""
    global _model, _tokenizer
    if _model is not None:
        return _model, _tokenizer

    import torch
    import torch.nn as nn
    from torchvision import models
    from transformers import AutoModel, AutoTokenizer

    device = get_device()

    class CNNEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            backbone     = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            self.encoder = nn.Sequential(*list(backbone.children())[:-1])
            for p in self.encoder.parameters():
                p.requires_grad = False   # fully frozen
        def forward(self, x):
            with torch.no_grad():
                f = self.encoder(x)
            return f.view(f.size(0), -1)

    class TextEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.bert = AutoModel.from_pretrained(FINBERT_MODEL)
            for p in self.bert.parameters():
                p.requires_grad = False   # fully frozen
        def forward(self, input_ids, attention_mask):
            with torch.no_grad():
                out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            return out.last_hidden_state[:, 0, :]

    class MultimodalFusionModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.cnn_encoder  = CNNEncoder()
            self.text_encoder = TextEncoder()
            joint_dim = CNN_EMBED_DIM + TEXT_EMBED_DIM + TECH_DIM  # 1283
            self.fusion_mlp = nn.Sequential(
                nn.Linear(joint_dim,      FUSION_HIDDEN1),
                nn.BatchNorm1d(FUSION_HIDDEN1),
                nn.ReLU(),
                nn.Dropout(DROPOUT_RATE),
                nn.Linear(FUSION_HIDDEN1, FUSION_HIDDEN2),
                nn.BatchNorm1d(FUSION_HIDDEN2),
                nn.ReLU(),
                nn.Dropout(DROPOUT_RATE),
                nn.Linear(FUSION_HIDDEN2, 1),
            )
        def forward(self, img, input_ids, attn_mask, tech):
            Ep = self.cnn_encoder(img)
            En = self.text_encoder(input_ids, attn_mask)
            return self.fusion_mlp(torch.cat([Ep, En, tech], dim=1)).squeeze(1)

        def predict_proba(self, img, input_ids, attn_mask, tech):
            return torch.sigmoid(self.forward(img, input_ids, attn_mask, tech))

    model = MultimodalFusionModel().to(device)

    if os.path.exists(MODEL_PATH):
        model.load_state_dict(
            torch.load(MODEL_PATH, map_location=device, weights_only=True))
        model.eval()
        print(f"[Model] Loaded weights from {MODEL_PATH}")
    else:
        model.eval()
        print(f"[Model] WARNING: No weights at {MODEL_PATH}. Running in demo mode.")

    tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL)
    _model, _tokenizer = model, tokenizer
    return model, tokenizer


def run_inference(prices, headline):
    """Core inference — returns (label, up_pct, down_pct, prob, confidence)."""
    import torch
    from torchvision import transforms
    from PIL import Image

    model, tokenizer = load_model()
    device = get_device()

    prices_arr = np.array(prices, dtype="float32")
    p_min, p_max = prices_arr.min(), prices_arr.max()
    norm  = (prices_arr - p_min) / (p_max - p_min + 1e-8)
    color = "#27ae60" if norm[-1] >= norm[0] else "#e74c3c"

    # Build color-coded 224×224 chart in memory
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
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    img_tensor = transform(img_pil).unsqueeze(0).to(device)

    # Technical features
    tech_tensor = torch.tensor(
        compute_tech_features(prices_arr), dtype=torch.float32
    ).unsqueeze(0).to(device)

    enc = tokenizer(str(headline).strip(), padding="max_length",
                    truncation=True, max_length=MAX_LEN, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attn_mask = enc["attention_mask"].to(device)

    with torch.no_grad():
        prob = model.predict_proba(img_tensor, input_ids,
                                   attn_mask, tech_tensor).item()

    up_pct     = round(prob * 100, 2)
    down_pct   = round((1 - prob) * 100, 2)
    label      = "Bullish" if prob >= 0.5 else "Bearish"
    confidence = ("High"   if prob >= 0.75 or prob <= 0.25 else
                  "Medium" if prob >= 0.60 or prob <= 0.40 else "Low")
    return label, up_pct, down_pct, round(prob, 4), confidence


# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder=FRONTEND, static_url_path="")
CORS(app)


@app.route("/")
def index():
    return send_from_directory(FRONTEND, "index.html")


@app.route("/api/model-status")
def model_status():
    exists  = os.path.exists(MODEL_PATH)
    size_mb = round(os.path.getsize(MODEL_PATH) / 1e6, 1) if exists else 0
    return jsonify({
        "model_loaded": exists,
        "model_path":   MODEL_PATH,
        "size_mb":      size_mb,
        "device":       get_device(),
        "mode":         "inference" if exists else "demo",
    })


@app.route("/api/live-prices")
def live_prices():
    try:
        import yfinance as yf
        ticker = request.args.get("ticker", "^GSPC")
        hist   = yf.Ticker(ticker).history(period="1mo")
        if hist.empty:
            raise ValueError("No data returned from Yahoo Finance.")
        tail   = hist.tail(10)
        prices = [round(float(p), 2) for p in tail["Close"].values]
        dates  = [str(d.date()) for d in tail.index]
        change = round(prices[-1] - prices[-2], 2) if len(prices) >= 2 else 0
        pct    = round((prices[-1] - prices[-2]) / prices[-2] * 100, 2) \
                 if len(prices) >= 2 else 0
        return jsonify({
            "status":       "success",
            "ticker":       ticker,
            "prices":       prices,
            "dates":        dates,
            "latest_close": prices[-1],
            "change":       change,
            "change_pct":   pct,
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/predict", methods=["POST"])
def predict():
    data     = request.get_json(force=True) or {}
    prices   = data.get("prices", [])
    headline = data.get("headline", "").strip()

    if len(prices) != 10:
        return jsonify({"error": f"Need exactly 10 prices, got {len(prices)}."}), 400
    if not headline:
        return jsonify({"error": "Headline cannot be empty."}), 400

    try:
        label, up_pct, down_pct, prob, confidence = run_inference(prices, headline)

        # Sparkline chart as base64 PNG for the frontend
        fig, ax = plt.subplots(figsize=(6, 2), dpi=100)
        fig.patch.set_facecolor("#0d1117")
        ax.set_facecolor("#0d1117")
        line_color = "#00ff88" if label == "Bullish" else "#ff4444"
        ax.plot(prices, color=line_color, linewidth=2.5, solid_capstyle="round")
        ax.fill_between(range(len(prices)), prices,
                        min(prices) * 0.999, alpha=0.15, color=line_color)
        ax.axis("off")
        plt.tight_layout(pad=0)
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight",
                    pad_inches=0, facecolor="#0d1117")
        plt.close(fig)
        import base64
        chart_b64 = base64.b64encode(buf.getvalue()).decode()

        return jsonify({
            "label":       label,
            "up_pct":      up_pct,
            "down_pct":    down_pct,
            "probability": prob,
            "confidence":  confidence,
            "chart":       chart_b64,
            "prices":      prices,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/history")
def history():
    if not os.path.exists(HISTORY_PATH):
        return jsonify({"status": "not_found",
                        "message": "No training history. Run minor.py first."}), 404
    with open(HISTORY_PATH, "rb") as f:
        h = pickle.load(f)
    return jsonify({
        "status":     "ok",
        "epochs":     len(h["train_loss"]),
        "train_loss": h["train_loss"],
        "val_loss":   h["val_loss"],
        "train_acc":  h["train_acc"],
        "val_acc":    h["val_acc"],
    })


@app.route("/api/comparison")
def comparison():
    comp_path = os.path.join(RESULT_DIR, "comparison_results.csv")
    if not os.path.exists(comp_path):
        return jsonify({"status": "not_found"}), 404
    import csv
    results = []
    with open(comp_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append({k: float(v) if k != 'Model' else v for k, v in row.items()})
    return jsonify({"status": "ok", "results": results})


if __name__ == "__main__":
    os.makedirs(MODEL_DIR,  exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)
    print("=" * 55)
    print("  MarketMind  —  Market Prediction API")
    print("=" * 55)
    print(f"  Frontend : http://localhost:5000")
    print(f"  Model    : {'FOUND' if os.path.exists(MODEL_PATH) else 'NOT FOUND — run minor.py first'}")
    print(f"  Device   : {get_device()}")
    print("=" * 55)
    app.run(host="0.0.0.0", port=5000, debug=False)
