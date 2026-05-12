# Multimodal Stock Market Prediction

ResNet18 (price charts) + FinBERT (news headlines) → MLP → Bullish / Bearish

## Setup

### 1. Install Python 3.10+
Download from https://www.python.org/downloads/

### 2. Install dependencies
```
pip install pandas numpy matplotlib Pillow torch torchvision transformers scikit-learn seaborn yfinance flask flask-cors
```

### 3. Place your dataset
Put `sp500_headlines_2008_2024.csv` inside the `data/` folder:
```
minor.py
data/
  sp500_headlines_2008_2024.csv   ← your Kaggle dataset goes here
charts/    (auto-created)
models/    (auto-created)
results/   (auto-created)
```
Download the dataset from Kaggle: https://www.kaggle.com/datasets

## Running

### Full pipeline (train + evaluate + live predict)
```
python minor.py
```

### Skip training (use existing saved model)
```
python minor.py --skip-train
```

### Live prediction only (model must already be trained)
```
python minor.py --predict-only
```

### Start the Flask REST API
```
python minor.py --flask
```
Then open `index.html` in your browser.

## Output files
| Path | Contents |
|------|----------|
| `models/best_model.pt` | Trained model weights |
| `results/confusion_matrix.png` | Confusion matrix |
| `results/roc_curve.png` | ROC curve |
| `results/training_curves.png` | Loss & accuracy over epochs |
| `results/tsne_clusters.png` | t-SNE embedding plot |
| `results/comparison_results.csv` | All model metrics |
| `results/live_predictions_log.txt` | Live prediction history |

## Notes
- Training takes ~30 min on CPU, ~5 min on GPU (CUDA)
- FinBERT weights (~440 MB) are downloaded automatically on first run
- ResNet18 weights (~45 MB) are downloaded automatically on first run
