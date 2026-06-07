# Sentiment Analysis — BiLSTM + Attention

Twitter sentiment classifier trained on [Sentiment140](http://help.sentiment140.com/for-students) (1.6M tweets).  
Model: Bidirectional LSTM with a custom attention mechanism and pre-trained GloVe embeddings.

---

## Project structure

```
├── config.py          # All hyperparameters and file paths
├── dataset.py         # Text cleaning + data loading + tokenization + GloVe matrix
├── model.py           # BiLSTM + Attention architecture
├── train.py           # Training pipeline + evaluation + plots
├── predict.py         # Single-tweet inference + attention visualization
├── requirements.txt
└── README.md
```

---

## Setup

```bash
pip install -r requirements.txt
```

Download the required data files and place them as shown:

```
data/
├── training.1600000.processed.noemoticon.csv   # Sentiment140
└── glove.6B.100d.txt                           # GloVe 6B 100-dim
```

- **Sentiment140**: http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip  
- **GloVe**: https://nlp.stanford.edu/projects/glove/ → glove.6B.zip

---

## Usage

**Train**
```bash
python train.py
```
Saves to `outputs/`: `tokenizer.pkl`, `best_model.keras`, `sentiment_bilstm_attention.keras`

**Predict**
```bash
python predict.py
```
Or in code:
```python
from predict import load_model_and_tokenizer, predict_sentiment, visualize_attention

model, tokenizer = load_model_and_tokenizer()
predict_sentiment("I absolutely love this!", model, tokenizer)
visualize_attention("I can't believe how bad this was", model, tokenizer)
```

---

## Model architecture

```
Input (60 tokens)
  └─ GloVe Embedding 100d (fine-tuned)
       └─ SpatialDropout1D(0.2)
            └─ Bidirectional LSTM(128 × 2 = 256)
                 └─ Attention (score → softmax → weighted sum)
                      └─ Dense(128) → Dropout(0.4)
                           └─ Dense(64) → Dropout(0.3)
                                └─ Dense(1, sigmoid)
```
