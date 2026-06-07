# predict.py
# ── Inference  +  Attention visualization ────────────────────────────────────

import pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from dataset import clean_tweet
from config import MAX_LEN, THRESHOLD, TOKENIZER_PATH, FINAL_MODEL_PATH


def load_model_and_tokenizer(model_path=FINAL_MODEL_PATH,
                              tokenizer_path=TOKENIZER_PATH):
    model = tf.keras.models.load_model(model_path)
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)
    print(f"Model    loaded ← {model_path}")
    print(f"Tokenizer loaded ← {tokenizer_path}")
    return model, tokenizer


def predict_sentiment(text: str, model, tokenizer) -> dict:
    """Predict sentiment for a single tweet and print the result."""
    cleaned = clean_tweet(text)
    seq     = tokenizer.texts_to_sequences([cleaned])
    pad     = pad_sequences(seq, maxlen=MAX_LEN, padding='post', truncating='post')
    prob    = model.predict(pad, verbose=0)[0][0]
    label   = 'Positive' if prob >= THRESHOLD else 'Negative'

    print(f"Tweet   : {text}")
    print(f"Cleaned : {cleaned}")
    print(f"Score   : {prob:.4f}  →  {label}\n")
    return {'text': text, 'cleaned': cleaned, 'score': float(prob), 'label': label}


def visualize_attention(text: str, model, tokenizer):
    """
    Bar chart of attention weights per token.
    Shows which words the model focused on for the prediction.
    """
    # Sub-model that exposes the attention softmax layer (index 6)
    attn_model = Model(inputs=model.input,
                       outputs=model.get_layer(index=6).output)

    cleaned = clean_tweet(text)
    tokens  = cleaned.split()
    seq     = tokenizer.texts_to_sequences([cleaned])
    pad     = pad_sequences(seq, maxlen=MAX_LEN, padding='post', truncating='post')

    prob    = model.predict(pad, verbose=0)[0][0]
    label   = 'Positive' if prob >= THRESHOLD else 'Negative'

    weights      = attn_model.predict(pad, verbose=0)[0]       # (MAX_LEN,)
    n            = min(len(tokens), MAX_LEN)
    w            = weights[:n] / weights[:n].sum()              # renormalize

    fig, ax = plt.subplots(figsize=(max(8, n * 0.8), 2))
    ax.bar(range(n), w, color=plt.cm.RdYlGn(w / w.max()), edgecolor='white')
    ax.set_xticks(range(n))
    ax.set_xticklabels(tokens[:n], rotation=45, ha='right', fontsize=11)
    ax.set_title(f'"{text}"\nPrediction: {label} ({prob:.3f})', fontsize=11)
    ax.set_ylabel('Attention weight')
    plt.tight_layout()
    plt.show()
    print(f"Top attended word: '{tokens[w.argmax()]}'  (weight: {w.max():.3f})\n")


# ── Quick demo ────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    model, tokenizer = load_model_and_tokenizer()

    tweets = [
        "I absolutely love this new phone, it's amazing!",
        "This movie was a complete waste of time, terrible acting.",
        "Just got back from the gym, feeling great!",
        "I can't believe how bad the service was today.",
        "Not bad at all, actually quite enjoyed it.",
        "I'm so tired of all this nonsense."
    ]

    print("\n── Predictions ──\n")
    for t in tweets:
        predict_sentiment(t, model, tokenizer)

    print("\n── Attention maps ──\n")
    for t in tweets[:4]:
        visualize_attention(t, model, tokenizer)
