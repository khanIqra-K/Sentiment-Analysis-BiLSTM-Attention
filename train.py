# train.py
# Training  +  Evaluation  +  Plots 

import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
)
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from config import (
    BATCH_SIZE, EPOCHS, LEARNING_RATE, VALIDATION_SPLIT,
    BEST_MODEL_PATH, FINAL_MODEL_PATH, THRESHOLD
)
from dataset import (
    load_data, apply_cleaning, split_data,
    build_tokenizer, get_padded_sequences, build_embedding_matrix
)
from model import build_model

os.makedirs('outputs', exist_ok=True)



# PLOT HELPERS

def plot_label_distribution(df: pd.DataFrame):
    counts = df['target'].value_counts()
    plt.figure(figsize=(5, 3))
    counts.plot(kind='bar', color=['#E8593C', '#3B8BD4'], edgecolor='white')
    plt.xticks([0, 1], ['Negative (0)', 'Positive (1)'], rotation=0)
    plt.title('Label distribution')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.show()


def plot_top_words(df: pd.DataFrame, col: str, title_suffix: str = ''):
    def top_words(texts, n=15):
        return Counter(' '.join(texts).lower().split()).most_common(n)

    neg = pd.DataFrame(top_words(df[df['target'] == 0][col]), columns=['word', 'count'])
    pos = pd.DataFrame(top_words(df[df['target'] == 1][col]), columns=['word', 'count'])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].barh(neg['word'][::-1], neg['count'][::-1], color='#E8593C')
    axes[0].set_title(f'Top words — Negative {title_suffix}')
    axes[1].barh(pos['word'][::-1], pos['count'][::-1], color='#3B8BD4')
    axes[1].set_title(f'Top words — Positive {title_suffix}')
    plt.tight_layout()
    plt.show()


def plot_training_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history['accuracy'],     label='Train')
    axes[0].plot(history.history['val_accuracy'], label='Validation')
    axes[0].set_title('Accuracy over epochs')
    axes[0].set_xlabel('Epoch')
    axes[0].legend()

    axes[1].plot(history.history['loss'],     label='Train')
    axes[1].plot(history.history['val_loss'], label='Validation')
    axes[1].set_title('Loss over epochs')
    axes[1].set_xlabel('Epoch')
    axes[1].legend()
    plt.tight_layout()
    plt.show()



# EVALUATION HELPERS

def evaluate_model(model, X_test_pad, y_test):
    """Accuracy, confusion matrix, classification report."""
    y_pred_prob = model.predict(X_test_pad, batch_size=BATCH_SIZE, verbose=1)
    y_pred      = (y_pred_prob >= THRESHOLD).astype(int).flatten()

    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest accuracy: {acc:.4f}  ({acc * 100:.2f}%)\n")

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Predicted Negative', 'Predicted Positive'],
                yticklabels=['Actual Negative',    'Actual Positive'])
    plt.title('Confusion matrix')
    plt.tight_layout()
    plt.show()

    tn, fp, fn, tp = cm.ravel()
    print(f"True Negatives  : {tn:,}\nFalse Positives : {fp:,}")
    print(f"False Negatives : {fn:,}\nTrue Positives  : {tp:,}")
    print("\n", classification_report(y_test, y_pred,
                                      target_names=['Negative', 'Positive']))
    return y_pred


def show_errors(X_test, y_test, y_pred, n: int = 10):
    """Print n random tweets the model predicted incorrectly."""
    errors = np.where(y_pred != y_test)[0]
    print(f"Total errors: {len(errors):,} / {len(y_test):,}  "
          f"({len(errors)/len(y_test)*100:.2f}%)\n")
    for idx in np.random.choice(errors, size=min(n, len(errors)), replace=False):
        print(f"Tweet     : {X_test[idx]}")
        print(f"True      : {'Positive' if y_test[idx] == 1 else 'Negative'}")
        print(f"Predicted : {'Positive' if y_pred[idx] == 1 else 'Negative'}\n")



# MAIN TRAINING PIPELINE

def train():
    print("=" * 60)
    print("GPU:", tf.config.list_physical_devices('GPU'))
    print("=" * 60)

    # 1. Load & clean
    df = load_data()
    plot_label_distribution(df)
    df = apply_cleaning(df)
    plot_top_words(df, col='clean_text', title_suffix='(after cleaning)')

    # 2. Split
    X_train, X_test, y_train, y_test = split_data(df)

    # 3. Tokenize & pad
    tokenizer              = build_tokenizer(X_train)
    X_train_pad, X_test_pad = get_padded_sequences(tokenizer, X_train, X_test)

    # 4. GloVe matrix
    embedding_matrix = build_embedding_matrix(tokenizer)

    # 5. Build & compile model
    model = build_model(embedding_matrix)
    model.summary()
    model.compile(
        loss      = 'binary_crossentropy',
        optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        metrics   = ['accuracy']
    )

    # 6. Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=4,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.3,
                          patience=2, min_lr=1e-6, verbose=1),
        ModelCheckpoint(filepath=BEST_MODEL_PATH, monitor='val_accuracy',
                        save_best_only=True, verbose=1)
    ]

    # 7. Train
    history = model.fit(
        X_train_pad, y_train,
        epochs           = EPOCHS,
        batch_size       = BATCH_SIZE,
        validation_split = VALIDATION_SPLIT,
        callbacks        = callbacks,
        verbose          = 1
    )

    # 8. Save
    model.save(FINAL_MODEL_PATH)
    print(f"Model saved → {FINAL_MODEL_PATH}")

    # 9. Plots & evaluation
    plot_training_history(history)
    y_pred = evaluate_model(model, X_test_pad, y_test)
    show_errors(X_test, y_test, y_pred)

    return model, tokenizer, X_test_pad, y_test, X_test


if __name__ == '__main__':
    train()
