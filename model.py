# model.py
# ── BiLSTM + Attention architecture ─────────────────────────────────────────

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Embedding, Bidirectional, LSTM,
    Dense, Dropout, SpatialDropout1D,
    Permute, Flatten, RepeatVector,
    Multiply, Lambda, Activation
)
from config import VOCAB_SIZE, EMBEDDING_DIM, MAX_LEN, LSTM_UNITS


def build_model(embedding_matrix: np.ndarray) -> Model:
    """
    BiLSTM + custom attention model for binary sentiment classification.

    Architecture:
        Input (MAX_LEN,)
          → GloVe Embedding (fine-tuned)
          → SpatialDropout1D(0.2)
          → Bidirectional LSTM(128) — returns all hidden states
          → Attention  (Dense tanh → Flatten → Softmax → weighted sum)
          → Dense(128, relu) → Dropout(0.4)
          → Dense(64,  relu) → Dropout(0.3)
          → Dense(1, sigmoid)
    """
    # ── Input ─────────────────────────────────────────────────────────────────
    inp = Input(shape=(MAX_LEN,))

    # ── Embedding ─────────────────────────────────────────────────────────────
    x = Embedding(
            input_dim    = VOCAB_SIZE,
            output_dim   = EMBEDDING_DIM,
            input_length = MAX_LEN,
            weights      = [embedding_matrix],
            trainable    = True            # fine-tune GloVe on tweets
        )(inp)
    x = SpatialDropout1D(0.2)(x)

    # ── Bidirectional LSTM ────────────────────────────────────────────────────
    # output: (batch, MAX_LEN, LSTM_UNITS*2)
    lstm_out = Bidirectional(
                   LSTM(LSTM_UNITS, return_sequences=True, dropout=0.3)
               )(x)

    # ── Attention ─────────────────────────────────────────────────────────────
    # Score each time step, then softmax → importance weights
    a = Dense(1, activation='tanh')(lstm_out)   # (batch, MAX_LEN, 1)
    a = Flatten()(a)                             # (batch, MAX_LEN)
    a = Activation('softmax')(a)                 # weights sum to 1

    # Expand to (batch, MAX_LEN, LSTM_UNITS*2) and multiply with LSTM output
    a = RepeatVector(LSTM_UNITS * 2)(a)          # (batch, LSTM_UNITS*2, MAX_LEN)
    a = Permute([2, 1])(a)                       # (batch, MAX_LEN, LSTM_UNITS*2)

    weighted       = Multiply()([lstm_out, a])
    context_vector = Lambda(lambda z: tf.reduce_sum(z, axis=1))(weighted)
    # context_vector shape: (batch, LSTM_UNITS*2)

    # ── Classification head ───────────────────────────────────────────────────
    x   = Dense(128, activation='relu')(context_vector)
    x   = Dropout(0.4)(x)
    x   = Dense(64,  activation='relu')(x)
    x   = Dropout(0.3)(x)
    out = Dense(1,   activation='sigmoid')(x)

    return Model(inp, out)
