# dataset.py
# Text cleaning  +  data loading / tokenization / GloVe matrix 

import re
import pickle
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from config import (
    DATA_PATH, GLOVE_PATH, TOKENIZER_PATH,
    VOCAB_SIZE, MAX_LEN, EMBEDDING_DIM,
    TEST_SIZE, RANDOM_STATE
)

# NLTK downloads
for pkg in ['stopwords', 'wordnet', 'averaged_perceptron_tagger',
            'averaged_perceptron_tagger_eng', 'punkt', 'omw-1.4']:
    nltk.download(pkg, quiet=True)

tqdm.pandas()
lemmatizer = WordNetLemmatizer()

# Custom stopword set 
STOPWORDS = set(stopwords.words('english'))

# keep negation words 
STOPWORDS -= {
    "not", "no", "nor", "never", "neither",
    "nothing", "nowhere", "hardly", "barely", "scarcely",
    "doesn't", "isn't", "wasn't", "shouldn't", "wouldn't",
    "couldn't", "won't", "can't", "don't"
}
# keep intensity words 
STOPWORDS -= {
    "very", "really", "so", "too", "such",
    "more", "most", "quite", "rather", "pretty",
    "extremely", "absolutely", "completely", "totally",
    "just", "even", "much", "well", "enough",
    "almost", "always", "often", "still"
}
# keep a/an
STOPWORDS -= {"a", "an"}



# 1. TEXT CLEANING

def clean_tweet(text: str) -> str:
    """
    Clean a raw tweet:
      - Lowercase, remove URLs / @mentions, strip # from hashtags
      - Remove non-alphabetic characters
      - Remove custom stopwords
      - POS-aware lemmatization
    Returns a cleaned string (may be empty if no meaningful tokens remain).
    """
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)    # remove URLs
    text = re.sub(r'@\w+', '', text)               # remove @mentions
    text = re.sub(r'#(\w+)', r'\1', text)          # strip # from hashtags
    text = re.sub(r'[^a-z\s]', '', text)           # keep letters only
    text = re.sub(r'\s+', ' ', text).strip()

    tokens = [w for w in text.split() if w not in STOPWORDS]

    if tokens:
        tagged  = nltk.pos_tag(tokens)
        tag_map = {'J': wordnet.ADJ, 'V': wordnet.VERB,
                   'N': wordnet.NOUN, 'R': wordnet.ADV}
        tokens  = [
            lemmatizer.lemmatize(w, tag_map.get(t[0].upper(), wordnet.NOUN))
            for w, t in tagged
        ]

    return ' '.join(tokens)


# 2. DATA LOADING & SPLITTING

def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Load Sentiment140 CSV → binary labels → drop duplicates."""
    cols = ['target', 'id', 'date', 'flag', 'user', 'text']
    df   = pd.read_csv(path, encoding='latin-1', names=cols)
    df['target'] = df['target'].replace(4, 1)
    df = df[['text', 'target']].drop_duplicates(subset='text').reset_index(drop=True)
    print(f"Loaded {df.shape[0]:,} unique tweets.")
    return df


def apply_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """Apply clean_tweet() to every row; drop rows that become empty."""
    print("Cleaning tweets (~8-12 min on full dataset)…")
    df['clean_text'] = df['text'].progress_apply(clean_tweet)
    empty = df['clean_text'].str.strip() == ''
    print(f"Dropping {empty.sum():,} empty tweets after cleaning.")
    return df[~empty].reset_index(drop=True)


def split_data(df: pd.DataFrame):
    X_train, X_test, y_train, y_test = train_test_split(
        df['clean_text'].values, df['target'].values,
        test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=df['target']
    )
    print(f"Train: {len(X_train):,}  |  Test: {len(X_test):,}")
    return X_train, X_test, y_train, y_test


# 3. TOKENIZATION & PADDING

def build_tokenizer(X_train, save_path: str = TOKENIZER_PATH) -> Tokenizer:
    """Fit a Keras Tokenizer on training texts and save it to disk."""
    tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train)
    print(f"Unique words: {len(tokenizer.word_index):,}  |  Kept: {VOCAB_SIZE:,}")
    with open(save_path, 'wb') as f:
        pickle.dump(tokenizer, f)
    print(f"Tokenizer saved → {save_path}")
    return tokenizer


def get_padded_sequences(tokenizer: Tokenizer, X_train, X_test):
    """Convert text arrays to padded integer sequences."""
    def _pad(seqs):
        return pad_sequences(seqs, maxlen=MAX_LEN, padding='post', truncating='post')

    X_train_pad = _pad(tokenizer.texts_to_sequences(X_train))
    X_test_pad  = _pad(tokenizer.texts_to_sequences(X_test))
    print(f"X_train_pad: {X_train_pad.shape}  |  X_test_pad: {X_test_pad.shape}")
    return X_train_pad, X_test_pad



# 4. GLOVE EMBEDDING MATRIX

def build_embedding_matrix(tokenizer: Tokenizer,
                            glove_path: str = GLOVE_PATH) -> np.ndarray:
    """Build a (VOCAB_SIZE × EMBEDDING_DIM) matrix from GloVe vectors."""
    print("Loading GloVe vectors…")
    glove = {}
    with open(glove_path, encoding='utf-8') as f:
        for line in f:
            parts       = line.split()
            glove[parts[0]] = np.array(parts[1:], dtype='float32')
    print(f"Loaded {len(glove):,} GloVe vectors.")

    matrix    = np.zeros((VOCAB_SIZE, EMBEDDING_DIM))
    found = not_found = 0
    for word, idx in tokenizer.word_index.items():
        if idx >= VOCAB_SIZE:
            continue
        vec = glove.get(word)
        if vec is not None:
            matrix[idx] = vec
            found += 1
        else:
            not_found += 1

    print(f"GloVe coverage: {found/(found+not_found)*100:.1f}%  "
          f"({found:,} found  |  {not_found:,} not found)")
    return matrix
