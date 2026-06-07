# config.py
# ── All hyperparameters and file paths ───────────────────────────────────────

# ── Data paths ────────────────────────────────────────────────────────────────
DATA_PATH  = 'data/training.1600000.processed.noemoticon.csv'
GLOVE_PATH = 'data/glove.6B.100d.txt'

# ── Output paths ──────────────────────────────────────────────────────────────
TOKENIZER_PATH   = 'outputs/tokenizer.pkl'
BEST_MODEL_PATH  = 'outputs/best_model.keras'
FINAL_MODEL_PATH = 'outputs/sentiment_bilstm_attention.keras'

# ── Preprocessing ─────────────────────────────────────────────────────────────
VOCAB_SIZE    = 50000
MAX_LEN       = 60
EMBEDDING_DIM = 100

# ── Model ─────────────────────────────────────────────────────────────────────
LSTM_UNITS = 128

# ── Training ──────────────────────────────────────────────────────────────────
BATCH_SIZE       = 512
EPOCHS           = 20
LEARNING_RATE    = 2e-4
VALIDATION_SPLIT = 0.1
TEST_SIZE        = 0.2
RANDOM_STATE     = 42

# ── Callbacks ─────────────────────────────────────────────────────────────────
EARLY_STOPPING_PATIENCE = 4
REDUCE_LR_PATIENCE      = 2
REDUCE_LR_FACTOR        = 0.3
MIN_LR                  = 1e-6

# ── Inference ─────────────────────────────────────────────────────────────────
THRESHOLD = 0.5
