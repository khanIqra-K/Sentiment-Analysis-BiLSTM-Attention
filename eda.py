# eda.py
# ── Generate and save all EDA graphs ─────────────────────────────────────────

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

from config import DATA_PATH

os.makedirs('graphs', exist_ok=True)

# ═════════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ═════════════════════════════════════════════════════════════════════════════

print("Loading data...")
cols = ['target', 'id', 'date', 'flag', 'user', 'text']
df   = pd.read_csv(DATA_PATH, encoding='latin-1', names=cols)
df['target'] = df['target'].replace(4, 1)
df = df[['text', 'target']].drop_duplicates(subset='text').reset_index(drop=True)
print(f"Loaded {df.shape[0]:,} tweets.")

df['char_count'] = df['text'].apply(len)
df['word_count'] = df['text'].apply(lambda x: len(x.split()))


# ═════════════════════════════════════════════════════════════════════════════
# 2. GRAPH 1 — Label distribution
# ═════════════════════════════════════════════════════════════════════════════

counts = df['target'].value_counts()
plt.figure(figsize=(5, 3))
counts.plot(kind='bar', color=['#E8593C', '#3B8BD4'], edgecolor='white')
plt.xticks([0, 1], ['Negative (0)', 'Positive (1)'], rotation=0)
plt.title('Label distribution')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('graphs/01_label_distribution.png', dpi=150)
plt.close()
print("Saved: graphs/01_label_distribution.png")


# ═════════════════════════════════════════════════════════════════════════════
# 3. GRAPH 2 — Character and word count distributions
# ═════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for label, color, name in [(0, '#E8593C', 'Negative'), (1, '#3B8BD4', 'Positive')]:
    subset = df[df['target'] == label]
    axes[0].hist(subset['char_count'], bins=50, alpha=0.6, color=color, label=name)
    axes[1].hist(subset['word_count'], bins=30, alpha=0.6, color=color, label=name)

axes[0].set_title('Character count distribution')
axes[0].set_xlabel('Characters')
axes[0].legend()
axes[1].set_title('Word count distribution')
axes[1].set_xlabel('Words')
axes[1].legend()
plt.tight_layout()
plt.savefig('graphs/02_length_distributions.png', dpi=150)
plt.close()
print("Saved: graphs/02_length_distributions.png")


# ═════════════════════════════════════════════════════════════════════════════
# 4. GRAPH 3 — Top words before cleaning
# ═════════════════════════════════════════════════════════════════════════════

def get_top_words(texts, n=15):
    return Counter(' '.join(texts).lower().split()).most_common(n)

neg_words = get_top_words(df[df['target'] == 0]['text'])
pos_words = get_top_words(df[df['target'] == 1]['text'])

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
neg_df = pd.DataFrame(neg_words, columns=['word', 'count'])
pos_df = pd.DataFrame(pos_words, columns=['word', 'count'])

axes[0].barh(neg_df['word'][::-1], neg_df['count'][::-1], color='#E8593C')
axes[0].set_title('Top words — Negative (before cleaning)')
axes[1].barh(pos_df['word'][::-1], pos_df['count'][::-1], color='#3B8BD4')
axes[1].set_title('Top words — Positive (before cleaning)')
plt.tight_layout()
plt.savefig('graphs/03_top_words_before_cleaning.png', dpi=150)
plt.close()
print("Saved: graphs/03_top_words_before_cleaning.png")


# ═════════════════════════════════════════════════════════════════════════════
# 5. GRAPH 4 — Top words after cleaning (loads clean_text if available)
# ═════════════════════════════════════════════════════════════════════════════

# Check if a cleaned version was already saved
CLEANED_PATH = 'outputs/cleaned_df.csv'

if os.path.exists(CLEANED_PATH):
    print("Found cleaned data, loading...")
    clean_df = pd.read_csv(CLEANED_PATH)

    neg_clean = get_top_words(clean_df[clean_df['target'] == 0]['clean_text'].dropna())
    pos_clean = get_top_words(clean_df[clean_df['target'] == 1]['clean_text'].dropna())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    neg_df2 = pd.DataFrame(neg_clean, columns=['word', 'count'])
    pos_df2 = pd.DataFrame(pos_clean, columns=['word', 'count'])

    axes[0].barh(neg_df2['word'][::-1], neg_df2['count'][::-1], color='#E8593C')
    axes[0].set_title('Top words — Negative (after cleaning)')
    axes[1].barh(pos_df2['word'][::-1], pos_df2['count'][::-1], color='#3B8BD4')
    axes[1].set_title('Top words — Positive (after cleaning)')
    plt.tight_layout()
    plt.savefig('graphs/04_top_words_after_cleaning.png', dpi=150)
    plt.close()
    print("Saved: graphs/04_top_words_after_cleaning.png")
else:
    print("Skipping graph 4 — cleaned data not found.")
    print("Run train.py first, then re-run eda.py to get graph 4.")


print("\nAll graphs saved to the 'graphs/' folder!")
