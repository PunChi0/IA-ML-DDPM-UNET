import re
from collections import Counter

import matplotlib.pyplot as plt

def tokenize_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s-]', '', text)
    text = text.replace('-', ' - ')
    return text.split()

def tokenize_dataset(texts):
    word_counts = Counter()
    for text in texts:
        word_counts.update(tokenize_text(text))

    special_tokens = ['<PAD>', '<UNK>', '<BOS>', '<EOS>']

    vocab = special_tokens + [word for word, count in word_counts.most_common()]

    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for word, idx in word2idx.items()}
    
    return vocab, word2idx, idx2word
    
def encode(text, max_len,word2idx):
    tokens = tokenize_text(text)
    ids = [word2idx['<BOS>']]
    ids += [word2idx.get(t, word2idx['<UNK>']) for t in tokens]
    ids += [word2idx['<EOS>']]
    # Pad or truncate
    ids = ids[:max_len]
    ids += [word2idx['<PAD>']] * (max_len - len(ids))
    return ids

def decode(ids,word2idx,idx2word):
    words = [idx2word[i] for i in ids 
             if i not in [word2idx['<PAD>'], word2idx['<BOS>'], word2idx['<EOS>']]]
    return ' '.join(words)

def plot_token_info(texts):
    token_lengths = [len(tokenize_text(text)) for text in texts]
    avg_len = sum(token_lengths) / len(token_lengths)

    # Histogram
    plt.figure(figsize=(10, 4))
    plt.hist(token_lengths, bins=20, color='steelblue', edgecolor='white')
    plt.axvline(avg_len, color='red', linestyle='--', label=f'Average ({avg_len:.1f})')
    plt.axvline(max(token_lengths), color='orange', linestyle='--', label=f'Max ({max(token_lengths)})')
    plt.xlabel("Number of tokens")
    plt.ylabel("Number of samples")
    plt.title("PixelDiffusion — Token length distribution")
    plt.legend()
    plt.tight_layout()
    plt.show()