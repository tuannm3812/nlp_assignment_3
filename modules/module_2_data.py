import re
import numpy as np
from collections import Counter

class BPETokenizer:
    def __init__(self, vocab_size=5000):
        self.vocab_size = vocab_size
        self.merges = []
        self.vocab = set()
        self.EOW = "</w>"

    def preprocess(self, text):
        """Standard cleaning from Lab 2.1"""
        text = text.lower()
        text = re.sub(r'<.*?>', '', text) # Remove HTML
        return text

    def train(self, dataset):
        """Implements the BPE learning loop from Lab 2.4"""
        # 1. Initialize character vocab
        corpus = []
        for text in dataset:
            text = self.preprocess(text)
            for word in text.split():
                corpus.append(list(word) + [self.EOW])
                self.vocab.update(corpus[-1])
        
        # 2. Iteratively merge pairs
        print("Training BPE Tokenizer...")
        for i in range(self.vocab_size):
            pairs = self.get_stats(corpus)
            if not pairs: break
            best = max(pairs, key=pairs.get)
            self.merges.append(best)
            corpus = self.merge_vocab(best, corpus)
            if i % 100 == 0: print(f"Merge {i}: {best}")
            
    def get_stats(self, corpus):
        pairs = Counter()
        for word in corpus:
            for i in range(len(word)-1):
                pairs[word[i], word[i+1]] += 1
        return pairs

    def merge_vocab(self, pair, corpus):
        bigram = re.escape(' '.join(pair))
        p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
        return [[c for c in word if c != ""] for word in corpus] # Simplified for brevity

    def tokenize(self, text):
        # Applies learned merges to new text
        # (Implementation of inference logic goes here)
        return [text] # Placeholder for full inference logic

def get_embedding_visual_data(tokens, embeddings):
    """Helper for t-SNE visualization (Lab 2.5)"""
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2)
    reduced = tsne.fit_transform(embeddings)
    return reduced, tokens