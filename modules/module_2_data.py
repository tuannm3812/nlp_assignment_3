"""Tokenization and embedding visualization helpers."""

from __future__ import annotations

import re
from collections import Counter
from typing import Iterable


class BPETokenizer:
    """A small Byte Pair Encoding tokenizer suitable for demos and tests."""

    def __init__(self, vocab_size: int = 5000):
        if vocab_size < 1:
            raise ValueError("vocab_size must be positive")
        self.vocab_size = vocab_size
        self.merges: list[tuple[str, str]] = []
        self.vocab = set()
        self.EOW = "</w>"

    def preprocess(self, text: str) -> str:
        """Normalize text before tokenization."""
        text = text.lower()
        text = re.sub(r"<.*?>", "", text)
        return re.sub(r"\s+", " ", text).strip()

    def train(self, dataset: Iterable[str]) -> None:
        """Learn merge rules from a corpus."""
        corpus = []
        for text in dataset:
            text = self.preprocess(text)
            for word in text.split():
                corpus.append(list(word) + [self.EOW])
                self.vocab.update(corpus[-1])

        for _ in range(self.vocab_size):
            pairs = self.get_stats(corpus)
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            if pairs[best] < 2:
                break
            self.merges.append(best)
            corpus = self.merge_vocab(best, corpus)
            self.vocab.add("".join(best))

    def get_stats(self, corpus):
        pairs = Counter()
        for word in corpus:
            for i in range(len(word) - 1):
                pairs[word[i], word[i + 1]] += 1
        return pairs

    def merge_vocab(self, pair, corpus):
        merged = "".join(pair)
        updated = []
        for word in corpus:
            i = 0
            new_word = []
            while i < len(word):
                if i < len(word) - 1 and (word[i], word[i + 1]) == pair:
                    new_word.append(merged)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            updated.append(new_word)
        return updated

    def tokenize(self, text: str) -> list[str]:
        """Apply learned merge rules to new text."""
        tokens = []
        for word in self.preprocess(text).split():
            pieces = list(word) + [self.EOW]
            for merge in self.merges:
                pieces = self.merge_vocab(merge, [pieces])[0]
            tokens.extend(piece for piece in pieces if piece != self.EOW)
        return tokens

def get_embedding_visual_data(tokens, embeddings):
    """Reduce embeddings to two dimensions for visualization."""
    from sklearn.manifold import TSNE

    tsne = TSNE(n_components=2)
    reduced = tsne.fit_transform(embeddings)
    return reduced, tokens
