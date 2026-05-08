"""Statistical baseline models and dataset loading utilities."""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from functools import lru_cache
from typing import Iterable

import pandas as pd


AFRICA_GALORE_URL = (
    "https://storage.googleapis.com/dm-educational/assets/"
    "ai_foundations/africa_galore.json"
)

SAMPLE_CORPUS = [
    "The village market was bustling with music food and conversation.",
    "Jollof rice simmered beside fresh yams cassava and pepper stew.",
    "Families gathered at sunset to share stories drums and dance.",
]


class NGramModel:
    """A compact word-level N-gram language model.

    The model is intentionally simple: it provides a transparent statistical
    baseline for comparing rule-based generation with neural approaches.
    """

    def __init__(self, n: int = 3, seed: int | None = None):
        if n < 2:
            raise ValueError("n must be at least 2")
        self.n = n
        self.model: dict[str, dict[str, float]] = {}
        self._rng = random.Random(seed)

    def train(self, corpus: Iterable[str]) -> None:
        """Builds the N-gram probability table from a list of text strings."""
        counts = defaultdict(Counter)
        for text in corpus:
            tokens = text.split()
            for i in range(len(tokens) - self.n + 1):
                ngram = tuple(tokens[i : i + self.n])
                context = " ".join(ngram[:-1])
                next_word = ngram[-1]
                counts[context][next_word] += 1

        for context, next_tokens in counts.items():
            total = sum(next_tokens.values())
            self.model[context] = {token: count / total for token, count in next_tokens.items()}

    def generate(self, prompt: str, length: int = 20) -> str:
        """Generates text based on the trained model."""
        words = prompt.split()
        if not words:
            return ""

        output = list(words)

        for _ in range(length):
            context = " ".join(output[-(self.n - 1) :])
            if context in self.model:
                next_word = self._rng.choices(
                    list(self.model[context].keys()),
                    weights=list(self.model[context].values()),
                )[0]
                output.append(next_word)
            else:
                break
        return " ".join(output)


@lru_cache(maxsize=1)
def load_africa_galore() -> list[str]:
    """Load the Africa Galore descriptions used by the demo.

    A small built-in corpus keeps the application usable when the remote dataset
    is temporarily unavailable, which is common in portfolio reviews and demos.
    """
    try:
        return pd.read_json(AFRICA_GALORE_URL)["description"].dropna().astype(str).tolist()
    except Exception:
        return SAMPLE_CORPUS.copy()
