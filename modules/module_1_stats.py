import random
from collections import Counter, defaultdict
import pandas as pd

class NGramModel:
    def __init__(self, n=3):
        self.n = n
        self.model = {}

    def train(self, corpus: list[str]):
        """Builds the N-gram probability table from a list of text strings."""
        print(f"Training {self.n}-gram model...")
        counts = defaultdict(Counter)
        for text in corpus:
            tokens = text.split(" ") # Simple space tokenization for baseline
            for i in range(len(tokens) - self.n + 1):
                ngram = tuple(tokens[i:i+self.n])
                context = " ".join(ngram[:-1])
                next_word = ngram[-1]
                counts[context][next_word] += 1
        
        # Convert counts to probabilities
        for context, next_tokens in counts.items():
            total = sum(next_tokens.values())
            self.model[context] = {t: c/total for t, c in next_tokens.items()}
        print("Training complete.")

    def generate(self, prompt: str, length=20):
        """Generates text based on the trained model."""
        words = prompt.split(" ")
        output = list(words)
        
        for _ in range(length):
            context = " ".join(output[-(self.n-1):])
            if context in self.model:
                next_word = random.choices(
                    list(self.model[context].keys()), 
                    weights=self.model[context].values()
                )[0]
                output.append(next_word)
            else:
                break # Stop if context is unknown
        return " ".join(output)

# Helper to load the specific dataset used in labs
def load_africa_galore():
    url = "https://storage.googleapis.com/dm-educational/assets/ai_foundations/africa_galore.json"
    return pd.read_json(url)["description"].tolist()