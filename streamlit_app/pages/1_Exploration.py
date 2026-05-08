from collections import Counter
from pathlib import Path
import sys

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from modules.module_1_stats import load_africa_galore


st.set_page_config(page_title="Exploration | AfriWeave", layout="wide")

st.title("Data Exploration & Insights")
st.markdown("Explore the Africa Galore corpus before training language models.")


@st.cache_data
def get_data():
    return load_africa_galore()


data = get_data()
st.sidebar.success(f"Dataset loaded: {len(data)} paragraphs")

tab1, tab2 = st.tabs(["N-Gram Analysis", "Embedding Space"])

with tab1:
    st.header("N-Gram Frequency")
    st.caption("What are the most common phrases in the dataset?")

    col1, col2 = st.columns([1, 3])
    with col1:
        n_size = st.slider("Select N", 2, 4, 2)
        top_k = st.slider("Number of top results", 5, 20, 10)

    with col2:
        tokens = " ".join(data).split()
        ngrams = zip(*[tokens[i:] for i in range(n_size)])
        counts = Counter(" ".join(ngram) for ngram in ngrams)

        df = pd.DataFrame(counts.most_common(top_k), columns=["Phrase", "Count"])
        fig = px.bar(
            df,
            x="Count",
            y="Phrase",
            orientation="h",
            title=f"Top {top_k} {n_size}-grams",
            color="Count",
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("Semantic Clustering")
    st.markdown("Visualizing how culturally related concepts can cluster in embedding space.")

    if st.button("Run Simulation"):
        import numpy as np

        words = ["Jollof", "Rice", "Spicy", "Lagos", "Accra", "Nairobi", "Market", "Drum", "Dance", "Happy"]
        categories = ["Food", "Food", "Food", "Place", "Place", "Place", "Setting", "Culture", "Culture", "Emotion"]

        data_points = []
        for category in categories:
            if category == "Food":
                base = [5, 5]
            elif category == "Place":
                base = [-5, 5]
            elif category == "Setting":
                base = [-5, -5]
            else:
                base = [5, -5]
            data_points.append(np.array(base) + np.random.normal(0, 1, 2))

        df_tsne = pd.DataFrame(data_points, columns=["x", "y"])
        df_tsne["Word"] = words
        df_tsne["Category"] = categories

        fig_tsne = px.scatter(df_tsne, x="x", y="y", text="Word", color="Category", size_max=60)
        st.plotly_chart(fig_tsne, use_container_width=True)
        st.info("Food items cluster together, distinct from geographical locations.")
