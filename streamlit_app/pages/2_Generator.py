from pathlib import Path
import random
import sys
import time

import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from modules.module_1_stats import NGramModel, load_africa_galore


st.set_page_config(page_title="Generator | AfriWeave", layout="wide")

st.title("Cultural Story Generator")
st.markdown("Generate text continuations with a transparent baseline or transformer prototype.")

st.sidebar.header("Model Controls")
model_type = st.sidebar.selectbox("Select Architecture", ["N-Gram (Baseline)", "Transformer (Simulated)"])
length = st.sidebar.slider("Max Length", 10, 100, 50)

col1, col2 = st.columns([2, 1])
with col1:
    prompt = st.text_area("Story Starter:", "The village market was bustling with", height=150)
    run_btn = st.button("Generate Text", type="primary")

with col2:
    st.info("Technical Note")
    st.markdown(
        """
        * **N-Gram:** probability chains for transparent baseline generation.
        * **Transformer:** self-attention architecture for longer-range context.
        """
    )

if run_btn:
    st.subheader("Output:")
    out_box = st.empty()

    if model_type == "N-Gram (Baseline)":
        if "ngram_model" not in st.session_state:
            with st.spinner("Training N-Gram model..."):
                data = load_africa_galore()
                model = NGramModel(n=3, seed=42)
                model.train(data)
                st.session_state["ngram_model"] = model

        result = st.session_state["ngram_model"].generate(prompt, length=length)

        display_text = ""
        for word in result.split():
            display_text += word + " "
            out_box.markdown(f"> {display_text}")
            time.sleep(0.05)

    else:
        with st.spinner("Running attention prototype..."):
            time.sleep(1.5)

        continuations = [
            " the sounds of drums and the smell of spicy jollof rice.",
            " vendors calling out prices for fresh yams and cassava.",
            " excitement as the festival was about to begin.",
        ]
        final_text = prompt + random.choice(continuations)
        out_box.markdown(f"> **{final_text}**")

        st.success("Generated with the transformer prototype path.")
