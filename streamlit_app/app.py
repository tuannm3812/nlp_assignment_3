from pathlib import Path
import sys

import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from modules.module_1_stats import NGramModel, load_africa_galore
from modules.module_2_data import BPETokenizer


st.set_page_config(page_title="AfriWeave AI", layout="wide")

st.title("AfriWeave: Cultural Text Generation Studio")
st.sidebar.info("Interactive NLP prototype")

tab1, tab2, tab3 = st.tabs(["Data & Statistics", "Tokenization", "Model Generation"])

with tab1:
    st.header("Baseline N-Gram Model")
    st.markdown("Train a transparent statistical model and inspect how it extends short prompts.")

    if st.button("Train N-Gram Model"):
        data = load_africa_galore()
        model = NGramModel(n=3, seed=42)
        model.train(data)
        st.session_state["ngram_model"] = model
        st.success("N-Gram model trained.")

    prompt = st.text_input("Enter prompt:", "Jide cooked")
    if "ngram_model" in st.session_state:
        if st.button("Generate (N-Gram)"):
            result = st.session_state["ngram_model"].generate(prompt)
            st.write(f"**Result:** {result}")
    else:
        st.warning("Train the model to enable generation.")

with tab2:
    st.header("BPE Tokenization Engine")
    st.markdown("Learn merge rules from the demo corpus and inspect token pieces.")

    user_text = st.text_area("Test Text:", "Jollof rice is delicious.")
    if st.button("Train Tokenizer"):
        tokenizer = BPETokenizer(vocab_size=200)
        tokenizer.train(load_africa_galore())
        st.session_state["tokenizer"] = tokenizer
        st.success(f"Tokenizer trained with {len(tokenizer.merges)} merges.")

    st.code(f"Input: {user_text}")
    if "tokenizer" in st.session_state:
        st.write("Tokens:", st.session_state["tokenizer"].tokenize(user_text))
    else:
        st.caption("Train the tokenizer to view learned BPE pieces.")

with tab3:
    st.header("Transformer SLM")
    st.markdown("Prototype architecture for a compact self-attention language model.")
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/3/3b/Attention-mechanism-for-transformer.png",
        caption="Scaled dot-product attention",
    )

    st.info("The repository includes the architecture code. Add trained weights to enable neural generation.")
