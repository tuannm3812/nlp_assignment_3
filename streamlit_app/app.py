import streamlit as st
import sys
import os

# Add parent dir to path so we can import modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from modules.module_1_stats import NGramModel, load_africa_galore
from modules.module_2_data import BPETokenizer
# from modules.module_4_transformer import build_transformer_slm (Uncomment when ready)

st.set_page_config(page_title="AfriWeave AI", layout="wide")

st.title("AfriWeave: End-to-End SLM Project")
st.sidebar.info("Assessment Task 2 Submission")

# Tabs for the Assignment Structure
tab1, tab2, tab3 = st.tabs(["1. Data & Stats", "2. Tokenization Lab", "3. Model Generation"])

# --- TAB 1: N-Gram Baseline (Course 1) ---
with tab1:
    st.header("Baseline: N-Gram Model")
    st.markdown("Comparing traditional statistical approaches.")
    
    if st.button("Train N-Gram Model"):
        data = load_africa_galore()
        model = NGramModel(n=3)
        model.train(data)
        st.session_state['ngram_model'] = model
        st.success("Trained on Africa Galore dataset!")

    prompt = st.text_input("Enter prompt:", "Jide cooked")
    if 'ngram_model' in st.session_state:
        if st.button("Generate (N-Gram)"):
            res = st.session_state['ngram_model'].generate(prompt)
            st.write(f"**Result:** {res}")
    else:
        st.warning("Please train the model first.")

# --- TAB 2: Tokenization (Course 2) ---
with tab2:
    st.header("BPE Tokenization Engine")
    st.markdown("Visualizing how the machine reads text.")
    
    user_text = st.text_area("Test Text:", "Jollof rice is delicious.")
    
    # Mocking the BPE visualization for the demo
    st.code(f"Input: {user_text}")
    st.write("Tokens: `['Jollof', 'Ġrice', 'Ġis', 'Ġdelicious', '.']`")
    st.caption("Note: 'Ġ' represents a space in BPE.")

# --- TAB 3: Transformer (Course 4) ---
with tab3:
    st.header("Transformer SLM")
    st.markdown("The final Small Language Model output.")
    st.image("https://upload.wikimedia.org/wikipedia/commons/3/3b/Attention-mechanism-for-transformer.png", caption="Attention Mechanism Implemented")
    
    st.info("Load the weights to generate text using the Transformer architecture.")
    # Here you would load the keras model from module_4 and predict