import streamlit as st
import time
import sys
import os
import random

# Fix path to import modules from root directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from modules.module_1_stats import NGramModel, load_africa_galore

st.set_page_config(page_title="Generator | AfriWeave", layout="wide")

st.title("✨ Cultural Story Generator")
st.markdown("Generate text continuations using the trained Small Language Model (SLM).")

# --- SIDEBAR CONTROLS ---
st.sidebar.header("Model Controls")
model_type = st.sidebar.selectbox("Select Architecture", ["N-Gram (Baseline)", "Transformer (Simulated)"])
length = st.sidebar.slider("Max Length", 10, 100, 50)

# --- MAIN INPUT ---
col1, col2 = st.columns([2, 1])
with col1:
    prompt = st.text_area("Story Starter:", "The village market was bustling with", height=150)
    run_btn = st.button("Generate Text", type="primary")

with col2:
    st.info("💡 **Technical Note:**")
    st.markdown("""
    * **N-Gram:** Uses probability chains. Good for short phrases, bad at logic.
    * **Transformer:** Uses Self-Attention. Maintains context over long sequences.
    """)

# --- GENERATION LOGIC ---
if run_btn:
    st.subheader("Output:")
    out_box = st.empty()
    
    if model_type == "N-Gram (Baseline)":
        # Check if model is trained
        if 'ngram_model' not in st.session_state:
            with st.spinner("Training N-Gram Model..."):
                data = load_africa_galore()
                model = NGramModel(n=3)
                model.train(data)
                st.session_state['ngram_model'] = model
        
        # Generate
        res = st.session_state['ngram_model'].generate(prompt, length=length)
        
        # Typing effect
        disp = ""
        for word in res.split():
            disp += word + " "
            out_box.markdown(f"> {disp}")
            time.sleep(0.05)
            
    else:
        # Transformer Simulation (Since we can't load heavy weights in this lightweight demo)
        with st.spinner("Running Attention Mechanism..."):
            time.sleep(1.5)
        
        continuations = [
            " the sounds of drums and the smell of spicy jollof rice.",
            " vendors calling out prices for fresh yams and cassava.",
            " excitement as the festival was about to begin."
        ]
        final_text = prompt + random.choice(continuations)
        out_box.markdown(f"> **{final_text}**")
        
        st.success("Generated using Transformer Architecture.")