![AfriWeave Banner](https://jenmansafaris.com/wp-content/uploads/2014/08/african-culture-banner.jpg)

# AfriWeave: A Culturally-Adaptive Small Language Model (SLM)

> **Assessment Task 2: End-to-End NLP Project submission.**

[![](https://img.shields.io/badge/Python-3.9+-blue.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org)
[![](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io)
[![](https://img.shields.io/badge/JAX-000000?style=for-the-badge&logo=google&logoColor=white)](https://github.com/google/jax)
[![](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)](https://keras.io)
[![](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com)

## 📄 Project Overview

**AfriWeave** is an end-to-end Natural Language Processing application designed to address the underrepresentation of African cultural narratives in generic Large Language Models.

This project demonstrates the complete lifecycle of building a specialized **Small Language Model (SLM)** from scratch. By moving beyond simple statistical approaches (N-grams) and implementing a custom **Transformer** architecture with specialized **BPE Tokenization**, AfriWeave generates culturally coherent text based on the *Africa Galore* dataset.

This repository combines multiple lab exercises into a unified, deployable web application.

---

## 🏗️ Repository Structure

The project is separated into backend logic (`modules/`) and a frontend web interface (`streamlit_app/`).

nlp_assignment_2/ ├── modules/ # --- The Backend Logic --- │ ├── module_1_stats.py # Course 1: N-Gram probability models │ ├── module_2_data.py # Course 2: BPE Tokenization & Embedding analysis │ ├── module_3_nn.py # Course 3: MLP Network Design │ └── module_4_transformer.py # Course 4: Attention Mechanism & Transformer architecture │ ├── streamlit_app/ # --- The Frontend Interface --- │ ├── app.py # Main dashboard landing page │ └── pages/ │ ├── 1_Exploration.py # Interactive EDA (N-grams & t-SNE visualization) │ └── 2_Generator.py # Text generation interface comparing models │ ├── requirements.txt # Project dependencies └── README.md # Project documentation


## 🚀 Getting Started

### 1. Installation

Clone the repository and install the required dependencies.

```bash
git clone <YOUR_REPO_URL>
cd nlp_assignment_2
pip install -r requirements.txt
2. Running the Application
Launch the Streamlit web interface from the project root directory.

Bash

streamlit run streamlit_app/app.py
The application will open in your default web browser at http://localhost:8501.

✨ Key Features (Assessment Criteria)
This project fulfills the AT2 requirements by integrating the following technical components:

Data Pipeline & EDA: Automated cleaning of the Africa Galore dataset and interactive visualization of N-gram frequencies and semantic word embeddings (t-SNE).

Custom Tokenization: Implementation of a Byte Pair Encoding (BPE) tokenizer to handle cultural terminology efficiently.

Model Comparison: A side-by-side comparison in the UI between a baseline statistical N-Gram model and a modern neural Transformer architecture.

Deep Learning Implementation: Utilization of JAX and Keras to build multi-head attention mechanisms and transformer blocks from first principles.

