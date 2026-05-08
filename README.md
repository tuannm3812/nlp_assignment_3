![AfriWeave Banner](https://jenmansafaris.com/wp-content/uploads/2014/08/african-culture-banner.jpg)

# AfriWeave

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-ff4b4b)
![Keras](https://img.shields.io/badge/Keras-JAX-d00000)
![Tests](https://img.shields.io/badge/Tests-Pytest-0a7f42)
![Status](https://img.shields.io/badge/Status-Prototype-orange)

AfriWeave is an interactive NLP prototype for exploring culturally focused text generation. It combines corpus exploration, a transparent N-gram baseline, a small BPE tokenizer, and a transformer architecture scaffold inside a Streamlit application.

The project is intentionally lightweight: it is suitable for demos, portfolio review, and experimentation without requiring large model weights.

## Features

- Corpus exploration for the Africa Galore dataset
- N-gram phrase frequency visualization
- Deterministic word-level N-gram text generation baseline
- Trainable Byte Pair Encoding tokenizer
- Keras/JAX transformer architecture components
- Streamlit interface for exploration and generation

## Project Structure

```text
.
|-- modules/
|   |-- module_1_stats.py         # Dataset loading and N-gram model
|   |-- module_2_data.py          # BPE tokenizer and embedding utilities
|   |-- module_3_nn.py            # Feed-forward neural model builder
|   `-- module_4_transformer.py   # Multi-head attention and transformer scaffold
|-- streamlit_app/
|   |-- app.py                    # Main Streamlit dashboard
|   `-- pages/
|       |-- 1_Exploration.py      # Corpus and phrase analysis
|       `-- 2_Generator.py        # Text generation interface
|-- tests/                        # Unit and smoke tests
|-- pyproject.toml                # Project metadata and tooling config
|-- requirements.txt              # Runtime dependencies
`-- README.md
```

## Getting Started

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run streamlit_app/app.py
```

The app will be available at http://localhost:8501.

## Development

Install test tooling:

```bash
pip install -e ".[dev]"
```

Run tests:

```bash
pytest
```

## Notes

The Africa Galore dataset is loaded from its public remote source. If the remote data is unavailable, AfriWeave falls back to a small built-in sample corpus so the app remains usable during demos.

The transformer path currently provides architecture code and a simulated UI generation path. To turn it into a full neural generator, add training scripts, checkpoint loading, and decoding logic.
