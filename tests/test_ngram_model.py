from modules.module_1_stats import NGramModel, SAMPLE_CORPUS, load_africa_galore


def test_ngram_model_generates_from_known_context():
    model = NGramModel(n=2, seed=1)
    model.train(["hello world", "hello there"])

    generated = model.generate("hello", length=1)

    assert generated in {"hello world", "hello there"}


def test_empty_prompt_returns_empty_string():
    model = NGramModel()

    assert model.generate("") == ""


def test_dataset_loader_returns_text_rows():
    data = load_africa_galore()

    assert data
    assert all(isinstance(row, str) for row in data)
    assert len(SAMPLE_CORPUS) >= 1
