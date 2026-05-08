from modules.module_2_data import BPETokenizer


def test_bpe_tokenizer_learns_and_applies_merges():
    tokenizer = BPETokenizer(vocab_size=20)
    tokenizer.train(["jollof jollof rice", "jollof rice"])

    tokens = tokenizer.tokenize("jollof rice")

    assert tokens
    assert tokens != ["jollof rice"]
    assert len(tokenizer.merges) > 0


def test_preprocess_removes_html_and_normalizes_spacing():
    tokenizer = BPETokenizer()

    assert tokenizer.preprocess("<b>Hello</b>   World") == "hello world"
