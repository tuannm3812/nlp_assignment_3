import pytest

pytest.importorskip("keras")

from modules.module_4_transformer import build_transformer_slm


def test_transformer_model_builds_with_expected_output_shape():
    model = build_transformer_slm(vocab_size=32, max_len=8)

    assert model.output_shape == (None, 8, 32)
