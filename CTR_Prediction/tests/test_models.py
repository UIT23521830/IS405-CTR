import pytest
import torch

from src.models.base import build_model


@pytest.mark.parametrize("name", ["lr", "fm", "deepfm", "autoint", "nam", "nafi"])
def test_model_forward(name: str) -> None:
    config = {
        "model": {"embedding_dim": 4, "hidden_units": [8], "dropout": 0.0},
        "nam": {"hidden_units": [4], "dropout": 0.0},
        "fin": {"num_heads": 2, "num_layers": 1, "attention_dropout": 0.0, "use_residual": True},
    }
    model = build_model(name, [10, 20, 30], config)
    x = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)
    out = model(x)
    assert out["logits"].shape == (2,)

