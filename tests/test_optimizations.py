import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from compare import collect_rows, format_table
from titanic.features import add_group_features, engineer, load_raw
from titanic.text import PROMPT_STYLES, describe, describe_structured

FIXTURES = Path(__file__).parent / "fixtures"


def engineered():
    return engineer(*load_raw(FIXTURES))


def test_structured_first_row_and_no_leakage():
    train, _ = engineered()
    text = describe_structured(train.iloc[0])
    assert text == (
        "Sex: male | Age: 22 | Class: 3 | Title: Mr | Fare per person: 7.25 | Embarked: Southampton | "
        "Family size: 2 | Deck: unknown | Ticket group size: 1"
    )
    for _, row in train.iterrows():
        lowered = describe_structured(row).lower()
        assert "survived" not in lowered.replace("group outcomes known", "") or "survived:" in lowered
        assert "passengerid" not in lowered and row["Name"].split(",")[0].lower() not in lowered
    flipped = train.iloc[3].copy()
    flipped["Survived"] = 1 - flipped["Survived"]
    assert describe_structured(train.iloc[3]) == describe_structured(flipped)


def test_structured_age_estimated_and_group_fields():
    train, test = engineered()
    assert "Age: " in describe_structured(train.iloc[5]) and "(estimated)" in describe_structured(train.iloc[5])
    fit, (val,) = add_group_features(train.iloc[:18], [train.iloc[18:]])
    texts = [describe_structured(r) for _, r in fit.iterrows()]
    assert all(t.count(" | ") >= 8 for t in texts)
    row = train.iloc[0].copy()
    for key, value in {"GroupSize": 4, "GroupKnownCount": 3, "GroupKnownSurvived": 2}.items():
        row[key] = value
    assert describe_structured(row).endswith("Group outcomes known: 3, survived: 2")
    row["GroupKnownCount"], row["GroupKnownSurvived"] = 0, 0
    assert describe_structured(row).endswith("Group outcomes known: 0")
    row["GroupSize"] = 1
    assert "Group outcomes" not in describe_structured(row)


def test_prompt_styles_registry():
    assert PROMPT_STYLES == {"prose": describe, "structured": describe_structured}


def test_assemble_embeddings_oof_order_and_test_mean():
    modal_app = pytest.importorskip("modal_app")
    r0 = {"val_ids": [3, 1], "val_emb": np.array([[3, 3], [1, 1]], dtype=np.float16),
          "test_ids": [10, 11], "test_emb": np.array([[0, 0], [2, 2]], dtype=np.float16)}
    r1 = {"val_ids": [2], "val_emb": np.array([[2, 2]], dtype=np.float16),
          "test_ids": [10, 11], "test_emb": np.array([[2, 2], [4, 4]], dtype=np.float16)}
    oof, test, ids = modal_app.assemble_embeddings([r0, r1], [1, 2, 3], [10, 11])
    assert oof[:, 0].tolist() == [1, 2, 3] and test[:, 0].tolist() == [1, 3]
    assert ids == {"train_ids": [1, 2, 3], "test_ids": [10, 11]}
    with pytest.raises(ValueError):
        modal_app.assemble_embeddings([r0], [1, 2, 3], [10, 11])  # row 2 never held out


def test_build_metrics_and_cfg_record_new_options():
    modal_app = pytest.importorskip("modal_app")
    cfg = modal_app.make_cfg(model_id="m", run_name="r", epochs=3, lr=1e-4, max_steps=-1)
    assert (cfg["split_seed"], cfg["head"], cfg["prompt_style"], cfg["save_embeddings"]) == (42, "lm", "prose", True)
    oof = pd.DataFrame({"PassengerId": [1, 2], "prob": [0.1, 0.9]})
    res = [{"fold": 0, "val_accuracy": 1.0}]
    m = modal_app.build_metrics(
        run_name="r", model_id="m", gpu="g", folds=1, epochs=3, lr=1e-4, lora_r=8, lora_alpha=16, seed=7,
        smoke=False, oof=oof, y_true=[0, 1], results=res, n_test=2, elapsed_seconds=1.0,
        split_seed=42, head="classification", prompt_style="structured", save_embeddings=True, embedding_dim=4,
    )
    assert (m["split_seed"], m["seed"], m["head"], m["prompt_style"], m["embedding_dim"]) == (
        42, 7, "classification", "structured", 4)


def test_compare_shows_new_columns_only_when_present(tmp_path):
    for name, extra in {"old": {}, "new": {"split_seed": 42, "seed": 7, "head": "lm", "prompt_style": "prose"}}.items():
        d = tmp_path / name
        d.mkdir()
        (d / "metrics.json").write_text(json.dumps(
            {"run_name": name, "oof_accuracy": 0.8, "fold_accuracies": [0.8], "epochs": 3, **extra}))
    both = format_table(collect_rows(tmp_path))
    assert all(c in both.splitlines()[0] for c in ("split_seed", "seed", "head", "prompt_style"))
    only_old = format_table([r for r in collect_rows(tmp_path) if r["run"] == "old"])
    assert "split_seed" not in only_old


def test_last_token_classifier_on_tiny_backbone(tmp_path):
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace as NS

    from titanic.llm import HEAD_NAME, last_token_index, make_last_token_classifier

    class Tiny(torch.nn.Module):
        base_model_prefix = ""

        def __init__(self):
            super().__init__()
            self.emb = torch.nn.Embedding(20, 6)
            self.config = NS(get_text_config=lambda: NS(hidden_size=6))

        def forward(self, input_ids=None, attention_mask=None, output_hidden_states=False):
            h = self.emb(input_ids)
            return NS(hidden_states=(h,), logits=h)

    model = make_last_token_classifier(Tiny())
    ids = torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]])
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]])
    assert last_token_index(mask).tolist() == [2, 1]
    out = model(input_ids=ids, attention_mask=mask, labels=torch.tensor([0, 1]))
    assert out["logits"].shape == (2, 2) and out["pooled"].shape == (2, 6) and out["loss"].ndim == 0
    assert torch.allclose(out["pooled"][1], model.backbone.emb(ids[1, 1]))
    assert getattr(model, HEAD_NAME).weight.dtype == torch.float32
