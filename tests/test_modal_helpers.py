from datetime import datetime

import pandas as pd
import pytest

pytest.importorskip("modal")
import modal_app  # noqa: E402


def result(fold, val_ids, val_probs, test_probs, test_ids=(10, 11)):
    return {
        "fold": fold,
        "val_ids": list(val_ids),
        "val_probs": list(val_probs),
        "test_ids": list(test_ids),
        "test_probs": list(test_probs),
        "val_accuracy": 1.0,
    }


def test_short_model_name():
    assert modal_app.short_model_name("google/gemma-4-E2B-it") == "e2b"
    assert modal_app.short_model_name("google/gemma-4-26B-A4B-it") == "26b-a4b"
    assert modal_app.short_model_name("org/Other-Model") == "other-model"


def test_default_run_name():
    name = modal_app.default_run_name("google/gemma-4-E4B-it", datetime(2026, 11, 17, 9, 5, 3))
    assert name == "gemma4-e4b-20261117-090503"
    assert modal_app.default_run_name("google/gemma-4-E4B-it", datetime(2026, 1, 1), smoke=True).endswith("-smoke")


def test_build_messages():
    msgs = modal_app.build_messages("A passenger.", "sys", "Q?")
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert msgs[1]["content"] == "A passenger.\n\nQ?"


def test_warmup_steps():
    assert modal_app.warmup_steps_for(712, 16, 3, -1) == 6
    assert modal_app.warmup_steps_for(64, 16, 1, 10) == 1


def test_accuracy():
    assert modal_app.accuracy([1, 0, 1, 0], [0.9, 0.2, 0.4, 0.6]) == 0.5
    with pytest.raises(ValueError):
        modal_app.accuracy([], [])


def test_assemble_oof_reorders_to_train_order():
    results = [result(0, [3, 1], [0.3, 0.1], [0, 0]), result(1, [2, 4], [0.2, 0.4], [0, 0])]
    oof = modal_app.assemble_oof(results, [1, 2, 3, 4])
    assert oof["PassengerId"].tolist() == [1, 2, 3, 4]
    assert oof["prob"].tolist() == pytest.approx([0.1, 0.2, 0.3, 0.4])


def test_assemble_oof_rejects_duplicates_and_gaps():
    dup = [result(0, [1, 2], [0.1, 0.2], [0, 0]), result(1, [2, 3], [0.2, 0.3], [0, 0])]
    with pytest.raises(ValueError, match="more than one"):
        modal_app.assemble_oof(dup, [1, 2, 3])
    gap = [result(0, [1], [0.1], [0, 0])]
    with pytest.raises(ValueError, match="missing"):
        modal_app.assemble_oof(gap, [1, 2])


def test_average_test_probs():
    results = [result(0, [1], [0.1], [0.2, 0.8]), result(1, [2], [0.2], [0.4, 0.6])]
    avg = modal_app.average_test_probs(results, [10, 11])
    assert avg["PassengerId"].tolist() == [10, 11]
    assert avg["prob"].tolist() == pytest.approx([0.3, 0.7])


def test_average_test_probs_rejects_mismatched_ids():
    with pytest.raises(ValueError):
        modal_app.average_test_probs([result(0, [1], [0.1], [0.2, 0.8])], [10, 12])


def test_build_metrics():
    results = [result(1, [2], [0.9], [0.5, 0.5]), result(0, [1], [0.1], [0.5, 0.5])]
    results[0]["val_accuracy"], results[1]["val_accuracy"] = 0.0, 1.0
    oof = pd.DataFrame({"PassengerId": [1, 2], "prob": [0.1, 0.9]})
    metrics = modal_app.build_metrics(
        run_name="r", model_id="google/gemma-4-E2B-it", gpu="L40S", folds=2, epochs=3,
        lr=1e-4, lora_r=16, lora_alpha=32, seed=42, smoke=False, oof=oof,
        y_true=[0, 1], results=results, n_test=2, elapsed_seconds=12.34,
    )
    assert metrics["oof_accuracy"] == 1.0
    assert metrics["fold_accuracies"] == [1.0, 0.0]
    assert metrics["model_id"] == "google/gemma-4-E2B-it" and metrics["folds"] == 2


def test_attn_candidates_order_and_shape():
    labels = [label for label, _ in modal_app.ATTN_CANDIDATES]
    assert labels[-1] == "eager" and modal_app.ATTN_CANDIDATES[-1][1] == "eager"
    first = modal_app.ATTN_CANDIDATES[0][1]
    assert first["audio_config"] == "eager" and first["text_config"] == "sdpa"
    assert modal_app.MODEL_CLASS_NAMES[0] == "AutoModelForCausalLM"


def test_is_attention_error():
    msg = "Gemma4AudioModel does not support an attention implementation through torch.nn.functional.scaled_dot_product_attention yet"
    assert modal_app.is_attention_error(ValueError(msg))
    assert not modal_app.is_attention_error(OSError("401 gated repo"))
