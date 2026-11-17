import pytest

from titanic.lora import LANGUAGE_PROJ_NAMES, select_lora_targets

LM = "model.language_model.layers"


def fake_modules():
    return [
        ("", "Gemma4ForConditionalGeneration"),
        ("model.language_model.embed_tokens", "Embedding"),
        (f"{LM}.1.self_attn.q_proj", "Linear"),
        (f"{LM}.0.self_attn.q_proj", "Linear"),
        (f"{LM}.0.self_attn.k_proj", "Linear"),
        (f"{LM}.0.self_attn.o_proj", "Linear"),
        (f"{LM}.0.mlp.gate_proj", "Linear"),
        (f"{LM}.0.mlp.up_proj", "Linear"),
        (f"{LM}.0.mlp.down_proj", "Linear"),
        (f"{LM}.0.input_layernorm", "Gemma4RMSNorm"),
        ("model.vision_tower.encoder.layers.0.self_attn.q_proj", "Gemma4ClippableLinear"),
        ("model.vision_tower.encoder.layers.0.self_attn.q_proj.linear", "Linear"),
        ("model.vision_tower.encoder.layers.0.mlp.up_proj", "Linear"),
        ("model.audio_tower.layers.0.self_attn.q_proj", "Linear"),
        ("model.embed_vision.embedding_projection", "Linear"),
        ("lm_head", "Linear"),
    ]


def test_keeps_only_language_model_projections():
    targets = select_lora_targets(fake_modules())
    assert targets == [
        f"{LM}.0.mlp.down_proj",
        f"{LM}.0.mlp.gate_proj",
        f"{LM}.0.mlp.up_proj",
        f"{LM}.0.self_attn.k_proj",
        f"{LM}.0.self_attn.o_proj",
        f"{LM}.0.self_attn.q_proj",
        f"{LM}.1.self_attn.q_proj",
    ]


def test_excludes_towers_embeddings_and_head():
    targets = select_lora_targets(fake_modules())
    assert not any("vision" in t or "audio" in t or "embed" in t for t in targets)
    assert "lm_head" not in targets


def test_rejects_clippable_wrapper_but_keeps_its_linear_child():
    modules = [
        (f"{LM}.0.self_attn.q_proj", "Gemma4ClippableLinear"),
        (f"{LM}.0.self_attn.q_proj.linear", "Linear"),
    ]
    assert select_lora_targets(modules) == [f"{LM}.0.self_attn.q_proj.linear"]


def test_linear_child_of_non_projection_is_ignored():
    modules = [
        (f"{LM}.0.router.linear", "Linear"),
        (f"{LM}.0.self_attn.v_proj", "Linear"),
    ]
    assert select_lora_targets(modules) == [f"{LM}.0.self_attn.v_proj"]


def test_non_linear_class_with_projection_name_is_ignored():
    modules = [
        (f"{LM}.0.self_attn.q_proj", "Gemma4ClippableLinear"),
        (f"{LM}.0.self_attn.k_proj", "Linear"),
    ]
    assert select_lora_targets(modules) == [f"{LM}.0.self_attn.k_proj"]


def test_deduplicates_and_sorts():
    name = f"{LM}.2.mlp.up_proj"
    modules = [(name, "Linear"), (f"{LM}.1.mlp.up_proj", "Linear"), (name, "Linear")]
    assert select_lora_targets(modules) == [f"{LM}.1.mlp.up_proj", name]


def test_accepts_generator_input():
    gen = ((f"{LM}.0.self_attn.{p}", "Linear") for p in ("q_proj", "v_proj"))
    assert len(select_lora_targets(gen)) == 2


def test_case_insensitive_exclusion():
    modules = [
        ("model.Vision_Tower.layers.0.q_proj", "Linear"),
        (f"{LM}.0.self_attn.q_proj", "Linear"),
    ]
    assert select_lora_targets(modules) == [f"{LM}.0.self_attn.q_proj"]


@pytest.mark.parametrize(
    "modules",
    [
        [],
        [("lm_head", "Linear"), ("model.language_model.embed_tokens", "Embedding")],
        [("model.vision_tower.layers.0.q_proj", "Linear")],
    ],
)
def test_raises_when_nothing_selected(modules):
    with pytest.raises(ValueError, match="No LoRA target"):
        select_lora_targets(modules)


def test_proj_names_constant():
    assert "q_proj" in LANGUAGE_PROJ_NAMES and len(LANGUAGE_PROJ_NAMES) == 7
