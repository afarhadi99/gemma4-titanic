"""Pure-Python LoRA target selection (no torch import, safe to unit test anywhere)."""

from __future__ import annotations

from collections.abc import Iterable

LANGUAGE_PROJ_NAMES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    # Fused projections (Phi-3/4: qkv_proj, gate_up_proj).
    "qkv_proj",
    "gate_up_proj",
    # Gated DeltaNet / linear-attention layers (Qwen3.5: in_proj_qkv, in_proj_z, out_proj;
    # Qwen3-Next: in_proj_qkvz). The tiny in_proj_a / in_proj_b gate projections are skipped.
    "in_proj_qkv",
    "in_proj_qkvz",
    "in_proj_z",
    "out_proj",
    # LFM3: SwiGLU MLP as w1 (gate) / w3 (up) / w2 (down), short-conv in_proj/out_proj.
    "w1",
    "w2",
    "w3",
    "in_proj",
)

_EXCLUDED_FRAGMENTS = ("vision", "visual", "audio", "speech", "multi_modal", "embed")


def _is_excluded(segments: list[str]) -> bool:
    return any(
        fragment in segment.lower()
        for segment in segments
        for fragment in _EXCLUDED_FRAGMENTS
    )


def _is_projection(segments: list[str]) -> bool:
    last = segments[-1]
    if last in LANGUAGE_PROJ_NAMES:
        return True
    return last == "linear" and len(segments) > 1 and segments[-2] in LANGUAGE_PROJ_NAMES


def select_lora_targets(modules: Iterable[tuple[str, str]]) -> list[str]:
    """Return sorted, unique full module names that are safe LoRA targets.

    `modules` holds (full module name, class name) pairs, e.g. from
    `[(n, type(m).__name__) for n, m in model.named_modules()]`. Only plain
    `Linear` layers inside the language model are kept: either a projection
    named in LANGUAGE_PROJ_NAMES, or the `.linear` child of such a projection
    (Gemma 4 wraps some projections in `Gemma4ClippableLinear`, which PEFT rejects).
    """
    selected: set[str] = set()
    for name, class_name in modules:
        if class_name != "Linear" or not name:
            continue
        segments = name.split(".")
        if _is_excluded(segments):
            continue
        if _is_projection(segments):
            selected.add(name)
    if not selected:
        raise ValueError(
            "No LoRA target modules found: expected torch.nn.Linear layers named "
            f"{', '.join(LANGUAGE_PROJ_NAMES)} outside vision/audio/embedding modules. "
            "The Gemma 4 module layout may differ from what this repo assumes."
        )
    return sorted(selected)
