"""Model-agnostic helpers for the yes/no LoRA classifier (pure Python; torch/transformers
are imported lazily so these are testable without GPU libraries)."""

from __future__ import annotations

SUB_CONFIGS = ("text_config", "vision_config", "audio_config")
MODEL_CLASS_NAMES = ("AutoModelForCausalLM", "AutoModelForMultimodalLM", "AutoModelForImageTextToText")
# Tried in order; the first pair where both words are single, distinct tokens wins.
LABEL_VARIANTS: tuple[tuple[str, str], ...] = (("no", "yes"), (" no", " yes"), ("No", "Yes"))


def attn_candidates(config) -> list[tuple[str, object]]:
    """Attention settings to try for `config`: sdpa, then per-sub-config sdpa (eager for
    audio) when the config has sub-configs, then eager."""
    out: list[tuple[str, object]] = [("sdpa", "sdpa")]
    subs = [k for k in SUB_CONFIGS if getattr(config, k, None) is not None]
    if subs:
        mapping: dict[str, str] = {"": "sdpa"}
        mapping.update({k: ("eager" if k == "audio_config" else "sdpa") for k in subs})
        label = "sdpa(" + ",".join(k.removesuffix("_config") for k in subs if k != "audio_config") + ")"
        if "audio_config" in subs:
            label += "+eager(audio)"
        out.append((label, mapping))
    out.append(("eager", "eager"))
    return out


def is_attention_error(exc: BaseException) -> bool:
    """True when `exc` complains about the attention implementation (worth trying another one)."""
    text = str(exc).lower()
    return any(k in text for k in ("attn_implementation", "attention implementation", "scaled_dot_product"))


def pick_label_tokens(encode, variants=LABEL_VARIANTS) -> tuple[tuple[str, str], list[int]]:
    """Choose the first (no, yes) word pair that encodes to two distinct single tokens.

    `encode(word) -> list[int]` (no special tokens). Returns (words, [no_id, yes_id])."""
    tried = []
    for words in variants:
        ids = [encode(w) for w in words]
        tried.append(dict(zip(words, ids)))
        if all(len(i) == 1 for i in ids) and ids[0][0] != ids[1][0]:
            return words, [ids[0][0], ids[1][0]]
    raise ValueError(f"No single-token distinct label pair found; tried {tried}")


def chat_prompt(tokenizer, messages: list[dict[str, str]]) -> str:
    """Generation prompt with thinking disabled; falls back if the template rejects the kwarg."""
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def load_causal_lm(model_id: str, *, dtype, config=None, trust_remote_code: bool = False, log=print):
    """Load a causal LM, trying each model class x attention candidate.

    With `config` given the model is built with `from_config` (random weights; for tests),
    otherwise `from_pretrained`. Returns (model, "Class + attn label")."""
    import transformers

    cfg = config
    if cfg is None:
        cfg = transformers.AutoConfig.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    attempts: list[str] = []
    last_exc: Exception | None = None
    for cls_name in MODEL_CLASS_NAMES:
        model_cls = getattr(transformers, cls_name, None)
        if model_cls is None:
            attempts.append(f"{cls_name}: not available in this transformers version")
            continue
        for attn_label, attn in attn_candidates(cfg):
            try:
                if config is None:
                    model = model_cls.from_pretrained(
                        model_id, dtype=dtype, attn_implementation=attn, trust_remote_code=trust_remote_code
                    )
                else:
                    model = model_cls.from_config(
                        config, dtype=dtype, attn_implementation=attn, trust_remote_code=trust_remote_code
                    )
            except Exception as exc:
                last_exc = exc
                attempts.append(f"{cls_name} + {attn_label}: {type(exc).__name__}: {str(exc)[:160]}")
                log(f"Load failed for {cls_name} + {attn_label}: {exc!r}")
                if not is_attention_error(exc):
                    break  # unrelated error: try the next model class, not another attention config
                continue
            desc = f"{cls_name} + attn_implementation={attn_label}"
            log(f"Loaded with {desc}")
            return model, desc
    raise RuntimeError(f"Could not load {model_id}. Tried: " + "; ".join(attempts)) from last_exc


# Repos whose custom code we have reviewed and agree to run without an explicit flag.
TRUST_REMOTE_CODE_ALLOWLIST = frozenset({"LiquidAI/LFM3-1.7B"})


def resolve_trust_remote_code(model_id: str, requested: bool = False, log=print) -> bool:
    """trust_remote_code is opt-in; only allowlisted repo ids enable it automatically."""
    trusted = bool(requested) or model_id in TRUST_REMOTE_CODE_ALLOWLIST
    if trusted:
        why = "requested" if requested else "allowlisted"
        log(f"WARNING: trust_remote_code=True for {model_id} ({why}); executing code from the model repo")
    return trusted
