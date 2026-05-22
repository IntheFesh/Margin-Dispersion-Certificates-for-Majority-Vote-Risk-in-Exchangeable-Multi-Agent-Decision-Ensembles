from __future__ import annotations
from typing import Iterable
import pandas as pd  # noqa: F401  (kept for downstream consumers)


def load_hf_model(model_name: str, revision: str | None = None, device: str | None = None):
    """Real Hugging Face loader. Imports torch/transformers lazily so that
    the rest of the codebase can be imported without GPU/model resources.

    Raises ImportError if torch/transformers are not installed.
    Raises RuntimeError on any model/tokenizer load failure (no silent
    fallback or dummy model is allowed).
    """
    try:
        import torch  # noqa: F401
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as e:
        raise ImportError(
            "torch and transformers are required for hf_generate.load_hf_model"
        ) from e
    try:
        tok = AutoTokenizer.from_pretrained(model_name, revision=revision, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(model_name, revision=revision)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load HF model '{model_name}' (revision={revision}): {e}"
        ) from e
    if device is not None:
        try:
            model = model.to(device)
        except Exception as e:
            raise RuntimeError(f"Failed to move model to device {device}: {e}") from e
    return model, tok


def generate_greedy(model, tok, prompts: Iterable[str], max_new_tokens: int = 256) -> list[str]:
    """Deterministic greedy decoding. No sampling, no temperature mixing.
    Returns the decoded text for each prompt. Raises RuntimeError if a
    generation step fails for any prompt (no silent skipping)."""
    try:
        import torch
    except Exception as e:
        raise ImportError("torch is required for generate_greedy") from e
    outs: list[str] = []
    for p in prompts:
        try:
            ids = tok(p, return_tensors="pt").to(model.device)
            with torch.no_grad():
                gen = model.generate(
                    **ids,
                    do_sample=False,
                    num_beams=1,
                    max_new_tokens=int(max_new_tokens),
                    pad_token_id=tok.pad_token_id or tok.eos_token_id,
                )
            text = tok.decode(gen[0][ids["input_ids"].shape[1] :], skip_special_tokens=True)
        except Exception as e:
            raise RuntimeError(f"generation failed for prompt: {e}") from e
        outs.append(text)
    return outs


def extract_prompt_representation(
    model,
    tok,
    prompts: Iterable[str],
    layer: int | str = "penultimate",
    pool: str = "mean",
) -> list:
    """Extract hidden-state representations from prompt tokens at the
    requested layer. Used by Panel B CKA.

    layer: int index, "last", "penultimate", or "mean_last_4".
    pool: "mean" (default per paper) or "last_token" (Appendix B
    robustness alternative).

    Raises RuntimeError if hidden-state extraction fails or the requested
    layer is out of range; no silent fallback.
    """
    try:
        import torch
    except Exception as e:
        raise ImportError("torch is required for extract_prompt_representation") from e
    reps: list = []
    for p in prompts:
        try:
            ids = tok(p, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model(**ids, output_hidden_states=True)
            hs = out.hidden_states  # tuple of [1, T, d]
            if isinstance(layer, int):
                if layer < 0 or layer >= len(hs):
                    raise RuntimeError(f"layer index {layer} out of range for {len(hs)} layers")
                h = hs[layer]
            elif layer == "last":
                h = hs[-1]
            elif layer == "penultimate":
                if len(hs) < 2:
                    raise RuntimeError("model has fewer than 2 hidden-state layers")
                h = hs[-2]
            elif layer == "mean_last_4":
                if len(hs) < 4:
                    raise RuntimeError("model has fewer than 4 hidden-state layers")
                h = torch.stack(hs[-4:]).mean(dim=0)
            else:
                raise RuntimeError(f"unsupported layer spec: {layer}")
            h0 = h.squeeze(0)  # [T, d]
            if pool == "mean":
                rep = h0.mean(dim=0).detach().cpu().numpy()
            elif pool == "last_token":
                rep = h0[-1].detach().cpu().numpy()
            else:
                raise RuntimeError(f"unsupported pool spec: {pool}")
        except Exception as e:
            raise RuntimeError(f"representation extraction failed: {e}") from e
        reps.append(rep)
    return reps
