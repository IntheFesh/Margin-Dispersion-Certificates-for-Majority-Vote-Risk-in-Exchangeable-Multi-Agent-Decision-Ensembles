"""vLLM-only generation client via the OpenAI-compatible HTTP API.

Panel A inference is served EXCLUSIVELY by a vLLM server exposing the
OpenAI-compatible ``/v1`` endpoints. We talk to it through the ``openai``
Python client pointed at the local vLLM ``base_url``. We never use
HuggingFace ``transformers`` for inference or hidden-state extraction.

The ``openai`` package is imported LAZILY inside method bodies so that this
module imports cleanly on hosts where ``openai`` (and ``vllm``) are not
installed (e.g. the code-authoring environment, Phase 0).
"""
from __future__ import annotations

from typing import Any


class VLLMClient:
    """Thin wrapper over an OpenAI-compatible client backed by a vLLM server.

    One ``generate`` call requests all ``n_samples`` completions for a prompt
    in a SINGLE HTTP request via the OpenAI ``n`` parameter; vLLM shares the
    prompt KV cache across the ``n`` parallel samples, which is the whole
    point of batching the samples server-side rather than issuing N requests.

    Per-generation determinism: the caller derives each request ``seed`` from
    ``(cell_id, instance_idx, sample_idx, global_seed)`` via
    ``src.utils.seeds.derive_seed`` and passes the resulting integer here as
    ``seed``. This client does not invent seeds.
    """

    def __init__(self, base_url: str, api_key: str = "EMPTY", model: str | None = None) -> None:
        """Store connection parameters; the client is created lazily.

        Parameters
        ----------
        base_url:
            Base URL of the vLLM OpenAI-compatible server, e.g.
            ``"http://localhost:8000/v1"``.
        api_key:
            Placeholder key; vLLM ignores it but the OpenAI client requires a
            non-empty value, hence the ``"EMPTY"`` default.
        model:
            Model name as registered by the vLLM server (typically the HF
            path). May be set later but must be non-None before ``generate``.
        """
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self._client: Any | None = None

    def _get_client(self) -> Any:
        """Lazily construct and cache the OpenAI client.

        ``openai`` is imported here (not at module top level) so the module
        stays importable without the dependency installed.
        """
        if self._client is None:
            try:
                from openai import OpenAI  # lazy import: see module docstring
            except ImportError as e:  # pragma: no cover - depends on env
                raise RuntimeError(
                    "openai package is required to talk to the vLLM server; "
                    "install it on the inference host"
                ) from e
            self._client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        return self._client

    def generate(
        self,
        prompt: str,
        n_samples: int,
        temperature: float,
        max_tokens: int,
        seed: int,
        top_p: float = 0.95,
    ) -> list[str]:
        """Generate ``n_samples`` completions for ``prompt`` in one request.

        Uses the chat/completions endpoint with the OpenAI ``n`` parameter so
        vLLM produces all samples while sharing the prompt KV cache. The
        ``seed`` is forwarded to vLLM for reproducible sampling and must have
        been derived deterministically by the caller from
        ``(cell_id, instance_idx, sample_idx, global_seed)``.

        Returns
        -------
        list[str]
            Exactly ``n_samples`` completion text strings.

        Raises
        ------
        ValueError
            If arguments are out of range or ``model`` is unset.
        RuntimeError
            On any API failure or if the server returns the wrong number of
            choices (no silent truncation / padding).
        """
        if self.model is None:
            raise ValueError("VLLMClient.model must be set before generate()")
        if n_samples < 1:
            raise ValueError(f"n_samples must be >= 1, got {n_samples}")
        if max_tokens < 1:
            raise ValueError(f"max_tokens must be >= 1, got {max_tokens}")

        client = self._get_client()
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                n=n_samples,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                seed=seed,
            )
        except Exception as e:  # noqa: BLE001 - re-raise as structured RuntimeError
            raise RuntimeError(
                f"vLLM chat.completions request failed (model={self.model}, "
                f"n={n_samples}, seed={seed}): {e!r}"
            ) from e

        choices = getattr(response, "choices", None)
        if choices is None or len(choices) != n_samples:
            got = None if choices is None else len(choices)
            raise RuntimeError(
                f"vLLM returned {got} choices but {n_samples} were requested "
                f"(model={self.model}, seed={seed})"
            )

        completions: list[str] = []
        for choice in choices:
            content = choice.message.content
            completions.append("" if content is None else content)
        return completions
