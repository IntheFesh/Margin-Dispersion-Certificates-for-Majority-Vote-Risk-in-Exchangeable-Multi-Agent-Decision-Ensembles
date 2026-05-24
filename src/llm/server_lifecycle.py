"""vLLM server lifecycle management.

Panel A inference runs against a vLLM server launched via the ``vllm serve``
CLI (NOT the Python ``vllm`` package imported in-process, and NOT HuggingFace
``transformers``). We spawn the server as a subprocess, poll its ``/health``
endpoint until ready, and shut it down gracefully.

``subprocess`` and ``urllib`` are stdlib, so importing them at module top
level is fine and adds no third-party dependency. We deliberately do NOT
``import vllm`` in Python; the engine is invoked purely through its CLI.
"""
from __future__ import annotations

import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

_HEALTH_TIMEOUT_S = 300.0  # 5 minutes
_HEALTH_POLL_INTERVAL_S = 2.0
_STOP_SIGINT_WAIT_S = 30.0


def start_vllm_server(
    model_path: str,
    port: int,
    gpu_memory_utilization: float = 0.90,
    max_model_len: int = 4096,
    enforce_eager: bool = False,
    seed: int = 0,
) -> "subprocess.Popen":
    """Spawn a ``vllm serve`` subprocess and wait until it is healthy.

    Builds the exact CLI invocation, prints it (so the launched command is
    captured in run logs), starts the process, then polls
    ``http://localhost:{port}/health`` until it returns HTTP 200.

    Parameters
    ----------
    model_path:
        HF path / local path served by vLLM (also the OpenAI ``model`` name).
    port:
        Port the OpenAI-compatible server binds to.
    gpu_memory_utilization, max_model_len, enforce_eager, seed:
        Passed straight through to ``vllm serve`` for reproducible serving.

    Returns
    -------
    subprocess.Popen
        The running server process. Pass it to :func:`stop_vllm_server`.

    Raises
    ------
    TimeoutError
        If ``/health`` does not return 200 within 5 minutes.
    RuntimeError
        If the subprocess exits before becoming healthy.
    """
    cmd: list[str] = [
        "vllm",
        "serve",
        model_path,
        "--port",
        str(port),
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
        "--max-model-len",
        str(max_model_len),
        "--seed",
        str(seed),
    ]
    if enforce_eager:
        cmd.append("--enforce-eager")

    # Log the exact command line so the launched server is reproducible.
    print(f"[server_lifecycle] launching vLLM: {' '.join(cmd)}", file=sys.stderr, flush=True)

    proc = subprocess.Popen(cmd)

    health_url = f"http://localhost:{port}/health"
    deadline = time.time() + _HEALTH_TIMEOUT_S
    while time.time() < deadline:
        # If the process died, fail loudly rather than polling a dead server.
        if proc.poll() is not None:
            raise RuntimeError(
                f"vLLM server process exited early with code {proc.returncode} "
                f"before becoming healthy (cmd: {' '.join(cmd)})"
            )
        try:
            with urllib.request.urlopen(health_url, timeout=5.0) as resp:
                if resp.status == 200:
                    print(
                        f"[server_lifecycle] vLLM healthy on port {port}",
                        file=sys.stderr,
                        flush=True,
                    )
                    return proc
        except (urllib.error.URLError, ConnectionError, OSError):
            # Server not up yet; keep polling until the deadline.
            pass
        time.sleep(_HEALTH_POLL_INTERVAL_S)

    # Timed out: stop the process we started, then raise.
    stop_vllm_server(proc)
    raise TimeoutError(
        f"vLLM server on port {port} did not become healthy within "
        f"{_HEALTH_TIMEOUT_S:.0f}s (cmd: {' '.join(cmd)})"
    )


def stop_vllm_server(proc: "subprocess.Popen") -> None:
    """Stop a vLLM server process gracefully.

    Sends SIGINT and waits up to 30 seconds for a clean shutdown; if the
    process is still alive, escalates to SIGTERM. Idempotent: a process that
    has already exited is left untouched.
    """
    if proc is None:
        return
    if proc.poll() is not None:
        return  # already exited

    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=_STOP_SIGINT_WAIT_S)
        return
    except subprocess.TimeoutExpired:
        pass

    proc.terminate()  # SIGTERM
    try:
        proc.wait(timeout=_STOP_SIGINT_WAIT_S)
    except subprocess.TimeoutExpired:
        # Last resort: ensure the process does not leak.
        proc.kill()
        proc.wait()
