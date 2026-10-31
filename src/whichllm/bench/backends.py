"""Decode-speed measurement on the runtimes people actually use.

Each backend reports the speed the runtime measured itself, except an
OpenAI-compatible server, which is timed from its stream. A backend that starts
a process or loads a model stops or unloads it again, even after Ctrl+C.
"""

from __future__ import annotations

import json
import shutil
import socket
import subprocess
import tempfile
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import httpx

OLLAMA_URL = "http://127.0.0.1:11434"
LM_STUDIO_URL = "http://127.0.0.1:1234"
LM_STUDIO_IDENTIFIER = "whichllm-bench"

# Raw text, not a chat turn: every runtime decodes from the same tokens, with no
# chat template, and a passage like this one has no natural place to stop.
PROMPT = (
    "The history of the printing press is a long one, and it begins well before "
    "Gutenberg. In the centuries that followed, the craft spread from one city "
    "to the next, and each of them added something of its own. To understand "
    "how that happened, we should start with"
)

_GENERATE_TIMEOUT_SECONDS = 1800.0
_REQUEST_TIMEOUT_SECONDS = 10.0
# llama.cpp may be downloading the model before it listens.
_LOAD_TIMEOUT_SECONDS = 7200.0
_STOP_TIMEOUT_SECONDS = 10.0
_LOG_TAIL_BYTES = 2000
_SEED = 42


class BenchError(Exception):
    """A measurement that could not run, with a message for the user."""


@dataclass(frozen=True)
class Sample:
    tok_per_sec: float
    tokens: int


class Backend:
    """A runtime to measure on. Use it as a context manager."""

    engine = "llama.cpp"
    app: str | None = None
    version: str | None = None

    def __enter__(self) -> Backend:
        try:
            self.start()
        except BaseException:
            self.stop()
            raise
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.stop()

    def start(self) -> None:
        """Get the model ready to generate."""

    def stop(self) -> None:
        """Undo what ``start`` did. Safe to call after a failed start."""

    def generate(self, tokens: int) -> Sample:
        raise NotImplementedError


def _sample(tok_per_sec: object, tokens: object) -> Sample:
    if not isinstance(tok_per_sec, (int, float)) or tok_per_sec <= 0:
        raise BenchError(f"The runtime reported no decode speed: {tok_per_sec!r}")
    if not isinstance(tokens, int) or tokens <= 0:
        raise BenchError(f"The runtime reported no generated tokens: {tokens!r}")
    return Sample(float(tok_per_sec), tokens)


def _post_json(url: str, body: dict, runtime: str) -> dict:
    try:
        response = httpx.post(url, json=body, timeout=_GENERATE_TIMEOUT_SECONDS)
    except httpx.HTTPError as error:
        raise BenchError(f"{runtime} did not answer at {url}: {error}") from error
    if response.status_code != 200:
        raise BenchError(
            f"{runtime} answered {response.status_code}: {response.text[:300]}"
        )
    return response.json()


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


class LlamaServer(Backend):
    """A llama-server this backend starts and stops itself."""

    def __init__(
        self,
        command: Sequence[str],
        *,
        load_timeout: float = _LOAD_TIMEOUT_SECONDS,
    ) -> None:
        self._command = list(command)
        self._load_timeout = load_timeout
        self._process: subprocess.Popen | None = None
        self._log = None
        self._url = ""

    def start(self) -> None:
        port = free_port()
        self._url = f"http://127.0.0.1:{port}"
        self._log = tempfile.TemporaryFile()
        try:
            self._process = subprocess.Popen(
                [*self._command, "--host", "127.0.0.1", "--port", str(port)],
                stdin=subprocess.DEVNULL,
                stdout=self._log,
                stderr=subprocess.STDOUT,
            )
        except OSError as error:
            raise BenchError(f"Could not start llama-server: {error}") from error
        self._wait_until_ready()
        self.version = self._build_info()

    def _log_tail(self) -> str:
        if self._log is None:
            return ""
        self._log.seek(0, 2)
        self._log.seek(max(0, self._log.tell() - _LOG_TAIL_BYTES))
        return self._log.read().decode("utf-8", errors="replace").strip()

    def _wait_until_ready(self) -> None:
        deadline = time.monotonic() + self._load_timeout
        while time.monotonic() < deadline:
            if self._process.poll() is not None:
                raise BenchError(
                    f"llama-server exited with code {self._process.returncode}:\n"
                    f"{self._log_tail()}"
                )
            try:
                response = httpx.get(
                    f"{self._url}/health", timeout=_REQUEST_TIMEOUT_SECONDS
                )
            except httpx.HTTPError:
                response = None
            if response is not None and response.status_code == 200:
                return
            time.sleep(0.2)
        raise BenchError("llama-server did not become ready in time")

    def _build_info(self) -> str | None:
        try:
            props = httpx.get(f"{self._url}/props", timeout=_REQUEST_TIMEOUT_SECONDS)
            return props.json().get("build_info")
        except (httpx.HTTPError, ValueError):
            return None

    def stop(self) -> None:
        process, self._process = self._process, None
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=_STOP_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        if self._log is not None:
            self._log.close()
            self._log = None

    def generate(self, tokens: int) -> Sample:
        result = _post_json(
            f"{self._url}/completion",
            {
                "prompt": PROMPT,
                "n_predict": tokens,
                "temperature": 0,
                "seed": _SEED,
                "cache_prompt": False,
                "ignore_eos": True,
            },
            "llama-server",
        )
        timings = result.get("timings") or {}
        return _sample(timings.get("predicted_per_second"), timings.get("predicted_n"))


class Ollama(Backend):
    """A running Ollama. A model it did not hold is pulled only after a yes."""

    app = "Ollama"

    def __init__(
        self,
        model: str,
        context_length: int,
        *,
        confirm_pull: Callable[[str], bool],
        url: str = OLLAMA_URL,
    ) -> None:
        self._model = model
        self._context_length = context_length
        self._confirm_pull = confirm_pull
        self._url = url
        self._loaded = False

    def start(self) -> None:
        try:
            version = httpx.get(
                f"{self._url}/api/version", timeout=_REQUEST_TIMEOUT_SECONDS
            )
            shown = httpx.post(
                f"{self._url}/api/show",
                json={"model": self._model},
                timeout=_REQUEST_TIMEOUT_SECONDS,
            )
        except httpx.HTTPError as error:
            raise BenchError(
                f"Ollama is not running at {self._url}: {error}"
            ) from error
        self.version = version.json().get("version")
        if shown.status_code == 404:
            if not self._confirm_pull(self._model):
                raise BenchError(f"Ollama does not have {self._model}")
            _post_json(
                f"{self._url}/api/pull",
                {"model": self._model, "stream": False},
                "Ollama",
            )
        self._loaded = True

    def stop(self) -> None:
        if not self._loaded:
            return
        self._loaded = False
        try:
            httpx.post(
                f"{self._url}/api/generate",
                json={"model": self._model, "keep_alive": 0},
                timeout=_REQUEST_TIMEOUT_SECONDS,
            )
        except httpx.HTTPError:
            pass

    def generate(self, tokens: int) -> Sample:
        result = _post_json(
            f"{self._url}/api/generate",
            {
                "model": self._model,
                "prompt": PROMPT,
                "raw": True,
                "stream": False,
                "options": {
                    "temperature": 0,
                    "seed": _SEED,
                    "num_predict": tokens,
                    "num_ctx": self._context_length,
                },
            },
            "Ollama",
        )
        count = result.get("eval_count")
        duration = result.get("eval_duration")
        if not count or not duration:
            raise BenchError("Ollama reported no decode timing")
        return _sample(count / (duration / 1e9), count)


class LMStudio(Backend):
    """LM Studio through its `lms` tool. The model must already be in its library."""

    app = "LM Studio"

    def __init__(
        self,
        filename: str,
        context_length: int,
        *,
        lms: str = "lms",
        url: str = LM_STUDIO_URL,
    ) -> None:
        self._filename = filename.rsplit("/", 1)[-1]
        self._context_length = context_length
        self._lms = lms
        self._url = url
        self._loaded = False

    def _run(self, *args: str, timeout: float = 600.0) -> str:
        try:
            result = subprocess.run(
                [self._lms, *args],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise BenchError(f"lms {args[0]} failed: {error}") from error
        if result.returncode != 0:
            raise BenchError(
                f"lms {args[0]} failed: {(result.stderr or result.stdout).strip()}"
            )
        return result.stdout

    def _model_key(self) -> str:
        try:
            listed = json.loads(self._run("ls", "--json", timeout=60.0))
        except ValueError as error:
            raise BenchError(f"lms ls printed no JSON: {error}") from error
        for entry in listed:
            if self._filename in str(entry.get("path", "")):
                return entry["modelKey"]
        raise BenchError(
            f"{self._filename} is not in LM Studio's library. Download it there first."
        )

    def _cli_version(self) -> str | None:
        try:
            return json.loads(self._run("version", "--json", timeout=30.0))["version"]
        except (ValueError, KeyError, TypeError):
            return None

    def start(self) -> None:
        self._run("server", "start", timeout=60.0)
        self.version = self._cli_version()
        key = self._model_key()
        self._loaded = True
        self._run(
            "load",
            key,
            "--identifier",
            LM_STUDIO_IDENTIFIER,
            "--context-length",
            str(self._context_length),
            "--yes",
        )

    def stop(self) -> None:
        if not self._loaded:
            return
        self._loaded = False
        try:
            self._run("unload", LM_STUDIO_IDENTIFIER, timeout=60.0)
        except BenchError:
            pass

    def generate(self, tokens: int) -> Sample:
        result = _post_json(
            f"{self._url}/api/v0/completions",
            {
                "model": LM_STUDIO_IDENTIFIER,
                "prompt": PROMPT,
                "max_tokens": tokens,
                "temperature": 0,
                "stream": False,
            },
            "LM Studio",
        )
        stats = result.get("stats") or {}
        usage = result.get("usage") or {}
        return _sample(stats.get("tokens_per_second"), usage.get("completion_tokens"))


class OpenAICompatible(Backend):
    """Any running server with an OpenAI completions route, timed from its stream."""

    def __init__(self, url: str, model: str, engine: str) -> None:
        self._url = url.rstrip("/")
        self._model = model
        self.engine = engine

    def generate(self, tokens: int) -> Sample:
        body = {
            "model": self._model,
            "prompt": PROMPT,
            "max_tokens": tokens,
            "temperature": 0,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        first = last = 0.0
        chunks = 0
        usage_tokens = None
        try:
            with httpx.stream(
                "POST",
                f"{self._url}/completions",
                json=body,
                timeout=_GENERATE_TIMEOUT_SECONDS,
            ) as response:
                if response.status_code != 200:
                    response.read()
                    raise BenchError(
                        f"The server answered {response.status_code}: "
                        f"{response.text[:300]}"
                    )
                for line in response.iter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line.removeprefix("data:").strip()
                    if data == "[DONE]":
                        break
                    event = json.loads(data)
                    usage = event.get("usage") or {}
                    usage_tokens = usage.get("completion_tokens", usage_tokens)
                    if not (event.get("choices") or [{}])[0].get("text"):
                        continue
                    last = time.perf_counter()
                    first = first or last
                    chunks += 1
        except httpx.HTTPError as error:
            raise BenchError(
                f"The server did not answer at {self._url}: {error}"
            ) from error
        generated = usage_tokens or chunks
        if generated < 2 or last <= first:
            raise BenchError("The server's stream was too short to time")
        # The clock starts at the first token, so it does not count.
        return _sample((generated - 1) / (last - first), generated)


def ollama_is_running(url: str = OLLAMA_URL) -> bool:
    try:
        return httpx.get(f"{url}/api/version", timeout=1.0).status_code == 200
    except httpx.HTTPError:
        return False


def auto_backend() -> str:
    """The first runtime on this machine that can measure a GGUF."""
    if shutil.which("llama-server"):
        return "llama-server"
    if ollama_is_running():
        return "ollama"
    if shutil.which("lms"):
        return "lm-studio"
    raise BenchError(
        "No runtime found. Install llama.cpp (brew install llama.cpp, winget "
        "install ggml.llamacpp), or start Ollama or LM Studio."
    )


def measure(backend: Backend, *, runs: int, tokens: int) -> list[Sample]:
    """One warm-up generation, then ``runs`` measured ones."""
    backend.generate(tokens)
    return [backend.generate(tokens) for _ in range(runs)]
