"""`whichllm bench`: fake runtimes, real processes and real HTTP."""

from __future__ import annotations

import getpass
import json
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import psutil
import pytest
from typer.testing import CliRunner

from whichllm.bench import backends
from whichllm.bench.backends import (
    BenchError,
    LlamaServer,
    LMStudio,
    Ollama,
    OpenAICompatible,
    Sample,
    auto_backend,
    measure,
)
from whichllm.bench.report import build_report, reports_dir, save_report
from whichllm.cli import app
from whichllm.commands import bench as bench_command
from whichllm.engine.placement import Placement
from whichllm.engine.types import CompatibilityResult
from whichllm.hardware.types import GPUInfo, HardwareInfo
from whichllm.models.types import GGUFVariant, ModelInfo

_GiB = 1024**3

FAKE_LLAMA_SERVER = """
import json, os, sys
from http.server import BaseHTTPRequestHandler, HTTPServer

port = int(sys.argv[sys.argv.index("--port") + 1])
open(os.environ["FAKE_PID_FILE"], "w").write(str(os.getpid()))
if os.environ.get("FAKE_FAIL"):
    print("error: model file not found", flush=True)
    sys.exit(3)
polls = {"count": 0}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _send(self, status, body):
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/health":
            polls["count"] += 1
            ready = polls["count"] >= 3
            self._send(200 if ready else 503, {"status": "ok"})
        elif self.path == "/props":
            self._send(200, {"build_info": "b9999-abcdef"})

    def do_POST(self):
        length = int(self.headers["Content-Length"])
        request = json.loads(self.rfile.read(length))
        with open(os.environ["FAKE_REQUEST_FILE"], "w") as handle:
            json.dump(request, handle)
        self._send(
            200,
            {"timings": {"predicted_per_second": 88.5, "predicted_n": request["n_predict"]}},
        )


HTTPServer(("127.0.0.1", port), Handler).serve_forever()
"""


@pytest.fixture
def fake_llama(tmp_path, monkeypatch) -> SimpleNamespace:
    script = tmp_path / "fake_llama_server.py"
    script.write_text(FAKE_LLAMA_SERVER, encoding="utf-8")
    pid_file = tmp_path / "pid"
    request_file = tmp_path / "request.json"
    monkeypatch.setenv("FAKE_PID_FILE", str(pid_file))
    monkeypatch.setenv("FAKE_REQUEST_FILE", str(request_file))
    return SimpleNamespace(
        command=[sys.executable, str(script)],
        pid=lambda: int(pid_file.read_text()),
        request=lambda: json.loads(request_file.read_text()),
    )


def _gone(pid: int) -> bool:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if not psutil.pid_exists(pid):
            return True
        time.sleep(0.05)
    return False


class _FakeHTTP:
    """Scripted JSON and stream answers on a real local port."""

    def __init__(self, routes: dict) -> None:
        self.requests: list[tuple[str, dict]] = []
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _answer(self, body: dict | None) -> None:
                self.requests = fake.requests
                route = routes[self.path]
                result = route(body) if callable(route) else route
                status, payload = result if isinstance(result, tuple) else (200, result)
                data = (
                    payload.encode()
                    if isinstance(payload, str)
                    else json.dumps(payload).encode()
                )
                self.send_response(status)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                self._answer(None)

            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length) or b"{}")
                fake.requests.append((self.path, body))
                self._answer(body)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def fake_http():
    servers: list[_FakeHTTP] = []

    def start(routes: dict) -> _FakeHTTP:
        server = _FakeHTTP(routes)
        servers.append(server)
        return server

    yield start
    for server in servers:
        server.close()


def test_llama_server_measures_and_stops(fake_llama):
    with LlamaServer(fake_llama.command) as server:
        pid = fake_llama.pid()
        sample = server.generate(64)
        assert sample == Sample(88.5, 64)
        assert server.version == "b9999-abcdef"

    request = fake_llama.request()
    assert request["n_predict"] == 64
    assert request["temperature"] == 0
    assert request["cache_prompt"] is False
    assert request["ignore_eos"] is True
    assert _gone(pid)


@pytest.mark.parametrize("error", [RuntimeError, KeyboardInterrupt])
def test_llama_server_stops_on_an_error_or_ctrl_c(fake_llama, error):
    with pytest.raises(error):
        with LlamaServer(fake_llama.command):
            pid = fake_llama.pid()
            raise error()

    assert _gone(pid)


def test_llama_server_stops_when_loading_is_interrupted(fake_llama, monkeypatch):
    def interrupted(self):
        raise KeyboardInterrupt

    monkeypatch.setattr(LlamaServer, "_wait_until_ready", interrupted)

    with pytest.raises(KeyboardInterrupt):
        with LlamaServer(fake_llama.command):
            pass

    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            pid = fake_llama.pid()
            break
        except (FileNotFoundError, ValueError):
            time.sleep(0.05)
    else:
        return  # Stopped before the child got as far as writing its pid.
    assert _gone(pid)


def test_llama_server_that_exits_shows_its_log(fake_llama, monkeypatch):
    monkeypatch.setenv("FAKE_FAIL", "1")

    with pytest.raises(BenchError, match="model file not found"):
        with LlamaServer(fake_llama.command):
            pass


def test_llama_server_that_never_gets_ready_is_stopped(fake_llama, monkeypatch):
    monkeypatch.setattr(backends.httpx, "get", _refuse)

    with pytest.raises(BenchError, match="did not become ready"):
        with LlamaServer(fake_llama.command, load_timeout=0.5):
            pass

    assert _gone(fake_llama.pid())


def _refuse(*args, **kwargs):
    raise backends.httpx.ConnectError("refused")


def _ollama_routes(known: bool = True) -> dict:
    return {
        "/api/version": {"version": "0.40.0"},
        "/api/show": {} if known else (404, {"error": "not found"}),
        "/api/pull": {"status": "success"},
        "/api/generate": lambda body: (
            {"done": True}
            if body.get("keep_alive") == 0
            else {"eval_count": 128, "eval_duration": 1_280_000_000}
        ),
    }


def test_ollama_reads_its_own_timings_and_unloads(fake_http):
    server = fake_http(_ollama_routes())

    with Ollama("hf.co/o/m:Q4_K_M", 8192, confirm_pull=lambda _: True, url=server.url):
        pass
    with Ollama(
        "hf.co/o/m:Q4_K_M", 8192, confirm_pull=lambda _: True, url=server.url
    ) as ollama:
        sample = ollama.generate(128)

    assert sample == Sample(100.0, 128)
    assert ollama.version == "0.40.0"
    path, body = next(r for r in server.requests if r[1].get("prompt"))
    assert path == "/api/generate"
    assert body["raw"] is True
    assert body["options"]["num_ctx"] == 8192
    assert server.requests[-1][1] == {"model": "hf.co/o/m:Q4_K_M", "keep_alive": 0}


def test_ollama_pulls_only_after_a_yes(fake_http):
    server = fake_http(_ollama_routes(known=False))

    with pytest.raises(BenchError, match="does not have"):
        with Ollama("m", 4096, confirm_pull=lambda _: False, url=server.url):
            pass
    assert not any(path == "/api/pull" for path, _ in server.requests)

    with Ollama("m", 4096, confirm_pull=lambda _: True, url=server.url):
        pass
    assert any(path == "/api/pull" for path, _ in server.requests)


def test_ollama_that_is_not_running_says_so():
    with pytest.raises(BenchError, match="not running"):
        with Ollama("m", 4096, confirm_pull=lambda _: True, url="http://127.0.0.1:9"):
            pass


def _fake_lms(monkeypatch, *, listed: list[dict], fail_load: bool = False) -> list:
    calls: list[list[str]] = []

    def run(command, **kwargs):
        calls.append(command[1:])
        action = command[1]
        stdout = ""
        code = 0
        if action == "ls":
            stdout = json.dumps(listed)
        elif action == "version":
            stdout = '{"version":"abc123"}\n'
        elif action == "load" and fail_load:
            code = 1
        return subprocess.CompletedProcess(command, code, stdout, "load failed")

    monkeypatch.setattr(backends.subprocess, "run", run)
    return calls


_LM_MODEL = {"modelKey": "qwen3.5-2b", "path": "unsloth/Qwen3.5-2B-GGUF/Q4_K_M.gguf"}


def test_lm_studio_loads_measures_and_unloads(fake_http, monkeypatch):
    calls = _fake_lms(monkeypatch, listed=[_LM_MODEL])
    server = fake_http(
        {
            "/api/v0/completions": {
                "stats": {"tokens_per_second": 37.5},
                "usage": {"completion_tokens": 128},
            }
        }
    )

    with LMStudio("sub/Q4_K_M.gguf", 4096, url=server.url) as lm_studio:
        sample = lm_studio.generate(128)

    assert sample == Sample(37.5, 128)
    assert lm_studio.version == "abc123"
    assert ["version", "--json"] in calls
    load = next(call for call in calls if call[0] == "load")
    assert load[1] == "qwen3.5-2b"
    assert calls[-1] == ["unload", backends.LM_STUDIO_IDENTIFIER]


def test_lm_studio_unloads_after_a_failed_load(monkeypatch):
    calls = _fake_lms(monkeypatch, listed=[_LM_MODEL], fail_load=True)

    with pytest.raises(BenchError, match="load failed"):
        with LMStudio("Q4_K_M.gguf", 4096):
            pass

    assert calls[-1][0] == "unload"


def test_lm_studio_says_when_a_model_is_missing(monkeypatch):
    calls = _fake_lms(monkeypatch, listed=[])

    with pytest.raises(BenchError, match="not in LM Studio"):
        with LMStudio("Q4_K_M.gguf", 4096):
            pass

    assert not any(call[0] == "unload" for call in calls)


def _stream(tokens: int, *, usage: int | None = None) -> str:
    lines = ["data: " + json.dumps({"choices": [{"text": "x"}]}) for _ in range(tokens)]
    if usage is not None:
        lines.append(
            "data: "
            + json.dumps({"choices": [], "usage": {"completion_tokens": usage}})
        )
    lines.append("data: [DONE]")
    return "\n\n".join(lines) + "\n\n"


def test_openai_times_the_stream(fake_http, monkeypatch):
    server = fake_http({"/v1/completions": _stream(5, usage=128)})
    ticks = iter(range(100))
    monkeypatch.setattr(backends.time, "perf_counter", lambda: float(next(ticks)))

    with OpenAICompatible(f"{server.url}/v1/", "m", "vllm") as openai:
        sample = openai.generate(128)

    # Four intervals between five chunks, and the server's own token count.
    assert sample == Sample(127 / 4.0, 128)
    assert openai.engine == "vllm"
    assert server.requests[0][1]["stream"] is True


def test_openai_rejects_a_stream_too_short_to_time(fake_http):
    server = fake_http({"/v1/completions": _stream(1)})

    with pytest.raises(BenchError, match="too short"):
        OpenAICompatible(f"{server.url}/v1", "m", "vllm").generate(128)


def test_openai_reports_a_failed_request(fake_http):
    server = fake_http({"/v1/completions": (404, {"error": "no such model"})})

    with pytest.raises(BenchError, match="404"):
        OpenAICompatible(f"{server.url}/v1", "m", "vllm").generate(128)


def test_measure_throws_the_warm_up_away():
    speeds = iter([5.0, 90.0, 91.0, 92.0])

    class Fake:
        def generate(self, tokens):
            return Sample(next(speeds), tokens)

    assert [s.tok_per_sec for s in measure(Fake(), runs=3, tokens=8)] == [90, 91, 92]


@pytest.mark.parametrize(
    ("which", "ollama", "expected"),
    [
        (
            {"llama-server": "/bin/llama-server", "lms": "/bin/lms"},
            True,
            "llama-server",
        ),
        ({"lms": "/bin/lms"}, True, "ollama"),
        ({"lms": "/bin/lms"}, False, "lm-studio"),
    ],
)
def test_auto_prefers_plain_llama_cpp(monkeypatch, which, ollama, expected):
    monkeypatch.setattr(backends.shutil, "which", which.get)
    monkeypatch.setattr(backends, "ollama_is_running", lambda: ollama)

    assert auto_backend() == expected


def test_auto_explains_what_to_install(monkeypatch):
    monkeypatch.setattr(backends.shutil, "which", lambda _: None)
    monkeypatch.setattr(backends, "ollama_is_running", lambda: False)

    with pytest.raises(BenchError, match="Install llama.cpp"):
        auto_backend()


def _hardware() -> HardwareInfo:
    return HardwareInfo(
        gpus=[
            GPUInfo(
                name="NVIDIA GeForce RTX 5080",
                vendor="nvidia",
                vram_bytes=16 * _GiB,
                memory_bandwidth_gbps=960.0,
            )
        ],
        cpu_name="Intel Core Ultra 9 285K",
        cpu_cores=24,
        ram_bytes=64 * _GiB,
        ram_bandwidth_gbps=89.6,
        ram_bandwidth_source="detected",
        disk_free_bytes=500 * _GiB,
        os="windows",
    )


def _model() -> tuple[ModelInfo, GGUFVariant]:
    variant = GGUFVariant("Model-Q4_K_M.gguf", "Q4_K_M", 2 * _GiB)
    model = ModelInfo(
        id="org/Model-GGUF",
        family_id="model",
        name="Model",
        parameter_count=int(3e9),
        gguf_variants=[variant],
    )
    return model, variant


def _pick(model: ModelInfo, variant: GGUFVariant) -> CompatibilityResult:
    return CompatibilityResult(
        model=model,
        gguf_variant=variant,
        can_run=True,
        vram_required_bytes=3 * _GiB,
        vram_available_bytes=16 * _GiB,
        estimated_tok_per_sec=80.0,
        speed_confidence="high",
        placement=Placement("gpu", 2 * _GiB, 0, n_gpu_layers=32),
    )


def _report() -> dict:
    model, variant = _model()
    return build_report(
        version="0.6.0",
        hardware=_hardware(),
        backend=SimpleNamespace(engine="llama.cpp", app=None, version="b9999"),
        model=model,
        variant=variant,
        served_model=None,
        context_length=4096,
        picked=_pick(model, variant),
        samples=[Sample(88.0, 128), Sample(91.0, 128), Sample(90.0, 128)],
        vram_used_mb=2500,
    )


def test_report_records_the_measurement_and_the_estimate():
    report = _report()

    assert report["schema_version"] == 1
    assert report["method"] == "auto"
    assert report["runtime"] == {"engine": "llama.cpp", "app": None, "version": "b9999"}
    assert report["measurement"] == {
        "tok_per_sec": [88.0, 91.0, 90.0],
        "median_tok_per_sec": 90.0,
        "tokens": 128,
        "vram_used_mb": 2500,
    }
    assert report["estimate"]["tok_per_sec"] == 80.0
    assert report["placement"] == {
        "kind": "gpu",
        "n_cpu_moe": None,
        "n_gpu_layers": 32,
    }
    assert report["model"]["file_size_bytes"] == 2 * _GiB


def test_report_holds_nothing_that_identifies_the_machine(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    text = save_report(_report()).read_text(encoding="utf-8")

    for secret in (getpass.getuser(), socket.gethostname(), str(Path.home())):
        assert secret.lower() not in text.lower()
    assert set(json.loads(text)) == {
        "schema_version",
        "whichllm_version",
        "method",
        "hardware",
        "runtime",
        "model",
        "placement",
        "measurement",
        "estimate",
    }
    assert set(json.loads(text)["hardware"]) == {
        "os",
        "cpu",
        "cpu_cores",
        "ram_gb",
        "ram_bandwidth_gbps",
        "ram_bandwidth_source",
        "gpus",
    }


def test_report_saves_under_the_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))

    path = save_report(_report())

    assert path.parent == reports_dir() == tmp_path / "whichllm" / "reports"
    assert json.loads(path.read_text(encoding="utf-8"))["schema_version"] == 1


@pytest.fixture
def stub_command(monkeypatch, tmp_path):
    """The command with hardware and model lookup stubbed, and a recording backend."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    model, variant = _model()
    recorded = SimpleNamespace(llama_command=None, started=False)

    class FakeServer(backends.Backend):
        version = "b9999"

        def __init__(self, command):
            recorded.llama_command = command

        def start(self):
            recorded.started = True

        def generate(self, tokens):
            return Sample(50.0, tokens)

    monkeypatch.setattr(bench_command, "_detected_hardware", lambda _: _hardware())
    monkeypatch.setattr(bench_command, "_load_models", lambda _: [model])
    monkeypatch.setattr(
        bench_command,
        "_resolve_model",
        lambda *args, **kwargs: (model, variant, _pick(model, variant)),
    )
    monkeypatch.setattr(bench_command, "LlamaServer", FakeServer)
    monkeypatch.setattr(bench_command, "_gpu_memory_used_mb", lambda: 1000)
    monkeypatch.setattr(
        bench_command, "_verified_local_path", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        bench_command.shutil,
        "which",
        lambda name: "/usr/bin/llama-server" if name == "llama-server" else None,
    )
    recorded.tmp = tmp_path
    return recorded


def test_a_named_model_gets_the_placement_the_estimate_assumed():
    model, _ = _model()

    picked = bench_command._rank_named_model(
        [model], model, _hardware(), context_length=4096, quant=None
    )

    assert picked.placement.kind == "gpu"
    assert picked.estimated_tok_per_sec > 0


def test_a_named_official_model_is_estimated_on_its_upload():
    official = ModelInfo(
        id="Qwen/M-8B", family_id="m", name="M-8B", parameter_count=int(8e9)
    )
    upload = ModelInfo(
        id="unsloth/M-8B-GGUF",
        family_id="m",
        name="M-8B-GGUF",
        parameter_count=int(8e9),
        gguf_variants=[GGUFVariant("M-8B-Q4_K_M.gguf", "Q4_K_M", 4 * _GiB)],
        base_model="Qwen/M-8B",
        base_model_relation="quantized",
        tags=["base_model:quantized:Qwen/M-8B"],
    )

    picked = bench_command._rank_named_model(
        [official, upload], official, _hardware(), context_length=4096, quant="Q4_K_M"
    )

    assert picked.gguf_variant.file_size_bytes == 4 * _GiB
    assert not picked.gguf_variant.is_estimated
    assert picked.artifact_model is upload


def test_a_model_too_large_to_rank_runs_without_a_placement():
    model, _ = _model()
    tiny = _hardware()
    tiny.gpus = []
    tiny.ram_bytes = _GiB

    assert (
        bench_command._rank_named_model(
            [model], model, tiny, context_length=4096, quant=None
        )
        is None
    )


def test_bench_runs_the_recipe_and_saves_a_report(stub_command):
    result = CliRunner().invoke(app, ["bench", "--backend", "llama-server", "--yes"])

    assert result.exit_code == 0, result.output
    assert "Measured" in result.output
    assert "50.0 tok/s" in result.output
    assert "80.0 tok/s" in result.output
    assert stub_command.llama_command[0] == "/usr/bin/llama-server"
    assert "-hf" in stub_command.llama_command
    saved = list((stub_command.tmp / "whichllm" / "reports").glob("*.json"))
    assert len(saved) == 1


def test_bench_asks_before_a_download(stub_command):
    result = CliRunner().invoke(
        app, ["bench", "--backend", "llama-server"], input="n\n"
    )

    assert result.exit_code != 0
    assert "Model-Q4_K_M.gguf" in result.output
    assert stub_command.started is False
    assert not (stub_command.tmp / "whichllm" / "reports").exists()


def test_a_failed_runtime_saves_no_report(stub_command, monkeypatch):
    def fail(*args):
        raise BenchError("no GPU memory left")

    monkeypatch.setattr(bench_command, "_run_benchmark", fail)

    result = CliRunner().invoke(app, ["bench", "--backend", "llama-server", "--yes"])

    assert result.exit_code == 1
    assert "no GPU memory left" in result.output
    assert not (stub_command.tmp / "whichllm" / "reports").exists()


@pytest.mark.parametrize(
    "args",
    [
        ["bench", "--backend", "vulkan"],
        ["bench", "--backend", "openai"],
        ["bench", "--backend", "openai", "--url", "http://x/v1"],
        ["bench", "--runs", "0"],
        ["bench", "--tokens", "1"],
    ],
)
def test_bench_rejects_bad_options(args):
    assert CliRunner().invoke(app, args).exit_code == 2


def test_openai_backend_needs_no_catalog(fake_http, tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.setattr(bench_command, "_detected_hardware", lambda _: _hardware())
    monkeypatch.setattr(bench_command, "_gpu_memory_used_mb", lambda: None)
    server = fake_http({"/v1/completions": _stream(16)})

    result = CliRunner().invoke(
        app,
        [
            "bench",
            "--backend",
            "openai",
            "--url",
            f"{server.url}/v1",
            "--model",
            "served-model",
            "--engine",
            "vllm",
            "--tokens",
            "16",
        ],
    )

    assert result.exit_code == 0, result.output
    saved = next((tmp_path / "whichllm" / "reports").glob("*.json"))
    report = json.loads(saved.read_text(encoding="utf-8"))
    assert report["model"]["repo"] == "served-model"
    assert report["runtime"]["engine"] == "vllm"
    assert report["estimate"] is None
