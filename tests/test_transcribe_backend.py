"""Choosing between openai-whisper (CPU) and mlx-whisper (Apple GPU), and using the choice."""

import sys
import types

import pytest

from pocket_libre import config as cfg
from pocket_libre import transcribe as tr

RESULT = {"text": " Hello there.", "language": "en",
          "segments": [{"start": 0.0, "end": 1.5, "text": " Hello there."}]}


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    """No ambient env var or real ~/.pocket-libre/config.toml."""
    monkeypatch.delenv("POCKET_LIBRE_WHISPER_BACKEND", raising=False)
    monkeypatch.setattr(cfg, "CONFIG_DIR", tmp_path / ".pocket-libre")
    monkeypatch.setattr(cfg, "CONFIG_FILE", tmp_path / ".pocket-libre" / "config.toml")


def machine(monkeypatch, *, apple_silicon, mlx_installed):
    monkeypatch.setattr(tr, "_is_apple_silicon", lambda: apple_silicon)
    monkeypatch.setattr(tr, "_have_mlx", lambda: mlx_installed)


# ── Choosing a backend ──────────────────────────


@pytest.mark.parametrize(
    "apple_silicon,mlx_installed,expected",
    [(True, True, "mlx"), (True, False, "whisper"), (False, True, "whisper"), (False, False, "whisper")],
)
def test_auto_uses_mlx_only_where_it_can_run(monkeypatch, apple_silicon, mlx_installed, expected):
    machine(monkeypatch, apple_silicon=apple_silicon, mlx_installed=mlx_installed)
    assert tr.resolve_backend() == expected


def test_default_is_auto(monkeypatch):
    machine(monkeypatch, apple_silicon=True, mlx_installed=True)
    assert tr.resolve_backend(None) == "mlx"


def test_whisper_can_be_forced_even_when_mlx_is_available(monkeypatch):
    machine(monkeypatch, apple_silicon=True, mlx_installed=True)
    assert tr.resolve_backend("whisper") == "whisper"


def test_forcing_mlx_without_it_installed_says_how_to_install(monkeypatch):
    machine(monkeypatch, apple_silicon=True, mlx_installed=False)
    with pytest.raises(RuntimeError, match=r"\.\[mlx\]"):
        tr.resolve_backend("mlx")


def test_forcing_mlx_off_apple_silicon_is_refused(monkeypatch):
    machine(monkeypatch, apple_silicon=False, mlx_installed=True)
    with pytest.raises(RuntimeError, match="Apple Silicon"):
        tr.resolve_backend("mlx")


def test_unknown_backend_is_rejected(monkeypatch):
    machine(monkeypatch, apple_silicon=True, mlx_installed=True)
    with pytest.raises(ValueError, match="gpu"):
        tr.resolve_backend("gpu")


def test_env_var_picks_the_backend(monkeypatch):
    machine(monkeypatch, apple_silicon=True, mlx_installed=True)
    monkeypatch.setenv("POCKET_LIBRE_WHISPER_BACKEND", "whisper")
    assert tr.resolve_backend() == "whisper"


def test_config_file_picks_the_backend(monkeypatch):
    machine(monkeypatch, apple_silicon=True, mlx_installed=True)
    cfg.save_config({"defaults": {"whisper_backend": "whisper"}})
    assert tr.resolve_backend() == "whisper"


def test_env_var_beats_the_config_file(monkeypatch):
    machine(monkeypatch, apple_silicon=True, mlx_installed=True)
    cfg.save_config({"defaults": {"whisper_backend": "whisper"}})
    monkeypatch.setenv("POCKET_LIBRE_WHISPER_BACKEND", "mlx")
    assert tr.resolve_backend() == "mlx"


# ── Model names ─────────────────────────────────


@pytest.mark.parametrize(
    "name,repo",
    [
        ("tiny.en", "mlx-community/whisper-tiny.en-mlx"),
        ("base.en", "mlx-community/whisper-base.en-mlx"),
        ("small.en", "mlx-community/whisper-small.en-mlx"),
        ("medium.en", "mlx-community/whisper-medium.en-mlx"),
        ("large", "mlx-community/whisper-large-v3-mlx"),
    ],
)
def test_every_documented_model_has_an_mlx_repo(name, repo):
    assert tr.mlx_repo(name) == repo


def test_a_huggingface_repo_name_passes_through():
    assert tr.mlx_repo("someone/whisper-custom-mlx") == "someone/whisper-custom-mlx"


def test_an_unknown_model_has_no_mlx_repo():
    assert tr.mlx_repo("small") is None


def test_auto_falls_back_to_whisper_for_a_model_mlx_cannot_load(monkeypatch):
    machine(monkeypatch, apple_silicon=True, mlx_installed=True)
    assert tr.resolve_backend(None, model_name="small") == "whisper"


def test_forced_mlx_with_an_unknown_model_is_an_error(monkeypatch):
    machine(monkeypatch, apple_silicon=True, mlx_installed=True)
    with pytest.raises(ValueError, match="small"):
        tr.resolve_backend("mlx", model_name="small")


# ── Running it ──────────────────────────────────


def _fake_modules(monkeypatch):
    """Fake mlx_whisper and whisper that record how they were called."""
    calls = {"mlx": [], "whisper_load": [], "whisper_run": []}

    mlx = types.ModuleType("mlx_whisper")

    def mlx_transcribe(path, **kwargs):
        calls["mlx"].append((path, kwargs))
        return dict(RESULT)

    mlx.transcribe = mlx_transcribe
    monkeypatch.setitem(sys.modules, "mlx_whisper", mlx)

    class _Model:
        def transcribe(self, path, **kwargs):
            calls["whisper_run"].append((path, kwargs))
            return dict(RESULT)

    whisper = types.ModuleType("whisper")

    def load_model(name):
        calls["whisper_load"].append(name)
        return _Model()

    whisper.load_model = load_model
    monkeypatch.setitem(sys.modules, "whisper", whisper)
    return calls


def test_mlx_backend_hands_the_file_and_repo_to_mlx_whisper(monkeypatch):
    machine(monkeypatch, apple_silicon=True, mlx_installed=True)
    calls = _fake_modules(monkeypatch)
    result = tr.transcribe_file("/a/b.mp3", "small.en")
    assert calls["mlx"] == [("/a/b.mp3", {"path_or_hf_repo": "mlx-community/whisper-small.en-mlx"})]
    assert calls["whisper_load"] == []
    assert result["segments"][0]["text"] == " Hello there."


def test_whisper_backend_loads_and_runs_the_openai_model(monkeypatch):
    machine(monkeypatch, apple_silicon=False, mlx_installed=False)
    calls = _fake_modules(monkeypatch)
    result = tr.transcribe_file("/a/b.mp3", "small.en")
    assert calls["whisper_load"] == ["small.en"]
    assert calls["whisper_run"] == [("/a/b.mp3", {"verbose": False})]
    assert calls["mlx"] == []
    assert result["language"] == "en"


def test_result_always_has_text_segments_and_language(monkeypatch):
    machine(monkeypatch, apple_silicon=True, mlx_installed=True)
    mlx = types.ModuleType("mlx_whisper")
    mlx.transcribe = lambda path, **kw: {"text": "hi"}
    monkeypatch.setitem(sys.modules, "mlx_whisper", mlx)
    result = tr.transcribe_file("/a/b.mp3", "small.en")
    assert result["segments"] == [] and result["language"] == "unknown"


# ── Call sites use the helper ───────────────────


@pytest.fixture
def mlx_machine(monkeypatch):
    machine(monkeypatch, apple_silicon=True, mlx_installed=True)
    calls = _fake_modules(monkeypatch)

    from pocket_libre import diarize, summarize

    monkeypatch.setattr(diarize, "diarize_auto",
                        lambda segments, **kw: diarize.diarize_simple(segments))
    monkeypatch.setattr(summarize, "summarize_transcript", lambda **kw: "Summary")
    return calls


def test_watch_processing_transcribes_with_the_chosen_backend(mlx_machine, tmp_path):
    from pocket_libre.watch import _process_recording

    audio = tmp_path / "PH1.mp3"
    audio.write_bytes(b"\x00")
    _process_recording(audio, "PH1", tmp_path, "small.en", "meeting", None, None)
    assert len(mlx_machine["mlx"]) == 1 and mlx_machine["whisper_load"] == []
    assert "Hello there." in (tmp_path / "PH1_transcript.txt").read_text()


def test_process_command_transcribes_with_the_chosen_backend(mlx_machine, tmp_path, monkeypatch):
    from click.testing import CliRunner

    from pocket_libre import cli

    audio = tmp_path / "PH1.mp3"
    audio.write_bytes(b"\x00")
    monkeypatch.setattr(cli, "load_config", lambda: {})
    result = CliRunner().invoke(cli.cli, ["process", "--input", str(audio), "--skip-summary"])
    assert result.exit_code == 0, result.output
    assert len(mlx_machine["mlx"]) == 1 and mlx_machine["whisper_load"] == []


def test_sync_command_transcribes_with_the_chosen_backend(mlx_machine, tmp_path, monkeypatch):
    from click.testing import CliRunner

    from pocket_libre import cli, commands
    from tests.test_download import FULL, REC, _FakeCommander

    class _Lister(_FakeCommander):
        async def list_all_recordings(self):
            return [REC]

    async def download(address, key, rec, progress_callback=None):
        return FULL, True

    monkeypatch.setattr(cli, "PocketCommander", _Lister)
    monkeypatch.setattr(commands, "download_checked", download)
    monkeypatch.setattr(cli, "load_config", lambda: {})
    result = CliRunner().invoke(cli.cli, [
        "sync", "--address", "addr", "--key", "k" * 16, "--output-dir", str(tmp_path),
    ])
    assert result.exit_code == 0, result.output
    assert len(mlx_machine["mlx"]) == 1 and mlx_machine["whisper_load"] == []


def test_transcribe_command_transcribes_with_the_chosen_backend(mlx_machine, tmp_path):
    audio = tmp_path / "PH1.mp3"
    audio.write_bytes(b"\x00")
    out = tmp_path / "out.txt"
    tr.transcribe_audio(str(audio), "small.en", output_path=str(out))
    assert len(mlx_machine["mlx"]) == 1 and mlx_machine["whisper_load"] == []
    assert out.read_text() == "Hello there."
