"""The on_summary hook: resolution, invocation, and failure tolerance."""

import stat

import pytest

from pocket_libre import config as cfg
from pocket_libre.hooks import run_summary_hook


@pytest.fixture(autouse=True)
def no_ambient_hook(monkeypatch, tmp_path):
    """Keep the real ~/.pocket-libre/config.toml and shell env out of every test."""
    monkeypatch.delenv("POCKET_LIBRE_ON_SUMMARY", raising=False)
    monkeypatch.setattr(cfg, "CONFIG_DIR", tmp_path / ".pocket-libre")
    monkeypatch.setattr(cfg, "CONFIG_FILE", tmp_path / ".pocket-libre" / "config.toml")


@pytest.fixture
def recorder(tmp_path):
    """An executable that appends each argv entry to a log, one per line."""
    log = tmp_path / "calls.log"
    script = tmp_path / "record-args"
    script.write_text(
        '#!/bin/sh\nfor a in "$@"; do printf "%s\\n" "$a" >> "' + str(log) + '"; done\n'
    )
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    return script, log


# ── Resolution ──────────────────────────────────


def test_hook_resolves_from_config_file():
    conf = {"hooks": {"on_summary": "/bin/true"}}
    assert cfg.resolve_summary_hook(conf) == "/bin/true"


def test_env_wins_over_config_file_for_hook(monkeypatch):
    monkeypatch.setenv("POCKET_LIBRE_ON_SUMMARY", "from-env")
    conf = {"hooks": {"on_summary": "from-file"}}
    assert cfg.resolve_summary_hook(conf) == "from-env"


def test_hook_is_empty_when_unconfigured():
    assert not cfg.resolve_summary_hook({})


# ── Invocation ──────────────────────────────────


def test_no_hook_configured_is_a_noop(tmp_path):
    assert run_summary_hook(tmp_path / "a_summary.md") is False


def test_hook_receives_the_summary_path(recorder, tmp_path):
    script, log = recorder
    summary = tmp_path / "a_summary.md"
    assert run_summary_hook(summary, command=str(script)) is True
    assert log.read_text().splitlines() == [str(summary)]


def test_path_with_spaces_arrives_as_one_argument(recorder, tmp_path):
    """The real output directory is '~/Pocket Libre' — it contains a space."""
    script, log = recorder
    spaced = tmp_path / "Pocket Libre" / "a_summary.md"
    assert run_summary_hook(spaced, command=str(script)) is True
    assert log.read_text().splitlines() == [str(spaced)]


def test_command_may_carry_its_own_arguments(recorder, tmp_path):
    script, log = recorder
    summary = tmp_path / "a_summary.md"
    run_summary_hook(summary, command=f"{script} --quiet")
    assert log.read_text().splitlines() == ["--quiet", str(summary)]


def test_leading_tilde_in_command_is_expanded(recorder, tmp_path, monkeypatch):
    script, log = recorder
    monkeypatch.setenv("HOME", str(script.parent))
    summary = tmp_path / "a_summary.md"
    assert run_summary_hook(summary, command="~/record-args") is True
    assert log.read_text().splitlines() == [str(summary)]


def test_hook_comes_from_config_when_no_command_given(recorder, tmp_path):
    script, log = recorder
    cfg.save_config({"hooks": {"on_summary": str(script)}})
    summary = tmp_path / "a_summary.md"
    assert run_summary_hook(summary) is True
    assert log.read_text().splitlines() == [str(summary)]


# ── Failure tolerance ───────────────────────────


def test_nonzero_exit_does_not_raise(tmp_path):
    assert run_summary_hook(tmp_path / "a_summary.md", command="/usr/bin/false") is False


def test_missing_command_does_not_raise(tmp_path):
    missing = str(tmp_path / "no-such-command")
    assert run_summary_hook(tmp_path / "a_summary.md", command=missing) is False


def test_unparseable_command_does_not_raise(tmp_path):
    assert run_summary_hook(tmp_path / "a_summary.md", command='broken "quote') is False


def test_hanging_command_is_cut_off(tmp_path):
    assert run_summary_hook(
        tmp_path / "a_summary.md", command="/bin/sleep 5", timeout=0.2,
    ) is False


# ── Call sites: every path that writes a summary fires the hook ─────────


@pytest.fixture
def fake_pipeline(monkeypatch):
    """Stand-ins for the heavy stages (Whisper, diarization, the Claude call)."""
    import sys
    import types

    class _Model:
        def transcribe(self, path, verbose=False):
            return {"segments": [{"start": 0.0, "end": 1.0, "text": "Hello"}]}

    whisper = types.ModuleType("whisper")
    whisper.load_model = lambda name: _Model()
    monkeypatch.setitem(sys.modules, "whisper", whisper)

    from pocket_libre import diarize, summarize

    monkeypatch.setattr(diarize, "diarize_auto",
                        lambda segments, **kw: diarize.diarize_simple(segments))
    monkeypatch.setattr(summarize, "summarize_transcript",
                        lambda **kw: "## Action Items\n- Me: ship it")


@pytest.fixture
def hook_env(recorder, monkeypatch):
    script, log = recorder
    monkeypatch.setenv("POCKET_LIBRE_ON_SUMMARY", str(script))
    return log


def test_process_command_fires_the_hook_with_the_summary_path(
        fake_pipeline, hook_env, tmp_path, monkeypatch):
    from click.testing import CliRunner

    from pocket_libre import cli

    audio = tmp_path / "PH1.mp3"
    audio.write_bytes(b"\x00")
    monkeypatch.setattr(cli, "load_config", lambda: {})

    result = CliRunner().invoke(cli.cli, [
        "process", "--input", str(audio), "--anthropic-key", "k",
    ])

    assert result.exit_code == 0, result.output
    assert hook_env.read_text().splitlines() == [str(tmp_path / "PH1_summary.md")]


def test_process_without_a_summary_does_not_fire_the_hook(
        fake_pipeline, hook_env, tmp_path, monkeypatch):
    from click.testing import CliRunner

    from pocket_libre import cli

    audio = tmp_path / "PH1.mp3"
    audio.write_bytes(b"\x00")
    monkeypatch.setattr(cli, "load_config", lambda: {})

    CliRunner().invoke(cli.cli, [
        "process", "--input", str(audio), "--anthropic-key", "k", "--skip-summary",
    ])

    assert not hook_env.exists()


def test_watch_processing_fires_the_hook_after_the_summary_is_on_disk(
        fake_pipeline, recorder, monkeypatch, tmp_path):
    """The hook must see a finished file, so it fires after the write."""
    from pocket_libre.watch import _process_recording

    script, _ = recorder
    seen = tmp_path / "seen.txt"
    checker = tmp_path / "check-exists"
    checker.write_text(f'#!/bin/sh\ncat "$1" > "{seen}"\n')
    checker.chmod(checker.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("POCKET_LIBRE_ON_SUMMARY", str(checker))

    rec_dir = tmp_path / "2026-10-20"
    rec_dir.mkdir()
    audio = rec_dir / "PH2.mp3"
    audio.write_bytes(b"\x00")

    _process_recording(audio, "PH2", rec_dir, "base.en", "meeting", "k", None)

    assert "ship it" in seen.read_text()


def test_sync_command_fires_the_hook_for_each_summary(
        fake_pipeline, hook_env, tmp_path, monkeypatch):
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
        "sync", "--address", "addr", "--key", "k" * 16,
        "--output-dir", str(tmp_path), "--anthropic-key", "k",
    ])

    assert result.exit_code == 0, result.output
    expected = tmp_path / REC.date / f"{REC.timestamp}_summary.md"
    assert hook_env.read_text().splitlines() == [str(expected)]
