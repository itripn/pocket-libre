"""The index is refreshed where summaries and names change, and `pocket-libre index`."""

import os
import time

import pytest
from click.testing import CliRunner

from pocket_libre import cli as cli_module
from pocket_libre import commands, voices
from tests.test_download import FULL, _FakeCommander
from tests.test_speakers import PRINTS, SEGMENTS, STEM, fake_diarize, make_recording
from tests.test_speakers_wiring import fake_transcribe

PH = "PH261020143032"


@pytest.fixture(autouse=True)
def pacific(monkeypatch):
    monkeypatch.setenv("TZ", "America/Los_Angeles")
    time.tzset()
    yield
    monkeypatch.undo()
    time.tzset()


@pytest.fixture
def summarizing(monkeypatch):
    """Summaries and titles come from stubs; no network, no user hook."""
    monkeypatch.setattr("pocket_libre.summarize.summarize_transcript", lambda **kw: "## Summary\nHi there.")
    monkeypatch.setattr("pocket_libre.summarize.make_title", lambda summary, key, *a, **k: "Budget sync")
    monkeypatch.setattr("pocket_libre.hooks.run_summary_hook", lambda path: None)
    monkeypatch.setattr(cli_module, "run_summary_hook", lambda path: None)
    monkeypatch.setattr("pocket_libre.diarize.diarize_auto", fake_diarize(PRINTS))


def test_watch_summary_has_title_heading_and_refreshes_index(tmp_path, monkeypatch, summarizing):
    from pocket_libre import watch

    monkeypatch.setattr("pocket_libre.transcribe.transcribe_file", fake_transcribe)
    monkeypatch.setattr("pocket_libre.config.load_config", lambda: {})
    folder = tmp_path / "2026-10-20"
    folder.mkdir()
    audio = folder / f"{PH}.mp3"
    audio.write_bytes(b"x")
    watch._process_recording(audio, PH, folder, "base.en", "meeting", "key", None)
    summary = (folder / f"{PH}_summary.md").read_text()
    assert summary.startswith("# Budget sync (2026-10-20 07:30 -07:00)\n")
    index = (tmp_path / "index.md").read_text()
    assert "Budget sync" in index and "07:30" in index


def test_watch_without_summary_leaves_no_index_behind(tmp_path, monkeypatch, summarizing):
    from pocket_libre import watch

    monkeypatch.setattr("pocket_libre.transcribe.transcribe_file", fake_transcribe)
    monkeypatch.setattr("pocket_libre.config.load_config", lambda: {})
    folder = tmp_path / "2026-10-20"
    folder.mkdir()
    audio = folder / f"{PH}.mp3"
    audio.write_bytes(b"x")
    watch._process_recording(audio, PH, folder, "base.en", "meeting", None, None)
    assert not (tmp_path / "index.md").exists()


def test_sync_refreshes_index(tmp_path, monkeypatch, summarizing):
    rec = commands.Recording("2026-10-20", PH, 0)

    class _Lister(_FakeCommander):
        async def list_all_recordings(self):
            return [rec]

    async def download(address, key, rec, progress_callback=None):
        return FULL, True

    monkeypatch.setattr(cli_module, "PocketCommander", _Lister)
    monkeypatch.setattr(commands, "download_checked", download)
    monkeypatch.setattr(cli_module, "transcribe_file", fake_transcribe)
    monkeypatch.setattr(cli_module, "load_config", lambda: {})
    result = CliRunner().invoke(cli_module.cli, [
        "sync", "--address", "addr", "--key", "k" * 16, "--anthropic-key", "key",
        "--output-dir", str(tmp_path),
    ])
    assert result.exit_code == 0, result.output
    assert "Budget sync" in (tmp_path / "index.md").read_text()


def test_process_refreshes_index_when_file_is_in_the_output_dir(tmp_path, monkeypatch, summarizing):
    out = tmp_path / "out"
    (out / "2026-10-20").mkdir(parents=True)
    audio = out / "2026-10-20" / f"{PH}.mp3"
    audio.write_bytes(b"x")
    monkeypatch.setattr(cli_module, "transcribe_file", fake_transcribe)
    monkeypatch.setattr(cli_module, "load_config", lambda: {"output": {"directory": str(out)}})
    result = CliRunner().invoke(cli_module.cli, [
        "process", "--input", str(audio), "--anthropic-key", "key",
    ])
    assert result.exit_code == 0, result.output
    assert "Budget sync" in (out / "index.md").read_text()


def test_process_elsewhere_does_not_drop_an_index_in_that_folder(tmp_path, monkeypatch, summarizing):
    elsewhere = tmp_path / "Downloads"
    elsewhere.mkdir()
    audio = elsewhere / f"{PH}.mp3"
    audio.write_bytes(b"x")
    monkeypatch.setattr(cli_module, "transcribe_file", fake_transcribe)
    monkeypatch.setattr(cli_module, "load_config",
                        lambda: {"output": {"directory": str(tmp_path / "out")}})
    result = CliRunner().invoke(cli_module.cli, [
        "process", "--input", str(audio), "--anthropic-key", "key",
    ])
    assert result.exit_code == 0, result.output
    assert list(tmp_path.rglob("index.md")) == []


def test_speakers_fix_refreshes_index(tmp_path, monkeypatch):
    make_recording(tmp_path / "out")
    monkeypatch.setattr(cli_module, "load_config",
                        lambda: {"output": {"directory": str(tmp_path / "out")}})
    monkeypatch.setattr(voices, "library_path", lambda: tmp_path / "home" / "voices.json")
    result = CliRunner().invoke(cli_module.cli, ["speakers", "fix", STEM, "Speaker 2", "Bob"])
    assert result.exit_code == 0, result.output
    assert "Bob" in (tmp_path / "out" / "index.md").read_text()


# ── `pocket-libre index` ──

def write_old_summary(folder, stem, body="We talked."):
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{stem}_summary.md"
    path.write_text(f"# {stem} (2026-10-20 11:03 -07:00)\n\n## Summary\n{body}\n\n---\n\n"
                    "## Full Transcript\n\n[00:05] Ron: hi\n", encoding="utf-8")
    return path


def invoke_index(monkeypatch, tmp_path, *args, env_key=None):
    monkeypatch.setattr(cli_module, "load_config",
                        lambda: {"output": {"directory": str(tmp_path / "out")}})
    if env_key:
        monkeypatch.setenv("ANTHROPIC_API_KEY", env_key)
    else:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    return CliRunner().invoke(cli_module.cli, ["index", *args])


def test_index_command_builds_index_without_touching_recordings(tmp_path, monkeypatch):
    s = write_old_summary(tmp_path / "out" / "2026-10-20", PH)
    before = (s.read_bytes(), s.stat().st_mtime_ns)
    result = invoke_index(monkeypatch, tmp_path)
    assert result.exit_code == 0, result.output
    assert (tmp_path / "out" / "index.md").exists()
    assert (s.read_bytes(), s.stat().st_mtime_ns) == before


def test_index_command_missing_output_dir_says_so(tmp_path, monkeypatch):
    result = invoke_index(monkeypatch, tmp_path)
    assert result.exit_code == 1
    assert "not found" in result.output.lower() or "no such" in result.output.lower()


def test_index_titles_adds_titles_keeping_body_and_mtime(tmp_path, monkeypatch):
    folder = tmp_path / "out" / "2026-10-20"
    missing = write_old_summary(folder, PH, body="We talked about Sifter.")
    titled = folder / "PH261020150132_summary.md"
    titled.write_text("# Already titled (2026-10-20 11:03 -07:00)\n\n## Summary\nx\n", encoding="utf-8")
    old = (1_700_000_000_000_000_000, 1_700_000_100_000_000_000)
    os.utime(missing, ns=old)
    seen = []
    monkeypatch.setattr("pocket_libre.summarize.make_title",
                        lambda summary, key, *a, **k: seen.append(summary) or "Sifter scaling")
    result = invoke_index(monkeypatch, tmp_path, "--titles", "--yes", env_key="key")
    assert result.exit_code == 0, result.output
    text = missing.read_text()
    assert text.startswith("# Sifter scaling (2026-10-20 11:03 -07:00)\n")
    assert PH in text.split("## Summary")[0]          # the id survives in a facts line
    assert "We talked about Sifter." in text and text.endswith("[00:05] Ron: hi\n")
    assert missing.stat().st_mtime_ns == old[1]
    assert titled.read_text().startswith("# Already titled")
    assert len(seen) == 1 and "Full Transcript" not in seen[0]
    assert "Sifter scaling" in (tmp_path / "out" / "index.md").read_text()


def test_index_titles_needs_an_api_key(tmp_path, monkeypatch):
    s = write_old_summary(tmp_path / "out" / "2026-10-20", PH)
    before = s.read_bytes()
    result = invoke_index(monkeypatch, tmp_path, "--titles", "--yes")
    assert result.exit_code == 1
    assert "key" in result.output.lower()
    assert s.read_bytes() == before


def test_index_titles_declined_changes_nothing(tmp_path, monkeypatch):
    s = write_old_summary(tmp_path / "out" / "2026-10-20", PH)
    before = s.read_bytes()
    monkeypatch.setattr("pocket_libre.summarize.make_title", lambda *a, **k: "T")
    monkeypatch.setattr(cli_module, "load_config",
                        lambda: {"output": {"directory": str(tmp_path / "out")}})
    monkeypatch.setenv("ANTHROPIC_API_KEY", "key")
    result = CliRunner().invoke(cli_module.cli, ["index", "--titles"], input="n\n")
    assert result.exit_code == 0
    assert s.read_bytes() == before


def test_index_titles_skips_when_the_model_gives_no_title(tmp_path, monkeypatch):
    s = write_old_summary(tmp_path / "out" / "2026-10-20", PH)
    before = s.read_bytes()
    monkeypatch.setattr("pocket_libre.summarize.make_title", lambda *a, **k: "")
    result = invoke_index(monkeypatch, tmp_path, "--titles", "--yes", env_key="key")
    assert result.exit_code == 0, result.output
    assert s.read_bytes() == before
