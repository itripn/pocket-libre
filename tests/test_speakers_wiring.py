"""Each place that transcribes a recording leaves a speaker sidecar next to the transcript."""

from click.testing import CliRunner

from pocket_libre import cli as cli_module
from pocket_libre import commands, speakers
from pocket_libre.commands import Recording
from tests.test_download import FULL, _FakeCommander
from tests.test_speakers import PRINTS, SEGMENTS, fake_diarize

REC = Recording("2026-10-03", "20261003090502", 0)


def fake_transcribe(audio_path, model_name, backend=None):
    return {"segments": SEGMENTS}


def assert_sidecar(folder, stem):
    side = speakers.sidecar_path(folder, stem)
    assert side.exists()
    assert set(speakers.read_sidecar(side)["speakers"]) == {"Speaker 1", "Speaker 2"}
    assert (folder / f"{stem}_transcript.txt").exists()


def test_process_writes_sidecar_next_to_transcript(tmp_path, monkeypatch):
    monkeypatch.setattr("pocket_libre.diarize.diarize_auto", fake_diarize(PRINTS))
    monkeypatch.setattr(cli_module, "transcribe_file", fake_transcribe)
    monkeypatch.setattr(cli_module, "load_config", lambda: {})
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    audio = tmp_path / "rec.mp3"
    audio.write_bytes(b"x")
    result = CliRunner().invoke(cli_module.cli, [
        "process", "--input", str(audio), "--skip-summary",
    ])
    assert result.exit_code == 0, result.output
    assert_sidecar(tmp_path, "rec")


def test_sync_writes_sidecar_next_to_transcript(tmp_path, monkeypatch):
    class _Lister(_FakeCommander):
        async def list_all_recordings(self):
            return [REC]

    async def download(address, key, rec, progress_callback=None):
        return FULL, True

    monkeypatch.setattr(cli_module, "PocketCommander", _Lister)
    monkeypatch.setattr(commands, "download_checked", download)
    monkeypatch.setattr(cli_module, "transcribe_file", fake_transcribe)
    monkeypatch.setattr("pocket_libre.diarize.diarize_auto", fake_diarize(PRINTS))
    monkeypatch.setattr(cli_module, "load_config", lambda: {})
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    result = CliRunner().invoke(cli_module.cli, [
        "sync", "--address", "addr", "--key", "k" * 16,
        "--output-dir", str(tmp_path),
    ])
    assert result.exit_code == 0, result.output
    assert_sidecar(tmp_path / REC.date, REC.timestamp)


def test_watch_writes_sidecar_next_to_transcript(tmp_path, monkeypatch):
    from pocket_libre import watch

    monkeypatch.setattr("pocket_libre.transcribe.transcribe_file", fake_transcribe)
    monkeypatch.setattr("pocket_libre.diarize.diarize_auto", fake_diarize(PRINTS))
    monkeypatch.setattr("pocket_libre.config.load_config", lambda: {})
    audio = tmp_path / "rec.mp3"
    audio.write_bytes(b"x")
    watch._process_recording(audio, "rec", tmp_path, "base.en", "meeting", None, None)
    assert_sidecar(tmp_path, "rec")
