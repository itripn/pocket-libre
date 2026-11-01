"""CLI tests for `pocket-libre speakers fix`."""

import json

from click.testing import CliRunner

from pocket_libre import cli as cli_module
from pocket_libre import speakers, voices
from tests.test_speakers import STEM, make_recording


def run(monkeypatch, tmp_path, *args):
    monkeypatch.setattr(cli_module, "load_config",
                        lambda: {"output": {"directory": str(tmp_path / "out")}})
    monkeypatch.setattr(voices, "library_path", lambda: tmp_path / "home" / "voices.json")
    return CliRunner().invoke(cli_module.cli, ["speakers", "fix", *args])


def test_fix_success(monkeypatch, tmp_path):
    side = make_recording(tmp_path / "out")
    result = run(monkeypatch, tmp_path, STEM, "Speaker 2", "Bob")
    assert result.exit_code == 0, result.output
    assert "Bob" in result.output
    assert speakers.read_sidecar(side)["speakers"]["Speaker 2"]["name"] == "Bob"


def test_fix_unknown_recording_exits_1(monkeypatch, tmp_path):
    (tmp_path / "out").mkdir()
    result = run(monkeypatch, tmp_path, "NOPE", "Speaker 2", "Bob")
    assert result.exit_code == 1
    assert "nothing matched" in result.output


def test_fix_bad_speaker_exits_1(monkeypatch, tmp_path):
    make_recording(tmp_path / "out")
    result = run(monkeypatch, tmp_path, STEM, "Speaker 9", "Bob")
    assert result.exit_code == 1
    assert "Speaker 1" in result.output


# ── list / forget / scores ──

def invoke(monkeypatch, tmp_path, *args, config=None, input=None):
    monkeypatch.setattr(cli_module, "load_config", lambda: {
        "output": {"directory": str(tmp_path / "out")}, **(config or {})})
    monkeypatch.setattr(voices, "library_path", lambda: tmp_path / "home" / "voices.json")
    return CliRunner().invoke(cli_module.cli, ["speakers", *args], input=input)


def sample(embedding, date="2026-10-01"):
    return {"embedding": embedding, "model": "pyannote/wespeaker", "source": "s",
            "label": "Speaker 1", "date": date, "seconds": 30.0}


def write_library(tmp_path):
    path = tmp_path / "home" / "voices.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"people": {
        "Ron": {"samples": [sample([1.0, 0.0], "2026-10-01"),
                            sample([1.0, 0.0], "2026-10-15")]},
        "Ann": {"samples": [sample([0.0, 1.0], "2026-09-30")]},
    }}))
    return path


def test_list_empty_and_populated(monkeypatch, tmp_path):
    result = invoke(monkeypatch, tmp_path, "list")
    assert result.exit_code == 0, result.output
    assert "No known voices" in result.output

    write_library(tmp_path)
    result = invoke(monkeypatch, tmp_path, "list")
    assert result.exit_code == 0, result.output
    assert "Ron" in result.output and "Ann" in result.output
    assert "2026-10-15" in result.output
    assert "0.0" not in result.output  # no embeddings


def test_list_corrupt_library_exits_1(monkeypatch, tmp_path):
    path = tmp_path / "home" / "voices.json"
    path.parent.mkdir(parents=True)
    path.write_text("{nope")
    result = invoke(monkeypatch, tmp_path, "list")
    assert result.exit_code == 1
    assert "voices.json" in result.output


def test_forget_requires_confirmation_and_deletes_samples_from_file(monkeypatch, tmp_path):
    path = write_library(tmp_path)
    before = path.read_bytes()
    result = invoke(monkeypatch, tmp_path, "forget", "Ron", input="n\n")
    assert result.exit_code == 0, result.output
    assert "2 voice sample" in result.output
    assert "Cancelled." in result.output
    assert path.read_bytes() == before

    result = invoke(monkeypatch, tmp_path, "forget", "Ron", input="y\n")
    assert result.exit_code == 0, result.output
    people = json.loads(path.read_text())["people"]
    assert "Ron" not in people and "Ann" in people


def test_forget_yes_flag(monkeypatch, tmp_path):
    path = write_library(tmp_path)
    result = invoke(monkeypatch, tmp_path, "forget", "--yes", "Ann")
    assert result.exit_code == 0, result.output
    assert list(json.loads(path.read_text())["people"]) == ["Ron"]


def test_forget_unknown_name_exits_1(monkeypatch, tmp_path):
    path = write_library(tmp_path)
    before = path.read_bytes()
    result = invoke(monkeypatch, tmp_path, "forget", "-y", "ron")
    assert result.exit_code == 1
    assert "ron" in result.output
    assert path.read_bytes() == before


def test_forget_unreadable_library_refuses(monkeypatch, tmp_path):
    path = tmp_path / "home" / "voices.json"
    path.parent.mkdir(parents=True)
    path.write_text("{nope")
    result = invoke(monkeypatch, tmp_path, "forget", "-y", "Ron")
    assert result.exit_code == 1
    assert path.read_text() == "{nope"


def test_scores_shows_scores_and_would_name_without_modifying_files(monkeypatch, tmp_path):
    side = make_recording(tmp_path / "out")
    lib = write_library(tmp_path)
    before = (lib.read_bytes(), side.read_bytes())
    result = invoke(monkeypatch, tmp_path, "scores", STEM,
                    config={"voices": {"min_seconds": 15}})
    assert result.exit_code == 0, result.output
    out = result.output
    assert "threshold" in out and "margin" in out and "min_seconds" in out
    assert "Ron" in out and "Ann" in out
    assert "1.00" in out  # Speaker 1 vs Ron
    # Speaker 1 clearly Ron; Speaker 2 clearly Ann; Speaker 20 is 12s < 15s
    rows = {line.split("│")[1].strip(): line.split("│")[-2].strip()
            for line in out.splitlines() if line.count("│") >= 4}
    assert rows["Speaker 1"] == "Ron"
    assert rows["Speaker 2"] == "Ann"
    assert rows["Speaker 20"] == "-"
    assert "12s" in out
    assert (lib.read_bytes(), side.read_bytes()) == before


def test_scores_empty_library_hints_at_fix(monkeypatch, tmp_path):
    make_recording(tmp_path / "out")
    result = invoke(monkeypatch, tmp_path, "scores", STEM)
    assert result.exit_code == 0, result.output
    assert "speakers fix" in result.output


def test_scores_unknown_recording_exits_1(monkeypatch, tmp_path):
    (tmp_path / "out").mkdir()
    result = invoke(monkeypatch, tmp_path, "scores", "NOPE")
    assert result.exit_code == 1


def test_list_and_forget_show_names_with_rich_markup_literally(monkeypatch, tmp_path):
    path = tmp_path / "home" / "voices.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"people": {
        "[bold]x[/bold]": {"samples": [sample([1.0, 0.0], "[red]2026[/red]")]},
    }}))
    result = invoke(monkeypatch, tmp_path, "list")
    assert result.exit_code == 0, result.output
    assert "[bold]x[/bold]" in result.output
    assert "[red]2026[/red]" in result.output

    result = invoke(monkeypatch, tmp_path, "forget", "-y", "[bold]y[/bold]")
    assert result.exit_code == 1
    assert "'[bold]y[/bold]'" in result.output
