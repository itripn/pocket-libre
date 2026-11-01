"""Tests for the speaker sidecar and label_transcript."""

import json
import math
import os
import stat
from pathlib import Path

import pytest

from pocket_libre import speakers, voices
from pocket_libre.diarize import SpeakerSegment

MODEL = "pyannote/wespeaker"
SEGMENTS = [
    {"start": 0.0, "end": 5.0, "text": "hello there"},
    {"start": 5.0, "end": 10.0, "text": "hi back"},
]


def vec(score: float) -> list[float]:
    return [score, math.sqrt(1 - score * score)]


def fake_diarize(voiceprints=None):
    def _diarize(whisper_segments, audio_path=None, hf_token=None,
                 anthropic_key=None, voiceprints_out=None):
        if voiceprints is not None:
            voiceprints_out.update(voiceprints)
        return [
            SpeakerSegment(0.0, 5.0, "Speaker 1"),
            SpeakerSegment(5.0, 10.0, "Speaker 2"),
        ]
    return _diarize


PRINTS = {
    "model": MODEL,
    "speakers": {
        "Speaker 1": {"embedding": [1.0, 0.0], "seconds": 30.0},
        "Speaker 2": {"embedding": [0.0, 1.0], "seconds": 30.0},
    },
}


@pytest.fixture
def lib_path(tmp_path, monkeypatch):
    path = tmp_path / "cfg" / "voices.json"
    monkeypatch.setattr(voices, "library_path", lambda: path)
    return path


def write_lib(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"people": {"Ron": {"samples": [{
        "embedding": [1.0, 0.0], "model": MODEL, "source": "old",
        "label": "Speaker 1", "date": "2026-10-01", "seconds": 30.0,
    }]}}}))


def run(monkeypatch, tmp_path, prints=PRINTS):
    monkeypatch.setattr("pocket_libre.diarize.diarize_auto", fake_diarize(prints))
    return speakers.label_transcript(
        SEGMENTS, "/a/rec.mp3", tmp_path, "rec", None, None, {},
    )


def test_label_transcript_names_matched_speaker_and_keeps_other_as_speaker_n(
        tmp_path, monkeypatch, lib_path):
    write_lib(lib_path)
    labeled = run(monkeypatch, tmp_path)
    assert [s["speaker"] for s in labeled] == ["Ron", "Speaker 2"]


def test_label_transcript_writes_private_sidecar(tmp_path, monkeypatch, lib_path):
    write_lib(lib_path)
    run(monkeypatch, tmp_path)
    path = speakers.sidecar_path(tmp_path, "rec")
    assert path == tmp_path / "rec_speakers.json"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    data = speakers.read_sidecar(path)
    assert data["version"] == 1
    assert data["model"] == MODEL
    assert data["audio"] == "/a/rec.mp3"
    assert data["speakers"]["Speaker 1"] == {
        "name": "Ron", "embedding": [1.0, 0.0], "seconds": 30.0}
    assert data["speakers"]["Speaker 2"]["name"] is None
    assert data["speakers"]["Speaker 2"]["embedding"] == [0.0, 1.0]


def test_label_transcript_without_library_changes_nothing(tmp_path, monkeypatch, lib_path):
    labeled = run(monkeypatch, tmp_path)
    assert [s["speaker"] for s in labeled] == ["Speaker 1", "Speaker 2"]
    assert not lib_path.exists()
    assert speakers.read_sidecar(speakers.sidecar_path(tmp_path, "rec"))["speakers"]


def test_label_transcript_survives_corrupt_library(tmp_path, monkeypatch, lib_path, capsys):
    lib_path.parent.mkdir(parents=True)
    lib_path.write_text("{not json")
    labeled = run(monkeypatch, tmp_path)
    assert [s["speaker"] for s in labeled] == ["Speaker 1", "Speaker 2"]
    assert speakers.sidecar_path(tmp_path, "rec").exists()
    assert "Voice library unreadable" in capsys.readouterr().out
    assert lib_path.read_text() == "{not json"


def test_label_transcript_survives_unexpected_library_failure(
        tmp_path, monkeypatch, lib_path, capsys):
    def boom():
        raise RuntimeError("kaboom")
    monkeypatch.setattr(voices, "load_library", boom)
    labeled = run(monkeypatch, tmp_path)
    assert [s["speaker"] for s in labeled] == ["Speaker 1", "Speaker 2"]
    assert speakers.sidecar_path(tmp_path, "rec").exists()
    assert "kaboom" in capsys.readouterr().out


def test_label_transcript_without_voiceprints_writes_no_sidecar(
        tmp_path, monkeypatch, lib_path):
    write_lib(lib_path)
    labeled = run(monkeypatch, tmp_path, prints=None)
    assert [s["speaker"] for s in labeled] == ["Speaker 1", "Speaker 2"]
    assert not speakers.sidecar_path(tmp_path, "rec").exists()


def test_label_transcript_skips_non_finite_embeddings_in_sidecar(
        tmp_path, monkeypatch, lib_path):
    prints = {"model": MODEL, "speakers": {
        "Speaker 1": {"embedding": [float("nan"), 0.0], "seconds": 30.0},
        "Speaker 2": {"embedding": [0.0, 1.0], "seconds": 30.0},
    }}
    run(monkeypatch, tmp_path, prints=prints)
    data = speakers.read_sidecar(speakers.sidecar_path(tmp_path, "rec"))
    assert list(data["speakers"]) == ["Speaker 2"]


def test_write_sidecar_leaves_no_temp_files(tmp_path):
    path = speakers.sidecar_path(tmp_path, "x")
    speakers.write_sidecar(path, {"version": 1})
    assert [p.name for p in tmp_path.iterdir()] == ["x_speakers.json"]


def test_label_transcript_survives_sidecar_write_failure(
        tmp_path, monkeypatch, lib_path, capsys):
    write_lib(lib_path)

    def boom(path, data):
        raise OSError("read-only file system")
    monkeypatch.setattr(speakers, "write_sidecar", boom)
    labeled = run(monkeypatch, tmp_path)
    assert [s["speaker"] for s in labeled] == ["Ron", "Speaker 2"]
    assert "read-only file system" in capsys.readouterr().out


def test_label_transcript_stores_absolute_audio_path(tmp_path, monkeypatch, lib_path):
    (tmp_path / "rec.mp3").write_bytes(b"x")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("pocket_libre.diarize.diarize_auto", fake_diarize(PRINTS))
    speakers.label_transcript(SEGMENTS, "rec.mp3", tmp_path, "rec", None, None, {})
    audio = speakers.read_sidecar(speakers.sidecar_path(tmp_path, "rec"))["audio"]
    assert Path(audio).is_absolute()
    assert Path(audio).samefile(tmp_path / "rec.mp3")


# ── find_recording / relabel / fix ──────────────────────────────

STEM = "PH261020213653"
SETTINGS = voices.VoiceSettings(threshold=0.7, margin=0.1, min_seconds=8.0)


def make_recording(root: Path, stem: str = STEM, day: str = "2026-10-20",
                   speakers_data: dict | None = None) -> Path:
    folder = root / day
    folder.mkdir(parents=True, exist_ok=True)
    data = {
        "version": 1, "model": MODEL, "audio": "/x.mp3",
        "speakers": speakers_data or {
            "Speaker 1": {"name": None, "embedding": [1.0, 0.0], "seconds": 30.0},
            "Speaker 2": {"name": None, "embedding": [0.0, 1.0], "seconds": 20.0},
            "Speaker 20": {"name": None, "embedding": [0.6, 0.8], "seconds": 12.0},
        },
    }
    side = speakers.sidecar_path(folder, stem)
    speakers.write_sidecar(side, data)
    (folder / f"{stem}_transcript.txt").write_text(
        "[00:01] Speaker 1: hi\n[00:05] Speaker 2: hello Speaker 1\n[00:09] Speaker 20: yo\n",
        encoding="utf-8")
    (folder / f"{stem}_summary.md").write_text(
        "Speaker 2 and Speaker 20 spoke.\n", encoding="utf-8")
    return side


@pytest.fixture
def lib_home(tmp_path, monkeypatch):
    monkeypatch.setattr(voices, "library_path", lambda: tmp_path / "home" / "voices.json")
    return tmp_path


def test_find_recording_by_id_path_and_latest(tmp_path):
    side = make_recording(tmp_path)
    other = make_recording(tmp_path, stem="PH999", day="2026-10-21")
    os.utime(side, (1, 1))
    assert speakers.find_recording(tmp_path, STEM) == side
    folder = side.parent
    for name in (f"{STEM}_summary.md", f"{STEM}_transcript.txt", f"{STEM}_speakers.json"):
        assert speakers.find_recording(tmp_path, str(folder / name)) == side
    assert speakers.find_recording(tmp_path, "latest") == other


def test_find_recording_unknown_lists_nothing_matched(tmp_path):
    make_recording(tmp_path)
    with pytest.raises(LookupError, match="nothing matched"):
        speakers.find_recording(tmp_path, "NOPE")
    with pytest.raises(LookupError, match="pocket-libre process"):
        speakers.find_recording(tmp_path, str(tmp_path / "2026-10-20" / "X_summary.md"))
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(LookupError):
        speakers.find_recording(empty, "latest")


def test_relabel_whole_word_only():
    text = "Speaker 2: hi Speaker 20 and Speaker 2."
    assert speakers.relabel_text(text, "Speaker 2", "Bob") == "Bob: hi Speaker 20 and Bob."


def test_relabel_name_with_regex_chars_and_backslash():
    assert speakers.relabel_text("Speaker 1: x", "Speaker 1", "Dr. O'Neil (Sr)") \
        == "Dr. O'Neil (Sr): x"
    assert speakers.relabel_text("Speaker 1: x", "Speaker 1", r"A\1B") == r"A\1B: x"
    assert speakers.relabel_text("Dr. O'Neil (Sr): x", "Dr. O'Neil (Sr)", "Al") == "Al: x"


def folder_of(side: Path) -> Path:
    return side.parent


def test_fix_relabels_transcript_and_summary_and_sidecar(lib_home):
    side = make_recording(lib_home)
    msg = speakers.fix_speaker(side, "Speaker 2", "Bob", SETTINGS, "2026-10-21")
    assert "Bob" in msg
    f = folder_of(side)
    t = (f / f"{STEM}_transcript.txt").read_text()
    assert "[00:05] Bob: hello Speaker 1" in t and "Speaker 20: yo" in t
    assert (f / f"{STEM}_summary.md").read_text() == "Bob and Speaker 20 spoke.\n"
    assert speakers.read_sidecar(side)["speakers"]["Speaker 2"]["name"] == "Bob"


def test_fix_adds_sample_with_model_source_label(lib_home):
    side = make_recording(lib_home)
    speakers.fix_speaker(side, "Speaker 2", "Bob", SETTINGS, "2026-10-21")
    sample = voices.load_library()["people"]["Bob"]["samples"][0]
    assert sample == {"source": STEM, "label": "Speaker 2", "model": MODEL,
                      "date": "2026-10-21", "seconds": 20.0, "embedding": [0.0, 1.0]}


def test_fix_short_speech_relabels_but_adds_no_sample(lib_home):
    side = make_recording(lib_home, speakers_data={
        "Speaker 1": {"name": None, "embedding": [1.0, 0.0], "seconds": 2.0}})
    msg = speakers.fix_speaker(side, "Speaker 1", "Bob", SETTINGS, "2026-10-21")
    assert "no sample" in msg.lower()
    assert "Bob" in (folder_of(side) / f"{STEM}_transcript.txt").read_text()
    assert voices.load_library() == {"people": {}}


def test_fix_is_idempotent(lib_home):
    side = make_recording(lib_home)
    speakers.fix_speaker(side, "Speaker 2", "Bob", SETTINGS, "2026-10-21")
    f = folder_of(side)
    before = (f / f"{STEM}_transcript.txt").read_text()
    lib_before = voices.load_library()
    speakers.fix_speaker(side, "Speaker 2", "Bob", SETTINGS, "2026-10-21")
    assert (f / f"{STEM}_transcript.txt").read_text() == before
    assert voices.load_library() == lib_before


def test_fix_by_current_name_confirms_and_adds_sample(lib_home):
    side = make_recording(lib_home)
    data = speakers.read_sidecar(side)
    data["speakers"]["Speaker 2"]["name"] = "Bob"
    speakers.write_sidecar(side, data)
    speakers.fix_speaker(side, "Bob", "Bob", SETTINGS, "2026-10-21")
    sample = voices.load_library()["people"]["Bob"]["samples"][0]
    assert sample["label"] == "Speaker 2"


def test_fix_renames_an_already_named_speaker(lib_home):
    side = make_recording(lib_home)
    speakers.fix_speaker(side, "Speaker 2", "Bob", SETTINGS, "2026-10-21")
    speakers.fix_speaker(side, "Bob", "Robert", SETTINGS, "2026-10-22")
    f = folder_of(side)
    assert "Robert: hello" in (f / f"{STEM}_transcript.txt").read_text()
    lib = voices.load_library()
    assert [s["label"] for s in lib["people"]["Robert"]["samples"]] == ["Speaker 2"]
    assert lib["people"]["Bob"]["samples"] == []


def test_fix_moves_sample_from_wrong_person(lib_home):
    side = make_recording(lib_home)
    speakers.fix_speaker(side, "Speaker 2", "Bob", SETTINGS, "2026-10-21")
    speakers.fix_speaker(side, "Speaker 2", "Carol", SETTINGS, "2026-10-21")
    lib = voices.load_library()
    assert lib["people"]["Bob"]["samples"] == []
    assert len(lib["people"]["Carol"]["samples"]) == 1


def test_fix_rejects_empty_name_and_name_held_by_other_speaker(lib_home):
    side = make_recording(lib_home)
    speakers.fix_speaker(side, "Speaker 1", "Bob", SETTINGS, "2026-10-21")
    for bad in ("", "   ", "bob", "Speaker 7"):
        with pytest.raises(ValueError):
            speakers.fix_speaker(side, "Speaker 2", bad, SETTINGS, "2026-10-21")


def test_fix_unknown_label_lists_available(lib_home):
    side = make_recording(lib_home)
    with pytest.raises(ValueError, match="Speaker 1") as e:
        speakers.fix_speaker(side, "Speaker 9", "Bob", SETTINGS, "2026-10-21")
    assert "Speaker 20" in str(e.value)


def test_fix_refuses_corrupt_library_without_touching_files(lib_home):
    side = make_recording(lib_home)
    lib_file = voices.library_path()
    lib_file.parent.mkdir(parents=True)
    lib_file.write_text("{not json")
    f = folder_of(side)
    snap = {p: p.read_bytes() for p in f.iterdir()}
    with pytest.raises(Exception):
        speakers.fix_speaker(side, "Speaker 2", "Bob", SETTINGS, "2026-10-21")
    assert {p: p.read_bytes() for p in f.iterdir()} == snap
    assert lib_file.read_text() == "{not json"


def test_fix_skips_missing_siblings_and_keeps_mode(lib_home):
    side = make_recording(lib_home)
    f = folder_of(side)
    (f / f"{STEM}_summary.md").unlink()
    t = f / f"{STEM}_transcript.txt"
    t.chmod(0o640)
    speakers.fix_speaker(side, "Speaker 2", "Bob", SETTINGS, "2026-10-21")
    assert stat.S_IMODE(t.stat().st_mode) == 0o640
    assert "Bob" in t.read_text()


def test_fix_keeps_transcript_and_summary_timestamps(lib_home):
    side = make_recording(lib_home)
    f = folder_of(side)
    old = (1_700_000_000_000_000_000, 1_700_000_100_000_000_000)  # (atime, mtime) ns
    paths = [f / f"{STEM}_transcript.txt", f / f"{STEM}_summary.md"]
    for p in paths:
        os.utime(p, ns=old)
    speakers.fix_speaker(side, "Speaker 2", "Bob", SETTINGS, "2026-10-21")
    for p in paths:
        assert "Bob" in p.read_text()
        assert p.stat().st_mtime_ns == old[1]


def test_fix_prefix_name_does_not_corrupt_longer_name(lib_home):
    side = make_recording(lib_home, speakers_data={
        "Speaker 1": {"name": "Ann", "embedding": [1.0, 0.0], "seconds": 30.0},
        "Speaker 2": {"name": "Ann Lee", "embedding": [0.0, 1.0], "seconds": 30.0}})
    f = folder_of(side)
    (f / f"{STEM}_transcript.txt").write_text("[0] Ann: hi\n[1] Ann Lee: yo\n")
    (f / f"{STEM}_summary.md").write_text("Ann met Ann Lee.\n")
    speakers.fix_speaker(side, "Speaker 1", "Beth", SETTINGS, "2026-10-21")
    assert (f / f"{STEM}_transcript.txt").read_text() == "[0] Beth: hi\n[1] Ann Lee: yo\n"
    assert (f / f"{STEM}_summary.md").read_text() == "Beth met Ann Lee.\n"


def test_fix_name_containing_generic_label_then_refix(lib_home):
    side = make_recording(lib_home)
    f = folder_of(side)
    speakers.fix_speaker(side, "Speaker 1", "Speaker 2's boss", SETTINGS, "2026-10-21")
    speakers.fix_speaker(side, "Speaker 2", "Bob", SETTINGS, "2026-10-21")
    t = (f / f"{STEM}_transcript.txt").read_text()
    assert "[00:01] Speaker 2's boss: hi" in t
    assert "[00:05] Bob: hello Speaker 2's boss" in t
    speakers.fix_speaker(side, "Speaker 2's boss", "Cy", SETTINGS, "2026-10-21")
    t = (f / f"{STEM}_transcript.txt").read_text()
    assert "[00:01] Cy: hi" in t and "[00:05] Bob: hello Cy" in t


def test_fix_rerun_heals_after_failed_text_write(lib_home, monkeypatch):
    side = make_recording(lib_home)
    f = folder_of(side)
    real = speakers._write_text_atomic

    def flaky(path, text):
        if path.name.endswith("_summary.md"):
            raise OSError("disk full")
        real(path, text)

    monkeypatch.setattr(speakers, "_write_text_atomic", flaky)
    with pytest.raises(OSError):
        speakers.fix_speaker(side, "Speaker 2", "Bob", SETTINGS, "2026-10-21")
    monkeypatch.setattr(speakers, "_write_text_atomic", real)
    speakers.fix_speaker(side, "Speaker 2", "Bob", SETTINGS, "2026-10-21")
    assert "Speaker 2:" not in (f / f"{STEM}_transcript.txt").read_text()
    assert (f / f"{STEM}_summary.md").read_text() == "Bob and Speaker 20 spoke.\n"


def test_latest_ignores_mtime_after_fixing_older_recording(lib_home):
    old = make_recording(lib_home, stem="PH261019000000", day="2026-10-19")
    new = make_recording(lib_home, stem="PH261020000000", day="2026-10-20")
    os.utime(new, (1, 1))
    speakers.fix_speaker(old, "Speaker 1", "Bob", SETTINGS, "2026-10-21")
    assert speakers.find_recording(lib_home, "latest") == new


def test_load_analyses_skips_speaker_sidecar(tmp_path):
    from pocket_libre.analyze import load_analyses
    (tmp_path / f"{STEM}_speakers.json").write_text(json.dumps({"speakers": {"Speaker 1": {}}}))
    (tmp_path / f"{STEM}_transcript.json").write_text("{}")
    (tmp_path / f"{STEM}_actions.json").write_text(json.dumps({"items": []}))
    assert load_analyses(tmp_path, STEM) == {"actions": {"items": []}}


# ── stale sidecars ──

def test_label_transcript_without_voiceprints_removes_stale_sidecar(
        tmp_path, monkeypatch, lib_path):
    stale = speakers.sidecar_path(tmp_path, "rec")
    stale.write_text("{}")
    run(monkeypatch, tmp_path, prints=None)
    assert not stale.exists()


def test_label_transcript_warns_when_stale_sidecar_cannot_be_removed(
        tmp_path, monkeypatch, lib_path, capsys):
    stale = speakers.sidecar_path(tmp_path, "rec")
    stale.write_text("{}")

    def boom(self, missing_ok=False):
        raise OSError("permission denied")
    monkeypatch.setattr(Path, "unlink", boom)
    labeled = run(monkeypatch, tmp_path, prints=None)
    assert [s["speaker"] for s in labeled] == ["Speaker 1", "Speaker 2"]
    assert "permission denied" in capsys.readouterr().out


def test_remove_stale_sidecar_is_a_noop_when_absent(tmp_path):
    speakers.remove_stale_sidecar(tmp_path, "rec")
    side = speakers.sidecar_path(tmp_path, "rec")
    side.write_text("{}")
    speakers.remove_stale_sidecar(tmp_path, "rec")
    assert not side.exists()


# ── settings errors ──

def test_label_transcript_reports_invalid_voice_settings_separately(
        tmp_path, monkeypatch, lib_path, capsys):
    write_lib(lib_path)
    monkeypatch.setattr("pocket_libre.diarize.diarize_auto", fake_diarize(PRINTS))
    labeled = speakers.label_transcript(
        SEGMENTS, "/a/rec.mp3", tmp_path, "rec", None, None,
        {"voices": {"threshold": "high"}},
    )
    out = capsys.readouterr().out
    assert "Voice settings invalid" in out
    assert "Voice library unreadable" not in out
    assert [s["speaker"] for s in labeled] == ["Speaker 1", "Speaker 2"]
    assert speakers.sidecar_path(tmp_path, "rec").exists()


# ── fix: case, undo, validation, purge ──

def test_fix_resolves_name_case_insensitively_to_existing_person(lib_home):
    side = make_recording(lib_home)
    other = make_recording(lib_home, stem="PH1", day="2026-10-19")
    speakers.fix_speaker(other, "Speaker 1", "Ron", SETTINGS, "2026-10-20")
    speakers.fix_speaker(side, "Speaker 2", "ron", SETTINGS, "2026-10-21")
    f = folder_of(side)
    assert "[00:05] Ron: hello Speaker 1" in (f / f"{STEM}_transcript.txt").read_text()
    assert (f / f"{STEM}_summary.md").read_text() == "Ron and Speaker 20 spoke.\n"
    assert speakers.read_sidecar(side)["speakers"]["Speaker 2"]["name"] == "Ron"
    lib = voices.load_library()
    assert list(lib["people"]) == ["Ron"]
    assert len(lib["people"]["Ron"]["samples"]) == 2


def test_fix_case_collision_with_other_speaker_stays_case_insensitive(lib_home):
    side = make_recording(lib_home)
    speakers.fix_speaker(side, "Speaker 1", "Ron", SETTINGS, "2026-10-21")
    with pytest.raises(ValueError, match="already"):
        speakers.fix_speaker(side, "Speaker 2", "RON", SETTINGS, "2026-10-21")


def test_fix_refuses_library_with_people_differing_only_by_case(lib_home):
    side = make_recording(lib_home)
    voices.save_library({"people": {"Ron": {"samples": []}, "ron": {"samples": []}}})
    f = folder_of(side)
    snap = {p: p.read_bytes() for p in f.iterdir()}
    with pytest.raises(ValueError, match="differ only by case"):
        speakers.fix_speaker(side, "Speaker 2", "RON", SETTINGS, "2026-10-21")
    assert {p: p.read_bytes() for p in f.iterdir()} == snap


def test_fix_can_undo_back_to_own_generic_label(lib_home):
    side = make_recording(lib_home)
    f = folder_of(side)
    speakers.fix_speaker(side, "Speaker 2", "Bob", SETTINGS, "2026-10-21")
    msg = speakers.fix_speaker(side, "Bob", "Speaker 2", SETTINGS, "2026-10-22")
    assert "undone" in msg.lower()
    assert "[00:05] Speaker 2: hello Speaker 1" in (f / f"{STEM}_transcript.txt").read_text()
    assert (f / f"{STEM}_summary.md").read_text() == "Speaker 2 and Speaker 20 spoke.\n"
    assert speakers.read_sidecar(side)["speakers"]["Speaker 2"]["name"] is None
    assert voices.load_library()["people"]["Bob"]["samples"] == []


def test_fix_rejects_another_speakers_generic_label_as_name(lib_home):
    side = make_recording(lib_home)
    with pytest.raises(ValueError, match="generic"):
        speakers.fix_speaker(side, "Speaker 20", "Speaker 2", SETTINGS, "2026-10-21")


def test_fix_short_speech_still_purges_earlier_sample_from_other_person(lib_home):
    side = make_recording(lib_home, speakers_data={
        "Speaker 1": {"name": None, "embedding": [1.0, 0.0], "seconds": 30.0}})
    speakers.fix_speaker(side, "Speaker 1", "Bob", SETTINGS, "2026-10-21")
    short = voices.VoiceSettings(threshold=0.7, margin=0.1, min_seconds=60.0)
    msg = speakers.fix_speaker(side, "Bob", "Carol", short, "2026-10-22")
    assert "no sample" in msg.lower()
    assert voices.load_library()["people"]["Bob"]["samples"] == []


@pytest.mark.parametrize("bad", [
    "Ann: B", "Ann\nB", "Ann\tB", "A/B", "A, B", "A & B", "A+B", "Ann and Bob", "AND",
])
def test_fix_rejects_names_that_break_transcript_lines_or_owner_splitting(lib_home, bad):
    side = make_recording(lib_home)
    with pytest.raises(ValueError):
        speakers.fix_speaker(side, "Speaker 2", bad, SETTINGS, "2026-10-21")


def test_fix_accepts_names_containing_and_as_part_of_a_word(lib_home):
    side = make_recording(lib_home)
    speakers.fix_speaker(side, "Speaker 2", "Sandy Anderson", SETTINGS, "2026-10-21")


# ── find_recording ──

def test_find_recording_glob_characters_are_literal(tmp_path):
    make_recording(tmp_path)
    for ref in ("PH*", "PH2610202136[0-9][0-9]", "PH?61020213653"):
        with pytest.raises(LookupError, match="nothing matched"):
            speakers.find_recording(tmp_path, ref)
