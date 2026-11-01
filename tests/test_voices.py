"""Tests for the voice library and speaker matching."""

import json
import math
import os
import stat

import pytest

from pocket_libre import voices
from pocket_libre.voices import VoiceSettings

MODEL = "pyannote/wespeaker"
SETTINGS = VoiceSettings(threshold=0.70, margin=0.10, min_seconds=8.0)


def vec(score: float) -> list[float]:
    """A unit vector whose cosine with [1, 0] is `score`."""
    return [score, math.sqrt(1 - score * score)]


def sample(embedding=None, model=MODEL, source="rec1", label="Speaker 1", date="2026-10-20"):
    return {
        "embedding": embedding if embedding is not None else [1.0, 0.0],
        "model": model,
        "source": source,
        "label": label,
        "date": date,
        "seconds": 30.0,
    }


def lib_with(**people):
    return {"people": {name: {"samples": list(samples)} for name, samples in people.items()}}


def speaker(embedding, seconds=60.0):
    return {"embedding": embedding, "seconds": seconds}


@pytest.fixture
def lib_path(tmp_path, monkeypatch):
    path = tmp_path / "pl" / "voices.json"
    monkeypatch.setattr(voices, "library_path", lambda: path)
    return path


def test_load_settings_defaults_and_overrides():
    assert voices.load_settings({}) == SETTINGS
    cfg = {"voices": {"threshold": 0.8, "margin": 0.2, "min_seconds": 5}}
    assert voices.load_settings(cfg) == VoiceSettings(0.8, 0.2, 5.0)


def test_load_missing_returns_empty(lib_path):
    assert voices.load_library() == {"people": {}}


def test_load_corrupt_raises_library_error(lib_path):
    lib_path.parent.mkdir(parents=True)
    lib_path.write_text("{not json", encoding="utf-8")
    with pytest.raises(voices.LibraryError):
        voices.load_library()


def test_save_roundtrip(lib_path):
    lib = lib_with(Ron=[sample()])
    voices.save_library(lib)
    assert voices.load_library() == lib


def test_save_sets_modes_600_and_700(lib_path):
    voices.save_library(lib_with(Ron=[sample()]))
    assert stat.S_IMODE(lib_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(lib_path.parent.stat().st_mode) == 0o700


def test_save_is_atomic(lib_path, monkeypatch):
    old = lib_with(Ron=[sample()])
    voices.save_library(old)

    def boom(src, dst):
        raise OSError("disk gone")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError):
        voices.save_library(lib_with(John=[sample()]))
    assert json.loads(lib_path.read_text(encoding="utf-8")) == old
    assert [p.name for p in lib_path.parent.iterdir()] == ["voices.json"]


def test_add_sample_replaces_same_source_label_from_other_person():
    lib = lib_with(John=[sample(source="rec1", label="Speaker 2")])
    voices.add_sample(lib, "Mary", sample(source="rec1", label="Speaker 2"))
    assert lib["people"]["John"]["samples"] == []
    assert len(lib["people"]["Mary"]["samples"]) == 1


def test_add_sample_caps_at_20_dropping_oldest():
    lib = {"people": {}}
    for i in range(25):
        voices.add_sample(lib, "Ron", sample(source=f"rec{i}"))
    sources = [s["source"] for s in lib["people"]["Ron"]["samples"]]
    assert sources == [f"rec{i}" for i in range(5, 25)]


def test_forget_removes_person_and_reports():
    lib = lib_with(Ron=[sample()])
    assert voices.forget(lib, "Ron") is True
    assert lib["people"] == {}
    assert voices.forget(lib, "Ron") is False


def test_cosine_identical_is_one_orthogonal_is_zero():
    assert voices.cosine([1.0, 2.0], [1.0, 2.0]) == pytest.approx(1.0)
    assert voices.cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_match_names_clear_match():
    lib = lib_with(Ron=[sample(vec(0.95))], John=[sample([0.0, 1.0])])
    got = voices.match_speakers({"Speaker 1": speaker([1.0, 0.0])}, lib, SETTINGS, MODEL)
    assert got == {"Speaker 1": "Ron"}


def test_match_needs_threshold():
    lib = lib_with(Ron=[sample(vec(0.69))])
    assert voices.match_speakers({"Speaker 1": speaker([1.0, 0.0])}, lib, SETTINGS, MODEL) == {}


def test_match_needs_margin_over_runner_up():
    lib = lib_with(Ron=[sample(vec(0.80))], John=[sample(vec(0.75))])
    spk = {"Speaker 1": speaker([1.0, 0.0])}
    assert voices.match_speakers(spk, lib, SETTINGS, MODEL) == {}
    lone = lib_with(Ron=[sample(vec(0.80))])
    assert voices.match_speakers(spk, lone, SETTINGS, MODEL) == {"Speaker 1": "Ron"}


def test_match_ignores_other_model_samples():
    lib = lib_with(Ron=[sample([1.0, 0.0], model="other/model")])
    assert voices.match_speakers({"Speaker 1": speaker([1.0, 0.0])}, lib, SETTINGS, MODEL) == {}


def test_match_skips_speakers_under_min_seconds():
    lib = lib_with(Ron=[sample([1.0, 0.0])])
    spk = {"Speaker 1": speaker([1.0, 0.0], seconds=7.9)}
    assert voices.match_speakers(spk, lib, SETTINGS, MODEL) == {}


def test_match_never_gives_one_name_to_two_speakers():
    lib = lib_with(Ron=[sample([1.0, 0.0])])
    spk = {"Speaker 1": speaker(vec(0.8)), "Speaker 2": speaker([1.0, 0.0])}
    assert voices.match_speakers(spk, lib, SETTINGS, MODEL) == {"Speaker 2": "Ron"}


def test_match_skips_non_finite_embeddings():
    lib = lib_with(Ron=[sample([1.0, 0.0])])
    spk = {"Speaker 1": speaker([float("nan"), 0.0])}
    assert voices.match_speakers(spk, lib, SETTINGS, MODEL) == {}
    bad = lib_with(Ron=[sample([float("inf"), 0.0])])
    assert voices.match_speakers({"Speaker 1": speaker([1.0, 0.0])}, bad, SETTINGS, MODEL) == {}


def test_person_score_is_best_sample():
    lib = lib_with(Ron=[sample(vec(0.3)), sample(vec(0.9))])
    scores = voices.score_speakers({"Speaker 1": speaker([1.0, 0.0])}, lib, MODEL)
    assert scores["Speaker 1"]["Ron"] == pytest.approx(0.9)


def test_score_speakers_best_sample_per_person_same_model_only():
    lib = lib_with(
        Ron=[sample(vec(0.4)), sample(vec(0.6)), sample([1.0, 0.0], model="other/model")],
        John=[sample([1.0, 0.0], model="other/model")],
        Mary=[sample([0.0, 1.0])],
    )
    spk = {
        "Speaker 1": speaker([1.0, 0.0]),
        "Speaker 2": speaker([float("nan"), 0.0]),
    }
    scores = voices.score_speakers(spk, lib, MODEL)
    assert set(scores) == {"Speaker 1"}
    assert set(scores["Speaker 1"]) == {"Ron", "Mary"}
    assert scores["Speaker 1"]["Ron"] == pytest.approx(0.6)
    assert scores["Speaker 1"]["Mary"] == pytest.approx(0.0)


def test_match_names_neither_of_two_close_speakers_for_one_person():
    lib = lib_with(Ron=[sample([1.0, 0.0])])
    spk = {"Speaker 1": speaker(vec(0.76)), "Speaker 2": speaker(vec(0.75))}
    assert voices.match_speakers(spk, lib, SETTINGS, MODEL) == {}


def test_match_names_the_clearly_better_of_two_speakers_for_one_person():
    lib = lib_with(Ron=[sample([1.0, 0.0])])
    spk = {"Speaker 1": speaker(vec(0.90)), "Speaker 2": speaker(vec(0.60))}
    assert voices.match_speakers(spk, lib, SETTINGS, MODEL) == {"Speaker 1": "Ron"}
