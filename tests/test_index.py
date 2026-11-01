"""The master index of recordings, grouped by local day."""

import json
import os
import time

import pytest

from pocket_libre import index as idx

SUMMARY = """# {title} ({stamp})

*Tue 20 Oct · 42 min · {stem}*

## Summary
{body}

---

## Full Transcript

{transcript}"""


@pytest.fixture(autouse=True)
def pacific(monkeypatch):
    monkeypatch.setenv("TZ", "America/Los_Angeles")
    time.tzset()
    yield
    monkeypatch.undo()
    time.tzset()


def add(root, folder, stem, *, title=None, body="We talked.", transcript="[00:05] Ron: hi\n[01:10] Alice: yo",
        mp3=True, summary=True, duration=None):
    d = root / folder
    d.mkdir(parents=True, exist_ok=True)
    if mp3:
        (d / f"{stem}.mp3").write_bytes(b"x")
    if transcript is not None:
        (d / f"{stem}_transcript.txt").write_text(transcript, encoding="utf-8")
    if summary:
        (d / f"{stem}_summary.md").write_text(
            SUMMARY.format(title=title or stem, stamp="2026-10-20 07:30 -07:00",
                           stem=stem, body=body, transcript=transcript or ""),
            encoding="utf-8")
    if duration is not None:
        f = d / ".downloads.json"
        data = json.loads(f.read_text()) if f.exists() else {}
        data[f"{stem}.mp3"] = {"size": 1, "duration_s": duration}
        f.write_text(json.dumps(data))


def test_groups_by_local_day_newest_first(tmp_path):
    add(tmp_path, "2026-10-20", "PH261020021437", title="Evening call")      # 19:14 on Oct 19 local
    add(tmp_path, "2026-10-20", "PH261020143032", title="Morning one-on-one")  # 07:30 on Oct 20 local
    text = idx.build_index(tmp_path)
    i19, i20 = text.index("Mon 19 Oct 2026"), text.index("Tue 20 Oct 2026")
    assert i20 < i19, "newest day first"
    assert text.index("Evening call") > i19
    assert text.index("Morning one-on-one") < i19
    assert "| 19:14 |" in text and "| 07:30 |" in text


def test_row_has_title_duration_speakers_and_relative_links(tmp_path):
    add(tmp_path, "2026-10-20", "PH261020143032", title="Budget sync")
    (tmp_path / "2026-10-20" / "PH261020143032.mp3").write_bytes(b"x" * (4000 * 42 * 60))
    text = idx.build_index(tmp_path)
    row = next(ln for ln in text.splitlines() if "Budget sync" in ln)
    assert "42 min" in row
    assert "Ron, Alice" in row
    assert "(2026-10-20/PH261020143032_summary.md)" in row
    assert "(2026-10-20/PH261020143032_transcript.txt)" in row
    assert "(2026-10-20/PH261020143032.mp3)" in row


def test_unnamed_speakers_are_counted_not_listed(tmp_path):
    add(tmp_path, "2026-10-20", "PH261020143032",
        transcript="[00:01] Ron: a\n[00:02] Speaker 1: b\n[00:03] Speaker 3: c\n[00:04] Unknown: d")
    row = next(ln for ln in idx.build_index(tmp_path).splitlines() if "07:30" in ln)
    assert "Ron" in row and "+2 unnamed" in row


def test_old_summary_without_title_falls_back_to_first_sentence(tmp_path):
    add(tmp_path, "2026-10-20", "PH261020143032", body="The team reviewed scaling. Then lunch.")
    row = next(ln for ln in idx.build_index(tmp_path).splitlines() if "07:30" in ln)
    assert "The team reviewed scaling." in row and "Then lunch" not in row


def test_duration_comes_from_audio_size_not_the_devices_listed_duration(tmp_path):
    # The device's listed duration can be a placeholder (30/60/120 s) for a 24-minute file.
    add(tmp_path, "2026-10-20", "PH261020143032", duration=30)
    (tmp_path / "2026-10-20" / "PH261020143032.mp3").write_bytes(b"x" * (4000 * 1476))
    row = next(ln for ln in idx.build_index(tmp_path).splitlines() if "07:30" in ln)
    assert "25 min" in row


def test_duration_falls_back_to_last_transcript_timestamp(tmp_path):
    add(tmp_path, "2026-10-20", "PH261020143032", transcript="[00:05] Ron: a\n[12:40] Ron: b", mp3=False)
    row = next(ln for ln in idx.build_index(tmp_path).splitlines() if "07:30" in ln)
    assert "13 min" in row


def test_recording_without_summary_still_listed(tmp_path):
    add(tmp_path, "2026-10-20", "PH261020143032", summary=False, transcript=None)
    row = next(ln for ln in idx.build_index(tmp_path).splitlines() if "07:30" in ln)
    assert "no summary" in row.lower()
    assert ".mp3" in row and "_summary.md" not in row


def test_title_cell_is_table_safe(tmp_path):
    add(tmp_path, "2026-10-20", "PH261020143032", title="A | B")
    row = next(ln for ln in idx.build_index(tmp_path).splitlines() if "07:30" in ln)
    assert row.count("|") == idx.COLUMNS + 1


def test_links_are_url_quoted_for_spaces_in_folder_names(tmp_path):
    add(tmp_path, "my folder", "PH261020143032", title="T")
    assert "(my%20folder/PH261020143032_summary.md)" in idx.build_index(tmp_path)


def test_ignores_unrelated_files_and_unreadable_ones(tmp_path):
    add(tmp_path, "2026-10-20", "PH261020143032", title="Kept")
    (tmp_path / "2026-10-20" / "notes.md").write_text("x")
    (tmp_path / "2026-10-20" / "PH261020143032_speakers.json").write_text("{}")
    bad = tmp_path / "2026-10-20" / "PH261020150132_transcript.txt"
    bad.write_bytes(b"\xff\xfe\x00bad")
    text = idx.build_index(tmp_path)
    assert "Kept" in text and "notes.md" not in text and "speakers.json" not in text


def test_empty_output_dir_gives_a_header_only(tmp_path):
    text = idx.build_index(tmp_path)
    assert text.startswith("# ") and "|" not in text


def test_write_index_is_atomic_idempotent_and_leaves_no_temp(tmp_path):
    add(tmp_path, "2026-10-20", "PH261020143032", title="T")
    path = idx.write_index(tmp_path)
    assert path == tmp_path / "index.md"
    first = path.read_bytes()
    assert idx.write_index(tmp_path).read_bytes() == first
    assert [p.name for p in tmp_path.iterdir() if p.name.endswith(".tmp")] == []


def test_write_index_does_not_touch_recordings(tmp_path):
    add(tmp_path, "2026-10-20", "PH261020143032", title="T")
    files = sorted((tmp_path / "2026-10-20").iterdir())
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in files}
    idx.write_index(tmp_path)
    assert {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in files} == before


def test_write_index_missing_directory_is_a_no_op(tmp_path):
    assert idx.write_index(tmp_path / "nope") is None
    assert not (tmp_path / "nope").exists()


def test_update_index_never_raises(tmp_path, monkeypatch, capsys):
    def boom(_):
        raise RuntimeError("disk on fire")
    monkeypatch.setattr(idx, "write_index", boom)
    idx.update_index(tmp_path)  # must not raise
    assert "disk on fire" in capsys.readouterr().out


def test_title_of_reads_heading_and_ignores_bare_stem(tmp_path):
    assert idx.title_of("# Budget sync (2026-10-20 07:30 -07:00)\n\nx", "PH1") == "Budget sync"
    assert idx.title_of("# PH261020143032 (2026-10-20 07:30 -07:00)\n\nx", "PH261020143032") == ""
    assert idx.title_of("no heading", "PH1") == ""
