"""Recording start times, summary titles and the summary document heading."""

import re
import time

import pytest

from pocket_libre import summarize
from pocket_libre.summarize import (
    clean_title,
    make_title,
    recording_start_local,
    summary_document,
)

# The same pattern pocket-todos uses to read a summary's date and UTC offset.
HEADER_STAMP = re.compile(r"#[^\n(]*\((\d{4}-\d{2}-\d{2}) \d{2}:\d{2} ([+-])(\d{2}):(\d{2})\)")


@pytest.fixture
def tz(monkeypatch):
    """Run a test in a fixed time zone (Pacific, UTC-7 in October)."""
    monkeypatch.setenv("TZ", "America/Los_Angeles")
    time.tzset()
    yield
    monkeypatch.undo()
    time.tzset()


def test_start_local_converts_ph_utc_stem_to_local(tz):
    start = recording_start_local("PH261020143032")
    assert (start.year, start.month, start.day, start.hour, start.minute) == (2026, 10, 20, 7, 30)
    assert start.utcoffset().total_seconds() == -7 * 3600


def test_start_local_evening_utc_is_previous_local_day(tz):
    start = recording_start_local("PH261020021437")
    assert (start.month, start.day, start.hour, start.minute) == (10, 19, 19, 14)


def test_start_local_accepts_old_style_stem(tz):
    start = recording_start_local("20261006230500")
    assert (start.month, start.day, start.hour) == (10, 6, 16)


@pytest.mark.parametrize("stem", ["my-meeting", "PH2610", "", "PH261320143032"])
def test_start_local_unparseable_is_none(stem):
    assert recording_start_local(stem) is None


@pytest.mark.parametrize("raw,expected", [
    ("One-on-one with Alice", "One-on-one with Alice"),
    ('"Sifter scaling review"', "Sifter scaling review"),
    ("# **Budget sync**", "Budget sync"),
    ("Title: Roadmap planning.", "Roadmap planning"),
    ("Planning (Q4)\nSecond line", "Planning Q4"),
    ("  spaced   out \t title ", "spaced out title"),
    ("Pipes | and\ttabs", "Pipes and tabs"),
    ("", ""),
    ("   \n  ", ""),
    ('""', ""),
])
def test_clean_title(raw, expected):
    assert clean_title(raw) == expected


def test_clean_title_caps_length():
    out = clean_title("word " * 40)
    assert 0 < len(out) <= 60
    assert not out.endswith(" ")


class _FakeClient:
    def __init__(self, text=None, error=None):
        self._text, self._error = text, error
        self.calls = []
        self.messages = self

    def create(self, **kw):
        self.calls.append(kw)
        if self._error:
            raise self._error
        return type("M", (), {"content": [type("B", (), {"text": self._text})()]})()


def _patch_client(monkeypatch, client):
    import anthropic
    monkeypatch.setattr(anthropic, "Anthropic", lambda api_key=None: client)


def test_make_title_returns_cleaned_title(monkeypatch):
    client = _FakeClient("\"Hiring plan (draft)\"")
    _patch_client(monkeypatch, client)
    assert make_title("## Summary\nWe discussed hiring.", "key") == "Hiring plan draft"
    sent = client.calls[0]
    assert sent["max_tokens"] <= 60
    assert "We discussed hiring." in sent["messages"][0]["content"]


def test_make_title_swallows_api_errors(monkeypatch):
    _patch_client(monkeypatch, _FakeClient(error=RuntimeError("boom")))
    assert make_title("summary", "key") == ""


def test_make_title_without_key_or_summary_makes_no_call(monkeypatch):
    client = _FakeClient("x")
    _patch_client(monkeypatch, client)
    assert make_title("summary", None) == ""
    assert make_title("", "key") == ""
    assert client.calls == []


def test_summary_document_with_title(tz):
    doc = summary_document("PH261020143032", "## Summary\nHi.", "[00:01] Ron: hi",
                           title="One-on-one with Alice", duration_s=42 * 60 + 10)
    first, _, rest = doc.partition("\n")
    assert first == "# One-on-one with Alice (2026-10-20 07:30 -07:00)"
    assert "*Tue 20 Oct · 42 min · PH261020143032*" in rest.splitlines()[1]
    assert "## Summary\nHi." in doc
    assert doc.endswith("---\n\n## Full Transcript\n\n[00:01] Ron: hi")
    m = HEADER_STAMP.match(doc)
    assert m.groups() == ("2026-10-20", "-", "07", "00")


def test_summary_document_without_title_keeps_the_id_in_the_heading(tz):
    doc = summary_document("PH261020143032", "S", "T")
    assert doc.startswith("# PH261020143032 (2026-10-20 07:30 -07:00)\n")
    assert HEADER_STAMP.match(doc)


def test_summary_document_stamps_the_recording_not_the_summary_time(tz):
    # Summarised long after recording: the heading still says when it was recorded.
    doc = summary_document("PH261020021437", "S", "T", title="Late call")
    assert doc.startswith("# Late call (2026-10-19 19:14 -07:00)")


def test_summary_document_non_recording_stem_falls_back_to_now(tz):
    doc = summary_document("my-meeting", "S", "T")
    assert re.match(r"# my-meeting \(\d{4}-\d{2}-\d{2} \d{2}:\d{2} [+-]\d{2}:\d{2}\)\n", doc)


@pytest.mark.parametrize("seconds,expected", [
    (20, "<1 min"), (60, "1 min"), (95, "2 min"), (42 * 60, "42 min"),
    (65 * 60, "1 h 05 min"), (3 * 3600, "3 h 00 min"),
])
def test_format_duration(seconds, expected):
    assert summarize.format_duration(seconds) == expected
