"""A master index of recordings, grouped by local day.

Folders and `PH...` names are in UTC (the device clock), so an evening call
sits in tomorrow's folder. The index shows what happened on each local day,
with titles, and links into the folders. It is rebuilt from the files on disk,
so it never needs repairing and never changes a recording.
"""

import os
import re
import tempfile
from collections import defaultdict
from pathlib import Path
from urllib.parse import quote

from rich.console import Console
from rich.markup import escape

from pocket_libre.protocol import BYTES_PER_SECOND
from pocket_libre.summarize import format_duration, recording_start_local

console = Console()

INDEX_NAME = "index.md"
COLUMNS = 5
_FILE = re.compile(r"(PH\d{12}|\d{14})(\.mp3|_transcript\.txt|_summary\.md)")
_HEADING = re.compile(r"#[ \t]+(.*?)[ \t]*\(\d{4}-\d{2}-\d{2} \d{2}:\d{2} [+-]\d{2}:\d{2}\)[ \t]*")
_TURN = re.compile(r"^\[(\d+):(\d{2})\] (.+?): ", re.MULTILINE)
_GENERIC = re.compile(r"speaker( \d+)?", re.IGNORECASE)
_SENTENCE = re.compile(r"(.+?[.!?])(\s|$)", re.DOTALL)
_KIND = {".mp3": "audio", "_transcript.txt": "transcript", "_summary.md": "summary"}
_LINK_ORDER = ("summary", "transcript", "audio")
FALLBACK_MAX = 80


def title_of(summary_text: str, stem: str) -> str:
    """The title in a summary's heading, or "" (old summaries put the id there)."""
    first = summary_text.split("\n", 1)[0]
    m = _HEADING.fullmatch(first)
    title = m.group(1).strip() if m else ""
    return "" if title == stem else title


_HEADING_PARTS = re.compile(r"#[ \t]+(.*?)[ \t]*(\(\d{4}-\d{2}-\d{2} \d{2}:\d{2} [+-]\d{2}:\d{2}\))[ \t]*")
_TRANSCRIPT_SPLIT = "\n\n---\n\n## Full Transcript"


def untitled_summaries(out_root) -> list[Path]:
    """Summaries whose heading still holds the bare recording id."""
    found = []
    for path in sorted(Path(out_root).glob("*/*_summary.md")):
        m = _FILE.fullmatch(path.name)
        if not m:
            continue
        text = _read(path)
        first = text.split("\n", 1)[0]
        if _HEADING_PARTS.fullmatch(first) and not title_of(text, m.group(1)):
            found.append(path)
    return found


def summary_body(text: str) -> str:
    """The summary itself: no heading, facts line or transcript."""
    head = text.split(_TRANSCRIPT_SPLIT, 1)[0]
    lines = head.split("\n")[1:]
    while lines and (not lines[0].strip() or lines[0].startswith("*")):
        lines.pop(0)
    return "\n".join(lines).strip()


def retitle(text: str, stem: str, title: str) -> str:
    """Put a title in the heading of a summary that has none, keeping the id on a facts line."""
    first, _, rest = text.partition("\n")
    m = _HEADING_PARTS.fullmatch(first)
    if not m:
        return text
    rest = rest.lstrip("\n")
    meta = ""
    if not rest.startswith("*"):
        start = recording_start_local(stem)
        day = f"{start:%a} {start.day} {start:%b} · " if start else ""
        meta = f"*{day}{stem}*\n\n"
    return f"# {title} {m.group(2)}\n\n{meta}{rest}"


def _first_sentence(summary_text: str) -> str:
    """First sentence of the `## Summary` section, for summaries without a title."""
    m = re.search(r"^## Summary[ \t]*\n+(.+?)(?:\n\s*\n|\n##|\n---|\Z)", summary_text, re.MULTILINE | re.DOTALL)
    if not m:
        return ""
    para = " ".join(m.group(1).split())
    s = _SENTENCE.match(para)
    out = s.group(1) if s else para
    return out if len(out) <= FALLBACK_MAX else out[:FALLBACK_MAX].rsplit(" ", 1)[0] + "…"


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _audio_seconds(audio: Path | None) -> float | None:
    """Length from the file size: the audio is constant 32 kbps.

    The device's listed duration is not used; it can be a placeholder
    (30, 60 or 120 s) for a recording that is really half an hour long.
    """
    try:
        size = audio.stat().st_size if audio else 0
    except OSError:
        return None
    return size / BYTES_PER_SECOND if size > 0 else None


def _speakers(transcript: str) -> str:
    named, unnamed = [], set()
    for m in _TURN.finditer(transcript):
        label = m.group(3)
        if label.lower() == "unknown":
            continue
        if _GENERIC.fullmatch(label):
            unnamed.add(label)
        elif label not in named:
            named.append(label)
    parts = list(named)
    if unnamed:
        parts.append(f"+{len(unnamed)} unnamed" if named else f"{len(unnamed)} unnamed")
    return ", ".join(parts)


def _cell(text: str) -> str:
    return " ".join(text.replace("|", "/").split())


def _row(root: Path, folder: Path, stem: str, kinds: dict[str, Path], start) -> str:
    summary = _read(kinds["summary"]) if "summary" in kinds else ""
    transcript = _read(kinds["transcript"]) if "transcript" in kinds else ""
    title = (title_of(summary, stem) or _first_sentence(summary)) if summary else ""
    seconds = _audio_seconds(kinds.get("audio"))
    if seconds is None:
        turns = _TURN.findall(transcript)
        seconds = int(turns[-1][0]) * 60 + int(turns[-1][1]) if turns else None
    links = " · ".join(
        f"[{kind}]({quote(kinds[kind].relative_to(root).as_posix(), safe='/')})"
        for kind in _LINK_ORDER if kind in kinds
    )
    return "| " + " | ".join([
        f"{start:%H:%M}" if start else "--:--",
        _cell(title) if title else "*(no summary)*",
        format_duration(seconds) if seconds else "",
        _cell(_speakers(transcript)),
        links,
    ]) + " |"


def build_index(out_root) -> str:
    """The index as Markdown text."""
    root = Path(out_root)
    days: dict[str, list] = defaultdict(list)  # sort key (local date) -> [(time key, row)]
    labels: dict[str, str] = {}
    folders = sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith("."))
    for folder in folders:
        found: dict[str, dict[str, Path]] = defaultdict(dict)
        for f in folder.iterdir():
            m = _FILE.fullmatch(f.name)
            if m:
                found[m.group(1)][_KIND[m.group(2)]] = f
        for stem, kinds in found.items():
            start = recording_start_local(stem)
            if start:
                key = start.date().isoformat()
                labels[key] = f"{start:%a} {start.day} {start:%b %Y}"
                time_key = start.isoformat()
            else:
                key, time_key = folder.name, stem
                labels[key] = folder.name
            days[key].append((time_key, _row(root, folder, stem, kinds, start)))

    out = ["# Pocket Libre recordings", "", "Times are local. Newest day first.", ""]
    for key in sorted(days, reverse=True):
        out += [f"## {labels[key]}", "",
                "| Time | Title | Length | Speakers | Files |", "|---|---|---|---|---|"]
        out += [row for _, row in sorted(days[key])]
        out.append("")
    return "\n".join(out).rstrip("\n") + "\n"


def write_index(out_root) -> Path | None:
    """Write `index.md` in out_root atomically; None if the folder doesn't exist."""
    root = Path(out_root)
    if not root.is_dir():
        return None
    text = build_index(root)
    path = root / INDEX_NAME
    fd, tmp = tempfile.mkstemp(dir=root, prefix=".index-", suffix=".tmp")
    try:
        os.fchmod(fd, 0o644)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return path


def update_index(out_root) -> None:
    """Refresh the index after a change; a failure only warns."""
    try:
        write_index(out_root)
    except Exception as e:
        console.print(f"[yellow]Could not update the recordings index: {escape(str(e))}[/yellow]")
