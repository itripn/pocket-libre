"""Summarization using Claude Haiku API. ~$0.001 per recording."""

import re
from datetime import datetime, timezone

from rich.console import Console

from pocket_libre.pricing import (
    DEFAULT_MODEL,
    INPUT_PRICE_PER_MTOK,
    OUTPUT_PRICE_PER_MTOK,
    format_cost,
)

console = Console()

SUMMARY_PROMPTS = {
    "meeting": """You are summarizing a meeting transcript. Be concise and direct.

Produce the following sections:

## Summary
2-3 sentences capturing the key discussion points and outcomes.

## Action Items
Bulleted list of action items. Format: "- [Owner]: [Task] by [Deadline if mentioned]"
Only include items that were clearly committed to.

## Key Decisions
Bulleted list of decisions that were made. Skip this section if none.

## Open Questions
Anything left unresolved. Skip this section if none.

Transcript:
{transcript}""",

    "notes": """You are processing voice notes. Extract the key ideas concisely.

## Key Ideas
The main thoughts captured, in order. Use bullets.

## Action Items
Any to-dos or follow-ups mentioned. Skip if none.

Transcript:
{transcript}""",

    "call": """You are summarizing a phone/video call. Be concise.

## Summary
2-3 sentences on what was discussed.

## Action Items
- [Owner]: [Task]

## Follow-ups Needed
Anything that requires a follow-up. Skip if none.

Transcript:
{transcript}""",

    "raw": """Summarize this transcript concisely. Pull out the most important information,
any action items, and any decisions made. Use markdown formatting.

Transcript:
{transcript}""",
}


def summarize_transcript(
    transcript_text: str,
    api_key: str,
    style: str = "meeting",
    model: str = DEFAULT_MODEL,
    custom_prompt: str | None = None,
) -> str:
    """Summarize a transcript using Claude Haiku.

    Args:
        transcript_text: The full transcript with speaker labels.
        api_key: Anthropic API key.
        style: One of "meeting", "notes", "call", "raw".
        model: Model to use (default: Haiku for cost efficiency).
        custom_prompt: Override the built-in prompt entirely.

    Returns:
        Markdown-formatted summary.
    """
    try:
        import anthropic
    except ImportError:
        console.print(
            "[red]anthropic package not installed.[/red] "
            "Run: [bold]pip install anthropic[/bold]"
        )
        return ""

    if custom_prompt:
        prompt = custom_prompt.replace("{transcript}", transcript_text)
    elif style in SUMMARY_PROMPTS:
        prompt = SUMMARY_PROMPTS[style].format(transcript=transcript_text)
    else:
        prompt = SUMMARY_PROMPTS["raw"].format(transcript=transcript_text)

    console.print(f"[dim]Summarizing with {model}...[/dim]")

    client = anthropic.Anthropic(api_key=api_key)

    message = client.messages.create(
        model=model,
        max_tokens=1500,
        messages=[
            {"role": "user", "content": prompt}
        ],
    )

    summary = message.content[0].text

    console.print(
        f"[dim]Tokens: "
        f"{format_cost(message.usage.input_tokens, message.usage.output_tokens)}[/dim]"
    )

    return summary


def estimate_cost(transcript_text: str) -> str:
    """Estimate API cost before calling. Returns human-readable string."""
    est_input = len(transcript_text) // 4
    est_output = 500
    cost = (
        est_input * INPUT_PRICE_PER_MTOK + est_output * OUTPUT_PRICE_PER_MTOK
    ) / 1_000_000
    return f"~${cost:.4f}"


def format_transcript_for_summary(segments: list[dict]) -> str:
    """Format labeled segments into a readable transcript string."""
    lines = []
    for seg in segments:
        timestamp = format_time(seg["start"])
        lines.append(f"[{timestamp}] {seg['speaker']}: {seg['text']}")
    return "\n".join(lines)


def format_time(seconds: float) -> str:
    """Format seconds as MM:SS."""
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}:{s:02d}"


def summary_stamp(now: datetime | None = None) -> str:
    """When a summary was written, in local time with its UTC offset.

    Recording names are UTC (the device clock is set to UTC), so the offset
    lets the two be compared: e.g. "2026-10-06 17:05 -06:00" for a recording
    named 20261006230500.
    """
    now = now or datetime.now()
    if now.tzinfo is None:
        now = now.astimezone()  # local time, with the machine's offset
    offset = now.strftime("%z")
    return f"{now:%Y-%m-%d %H:%M} {offset[:3]}:{offset[3:]}"


_PH_STEM = re.compile(r"PH(\d{12})")
_OLD_STEM = re.compile(r"\d{14}")
TITLE_MAX = 60


def recording_start_local(stem: str) -> datetime | None:
    """When a recording started, in local time, from its UTC-named stem.

    Handles `PHyymmddHHMMSS` and the older `YYYYMMDDHHMMSS`; None for any
    other name (e.g. a file passed to `process`).
    """
    if _PH_STEM.fullmatch(stem):
        raw = "20" + stem[2:]
    elif _OLD_STEM.fullmatch(stem):
        raw = stem
    else:
        return None
    try:
        return datetime.strptime(raw, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc).astimezone()
    except ValueError:
        return None


def format_duration(seconds: float) -> str:
    """"42 min", "1 h 05 min", or "<1 min"."""
    minutes = round(seconds / 60)
    if seconds < 60 and minutes < 1:
        return "<1 min"
    minutes = max(minutes, 1)
    if minutes >= 60:
        return f"{minutes // 60} h {minutes % 60:02d} min"
    return f"{minutes} min"


def clean_title(text: str) -> str:
    """Reduce model output to one plain line safe for a heading and a table cell.

    Parentheses are removed because pocket-todos reads the "(date time offset)"
    after the title; pipes because the index is a Markdown table.
    """
    line = next((ln for ln in text.splitlines() if ln.strip()), "")
    line = re.sub(r"^[\s#*_>`\"'\u201c\u201d]+", "", line)
    line = re.sub(r"^title\s*:\s*", "", line, flags=re.IGNORECASE)
    line = re.sub(r"[()|]", "", line)
    line = re.sub(r"[*_`\"\u201c\u201d]", "", line)
    line = re.sub(r"\s+", " ", line).strip(" .'\u2019")
    if len(line) > TITLE_MAX:
        line = line[:TITLE_MAX].rsplit(" ", 1)[0] if " " in line[:TITLE_MAX] else line[:TITLE_MAX]
    return line.strip()


TITLE_PROMPT = """Write a title for this meeting summary: at most 8 words, plain text, \
no quotes, no trailing period. Reply with only the title.

{summary}"""


def make_title(summary: str, api_key: str | None, model: str = DEFAULT_MODEL) -> str:
    """A short title for a summary, or "" if none can be made.

    One small extra call on the summary text (never the transcript), so it
    works for every summary style. Any failure just means no title.
    """
    if not api_key or not summary.strip():
        return ""
    try:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model=model,
            max_tokens=40,
            messages=[{"role": "user", "content": TITLE_PROMPT.format(summary=summary[:6000])}],
        )
        return clean_title(message.content[0].text)
    except Exception:
        return ""


def summary_document(
    stem: str,
    summary: str,
    transcript_text: str,
    title: str = "",
    duration_s: float | None = None,
) -> str:
    """The summary file: heading, one line of facts, summary, full transcript.

    The heading ends in "(YYYY-MM-DD HH:MM +hh:mm)", the recording's own local
    start, which pocket-todos reads with the stem's UTC time to find the day.
    """
    start = recording_start_local(stem)
    if start is not None:
        offset = start.strftime("%z")
        stamp = f"{start:%Y-%m-%d %H:%M} {offset[:3]}:{offset[3:]}"
    else:
        stamp = summary_stamp()
    facts = []
    if start is not None:
        facts.append(f"{start:%a} {start.day} {start:%b}")
    if duration_s:
        facts.append(format_duration(duration_s))
    if title:
        facts.append(stem)
    meta = f"*{' · '.join(facts)}*\n\n" if facts else ""
    return (
        f"# {title or stem} ({stamp})\n\n{meta}{summary}"
        f"\n\n---\n\n## Full Transcript\n\n{transcript_text}"
    )
