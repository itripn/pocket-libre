"""Speaker diarization. Identifies who is speaking at each point in the audio.

Uses pyannote.audio if available (best quality), falls back to a simple
energy-based heuristic for basic speaker separation.

pyannote.audio requires:
  pip install pyannote.audio
  A HuggingFace token (free): https://huggingface.co/settings/tokens
  Accept terms at: https://huggingface.co/pyannote/speaker-diarization-3.1
"""

import inspect
import json
import os
import re
from dataclasses import dataclass

from rich.console import Console

from pocket_libre.pricing import DEFAULT_MODEL, format_cost

console = Console()

DEFAULT_PYANNOTE_MODEL = "pyannote/speaker-diarization-3.1"

# One speaker-change entry is ~12 tokens and there can be one per line, so the
# reply budget grows with the transcript. Capped so the request stays non-streaming.
LLM_TOKENS_PER_LINE = 20
LLM_MIN_TOKENS = 1500
LLM_MAX_TOKENS = 16000


@dataclass
class SpeakerSegment:
    """A labeled segment of speech."""
    start: float
    end: float
    speaker: str


def diarize_pyannote(
    audio_path: str,
    hf_token: str,
    voiceprints_out: dict | None = None,
) -> list[SpeakerSegment]:
    """Diarize using pyannote.audio (high quality, needs HuggingFace token).

    When voiceprints_out is a dict, fill it with the per-speaker embeddings
    pyannote computes anyway: {"model": id, "speakers": {"Speaker 1":
    {"embedding": [...], "seconds": n}}}. Left empty if they are unavailable.
    """
    try:
        from pyannote.audio import Pipeline
    except ImportError:
        console.print(
            "[red]pyannote.audio not installed.[/red] "
            "Run: [bold]pip install pyannote.audio[/bold]"
        )
        return []

    model = os.environ.get("POCKET_LIBRE_DIARIZE_MODEL") or DEFAULT_PYANNOTE_MODEL
    console.print(f"[dim]Loading pyannote speaker diarization model ({model})...[/dim]")
    # pyannote 4 renamed use_auth_token to token.
    token_kwarg = ("token" if "token" in inspect.signature(Pipeline.from_pretrained).parameters
                   else "use_auth_token")
    pipeline = Pipeline.from_pretrained(model, **{token_kwarg: hf_token})
    if pipeline is None:
        # pyannote returns None, not an error, when the account hasn't accepted the terms.
        raise RuntimeError(
            f"Could not load {model}. Accept its terms at https://huggingface.co/{model} "
            "(signed in as the account that owns your HuggingFace token) and check the token."
        )

    import torch

    device = _pick_device(torch)
    pipeline.to(torch.device(device))
    console.print(f"[dim]Running diarization on {device}...[/dim]")
    # 3.x only returns embeddings when asked; 4.x always carries them on the output.
    want_3x_embeddings = (voiceprints_out is not None and token_kwarg == "use_auth_token"
                          and _accepts_return_embeddings(pipeline))
    run_kwargs = {"return_embeddings": True} if want_3x_embeddings else {}
    try:
        output = pipeline(audio_path, **run_kwargs)
    except Exception as e:
        if device == "cpu":
            raise
        console.print(f"[yellow]Diarization failed on {device} ({e}); retrying on cpu[/yellow]")
        pipeline.to(torch.device("cpu"))
        output = pipeline(audio_path, **run_kwargs)

    embeddings = None
    if want_3x_embeddings:
        output, embeddings = output
    else:
        embeddings = getattr(output, "speaker_embeddings", None)

    # pyannote 4 returns a DiarizeOutput; 3.x returned the Annotation itself.
    diarization = getattr(output, "speaker_diarization", output)

    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append(SpeakerSegment(
            start=turn.start,
            end=turn.end,
            speaker=speaker,
        ))

    # Rename speakers to friendly names (Speaker 1, Speaker 2, etc.)
    unique_speakers = list(dict.fromkeys(seg.speaker for seg in segments))
    name_map = {spk: f"Speaker {i+1}" for i, spk in enumerate(unique_speakers)}

    for seg in segments:
        seg.speaker = name_map[seg.speaker]

    if voiceprints_out is not None and embeddings is not None:
        # Embedding row i belongs to labels()[i], not to the order speakers first talk.
        labels = list(diarization.labels())
        if len(embeddings) == len(labels) == len(unique_speakers):
            seconds: dict[str, float] = {}
            for seg in segments:
                seconds[seg.speaker] = seconds.get(seg.speaker, 0.0) + (seg.end - seg.start)
            voiceprints_out["model"] = model
            voiceprints_out["speakers"] = {
                name_map[label]: {
                    "embedding": embeddings[i].tolist(),
                    "seconds": seconds[name_map[label]],
                }
                for i, label in enumerate(labels)
            }

    console.print(f"[green]Found {len(unique_speakers)} speaker(s)[/green]")
    return segments


def _accepts_return_embeddings(pipeline) -> bool:
    """Whether a 3.x pipeline call takes return_embeddings (named or via **kwargs)."""
    try:
        params = inspect.signature(pipeline.__call__).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(p.name == "return_embeddings" or p.kind is p.VAR_KEYWORD for p in params)


def _pick_device(torch) -> str:
    """Fastest device available, or whatever POCKET_LIBRE_DIARIZE_DEVICE names."""
    forced = os.environ.get("POCKET_LIBRE_DIARIZE_DEVICE", "").strip().lower()
    if forced:
        return forced
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _salvage_entries(text: str) -> list[dict]:
    """Complete {"line": ..} objects from a reply that was cut off mid-array."""
    entries = []
    for match in re.finditer(r"\{[^{}]*\}", text):
        try:
            entry = json.loads(match.group())
        except json.JSONDecodeError:
            continue
        if isinstance(entry, dict) and "line" in entry:
            entries.append(entry)
    return entries


def diarize_llm(whisper_segments: list[dict], api_key: str) -> list[SpeakerSegment]:
    """Use Claude Haiku to identify speakers from transcript content.

    Analyzes the text of each segment to determine speaker changes based
    on context, tone, and conversational flow. ~$0.001 per call.
    """
    try:
        import anthropic
    except ImportError:
        console.print("[yellow]anthropic not installed, falling back to simple diarization[/yellow]")
        return diarize_simple(whisper_segments)

    # Build a numbered transcript for Claude to label
    lines = []
    for i, seg in enumerate(whisper_segments):
        t = int(seg["start"])
        lines.append(f"{i}: [{t//60:02d}:{t%60:02d}] {seg['text'].strip()}")
    numbered_transcript = "\n".join(lines)

    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model=DEFAULT_MODEL,
        max_tokens=min(LLM_MAX_TOKENS,
                       max(LLM_MIN_TOKENS, LLM_TOKENS_PER_LINE * len(whisper_segments))),
        system=(
            "You identify distinct speakers in transcripts. Analyze speaking patterns, "
            "context, and conversational flow to determine when the speaker changes. "
            "Return ONLY a JSON array mapping line numbers to speaker labels. "
            "Use descriptive labels when possible (e.g. 'Interviewer', 'Host', 'Manager') "
            "or 'Speaker 1', 'Speaker 2' etc. if roles are unclear."
        ),
        messages=[{
            "role": "user",
            "content": (
                f"Identify the speakers in this transcript. Each line is numbered.\n"
                f"Return a JSON array like: [{{\"line\": 0, \"speaker\": \"Speaker 1\"}}, ...]\n"
                f"Only include lines where the speaker CHANGES (first line always included).\n\n"
                f"{numbered_transcript}"
            ),
        }],
    )

    console.print(
        f"[dim]Speaker ID: "
        f"{format_cost(message.usage.input_tokens, message.usage.output_tokens)}[/dim]"
    )

    # Parse response
    truncated = getattr(message, "stop_reason", None) == "max_tokens"
    text = message.content[0].text.strip()
    if text.startswith("```"):
        text = "\n".join(ln for ln in text.split("\n") if not ln.strip().startswith("```"))

    # Models routinely wrap the array in a sentence ("Here are the changes:") or a
    # fence that doesn't start the reply. The fence check above only fires when the
    # text *begins* with backticks, so a preamble defeats it and json.loads dies on
    # the first character. Slice to the outermost brackets instead — that tolerates
    # prose and fences alike.
    start, end = text.find("["), text.rfind("]")
    if start != -1 and end > start:
        text = text[start:end + 1]
    elif start != -1:
        text = text[start:]  # no closing bracket: the reply was cut off

    try:
        changes = json.loads(text)
        if not isinstance(changes, list):
            raise json.JSONDecodeError("expected a JSON array", text, 0)
    except json.JSONDecodeError:
        # A reply cut off by the token limit still holds many good entries.
        changes = _salvage_entries(text) if start != -1 else []
        if not changes:
            console.print("[yellow]Could not parse speaker labels, falling back to simple[/yellow]")
            console.print(f"[dim]Got: {text[:200]}[/dim]")
            return diarize_simple(whisper_segments)
        reason = "cut off (hit the token limit)" if truncated else "malformed"
        console.print(
            f"[yellow]Speaker labels were {reason}; "
            f"kept the {len(changes)} complete entries.[/yellow]"
        )

    # Build a line->speaker mapping
    speaker_map = {}
    current_speaker = "Speaker 1"
    changes = [c for c in changes if isinstance(c, dict)]
    for change in sorted(changes, key=lambda c: c.get("line", 0)):
        line_num = change.get("line", 0)
        current_speaker = change.get("speaker", current_speaker)
        speaker_map[line_num] = current_speaker

    # Assign speakers to all segments
    segments = []
    active_speaker = "Speaker 1"
    unique_speakers = set()
    for i, seg in enumerate(whisper_segments):
        if i in speaker_map:
            active_speaker = speaker_map[i]
        unique_speakers.add(active_speaker)
        segments.append(SpeakerSegment(
            start=seg["start"],
            end=seg["end"],
            speaker=active_speaker,
        ))

    console.print(f"[green]Identified {len(unique_speakers)} speaker(s) via AI[/green]")
    return segments


def diarize_simple(whisper_segments: list[dict]) -> list[SpeakerSegment]:
    """Simple fallback: treat each Whisper segment as a single speaker turn.

    This doesn't actually identify speakers, but provides the segment
    structure so the transcript is still time-stamped. Speaker labels
    are all "Speaker" since we can't differentiate without pyannote.
    """
    return [
        SpeakerSegment(
            start=seg["start"],
            end=seg["end"],
            speaker="Speaker",
        )
        for seg in whisper_segments
    ]


def diarize_auto(
    whisper_segments: list[dict],
    audio_path: str | None = None,
    hf_token: str | None = None,
    anthropic_key: str | None = None,
    voiceprints_out: dict | None = None,
) -> list[SpeakerSegment]:
    """Auto-select best diarization method.

    Priority: pyannote (if installed + token) > Claude Haiku (if key) > simple fallback.
    Only pyannote fills voiceprints_out.
    """
    # Try pyannote first
    if hf_token and audio_path:
        try:
            segments = diarize_pyannote(audio_path, hf_token, voiceprints_out)
            if segments:
                return segments
        except Exception as e:
            console.print(f"[yellow]pyannote failed: {e}[/yellow]")

    # Try LLM diarization
    if anthropic_key and whisper_segments:
        try:
            segments = diarize_llm(whisper_segments, anthropic_key)
            if segments:
                return segments
        except Exception as e:
            console.print(f"[dim]LLM diarization failed: {e}[/dim]")

    # Simple fallback
    return diarize_simple(whisper_segments)


def merge_transcript_with_speakers(
    whisper_segments: list[dict],
    speaker_segments: list[SpeakerSegment],
) -> list[dict]:
    """Merge Whisper transcript segments with speaker labels.

    Each Whisper segment goes to the speaker who talks for most of it. Speaker
    turns rarely line up with Whisper's segments, so looking only at the
    midpoint left a large share of segments unlabeled. A zero-length segment,
    which has no span to overlap, is labeled by the turn it sits in. A segment
    that overlaps no turn at all is "Unknown".
    """
    if not speaker_segments:
        return [
            {
                "start": seg["start"],
                "end": seg["end"],
                "speaker": "Speaker",
                "text": seg["text"].strip(),
            }
            for seg in whisper_segments
        ]

    result = []
    for seg in whisper_segments:
        speaker = "Unknown"
        if seg["end"] > seg["start"]:
            overlap: dict[str, float] = {}
            for spk_seg in speaker_segments:
                shared = min(seg["end"], spk_seg.end) - max(seg["start"], spk_seg.start)
                if shared > 0:
                    overlap[spk_seg.speaker] = overlap.get(spk_seg.speaker, 0.0) + shared
            if overlap:
                speaker = max(overlap, key=overlap.get)
        else:
            for spk_seg in speaker_segments:
                if spk_seg.start <= seg["start"] <= spk_seg.end:
                    speaker = spk_seg.speaker
                    break

        result.append({
            "start": seg["start"],
            "end": seg["end"],
            "speaker": speaker,
            "text": seg["text"].strip(),
        })

    # Merge consecutive segments from the same speaker
    merged = []
    for seg in result:
        if merged and merged[-1]["speaker"] == seg["speaker"]:
            merged[-1]["end"] = seg["end"]
            merged[-1]["text"] += " " + seg["text"]
        else:
            merged.append(dict(seg))

    return merged
