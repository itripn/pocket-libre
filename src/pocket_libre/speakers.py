"""Per-recording speaker sidecar and voice-library labelling.

The sidecar (<stem>_speakers.json, owner-only) keeps each speaker's voiceprint
under its original "Speaker N" label so the speakers can be confirmed later.
"""

import glob
import json
import math
import os
import re
import tempfile
from dataclasses import replace
from pathlib import Path

from rich.console import Console

console = Console()

SIDECAR_VERSION = 1


def sidecar_path(out_dir: Path, stem: str) -> Path:
    return out_dir / f"{stem}_speakers.json"


def read_sidecar(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_sidecar(path: Path, data: dict) -> None:
    """Write the sidecar atomically with mode 600."""
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".speakers-", suffix=".tmp")
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def remove_stale_sidecar(out_dir: Path, stem: str) -> None:
    """Delete a leftover sidecar when a re-run produced no voiceprints.

    Its "Speaker N" labels belong to the earlier diarization, so confirming a
    speaker from it could teach the library the wrong voice. Never raises.
    """
    try:
        sidecar_path(out_dir, stem).unlink(missing_ok=True)
    except OSError as e:
        console.print(f"[yellow]Could not remove outdated speaker voiceprints ({e}).[/yellow]")


def _finite(embedding) -> bool:
    return bool(embedding) and all(math.isfinite(x) for x in embedding)


def label_transcript(
    segments: list[dict],
    audio_path: str,
    out_dir: Path,
    stem: str,
    hf_token: str | None,
    anthropic_key: str | None,
    config: dict,
) -> list[dict]:
    """Diarize, name speakers found in the voice library, and merge with the transcript.

    Writes the speaker sidecar when pyannote produced voiceprints. Unmatched
    speakers keep their "Speaker N" label. The library never fails the pipeline.
    """
    from pocket_libre import diarize, voices

    voiceprints: dict = {}
    speaker_segments = diarize.diarize_auto(
        segments, audio_path=audio_path,
        hf_token=hf_token, anthropic_key=anthropic_key,
        voiceprints_out=voiceprints,
    )

    prints = {
        label: spk for label, spk in voiceprints.get("speakers", {}).items()
        if _finite(spk.get("embedding"))
    }
    if prints:
        model = voiceprints["model"]
        matches = {}
        try:
            settings = voices.load_settings(config)
        except Exception as e:
            console.print(f"[yellow]Voice settings invalid ({e}); leaving speakers unnamed.[/yellow]")
        else:
            try:
                matches = voices.match_speakers(prints, voices.load_library(), settings, model)
            except Exception as e:
                console.print(f"[yellow]Voice library unreadable ({e}); leaving speakers unnamed.[/yellow]")

        sidecar = {
            "version": SIDECAR_VERSION,
            "model": model,
            "audio": str(Path(audio_path).resolve()),
            "speakers": {
                label: {
                    "name": matches.get(label),
                    "embedding": spk["embedding"],
                    "seconds": spk["seconds"],
                }
                for label, spk in prints.items()
            },
        }
        try:
            write_sidecar(sidecar_path(out_dir, stem), sidecar)
        except OSError as e:
            console.print(f"[yellow]Could not save speaker voiceprints ({e}).[/yellow]")
        speaker_segments = [
            replace(s, speaker=matches.get(s.speaker, s.speaker))
            for s in speaker_segments
        ]
    else:
        remove_stale_sidecar(out_dir, stem)

    return diarize.merge_transcript_with_speakers(segments, speaker_segments)


# ── Confirming speakers after the fact ──────────────────────────

_SIDECAR_SUFFIX = "_speakers.json"
_GENERIC_LABEL = re.compile(r"Speaker \d+", re.IGNORECASE)
# These break "[ts] Name: text" lines and pocket-todos' owner splitting.
_BAD_NAME_CHARS = re.compile(r"[:/,&+\x00-\x1f\x7f]")
_BAD_NAME_WORD = re.compile(r"(?<!\w)and(?!\w)", re.IGNORECASE)


def find_recording(output_dir: Path, ref: str) -> Path:
    """Resolve a recording reference to its speaker sidecar path.

    ref is a path to the recording's _summary.md, _transcript.txt or
    _speakers.json, a recording id found under output_dir/<date>/, or "latest".
    """
    if ref == "latest":
        found = list(Path(output_dir).glob(f"*/*{_SIDECAR_SUFFIX}"))
        if not found:
            raise LookupError(
                f"No speaker data found in {output_dir}; "
                "re-run `pocket-libre process` on a recording to create it."
            )
        # Dated folders and PH<YYMMDDHHMMSS> ids sort chronologically; mtime would
        # jump to whichever sidecar a fix last rewrote.
        return max(found, key=lambda p: (p.parent.name, p.name))

    path = Path(ref).expanduser()
    for suffix in ("_summary.md", "_transcript.txt", _SIDECAR_SUFFIX):
        if path.name.endswith(suffix) and path.name != suffix:
            side = path.with_name(path.name[: -len(suffix)] + _SIDECAR_SUFFIX)
            if not side.exists():
                raise LookupError(
                    f"No speaker data for {path.name} ({side.name} is missing); "
                    "re-run `pocket-libre process` on the recording to create it."
                )
            return side

    matches = sorted(Path(output_dir).glob(f"*/{glob.escape(ref)}{_SIDECAR_SUFFIX}"))
    if matches:
        return matches[-1]
    raise LookupError(
        f"nothing matched {ref!r} in {output_dir}. Give a path to the recording's "
        "summary, transcript or speakers file, a recording id, or `latest`; "
        "recordings made before voice enrollment need `pocket-libre process` re-run."
    )


def relabel_text(text: str, old: str, new: str) -> str:
    """Replace whole-word occurrences of old with new ("Speaker 2" never hits "Speaker 20")."""
    pattern = r"(?<!\w)" + re.escape(old) + r"(?!\w)"
    return re.sub(pattern, lambda _m: new, text)


def relabel_speaker(text: str, targets: set[str], protected: set[str], new: str) -> str:
    """Replace whole-word occurrences of any target string with new.

    Longer strings win, so a protected name that contains a target ("Ann Lee"
    when renaming "Ann") is matched whole and left alone.
    """
    alternatives = sorted(targets | protected, key=len, reverse=True)
    pattern = r"(?<!\w)(?:" + "|".join(re.escape(a) for a in alternatives) + r")(?!\w)"
    return re.sub(pattern, lambda m: new if m.group(0) in targets else m.group(0), text)


def _write_text_atomic(path: Path, text: str) -> None:
    """Replace a text file atomically, keeping its mode and timestamps.

    The modified time tells when a meeting happened, so a relabel must not move it.
    """
    st = path.stat()
    mode = st.st_mode & 0o777
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".relabel-", suffix=".tmp")
    try:
        os.fchmod(fd, mode)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.utime(tmp, ns=(st.st_atime_ns, st.st_mtime_ns))
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def fix_speaker(sidecar: Path, speaker: str, name: str, settings, today: str) -> str:
    """Confirm who a speaker is: relabel the recording's files and teach the library.

    speaker is a sidecar label ("Speaker 2") or the name currently assigned.
    Everything is validated, and the library loaded, before anything is written,
    so a failed fix changes nothing. Returns a one-line result message.
    """
    from pocket_libre import voices

    data = read_sidecar(sidecar)
    people = data["speakers"]
    display = {label: spk.get("name") or label for label, spk in people.items()}

    label = next((l for l in people if l == speaker), None)
    if label is None:
        label = next((l for l, d in display.items() if d == speaker), None)
    if label is None:
        known = ", ".join(sorted({*people, *(d for d in display.values())}))
        raise ValueError(f"Unknown speaker {speaker!r}; this recording has: {known}")

    name = name.strip()
    if not name:
        raise ValueError("The name cannot be empty.")
    # A generic label is only a valid target to undo a wrong name on that speaker.
    undo = name.lower() == label.lower()
    if undo:
        name = label
    elif _GENERIC_LABEL.fullmatch(name):
        raise ValueError(f"{name!r} is a generic label, not a name.")
    elif _BAD_NAME_CHARS.search(name) or _BAD_NAME_WORD.search(name):
        raise ValueError(
            f"{name!r} cannot be used as a name: avoid : / , & + line breaks and the word 'and'."
        )

    lib = voices.load_library()
    if not undo:
        same = [p for p in lib["people"] if p.lower() == name.lower()]
        if len(same) > 1:
            raise ValueError(
                f"The voice library has {' and '.join(map(repr, same))}, which differ only by "
                "case; `speakers forget` one of them first."
            )
        if same:
            name = same[0]
    for other, shown in display.items():
        if other != label and shown.lower() == name.lower():
            raise ValueError(f"{name!r} is already {other} in this recording.")

    stem = sidecar.name[: -len(_SIDECAR_SUFFIX)]
    spk = people[label]
    embedding = spk.get("embedding")
    seconds = spk.get("seconds", 0.0)
    add = not undo and bool(embedding) and seconds >= settings.min_seconds
    # Whoever this speaker was filed under before is wrong now, however this fix ends.
    changed = voices.remove_sample(lib, stem, label)
    if add:
        voices.add_sample(lib, name, {
            "source": stem,
            "label": label,
            "model": data["model"],
            "date": today,
            "seconds": seconds,
            "embedding": embedding,
        })
        changed = True

    current = display[label]
    # The stable label can never be a name, so replacing it too is always safe;
    # it lets a re-run finish files an earlier failed run left half-relabelled.
    targets = {label, current}
    protected = {t for other, shown in display.items() if other != label for t in (other, shown)}
    protected -= targets
    new_texts = []
    for sibling in (f"{stem}_transcript.txt", f"{stem}_summary.md"):
        path = sidecar.with_name(sibling)
        if path.exists():
            new_texts.append((path, relabel_speaker(path.read_text(encoding="utf-8"), targets, protected, name)))
    spk["name"] = None if undo else name

    if changed:
        voices.save_library(lib)
    write_sidecar(sidecar, data)
    for path, text in new_texts:
        _write_text_atomic(path, text)

    if undo:
        return f"{current} is {label} again; the naming was undone."
    if add:
        return f"{current} is now {name}; voice sample added."
    return f"{current} is now {name}; no sample added (too little speech or no voiceprint)."
