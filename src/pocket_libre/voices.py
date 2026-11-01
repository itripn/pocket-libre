"""Voice library: remembered speaker voiceprints and cross-recording matching.

The library lives in ~/.pocket-libre/voices.json (owner-only). Each person has
a list of samples; a sample is one speaker embedding from one recording, tagged
with the embedding model id. Embeddings from different models are never compared.
"""

import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .config import CONFIG_DIR, get

MAX_SAMPLES = 20

DEFAULT_THRESHOLD = 0.70
DEFAULT_MARGIN = 0.10
DEFAULT_MIN_SECONDS = 8.0


class LibraryError(Exception):
    """voices.json exists but cannot be read as a library."""


@dataclass(frozen=True)
class VoiceSettings:
    threshold: float
    margin: float
    min_seconds: float


def load_settings(config: dict) -> VoiceSettings:
    """Read [voices] threshold|margin|min_seconds, falling back to the defaults."""
    return VoiceSettings(
        threshold=float(get(config, "voices", "threshold", default=DEFAULT_THRESHOLD)),
        margin=float(get(config, "voices", "margin", default=DEFAULT_MARGIN)),
        min_seconds=float(get(config, "voices", "min_seconds", default=DEFAULT_MIN_SECONDS)),
    )


def library_path() -> Path:
    return CONFIG_DIR / "voices.json"


def load_library() -> dict:
    """Read the library; an absent file is an empty library."""
    path = library_path()
    if not path.exists():
        return {"people": {}}
    try:
        lib = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise LibraryError(f"cannot read {path}: {e}") from e
    if not isinstance(lib, dict) or not isinstance(lib.get("people"), dict):
        raise LibraryError(f"{path} is not a voice library")
    return lib


def save_library(lib: dict) -> None:
    """Write the library atomically: file 600, directory 700."""
    path = library_path()
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path.parent, 0o700)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".voices-", suffix=".tmp")
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(lib, f)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def remove_sample(lib: dict, source: str, label: str) -> bool:
    """Remove the (source, label) sample from every person. True if any was removed."""
    removed = False
    for person in lib["people"].values():
        kept = [s for s in person["samples"] if (s["source"], s["label"]) != (source, label)]
        removed = removed or len(kept) != len(person["samples"])
        person["samples"] = kept
    return removed


def add_sample(lib: dict, name: str, sample: dict) -> None:
    """Add a confirmed sample to a person, creating them if new.

    A recording's speaker belongs to one person, so any earlier sample with the
    same (source, label) is removed first. The oldest samples beyond the cap go.
    """
    remove_sample(lib, sample["source"], sample["label"])
    samples = lib["people"].setdefault(name, {"samples": []})["samples"]
    samples.append(sample)
    del samples[:-MAX_SAMPLES]


def forget(lib: dict, name: str) -> bool:
    """Delete a person and all their samples. False if they were not known."""
    return lib["people"].pop(name, None) is not None


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


def _usable(embedding) -> bool:
    return bool(embedding) and all(math.isfinite(x) for x in embedding)


def score_speakers(speakers: dict, lib: dict, model: str) -> dict[str, dict[str, float]]:
    """Per speaker label, each known person's best cosine over their same-model samples.

    People with no same-model sample, and speakers with non-finite embeddings,
    are left out.
    """
    scores: dict[str, dict[str, float]] = {}
    for label, spk in speakers.items():
        emb = spk.get("embedding")
        if not _usable(emb):
            continue
        per_person: dict[str, float] = {}
        for name, person in lib["people"].items():
            best = None
            for s in person["samples"]:
                if s.get("model") != model or not _usable(s.get("embedding")):
                    continue
                if len(s["embedding"]) != len(emb):
                    continue
                c = cosine(emb, s["embedding"])
                if best is None or c > best:
                    best = c
            if best is not None:
                per_person[name] = best
        scores[label] = per_person
    return scores


def match_speakers(speakers: dict, lib: dict, settings: VoiceSettings,
                   model: str) -> dict[str, str]:
    """Name the speakers whose voice clearly matches a known person.

    A match needs the threshold and a margin over the runner-up person, and over
    any other speaker in the recording who also scores as that person. Speakers
    with too little speech are skipped. No name is given to two speakers.
    """
    eligible = {
        label: spk for label, spk in speakers.items()
        if spk.get("seconds", 0.0) >= settings.min_seconds
    }
    scores = score_speakers(eligible, lib, model)
    candidates = []
    for label, per_person in scores.items():
        ranked = sorted(per_person.items(), key=lambda kv: kv[1], reverse=True)
        if not ranked:
            continue
        name, best = ranked[0]
        if best < settings.threshold:
            continue
        if len(ranked) > 1 and best - ranked[1][1] < settings.margin:
            continue
        # Two similar voices in one recording can both look like the same person.
        if any(
            other != label and rival.get(name, -1.0) >= settings.threshold
            and best - rival[name] < settings.margin
            for other, rival in scores.items()
        ):
            continue
        candidates.append((best, label, name))

    matched: dict[str, str] = {}
    used = set()
    for _, label, name in sorted(candidates, reverse=True):
        if name not in used:
            matched[label] = name
            used.add(name)
    return matched
