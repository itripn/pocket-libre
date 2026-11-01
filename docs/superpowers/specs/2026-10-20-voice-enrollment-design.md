# Voice enrollment (speaker recognition) - design

Status: draft for approval, 2026-10-20. Nothing built yet.

## Why

pocket-libre labels speakers `Speaker 1`, `Speaker 2` per recording, so the labels mean nothing across recordings. The visible damage: `pocket-todos` cannot tell which action items are Ron's. In a dry run over one day's meetings, about two thirds of extracted tasks were flagged `Owner unclear: Speaker N`, and several plainly belonged to the other person.

Ron wants the system to recognise his own voice and the voices of people he meets often, getting better as it hears more. Labels then feed action-item ownership. Later, they feed search for what a given person said.

## Goals

- Recognise Ron and frequent colleagues across recordings, using the speaker voiceprints pyannote already computes.
- Improve with use: every confirmed correction adds a sample to that person's profile.
- Let Ron correct a wrong or missing label from the CLI, and have that correction teach the library.
- Make `pocket-todos` use real names: Ron's tasks go in clean, other named people's tasks are dropped, unmatched `Speaker N` tasks stay flagged as today.

## Non-goals (this phase)

- Search by speaker. Later phase, once labels prove reliable.
- Web UI for corrections. CLI first.
- Cloud anything. Voiceprints never leave repromax and are never sent to kengram, Strata or Claude. Transcript text still goes to Claude for the summary, as it does today.
- Changing tasks already created in Structured after a later correction.
- Identity from context alone. A name heard in the audio ("thanks, John") never applies a label by itself.

## Anti-goals (would count as failure even if it works)

- A wrong name shown with confidence, or learned silently. A wrong label that spreads into summaries and the inbox is worse than `Speaker 2`.
- Voiceprints readable by other users or committed to git.
- A setup ritual before anything works.

## Behaviour Ron decided

- Uncertain voice: show `Speaker N` and stay out of the way. No `John?` labels. Only a clear match gets a name.
- Corrections come from a CLI command. Guessing is allowed, but guesses are not trusted until Ron confirms them.
- Unmatched speakers stay flagged in `pocket-todos` exactly as today.

## Design

### 1. Keep the voiceprints
`diarize_pyannote` already receives per-speaker embeddings (`DiarizeOutput.speaker_embeddings`) and discards them. Keep them: for each recording, write a sidecar `<ts>_speakers.json` next to the transcript holding, per local label (`Speaker 1`...), the embedding, the total speaking time, and the embedding model id. This sidecar is what makes a later correction possible on an old recording. Recordings processed before this feature have no sidecar; re-running `pocket-libre process` on them creates one.

### 2. Voice library
A private file `~/.pocket-libre/voices.json`, mode 600, owner-only directory. For each person: name, and a list of samples (embedding, source recording, date, speaking seconds). Every sample carries the embedding model id; samples from a different model are never compared (pyannote model changes make old vectors meaningless).

### 3. Matching
After diarization, compare each speaker's embedding to each person's samples by cosine similarity. A match must clear a threshold AND beat the runner-up by a margin; otherwise the label stays `Speaker N`. A speaker with very little speech (seconds, builder to pick) is never matched. Two speakers in one recording can never get the same name. The threshold and margin are configuration with conservative defaults and are calibrated on Ron's real recordings during the build.

### 4. Correction commands
- `pocket-libre speakers fix <recording> "Speaker 2" John`: relabels that speaker in the recording's transcript and summary, and adds the speaker's embedding to John's profile (creating John if new). This is the only operation that teaches the library.
- `pocket-libre speakers list`: who is known, sample count per person, last heard.
- `pocket-libre speakers forget <name>`: deletes a person and all their samples. Voiceprints are biometric data, so deletion has to exist.
- `fix` also works for correcting a wrong named label (e.g. matched as John, actually Mary): the wrong sample is not added to John, and if the earlier match had been learned, removing it is part of `fix`.

### 5. First-time setup for Ron
No separate enrolment step. Ron runs `speakers fix` once on a recording where he knows which speaker is him, with the name "Ron". After that, matching takes over.

### 6. Where labels flow
Named labels replace `Speaker N` in the labeled transcript before the summary is generated, so the Claude summary says "Ron" and "John" naturally. `pocket-todos` needs no new logic: its existing owner policy already includes tasks owned by `POCKET_TODOS_OWNER` (Ron), drops other names, and flags `Speaker N`. The expected effect is fewer flagged tasks, and other people's tasks dropped instead of flagged.

## Constraints

- Existing stack: pyannote 4.x on `mps` with CPU fallback, merge by max overlap, Python, pytest. Follow current code style. TDD.
- Library and sidecars are written atomically (no half-written files).
- Hook behaviour and `pocket-todos` stay unchanged.
- Everything uncommitted until Ron asks for a commit.

## Left to the builder

Threshold, margin and minimum-speech values; sample storage (keep all, cap, or running centroid); the exact sidecar and library schemas; command-line ergonomics for naming a recording (path, timestamp id, or "latest").

## My calls (Ron to check)

1. A name heard in the audio is not used automatically. Ron said guessing from context was fine, and also said to stay out of the way when unsure; I resolved that in favour of the second, so context guesses do not change labels. They could be shown as hints in `speakers fix` output later.
2. `speakers list` and `speakers forget` are in scope although Ron did not ask for them.
3. No automatic backfill of old recordings.
4. Mode 600 file under `~/.pocket-libre/`, never in git or kengram.
