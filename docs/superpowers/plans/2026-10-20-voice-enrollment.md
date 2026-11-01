# Voice Enrollment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** pocket-libre recognises Ron and frequent colleagues across recordings from their voiceprints, names only clear matches, and lets Ron correct labels from the CLI, with each correction teaching the voice library.

**Architecture:** `diarize_pyannote` already receives per-speaker embeddings and drops them; keep them in a per-recording sidecar (`<stem>_speakers.json`). A private voice library (`~/.pocket-libre/voices.json`) holds named samples. `speakers.label_transcript` replaces the three duplicated diarize-and-merge blocks (cli `sync`, cli `process`, `watch`): it diarizes, writes the sidecar, names clear matches, and merges. A `speakers` click group (`fix`, `list`, `forget`, `scores`) edits labels and the library. `pocket-todos` is unchanged.

**Tech Stack:** Python, pyannote.audio 4.x (`DiarizeOutput.speaker_embeddings`), numpy (already a pyannote dependency), click, rich, pytest. Tests run with `/Users/rjf/code/pocket/pocket-libre/.venv/bin/python -m pytest -q -p no:cacheprovider` (baseline 384 passing).

**Spec:** `docs/superpowers/specs/2026-10-20-voice-enrollment-design.md`

## Global Constraints

- Voiceprints never leave repromax: not in kengram, Strata, Claude prompts, or git.
- `voices.json` and every `*_speakers.json` sidecar are mode 600; the directory holding `voices.json` is 700. All writes are atomic (write a temp file in the same directory, then `os.replace`).
- A voice that is not a clear match keeps its `Speaker N` label. No `John?` labels. Only `speakers fix` adds samples to the library.
- Embeddings are only ever compared when their `model` id is equal.
- Two speakers in one recording never get the same name.
- Existing behaviour is unchanged when no voice library exists, no HF token is set, or any voice step fails: the pipeline must still produce a transcript and summary.
- Follow existing style (rich `console.print` with `[yellow]` for warnings). TDD. No commits unless Ron asks; the checkpoint after each task is a full-suite run.
- Defaults (placeholders, calibrated in Task 6): `threshold = 0.70`, `margin = 0.10`, `min_seconds = 8.0`, at most `20` samples per person.

## Review Focus

- `voices.json` is corrupt or unreadable: the pipeline still labels `Speaker N` with a yellow warning; `fix` refuses to overwrite a file it cannot parse.
- Library holds samples from a different embedding model: they are ignored, nothing crashes.
- pyannote returns no embeddings, or a count that does not match the speakers: no sidecar entries and no names, labels unchanged.
- `fix` names that would break text replacement (`Dr. O'Neil (Sr)`, a name with a backslash, `Speaker 20` next to `Speaker 2`) and an empty or whitespace name.
- `fix` to a name another speaker in the same recording already has, and `fix` on a label that does not exist (error lists the labels that do).
- A recording re-processed after being fixed: the library now names the speaker again; the fix is idempotent.

## File Structure

- `src/pocket_libre/diarize.py` (modify): `diarize_pyannote` and `diarize_auto` gain an optional `voiceprints_out` parameter.
- `src/pocket_libre/voices.py` (create): library file I/O, sample management, cosine matching, settings. No pipeline knowledge.
- `src/pocket_libre/speakers.py` (create): sidecar I/O, `label_transcript`, recording lookup, relabelling, the `fix` operation.
- `src/pocket_libre/cli.py` (modify): `speakers` command group; replace two diarize blocks with `label_transcript`.
- `src/pocket_libre/watch.py` (modify): replace the diarize block with `label_transcript`.
- Tests: `tests/test_diarize.py` (extend), `tests/test_voices.py`, `tests/test_speakers.py`, `tests/test_speakers_cli.py` (create).
- `README.md`, `CHANGELOG.md` (modify).

---

### Task 1: Keep the voiceprints pyannote computes

**Files:**
- Modify: `src/pocket_libre/diarize.py` (`diarize_pyannote` line 41, `diarize_auto` line 251)
- Test: `tests/test_diarize.py`

**Interfaces:**
- Produces: `diarize_pyannote(audio_path: str, hf_token: str, voiceprints_out: dict | None = None) -> list[SpeakerSegment]` and `diarize_auto(..., voiceprints_out: dict | None = None)`. When `voiceprints_out` is a dict, pyannote fills it as `{"model": str, "speakers": {"Speaker 1": {"embedding": list[float], "seconds": float}, ...}}` (keys are the friendly names, `seconds` is the speaker's summed turn duration). Left empty if embeddings are unavailable or their count differs from the speaker count. Other diarizers never touch it.

- [ ] **Step 1: Write failing tests** in `tests/test_diarize.py`, extending the existing fake-pyannote helper: `test_voiceprints_out_filled_from_diarize_output` (fake `DiarizeOutput` with `speaker_embeddings` rows aligned with `labels()` order; assert `voiceprints_out["speakers"]["Speaker 1"]["embedding"] == [..]` for the speaker that speaks first even when `labels()` sorts it second, and `seconds` equals the summed turn lengths, and `model` equals the model id used); `test_voiceprints_out_empty_when_embedding_count_mismatches`; `test_voiceprints_out_empty_when_no_embeddings` (3.x-style `Annotation` return, fake pipeline called with `return_embeddings=True` returning `(annotation, embeddings)` is covered by the first case in a second parametrisation); `test_diarize_auto_passes_voiceprints_out`; `test_voiceprints_out_none_changes_nothing`.
- [ ] **Step 2: Run** `pytest tests/test_diarize.py -q`. Expected: the new tests FAIL (unexpected keyword / empty dict), existing ones pass.
- [ ] **Step 3: Implement.** For 4.x read `output.speaker_embeddings`; for 3.x call `pipeline(audio_path, return_embeddings=True)` only when `voiceprints_out is not None` and the call signature accepts it, unpacking `(annotation, embeddings)`. Embedding row `i` belongs to `diarization.labels()[i]`; map through the existing friendly-name map. Convert with `.tolist()`.
- [ ] **Step 4: Run** the new tests: PASS. Then the full suite: 384 + new, all pass.
- [ ] **Step 5: Real-engine smoke check.** With the venv python, run `diarize_pyannote` on a short real clip from `~/Pocket Libre/` (copy a 2-3 minute slice to the scratchpad) with `voiceprints_out={}`; confirm one embedding per found speaker and equal dimensions. Record the dimension in the commit-free notes for Task 3.

### Task 2: Voice library and matching

**Files:**
- Create: `src/pocket_libre/voices.py`
- Test: `tests/test_voices.py`

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True) class VoiceSettings: threshold: float; margin: float; min_seconds: float` and `load_settings(config: dict) -> VoiceSettings` (reads `[voices] threshold|margin|min_seconds` via `config.get`, falling back to the defaults above).
  - `library_path() -> Path` (`CONFIG_DIR / "voices.json"`; tests monkeypatch it).
  - `load_library() -> dict` returns `{"people": {name: {"samples": [sample, ...]}}}` or an empty library when the file is absent; raises `LibraryError(Exception)` when the file exists but cannot be parsed.
  - `save_library(lib: dict) -> None` atomic, file 600, directory 700.
  - `sample` is `{"embedding": list[float], "model": str, "source": str, "label": str, "date": str, "seconds": float}`.
  - `add_sample(lib: dict, name: str, sample: dict) -> None`: first removes any sample with the same `(source, label)` from every person (fixes a wrong earlier correction), appends, then drops the oldest beyond 20.
  - `forget(lib: dict, name: str) -> bool`.
  - `cosine(a: list[float], b: list[float]) -> float`.
  - `match_speakers(speakers: dict, lib: dict, settings: VoiceSettings, model: str) -> dict[str, str]`: `speakers` is the sidecar `speakers` mapping; returns `{label: name}` for clear matches only.

- [ ] **Step 1: Write failing tests** (`tests/test_voices.py`): `test_load_missing_returns_empty`; `test_load_corrupt_raises_library_error`; `test_save_sets_modes_600_and_700` (stat the file and directory); `test_save_is_atomic` (failure in `os.replace` leaves the old file intact); `test_add_sample_replaces_same_source_label_from_other_person`; `test_add_sample_caps_at_20_dropping_oldest`; `test_forget_removes_person_and_reports`; `test_cosine_identical_is_one_orthogonal_is_zero`; `test_match_names_clear_match`; `test_match_needs_threshold` (best score 0.69 -> no name); `test_match_needs_margin_over_runner_up` (0.80 vs 0.75 -> no name; a lone known person passes on threshold alone); `test_match_ignores_other_model_samples`; `test_match_skips_speakers_under_min_seconds`; `test_match_never_gives_one_name_to_two_speakers` (higher score wins, the other stays unmatched); `test_person_score_is_best_sample`.
- [ ] **Step 2: Run** `pytest tests/test_voices.py -q`. Expected: FAIL (module missing).
- [ ] **Step 3: Implement** `voices.py` to the interfaces above. Atomic write: `tempfile.mkstemp` in the library directory, `os.fchmod(fd, 0o600)`, write, `os.replace`. Greedy assignment: sort all (label, name, score) candidates passing threshold and margin by score descending, take each if neither label nor name is used.
- [ ] **Step 4: Run** the new file, then the full suite: all pass.

### Task 3: Sidecar, `label_transcript`, and wiring into the pipeline

**Files:**
- Create: `src/pocket_libre/speakers.py`
- Modify: `src/pocket_libre/cli.py:1148-1157` (`sync`), `src/pocket_libre/cli.py:1246-1254` (`process`), `src/pocket_libre/watch.py:171-187`
- Test: `tests/test_speakers.py`

**Interfaces:**
- Consumes: `diarize_auto(..., voiceprints_out)`, `merge_transcript_with_speakers`, `voices.load_library`, `voices.match_speakers`, `voices.load_settings`, `voices.LibraryError`.
- Produces:
  - `sidecar_path(out_dir: Path, stem: str) -> Path` = `out_dir / f"{stem}_speakers.json"`.
  - Sidecar JSON: `{"version": 1, "model": str, "audio": str, "speakers": {"Speaker 1": {"name": "Ron" | null, "embedding": [...], "seconds": float}}}`. Written mode 600, atomically.
  - `read_sidecar(path: Path) -> dict`, `write_sidecar(path: Path, data: dict) -> None`.
  - `label_transcript(segments: list[dict], audio_path: str, out_dir: Path, stem: str, hf_token: str | None, anthropic_key: str | None, config: dict) -> list[dict]`: runs `diarize_auto` with a `voiceprints_out`, writes the sidecar when voiceprints exist, renames clear matches in the speaker segments, and returns `merge_transcript_with_speakers(segments, speaker_segments)`. Unmatched speakers keep `Speaker N`. A `LibraryError` is printed as a yellow warning and treated as an empty library. The call sites keep their existing `try/except` fallbacks around it.

- [ ] **Step 1: Write failing tests** with a fake `diarize_auto` and a monkeypatched `library_path`: `test_label_transcript_names_matched_speaker_and_keeps_other_as_speaker_n` (library holds Ron; speaker 1 matches, speaker 2 stays `Speaker 2`, not renumbered); `test_label_transcript_writes_private_sidecar` (mode 600, contains both embeddings and `name`); `test_label_transcript_without_library_changes_nothing`; `test_label_transcript_survives_corrupt_library` (labels `Speaker N`, sidecar still written, yellow warning printed); `test_label_transcript_without_voiceprints_writes_no_sidecar` (LLM or simple diarizer path).
- [ ] **Step 2: Run** `pytest tests/test_speakers.py -q`. Expected: FAIL (module missing).
- [ ] **Step 3: Implement** `speakers.py` as specified. In `cli.py` `sync`/`process` and `watch._process_recording`, replace the `diarize_auto` + `merge_transcript_with_speakers` pair with `label_transcript(...)` using `rec_dir`/`out_dir` and `rec.timestamp`/`stem`/`timestamp` as the stem, passing the loaded config.
- [ ] **Step 4: Run** the new tests and the full suite. The existing hook and pipeline tests must pass unchanged.

### Task 4: Relabel and `speakers fix`

**Files:**
- Modify: `src/pocket_libre/speakers.py`, `src/pocket_libre/cli.py` (new `speakers` click group, command `fix`)
- Test: `tests/test_speakers.py`, `tests/test_speakers_cli.py`

**Interfaces:**
- Consumes: Task 2 library functions; Task 3 sidecar functions.
- Produces:
  - `find_recording(output_dir: Path, ref: str) -> Path` returns the sidecar path. `ref` is a path to any of `*_summary.md`, `*_transcript.txt`, `*_speakers.json`; a recording id like `PH261020213653` found under `output_dir/*/`; or `latest` (the newest sidecar by mtime). Raises `LookupError` with a readable message when nothing matches.
  - `relabel_text(text: str, old: str, new: str) -> str` replaces whole-word occurrences (`\bold\b` semantics that also hold for names like `Speaker 2` vs `Speaker 20`), using `re.escape` on `old` and a function replacement so `new` is never interpreted as a regex template.
  - `fix_speaker(sidecar: Path, speaker: str, name: str, settings: VoiceSettings, today: str) -> str` where `speaker` is either a sidecar label (`Speaker 2`) or a name currently assigned. Steps: validate `name` (non-empty after strip, not equal to another speaker's current display name in this recording); relabel the sibling `<stem>_transcript.txt` and `<stem>_summary.md` when they exist; set the sidecar speaker's `name`; add the sample to the library unless `seconds < min_seconds` (still relabels, returns a message saying no sample was added); returns a one-line result message. An unknown `speaker` raises `ValueError` listing the labels that exist.
  - CLI: `pocket-libre speakers fix <recording> <speaker> <name>`, resolving the output directory the way other commands do (`config.get_output_dir`).

- [ ] **Step 1: Write failing tests:** `test_find_recording_by_id_path_and_latest`; `test_find_recording_unknown_lists_nothing_matched`; `test_relabel_whole_word_only` (`Speaker 2` changes, `Speaker 20` does not); `test_relabel_name_with_regex_chars_and_backslash` (`Dr. O'Neil (Sr)`, `A\1B`); `test_fix_relabels_transcript_and_summary_and_sidecar`; `test_fix_adds_sample_with_model_source_label`; `test_fix_short_speech_relabels_but_adds_no_sample`; `test_fix_is_idempotent`; `test_fix_by_current_name_confirms_and_adds_sample`; `test_fix_moves_sample_from_wrong_person`; `test_fix_rejects_empty_name_and_name_held_by_other_speaker`; `test_fix_unknown_label_lists_available`; `test_fix_refuses_corrupt_library_without_touching_files`; CLI test via `click.testing.CliRunner`: exit 0 and message on success, exit 1 with the message on `LookupError`/`ValueError`.
- [ ] **Step 2: Run** both files. Expected: FAIL.
- [ ] **Step 3: Implement** to the interfaces. Do all validation and the library load before writing any file, so a failed `fix` changes nothing.
- [ ] **Step 4: Run** new tests and full suite: all pass.

### Task 5: `speakers list`, `forget`, `scores`

**Files:**
- Modify: `src/pocket_libre/cli.py` (`speakers` group)
- Test: `tests/test_speakers_cli.py`

**Interfaces:**
- Consumes: `voices.load_library`, `voices.forget`, `voices.save_library`, `voices.cosine`, `speakers.find_recording`, `speakers.read_sidecar`.
- Produces: `speakers list` (table: name, sample count, most recent sample date; says so when the library is empty); `speakers forget <name>` (confirms before deleting, `--yes` skips; exit 1 and a message when the name is unknown); `speakers scores <recording>` (read-only table of each sidecar speaker's score against each known person, plus whether it would be named under current settings; used for calibration).

- [ ] **Step 1: Write failing CLI tests:** `test_list_empty_and_populated`; `test_forget_requires_confirmation_and_deletes_samples_from_file`; `test_forget_yes_flag`; `test_forget_unknown_name_exits_1`; `test_scores_shows_scores_and_would_name_without_modifying_files` (compare file bytes before and after).
- [ ] **Step 2: Run** `pytest tests/test_speakers_cli.py -q`. Expected: FAIL for the new tests.
- [ ] **Step 3: Implement** the three commands with rich tables, matching the style of existing commands such as `list`/`status`.
- [ ] **Step 4: Run** new tests and full suite: all pass.

### Task 6: Docs, calibration, and acceptance (Ron's real recordings)

**Files:**
- Modify: `README.md` (section "Speaker recognition": what is stored and where, the four commands, enrolling as "Ron", that sidecars contain voiceprints), `CHANGELOG.md` (Added), and the `[voices]` defaults if calibration changes them.

- [ ] **Step 1: Docs.** Write the README section and CHANGELOG entry. State plainly that `~/Pocket Libre/**/_speakers.json` files are voiceprints and should not be shared.
- [ ] **Step 2: Create sidecars for real recordings.** Re-run `pocket-libre process` on four to six of today's recordings that include Ron (it re-summarises; the summary hook is still off). Confirm each has a `_speakers.json` with mode 600.
- [ ] **Step 3: Enroll Ron.** Run `pocket-libre speakers fix <recording> "Speaker N" Ron` on a recording where Ron knows which speaker is him. The name must be exactly `Ron` (`pocket-todos` matches its `POCKET_TODOS_OWNER`). Enroll one or two frequent colleagues the same way.
- [ ] **Step 4: Calibrate.** Run `speakers scores` on the remaining recordings; compare Ron's score on recordings he is in against recordings he is not in. Set `threshold`, `margin`, `min_seconds` so no stranger is named and Ron usually is. Report the numbers to Ron before changing the defaults.
- [ ] **Step 5: End to end.** Re-process one recording; confirm Ron's lines are labelled `Ron`, the summary says "Ron", and `pocket-todos --dry-run` on a scratch copy of that summary creates Ron's tasks without `Owner unclear`, drops other named people's tasks, and still flags unmatched `Speaker N`.
- [ ] **Step 6: Final full-suite run;** report the counts and any failures by name.

---

## Self-review notes

- Spec coverage: voiceprint capture (Task 1), library, privacy, model separation, caps (Task 2), matching with threshold/margin/min-speech/unique names (Task 2), sidecar and flow into transcript and summary (Task 3), `fix`/`list`/`forget` and first-time enrolment (Tasks 4-5, 6), todo ownership without code change (Task 6 step 5), search later and web UI later (not in plan, per spec).
- Additions beyond the spec, flagged for Ron: `speakers scores` (calibration needs visibility into scores); sidecars are mode 600 because they hold voiceprints too; `fix` on a very short speaker relabels without adding a sample.
