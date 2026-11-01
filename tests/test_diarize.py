"""Speaker ID: the Claude path survives truncation, the pyannote path survives its 4.x API."""

import sys
import types
from collections import namedtuple

import pytest

from pocket_libre import diarize

Turn = namedtuple("Turn", "start end")


def _segments(n):
    return [{"start": float(i), "end": float(i) + 1.0, "text": f" line {i}"} for i in range(n)]


def _speakers(segments):
    return [s.speaker for s in segments]


# ── Claude path ─────────────────────────────────


class _FakeAnthropic:
    """Stands in for the anthropic module; records what diarize_llm asked for."""

    def __init__(self, text, stop_reason="end_turn"):
        self.text = text
        self.stop_reason = stop_reason
        self.requests = []

    def install(self, monkeypatch):
        outer = self

        class _Messages:
            def create(self, **kwargs):
                outer.requests.append(kwargs)
                usage = types.SimpleNamespace(input_tokens=10, output_tokens=5)
                block = types.SimpleNamespace(text=outer.text)
                return types.SimpleNamespace(
                    content=[block], usage=usage, stop_reason=outer.stop_reason)

        class _Client:
            def __init__(self, api_key=None):
                self.messages = _Messages()

        module = types.ModuleType("anthropic")
        module.Anthropic = _Client
        monkeypatch.setitem(sys.modules, "anthropic", module)
        return self


def test_complete_reply_labels_each_stretch(monkeypatch):
    reply = '[{"line": 0, "speaker": "Ann"}, {"line": 2, "speaker": "Bob"}]'
    _FakeAnthropic(reply).install(monkeypatch)
    result = diarize.diarize_llm(_segments(4), "key")
    assert _speakers(result) == ["Ann", "Ann", "Bob", "Bob"]


def test_reply_wrapped_in_prose_and_a_fence_still_parses(monkeypatch):
    reply = 'Here you go:\n```json\n[{"line": 0, "speaker": "Ann"}]\n```'
    _FakeAnthropic(reply).install(monkeypatch)
    assert set(_speakers(diarize.diarize_llm(_segments(3), "key"))) == {"Ann"}


def test_truncated_reply_keeps_every_complete_entry(monkeypatch):
    """The model ran out of tokens mid-object; what it did finish is still good."""
    reply = (
        '[\n  {"line": 0, "speaker": "Speaker 1"},\n  {"line": 3, "speaker": "Speaker 2"},\n'
        '  {"line": 5, "speaker": "Speaker 1"},\n  {'
    )
    _FakeAnthropic(reply, stop_reason="max_tokens").install(monkeypatch)
    result = diarize.diarize_llm(_segments(8), "key")
    assert _speakers(result) == [
        "Speaker 1", "Speaker 1", "Speaker 1",
        "Speaker 2", "Speaker 2",
        "Speaker 1", "Speaker 1", "Speaker 1",
    ]


def test_truncated_reply_says_so(monkeypatch, capsys):
    reply = '[{"line": 0, "speaker": "Ann"}, {"line": 1, "spe'
    _FakeAnthropic(reply, stop_reason="max_tokens").install(monkeypatch)
    diarize.diarize_llm(_segments(3), "key")
    assert "cut off" in capsys.readouterr().out


def test_truncated_reply_with_nothing_usable_falls_back_to_simple(monkeypatch):
    _FakeAnthropic('[{"line": 0, "spea', stop_reason="max_tokens").install(monkeypatch)
    result = diarize.diarize_llm(_segments(3), "key")
    assert _speakers(result) == ["Speaker", "Speaker", "Speaker"]


def test_non_array_reply_falls_back_to_simple(monkeypatch):
    _FakeAnthropic('{"line": 0, "speaker": "Ann"}').install(monkeypatch)
    assert set(_speakers(diarize.diarize_llm(_segments(3), "key"))) == {"Speaker"}


def test_long_transcripts_are_given_room_to_answer(monkeypatch):
    """One entry per speaker change is ~12 tokens; 1500 ran out on a 24-minute recording."""
    fake = _FakeAnthropic('[{"line": 0, "speaker": "Ann"}]').install(monkeypatch)
    diarize.diarize_llm(_segments(400), "key")
    assert fake.requests[-1]["max_tokens"] >= 400 * 12


def test_short_transcripts_keep_the_old_floor(monkeypatch):
    fake = _FakeAnthropic('[{"line": 0, "speaker": "Ann"}]').install(monkeypatch)
    diarize.diarize_llm(_segments(5), "key")
    assert fake.requests[-1]["max_tokens"] >= 1500


# ── pyannote path ───────────────────────────────


class _Annotation:
    def __init__(self, turns):
        self._turns = turns

    def itertracks(self, yield_label=False):
        for start, end, label in self._turns:
            yield Turn(start, end), None, label

    def labels(self):
        return sorted({label for _, _, label in self._turns})


TURNS = [(0.0, 2.0, "SPEAKER_01"), (2.0, 4.0, "SPEAKER_00"), (4.0, 6.0, "SPEAKER_01")]


class _FakePipeline:
    def __init__(self, result, fail_on=None):
        self.result = result
        self.moved_to = []
        self.fail_on = fail_on  # device name that makes a run blow up
        self.embeddings = None  # what a 3.x run returns alongside the annotation
        self.call_kwargs = {}

    def to(self, device):
        self.moved_to.append(device)
        return self

    def __call__(self, audio_path, **kwargs):
        if self.fail_on and self.moved_to and self.moved_to[-1] == f"device:{self.fail_on}":
            raise RuntimeError("MPS does not support this op")
        self.call_kwargs = kwargs
        if kwargs.get("return_embeddings"):
            return self.result, self.embeddings
        return self.result


def _install_pyannote(monkeypatch, *, new_api, pipeline=None, mps=True, cuda=False):
    """Fake pyannote.audio and torch. new_api mimics 4.x: token= and a DiarizeOutput."""
    calls = {}
    annotation = _Annotation(TURNS)
    result = types.SimpleNamespace(speaker_diarization=annotation) if new_api else annotation
    pipe = pipeline or _FakePipeline(result)

    class Pipeline:
        if new_api:
            @staticmethod
            def from_pretrained(checkpoint, token=None):
                calls.update(checkpoint=checkpoint, token=token)
                return pipe
        else:
            @staticmethod
            def from_pretrained(checkpoint, use_auth_token=None):
                calls.update(checkpoint=checkpoint, token=use_auth_token)
                return pipe

    audio = types.ModuleType("pyannote.audio")
    audio.Pipeline = Pipeline
    monkeypatch.setitem(sys.modules, "pyannote", types.ModuleType("pyannote"))
    monkeypatch.setitem(sys.modules, "pyannote.audio", audio)

    torch = types.ModuleType("torch")
    torch.device = lambda name: f"device:{name}"
    torch.backends = types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: mps))
    torch.cuda = types.SimpleNamespace(is_available=lambda: cuda)
    monkeypatch.setitem(sys.modules, "torch", torch)
    return pipe, calls


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    monkeypatch.delenv("POCKET_LIBRE_DIARIZE_DEVICE", raising=False)
    monkeypatch.delenv("POCKET_LIBRE_DIARIZE_MODEL", raising=False)


@pytest.mark.parametrize("new_api", [True, False], ids=["pyannote-4", "pyannote-3"])
def test_pyannote_works_with_both_api_generations(monkeypatch, new_api):
    _, calls = _install_pyannote(monkeypatch, new_api=new_api)
    segments = diarize.diarize_pyannote("a.mp3", "hf_token")
    assert calls["token"] == "hf_token"
    assert [(s.start, s.end, s.speaker) for s in segments] == [
        (0.0, 2.0, "Speaker 1"), (2.0, 4.0, "Speaker 2"), (4.0, 6.0, "Speaker 1"),
    ]


def test_model_can_be_chosen_with_an_env_var(monkeypatch):
    _, calls = _install_pyannote(monkeypatch, new_api=True)
    monkeypatch.setenv("POCKET_LIBRE_DIARIZE_MODEL", "pyannote/speaker-diarization-community-1")
    diarize.diarize_pyannote("a.mp3", "hf_token")
    assert calls["checkpoint"] == "pyannote/speaker-diarization-community-1"


def test_default_model_is_unchanged(monkeypatch):
    _, calls = _install_pyannote(monkeypatch, new_api=True)
    diarize.diarize_pyannote("a.mp3", "hf_token")
    assert calls["checkpoint"] == "pyannote/speaker-diarization-3.1"


# ── Device choice ───────────────────────────────


@pytest.mark.parametrize(
    "mps,cuda,expected",
    [(True, False, "mps"), (False, True, "cuda"), (True, True, "cuda"), (False, False, "cpu")],
)
def test_device_prefers_the_fastest_available(monkeypatch, mps, cuda, expected):
    pipe, _ = _install_pyannote(monkeypatch, new_api=True, mps=mps, cuda=cuda)
    diarize.diarize_pyannote("a.mp3", "hf_token")
    assert pipe.moved_to[0] == f"device:{expected}"


def test_device_can_be_forced_with_an_env_var(monkeypatch):
    pipe, _ = _install_pyannote(monkeypatch, new_api=True, mps=True)
    monkeypatch.setenv("POCKET_LIBRE_DIARIZE_DEVICE", "cpu")
    diarize.diarize_pyannote("a.mp3", "hf_token")
    assert pipe.moved_to[0] == "device:cpu"


def test_a_gpu_failure_retries_on_the_cpu(monkeypatch):
    """Some ops are missing on MPS; the run should finish on the CPU, not die."""
    annotation = _Annotation(TURNS)
    pipe = _FakePipeline(types.SimpleNamespace(speaker_diarization=annotation), fail_on="mps")
    _install_pyannote(monkeypatch, new_api=True, pipeline=pipe, mps=True)
    segments = diarize.diarize_pyannote("a.mp3", "hf_token")
    assert len(segments) == 3
    assert pipe.moved_to == ["device:mps", "device:cpu"]


def test_a_gated_model_says_what_to_do(monkeypatch):
    """pyannote returns None, not an error, when the HF account hasn't accepted the terms."""
    pipe, _ = _install_pyannote(monkeypatch, new_api=True)
    sys.modules["pyannote.audio"].Pipeline.from_pretrained = staticmethod(
        lambda checkpoint, token=None: None)
    with pytest.raises(RuntimeError) as err:
        diarize.diarize_pyannote("a.mp3", "hf_token")
    assert "huggingface.co/pyannote/speaker-diarization-3.1" in str(err.value)


# ── Voiceprints ─────────────────────────────────


class _Rows:
    """A tiny stand-in for the numpy array pyannote returns."""

    def __init__(self, rows):
        self._rows = rows

    def __len__(self):
        return len(self._rows)

    def __getitem__(self, i):
        return types.SimpleNamespace(tolist=lambda: list(self._rows[i]))


# labels() sorts SPEAKER_00 first, but SPEAKER_01 speaks first (and becomes "Speaker 1").
ROWS = [[0.0, 1.0], [2.0, 3.0]]


def _output_with(embeddings):
    return types.SimpleNamespace(
        speaker_diarization=_Annotation(TURNS), speaker_embeddings=embeddings)


def test_voiceprints_out_filled_from_diarize_output(monkeypatch):
    pipe = _FakePipeline(_output_with(_Rows(ROWS)))
    _install_pyannote(monkeypatch, new_api=True, pipeline=pipe)
    monkeypatch.setenv("POCKET_LIBRE_DIARIZE_MODEL", "pyannote/speaker-diarization-community-1")
    out = {}
    diarize.diarize_pyannote("a.mp3", "hf_token", voiceprints_out=out)
    assert out["model"] == "pyannote/speaker-diarization-community-1"
    assert out["speakers"] == {
        "Speaker 1": {"embedding": [2.0, 3.0], "seconds": 4.0},
        "Speaker 2": {"embedding": [0.0, 1.0], "seconds": 2.0},
    }


def test_voiceprints_out_filled_on_pyannote_3(monkeypatch):
    pipe = _FakePipeline(_Annotation(TURNS))
    pipe.embeddings = _Rows(ROWS)
    _install_pyannote(monkeypatch, new_api=False, pipeline=pipe)
    out = {}
    diarize.diarize_pyannote("a.mp3", "hf_token", voiceprints_out=out)
    assert pipe.call_kwargs == {"return_embeddings": True}
    assert out["model"] == "pyannote/speaker-diarization-3.1"
    assert out["speakers"]["Speaker 1"] == {"embedding": [2.0, 3.0], "seconds": 4.0}
    assert out["speakers"]["Speaker 2"] == {"embedding": [0.0, 1.0], "seconds": 2.0}


def test_voiceprints_out_empty_when_embedding_count_mismatches(monkeypatch):
    pipe = _FakePipeline(_output_with(_Rows([[0.0, 1.0]])))
    _install_pyannote(monkeypatch, new_api=True, pipeline=pipe)
    out = {}
    segments = diarize.diarize_pyannote("a.mp3", "hf_token", voiceprints_out=out)
    assert out == {}
    assert len(segments) == 3


def test_voiceprints_out_empty_when_no_embeddings(monkeypatch):
    pipe = _FakePipeline(_output_with(None))
    _install_pyannote(monkeypatch, new_api=True, pipeline=pipe)
    out = {}
    diarize.diarize_pyannote("a.mp3", "hf_token", voiceprints_out=out)
    assert out == {}


def test_voiceprints_out_empty_when_pyannote_3_cannot_return_embeddings(monkeypatch):
    class NoEmbeddings(_FakePipeline):
        def __call__(self, audio_path):
            return self.result

    _install_pyannote(monkeypatch, new_api=False, pipeline=NoEmbeddings(_Annotation(TURNS)))
    out = {}
    segments = diarize.diarize_pyannote("a.mp3", "hf_token", voiceprints_out=out)
    assert out == {}
    assert len(segments) == 3


def test_voiceprints_out_none_changes_nothing(monkeypatch):
    pipe = _FakePipeline(_Annotation(TURNS))
    _install_pyannote(monkeypatch, new_api=False, pipeline=pipe)
    segments = diarize.diarize_pyannote("a.mp3", "hf_token")
    assert pipe.call_kwargs == {}
    assert len(segments) == 3


def test_diarize_auto_passes_voiceprints_out(monkeypatch):
    seen = {}

    def fake(audio_path, hf_token, voiceprints_out=None):
        seen["out"] = voiceprints_out
        return [diarize.SpeakerSegment(0.0, 1.0, "Speaker 1")]

    monkeypatch.setattr(diarize, "diarize_pyannote", fake)
    out = {}
    diarize.diarize_auto(_segments(1), "a.mp3", "hf_token", voiceprints_out=out)
    assert seen["out"] is out
