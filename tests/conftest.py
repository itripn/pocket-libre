"""Shared test setup."""

import pytest

from pocket_libre import voices


@pytest.fixture(autouse=True)
def default_to_openai_whisper(monkeypatch):
    """Tests that fake `whisper` must not silently get a real mlx-whisper instead.

    "auto" picks MLX whenever it is installed on an Apple Silicon Mac, so
    without this the result depends on the machine running the tests. Tests of
    backend choice clear this themselves.
    """
    monkeypatch.setenv("POCKET_LIBRE_WHISPER_BACKEND", "whisper")


@pytest.fixture(autouse=True)
def isolated_voice_library(tmp_path_factory, monkeypatch):
    """No test may read or write the real ~/.pocket-libre/voices.json.

    Tests that care about the library point library_path at their own file.
    """
    path = tmp_path_factory.mktemp("voice-home") / "voices.json"
    monkeypatch.setattr(voices, "library_path", lambda: path)
