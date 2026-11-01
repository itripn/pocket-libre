"""Local transcription using Whisper. Everything stays on your machine.

Two implementations of the same model are supported: openai-whisper (runs on
the CPU) and, on Apple Silicon, mlx-whisper (runs on the Apple GPU and is
several times faster). `transcribe_file` picks one; see `resolve_backend`.
"""

import json
import platform
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

from pocket_libre.config import load_config, resolve_whisper_backend

console = Console()

BACKENDS = ("auto", "mlx", "whisper")

# Whisper model names pocket-libre accepts, and the MLX conversion of each.
MLX_REPOS = {
    "tiny.en": "mlx-community/whisper-tiny.en-mlx",
    "base.en": "mlx-community/whisper-base.en-mlx",
    "small.en": "mlx-community/whisper-small.en-mlx",
    "medium.en": "mlx-community/whisper-medium.en-mlx",
    "large": "mlx-community/whisper-large-v3-mlx",
}


def _is_apple_silicon() -> bool:
    return sys.platform == "darwin" and platform.machine() == "arm64"


def _have_mlx() -> bool:
    try:
        import mlx_whisper  # noqa: F401
    except ImportError:
        return False
    return True


def mlx_repo(model_name: str) -> str | None:
    """The MLX build of a Whisper model, or None if there isn't a known one."""
    if "/" in model_name:  # already a HuggingFace repo
        return model_name
    return MLX_REPOS.get(model_name)


def resolve_backend(requested: str | None = None, model_name: str | None = None) -> str:
    """Pick "mlx" or "whisper".

    ``requested`` defaults to POCKET_LIBRE_WHISPER_BACKEND, then
    ``[defaults] whisper_backend``, then "auto". Auto uses MLX when this is an
    Apple Silicon Mac with mlx-whisper installed and the model has an MLX build,
    and openai-whisper otherwise. Asking for MLX outright when it can't run is
    an error rather than a quiet fallback.
    """
    requested = requested or resolve_whisper_backend(load_config())
    if requested not in BACKENDS:
        raise ValueError(
            f"Unknown whisper backend {requested!r}; use one of: {', '.join(BACKENDS)}."
        )
    if requested == "whisper":
        return "whisper"

    can_run = _is_apple_silicon() and _have_mlx()
    has_build = model_name is None or mlx_repo(model_name) is not None
    if requested == "auto":
        return "mlx" if can_run and has_build else "whisper"

    if not _is_apple_silicon():
        raise RuntimeError("The mlx backend needs an Apple Silicon Mac.")
    if not _have_mlx():
        raise RuntimeError('mlx-whisper is not installed. Run: pip install -e ".[mlx]"')
    if not has_build:
        raise ValueError(
            f"No MLX build is known for Whisper model {model_name!r}. "
            f"Use one of: {', '.join(MLX_REPOS)}, or a HuggingFace repo name."
        )
    return "mlx"


def transcribe_file(audio_path, model_name: str, backend: str | None = None) -> dict:
    """Transcribe one audio file. Returns Whisper's result: text, language, segments."""
    backend = resolve_backend(backend, model_name)
    if backend == "mlx":
        import mlx_whisper

        repo = mlx_repo(model_name)
        console.print(f"[dim]Transcribing with mlx-whisper ({repo})...[/dim]")
        result = mlx_whisper.transcribe(str(audio_path), path_or_hf_repo=repo)
    else:
        import whisper

        model = whisper.load_model(model_name)
        result = model.transcribe(str(audio_path), verbose=False)

    return {
        **result,
        "text": result.get("text", ""),
        "language": result.get("language", "unknown"),
        "segments": result.get("segments", []),
    }


def transcribe_audio(
    input_path: str,
    model_name: str = "base.en",
    output_path: str | None = None,
    output_format: str = "text",
):
    """Transcribe an audio file using Whisper locally.

    No network calls. No API keys. The model runs entirely on your machine.
    """
    input_file = Path(input_path)
    if not input_file.exists():
        console.print(f"[red]File not found: {input_path}[/red]")
        return

    console.print(f"[bold]Loading Whisper model: {model_name}[/bold]")
    console.print("[dim]First run will download the model (one-time).[/dim]\n")

    console.print(f"[cyan]Transcribing: {input_path}[/cyan]\n")

    try:
        result = transcribe_file(input_file, model_name)
    except ImportError:
        console.print(
            "[red]Whisper not installed.[/red] "
            "Run: [bold]pip install openai-whisper[/bold]"
        )
        return

    text = result["text"].strip()
    segments = result.get("segments", [])
    language = result.get("language", "unknown")

    if output_format == "text":
        formatted = text
    elif output_format == "json":
        formatted = json.dumps(
            {
                "text": text,
                "language": language,
                "segments": [
                    {
                        "start": seg["start"],
                        "end": seg["end"],
                        "text": seg["text"].strip(),
                    }
                    for seg in segments
                ],
            },
            indent=2,
        )
    elif output_format == "srt":
        lines = []
        for i, seg in enumerate(segments, 1):
            start = format_timestamp_srt(seg["start"])
            end = format_timestamp_srt(seg["end"])
            lines.append(f"{i}")
            lines.append(f"{start} --> {end}")
            lines.append(seg["text"].strip())
            lines.append("")
        formatted = "\n".join(lines)
    else:
        formatted = text

    if output_path:
        Path(output_path).write_text(formatted, encoding="utf-8")
        console.print(f"[bold green]Transcript saved to: {output_path}[/bold green]")
    else:
        console.print(Panel(formatted, title="Transcript", border_style="green"))

    console.print(f"\n[dim]Language: {language} | Segments: {len(segments)}[/dim]")


def format_timestamp_srt(seconds: float) -> str:
    """Format seconds as SRT timestamp (HH:MM:SS,mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
