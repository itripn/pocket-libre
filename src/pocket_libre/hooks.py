"""User-supplied commands run at points in the pipeline.

Today there is one: ``[hooks] on_summary`` runs after a summary file is
written, with the path to that file as its last argument. It exists so
summaries can be handed to other tools without pocket-libre knowing about
them. A hook that fails never fails the sync that triggered it.
"""

from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path

from rich.console import Console

from pocket_libre.config import load_config, resolve_summary_hook

console = Console()

# Hooks may call out over the network (an LLM, an MCP server), so leave room.
DEFAULT_TIMEOUT_SECONDS = 300.0


def run_summary_hook(
    summary_path: Path,
    command: str | None = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> bool:
    """Run the on_summary hook for one summary file. Returns True if it ran cleanly.

    ``command`` defaults to the configured hook. It is split shell-style but
    never passed through a shell, so a path containing spaces (the default
    output directory does) arrives as a single argument.
    """
    if command is None:
        command = resolve_summary_hook(load_config())
    if not command:
        return False

    try:
        argv = shlex.split(command)
    except ValueError as e:
        console.print(f"[yellow]on_summary hook is not a valid command: {e}[/yellow]")
        return False
    if not argv:
        return False
    argv[0] = os.path.expanduser(argv[0])
    argv.append(str(summary_path))

    try:
        result = subprocess.run(argv, timeout=timeout, check=False)
    except FileNotFoundError:
        console.print(f"[yellow]on_summary hook not found: {argv[0]}[/yellow]")
        return False
    except subprocess.TimeoutExpired:
        console.print(f"[yellow]on_summary hook timed out after {timeout:.0f}s[/yellow]")
        return False
    except OSError as e:
        console.print(f"[yellow]on_summary hook failed to start: {e}[/yellow]")
        return False

    if result.returncode != 0:
        console.print(f"[yellow]on_summary hook exited {result.returncode}[/yellow]")
        return False
    return True
