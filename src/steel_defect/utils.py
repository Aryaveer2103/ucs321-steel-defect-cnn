"""Small shared helpers for JSON, command errors and reproducibility."""
import json
from pathlib import Path


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def cli_guard(function):
    """Keep expected setup errors actionable; retain tracebacks for programming bugs."""
    try:
        function()
    except (ValueError, FileNotFoundError, FileExistsError, ImportError) as error:
        raise SystemExit(f"Error: {error}") from error
