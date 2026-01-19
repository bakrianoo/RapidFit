"""Prompt template utilities."""

from pathlib import Path

_PROMPTS_DIR = Path(__file__).parent


def load_prompt(name: str) -> str:
    """
    Load a prompt template by name.

    Args:
        name: Prompt file name without extension.

    Returns:
        Prompt template content.
    """
    return (_PROMPTS_DIR / f"{name}.txt").read_text()
