"""Read the actual Markdown examples exercised by the documentation tests."""

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]


def extract_fenced_blocks(path, language="python"):
    """Return fences in document order, including VitePress code-group labels."""
    source = (ROOT / path).read_text(encoding="utf-8")
    pattern = rf"^```{re.escape(language)}(?:[ \t]+[^\n]*)?\n(.*?)^```[ \t]*$"
    return re.findall(pattern, source, flags=re.MULTILINE | re.DOTALL)
