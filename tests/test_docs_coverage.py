"""Require execution coverage when documentation adds or moves code blocks."""

from collections import Counter
import re
import textwrap

import pytest

from docs_examples import ROOT, extract_fenced_blocks


# Python blocks are executed by test_docs_api.py, test_docs_examples.py,
# test_docs_graph_environments.py, and test_docs_drl.py. The graph/environment
# tests also check game-format excerpts;
# test_docs_commands.py handles shell examples. Other languages are inventoried
# explicitly so they cannot be mistaken for executed Python or shell commands.
EXPECTED_FENCES = {
    "docs/contributing.md": {"sh": 2},
    "docs/guide/algorithms.md": {"python": 1},
    "docs/guide/computation-graph.md": {"python": 19, "mermaid": 1},
    "docs/guide/concepts.md": {"mermaid": 1},
    "docs/guide/deep-learning.md": {"python": 1},
    "docs/guide/deep-learning/objectives.md": {"python": 1},
    "docs/guide/deep-learning/policies.md": {"python": 1},
    "docs/guide/deep-learning/environments.md": {"python": 1},
    "docs/guide/environments.md": {"python": 9},
    "docs/guide/environments/file-env.md": {"python": 3, "text": 3},
    "docs/guide/environments/open-spiel.md": {"python": 10, "text": 1},
    "docs/guide/examples.md": {"python": 7, "sh": 6},
    "docs/guide/installation.md": {"sh": 11, "powershell": 1},
    "docs/guide/quick-start.md": {"python": 2, "sh": 1},
    "docs/index.md": {"python": 1},
    "docs/research.md": {"bibtex": 1},
}


def test_all_fenced_examples_have_reviewed_page_and_language_coverage():
    actual = {}
    # Inspect all fence languages independently of the Python/sh extractor.
    # This also notices newly added pages and unlabeled or tilde-style fences.
    for path in sorted((ROOT / "docs").rglob("*.md")):
        languages = Counter()
        opening = None
        for line in path.read_text(encoding="utf-8").splitlines():
            if opening is not None:
                if re.fullmatch(rf" {{0,3}}{re.escape(opening)}[ \t]*", line):
                    opening = None
                continue
            match = re.match(r"^ {0,3}(`{3,}|~{3,})(.*)$", line)
            if match:
                opening, info = match.groups()
                languages[info.strip().split()[0] if info.strip() else ""] += 1
        assert opening is None, f"Unclosed code fence in {path}"
        if languages:
            actual[path.relative_to(ROOT).as_posix()] = dict(languages)
    assert actual == EXPECTED_FENCES, (
        "Assign new or moved examples to an execution harness before updating "
        "the page/language coverage inventory"
    )


@pytest.mark.parametrize("path", [
    path for path, languages in EXPECTED_FENCES.items() if "python" in languages
])
def test_python_fences_are_extracted_and_compile(path):
    blocks = extract_fenced_blocks(path)
    assert len(blocks) == EXPECTED_FENCES[path]["python"]
    for index, source in enumerate(blocks, 1):
        compile(textwrap.dedent(source), f"{path}:python[{index}]", "exec")
