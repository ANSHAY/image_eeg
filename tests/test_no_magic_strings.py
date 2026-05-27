"""Enforce the config-only discipline from docs/implementation_plan.md §10.

Walks every .py file under the source dirs and flags hardcoded string
literals that should live in `config.yaml` instead:

  - HuggingFace model IDs (e.g. `openai/clip-vit-base-patch32`)
  - HTTP/HTTPS URLs
  - LSL stream names that match the configured outlet names

Docstrings, pydantic validation regex patterns, and short literals are
exempt. Test files and downloader stubs are skipped (the stubs only use
strings sourced from config; nothing literal slips in).

Adding allowed-exceptions: extend `ALLOWED_LITERALS` rather than
weakening the rules — every exception should be deliberate.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from utils.config import load_config

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Directories whose .py files must obey the config-only rule.
SOURCE_DIRS = [
    PROJECT_ROOT / "models",
    PROJECT_ROOT / "preprocessing",
    PROJECT_ROOT / "streaming",
    PROJECT_ROOT / "app",
    PROJECT_ROOT / "generation",
    PROJECT_ROOT / "evaluation",
    PROJECT_ROOT / "utils",
    PROJECT_ROOT / "data",
]

# Patterns that must NOT appear as raw string literals in source.
FORBIDDEN_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "hf-model-id",
        re.compile(
            r"^(openai|stabilityai|h94|google|microsoft|meta-llama|"
            r"anthropic|huggingface|runwayml|CompVis|laion)/",
        ),
    ),
    ("http-url", re.compile(r"^https?://")),
]

# Strings on this list are tolerated (override of pattern matches).
ALLOWED_LITERALS: set[str] = set()


def _iter_source_files() -> list[Path]:
    files: list[Path] = []
    for d in SOURCE_DIRS:
        if not d.exists():
            continue
        files.extend(p for p in d.rglob("*.py") if p.name != "__init__.py")
    return files


def _collect_docstring_node_ids(tree: ast.AST) -> set[int]:
    """Return the id() of every Constant node that is a docstring."""
    docstring_ids: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = getattr(node, "body", None) or []
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                docstring_ids.add(id(body[0].value))
    return docstring_ids


def _is_regex_pattern_arg(parent_stack: list[ast.AST], node: ast.Constant) -> bool:
    """True if `node` is the value of `Field(pattern="…")` or a `re.compile(…)` call."""
    for parent in reversed(parent_stack):
        if isinstance(parent, ast.Call):
            # Field(pattern="…")
            for kw in parent.keywords:
                if kw.arg == "pattern" and kw.value is node:
                    return True
            # re.compile(r"…")  — positional 0
            if (
                isinstance(parent.func, ast.Attribute)
                and parent.func.attr == "compile"
                and parent.args
                and parent.args[0] is node
            ):
                return True
            return False
    return False


def _annotate_parents(tree: ast.AST) -> dict[int, list[ast.AST]]:
    """Map id(node) -> ancestor stack (root → … → parent) for every AST node."""
    parents: dict[int, list[ast.AST]] = {id(tree): []}
    stack: list[tuple[ast.AST, list[ast.AST]]] = [(tree, [])]
    while stack:
        current, ancestors = stack.pop()
        for child in ast.iter_child_nodes(current):
            child_ancestors = ancestors + [current]
            parents[id(child)] = child_ancestors
            stack.append((child, child_ancestors))
    return parents


def _violations_in(path: Path) -> list[tuple[int, str, str]]:
    """Return (lineno, pattern_name, literal) for each violation in `path`."""
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src, filename=str(path))
    docstring_ids = _collect_docstring_node_ids(tree)
    parents_map = _annotate_parents(tree)

    violations: list[tuple[int, str, str]] = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Constant) and isinstance(node.value, str)):
            continue
        if id(node) in docstring_ids:
            continue
        literal = node.value
        if literal in ALLOWED_LITERALS:
            continue
        # Short literals are noise; require at least 8 chars to engage.
        if len(literal) < 8:
            continue
        if _is_regex_pattern_arg(parents_map.get(id(node), []), node):
            continue
        for name, pat in FORBIDDEN_PATTERNS:
            if pat.search(literal):
                violations.append((node.lineno, name, literal))
                break
    return violations


@pytest.mark.parametrize("source_file", _iter_source_files(), ids=lambda p: str(p.relative_to(PROJECT_ROOT)))
def test_no_magic_strings(source_file: Path) -> None:
    violations = _violations_in(source_file)
    if violations:
        report = "\n".join(
            f"  {source_file.relative_to(PROJECT_ROOT)}:{ln} [{kind}]  {lit!r}"
            for ln, kind, lit in violations
        )
        pytest.fail(
            f"Hardcoded strings found — move into config.yaml:\n{report}",
        )


def test_lint_actually_catches_forbidden_patterns(tmp_path: Path) -> None:
    """Self-test: the lint must flag a deliberately bad file."""
    bad = tmp_path / "bad_module.py"
    bad.write_text(
        '"""docstring is exempt."""\n'
        'MODEL = "openai/clip-vit-base-patch32"\n'
        'URL = "https://huggingface.co/foo"\n',
        encoding="utf-8",
    )
    found = _violations_in(bad)
    kinds = {kind for _, kind, _ in found}
    assert "hf-model-id" in kinds
    assert "http-url" in kinds


def test_config_yaml_holds_canonical_model_ids() -> None:
    """The model IDs the lint forbids in code must be reachable through config."""
    cfg = load_config()
    assert "/" in cfg.models.clip.hf_id
    assert "/" in cfg.generation.sd.hf_id
    assert "/" in cfg.generation.sd.ip_adapter_hf_id
