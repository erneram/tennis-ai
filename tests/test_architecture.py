"""The core must stay stroke-agnostic: nothing under core/ may import strokes."""

import ast
from pathlib import Path

CORE = Path(__file__).resolve().parents[1] / "core"


def imported_modules(path: Path) -> list[str]:
    modules = []
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            modules += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            modules.append(node.module or "")
    return modules


def test_core_does_not_import_strokes():
    offenders = [
        str(path.relative_to(CORE.parent))
        for path in CORE.rglob("*.py")
        if any(m.split(".")[0] == "strokes" for m in imported_modules(path))
    ]
    assert not offenders, f"core/ must not import strokes: {offenders}"
