#!/usr/bin/env python
"""Import hygiene check for the xichen package.

Walks every ``xichen/**/*.py`` file, collects top-level import module names via
AST, and fails if any import is neither (a) stdlib, (b) the ``xichen`` package
itself, nor (c) a declared third-party dependency. Catches leftovers from the
research mono-repo (``src.*``, ``inference.*``, ``plots.*``) and undeclared
deps before they reach users.

Usage: python scripts/check_imports.py
"""
import ast
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PKG = REPO_ROOT / "xichen"

# Declared third-party dependencies (import name, not PyPI name).
DECLARED = {
    "torch",
    "numpy",
    "timm",
    "einops",
    "scipy",
    "xarray",
    "netCDF4",
    "h5netcdf",
    "pandas",
    "matplotlib",
    "click",
    "dateutil",
}

STDLIB = set(getattr(sys, "stdlib_module_names", ())) | set(sys.builtin_module_names)


def iter_imports(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield node.lineno, alias.name.split(".")[0]
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:  # absolute import only
                yield node.lineno, node.module.split(".")[0]


def main() -> int:
    bad = []
    for path in sorted(PKG.rglob("*.py")):
        for lineno, top in iter_imports(path):
            if top in STDLIB or top == "xichen" or top in DECLARED:
                continue
            bad.append(f"{path.relative_to(REPO_ROOT)}:{lineno}: imports {top!r}")
    if bad:
        print("Undeclared / leftover imports found:")
        print("\n".join(f"  {b}" for b in bad))
        return 1
    print(f"OK: all imports in {PKG.relative_to(REPO_ROOT)}/ are stdlib, xichen, or declared deps")
    return 0


if __name__ == "__main__":
    sys.exit(main())
