from __future__ import annotations

import sys
import shutil
import re
from pathlib import Path
from typing import List, Sequence, Tuple


def _import_nbformat():
    def _normalize_sys_path() -> None:
        for entry in list(sys.path):
            if not entry:
                continue
            match = re.search(r"python(\d+\.\d+)", entry)
            if match and match.group(1) != f"{sys.version_info.major}.{sys.version_info.minor}":
                try:
                    sys.path.remove(entry)
                except ValueError:
                    pass

    def _candidate_site_packages() -> List[Path]:
        candidates: List[Path] = []
        for exe_name in ("jupyter-lab", "jupyter"):
            exe_path = shutil.which(exe_name)
            if not exe_path:
                continue
            resolved = Path(exe_path).resolve()
            for parent in [resolved, *resolved.parents]:
                candidates.extend(
                    path
                    for path in parent.glob("lib/python*/site-packages")
                    if path.exists()
                )
                candidates.extend(
                    path
                    for path in parent.glob("libexec/lib/python*/site-packages")
                    if path.exists()
                )

        return candidates

    _normalize_sys_path()

    try:
        import nbformat  # type: ignore
        from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook
        return nbformat, new_code_cell, new_markdown_cell, new_notebook
    except ModuleNotFoundError:
        for candidate in _candidate_site_packages():
            if str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
        _normalize_sys_path()
        import nbformat  # type: ignore
        from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook
        return nbformat, new_code_cell, new_markdown_cell, new_notebook


nbformat, new_code_cell, new_markdown_cell, new_notebook = _import_nbformat()


ROOT = Path(__file__).resolve().parent
OUTPUT_NOTEBOOK = ROOT / "submission_notebook.ipynb"
SOURCE_FILES: Sequence[Tuple[str, str]] = (
    ("src/pipeline.py", "Shared Pipeline and Base Abstractions"),
    ("src/models_trees.py", "Tree-Based Model Family"),
    ("src/models_sequences.py", "Sequence Model Family"),
    ("src/models_linear.py", "Linear Model Family"),
    ("src/models_spatial.py", "Spatial Model Family"),
    ("src/models_kernels.py", "Kernel Model Family"),
    ("src/models_tabnet.py", "TabNet Model Family"),
    ("src/models_baselines.py", "Baseline Model Family"),
    ("src/blend.py", "Blending and Submission Pipeline"),
)

MAIN_GUARDS = (
    'if __name__ == "__main__":',
    "if __name__ == '__main__':",
)


def strip_main_guard(source: str) -> str:
    """Remove top-level smoke-test blocks guarded by __main__ checks."""
    lines = source.splitlines()
    cleaned: List[str] = []
    skipping = False
    guard_indent = 0

    for line in lines:
        stripped = line.lstrip()
        indent = len(line) - len(stripped)

        if not skipping and stripped in MAIN_GUARDS:
            skipping = True
            guard_indent = indent
            continue

        if skipping:
            if not stripped:
                continue
            if indent > guard_indent:
                continue
            skipping = False

        cleaned.append(line)

    result = "\n".join(cleaned)
    if source.endswith("\n"):
        result += "\n"
    return result


def build_notebook_cells(source_files: Sequence[Tuple[str, str]]) -> List[object]:
    cells: List[object] = []
    for relative_path, section_title in source_files:
        source_path = ROOT / relative_path
        if not source_path.exists():
            raise FileNotFoundError(f"Source file not found: {source_path}")

        raw_source = source_path.read_text(encoding="utf-8")
        cleaned_source = strip_main_guard(raw_source)
        cells.append(new_markdown_cell(f"## {section_title}\n\n`{relative_path}`"))

        cells.append(new_code_cell(cleaned_source))

    cells.append(
        new_code_cell(
            'import sys\n'
            'sys.argv = ["blend.py", "--full"]\n'
            'main()\n'
        )
    )
    return cells


def compile_notebook(output_path: Path = OUTPUT_NOTEBOOK) -> Path:
    notebook = new_notebook(
        cells=build_notebook_cells(SOURCE_FILES),
        metadata={
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "pygments_lexer": "ipython3",
            },
        },
    )

    with output_path.open("w", encoding="utf-8") as handle:
        nbformat.write(notebook, handle)

    return output_path


def main() -> int:
    output_path = compile_notebook()
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
