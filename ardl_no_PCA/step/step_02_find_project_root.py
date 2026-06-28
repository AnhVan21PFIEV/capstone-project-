from __future__ import annotations

from .common import find_project_root


def run(context: dict) -> dict:
    project_root = find_project_root()
    context["PROJECT_ROOT"] = project_root
    context["SPLITS_DIR"] = project_root / "data/processed/splits"
    context["CORE_DIR"] = project_root / "data/processed/core"
    print("ARDL step 2: project root ->", project_root)
    return context