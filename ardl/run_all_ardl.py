from __future__ import annotations

import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from step.step_01_setup import run as step_01_setup
from step.step_02_find_project_root import run as step_02_find_project_root
from step.step_03_load_data import run as step_03_load_data
from step.step_04_validate_pca import run as step_04_validate_pca
from step.step_05_sweep_ardl import run as step_05_sweep_ardl
from step.step_06_select_and_forecast import run as step_06_select_and_forecast
from step.step_07_export_pkl import run as step_07_export_pkl
from step.step_08_summary import run as step_08_summary
from step.step_09_plot import run as step_09_plot


def main() -> dict:
    context = {"SELECTED_PAIR": (2, 0)}

    for step in [
        step_01_setup,
        step_02_find_project_root,
        step_03_load_data,
        step_04_validate_pca,
        step_05_sweep_ardl,
        step_06_select_and_forecast,
        step_07_export_pkl,
        step_08_summary,
        step_09_plot,
    ]:
        context = step(context)

    return context


if __name__ == "__main__":
    main()
