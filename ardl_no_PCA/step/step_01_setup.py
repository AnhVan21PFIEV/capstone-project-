from __future__ import annotations

import os
import random
import warnings
import numpy as np
import pandas as pd
from pathlib import Path


def run(context: dict) -> dict:
    warnings.filterwarnings("ignore")
    pd.set_option("display.max_columns", 200)
    pd.set_option("display.width", 140)

    seed = 42
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    context.update({"seed": seed})
    
    # ===== TẠO THƯ MỤC logs/figures/ardl/ =====
    figures_dir = Path("E:\\code for vsc\\New folder\\capstone-project-\\logs\\figures\\ardl")
    figures_dir.mkdir(parents=True, exist_ok=True)
    context["figures_dir"] = figures_dir
    print(f" Figures directory: {figures_dir.resolve()}")
    # ===== KẾT THÚC =====

    print("ARDL step 1: setup complete")
    return context