from __future__ import annotations

import os
import random
import warnings

import numpy as np
import pandas as pd


def run(context: dict) -> dict:
    warnings.filterwarnings("ignore")
    pd.set_option("display.max_columns", 200)
    pd.set_option("display.width", 140)

    seed = 42
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    context.update({"seed": seed})
    print("ARDL step 1: setup complete")
    return context
