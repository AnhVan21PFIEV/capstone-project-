from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.stats.stattools import jarque_bera


def find_project_root() -> Path:
    expected_rel = Path("data/processed/pca/train_pca.csv")
    candidates = [
        Path.cwd(),
        Path.cwd().parent,
        Path.cwd().parent.parent,
        Path(__file__).resolve().parents[2],
        Path("/content/capstone-project-"),
        Path("/content/drive/MyDrive/capstone-project-"),
    ]

    for root in candidates:
        if (root / expected_rel).exists():
            return root

    if Path("/content").exists():
        for p in Path("/content").rglob("train_pca.csv"):
            if p.parent.name == "pca":
                root = p.parents[3]
                if (root / expected_rel).exists():
                    return root

    raise FileNotFoundError(
        "Cannot find data/processed/pca/train_pca.csv. Check the project folder location."
    )


def paired_valid(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    n = min(len(y_true), len(y_pred))
    y_true = y_true[:n]
    y_pred = y_pred[:n]
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    return y_true[mask], y_pred[mask]


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mape(y_true, y_pred) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    eps = 1e-8
    return float(np.mean(np.abs((y_true - y_pred) / (y_true + eps))) * 100.0)


def diagnostics_from_residuals(resid) -> Dict[str, float]:
    lb_df = acorr_ljungbox(resid, lags=[1, 10], return_df=True)
    jb_stat, jb_pvalue, skew, kurt = jarque_bera(resid)
    arch_stat, arch_pvalue, _, _ = het_arch(resid, nlags=5)

    return {
        "LjungBox_Q_L1": float(lb_df.loc[1, "lb_stat"]),
        "LjungBox_p_L1": float(lb_df.loc[1, "lb_pvalue"]),
        "LjungBox_Q_L10": float(lb_df.loc[10, "lb_stat"]),
        "LjungBox_p_L10": float(lb_df.loc[10, "lb_pvalue"]),
        "JarqueBera": float(jb_stat),
        "JB_pvalue": float(jb_pvalue),
        "Skew": float(skew),
        "Kurtosis": float(kurt),
        "ARCH_stat": float(arch_stat),
        "ARCH_pvalue": float(arch_pvalue),
    }


def load_inputs(project_root: Path):
    pca_dir = project_root / "data/processed/pca"
    core_dir = project_root / "data/processed/core"

    train_pca = pd.read_csv(pca_dir / "train_pca.csv", parse_dates=["Ngày"]).set_index("Ngày")
    val_pca = pd.read_csv(pca_dir / "val_pca.csv", parse_dates=["Ngày"]).set_index("Ngày")
    test_pca = pd.read_csv(pca_dir / "test_pca.csv", parse_dates=["Ngày"]).set_index("Ngày")
    vnindex = pd.read_csv(core_dir / "vnindex_target.csv", parse_dates=["Ngày"]).set_index("Ngày")

    train_df = train_pca.join(vnindex, how="inner")
    val_df = val_pca.join(vnindex, how="inner")
    test_df = test_pca.join(vnindex, how="inner")
    pc_cols = [c for c in train_df.columns if c.startswith("PC")]

    return {
        "pca_dir": pca_dir,
        "core_dir": core_dir,
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "pc_cols": pc_cols,
        "target_col": "VNINDEX",
    }
