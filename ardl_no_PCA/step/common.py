from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.stats.stattools import jarque_bera


def find_project_root() -> Path:
    """Tìm project root chứa thư mục data/"""
    # Sửa: tìm đúng đường dẫn data/processed/splits/train_scaled.csv
    expected_rel = Path("data/processed/splits/train_scaled.csv")
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
        for p in Path("/content").rglob("train_scaled.csv"):
            if p.parent.name == "splits":
                root = p.parents[3]  # .../data/processed/splits/train_scaled.csv
                if (root / expected_rel).exists():
                    return root

    raise FileNotFoundError(
        "Cannot find data/processed/splits/train_scaled.csv. Check the project folder location."
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
    """Load dữ liệu đã scale từ thư mục data/processed/splits/"""
    # Sửa: đúng đường dẫn data/processed/splits/
    splits_dir = project_root / "data/processed/splits"
    core_dir = project_root / "data/processed/core"

    # Load dữ liệu scaled
    train_scaled = pd.read_csv(splits_dir / "train_scaled.csv", index_col=0, parse_dates=True)
    val_scaled = pd.read_csv(splits_dir / "val_scaled.csv", index_col=0, parse_dates=True)
    test_scaled = pd.read_csv(splits_dir / "test_scaled.csv", index_col=0, parse_dates=True)
    
    # Load VNINDEX target
    vnindex = pd.read_csv(core_dir / "vnindex_target.csv", parse_dates=["Ngày"]).set_index("Ngày")

    # Merge target vào từng tập
    train_df = train_scaled.join(vnindex, how="inner")
    val_df = val_scaled.join(vnindex, how="inner")
    test_df = test_scaled.join(vnindex, how="inner")

    # Lấy danh sách features (tất cả các cột trừ VNINDEX)
    feature_cols = [c for c in train_df.columns if c != "VNINDEX"]

    return {
        "splits_dir": splits_dir,
        "core_dir": core_dir,
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "feature_cols": feature_cols,
        "target_col": "VNINDEX",
    }