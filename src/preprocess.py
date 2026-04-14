"""
preprocess.py  –  VN-Index Preprocessing Pipeline
====================================================
Chạy: python src/preprocess.py --config config/config.yaml

Output artifacts (mỗi bước lưu riêng):
  data/processed/
        core/cleaned_data.csv         ← ma trận wide (ngày × mã)
        core/vnindex_target.csv       ← chuỗi VN-Index (biến Y)
        core/valid_stocks.csv         ← danh sách mã được giữ lại
        core/removed_stocks.csv       ← danh sách mã bị loại + lý do
        quality/outlier_log.csv       ← log ngoại lai IQR theo mã
        quality/missing_dist.csv      ← phân bố tỷ lệ thiếu
        quality/corr_summary.csv      ← phân tích tương quan
        quality/corr_matrix.csv       ← ma trận tương quan đầy đủ
        splits/train|val|test_scaled.csv
        splits/split_summary.csv      ← thông tin phân chia thời gian
        stationarity/stationarity_results.csv
        stationarity/stationarity_logreturn.csv
  models/
    scaler_params.pkl             ← Z-score scaler (fit on train only)
  logs/figures/
    preprocess_summary.png
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from helpers.utils import (
    scale_by_train_stats,
    split_by_time,
)
from preprocess_steps import (
    analyze_correlation,
    clean_and_separate,
    fill_and_clean,
    filter_by_observation_ratio,
    load_raw_data,
    pivot_to_wide,
    remove_outliers_with_log,
    run_log_return_adf,
    run_stationarity_checks,
    save_preprocess_figure,
)


# ─────────────────────────────────────────────────────────────────
# HELPER: LOAD CONFIG
# ─────────────────────────────────────────────────────────────────
def load_config(config_path: str | Path) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────
def preprocess_pipeline(project_root: Path, config_path: Path) -> None:
    cfg   = load_config(config_path)
    paths = cfg["paths"]
    cols  = cfg["columns"]
    prep  = cfg["preprocess"]

    raw_path      = project_root / paths["raw_file"]
    processed_dir = project_root / paths["processed_dir"]
    models_dir    = project_root / paths["models_dir"]
    figures_dir   = project_root / paths.get("figures_dir", "logs/figures")

    subdirs = paths.get("processed_subdirs", {})
    core_dir = processed_dir / subdirs.get("core", "core")
    quality_dir = processed_dir / subdirs.get("quality", "quality")
    splits_dir = processed_dir / subdirs.get("splits", "splits")
    stationarity_dir = processed_dir / subdirs.get("stationarity", "stationarity")

    for d in [
        processed_dir,
        core_dir,
        quality_dir,
        splits_dir,
        stationarity_dir,
        models_dir,
        figures_dir,
    ]:
        d.mkdir(parents=True, exist_ok=True)

    # ── 1. Load ──────────────────────────────────────────────────
    df = load_raw_data(raw_path)

    # ── 2. Clean & separate VNINDEX ──────────────────────────────
    stocks_df, vnindex_series = clean_and_separate(
        df, cols, remove_weekends=prep.get("remove_weekends", True)
    )

    # Save VNINDEX target artifact
    vnindex_series.to_csv(core_dir / "vnindex_target.csv", header=True)
    print(f"[SAVED] vnindex_target.csv")

    # ── 3. IQR outlier removal ───────────────────────────────────
    stocks_clean = remove_outliers_with_log(
        stocks_df,
        symbol_col=cols["symbol"],
        close_col=cols["close"],
        k=float(prep.get("outlier_k", 1.5)),
        output_path=quality_dir / "outlier_log.csv",
    )

    # ── 4. Pivot to wide format ───────────────────────────────────
    df_pivot = pivot_to_wide(
        stocks_clean,
        date_col=cols["date"],
        symbol_col=cols["symbol"],
        close_col=cols["close"],
    )

    # ── 5. Filter ≥ 80% observations ─────────────────────────────
    threshold = float(prep.get("missing_threshold", 0.2))
    df_pivot, missing_dist = filter_by_observation_ratio(
        df_pivot,
        threshold=threshold,
        core_dir=core_dir,
        quality_dir=quality_dir,
    )

    # ── 6. Fill missing + clean ───────────────────────────────────
    df_pivot = fill_and_clean(df_pivot)

    # ── 7. Correlation analysis ───────────────────────────────────
    corr_summary = analyze_correlation(df_pivot, quality_dir)

    # ── 8. Save cleaned wide data ─────────────────────────────────
    cleaned_path = core_dir / "cleaned_data.csv"
    df_pivot.to_csv(cleaned_path)
    print(f"[SAVED] cleaned_data.csv  shape={df_pivot.shape}")

    # ── 9. Time-based split ───────────────────────────────────────
    splits = split_by_time(
        df_pivot,
        train_ratio=float(prep.get("train_ratio", 0.7)),
        val_ratio=float(prep.get("val_ratio", 0.15)),
    )
    n_train = len(splits.train)
    n_val   = len(splits.val)
    n_test  = len(splits.test)

    split_summary = pd.DataFrame({
        "Split": ["Train", "Validation", "Test"],
        "From":  [splits.train.index.min(), splits.val.index.min(), splits.test.index.min()],
        "To":    [splits.train.index.max(), splits.val.index.max(), splits.test.index.max()],
        "Rows":  [n_train, n_val, n_test],
        "Ratio": ["70%", "15%", "15%"],
    })
    split_summary.to_csv(splits_dir / "split_summary.csv", index=False)
    print("[SPLIT]")
    print(split_summary.to_string(index=False))

    # ── 10. Z-score scaling (fit on train only) ───────────────────
    train_scaled, val_scaled, test_scaled, train_mean, train_std = scale_by_train_stats(
        splits.train, splits.val, splits.test
    )
    train_scaled.to_csv(splits_dir / "train_scaled.csv")
    val_scaled.to_csv(splits_dir / "val_scaled.csv")
    test_scaled.to_csv(splits_dir / "test_scaled.csv")

    scaler_params = {
        "mean": train_mean,
        "std": train_std,
        "feature_names": df_pivot.columns.tolist(),
        "fit_date_range": (splits.train.index.min(), splits.train.index.max()),
    }
    scaler_path = models_dir / "scaler_params.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler_params, f)

    print(f"[VERIFY] Train scaled mean≈0: {train_scaled.mean().mean():.6f}")
    print(f"[VERIFY] Train scaled std≈1 : {train_scaled.std().mean():.6f}")
    print(f"[SAVED] scaler_params.pkl")

    # ── 11. Stationarity tests ────────────────────────────────────
    n_stat   = int(prep.get("stationarity_sample_size", 30))
    n_diff   = int(prep.get("stationarity_after_diff_sample_size", 10))
    sample_s = list(df_pivot.columns[:n_stat])

    stat_df = run_stationarity_checks(
        splits.train,
        sample_s,
        stationarity_dir / "stationarity_results.csv",
    )
    run_log_return_adf(
        splits.train, stat_df, sample_s[:n_diff],
        stationarity_dir / "stationarity_logreturn.csv",
    )

    # ── 12. Figures ───────────────────────────────────────────────
    save_preprocess_figure(
        figures_dir=figures_dir,
        vnindex_series=vnindex_series,
        df_pivot=df_pivot,
        train_data=splits.train,
        train_scaled=train_scaled,
        stationarity_df=stat_df,
        corr_summary=corr_summary,
        n_train=n_train,
        n_val=n_val,
    )

    # ── 13. Summary ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("ARTIFACTS SAVED  (output của bước này = input bước sau)")
    print("=" * 60)
    artifacts = [
        ("core/vnindex_target.csv",          f"VNINDEX target {len(vnindex_series)} ngày"),
        ("core/valid_stocks.csv",            f"{df_pivot.shape[1]} mã được giữ lại"),
        ("core/removed_stocks.csv",          "Mã bị loại + lý do"),
        ("core/cleaned_data.csv",            f"Wide matrix {df_pivot.shape}"),
        ("quality/outlier_log.csv",          "Log ngoại lai IQR theo mã"),
        ("quality/missing_dist.csv",         "Phân bố missing ratio"),
        ("quality/corr_summary.csv",         "Tóm tắt phân tích tương quan"),
        ("quality/corr_matrix.csv",          "Ma trận tương quan đầy đủ"),
        ("splits/split_summary.csv",         "Train/Val/Test timeline"),
        ("splits/train|val|test_scaled.csv", "Dữ liệu đã Z-score (fit on train)"),
        ("stationarity/stationarity_results.csv",   "ADF + KPSS trên 30 mã"),
        ("stationarity/stationarity_logreturn.csv", "ADF sau log-return"),
        ("../models/scaler_params.pkl",      "Scaler object cho inverse-transform"),
        ("../logs/figures/preprocess_summary.png", "Biểu đồ tổng hợp 6 panel"),
    ]
    for name, desc in artifacts:
        print(f"  ✅ {name:<40} {desc}")
    print(f"\n➡️  Bước tiếp theo: python src/pca_model.py --config config/config.yaml")


# ─────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="VN-Index preprocessing pipeline")
    p.add_argument("--config", default="config/config.yaml")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    preprocess_pipeline(project_root=root, config_path=root / args.config)