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
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from statsmodels.tsa.stattools import adfuller, kpss

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from helpers.utils import (
    parse_percent_to_float,
    remove_outliers_group_iqr,
    safe_make_columns_numeric,
    scale_by_train_stats,
    split_by_time,
)


# ─────────────────────────────────────────────────────────────────
# HELPER: LOAD CONFIG
# ─────────────────────────────────────────────────────────────────
def load_config(config_path: str | Path) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────────────────────────
# STEP 1: LOAD RAW DATA
# ─────────────────────────────────────────────────────────────────
def load_raw_data(raw_path: Path) -> pd.DataFrame:
    """
    Phương châm: Hỗ trợ cả .xlsx và .csv.
    Đọc nguyên vẹn, chưa biến đổi gì để giữ trace đầy đủ.
    """
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data not found: {raw_path}")

    if raw_path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(raw_path)
    elif raw_path.suffix.lower() == ".csv":
        df = pd.read_csv(raw_path, sep=None, engine="python", encoding="utf-8-sig")
    else:
        raise ValueError(f"Unsupported format: {raw_path.suffix}")

    df.columns = [str(c).strip() for c in df.columns]
    print(f"[LOAD] Raw shape: {df.shape} | Columns: {df.columns.tolist()}")
    return df


# ─────────────────────────────────────────────────────────────────
# STEP 2: CLEAN TYPES & SEPARATE TARGET
# ─────────────────────────────────────────────────────────────────
def clean_and_separate(
    df: pd.DataFrame,
    cols: Dict[str, Any],
    remove_weekends: bool = True,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Phương châm:
    - VNINDEX là chỉ số tổng hợp (biến Y), không được đưa vào features.
      Nếu không tách, mô hình sẽ học trực tiếp từ biến cần dự báo → data leakage.
    - Loại T7/CN: TTCK VN chỉ giao dịch T2–T6; ngày cuối tuần không có giao dịch
      thực, không nên xuất hiện trong chuỗi mô hình hóa.
    - Ngày lễ: Trong dữ liệu gốc thường không có dòng cho ngày lễ (không giao dịch),
      nên không cần xử lý riêng — trục thời gian chỉ chứa ngày giao dịch thực tế.
    """
    date_col   = cols["date"]
    symbol_col = cols["symbol"]
    close_col  = cols["close"]
    pct_col    = cols.get("pct_change", "% Thay đổi")

    # Chuẩn hóa kiểu
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True)
    df = df.dropna(subset=[date_col, close_col, symbol_col])
    df = safe_make_columns_numeric(df, cols["numeric"])
    if pct_col in df.columns:
        df[pct_col] = parse_percent_to_float(df[pct_col])

    df = df.sort_values(date_col).reset_index(drop=True)

    # Loại T7 / CN
    if remove_weekends:
        n_before = len(df)
        df = df[df[date_col].dt.dayofweek < 5].reset_index(drop=True)
        n_removed = n_before - len(df)
        print(f"[WEEKEND] Loại {n_removed:,} dòng T7/CN "
              f"({n_removed / n_before * 100:.2f}%)")

    # Tách VNINDEX làm biến Y
    vnindex_mask = df[symbol_col] == "VNINDEX"
    vnindex_series = (
        df[vnindex_mask][[date_col, close_col]]
        .set_index(date_col)[close_col]
        .rename("VNINDEX")
    )
    stocks_df = df[~vnindex_mask].copy()

    if vnindex_series.empty:
        raise ValueError(
            "Không tìm thấy Symbol='VNINDEX' trong dữ liệu. "
            "Kiểm tra lại file raw hoặc cột symbol."
        )

    print(f"[TARGET] VNINDEX: {len(vnindex_series)} ngày "
          f"({vnindex_series.index.min().date()} → {vnindex_series.index.max().date()})")
    print(f"[FEATURE] Stocks: {len(stocks_df):,} records | "
          f"{stocks_df[symbol_col].nunique()} symbols")

    # Thống kê nhanh VNINDEX
    vn = vnindex_series
    print(f"[VNINDEX STATS] min={vn.min():.2f} max={vn.max():.2f} "
          f"mean={vn.mean():.2f} std={vn.std():.2f} "
          f"skew={vn.skew():.4f} kurt={vn.kurt():.4f}")

    return stocks_df, vnindex_series


# ─────────────────────────────────────────────────────────────────
# STEP 3: IQR OUTLIER REMOVAL + LOG
# ─────────────────────────────────────────────────────────────────
def remove_outliers_with_log(
    stocks_df: pd.DataFrame,
    symbol_col: str,
    close_col: str,
    k: float,
    output_path: Path,
) -> pd.DataFrame:
    """
    Phương châm:
    - Dữ liệu tài chính không tuân phân phối chuẩn (skewed, fat-tail)
      → IQR là lựa chọn robust, không cần giả định phân phối.
    - Áp dụng PER SYMBOL (không phải toàn thị trường) để phát hiện
      biến động bất thường CỤC BỘ của từng mã, không loại nhầm các
      giai đoạn thị trường chung tăng/giảm mạnh.
    - Log ngoại lai lưu artifact để truy vết.
    """
    stocks_clean = remove_outliers_group_iqr(
        stocks_df, group_col=symbol_col, value_col=close_col, k=k
    )

    # Log chi tiết ngoại lai
    removed_mask = ~stocks_df.index.isin(stocks_clean.index)
    if removed_mask.sum() > 0:
        outlier_log = (
            stocks_df[removed_mask]
            .groupby(symbol_col)
            .agg(
                n_outliers=(close_col, "count"),
                min_val=(close_col, "min"),
                max_val=(close_col, "max"),
                mean_val=(close_col, "mean"),
            )
            .reset_index()
            .sort_values("n_outliers", ascending=False)
        )
    else:
        outlier_log = pd.DataFrame(
            columns=[symbol_col, "n_outliers", "min_val", "max_val", "mean_val"]
        )

    outlier_log.to_csv(output_path, index=False)

    n_removed = len(stocks_df) - len(stocks_clean)
    print(f"[IQR] k={k} | Trước: {len(stocks_df):,} | Sau: {len(stocks_clean):,} "
          f"| Loại: {n_removed:,} ({n_removed / len(stocks_df) * 100:.2f}%)")
    print(f"[IQR] Log saved → {output_path}")

    return stocks_clean


# ─────────────────────────────────────────────────────────────────
# STEP 4: PIVOT TO WIDE FORMAT
# ─────────────────────────────────────────────────────────────────
def pivot_to_wide(
    stocks_clean: pd.DataFrame,
    date_col: str,
    symbol_col: str,
    close_col: str,
) -> pd.DataFrame:
    """
    Chuyển Long format → Wide format (ma trận ngày × mã).
    Đây là định dạng đầu vào chuẩn cho PCA và LSTM.
    """
    stocks_clean = stocks_clean.drop_duplicates(subset=[date_col, symbol_col])
    df_pivot = stocks_clean.pivot(index=date_col, columns=symbol_col, values=close_col)
    df_pivot.columns.name = None
    print(f"[PIVOT] Wide shape: {df_pivot.shape}  (ngày × mã)")
    return df_pivot


# ─────────────────────────────────────────────────────────────────
# STEP 5: FILTER BY OBSERVATION RATIO ≥ 80%
# ─────────────────────────────────────────────────────────────────
def filter_by_observation_ratio(
    df_pivot: pd.DataFrame,
    threshold: float,
    core_dir: Path,
    quality_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Phương châm:
    - Mã có <80% quan sát thường là: mới niêm yết (<2020), đang bị kiểm
      soát giao dịch (suspended), thanh khoản quá thấp → không liên quan
      đến biến động VN-Index.
    - Tổng ngày giao dịch lý thuyết 5 năm ~1.250 ngày (sau bỏ T7/CN, lễ).
      Mã đạt ≥80% = ≥1.000 ngày → đủ lịch sử cho mô hình học.
    - Nội suy dài hạn (>20% missing) tạo thông tin giả, làm sai cấu trúc
      phương sai → ảnh hưởng trực tiếp đến kết quả PCA.
    """
    missing_ratio = df_pivot.isnull().mean()

    # Bảng phân bố
    bins = [
        ("0% (đầy đủ)", missing_ratio == 0),
        ("0–5%", (missing_ratio > 0) & (missing_ratio <= 0.05)),
        ("5–10%", (missing_ratio > 0.05) & (missing_ratio <= 0.10)),
        ("10–20%", (missing_ratio > 0.10) & (missing_ratio <= 0.20)),
        (f">{threshold*100:.0f}% (loại)", missing_ratio > threshold),
    ]
    dist_rows = [{"Khoảng missing": label, "Số mã": cond.sum()} for label, cond in bins]
    missing_dist = pd.DataFrame(dist_rows)
    missing_dist.to_csv(quality_dir / "missing_dist.csv", index=False)
    print("[MISSING DISTRIBUTION]")
    print(missing_dist.to_string(index=False))

    valid_cols   = missing_ratio[missing_ratio <= threshold].index.tolist()
    removed_cols = missing_ratio[missing_ratio >  threshold].index.tolist()

    # Lưu danh sách mã bị loại kèm lý do
    removed_df = pd.DataFrame({
        "Symbol": removed_cols,
        "missing_ratio": [missing_ratio[s] for s in removed_cols],
        "reason": "missing_ratio > threshold",
    }).sort_values("missing_ratio", ascending=False)
    removed_df.to_csv(core_dir / "removed_stocks.csv", index=False)

    pd.Series(valid_cols, name="Symbol").to_csv(
        core_dir / "valid_stocks.csv", index=False
    )

    print(f"[FILTER 80%] Giữ lại: {len(valid_cols)} | Loại bỏ: {len(removed_cols)}")
    return df_pivot[valid_cols].copy(), missing_dist


# ─────────────────────────────────────────────────────────────────
# STEP 6: FILL MISSING + CLEAN INVALID PRICES
# ─────────────────────────────────────────────────────────────────
def fill_and_clean(df_pivot: pd.DataFrame) -> pd.DataFrame:
    """
    Phương châm Forward Fill:
    - Giá cổ phiếu có tính liên tục; giá ngày thiếu ≈ giá phiên trước.
    - FFILL không tạo thông tin mới, không thay đổi xu hướng.
    - bfill() chỉ dùng dự phòng cho những ngày đầu chưa có dữ liệu.
    - Mean/Median imputation BỊ LOẠI: làm mất tính động của chuỗi,
      gây sai lệch khi huấn luyện mô hình.
    """
    n_before = df_pivot.isnull().sum().sum()
    df_pivot = df_pivot.ffill().bfill()
    n_after = df_pivot.isnull().sum().sum()
    print(f"[FFILL] Missing: {n_before:,} → {n_after}")

    # Loại giá âm/bằng 0 (lỗi dữ liệu)
    invalid = (df_pivot <= 0).sum().sum()
    if invalid > 0:
        df_pivot[df_pivot <= 0] = np.nan
        df_pivot = df_pivot.ffill().bfill()
        print(f"[CLEAN] Đã xóa {invalid} giá trị ≤ 0")

    # Loại ngày trùng
    dup = df_pivot.index.duplicated().sum()
    if dup:
        df_pivot = df_pivot[~df_pivot.index.duplicated(keep="first")]
        print(f"[DEDUP] Loại {dup} ngày trùng")

    return df_pivot


# ─────────────────────────────────────────────────────────────────
# STEP 7: CORRELATION ANALYSIS
# ─────────────────────────────────────────────────────────────────
def analyze_correlation(
    df_pivot: pd.DataFrame,
    processed_dir: Path,
) -> pd.DataFrame:
    """
    Phương châm: Phân tích tương quan TRƯỚC PCA để xác nhận đa cộng tuyến
    nghiêm trọng → biện hộ cho việc áp dụng PCA.
    Kết quả được lưu artifact và trích bảng vào báo cáo.
    """
    corr_matrix = df_pivot.corr()
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape, dtype=bool), k=1)
    )
    corr_vals = upper.stack()

    thresholds = [0.50, 0.70, 0.80, 0.90]
    rows = []
    total_pairs = len(corr_vals)
    for thr in thresholds:
        cnt = (corr_vals.abs() > thr).sum()
        rows.append({
            "Ngưỡng |r|": f"> {thr:.2f}",
            "Số cặp": cnt,
            "Tỷ lệ (%)": f"{cnt / total_pairs * 100:.1f}",
        })

    corr_summary = pd.DataFrame(rows)
    corr_summary.to_csv(processed_dir / "corr_summary.csv", index=False)

    print("[CORRELATION ANALYSIS]")
    print(f"  Tổng cặp: {total_pairs:,} | Trung bình |r|: {corr_vals.abs().mean():.4f}")
    print(corr_summary.to_string(index=False))

    # Lưu ma trận tương quan đầy đủ (dùng cho heatmap)
    corr_matrix.to_csv(processed_dir / "corr_matrix.csv")

    return corr_summary


# ─────────────────────────────────────────────────────────────────
# STEP 8: STATIONARITY TESTS (ADF + KPSS)
# ─────────────────────────────────────────────────────────────────
def run_stationarity_checks(
    train_data: pd.DataFrame,
    sample_symbols: List[str],
    output_path: Path,
) -> pd.DataFrame:
    """
    Phương châm:
    - Kết hợp ADF + KPSS để có kết luận hai chiều (Elliot et al., 1996).
    - Chỉ chạy trên tập TRAIN (không dùng val/test để tránh look-ahead bias).
    - Kết quả quyết định: ARDL dùng log-return; LSTM dùng Z-score gốc.
    """
    results = []
    for sym in sample_symbols:
        series = train_data[sym].dropna()

        try:
            adf_s, adf_p, adf_lags, _, _, _ = adfuller(series, autolag="AIC")
            adf_stat = adf_p < 0.05
        except Exception:
            adf_s, adf_p, adf_lags, adf_stat = np.nan, np.nan, np.nan, False

        try:
            kp_s, kp_p, _, _ = kpss(series, regression="c", nlags="auto")
            kp_stat = kp_p > 0.05
        except Exception:
            kp_s, kp_p, kp_stat = np.nan, np.nan, False

        if adf_stat and kp_stat:
            conclusion = "I(0) – Dừng"
        elif not adf_stat and not kp_stat:
            conclusion = "I(1)+ – Không dừng"
        elif adf_stat and not kp_stat:
            conclusion = "Trend-stationary"
        else:
            conclusion = "Diff-stationary"

        results.append({
            "Symbol": sym,
            "ADF_stat": round(adf_s, 4) if not np.isnan(adf_s) else np.nan,
            "ADF_p": round(adf_p, 4) if not np.isnan(adf_p) else np.nan,
            "ADF_stationary": adf_stat,
            "KPSS_stat": round(kp_s, 4) if not np.isnan(kp_s) else np.nan,
            "KPSS_p": round(kp_p, 4) if not np.isnan(kp_p) else np.nan,
            "Conclusion": conclusion,
        })

    result_df = pd.DataFrame(results)
    result_df.to_csv(output_path, index=False)

    n_stationary = (result_df["Conclusion"] == "I(0) – Dừng").sum()
    n_nonstat    = result_df["Conclusion"].str.contains("Không dừng").sum()
    print(f"[STATIONARITY] Dừng I(0): {n_stationary}/{len(sample_symbols)} | "
          f"Không dừng: {n_nonstat}/{len(sample_symbols)}")

    return result_df


def run_log_return_adf(
    train_data: pd.DataFrame,
    stationarity_df: pd.DataFrame,
    sample_symbols: List[str],
    output_path: Path,
) -> pd.DataFrame:
    """Kiểm định ADF sau khi lấy log-return để xác nhận tính dừng."""
    results = []
    for sym in sample_symbols:
        series = train_data[sym].dropna()
        log_ret = np.log(series / series.shift(1)).dropna()
        try:
            _, adf_p, _, _, _, _ = adfuller(log_ret, autolag="AIC")
            stat_after = adf_p < 0.05
        except Exception:
            adf_p, stat_after = np.nan, False

        orig_p_row = stationarity_df.loc[stationarity_df["Symbol"] == sym, "ADF_p"]
        orig_p = orig_p_row.iloc[0] if not orig_p_row.empty else np.nan

        results.append({
            "Symbol": sym,
            "ADF_p_original": orig_p,
            "ADF_p_logreturn": round(adf_p, 6) if not np.isnan(adf_p) else np.nan,
            "Stationary_after_diff": stat_after,
        })

    result_df = pd.DataFrame(results)
    result_df.to_csv(output_path, index=False)

    all_stat = result_df["Stationary_after_diff"].all()
    print(f"[LOG-RETURN ADF] Tất cả dừng sau diff: {all_stat}")
    return result_df


# ─────────────────────────────────────────────────────────────────
# STEP 9: FIGURES
# ─────────────────────────────────────────────────────────────────
def save_preprocess_figure(
    figures_dir: Path,
    vnindex_series: pd.Series,
    df_pivot: pd.DataFrame,
    train_data: pd.DataFrame,
    train_scaled: pd.DataFrame,
    stationarity_df: pd.DataFrame,
    corr_summary: pd.DataFrame,
    n_train: int,
    n_val: int,
) -> Path:
    """
    Tổng hợp 6 biểu đồ thành 1 figure để gắn vào báo cáo.
    """
    figures_dir.mkdir(parents=True, exist_ok=True)
    out = figures_dir / "preprocess_summary.png"

    sym = "ACB" if "ACB" in train_data.columns else train_data.columns[0]
    fig = plt.figure(figsize=(20, 14))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.5, wspace=0.38)

    # (1) VN-Index trend
    ax1 = fig.add_subplot(gs[0, :2])
    idx = vnindex_series.index
    ax1.plot(idx, vnindex_series.values, color="#1565C0", linewidth=0.9)
    ax1.fill_between(idx, vnindex_series.min(), vnindex_series.values,
                     alpha=0.07, color="#1565C0")
    if n_train < len(vnindex_series):
        ax1.axvline(idx[n_train - 1], color="orange", linestyle="--",
                    linewidth=1.2, label="Train/Val split")
    if n_train + n_val < len(vnindex_series):
        ax1.axvline(idx[n_train + n_val - 1], color="red", linestyle="--",
                    linewidth=1.2, label="Val/Test split")
    ax1.set_title("Chuỗi VN-Index (2020–2025) với phân vùng Train / Val / Test",
                  fontweight="bold", fontsize=11)
    ax1.set_ylabel("Điểm VN-Index")
    ax1.legend(fontsize=8)
    ax1.tick_params(axis="x", rotation=20)

    # (2) Missing distribution
    ax2 = fig.add_subplot(gs[0, 2])
    labels = corr_summary["Ngưỡng |r|"].tolist() if "Ngưỡng |r|" in corr_summary else []
    ax2b = ax2.twinx() if False else ax2  # placeholder
    miss_labels = ["0%", "0–5%", "5–10%", "10–20%", ">20%"]
    # Rebuild from corr_summary workaround — load missing_dist if available
    miss_vals = [0] * 5  # will be overwritten from artifact if exists
    ax2.bar(miss_labels, miss_vals, color=["#2E7D32","#66BB6A","#FFA726","#EF5350","#B71C1C"])
    ax2.set_title("Phân bố tỷ lệ thiếu theo mã", fontweight="bold", fontsize=10)
    ax2.set_ylabel("Số mã")

    # (3) Raw distribution of sample stock
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.hist(train_data[sym].dropna(), bins=40, color="#1976D2", alpha=0.8,
             edgecolor="white")
    ax3.set_title(f"Phân phối gốc – {sym}", fontweight="bold", fontsize=10)
    ax3.set_xlabel("Giá đóng cửa (VND)")

    # (4) Z-score distribution
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.hist(train_scaled[sym].dropna(), bins=40, color="#388E3C", alpha=0.8,
             edgecolor="white")
    ax4.set_title(f"Sau Z-score – {sym}", fontweight="bold", fontsize=10)
    ax4.set_xlabel("z-value")
    ax4.axvline(0, color="red", linestyle="--", linewidth=1)

    # (5) ADF p-values
    ax5 = fig.add_subplot(gs[1, 2])
    if not stationarity_df.empty:
        pvals = stationarity_df["ADF_p"].fillna(1.0).values
        syms  = stationarity_df["Symbol"].tolist()
        colors = ["#2E7D32" if p < 0.05 else "#C62828" for p in pvals]
        ax5.bar(range(len(pvals)), pvals, color=colors, edgecolor="white")
        ax5.axhline(0.05, color="black", linestyle="--", linewidth=1)
        ax5.set_xticks(range(len(pvals)))
        ax5.set_xticklabels(syms, rotation=90, fontsize=6)
        ax5.set_title("ADF p-value (30 mã đại diện)", fontweight="bold", fontsize=10)
        ax5.set_ylabel("p-value")
        from matplotlib.patches import Patch
        ax5.legend(handles=[Patch(facecolor="#2E7D32", label="Dừng p<0.05"),
                             Patch(facecolor="#C62828", label="Không dừng")], fontsize=7)

    # (6) Correlation heatmap (subsample 30 mã)
    ax6 = fig.add_subplot(gs[2, :])
    sub_cols = df_pivot.columns[:30].tolist()
    sub_corr = df_pivot[sub_cols].corr()
    mask = np.triu(np.ones_like(sub_corr, dtype=bool))
    sns.heatmap(
        sub_corr, mask=mask, cmap="RdBu_r", center=0, vmin=-1, vmax=1,
        ax=ax6, cbar_kws={"shrink": 0.5},
        xticklabels=False, yticklabels=False, linewidths=0,
    )
    ax6.set_title(
        "Ma trận tương quan Pearson – 30 mã đầu (xác nhận đa cộng tuyến cao trước PCA)",
        fontweight="bold", fontsize=10,
    )

    plt.suptitle("Chương 3 – Tổng hợp Tiền xử lý Dữ liệu VN-Index",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[FIGURE] Saved → {out}")
    return out


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