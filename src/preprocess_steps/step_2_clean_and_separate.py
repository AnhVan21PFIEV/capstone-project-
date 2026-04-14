from __future__ import annotations

from typing import Any, Dict, Tuple

import pandas as pd

from helpers.utils import parse_percent_to_float, safe_make_columns_numeric


def clean_and_separate(
    df: pd.DataFrame,
    cols: Dict[str, Any],
    remove_weekends: bool = True,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Phương châm:
    - VNINDEX là chỉ số tổng hợp (biến Y), không được đưa vào features.
      Nếu không tách, mô hình sẽ học trực tiếp từ biến cần dự báo -> data leakage.
    - Loại T7/CN: TTCK VN chỉ giao dịch T2-T6; ngày cuối tuần không có giao dịch
      thực, không nên xuất hiện trong chuỗi mô hình hóa.
    - Ngày lễ: Trong dữ liệu gốc thường không có dòng cho ngày lễ (không giao dịch),
      nên không cần xử lý riêng - trục thời gian chỉ chứa ngày giao dịch thực tế.
    """
    date_col = cols["date"]
    symbol_col = cols["symbol"]
    close_col = cols["close"]
    pct_col = cols.get("pct_change", "% Thay đổi")

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True)
    df = df.dropna(subset=[date_col, close_col, symbol_col])
    df = safe_make_columns_numeric(df, cols["numeric"])
    if pct_col in df.columns:
        df[pct_col] = parse_percent_to_float(df[pct_col])

    df = df.sort_values(date_col).reset_index(drop=True)

    if remove_weekends:
        n_before = len(df)
        df = df[df[date_col].dt.dayofweek < 5].reset_index(drop=True)
        n_removed = n_before - len(df)
        print(f"[WEEKEND] Loại {n_removed:,} dòng T7/CN ({n_removed / n_before * 100:.2f}%)")

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

    print(
        f"[TARGET] VNINDEX: {len(vnindex_series)} ngày "
        f"({vnindex_series.index.min().date()} -> {vnindex_series.index.max().date()})"
    )
    print(
        f"[FEATURE] Stocks: {len(stocks_df):,} records | "
        f"{stocks_df[symbol_col].nunique()} symbols"
    )

    vn = vnindex_series
    print(
        f"[VNINDEX STATS] min={vn.min():.2f} max={vn.max():.2f} "
        f"mean={vn.mean():.2f} std={vn.std():.2f} "
        f"skew={vn.skew():.4f} kurt={vn.kurt():.4f}"
    )

    return stocks_df, vnindex_series
