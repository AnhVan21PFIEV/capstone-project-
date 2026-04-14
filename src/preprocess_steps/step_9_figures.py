from __future__ import annotations

from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


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

    ax1 = fig.add_subplot(gs[0, :2])
    idx = vnindex_series.index
    ax1.plot(idx, vnindex_series.values, color="#1565C0", linewidth=0.9)
    ax1.fill_between(idx, vnindex_series.min(), vnindex_series.values, alpha=0.07, color="#1565C0")
    if n_train < len(vnindex_series):
        ax1.axvline(idx[n_train - 1], color="orange", linestyle="--", linewidth=1.2, label="Train/Val split")
    if n_train + n_val < len(vnindex_series):
        ax1.axvline(idx[n_train + n_val - 1], color="red", linestyle="--", linewidth=1.2, label="Val/Test split")
    ax1.set_title("Chuỗi VN-Index (2020-2025) với phân vùng Train / Val / Test", fontweight="bold", fontsize=11)
    ax1.set_ylabel("Điểm VN-Index")
    ax1.legend(fontsize=8)
    ax1.tick_params(axis="x", rotation=20)

    ax2 = fig.add_subplot(gs[0, 2])
    miss_labels = ["0%", "0-5%", "5-10%", "10-20%", ">20%"]
    miss_vals = [0] * 5
    ax2.bar(miss_labels, miss_vals, color=["#2E7D32", "#66BB6A", "#FFA726", "#EF5350", "#B71C1C"])
    ax2.set_title("Phân bố tỷ lệ thiếu theo mã", fontweight="bold", fontsize=10)
    ax2.set_ylabel("Số mã")

    ax3 = fig.add_subplot(gs[1, 0])
    ax3.hist(train_data[sym].dropna(), bins=40, color="#1976D2", alpha=0.8, edgecolor="white")
    ax3.set_title(f"Phân phối gốc - {sym}", fontweight="bold", fontsize=10)
    ax3.set_xlabel("Giá đóng cửa (VND)")

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.hist(train_scaled[sym].dropna(), bins=40, color="#388E3C", alpha=0.8, edgecolor="white")
    ax4.set_title(f"Sau Z-score - {sym}", fontweight="bold", fontsize=10)
    ax4.set_xlabel("z-value")
    ax4.axvline(0, color="red", linestyle="--", linewidth=1)

    ax5 = fig.add_subplot(gs[1, 2])
    if not stationarity_df.empty:
        pvals = stationarity_df["ADF_p"].fillna(1.0).values
        syms = stationarity_df["Symbol"].tolist()
        colors = ["#2E7D32" if p < 0.05 else "#C62828" for p in pvals]
        ax5.bar(range(len(pvals)), pvals, color=colors, edgecolor="white")
        ax5.axhline(0.05, color="black", linestyle="--", linewidth=1)
        ax5.set_xticks(range(len(pvals)))
        ax5.set_xticklabels(syms, rotation=90, fontsize=6)
        ax5.set_title("ADF p-value (30 mã đại diện)", fontweight="bold", fontsize=10)
        ax5.set_ylabel("p-value")

        from matplotlib.patches import Patch

        ax5.legend(
            handles=[
                Patch(facecolor="#2E7D32", label="Dừng p<0.05"),
                Patch(facecolor="#C62828", label="Không dừng"),
            ],
            fontsize=7,
        )

    ax6 = fig.add_subplot(gs[2, :])
    sub_cols = df_pivot.columns[:30].tolist()
    sub_corr = df_pivot[sub_cols].corr()
    mask = np.triu(np.ones_like(sub_corr, dtype=bool))
    sns.heatmap(
        sub_corr,
        mask=mask,
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        ax=ax6,
        cbar_kws={"shrink": 0.5},
        xticklabels=False,
        yticklabels=False,
        linewidths=0,
    )
    ax6.set_title(
        "Ma trận tương quan Pearson - 30 mã đầu (xác nhận đa cộng tuyến cao trước PCA)",
        fontweight="bold",
        fontsize=10,
    )

    plt.suptitle("Chương 3 - Tổng hợp Tiền xử lý Dữ liệu VN-Index", fontsize=13, fontweight="bold", y=1.01)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[FIGURE] Saved -> {out}")
    return out
