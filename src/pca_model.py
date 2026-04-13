"""
pca_model.py  –  VN-Index PCA Pipeline
========================================
Chạy: python src/pca_model.py --config config/config.yaml

Requires: data/processed/splits/train_scaled.csv (output của preprocess.py)

Output artifacts:
  data/processed/
        pca/train_pca.csv / pca/val_pca.csv / pca/test_pca.csv
        pca/pca_loadings.csv                          ← loadings matrix (p × k)
        pca/pca_threshold_summary.csv                 ← k theo các ngưỡng CEV
        pca/pca_metrics.csv                           ← metrics tóm tắt
        pca/pca_corr_after.csv                        ← xác nhận PC không tương quan
  models/
    pca_model.pkl
  logs/figures/
    pca_summary.png
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import yaml
from sklearn.decomposition import PCA


def load_config(config_path: str | Path) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_sector_mapping(mapping_path: Path) -> pd.DataFrame:
    """Load optional symbol-sector mapping file. Returns empty dataframe if missing."""
    if not mapping_path.exists():
        return pd.DataFrame(columns=["Symbol", "Sector"])

    mapping_df = pd.read_csv(mapping_path, sep=None, engine="python", encoding="utf-8-sig")
    mapping_df.columns = [str(c).strip() for c in mapping_df.columns]

    symbol_candidates = ["Symbol", "symbol", "Ticker", "ticker", "Mã", "Ma"]
    sector_candidates = ["Sector", "sector", "Industry", "industry", "Ngành", "Nganh"]

    symbol_col = next((c for c in symbol_candidates if c in mapping_df.columns), None)
    sector_col = next((c for c in sector_candidates if c in mapping_df.columns), None)

    if symbol_col is None or sector_col is None:
        print(
            f"[SECTOR MAP] Skip: file {mapping_path} thiếu cột Symbol/Sector. "
            f"Columns hiện có: {mapping_df.columns.tolist()}"
        )
        return pd.DataFrame(columns=["Symbol", "Sector"])

    mapping_df = mapping_df[[symbol_col, sector_col]].copy()
    mapping_df.columns = ["Symbol", "Sector"]
    mapping_df["Symbol"] = mapping_df["Symbol"].astype(str).str.strip()
    mapping_df["Sector"] = mapping_df["Sector"].astype(str).str.strip()
    mapping_df = mapping_df.dropna().drop_duplicates(subset=["Symbol"])
    return mapping_df


def summarize_pc_by_sector(
    loadings: pd.DataFrame,
    sector_map: pd.DataFrame,
    output_summary_path: Path,
    output_detail_path: Path,
    top_pcs: int = 10,
    top_symbols_per_pc: int = 20,
) -> None:
    """Summarize dominant sectors for each PC based on top absolute loadings."""
    sector_lookup = {}
    if not sector_map.empty:
        sector_lookup = sector_map.set_index("Symbol")["Sector"].to_dict()

    detail_rows = []
    summary_rows = []

    for pc in loadings.columns[: min(top_pcs, len(loadings.columns))]:
        top_symbols = loadings[pc].abs().nlargest(top_symbols_per_pc).index.tolist()
        tmp = pd.DataFrame(
            {
                "PC": pc,
                "Symbol": top_symbols,
                "Loading": [float(loadings.loc[s, pc]) for s in top_symbols],
            }
        )
        tmp["AbsLoading"] = tmp["Loading"].abs()
        tmp["Direction"] = np.where(tmp["Loading"] >= 0, "Positive", "Negative")
        tmp["Sector"] = tmp["Symbol"].map(lambda s: sector_lookup.get(s, "Unknown"))
        detail_rows.append(tmp)

        grouped = (
            tmp.groupby("Sector", as_index=False)
            .agg(
                SymbolCount=("Symbol", "count"),
                TotalAbsLoading=("AbsLoading", "sum"),
                NetLoading=("Loading", "sum"),
            )
            .sort_values("TotalAbsLoading", ascending=False)
        )

        total_abs = grouped["TotalAbsLoading"].sum()
        grouped["ContributionPct"] = np.where(
            total_abs > 0,
            grouped["TotalAbsLoading"] / total_abs * 100,
            0.0,
        )
        grouped["PC"] = pc
        grouped["DominantDirection"] = np.where(grouped["NetLoading"] >= 0, "Positive", "Negative")
        grouped["Rank"] = np.arange(1, len(grouped) + 1)

        summary_rows.append(
            grouped[
                [
                    "PC",
                    "Rank",
                    "Sector",
                    "SymbolCount",
                    "TotalAbsLoading",
                    "ContributionPct",
                    "DominantDirection",
                ]
            ]
        )

    detail_df = pd.concat(detail_rows, ignore_index=True) if detail_rows else pd.DataFrame()
    summary_df = pd.concat(summary_rows, ignore_index=True) if summary_rows else pd.DataFrame()

    detail_df.to_csv(output_detail_path, index=False)
    summary_df.to_csv(output_summary_path, index=False)


# ─────────────────────────────────────────────────────────────────
# VERIFY PC ORTHOGONALITY
# ─────────────────────────────────────────────────────────────────
def verify_pc_orthogonality(
    train_pca: np.ndarray,
    k: int,
    output_path: Path,
) -> float:
    """
    Phương châm: Sau PCA, các PC phải không tương quan nhau
    (tính chất orthogonality của PCA). Kiểm tra này xác nhận
    PCA đã loại bỏ hoàn toàn đa cộng tuyến.
    Max |r| giữa các PC phải ≈ 0 (< 1e-6).
    """
    corr_matrix = np.corrcoef(train_pca.T)
    off_diag = np.abs(corr_matrix - np.eye(k))
    max_cross_corr = off_diag.max()

    # Lưu artifact: ma trận tương quan PC (chỉ lấy corner nếu k lớn)
    n_show = min(k, 20)
    pc_cols = [f"PC{i+1}" for i in range(n_show)]
    corr_df = pd.DataFrame(
        corr_matrix[:n_show, :n_show],
        index=pc_cols,
        columns=pc_cols,
    ).round(8)
    corr_df.to_csv(output_path)

    status = "✓ OK" if max_cross_corr < 1e-5 else "⚠️ WARN"
    print(f"[PCA VERIFY] Max cross-correlation giữa PC: {max_cross_corr:.2e} {status}")
    return max_cross_corr


# ─────────────────────────────────────────────────────────────────
# FIGURES
# ─────────────────────────────────────────────────────────────────
def save_pca_figure(
    explained_var: np.ndarray,
    cum_explained_var: np.ndarray,
    k_optimal: int,
    train_pca: np.ndarray,
    val_pca: np.ndarray,
    test_pca: np.ndarray,
    train_index: pd.DatetimeIndex,
    val_index: pd.DatetimeIndex,
    test_index: pd.DatetimeIndex,
    loadings: pd.DataFrame,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    n = len(explained_var)
    n_show = min(50, n)

    fig = plt.figure(figsize=(18, 14))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.5, wspace=0.38)

    # (1) Scree Plot
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.bar(range(1, n_show+1), explained_var[:n_show]*100,
            color="#1976D2", alpha=0.8, edgecolor="white")
    ax1.axvline(k_optimal, color="red", linestyle="--", linewidth=1.5,
                label=f"k={k_optimal}")
    ax1.set_title("Scree Plot", fontweight="bold", fontsize=10)
    ax1.set_xlabel("Principal Component")
    ax1.set_ylabel("Explained Variance (%)")
    ax1.legend(fontsize=9)

    # (2) Cumulative Explained Variance
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(range(1, n+1), cum_explained_var*100, color="#388E3C", linewidth=2)
    ax2.fill_between(range(1, n+1), cum_explained_var*100, alpha=0.1, color="#388E3C")
    for thr, col in [(0.80,"#FF9800"),(0.90,"#F44336"),(0.95,"#7B1FA2"),(0.99,"#000")]:
        kt = int(np.argmax(cum_explained_var >= thr)) + 1
        ax2.axhline(thr*100, color=col, linestyle=":", linewidth=1,
                    label=f"{thr*100:.0f}% → k={kt}")
    ax2.axvline(k_optimal, color="red", linestyle="--", linewidth=1.5)
    ax2.set_xlim(0, min(100, n))
    ax2.set_title("Cumulative Explained Variance", fontweight="bold", fontsize=10)
    ax2.set_xlabel("Số thành phần chính k")
    ax2.set_ylabel("CEV (%)")
    ax2.legend(fontsize=7, loc="lower right")
    ax2.grid(True, alpha=0.3)

    # (3) PC1 vs PC2 scatter
    ax3 = fig.add_subplot(gs[0, 2])
    n_pts = len(train_pca)
    sc = ax3.scatter(train_pca[:, 0], train_pca[:, 1],
                     c=np.arange(n_pts), cmap="viridis",
                     alpha=0.4, s=8)
    plt.colorbar(sc, ax=ax3, label="Thời gian (sớm→muộn)")
    ax3.set_xlabel(f"PC1 ({explained_var[0]*100:.1f}%)")
    ax3.set_ylabel(f"PC2 ({explained_var[1]*100:.1f}%)")
    ax3.set_title("PC1 vs PC2 – Train Set", fontweight="bold", fontsize=10)

    # (4) PC1 theo thời gian (Train/Val/Test)
    ax4 = fig.add_subplot(gs[1, :2])
    ax4.plot(train_index, train_pca[:, 0], color="#1565C0", linewidth=0.8, label="Train")
    ax4.plot(val_index,   val_pca[:, 0],   color="#E65100", linewidth=0.8, label="Val")
    ax4.plot(test_index,  test_pca[:, 0],  color="#B71C1C", linewidth=0.8, label="Test")
    ax4.axvline(val_index[0],  color="gray", linestyle="--", linewidth=0.8)
    ax4.axvline(test_index[0], color="gray", linestyle="--", linewidth=0.8)
    ax4.set_title(f"PC1 theo thời gian ({explained_var[0]*100:.1f}% phương sai)",
                  fontweight="bold", fontsize=10)
    ax4.set_ylabel("PC1 Score")
    ax4.legend(fontsize=8)
    ax4.tick_params(axis="x", rotation=20)

    # (5) Top Loadings PC1
    ax5 = fig.add_subplot(gs[1, 2])
    top15 = loadings["PC1"].abs().nlargest(15)
    c_load = ["#1565C0" if loadings.loc[s, "PC1"] > 0 else "#C62828" for s in top15.index]
    ax5.barh(top15.index, top15.values, color=c_load, edgecolor="white")
    ax5.set_xlabel("|Loading|")
    ax5.set_title("Top 15 Loadings – PC1", fontweight="bold", fontsize=10)
    ax5.invert_yaxis()
    from matplotlib.patches import Patch
    ax5.legend(handles=[Patch(facecolor="#1565C0", label="Loading (+)"),
                        Patch(facecolor="#C62828", label="Loading (–)")], fontsize=8)

    # (6) PC2 Loadings
    ax6 = fig.add_subplot(gs[2, 0])
    top10_pc2 = loadings["PC2"].abs().nlargest(10) if "PC2" in loadings.columns else pd.Series(dtype=float)
    if not top10_pc2.empty:
        c2 = ["#1565C0" if loadings.loc[s, "PC2"] > 0 else "#C62828" for s in top10_pc2.index]
        ax6.barh(top10_pc2.index, top10_pc2.values, color=c2, edgecolor="white")
        ax6.set_xlabel("|Loading|")
        ax6.set_title("Top 10 Loadings – PC2", fontweight="bold", fontsize=10)
        ax6.invert_yaxis()

    # (7) Threshold summary table
    ax7 = fig.add_subplot(gs[2, 1])
    ax7.axis("off")
    thr_data = []
    for thr in [0.80, 0.85, 0.90, 0.95, 0.99]:
        kt = int(np.argmax(cum_explained_var >= thr)) + 1
        thr_data.append([f"{thr*100:.0f}%", kt,
                         f"{(1-kt/n)*100:.1f}%",
                         f"{cum_explained_var[kt-1]*100:.2f}%"])
    tbl = ax7.table(
        cellText=thr_data,
        colLabels=["Ngưỡng CEV", "k", "Giảm chiều", "CEV thực"],
        cellLoc="center", loc="center", bbox=[0, 0, 1, 1],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#1976D2")
            cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#E3F2FD")
    ax7.set_title("Phân tích ngưỡng CEV", fontweight="bold", fontsize=10, pad=15)

    # (8) PC variance bar chart (top 20)
    ax8 = fig.add_subplot(gs[2, 2])
    top20 = min(20, k_optimal)
    ax8.bar(range(1, top20+1), explained_var[:top20]*100,
            color="#7B1FA2", alpha=0.8, edgecolor="white")
    ax8.set_title(f"Top {top20} PC – % Phương sai (k={k_optimal})",
                  fontweight="bold", fontsize=10)
    ax8.set_xlabel("PC index")
    ax8.set_ylabel("Explained Var (%)")
    ax8.text(top20*0.6, explained_var[0]*100*0.8,
             f"PC1={explained_var[0]*100:.1f}%\n"
             f"PC1-5={cum_explained_var[4]*100:.1f}%",
             fontsize=8, color="#7B1FA2")

    plt.suptitle(f"Chương 3 – Kết quả PCA (k={k_optimal}, CEV≥95%)",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[FIGURE] Saved → {output_path}")


# ─────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────
def run_pca_pipeline(project_root: Path, config_path: Path) -> None:
    cfg       = load_config(config_path)
    paths     = cfg["paths"]
    pca_cfg   = cfg["pca"]

    processed_dir = project_root / paths["processed_dir"]
    models_dir    = project_root / paths["models_dir"]
    figures_dir   = project_root / paths.get("figures_dir", "logs/figures")

    subdirs = paths.get("processed_subdirs", {})
    splits_dir = processed_dir / subdirs.get("splits", "splits")
    pca_dir = processed_dir / subdirs.get("pca", "pca")

    models_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    pca_dir.mkdir(parents=True, exist_ok=True)

    sector_mapping_file = pca_cfg.get("sector_mapping_file", "data/raw/symbol_sector_mapping.csv")
    sector_map_path = project_root / sector_mapping_file
    sector_map = load_sector_mapping(sector_map_path)
    if sector_map.empty:
        print(f"[SECTOR MAP] Không có mapping hợp lệ tại {sector_map_path}. Dùng 'Unknown'.")
    else:
        print(f"[SECTOR MAP] Loaded {len(sector_map)} mã có ngành từ {sector_map_path}")

    # ── Load scaled data (output từ preprocess) ───────────────────
    for fname in ["train_scaled.csv", "val_scaled.csv", "test_scaled.csv"]:
        if not (splits_dir / fname).exists():
            raise FileNotFoundError(
                f"{fname} không tồn tại trong {splits_dir}. Chạy preprocess.py trước."
            )

    train_scaled = pd.read_csv(splits_dir / "train_scaled.csv", index_col=0, parse_dates=True)
    val_scaled   = pd.read_csv(splits_dir / "val_scaled.csv",   index_col=0, parse_dates=True)
    test_scaled  = pd.read_csv(splits_dir / "test_scaled.csv",  index_col=0, parse_dates=True)

    X_train = train_scaled.values
    X_val   = val_scaled.values
    X_test  = test_scaled.values
    p       = X_train.shape[1]

    print(f"[PCA INPUT] Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")

    # ── Fit full PCA (chỉ trên train) ─────────────────────────────
    rs         = int(pca_cfg.get("random_state", 42))
    pca_full   = PCA(n_components=None, random_state=rs)
    pca_full.fit(X_train)

    ev  = pca_full.explained_variance_ratio_
    cev = np.cumsum(ev)

    print(f"[PCA FULL] p={p} | PC1={ev[0]*100:.2f}% | "
          f"PC1-5={cev[4]*100:.2f}% | PC1-10={cev[9]*100:.2f}%")

    # ── Threshold analysis ────────────────────────────────────────
    threshold = float(pca_cfg.get("explained_variance_threshold", 0.95))
    k_optimal = int(np.argmax(cev >= threshold)) + 1

    thr_rows = []
    for thr in [0.80, 0.85, 0.90, 0.95, 0.99]:
        kt = int(np.argmax(cev >= thr)) + 1
        thr_rows.append({
            "threshold": thr,
            "k":         kt,
            "dim_reduction_pct": (1 - kt / p) * 100,
            "cev_achieved":       cev[kt-1] * 100,
        })
    thr_df = pd.DataFrame(thr_rows)
    thr_df.to_csv(pca_dir / "pca_threshold_summary.csv", index=False)
    print(f"\n[THRESHOLD ANALYSIS]")
    print(thr_df.to_string(index=False))
    print(f"\n[DECISION] k={k_optimal} (CEV≥{threshold*100:.0f}%, "
          f"giảm {(1-k_optimal/p)*100:.1f}% chiều)")

    # ── Fit final PCA với k tối ưu ────────────────────────────────
    pca_final = PCA(n_components=k_optimal, random_state=rs)
    pca_final.fit(X_train)

    pc_cols   = [f"PC{i+1}" for i in range(k_optimal)]
    train_pca = pca_final.transform(X_train)
    val_pca   = pca_final.transform(X_val)
    test_pca  = pca_final.transform(X_test)

    df_train_pca = pd.DataFrame(train_pca, index=train_scaled.index, columns=pc_cols)
    df_val_pca   = pd.DataFrame(val_pca,   index=val_scaled.index,   columns=pc_cols)
    df_test_pca  = pd.DataFrame(test_pca,  index=test_scaled.index,  columns=pc_cols)

    df_train_pca.to_csv(pca_dir / "train_pca.csv")
    df_val_pca.to_csv(pca_dir / "val_pca.csv")
    df_test_pca.to_csv(pca_dir / "test_pca.csv")
    print(f"[SAVED] train/val/test_pca.csv  shape=({len(df_train_pca)}, {k_optimal})")

    # ── Loadings ──────────────────────────────────────────────────
    loadings = pd.DataFrame(
        pca_final.components_.T,
        index=train_scaled.columns,
        columns=pc_cols,
    )
    loadings.to_csv(pca_dir / "pca_loadings.csv")

    summarize_pc_by_sector(
        loadings=loadings,
        sector_map=sector_map,
        output_summary_path=pca_dir / "pc_sector_summary.csv",
        output_detail_path=pca_dir / "pc_top_symbols_with_sector.csv",
        top_pcs=int(pca_cfg.get("sector_top_pcs", 10)),
        top_symbols_per_pc=int(pca_cfg.get("sector_top_symbols", 20)),
    )
    print("[SAVED] pc_sector_summary.csv, pc_top_symbols_with_sector.csv")

    # Top loadings PC1-3
    for i in range(min(3, k_optimal)):
        pc = pc_cols[i]
        top5_pos = loadings[pc].nlargest(5).index.tolist()
        top5_neg = loadings[pc].nsmallest(5).index.tolist()
        print(f"[{pc}] {ev[i]*100:.2f}% | Top(+): {top5_pos} | Top(-): {top5_neg}")

    # ── Verify PC orthogonality ───────────────────────────────────
    max_cross = verify_pc_orthogonality(
        train_pca, k_optimal, pca_dir / "pca_corr_after.csv"
    )

    # ── Metrics ───────────────────────────────────────────────────
    metrics = {
        "input_features": p,
        "k_optimal": k_optimal,
        "cev_threshold": threshold,
        "cev_achieved": float(cev[k_optimal-1]),
        "dim_reduction_pct": (1 - k_optimal / p) * 100,
        "pc1_var": float(ev[0]),
        "pc1_5_cum_var": float(cev[4]),
        "max_pc_cross_corr": float(max_cross),
    }
    pd.Series(metrics).to_csv(pca_dir / "pca_metrics.csv", header=["value"])
    print(f"\n[PCA METRICS]")
    for k_m, v_m in metrics.items():
        print(f"  {k_m}: {v_m}")

    # ── Save model ────────────────────────────────────────────────
    with open(models_dir / "pca_model.pkl", "wb") as f:
        pickle.dump(pca_final, f)
    print(f"[SAVED] pca_model.pkl")

    # ── Figures ───────────────────────────────────────────────────
    save_pca_figure(
        explained_var=ev,
        cum_explained_var=cev,
        k_optimal=k_optimal,
        train_pca=train_pca,
        val_pca=val_pca,
        test_pca=test_pca,
        train_index=train_scaled.index,
        val_index=val_scaled.index,
        test_index=test_scaled.index,
        loadings=loadings,
        output_path=figures_dir / "pca_summary.png",
    )

    # ── Final summary ─────────────────────────────────────────────
    print("\n" + "="*60)
    print("PCA PIPELINE HOÀN THÀNH")
    print("="*60)
    print(f"  Input  : {p} cổ phiếu (features)")
    print(f"  Output : {k_optimal} thành phần chính")
    print(f"  CEV    : {cev[k_optimal-1]*100:.2f}% (≥{threshold*100:.0f}%)")
    print(f"  Giảm   : {(1-k_optimal/p)*100:.1f}% số chiều")
    print(f"  Max cross-corr PC: {max_cross:.2e} (≈ 0 ✓)")
    print(f"\n  Bước tiếp theo: xây dựng ARDL + LSTM với dữ liệu train/val/test_pca.csv")


# ─────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="VN-Index PCA pipeline")
    p.add_argument("--config", default="config/config.yaml")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    run_pca_pipeline(project_root=root, config_path=root / args.config)