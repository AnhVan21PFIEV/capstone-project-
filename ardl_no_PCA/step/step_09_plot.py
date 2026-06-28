from __future__ import annotations

import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
from pathlib import Path


def run(context: dict) -> dict:
    y_test = context["y_test"]
    pred_test = context["pred_test"]
    selected_pair = context["SELECTED_PAIR"]
    
    figures_dir = context.get("figures_dir")
    
    if figures_dir is None:
        figures_dir = Path("logs/figures/ardl_no_pca")
        figures_dir.mkdir(parents=True, exist_ok=True)
        context["figures_dir"] = figures_dir

    figures_dir.mkdir(parents=True, exist_ok=True)

    resid = y_test.values - pred_test.values
    
    print("\n" + "="*70)
    print(" ARDL STEP 9: ĐANG XUẤT ẢNH (NO PCA)...")
    print("="*70)
    print(f" Lưu vào: {figures_dir.resolve()}")

    # ===== ẢNH 1: Actual vs Predicted =====
    fig1, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(y_test.index, y_test.values, label="Actual VNINDEX", linewidth=2, color='blue')
    ax1.plot(pred_test.index, pred_test.values, label=f"ARDL Forecast (p={selected_pair[0]}, q={selected_pair[1]})", 
             linewidth=2, color='red', linestyle='--')
    ax1.set_title(f"VNINDEX Forecast on Test Set (ARDL - NO PCA)", fontsize=14)
    ax1.set_xlabel("Date")
    ax1.set_ylabel("VNINDEX")
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    fig1_path = figures_dir / f"ardl_no_pca_forecast_P{selected_pair[0]}_Q{selected_pair[1]}.png"
    fig1.savefig(fig1_path, dpi=300, bbox_inches="tight")
    plt.close(fig1)
    print(f"   Saved: {fig1_path.name}")

    # ===== ẢNH 2: Residuals theo thời gian =====
    fig2, ax2 = plt.subplots(figsize=(12, 4))
    ax2.plot(y_test.index, resid, color='red', linewidth=1, label='Residual')
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
    ax2.axhline(y=np.std(resid), color='gray', linestyle=':', linewidth=0.8, label=f'±1σ ({np.std(resid):.2f})')
    ax2.axhline(y=-np.std(resid), color='gray', linestyle=':', linewidth=0.8)
    ax2.set_title(f"ARDL Residuals on Test Set (NO PCA)", fontsize=14)
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Residual")
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    fig2_path = figures_dir / f"ardl_no_pca_residuals_P{selected_pair[0]}_Q{selected_pair[1]}.png"
    fig2.savefig(fig2_path, dpi=300, bbox_inches="tight")
    plt.close(fig2)
    print(f"   Saved: {fig2_path.name}")

    # ===== ẢNH 3: QQ-plot =====
    fig3, ax3 = plt.subplots(figsize=(6, 6))
    stats.probplot(resid, dist="norm", plot=ax3)
    ax3.set_title("QQ-plot of ARDL Residuals (NO PCA)", fontsize=14)
    
    fig3_path = figures_dir / f"ardl_no_pca_qqplot_P{selected_pair[0]}_Q{selected_pair[1]}.png"
    fig3.savefig(fig3_path, dpi=300, bbox_inches="tight")
    plt.close(fig3)
    print(f"   Saved: {fig3_path.name}")

    # ===== ẢNH 4: Histogram residuals =====
    fig4, ax4 = plt.subplots(figsize=(8, 5))
    ax4.hist(resid, bins=20, edgecolor='black', alpha=0.7, color='steelblue')
    ax4.axvline(x=0, color='red', linestyle='--', linewidth=1.5, label='Mean = 0')
    ax4.axvline(x=np.mean(resid), color='green', linestyle='--', linewidth=1.5, label=f'Mean = {np.mean(resid):.2f}')
    ax4.set_title("Histogram of ARDL Residuals (NO PCA)", fontsize=14)
    ax4.set_xlabel("Residual")
    ax4.set_ylabel("Frequency")
    ax4.legend()
    ax4.grid(alpha=0.3)
    
    fig4_path = figures_dir / f"ardl_no_pca_histogram_P{selected_pair[0]}_Q{selected_pair[1]}.png"
    fig4.savefig(fig4_path, dpi=300, bbox_inches="tight")
    plt.close(fig4)
    print(f"   Saved: {fig4_path.name}")

    # ===== ẢNH 5: Actual vs Predicted scatter =====
    fig5, ax5 = plt.subplots(figsize=(6, 6))
    ax5.scatter(y_test.values, pred_test.values, alpha=0.6, color='steelblue')
    ax5.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2, label='Perfect fit')
    ax5.set_xlabel("Actual VNINDEX")
    ax5.set_ylabel("Predicted VNINDEX")
    ax5.set_title(f"Actual vs Predicted (Test Set - NO PCA)\nR² = {context['metrics']['R2_test']:.4f}", fontsize=12)
    ax5.legend()
    ax5.grid(alpha=0.3)
    
    fig5_path = figures_dir / f"ardl_no_pca_scatter_P{selected_pair[0]}_Q{selected_pair[1]}.png"
    fig5.savefig(fig5_path, dpi=300, bbox_inches="tight")
    plt.close(fig5)
    print(f"   Saved: {fig5_path.name}")

    print(f"\n All {5} figures saved to: {figures_dir.resolve()}")
    print("="*70 + "\n")

    context["figures_dir"] = figures_dir
    return context