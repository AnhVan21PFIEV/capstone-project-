"""Step 08: Export LSTM forecast figures to logs/figures/lstm_no_pca/"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from scipy import stats
from pathlib import Path
from step_02_paths import PROJECT_ROOT


# ===== ĐỊNH NGHĨA ĐƯỜNG DẪN =====
figures_dir = PROJECT_ROOT / "logs" / "figures" / "lstm_no_pca"
figures_dir.mkdir(parents=True, exist_ok=True)
print(f"Figures directory: {figures_dir}")

# ===== SỬA: ĐỌC FILE DỰ BÁO CHO CẶP TỐI ƯU (LB=30, BS=16) =====
results_dir = PROJECT_ROOT / "outputs_no_PCA" / "lstm_vnindex_sweep"
selected_lookback = 30
selected_batch_size = 16
pred_file = results_dir / f"predictions_lookback_{selected_lookback}_batch_{selected_batch_size}.csv"

if not pred_file.exists():
    print(f"ERROR: File not found: {pred_file}")
    print(f"Please run step_05 first to generate predictions for lookback={selected_lookback}, batch_size={selected_batch_size}.")
    exit(1)

df = pd.read_csv(pred_file, parse_dates=['Date'])

# Tính sai số
df['Residual'] = df['Actual_VNINDEX'] - df['Predicted_VNINDEX']
df['APE_%'] = (abs(df['Residual']) / df['Actual_VNINDEX']) * 100

# Thống kê cơ bản
rmse = np.sqrt(np.mean(df['Residual']**2))
mae = np.mean(abs(df['Residual']))
mape = np.mean(df['APE_%'])
std_res = np.std(df['Residual'])
mean_res = np.mean(df['Residual'])
skew_res = stats.skew(df['Residual'])
kurt_res = stats.kurtosis(df['Residual'])

print(f"\nLoaded predictions: {len(df)} samples")
print(f"   Lookback: {selected_lookback}, Batch_size: {selected_batch_size}")
print(f"   Date range: {df['Date'].min().strftime('%d/%m/%Y')} -> {df['Date'].max().strftime('%d/%m/%Y')}")
print(f"   RMSE: {rmse:.4f}")
print(f"   MAE: {mae:.4f}")
print(f"   MAPE: {mape:.4f}%")
print(f"   Residual Mean: {mean_res:.4f}")
print(f"   Residual Std: {std_res:.4f}")
print(f"   Skewness: {skew_res:.4f}")
print(f"   Kurtosis: {kurt_res:.4f}")

# Định nghĩa tag cho tên file
model_tag = f"NO_PCA_LB{selected_lookback}_BS{selected_batch_size}"

print("\n" + "="*80)
print(f"ĐANG XUẤT ẢNH LSTM (NO PCA, LB={selected_lookback}, BS={selected_batch_size})...")
print("="*80)

# ===== 1. BIỂU ĐỒ SO SÁNH DỰ BÁO =====
print(f"\n  1. LSTM Forecast Plot (NO PCA, LB={selected_lookback}, BS={selected_batch_size})...")
fig1, ax1 = plt.subplots(figsize=(14, 6))
ax1.plot(df['Date'], df['Actual_VNINDEX'], 
         label='Actual VNINDEX', 
         linewidth=2, 
         color='blue')
ax1.plot(df['Date'], df['Predicted_VNINDEX'], 
         label=f'LSTM Forecast (NO PCA, LB={selected_lookback}, BS={selected_batch_size})', 
         linewidth=2, 
         color='red', 
         linestyle='--')
ax1.set_title(f'VNINDEX Forecast on Test Set (LSTM - NO PCA, LB={selected_lookback}, BS={selected_batch_size})', fontsize=14)
ax1.set_xlabel('Date')
ax1.set_ylabel('VNINDEX (points)')
ax1.legend(loc='best')
ax1.grid(alpha=0.3)
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%Y'))
ax1.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
plt.xticks(rotation=45)
plt.tight_layout()
fig1.savefig(figures_dir / f'lstm_no_pca_forecast_{model_tag}.png', dpi=300, bbox_inches='tight')
plt.close(fig1)
print(f"     Saved: lstm_no_pca_forecast_{model_tag}.png")

# ===== 2. BIỂU ĐỒ PHẦN DƯ THEO THỜI GIAN =====
print(f"  2. LSTM Residuals Plot (NO PCA, LB={selected_lookback}, BS={selected_batch_size})...")
fig2, ax2 = plt.subplots(figsize=(14, 5))
ax2.plot(df['Date'], df['Residual'], color='blue', linewidth=1.5, label='Residuals')
ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax2.axhline(y=std_res, color='red', linestyle='--', linewidth=0.8, label=f'+/-1σ ({std_res:.2f})')
ax2.axhline(y=-std_res, color='red', linestyle='--', linewidth=0.8)
ax2.fill_between(df['Date'], -std_res, std_res, alpha=0.1, color='red')
ax2.set_title(f'LSTM Residuals on Test Set (NO PCA, LB={selected_lookback}, BS={selected_batch_size})', fontsize=14)
ax2.set_xlabel('Date')
ax2.set_ylabel('Residual (points)')
ax2.legend(loc='best')
ax2.grid(alpha=0.3)
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%Y'))
ax2.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
plt.xticks(rotation=45)
plt.tight_layout()
fig2.savefig(figures_dir / f'lstm_no_pca_residuals_{model_tag}.png', dpi=300, bbox_inches='tight')
plt.close(fig2)
print(f"     Saved: lstm_no_pca_residuals_{model_tag}.png")

# ===== 3. HISTOGRAM PHẦN DƯ =====
print(f"  3. LSTM Histogram (NO PCA, LB={selected_lookback}, BS={selected_batch_size})...")
fig3, ax3 = plt.subplots(figsize=(10, 6))
n, bins, patches = ax3.hist(df['Residual'], bins=30, color='blue', alpha=0.7, edgecolor='black', density=True)
mu, sigma = np.mean(df['Residual']), np.std(df['Residual'])
x = np.linspace(df['Residual'].min(), df['Residual'].max(), 100)
ax3.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, label='Normal Distribution')
ax3.axvline(x=0, color='green', linestyle='-', linewidth=1.5, label=f'Mean = {mu:.2f}')
ax3.set_title(f'Histogram of LSTM Residuals (NO PCA, LB={selected_lookback}, BS={selected_batch_size})', fontsize=14)
ax3.set_xlabel('Residual (points)')
ax3.set_ylabel('Density')
ax3.legend(loc='best')
ax3.grid(alpha=0.3)
plt.tight_layout()
fig3.savefig(figures_dir / f'lstm_no_pca_histogram_{model_tag}.png', dpi=300, bbox_inches='tight')
plt.close(fig3)
print(f"     Saved: lstm_no_pca_histogram_{model_tag}.png")

# ===== 4. QQ-PLOT =====
print(f"  4. LSTM QQ-Plot (NO PCA, LB={selected_lookback}, BS={selected_batch_size})...")
fig4, ax4 = plt.subplots(figsize=(8, 8))
stats.probplot(df['Residual'], dist="norm", plot=ax4)
ax4.set_title(f'Q-Q Plot of LSTM Residuals (NO PCA, LB={selected_lookback}, BS={selected_batch_size})', fontsize=14)
ax4.grid(alpha=0.3)
plt.tight_layout()
fig4.savefig(figures_dir / f'lstm_no_pca_qqplot_{model_tag}.png', dpi=300, bbox_inches='tight')
plt.close(fig4)
print(f"     Saved: lstm_no_pca_qqplot_{model_tag}.png")

# ===== 5. SCATTER PLOT =====
print(f"  5. LSTM Scatter Plot (NO PCA, LB={selected_lookback}, BS={selected_batch_size})...")
fig5, ax5 = plt.subplots(figsize=(8, 8))
ax5.scatter(df['Actual_VNINDEX'], df['Predicted_VNINDEX'], alpha=0.5, color='blue', s=30)
min_val = min(df['Actual_VNINDEX'].min(), df['Predicted_VNINDEX'].min())
max_val = max(df['Actual_VNINDEX'].max(), df['Predicted_VNINDEX'].max())
ax5.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Fit')
corr = np.corrcoef(df['Actual_VNINDEX'], df['Predicted_VNINDEX'])[0, 1]
r2 = corr**2
ax5.text(0.05, 0.95, f'R2 = {r2:.4f}', transform=ax5.transAxes, fontsize=12, verticalalignment='top')
ax5.set_xlabel('Actual VNINDEX (points)')
ax5.set_ylabel('Predicted VNINDEX (points)')
ax5.set_title(f'Actual vs Predicted VNINDEX (LSTM - NO PCA, LB={selected_lookback}, BS={selected_batch_size})', fontsize=14)
ax5.legend(loc='best')
ax5.grid(alpha=0.3)
plt.tight_layout()
fig5.savefig(figures_dir / f'lstm_no_pca_scatter_{model_tag}.png', dpi=300, bbox_inches='tight')
plt.close(fig5)
print(f"     Saved: lstm_no_pca_scatter_{model_tag}.png")

# ===== 6. BOXPLOT PHẦN DƯ =====
print(f"  6. LSTM Boxplot (NO PCA, LB={selected_lookback}, BS={selected_batch_size})...")
fig6, ax6 = plt.subplots(figsize=(8, 6))
bp = ax6.boxplot(df['Residual'], vert=True, patch_artist=True,
                  boxprops=dict(facecolor='lightblue', color='blue'),
                  whiskerprops=dict(color='blue'),
                  capprops=dict(color='blue'),
                  medianprops=dict(color='red', linewidth=2))
ax6.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax6.set_title(f'Boxplot of LSTM Residuals (NO PCA, LB={selected_lookback}, BS={selected_batch_size})', fontsize=14)
ax6.set_ylabel('Residual (points)')
ax6.set_xticklabels(['Residuals'])
ax6.grid(alpha=0.3, axis='y')
plt.tight_layout()
fig6.savefig(figures_dir / f'lstm_no_pca_boxplot_{model_tag}.png', dpi=300, bbox_inches='tight')
plt.close(fig6)
print(f"     Saved: lstm_no_pca_boxplot_{model_tag}.png")

# ===== TỔNG KẾT =====
print("\n" + "="*80)
print(f"All 6 figures saved to: {figures_dir}")
print("="*80)
print(f"\nFigure files (LB={selected_lookback}, BS={selected_batch_size}):")
print(f"   lstm_no_pca_forecast_{model_tag}.png      (Forecast comparison)")
print(f"   lstm_no_pca_residuals_{model_tag}.png     (Residuals over time)")
print(f"   lstm_no_pca_histogram_{model_tag}.png     (Histogram of residuals)")
print(f"   lstm_no_pca_qqplot_{model_tag}.png        (Q-Q plot)")
print(f"   lstm_no_pca_scatter_{model_tag}.png       (Actual vs Predicted)")
print(f"   lstm_no_pca_boxplot_{model_tag}.png       (Boxplot of residuals)")
print("="*80)

# ===== LƯU FILE CSV THỐNG KÊ PHẦN DƯ =====
stats_df = pd.DataFrame({
    'Metric': ['RMSE', 'MAE', 'MAPE', 'Mean_Residual', 'Std_Residual', 'Min_Residual', 'Max_Residual', 'Skewness', 'Kurtosis'],
    'Value': [rmse, mae, mape, mean_res, std_res, df['Residual'].min(), df['Residual'].max(), skew_res, kurt_res]
})
stats_file = results_dir / f"residual_statistics_no_pca_lb{selected_lookback}_bs{selected_batch_size}.csv"
stats_df.to_csv(stats_file, index=False)
print(f"\nSaved residual statistics to: {stats_file}")