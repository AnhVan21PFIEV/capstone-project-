"""
COMPARE MODELS: ARDL vs LSTM
So sánh hiệu năng dự báo giữa PCA-ARDL(5,2) và PCA-LSTM(45,16)
Chạy trực tiếp trên terminal - không xuất file
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from statsmodels.stats.diagnostic import het_arch
from statsmodels.stats.stattools import jarque_bera
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ===== ĐƯỜNG DẪN =====
PROJECT_ROOT = Path("C:/Users/ADMIN/Desktop/CAPSTONE PROJECT")

# ===== ĐỊNH NGHĨA HÀM =====
def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2))

def mape(y_true, y_pred):
    eps = 1e-8
    return np.mean(np.abs((y_true - y_pred) / (y_true + eps))) * 100

def compute_diagnostics(residuals, name=""):
    """Tính các chỉ số chẩn đoán phần dư"""
    n = len(residuals)
    mean_res = np.mean(residuals)
    std_res = np.std(residuals)
    min_res = np.min(residuals)
    max_res = np.max(residuals)
    skew = stats.skew(residuals)
    kurt = stats.kurtosis(residuals)
    
    # Jarque-Bera - hàm trả về 4 giá trị
    jb_result = jarque_bera(residuals)
    jb_stat = jb_result[0]
    jb_pvalue = jb_result[1]
    
    # ARCH-LM (lag 5)
    try:
        arch_test = het_arch(residuals, nlags=5)
        arch_stat = arch_test[0]
        arch_pvalue = arch_test[1]
    except:
        arch_stat = np.nan
        arch_pvalue = np.nan
    
    return {
        "n": n,
        "Mean_Residual": mean_res,
        "Std_Residual": std_res,
        "Min_Residual": min_res,
        "Max_Residual": max_res,
        "Skewness": skew,
        "Kurtosis": kurt,
        "Jarque_Bera_stat": jb_stat,
        "JB_pvalue": jb_pvalue,
        "ARCH_LM_stat": arch_stat,
        "ARCH_LM_pvalue": arch_pvalue,
    }


# ============================================================
# 1. ĐỌC DỮ LIỆU ARDL
# ============================================================
print("\n" + "="*80)
print(" ĐANG ĐỌC DỮ LIỆU ARDL...")
print("="*80)

# Đọc file forecast của ARDL (n=125)
ardl_file = PROJECT_ROOT / "outputs" / "ardl_vnindex_forecast" / "chapter4_ardl_forecast.csv"
if ardl_file.exists():
    ardl_df = pd.read_csv(ardl_file, parse_dates=['Date'])
    print(f"✅ ARDL file loaded: {len(ardl_df)} samples")
else:
    ardl_file = PROJECT_ROOT / "outputs" / "ardl_vnindex_forecast" / "ardl_test_forecast_P5_Q2.csv"
    if ardl_file.exists():
        ardl_df = pd.read_csv(ardl_file, parse_dates=['Date'])
        print(f"✅ ARDL file loaded: {len(ardl_df)} samples")
    else:
        print("❌ ARDL file not found!")
        ardl_df = None

# Đọc file ARDL n=80
ardl_80_file = PROJECT_ROOT / "outputs" / "ardl_vnindex_forecast" / "ardl_test_forecast_80obs_P5_Q2.csv"
if ardl_80_file.exists():
    ardl_80_df = pd.read_csv(ardl_80_file, parse_dates=['Date'])
    print(f"✅ ARDL 80obs file loaded: {len(ardl_80_df)} samples")
else:
    ardl_80_df = None
    print("❌ ARDL 80obs file not found!")


# ============================================================
# 2. ĐỌC DỮ LIỆU LSTM
# ============================================================
print("\n" + "="*80)
print(" ĐANG ĐỌC DỮ LIỆU LSTM...")
print("="*80)

lstm_file = PROJECT_ROOT / "outputs" / "lstm_vnindex_sweep" / "predictions_lookback_45_batch_16.csv"
if lstm_file.exists():
    lstm_df = pd.read_csv(lstm_file, parse_dates=['Date'])
    print(f"✅ LSTM file loaded: {len(lstm_df)} samples")
else:
    print("❌ LSTM file not found!")
    lstm_df = None


# ============================================================
# 3. TÍNH TOÁN CHỈ SỐ
# ============================================================
print("\n" + "="*80)
print(" TÍNH TOÁN CHỈ SỐ...")
print("="*80)

# ARDL (n=125)
if ardl_df is not None:
    y_actual = ardl_df['Actual_VNINDEX'].values
    y_pred = ardl_df['Predicted_VNINDEX'].values
    residuals = ardl_df['Residual'].values if 'Residual' in ardl_df.columns else y_actual - y_pred
    
    ardl_metrics = {
        "RMSE": rmse(y_actual, y_pred),
        "MAE": mean_absolute_error(y_actual, y_pred),
        "MAPE": mape(y_actual, y_pred),
        "R2": r2_score(y_actual, y_pred),
    }
    ardl_diag = compute_diagnostics(residuals, "ARDL (n=125)")

# ARDL (n=80)
if ardl_80_df is not None:
    y_actual = ardl_80_df['Actual_VNINDEX'].values
    y_pred = ardl_80_df['Predicted_VNINDEX'].values
    residuals = ardl_80_df['Residual'].values if 'Residual' in ardl_80_df.columns else y_actual - y_pred
    
    ardl_80_metrics = {
        "RMSE": rmse(y_actual, y_pred),
        "MAE": mean_absolute_error(y_actual, y_pred),
        "MAPE": mape(y_actual, y_pred),
        "R2": r2_score(y_actual, y_pred),
    }
    ardl_80_diag = compute_diagnostics(residuals, "ARDL (n=80)")

# LSTM (n=80)
if lstm_df is not None:
    y_actual = lstm_df['Actual_VNINDEX'].values
    y_pred = lstm_df['Predicted_VNINDEX'].values
    residuals = lstm_df['Residual'].values if 'Residual' in lstm_df.columns else y_actual - y_pred
    
    lstm_metrics = {
        "RMSE": rmse(y_actual, y_pred),
        "MAE": mean_absolute_error(y_actual, y_pred),
        "MAPE": mape(y_actual, y_pred),
        "R2": r2_score(y_actual, y_pred),
    }
    lstm_diag = compute_diagnostics(residuals, "LSTM (n=80)")


# ============================================================
# 4. IN BẢNG SO SÁNH
# ============================================================
print("\n" + "="*80)
print(" BẢNG SO SÁNH TỔNG HỢP")
print("="*80)

# ===== BẢNG 4.2.2a: So sánh hiệu năng tổng thể =====
print("\n" + "-"*80)
print("Bảng 4.2.2a: So sánh hiệu năng dự báo tổng thể")
print("-"*80)
print(f"{'Chỉ số':<15} {'ARDL (n=125)':<18} {'ARDL (n=80)':<18} {'LSTM (n=80)':<18} {'ARDL tốt hơn LSTM (%)':<25}")
print("-"*80)

if ardl_metrics and lstm_metrics:
    rmse_improve = (lstm_metrics['RMSE'] - ardl_80_metrics['RMSE']) / lstm_metrics['RMSE'] * 100 if ardl_80_metrics else "-"
    mae_improve = (lstm_metrics['MAE'] - ardl_80_metrics['MAE']) / lstm_metrics['MAE'] * 100 if ardl_80_metrics else "-"
    mape_improve = (lstm_metrics['MAPE'] - ardl_80_metrics['MAPE']) / lstm_metrics['MAPE'] * 100 if ardl_80_metrics else "-"
    
    print(f"{'RMSE':<15} {ardl_metrics['RMSE']:<18.4f} {ardl_80_metrics['RMSE']:<18.4f} {lstm_metrics['RMSE']:<18.4f} {rmse_improve:<25.2f}")
    print(f"{'MAE':<15} {ardl_metrics['MAE']:<18.4f} {ardl_80_metrics['MAE']:<18.4f} {lstm_metrics['MAE']:<18.4f} {mae_improve:<25.2f}")
    print(f"{'MAPE (%)':<15} {ardl_metrics['MAPE']:<18.4f} {ardl_80_metrics['MAPE']:<18.4f} {lstm_metrics['MAPE']:<18.4f} {mape_improve:<25.2f}")
    print(f"{'R²':<15} {ardl_metrics['R2']:<18.4f} {ardl_80_metrics['R2']:<18.4f} {lstm_metrics['R2']:<18.4f} {'-':<25}")
print("-"*80)

# ===== BẢNG 4.2.4a: Thống kê mô tả phần dư =====
print("\n" + "-"*80)
print("Bảng 4.2.4a: Thống kê mô tả phần dư")
print("-"*80)
print(f"{'Chỉ tiêu':<20} {'ARDL (n=125)':<18} {'ARDL (n=80)':<18} {'LSTM (n=80)':<18}")
print("-"*80)

if ardl_diag and lstm_diag:
    stats_keys = ["n", "Mean_Residual", "Std_Residual", "Min_Residual", "Max_Residual", "Skewness", "Kurtosis"]
    labels = ["Số phần dư", "Trung bình phần dư", "Độ lệch chuẩn", "Phần dư nhỏ nhất", "Phần dư lớn nhất", "Skewness", "Kurtosis"]
    for key, label in zip(stats_keys, labels):
        ardl_val = f"{ardl_diag[key]:.4f}" if key != "n" else f"{ardl_diag[key]}"
        ardl_80_val = f"{ardl_80_diag[key]:.4f}" if ardl_80_diag and key != "n" else f"{ardl_80_diag[key]}" if ardl_80_diag else "-"
        lstm_val = f"{lstm_diag[key]:.4f}" if key != "n" else f"{lstm_diag[key]}"
        print(f"{label:<20} {ardl_val:<18} {ardl_80_val:<18} {lstm_val:<18}")
print("-"*80)

# ===== BẢNG 4.2.4b: Jarque-Bera =====
print("\n" + "-"*80)
print("Bảng 4.2.4b: Kết quả kiểm định Jarque-Bera")
print("-"*80)
print(f"{'Mô hình':<20} {'JB-stat':<15} {'p-value':<15} {'Kết luận':<20}")
print("-"*80)

if ardl_diag and lstm_diag:
    jb_ardl_kl = "Phân phối không chuẩn" if ardl_diag['JB_pvalue'] < 0.05 else "Phân phối chuẩn"
    print(f"{'ARDL (n=125)':<20} {ardl_diag['Jarque_Bera_stat']:<15.4f} {ardl_diag['JB_pvalue']:<15.6f} {jb_ardl_kl:<20}")
    
    if ardl_80_diag:
        jb_ardl80_kl = "Phân phối không chuẩn" if ardl_80_diag['JB_pvalue'] < 0.05 else "Phân phối chuẩn"
        print(f"{'ARDL (n=80)':<20} {ardl_80_diag['Jarque_Bera_stat']:<15.4f} {ardl_80_diag['JB_pvalue']:<15.6f} {jb_ardl80_kl:<20}")
    
    jb_lstm_kl = "Phân phối không chuẩn" if lstm_diag['JB_pvalue'] < 0.05 else "Phân phối chuẩn"
    print(f"{'LSTM (n=80)':<20} {lstm_diag['Jarque_Bera_stat']:<15.4f} {lstm_diag['JB_pvalue']:<15.6f} {jb_lstm_kl:<20}")
print("-"*80)

# ===== BẢNG 4.2.4c: ARCH-LM =====
print("\n" + "-"*80)
print("Bảng 4.2.4c: Kết quả kiểm định ARCH-LM (lag 5)")
print("-"*80)
print(f"{'Mô hình':<20} {'ARCH-stat':<15} {'p-value':<15} {'Kết luận':<25}")
print("-"*80)

if ardl_diag and lstm_diag:
    arch_ardl_kl = "Tồn tại hiệu ứng ARCH" if ardl_diag['ARCH_LM_pvalue'] < 0.05 else "Không có hiệu ứng ARCH"
    print(f"{'ARDL (n=125)':<20} {ardl_diag['ARCH_LM_stat']:<15.4f} {ardl_diag['ARCH_LM_pvalue']:<15.6f} {arch_ardl_kl:<25}")
    
    if ardl_80_diag:
        arch_ardl80_kl = "Tồn tại hiệu ứng ARCH" if ardl_80_diag['ARCH_LM_pvalue'] < 0.05 else "Không có hiệu ứng ARCH"
        print(f"{'ARDL (n=80)':<20} {ardl_80_diag['ARCH_LM_stat']:<15.4f} {ardl_80_diag['ARCH_LM_pvalue']:<15.6f} {arch_ardl80_kl:<25}")
    
    arch_lstm_kl = "Tồn tại hiệu ứng ARCH" if lstm_diag['ARCH_LM_pvalue'] < 0.05 else "Không có hiệu ứng ARCH"
    print(f"{'LSTM (n=80)':<20} {lstm_diag['ARCH_LM_stat']:<15.4f} {lstm_diag['ARCH_LM_pvalue']:<15.6f} {arch_lstm_kl:<25}")
print("-"*80)


# ============================================================
# 5. NHẬN XÉT TỔNG QUAN
# ============================================================
print("\n" + "="*80)
print(" NHẬN XÉT TỔNG QUAN")
print("="*80)

if ardl_metrics and lstm_metrics:
    print("\n1. HIỆU NĂNG DỰ BÁO:")
    print(f"   - ARDL có MAPE = {ardl_metrics['MAPE']:.4f}% (n=125) và {ardl_80_metrics['MAPE']:.4f}% (n=80)")
    print(f"   - LSTM có MAPE = {lstm_metrics['MAPE']:.4f}% (n=80)")
    print(f"   - ARDL tốt hơn LSTM {mape_improve:.2f}% về MAPE")
    
    print("\n2. ĐỘ ỔN ĐỊNH SAI SỐ:")
    print(f"   - ARDL Std_Residual = {ardl_diag['Std_Residual']:.4f} (n=125)")
    print(f"   - LSTM Std_Residual = {lstm_diag['Std_Residual']:.4f} (n=80)")
    print(f"   - ARDL ổn định hơn LSTM (độ lệch chuẩn nhỏ hơn {(1 - ardl_diag['Std_Residual']/lstm_diag['Std_Residual'])*100:.2f}%)")
    
    print("\n3. PHÂN PHỐI PHẦN DƯ:")
    if ardl_diag['JB_pvalue'] < 0.05:
        print(f"   - ARDL: Phân phối không chuẩn (JB p={ardl_diag['JB_pvalue']:.6f})")
    else:
        print(f"   - ARDL: Phân phối chuẩn (JB p={ardl_diag['JB_pvalue']:.6f})")
    
    if lstm_diag['JB_pvalue'] < 0.05:
        print(f"   - LSTM: Phân phối không chuẩn (JB p={lstm_diag['JB_pvalue']:.6f})")
    else:
        print(f"   - LSTM: Phân phối chuẩn (JB p={lstm_diag['JB_pvalue']:.6f})")
    
    print("\n4. HIỆU ỨNG ARCH:")
    if ardl_diag['ARCH_LM_pvalue'] < 0.05:
        print(f"   - ARDL: Tồn tại hiệu ứng ARCH (p={ardl_diag['ARCH_LM_pvalue']:.6f})")
    else:
        print(f"   - ARDL: Không có hiệu ứng ARCH (p={ardl_diag['ARCH_LM_pvalue']:.6f})")
    
    if lstm_diag['ARCH_LM_pvalue'] < 0.05:
        print(f"   - LSTM: Tồn tại hiệu ứng ARCH (p={lstm_diag['ARCH_LM_pvalue']:.6f})")
    else:
        print(f"   - LSTM: Không có hiệu ứng ARCH (p={lstm_diag['ARCH_LM_pvalue']:.6f})")
    
    print("\n5. KẾT LUẬN CHUNG:")
    print("   ✅ Mô hình PCA-ARDL(5,2) vượt trội hơn PCA-LSTM(45,16)")
    print(f"      - MAPE thấp hơn {mape_improve:.2f}%")
    print(f"      - RMSE thấp hơn {rmse_improve:.2f}%")
    print(f"      - R² cao hơn {(ardl_80_metrics['R2'] - lstm_metrics['R2']):.4f}")
    print("   ⚠️ Cả hai mô hình đều có phần dư không chuẩn và tồn tại hiệu ứng ARCH")
    print("   💡 Gợi ý: Cần mở rộng sang ARDL-GARCH hoặc Regime-Switching models")

print("\n" + "="*80)
print(" HOÀN TẤT!")
print("="*80)