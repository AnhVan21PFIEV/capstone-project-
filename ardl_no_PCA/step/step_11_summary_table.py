from __future__ import annotations

import pandas as pd
import numpy as np

# Kiểm tra xem có tabulate không, nếu không thì dùng print thường
from tabulate import tabulate


def run(context: dict) -> dict:
    """In bảng tóm tắt mô hình ARDL + PCA"""
    
    res = context["ardl_res"]
    selected_pair = context["SELECTED_PAIR"]
    metrics = context["metrics"]
    diag = context["diag"]
    
    print("\nTóm tắt mô hình ARDL:")
    print("\n" + "="*90)
    print("                            ARDL + PCA Regression Results")
    print("="*90)
    
    # ===== 1. THÔNG TIN MÔ HÌNH =====
    info_data = [
        ["Dep. Variable:", "VNINDEX", "No. Observations:", len(context["y_trainval"])],
        ["Model:", f"ARDL({selected_pair[0]},{selected_pair[1]})+PCA", "Log Likelihood:", f"{res.llf:.3f}"],
        ["Date:", pd.Timestamp.now().strftime("%a, %d %b %Y"), "AIC:", f"{res.aic:.3f}"],
        ["Time:", pd.Timestamp.now().strftime("%H:%M:%S"), "BIC:", f"{res.bic:.3f}"],
        ["Sample:", f"0 - {len(context['y_trainval'])-1}", "HQIC:", f"{res.hqic:.3f}"],
        ["Covariance Type:", "nonrobust", "", ""]
    ]
    
    print(tabulate(info_data, tablefmt="plain", numalign="left", stralign="left"))
    
    # ===== 2. HỆ SỐ HỒI QUY =====
    print("\n" + "="*90)
    print("                 coef    std err        t      P>|t|      [0.025      0.975]")
    print("-"*90)
    
    # Lấy tên biến (loại bỏ khoảng trắng thừa)
    for idx, param in enumerate(res.params.index):
        param_name = param.strip()
        coef_val = res.params.iloc[idx]
        se_val = res.bse.iloc[idx]
        t_val = res.tvalues.iloc[idx]
        p_val = res.pvalues.iloc[idx]
        ci_lower = coef_val - 1.96 * se_val
        ci_upper = coef_val + 1.96 * se_val
        
        # Format theo đúng style của statsmodels
        print(f"{param_name:>18} {coef_val:10.4f} {se_val:10.4f} {t_val:9.3f} {p_val:9.3f} {ci_lower:10.4f} {ci_upper:10.4f}")
    
    # ===== 3. THỐNG KÊ PHẦN DƯ =====
    print("\n" + "="*90)
    
    # Lấy residuals từ context
    forecast_table = context.get("forecast_table")
    if forecast_table is not None and "Residual" in forecast_table.columns:
        resid = forecast_table["Residual"]
        resid_arr = np.array(resid.dropna())
        
        # Tính các thống kê phần dư
        jb_stat = diag.get('JarqueBera', 0)
        jb_p = diag.get('JB_pvalue', 0)
        skew = diag.get('Skew', 0)
        kurt = diag.get('Kurtosis', 0)
        
        # Lấy LB test từ diag (có thể được lưu từ step 9)
        lb_q1 = diag.get('LjungBox_Q_L1', 0)
        lb_p1 = diag.get('LjungBox_p_L1', 0)
        
        arch_stat = diag.get('ARCH_stat', 0)
        arch_p = diag.get('ARCH_pvalue', 0)
        
        # In thống kê dạng mẫu
        print(f"Ljung-Box (L1) (Q):       {lb_q1:>8.2f}        Jarque-Bera (JB):      {jb_stat:>8.2f}")
        print(f"Prob(Q):                   {lb_p1:>8.3f}        Prob(JB):               {jb_p:>8.3f}")
        print(f"ARCH LM:                   {arch_stat:>8.2f}        Prob(ARCH):             {arch_p:>8.3f}")
        print(f"Skew:                      {skew:>8.3f}        Kurtosis:               {kurt:>8.3f}")
    
    # ===== 4. HIỆU NĂNG DỰ BÁO =====
    print("\n" + "="*90)
    print(f"RMSE Train+Validation : {metrics['RMSE_trainval']:.4f}")
    print(f"RMSE Test             : {metrics['RMSE_test']:.4f}")
    
    # ===== 5. NHẬN XÉT TỔNG QUAN =====
    print("\n" + "="*90)
    print("NHẬN XÉT TỔNG QUAN")
    print("-"*90)
    
    # Nhận xét về hiệu năng dự báo
    if metrics['MAPE_test(%)'] < 2:
        mape_nx = "MAPE trên tập test < 2% cho thấy mô hình có độ chính xác rất cao"
    elif metrics['MAPE_test(%)'] < 5:
        mape_nx = "MAPE trên tập test < 5% cho thấy mô hình có độ chính xác tốt"
    else:
        mape_nx = "MAPE trên tập test ở mức trung bình"
    
    # Nhận xét về R²
    if metrics['R2_test'] > 0.7:
        r2_nx = f"R² = {metrics['R2_test']:.4f} cho thấy mô hình giải thích được trên 70% biến động của VN-Index"
    elif metrics['R2_test'] > 0.5:
        r2_nx = f"R² = {metrics['R2_test']:.4f} cho thấy mô hình giải thích được trên 50% biến động của VN-Index"
    else:
        r2_nx = f"R² = {metrics['R2_test']:.4f} cho thấy mô hình giải thích được dưới 50% biến động của VN-Index"
    
    # Nhận xét về phần dư
    jb_p = diag['JB_pvalue']
    if jb_p < 0.05:
        resid_nx = "Phần dư không tuân theo phân phối chuẩn (Jarque-Bera p < 0.05) - có thể tồn tại đuôi dày hoặc bất đối xứng"
    else:
        resid_nx = "Phần dư tuân theo phân phối chuẩn"
    
    arch_p = diag['ARCH_pvalue']
    if arch_p < 0.05:
        arch_nx = "Tồn tại hiệu ứng ARCH trong phần dư (p < 0.05) - phương sai sai số thay đổi theo thời gian"
    else:
        arch_nx = "Không có hiệu ứng ARCH trong phần dư"
    
    print(f"  • {mape_nx}.")
    print(f"  • {r2_nx}.")
    print(f"  • {resid_nx}.")
    print(f"  • {arch_nx}.")
    
    # Xác định mô hình phù hợp
    print("\n  Kết luận:")
    if metrics['MAPE_test(%)'] < 2 and metrics['R2_test'] > 0.6:
        print("     Mô hình PCA-ARDL(5,2) cho thấy hiệu năng dự báo tốt trên tập test,")
        print("     phù hợp cho bài toán dự báo VN-Index trong giai đoạn nghiên cứu.")
    elif metrics['MAPE_test(%)'] < 5:
        print("     Mô hình PCA-ARDL(5,2) đạt độ chính xác ở mức chấp nhận được,")
        print("     tuy nhiên cần xem xét các yếu tố phi tuyến để cải thiện thêm.")
    else:
        print("     Mô hình PCA-ARDL(5,2) có hiệu năng dự báo hạn chế trên tập test,")
        print("     cần xem xét các mô hình phi tuyến như LSTM để cải thiện.")
    
    print("\n" + "="*90)
    print(" Tóm tắt mô hình ARDL hoàn tất!")
    print("="*90 + "\n")
    
    return context