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
    
    print("\n" + "="*80)
    print(" TÓM TẮT MÔ HÌNH ARDL + PCA")
    print("="*80)
    
    # ===== 1. THÔNG TIN MÔ HÌNH =====
    print("\n" + "-"*80)
    print("THÔNG TIN MÔ HÌNH")
    print("-"*80)
    
    info_data = [
        ["Mô hình", f"ARDL({selected_pair[0]}, {selected_pair[1]})"],
        ["Biến phụ thuộc", "VNINDEX"],
        ["Số quan sát (Train+Val)", len(context["y_trainval"])],
        ["Số quan sát (Test)", len(context["y_test"])],
        ["Số thành phần PCA (k)", len(context["pc_cols"])],
        ["Số tham số", len(res.params)],
        ["AIC", f"{res.aic:.4f}"],
        ["BIC", f"{res.bic:.4f}"],
        ["HQIC", f"{res.hqic:.4f}"],
    ]
    
    print(tabulate(info_data, headers=["Chỉ tiêu", "Giá trị"], tablefmt="grid"))

    
    # ===== 2. HỆ SỐ HỒI QUY =====
    print("\n" + "-"*80)
    print("HỆ SỐ HỒI QUY")
    print("-"*80)
    
    coef_data = []
    for idx, param in enumerate(res.params.index):
        pval = res.pvalues.iloc[idx]
        sig = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else ""
        coef_data.append([
            param,
            f"{res.params.iloc[idx]:.6f}",
            f"{res.bse.iloc[idx]:.6f}",
            f"{res.tvalues.iloc[idx]:.4f}",
            f"{pval:.4f}",
            sig
        ])
    
  
        print(tabulate(coef_data, 
                       headers=["Biến", "Hệ số", "Sai số chuẩn", "t-stat", "p-value", "Sig."],
                       tablefmt="grid"))
    
    # ===== 3. HIỆU NĂNG DỰ BÁO =====
    print("\n" + "-"*80)
    print("HIỆU NĂNG DỰ BÁO")
    print("-"*80)
    
    perf_data = [
        ["RMSE", f"{metrics['RMSE_trainval']:.4f}", f"{metrics['RMSE_test']:.4f}"],
        ["MAE", f"{metrics['MAE_trainval']:.4f}", f"{metrics['MAE_test']:.4f}"],
        ["MAPE (%)", f"{metrics['MAPE_trainval(%)']:.4f}", f"{metrics['MAPE_test(%)']:.4f}"],
        ["R²", f"{metrics['R2_trainval']:.4f}", f"{metrics['R2_test']:.4f}"],
    ]
    

    print(tabulate(perf_data, headers=["Chỉ số", "Train+Val", "Test"], tablefmt="grid"))
    # ===== 4. CHẨN ĐOÁN PHẦN DƯ =====
    print("\n" + "-"*80)
    print("CHẨN ĐOÁN PHẦN DƯ")
    print("-"*80)
    
    # Lấy residuals từ context
    forecast_table = context.get("forecast_table")
    if forecast_table is not None and "Residual" in forecast_table.columns:
        resid = forecast_table["Residual"]
        resid_arr = np.array(resid.dropna())
        print(f"  Số phần dư: {len(resid_arr)}")
        print(f"  Trung bình phần dư: {np.mean(resid_arr):.6f}")
        print(f"  Độ lệch chuẩn phần dư: {np.std(resid_arr):.6f}")
        print(f"  Min phần dư: {np.min(resid_arr):.6f}")
        print(f"  Max phần dư: {np.max(resid_arr):.6f}")
        print()
    
    diag_data = [
        ["Ljung-Box Q (lag 1)", f"{diag['LjungBox_Q_L1']:.4f}", f"{diag['LjungBox_p_L1']:.4f}"],
        ["Ljung-Box Q (lag 10)", f"{diag['LjungBox_Q_L10']:.4f}", f"{diag['LjungBox_p_L10']:.4f}"],
        ["Jarque-Bera", f"{diag['JarqueBera']:.4f}", f"{diag['JB_pvalue']:.4f}"],
        ["Skewness", f"{diag['Skew']:.4f}", ""],
        ["Kurtosis", f"{diag['Kurtosis']:.4f}", ""],
        ["ARCH-LM (lag 5)", f"{diag['ARCH_stat']:.4f}", f"{diag['ARCH_pvalue']:.4f}"],
    ]
    
    print(tabulate(diag_data, headers=["Kiểm định", "Thống kê", "p-value"], tablefmt="grid"))
    
    # ===== NHẬN XÉT TỔNG QUAN =====
    print("\n" + "-"*80)
    print("NHẬN XÉT TỔNG QUAN")
    print("-"*80)
    
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
    
    print("\n" + "="*80)
    print(" Tóm tắt mô hình ARDL hoàn tất!")
    print("="*80 + "\n")
    
    return context