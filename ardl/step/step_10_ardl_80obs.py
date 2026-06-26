from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

from .common import mape, paired_valid, rmse


def run(context: dict) -> dict:
    """
    Tính kết quả ARDL trên 80 quan sát (30/12/2024 - 29/04/2025)
    """
    selected_pair = context.get("SELECTED_PAIR", (5, 2))
    y_test = context["y_test"]
    pred_test = context["pred_test"]
    
    # Lọc dữ liệu từ 30/12/2024
    start_date_80 = pd.Timestamp('2024-12-30')
    mask_80 = y_test.index >= start_date_80
    
    y_test_80 = y_test[mask_80]
    pred_test_80 = pred_test[mask_80]
    
    y_test_eval, pred_test_eval = paired_valid(y_test_80, pred_test_80)
    
    # Tính metrics
    metrics_80 = {
        "n_obs": len(y_test_eval),
        "RMSE": rmse(y_test_eval, pred_test_eval),
        "MAE": float(mean_absolute_error(y_test_eval, pred_test_eval)),
        "MAPE": mape(y_test_eval, pred_test_eval),
        "R2": float(r2_score(y_test_eval, pred_test_eval)),
    }
    
    print("\n" + "="*70)
    print("📊 KẾT QUẢ ARDL TRÊN 80 QUAN SÁT")
    print("Giai đoạn: 30/12/2024 - 29/04/2025")
    print("="*70)
    print(f"Số quan sát:     {metrics_80['n_obs']}")
    print(f"RMSE:            {metrics_80['RMSE']:.4f} điểm")
    print(f"MAE:             {metrics_80['MAE']:.4f} điểm")
    print(f"MAPE:            {metrics_80['MAPE']:.4f}%")
    print(f"R²:              {metrics_80['R2']:.4f}")
    print("="*70)
    
    # So sánh với LSTM
    lstm_rmse = 43.6124
    lstm_mae = 33.6747
    lstm_mape = 2.6883
    
    print("\n📊 SO SÁNH VỚI PCA-LSTM (cùng 80 quan sát)")
    print("="*70)
    print(f"Chỉ số     ARDL n=80     LSTM n=80     ARDL tốt hơn")
    print("-"*70)
    print(f"RMSE       {metrics_80['RMSE']:.4f}      {lstm_rmse:.4f}      {(lstm_rmse - metrics_80['RMSE'])/lstm_rmse*100:.2f}%")
    print(f"MAE        {metrics_80['MAE']:.4f}      {lstm_mae:.4f}      {(lstm_mae - metrics_80['MAE'])/lstm_mae*100:.2f}%")
    print(f"MAPE       {metrics_80['MAPE']:.4f}%    {lstm_mape:.4f}%    {(lstm_mape - metrics_80['MAPE'])/lstm_mape*100:.2f}%")
    print("="*70)
    
    # Lưu kết quả
    forecast_80 = pd.DataFrame({
        "Date": y_test_80.index,
        "Actual_VNINDEX": y_test_80.values,
        "Predicted_VNINDEX": pred_test_80.values,
        "Residual": y_test_80.values - pred_test_80.values,
        "APE_%": np.abs((y_test_80.values - pred_test_80.values) / (y_test_80.values + 1e-8)) * 100
    })
    
    results_dir = context["PROJECT_ROOT"] / "outputs" / "ardl_vnindex_forecast"
    results_dir.mkdir(parents=True, exist_ok=True)
    forecast_path = results_dir / f"ardl_test_forecast_80obs_P{selected_pair[0]}_Q{selected_pair[1]}.csv"
    forecast_80.to_csv(forecast_path, index=False)
    print(f"\n✅ Đã lưu kết quả vào: {forecast_path}")
    
    context.update({
        "metrics_80": metrics_80,
        "forecast_80_path": forecast_path,
        "forecast_80": forecast_80,
    })
    
    return context