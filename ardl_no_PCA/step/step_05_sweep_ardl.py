from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from statsmodels.tsa.ardl import ARDL

from .common import mape, paired_valid, rmse


def run(context: dict) -> dict:
    trainval_df = pd.concat([context["train_df"], context["val_df"]], axis=0).sort_index()
    y_trainval = trainval_df[context["target_col"]].astype(float)
    X_trainval = trainval_df[context["feature_cols"]].astype(float)
    y_test = context["test_df"][context["target_col"]].astype(float)
    X_test = context["test_df"][context["feature_cols"]].astype(float)

    print("=" * 80)
    print("ARDL step 5: SWEEP với dữ liệu GỐC (NO PCA)")
    print("=" * 80)
    print(f"Train+Val period: {y_trainval.index.min().date()} -> {y_trainval.index.max().date()}")
    print(f"Test period: {y_test.index.min().date()} -> {y_test.index.max().date()}")
    print(f"Số features: {len(context['feature_cols'])}")
    print(f"Số samples train+val: {len(y_trainval)}")
    print(f"Số samples test: {len(y_test)}")
    print("-" * 80)

    # Giới hạn P và Q để tránh quá tải
    pq_pairs = [
        (1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (1, 5),
        (2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5),
        (3, 0), (3, 1), (3, 2), (3, 3), (3, 4), (3, 5),
        (4, 0), (4, 1), (4, 2), (4, 3), (4, 4), (4, 5),
        (5, 0), (5, 1), (5, 2), (5, 3), (5, 4), (5, 5),
    ]

    sweep_dir = context["PROJECT_ROOT"] / "outputs_no_PCA" / "ardl_vnindex_sweep"
    sweep_dir.mkdir(parents=True, exist_ok=True)

    ardl_results_by_pair = {}
    sweep_rows = []

    # In bảng header
    print(f"{'P':<4} {'Q':<4} {'Params':<8} {'AIC':<12} {'BIC':<12} {'Status':<15} {'Lý do'}")
    print("-" * 80)

    for p, q in pq_pairs:
        row = {"P": p, "Q": q}
        
        # Tính số tham số dự kiến
        n_features = len(context["feature_cols"])
        expected_params = 1 + p + (q + 1) * n_features  # intercept + p lags của y + (q+1)*features
        row["Expected_Params"] = expected_params
        
        try:
            model = ARDL(endog=y_trainval, lags=p, exog=X_trainval, order=q, trend="c")
            res = model.fit()

            fitted = res.fittedvalues
            pred_test = res.predict(start=len(y_trainval), end=len(y_trainval) + len(y_test) - 1, exog_oos=X_test)
            pred_test.index = y_test.index

            y_tr_eval, y_tr_pred = paired_valid(y_trainval, fitted)
            y_te_eval, y_te_pred = paired_valid(y_test, pred_test)

            row.update({
                "Status": "OK",
                "Num_Params": int(len(res.params)),
                "AIC": float(res.aic),
                "BIC": float(res.bic),
                "HQIC": float(res.hqic),
                "RMSE_trainval": rmse(y_tr_eval, y_tr_pred),
                "MAE_trainval": float(mean_absolute_error(y_tr_eval, y_tr_pred)),
                "MAPE_trainval(%)": mape(y_tr_eval, y_tr_pred),
                "R2_trainval": float(r2_score(y_tr_eval, y_tr_pred)),
                "RMSE_test": rmse(y_te_eval, y_te_pred),
                "MAE_test": float(mean_absolute_error(y_te_eval, y_te_pred)),
                "MAPE_test(%)": mape(y_te_eval, y_te_pred),
                "R2_test": float(r2_score(y_te_eval, y_te_pred)),
                "Reason": "OK"
            })

            pair_path = sweep_dir / f"forecast_P{p}_Q{q}.csv"
            pd.DataFrame({
                "Date": y_test.index,
                "Actual_VNINDEX": y_test.values,
                "Predicted_VNINDEX": pred_test.values,
            }).to_csv(pair_path, index=False)

            ardl_results_by_pair[(p, q)] = {
                "model": model,
                "res": res,
                "pred_test": pred_test,
                "fitted": fitted,
                "forecast_path": str(pair_path),
            }
            
            print(f"{p:<4} {q:<4} {len(res.params):<8} {res.aic:<12.4f} {res.bic:<12.4f} {'✅ OK':<15} {'':<20}")
            
        except Exception as exc:
            error_type = type(exc).__name__
            error_msg = str(exc)[:50]
            
            # Phân tích nguyên nhân lỗi
            if "singular" in str(exc).lower() or "Singular" in str(exc):
                reason = "Ma trận suy biến (Singular matrix) - do quá nhiều features"
            elif "memory" in str(exc).lower():
                reason = "Out of memory - quá nhiều tham số"
            elif "LinAlgError" in error_type:
                reason = f"Lỗi đại số tuyến tính: {error_msg}"
            elif "ValueError" in error_type:
                reason = f"ValueError: {error_msg}"
            else:
                reason = f"{error_type}: {error_msg}"
            
            row.update({
                "Status": f"FAIL: {error_type}",
                "Num_Params": np.nan,
                "AIC": np.nan,
                "BIC": np.nan,
                "HQIC": np.nan,
                "RMSE_trainval": np.nan,
                "MAE_trainval": np.nan,
                "MAPE_trainval(%)": np.nan,
                "R2_trainval": np.nan,
                "RMSE_test": np.nan,
                "MAE_test": np.nan,
                "MAPE_test(%)": np.nan,
                "R2_test": np.nan,
                "Reason": reason,
            })
            
            print(f"{p:<4} {q:<4} {expected_params:<8} {'':<12} {'':<12} {'❌ FAIL':<15} {reason[:40]}")
            
            # Lưu log chi tiết lỗi
            error_log_path = sweep_dir / f"error_log_P{p}_Q{q}.txt"
            with open(error_log_path, "w", encoding="utf-8") as f:
                f.write(f"P={p}, Q={q}\n")
                f.write(f"Error type: {error_type}\n")
                f.write(f"Error message: {exc}\n")
                f.write(f"Expected parameters: {expected_params}\n")
                f.write(f"Features: {len(context['feature_cols'])}\n")
                f.write(f"Train samples: {len(y_trainval)}\n")

        sweep_rows.append(row)

    ardl_sweep_table = pd.DataFrame(sweep_rows)
    sweep_csv = sweep_dir / "sweep_results.csv"
    ardl_sweep_table.to_csv(sweep_csv, index=False)
    
    print("-" * 80)
    print("ARDL step 5: sweep saved to", sweep_csv)
    print("=" * 80)

    context.update({
        "trainval_df": trainval_df,
        "y_trainval": y_trainval,
        "X_trainval": X_trainval,
        "y_test": y_test,
        "X_test": X_test,
        "pq_pairs": pq_pairs,
        "ardl_results_by_pair": ardl_results_by_pair,
        "ardl_sweep_table": ardl_sweep_table,
        "sweep_csv": sweep_csv,
    })
    return context