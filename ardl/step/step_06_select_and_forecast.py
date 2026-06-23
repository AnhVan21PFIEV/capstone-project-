from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

from .common import diagnostics_from_residuals, mape, paired_valid, rmse


def run(context: dict) -> dict:
    selected_pair = context.get("SELECTED_PAIR", (5, 2))
    if selected_pair not in context["ardl_results_by_pair"]:
        raise KeyError(f"Selected pair {selected_pair} was not fit successfully in the sweep.")

    selected_entry = context["ardl_results_by_pair"][selected_pair]
    ardl_model = selected_entry["model"]
    ardl_res = selected_entry["res"]

    y_trainval = context["y_trainval"]
    X_trainval = context["X_trainval"]
    y_test = context["y_test"]
    X_test = context["X_test"]

    pred_trainval = ardl_res.predict(start=y_trainval.index[0], end=y_trainval.index[-1], exog=X_trainval)
    pred_test = ardl_res.predict(start=len(y_trainval), end=len(y_trainval) + len(y_test) - 1, exog_oos=X_test)
    pred_test.index = y_test.index

    y_trainval_eval, pred_trainval_eval = paired_valid(y_trainval, ardl_res.fittedvalues)
    y_test_eval, pred_test_eval = paired_valid(y_test, pred_test)

    metrics = {
        "RMSE_trainval": rmse(y_trainval_eval, pred_trainval_eval),
        "MAE_trainval": float(mean_absolute_error(y_trainval_eval, pred_trainval_eval)),
        "MAPE_trainval(%)": mape(y_trainval_eval, pred_trainval_eval),
        "R2_trainval": float(r2_score(y_trainval_eval, pred_trainval_eval)),
        "RMSE_test": rmse(y_test_eval, pred_test_eval),
        "MAE_test": float(mean_absolute_error(y_test_eval, pred_test_eval)),
        "MAPE_test(%)": mape(y_test_eval, pred_test_eval),
        "R2_test": float(r2_score(y_test_eval, pred_test_eval)),
    }

    forecast_table = pd.DataFrame({
        "Date": y_test.index,
        "Actual_VNINDEX": y_test.values,
        "Predicted_VNINDEX": pred_test.values,
        "Residual": y_test.values - pred_test.values,
    })

    results_dir = context["PROJECT_ROOT"] / "outputs" / "ardl_vnindex_forecast"
    results_dir.mkdir(parents=True, exist_ok=True)
    forecast_path = results_dir / f"ardl_test_forecast_P{selected_pair[0]}_Q{selected_pair[1]}.csv"
    forecast_table.to_csv(forecast_path, index=False)

    diag = diagnostics_from_residuals(ardl_res.resid)

    context.update({
        "SELECTED_PAIR": selected_pair,
        "ardl_model": ardl_model,
        "ardl_res": ardl_res,
        "pred_trainval": pred_trainval,
        "pred_test": pred_test,
        "metrics": metrics,
        "forecast_table": forecast_table,
        "forecast_path": forecast_path,
        "diag": diag,
    })

    print("ARDL step 6: selected pair", selected_pair)
    print("  RMSE_test =", f"{metrics['RMSE_test']:.6f}")
    print("  MAE_test  =", f"{metrics['MAE_test']:.6f}")
    print("  MAPE_test =", f"{metrics['MAPE_test(%)']:.6f}")
    print("  R2_test   =", f"{metrics['R2_test']:.6f}")
    print("  Forecast  =", forecast_path)
    return context
