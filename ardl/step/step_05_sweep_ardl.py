from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from statsmodels.tsa.ardl import ARDL

from .common import mape, paired_valid, rmse


def run(context: dict) -> dict:
    trainval_df = pd.concat([context["train_df"], context["val_df"]], axis=0).sort_index()
    y_trainval = trainval_df[context["target_col"]].astype(float)
    X_trainval = trainval_df[context["pc_cols"]].astype(float)
    y_test = context["test_df"][context["target_col"]].astype(float)
    X_test = context["test_df"][context["pc_cols"]].astype(float)

    print("ARDL step 5: train+val period", y_trainval.index.min().date(), "->", y_trainval.index.max().date())
    print("ARDL step 5: test period", y_test.index.min().date(), "->", y_test.index.max().date())

    pq_pairs = [
        (1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (1, 5),
        (2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5),
        (3, 0), (3, 1), (3, 2), (3, 3), (3, 4), (3, 5),
        (4, 0), (4, 1), (4, 2), (4, 3), (4, 4), (4, 5),
        (5, 0), (5, 1), (5, 2), (5, 3), (5, 4), (5, 5),
    ]

    sweep_dir = context["PROJECT_ROOT"] / "outputs" / "ardl_vnindex_pca_sweep"
    sweep_dir.mkdir(parents=True, exist_ok=True)

    ardl_results_by_pair = {}
    sweep_rows = []

    for p, q in pq_pairs:
        row = {"P": p, "Q": q}
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
            print(f"(P={p}, Q={q}) -> OK | params={len(res.params)} | AIC={res.aic:.6f} | BIC={res.bic:.6f}")
        except Exception as exc:
            row.update({
                "Status": f"FAIL: {type(exc).__name__}",
                "Num_Params": np.nan,
                "AIC": np.nan,
                "BIC": np.nan,
                "HQIC": np.nan,
            })
            print(f"(P={p}, Q={q}) -> FAIL: {type(exc).__name__}: {exc}")

        sweep_rows.append(row)

    ardl_sweep_table = pd.DataFrame(sweep_rows)
    sweep_csv = sweep_dir / "sweep_results.csv"
    ardl_sweep_table.to_csv(sweep_csv, index=False)
    print("ARDL step 5: sweep saved to", sweep_csv)

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
