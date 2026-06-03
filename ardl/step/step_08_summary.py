from __future__ import annotations


def run(context: dict) -> dict:
    ardl_model = context["ardl_model"]
    ardl_res = context["ardl_res"]
    metrics = context["metrics"]
    diag = context["diag"]
    forecast_path = context["forecast_path"]

    print("TOM TAT MO HINH ARDL (with PCA exogenous variables):")
    print(f"Selected pair: {context['SELECTED_PAIR']}")
    print("Model used: fixed ARDL with PCA exogenous variables")
    print("  AR lags (p):", max(ardl_model._lags) if ardl_model._lags else 0)
    print("  AR lag list :", ardl_model._lags)
    print("  Exog lags map:", ardl_model._order)
    print("  Number of parameters:", len(ardl_res.params))
    print("")

    print(ardl_res.summary().tables[1])

    print("\nFINAL SCOREBOARD")
    print("=" * 70)
    print(f"So quan sat Train+Val: {len(context['y_trainval'])}")
    print(f"So quan sat Test     : {len(context['y_test'])}")
    print(f"So thanh phan PCA k  : {len(context['pc_cols'])}")
    print(f"AIC                  : {ardl_res.aic:.6f}")
    print(f"BIC                  : {ardl_res.bic:.6f}")
    print(f"HQIC                 : {ardl_res.hqic:.6f}")
    print("-" * 70)
    print(f"RMSE tren tap Train+Val: {metrics['RMSE_trainval']:.6f}")
    print(f"RMSE tren tap Test     : {metrics['RMSE_test']:.6f}")
    print(f"MAE tren tap Test      : {metrics['MAE_test']:.6f}")
    print(f"MAPE tren tap Test (%) : {metrics['MAPE_test(%)']:.6f}")
    print(f"R2 tren tap Test       : {metrics['R2_test']:.6f}")
    print("=" * 70)
    print(f"Forecast file           : {forecast_path}")

    print("\nDiagnostics:")
    for name, value in diag.items():
        print(f"  {name}: {value:.6f}")

    context["summary_displayed"] = True
    return context
