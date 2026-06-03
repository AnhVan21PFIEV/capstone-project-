from __future__ import annotations

import pickle
from datetime import datetime


def run(context: dict) -> dict:
    export_dir = context["PROJECT_ROOT"] / "outputs" / "ardl_vnindex_forecast"
    export_dir.mkdir(parents=True, exist_ok=True)
    selected_pair = context["SELECTED_PAIR"]
    model_pkl_path = export_dir / f"ardl_model_P{selected_pair[0]}_Q{selected_pair[1]}.pkl"

    model_bundle = {
        "model_type": "statsmodels_ARDL",
        "selected_pair": selected_pair,
        "model_class": type(context["ardl_model"]).__name__,
        "results_class": type(context["ardl_res"]).__name__,
        "p_lags": list(context["ardl_model"]._lags),
        "q_lags_map": {k: list(v) for k, v in context["ardl_model"]._order.items()},
        "feature_columns": context["pc_cols"],
        "target_col": context["target_col"],
        "metrics": context["metrics"],
        "diagnostics": context["diag"],
        "forecast_path": str(context["forecast_path"]),
        "forecast_table_head": context["forecast_table"].head(10).to_dict(orient="records"),
        "model": context["ardl_model"],
        "results": context["ardl_res"],
        "exported_at": datetime.now().isoformat(),
        "version": "1.0",
    }

    with open(model_pkl_path, "wb") as f:
        pickle.dump(model_bundle, f)

    with open(model_pkl_path, "rb") as f:
        loaded_bundle = pickle.load(f)

    context.update({"model_pkl_path": model_pkl_path, "model_bundle": model_bundle, "loaded_bundle": loaded_bundle})
    print("ARDL step 7: saved pickle ->", model_pkl_path)
    print("  loaded pair:", loaded_bundle.get("selected_pair"))
    return context
