"""Step 06: Export the selected LSTM model to a pickle bundle."""
import pickle
from datetime import datetime
from step_02_paths import PROJECT_ROOT
from step_04_prepare_data import epochs, pc_cols, target_col, x_scaler, y_scaler
from step_05_train_and_evaluate import (
    selected_model, selected_train_dates, selected_val_dates, selected_test_dates,
    selected_metrics_row, summary_results, X_train_final
)


export_dir = PROJECT_ROOT / "outputs" / "lstm_vnindex_sweep"
export_dir.mkdir(parents=True, exist_ok=True)
model_pkl_path = export_dir / "lstm_vnindex_lb45_bs16.pkl"

# Explicitly define the chosen optimal pair for export
selected_lookback = 45
selected_batch_size = 16
selected_forecast_horizon = 1
selected_random_seed = 42
selected_optimizer_name = "Adam"
selected_loss_function = "mse"
selected_scaler_type = "StandardScaler"
selected_train_ratio = 0.7

# pick the row corresponding to the chosen pair if present
try:
    selected_row_df = summary_results.loc[(summary_results['Lookback'] == selected_lookback) & (summary_results['Batch_size'] == selected_batch_size)]
    if not selected_row_df.empty:
        selected_metrics = selected_row_df.iloc[0].to_dict()
    else:
        selected_metrics = summary_results.iloc[0].to_dict() if not summary_results.empty else {}
except Exception:
    selected_metrics = summary_results.iloc[0].to_dict() if not summary_results.empty else {}

model_bundle = {
    # MODEL
    "model_type": "keras_sequential_lstm",
    "model_config_json": selected_model.to_json() if selected_model is not None else None,
    "model_weights": selected_model.get_weights() if selected_model is not None else None,

    # HYPERPARAMETERS
    "lookback": selected_lookback,
    "batch_size": selected_batch_size,
    "epochs": epochs,
    "forecast_horizon": selected_forecast_horizon,

    # TRAIN CONFIG
    "optimizer_name": selected_optimizer_name,
    "loss_function": selected_loss_function,
    "random_seed": selected_random_seed,

    # DATA INFO
    "feature_columns": pc_cols,
    "target_col": target_col,
    "feature_count_per_timestep": X_train_final.shape[-1] if 'X_train_final' in dir() else None,

    # NORMALIZATION
    "x_scaler": x_scaler,
    "y_scaler": y_scaler,
    "scaler_type": selected_scaler_type,

    # EVALUATION
    "metrics": {
        "RMSE": selected_metrics.get("Test_RMSE"),
        "MAE": selected_metrics.get("Test_MAE"),
        "MAPE": selected_metrics.get("Test_MAPE(%)"),
    },

    # DATASET INFO
    "train_ratio": selected_train_ratio,

    # METADATA
    "selected_metrics": selected_metrics_row if selected_metrics_row else selected_metrics,
    "exported_at": datetime.now().isoformat(),
    "version": "1.0",
}

with open(model_pkl_path, "wb") as f:
    pickle.dump(model_bundle, f)

print(f"Saved LSTM model bundle to: {model_pkl_path}")
print("Bundle contents:")
print("- model_config_json")
print("- model_weights")
print("- x_scaler")
print("- y_scaler")
print("- lookback / batch_size / epochs / forecast_horizon")
print("- feature metadata and selected metrics")
