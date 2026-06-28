"""Step 06: Export the selected LSTM model to a pickle bundle (NO PCA)."""
import pickle
import numpy as np
from datetime import datetime
from step_02_paths import PROJECT_ROOT
from step_04_prepare_data import epochs, feature_cols, target_col, x_scaler, y_scaler

# Load context từ step 05
results_dir = PROJECT_ROOT / "outputs_no_PCA" / "lstm_vnindex_sweep"
context_path = results_dir / "context.pkl"

if not context_path.exists():
    print("ERROR: context.pkl not found. Please run step_05 first.")
    exit(1)

with open(context_path, 'rb') as f:
    context = pickle.load(f)

selected_model = context['selected_model']
selected_metrics_row = context['selected_metrics_row']
summary_results = context['summary_results']

export_dir = PROJECT_ROOT / "outputs_no_PCA" / "lstm_vnindex_sweep"
export_dir.mkdir(parents=True, exist_ok=True)

# ===== SỬA: Chọn cặp tối ưu là lookback=30, batch_size=16 =====
selected_lookback = 30
selected_batch_size = 16

model_pkl_path = export_dir / f"lstm_vnindex_lb{selected_lookback}_bs{selected_batch_size}_no_pca.pkl"

selected_forecast_horizon = 1
selected_random_seed = 42
selected_optimizer_name = "Adam"
selected_loss_function = "mse"
selected_scaler_type = "StandardScaler"
selected_train_ratio = 0.7

print("=" * 60)
print("EXPORTING LSTM MODEL (NO PCA)")
print(f"Selected lookback: {selected_lookback}")
print(f"Selected batch_size: {selected_batch_size}")
print("=" * 60)

# pick the row corresponding to the chosen pair
try:
    selected_row_df = summary_results.loc[(summary_results['Lookback'] == selected_lookback) & (summary_results['Batch_size'] == selected_batch_size)]
    if not selected_row_df.empty:
        selected_metrics = selected_row_df.iloc[0].to_dict()
        print("\nSelected model metrics:")
        print(f"  Train RMSE: {selected_metrics.get('Train_RMSE', 'N/A'):.4f}")
        print(f"  Val RMSE: {selected_metrics.get('Val_RMSE', 'N/A'):.4f}")
        print(f"  Test RMSE: {selected_metrics.get('Test_RMSE', 'N/A'):.4f}")
        print(f"  Test MAE: {selected_metrics.get('Test_MAE', 'N/A'):.4f}")
        print(f"  Test MAPE: {selected_metrics.get('Test_MAPE(%)', 'N/A'):.4f}%")
    else:
        print(f"WARNING: No results found for lookback={selected_lookback}, batch_size={selected_batch_size}")
        print("Using first row of summary_results as fallback.")
        selected_metrics = summary_results.iloc[0].to_dict() if not summary_results.empty else {}
except Exception as e:
    print(f"Error selecting metrics: {e}")
    selected_metrics = summary_results.iloc[0].to_dict() if not summary_results.empty else {}

# Lấy feature_count_per_timestep từ model
try:
    if selected_model is not None:
        feature_count = selected_model.input_shape[-1]
    else:
        feature_count = len(feature_cols) + 1  # features + target history
except Exception:
    feature_count = len(feature_cols) + 1

model_bundle = {
    # MODEL
    "model_type": "keras_sequential_lstm_no_pca",
    "model_config_json": selected_model.to_json() if selected_model is not None else None,
    "model_weights": selected_model.get_weights() if selected_model is not None else None,

    # HYPERPARAMETERS - SỬA
    "lookback": selected_lookback,
    "batch_size": selected_batch_size,
    "epochs": epochs,
    "forecast_horizon": selected_forecast_horizon,

    # TRAIN CONFIG
    "optimizer_name": selected_optimizer_name,
    "loss_function": selected_loss_function,
    "random_seed": selected_random_seed,

    # DATA INFO
    "feature_columns": feature_cols,
    "target_col": target_col,
    "feature_count_per_timestep": feature_count,

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
    "note": f"NO PCA - using original scaled features. Optimal: lookback={selected_lookback}, batch_size={selected_batch_size}"
}

with open(model_pkl_path, "wb") as f:
    pickle.dump(model_bundle, f)

print(f"\n✅ Saved LSTM model bundle (NO PCA) to: {model_pkl_path}")
print("\nBundle contents:")
print("- model_config_json")
print("- model_weights")
print("- x_scaler")
print("- y_scaler")
print(f"- lookback: {selected_lookback}")
print(f"- batch_size: {selected_batch_size}")
print("- epochs / forecast_horizon")
print("- feature metadata and selected metrics")
print("- note: NO PCA")
print("=" * 60)