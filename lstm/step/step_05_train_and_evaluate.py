"""Step 05: Train and evaluate LSTM models."""
import random
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow import keras
from tensorflow.keras import layers
from step_02_paths import PROJECT_ROOT
from step_04_prepare_data import (
    lookback_values, batch_size_values, epochs,
    make_windowed_data, add_target_history,
    train_scaled_df, val_scaled_df, test_scaled_df,
    x_scaler, y_scaler, pc_cols, target_col
)


def build_lstm_model(lookback: int, n_features: int):
    """Build LSTM model with deterministic weights."""
    try:
        tf.keras.utils.set_random_seed(42)
    except Exception:
        tf.random.set_seed(42)
    np.random.seed(42)
    random.seed(42)

    model = keras.Sequential([
        layers.Input(shape=(lookback, n_features)),
        layers.LSTM(64, return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(32),
        layers.Dropout(0.2),
        layers.Dense(16, activation="relu"),
        layers.Dense(1),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse",
        metrics=[keras.metrics.MeanAbsoluteError(name="mae")],
    )
    return model


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mape(y_true, y_pred):
    eps = 1e-8
    return float(np.mean(np.abs((y_true - y_pred) / (y_true + eps))) * 100.0)


results_dir = PROJECT_ROOT / "outputs" / "lstm_vnindex_sweep"
results_dir.mkdir(parents=True, exist_ok=True)

all_results = []
selected_train_dates = None
selected_val_dates = None
selected_test_dates = None
selected_history = None
selected_model = None

for lookback in lookback_values:
    X_train, y_train, train_dates = make_windowed_data(train_scaled_df, pc_cols, target_col, lookback)
    X_val, y_val, val_dates = make_windowed_data(val_scaled_df, pc_cols, target_col, lookback)
    X_test, y_test, test_dates = make_windowed_data(test_scaled_df, pc_cols, target_col, lookback)

    X_train_hist = add_target_history(train_scaled_df, target_col, lookback)
    X_val_hist = add_target_history(val_scaled_df, target_col, lookback)
    X_test_hist = add_target_history(test_scaled_df, target_col, lookback)

    X_train_final = np.concatenate([X_train, X_train_hist], axis=2)
    X_val_final = np.concatenate([X_val, X_val_hist], axis=2)
    X_test_final = np.concatenate([X_test, X_test_hist], axis=2)

    print("=" * 90)
    print(f"LOOKBACK = {lookback}")
    print(f"Train samples: {len(X_train_final)} | Validation samples: {len(X_val_final)} | Test samples: {len(X_test_final)}")

    for batch_size in batch_size_values:
        print("-" * 90)
        print(f"Training with batch_size = {batch_size}")

        model = build_lstm_model(lookback, X_train_final.shape[-1])
        callbacks = [
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=7, min_lr=1e-5),
        ]

        history = model.fit(
            X_train_final, y_train,
            validation_data=(X_val_final, y_val),
            epochs=epochs, batch_size=batch_size,
            callbacks=callbacks, verbose=1, shuffle=False,
        )

        train_pred_scaled = model.predict(X_train_final, verbose=0)
        val_pred_scaled = model.predict(X_val_final, verbose=0)
        test_pred_scaled = model.predict(X_test_final, verbose=0)

        train_pred = y_scaler.inverse_transform(train_pred_scaled).ravel()
        val_pred = y_scaler.inverse_transform(val_pred_scaled).ravel()
        test_pred = y_scaler.inverse_transform(test_pred_scaled).ravel()

        y_train_true = y_scaler.inverse_transform(y_train).ravel()
        y_val_true = y_scaler.inverse_transform(y_val).ravel()
        y_test_true = y_scaler.inverse_transform(y_test).ravel()

        metrics = {
            "Train": {"RMSE": rmse(y_train_true, train_pred), "MAE": mean_absolute_error(y_train_true, train_pred), "MAPE": mape(y_train_true, train_pred)},
            "Validation": {"RMSE": rmse(y_val_true, val_pred), "MAE": mean_absolute_error(y_val_true, val_pred), "MAPE": mape(y_val_true, val_pred)},
            "Test": {"RMSE": rmse(y_test_true, test_pred), "MAE": mean_absolute_error(y_test_true, test_pred), "MAPE": mape(y_test_true, test_pred)},
        }

        print("Train  - RMSE:", f"{metrics['Train']['RMSE']:.5f}", "MAE:", f"{metrics['Train']['MAE']:.5f}", "MAPE:", f"{metrics['Train']['MAPE']:.5f}%")
        print("Val    - RMSE:", f"{metrics['Validation']['RMSE']:.5f}", "MAE:", f"{metrics['Validation']['MAE']:.5f}", "MAPE:", f"{metrics['Validation']['MAPE']:.5f}%")
        print("Test   - RMSE:", f"{metrics['Test']['RMSE']:.5f}", "MAE:", f"{metrics['Test']['MAE']:.5f}", "MAPE:", f"{metrics['Test']['MAPE']:.5f}%")

        pred_table = pd.DataFrame({
            "Date": test_dates,
            "Actual_VNINDEX": y_test_true,
            "Predicted_VNINDEX": test_pred,
            "Residual": y_test_true - test_pred,
        })
        pred_filename = f"predictions_lookback_{lookback}_batch_{batch_size}.csv"
        pred_table.to_csv(results_dir / pred_filename, index=False)
        print(f"Saved: {results_dir / pred_filename}")

        try:
            best_epoch_run = int(np.argmin(history.history['val_loss']) + 1) if 'val_loss' in history.history else len(history.history.get('loss', []))
        except Exception:
            best_epoch_run = 'N/A'

        all_results.append({
            "Lookback": lookback, "Batch_size": batch_size,
            "Train_RMSE": metrics['Train']['RMSE'], "Train_MAE": metrics['Train']['MAE'], "Train_MAPE(%)": metrics['Train']['MAPE'],
            "Val_RMSE": metrics['Validation']['RMSE'], "Val_MAE": metrics['Validation']['MAE'], "Val_MAPE(%)": metrics['Validation']['MAPE'],
            "Test_RMSE": metrics['Test']['RMSE'], "Test_MAE": metrics['Test']['MAE'], "Test_MAPE(%)": metrics['Test']['MAPE'],
            "Train_samples": len(X_train_final), "Val_samples": len(X_val_final), "Test_samples": len(X_test_final),
            "Train_period_start": train_dates.min().strftime('%Y-%m-%d') if len(train_dates) > 0 else None,
            "Train_period_end": train_dates.max().strftime('%Y-%m-%d') if len(train_dates) > 0 else None,
            "Val_period_start": val_dates.min().strftime('%Y-%m-%d') if len(val_dates) > 0 else None,
            "Val_period_end": val_dates.max().strftime('%Y-%m-%d') if len(val_dates) > 0 else None,
            "Test_period_start": test_dates.min().strftime('%Y-%m-%d') if len(test_dates) > 0 else None,
            "Test_period_end": test_dates.max().strftime('%Y-%m-%d') if len(test_dates) > 0 else None,
            "Best_Epoch": best_epoch_run,
        })

        if lookback == 45 and batch_size == 16:
            selected_train_dates = train_dates
            selected_val_dates = val_dates
            selected_test_dates = test_dates
            selected_history = history
            selected_model = model
            selected_metrics_row = {
                'Train_RMSE': metrics['Train']['RMSE'], 'Train_MAE': metrics['Train']['MAE'], 'Train_MAPE(%)': metrics['Train']['MAPE'],
                'Val_RMSE': metrics['Validation']['RMSE'], 'Val_MAE': metrics['Validation']['MAE'], 'Val_MAPE(%)': metrics['Validation']['MAPE'],
                'Test_RMSE': metrics['Test']['RMSE'], 'Test_MAE': metrics['Test']['MAE'], 'Test_MAPE(%)': metrics['Test']['MAPE'],
                'Best_Epoch': best_epoch_run,
            }

summary_results = pd.DataFrame(all_results)
summary_results.to_csv(results_dir / "sweep_summary.csv", index=False)
print(f"\nSaved sweep summary to: {results_dir / 'sweep_summary.csv'}")
print("\nSummary Results:")
print(summary_results)
