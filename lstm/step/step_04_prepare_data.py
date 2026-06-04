"""Step 04: Prepare data - scaling and windowing."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from step_03_load_data import train_df, val_df, test_df, pc_cols, target_col

# Hyperparameters for sweep
lookback_values = [30, 45, 60]
batch_size_values = [16, 32, 60]
epochs = 100

x_scaler = StandardScaler()
y_scaler = StandardScaler()

x_train_raw = train_df[pc_cols].astype(float)
y_train_raw = train_df[[target_col]].astype(float)
x_val_raw = val_df[pc_cols].astype(float)
y_val_raw = val_df[[target_col]].astype(float)
x_test_raw = test_df[pc_cols].astype(float)
y_test_raw = test_df[[target_col]].astype(float)

x_scaler.fit(x_train_raw)
y_scaler.fit(y_train_raw)

train_scaled_df = pd.DataFrame(x_scaler.transform(x_train_raw), index=train_df.index, columns=pc_cols)
val_scaled_df = pd.DataFrame(x_scaler.transform(x_val_raw), index=val_df.index, columns=pc_cols)
test_scaled_df = pd.DataFrame(x_scaler.transform(x_test_raw), index=test_df.index, columns=pc_cols)

train_scaled_df[target_col] = y_scaler.transform(y_train_raw).ravel()
val_scaled_df[target_col] = y_scaler.transform(y_val_raw).ravel()
test_scaled_df[target_col] = y_scaler.transform(y_test_raw).ravel()


def make_windowed_data(df: pd.DataFrame, feature_cols: list, target_col: str, lookback: int):
    """Create windowed time-series dataset."""
    x_values = df[feature_cols].astype(float).values
    y_values = df[[target_col]].astype(float).values
    x_windows, y_windows, end_dates = [], [], []
    for end_idx in range(lookback, len(df)):
        x_windows.append(x_values[end_idx - lookback:end_idx])
        y_windows.append(y_values[end_idx])
        end_dates.append(df.index[end_idx])
    return np.array(x_windows), np.array(y_windows), pd.Index(end_dates)


def add_target_history(df: pd.DataFrame, target_col: str, lookback: int):
    """Add target variable history as additional features."""
    target_hist = []
    values = df[target_col].values
    for end_idx in range(lookback, len(df)):
        target_hist.append(values[end_idx - lookback:end_idx].reshape(lookback, 1))
    return np.array(target_hist)


print("Prepared scaled datasets for sweep experiments.")
print("Lookback values:", lookback_values)
print("Batch sizes:", batch_size_values)
print("Epochs:", epochs)
