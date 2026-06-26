"""Step 07: Generate LSTM model summary report."""
import numpy as np
import pandas as pd
import tensorflow.keras.backend as K
from datetime import datetime
from step_02_paths import PROJECT_ROOT
from step_03_load_data import train_df, val_df, test_df, pc_cols, target_col
from step_04_prepare_data import epochs, x_scaler, y_scaler
from step_05_train_and_evaluate import (
    selected_model, selected_train_dates, selected_val_dates, selected_test_dates,
    selected_history, selected_metrics_row, summary_results
)


# Config for selected model
selected_lookback = 45
selected_batch_size = 16
selected_optimizer_name = "Adam"
selected_loss_function = "mse"
selected_scaler_type = "StandardScaler"
selected_train_ratio = 0.7

# Periods
try:
    train_period = f"{selected_train_dates.min().strftime('%d/%m/%Y')} - {selected_train_dates.max().strftime('%d/%m/%Y')}" if selected_train_dates is not None and len(selected_train_dates) > 0 else f"{train_df.index.min().strftime('%d/%m/%Y')} - {train_df.index.max().strftime('%d/%m/%Y')}"
except Exception:
    train_period = "N/A"

try:
    val_period = f"{selected_val_dates.min().strftime('%d/%m/%Y')} - {selected_val_dates.max().strftime('%d/%m/%Y')}" if selected_val_dates is not None and len(selected_val_dates) > 0 else f"{val_df.index.min().strftime('%d/%m/%Y')} - {val_df.index.max().strftime('%d/%m/%Y')}"
except Exception:
    val_period = "N/A"

try:
    test_period = f"{selected_test_dates.min().strftime('%d/%m/%Y')} - {selected_test_dates.max().strftime('%d/%m/%Y')}" if selected_test_dates is not None and len(selected_test_dates) > 0 else f"{test_df.index.min().strftime('%d/%m/%Y')} - {test_df.index.max().strftime('%d/%m/%Y')}"
except Exception:
    test_period = "N/A"

# Samples
train_samples = len(selected_train_dates) if selected_train_dates is not None else 'N/A'
val_samples = len(selected_val_dates) if selected_val_dates is not None else 'N/A'
test_samples = len(selected_test_dates) if selected_test_dates is not None else 'N/A'

try:
    total_observations = int(train_samples) + int(val_samples) + int(test_samples)
except Exception:
    try:
        total_observations = len(train_df) + len(val_df) + len(test_df)
    except Exception:
        total_observations = 'N/A'

# Feature list
feature_list_text = 'PCA features: ' + ', '.join(pc_cols)

# Learning rate and dropout
try:
    lr = float(K.get_value(selected_model.optimizer.learning_rate))
    lr_text = f"{lr:.6f}"
except Exception:
    lr_text = "N/A"

try:
    drops = [l for l in selected_model.layers if 'Dropout' in l.__class__.__name__]
    dropout_rate = getattr(drops[0], 'rate', 'N/A') if drops else 'N/A'
except Exception:
    dropout_rate = 'N/A'

# ===== SỬA PHẦN LẤY ARCHITECTURE =====
arch_rows = []
try:
    if selected_model is not None:
        # Đảm bảo model đã được build
        # Lấy thông tin các layer
        for i, layer in enumerate(selected_model.layers):
            # Lấy tên layer
            layer_name = layer.__class__.__name__
            
            # Lấy output shape
            try:
                if hasattr(layer, 'output_shape'):
                    out_shape = layer.output_shape
                    # Chuyển đổi None thành '?' để dễ đọc
                    if out_shape is not None:
                        if isinstance(out_shape, tuple):
                            # Xử lý tuple có None
                            out_shape_str = str(tuple('?' if dim is None else dim for dim in out_shape))
                        else:
                            out_shape_str = str(out_shape)
                    else:
                        out_shape_str = 'unknown'
                elif hasattr(layer, 'output'):
                    out_shape = layer.output.shape
                    out_shape_str = str(tuple('?' if dim is None else dim for dim in out_shape))
                else:
                    out_shape_str = 'unknown'
            except Exception:
                out_shape_str = 'unknown'
            
            # Lấy số tham số
            try:
                params = layer.count_params()
            except Exception:
                params = 'N/A'
            
            arch_rows.append((layer_name, out_shape_str, params))
        
        # Lấy tổng tham số
        try:
            total_params = selected_model.count_params()
        except:
            total_params = 'N/A'
            
        try:
            trainable_params = int(sum([K.count_params(w) for w in selected_model.trainable_weights]))
        except:
            trainable_params = 'N/A'
            
        try:
            if isinstance(total_params, (int, float)):
                if isinstance(trainable_params, (int, float)):
                    non_trainable_params = int(total_params - trainable_params)
                else:
                    non_trainable_params = 'N/A'
            else:
                non_trainable_params = 'N/A'
        except:
            non_trainable_params = 'N/A'
    else:
        arch_rows = []
        total_params = 'N/A'
        trainable_params = 'N/A'
        non_trainable_params = 'N/A'
except Exception as e:
    print(f"Warning: Could not retrieve model architecture: {e}")
    arch_rows = []
    total_params = 'N/A'
    trainable_params = 'N/A'
    non_trainable_params = 'N/A'

# Best epoch
best_epoch = 'N/A'
try:
    if selected_history is not None and hasattr(selected_history, 'history'):
        if 'val_loss' in selected_history.history:
            best_epoch = int(np.argmin(selected_history.history['val_loss']) + 1)
except Exception:
    pass

# Metrics
S_metrics = selected_metrics_row
if S_metrics is None:
    try:
        sel_df = summary_results.loc[(summary_results['Lookback'] == selected_lookback) & (summary_results['Batch_size'] == selected_batch_size)]
        if not sel_df.empty:
            S_metrics = sel_df.iloc[0].to_dict()
        else:
            S_metrics = summary_results.iloc[0].to_dict() if not summary_results.empty else None
    except Exception:
        S_metrics = summary_results.iloc[0].to_dict() if not summary_results.empty else None

# Print formatted report
line = "=" * 56
dash = "-" * 56
print(line)
print('                 LSTM MODEL SUMMARY')
print(line)
print('\nModel Type          : Sequential LSTM')
print(f'Target Variable     : {target_col}')
print('Forecast Horizon    : T+1')
print('\n' + dash)
print('DATASET INFORMATION')
print(dash)
print(f'Training Period     : {train_period}')
print(f'Validation Period   : {val_period}')
print(f'Testing Period      : {test_period}')
print('\nTotal Observations  :', total_observations)
print('Train Samples       :', train_samples)
print('Validation Samples  :', val_samples)
print('Test Samples        :', test_samples)
print('\nFeature Columns     :')
print(feature_list_text)
print('\nTarget Column       :', target_col)
print('\n' + dash)
print('HYPERPARAMETERS')
print(dash)
print('Look-back           :', selected_lookback)
print('Batch Size          :', selected_batch_size)
print('Epochs              :', epochs)
print('Optimizer           :', selected_optimizer_name)
print('Loss Function       : Mean Squared Error (MSE)')
print('\nScaler Type         :', selected_scaler_type)
print('Dropout Rate        :', dropout_rate)
print('Learning Rate       :', lr_text)
print('\n' + dash)
print('MODEL ARCHITECTURE')
print(dash)
print(f"{'Layer (Type)':35} {'Param #':15}")
for name, out_shape, params in arch_rows:
    # Định dạng để hiển thị đẹp, bỏ out_shape
    print(f"{name:35} {str(params):15}")
print('\n' + dash)
print('Total Parameters          :', total_params)
print('Trainable Parameters      :', trainable_params)
print('Non-trainable Parameters  :', non_trainable_params)
print('\n' + dash)
print('MODEL PERFORMANCE')
print(dash)

if S_metrics is not None:
    print('RMSE on Train Set         :', f"{S_metrics.get('Train_RMSE'):.5f}" if S_metrics.get('Train_RMSE') is not None else 'N/A')
    print('MAE on Train Set          :', f"{S_metrics.get('Train_MAE'):.5f}" if S_metrics.get('Train_MAE') is not None else 'N/A')
    print('MAPE on Train Set         :', f"{S_metrics.get('Train_MAPE(%)'):.5f} %" if S_metrics.get('Train_MAPE(%)') is not None else 'N/A')
    print('\nRMSE on Validation Set    :', f"{S_metrics.get('Val_RMSE'):.5f}" if S_metrics.get('Val_RMSE') is not None else 'N/A')
    print('MAE on Validation Set     :', f"{S_metrics.get('Val_MAE'):.5f}" if S_metrics.get('Val_MAE') is not None else 'N/A')
    print('MAPE on Validation Set    :', f"{S_metrics.get('Val_MAPE(%)'):.5f} %" if S_metrics.get('Val_MAPE(%)') is not None else 'N/A')
    print('\nRMSE on Test Set          :', f"{S_metrics.get('Test_RMSE'):.5f}" if S_metrics.get('Test_RMSE') is not None else 'N/A')
    print('MAE on Test Set           :', f"{S_metrics.get('Test_MAE'):.5f}" if S_metrics.get('Test_MAE') is not None else 'N/A')
    print('MAPE on Test Set          :', f"{S_metrics.get('Test_MAPE(%)'):.5f} %" if S_metrics.get('Test_MAPE(%)') is not None else 'N/A')
else:
    print('No metrics available.')
print('\n' + dash)
print('TRAINING STATUS')
print(dash)
print('Training Completed       :', 'Yes' if selected_history is not None else 'N/A')
print('Best Epoch               :', best_epoch)
print('Early Stopping           :', 'Enabled')
print('\n' + line)