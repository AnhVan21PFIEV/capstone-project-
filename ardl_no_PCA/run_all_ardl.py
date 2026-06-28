# run_all_ardl.py
import sys
import os
from pathlib import Path

# Thêm đường dẫn gốc vào sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# Import các step
from ardl_no_PCA.step.step_01_setup import run as step_01_setup
from ardl_no_PCA.step.step_02_find_project_root import run as step_02_find_project_root
from ardl_no_PCA.step.step_03_load_data import run as step_03_load_data
from ardl_no_PCA.step.step_04_validate_data import run as step_04_validate_data
from ardl_no_PCA.step.step_04a_adf_stationarity_test import run as step_04a_adf_stationarity_test
from ardl_no_PCA.step.step_05_sweep_ardl import run as step_05_sweep_ardl
from ardl_no_PCA.step.step_06_select_and_forecast import run as step_06_select_and_forecast
from ardl_no_PCA.step.step_07_export_pkl import run as step_07_export_pkl
from ardl_no_PCA.step.step_08_summary import run as step_08_summary
from ardl_no_PCA.step.step_09_plot import run as step_09_plot
from ardl_no_PCA.step.step_10_ardl_80obs import run as step_10_ardl_80obs
from ardl_no_PCA.step.step_11_summary_table import run as step_11_summary_table
from ardl_no_PCA.step.step_13_check_singularity import run as step_13_check_singularity


def run_all():
    print("=" * 80)
    print("🚀 CHẠY ARDL (NO PCA) - FULL PIPELINE")
    print("=" * 80)
    
    # Đảm bảo chạy từ thư mục gốc
    current_dir = Path(__file__).resolve().parent.parent
    os.chdir(current_dir)
    
    context = {}
    
    # Chạy từng step
    context = step_01_setup(context)
    context = step_02_find_project_root(context)
    context = step_03_load_data(context)
    context = step_04_validate_data(context)
    context = step_04a_adf_stationarity_test(context)
    context = step_05_sweep_ardl(context)
    context = step_06_select_and_forecast(context)
    context = step_07_export_pkl(context)
    context = step_08_summary(context)
    context = step_09_plot(context)
    context = step_10_ardl_80obs(context)
    context = step_11_summary_table(context)
    context = step_13_check_singularity(context)

    
    print("\n" + "=" * 80)
    print("✅ ARDL (NO PCA) PIPELINE HOÀN THÀNH!")
    print(f"📁 Kết quả lưu tại: {context['PROJECT_ROOT'] / 'outputs_no_PCA'}")
    print("=" * 80)
    
    return context


if __name__ == "__main__":
    run_all()