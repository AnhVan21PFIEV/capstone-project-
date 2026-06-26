#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_all.py - Run all steps for LSTM VNINDEX prediction
Execute this file to run the entire pipeline from imports to model summary.
"""

import sys
import os
import time
import traceback
from pathlib import Path


# Thêm thư mục step vào sys.path để có thể import các module
def setup_path():
    """Add step directory to Python path."""
    current_dir = Path(__file__).parent.absolute()
    step_dir = current_dir / "step"
    
    if step_dir.exists() and step_dir.is_dir():
        sys.path.insert(0, str(step_dir))
        return step_dir
    else:
        raise FileNotFoundError(f"Step directory not found: {step_dir}")


def find_data_root():
    """
    Find the root directory that contains the 'data' folder.
    Based on the actual structure: data/ is in the parent directory of lstm/
    """
    current_dir = Path(__file__).parent.absolute()  # lstm/
    
    # Kiểm tra các vị trí có thể chứa thư mục data
    # Cấu trúc thực tế: data/ nằm cùng cấp với lstm/
    candidates = [
        current_dir.parent,  # Thư mục chứa lstm/ (nơi có data/)
        current_dir,         # Trong trường hợp data/ nằm trong lstm/
        current_dir.parent.parent,
        Path("/content"),
        Path("/content/drive/MyDrive"),
    ]
    
    # Kiểm tra từng vị trí
    for root in candidates:
        data_dir = root / "data"
        if data_dir.exists() and data_dir.is_dir():
            # Kiểm tra xem có file PCA không
            pca_dir = data_dir / "processed" / "pca"
            if pca_dir.exists():
                # Kiểm tra các file cần thiết
                required_files = [
                    pca_dir / "train_pca.csv",
                    pca_dir / "val_pca.csv",
                    pca_dir / "test_pca.csv",
                    data_dir / "processed" / "core" / "vnindex_target.csv",
                ]
                # Nếu có ít nhất 1 file, coi như tìm thấy
                if any(f.exists() for f in required_files):
                    return root
    
    # Nếu không tìm thấy, thử tìm kiếm đệ quy
    try:
        for root in [current_dir.parent, Path("/content"), Path("/content/drive/MyDrive")]:
            if root.exists():
                for p in root.rglob("train_pca.csv"):
                    if "pca" in str(p.parent):
                        # Tìm thấy file, trả về thư mục chứa data
                        data_dir = p.parents[2]  # data/processed/pca/ -> data/
                        if data_dir.name == "data":
                            return data_dir.parent
    except Exception:
        pass
    
    raise FileNotFoundError(
        "Cannot find data/processed/pca/train_pca.csv. "
        "Expected structure:\n"
        "  project_root/\n"
        "  ├── data/\n"
        "  │   └── processed/\n"
        "  │       ├── pca/\n"
        "  │       │   ├── train_pca.csv\n"
        "  │       │   ├── val_pca.csv\n"
        "  │       │   └── test_pca.csv\n"
        "  │       └── core/\n"
        "  │           └── vnindex_target.csv\n"
        "  └── lstm/\n"
        "      ├── step/\n"
        "      └── run_all.py"
    )


def print_header(text: str, width: int = 80):
    """Print a formatted header."""
    print("\n" + "=" * width)
    print(f" {text} ".center(width))
    print("=" * width)


def print_directory_tree(path: Path, indent: str = "", max_depth: int = 2):
    """Print a simple directory tree for visualization."""
    if max_depth < 0:
        print(f"{indent}└── ...")
        return
    
    try:
        items = sorted([p for p in path.iterdir() if p.is_dir()])
        for i, item in enumerate(items):
            is_last = i == len(items) - 1
            prefix = "└── " if is_last else "├── "
            print(f"{indent}{prefix}{item.name}")
            if max_depth > 0:
                new_indent = indent + ("    " if is_last else "│   ")
                print_directory_tree(item, new_indent, max_depth - 1)
    except Exception:
        pass


def run_step(step_num: int, step_name: str, module_name: str):
    """
    Run a specific step by importing and executing its main code.
    
    Args:
        step_num: Step number
        step_name: Descriptive name of the step
        module_name: Name of the module to import (without .py extension)
    
    Returns:
        bool: True if successful, False otherwise
    """
    print_header(f"Step {step_num:02d}: {step_name}")
    print(f"Executing {module_name}.py...\n")
    
    start_time = time.time()
    
    try:
        # Import the module - this executes all code at module level
        __import__(module_name)
        
        elapsed = time.time() - start_time
        print(f"\n✓ Step {step_num} completed successfully in {elapsed:.2f} seconds")
        return True
        
    except FileNotFoundError as e:
        print(f"\n✗ ERROR: File not found - {e}")
        print("Please make sure all data files exist in the correct locations.")
        
    except ImportError as e:
        print(f"\n✗ ERROR: Import failed - {e}")
        print("Please check that all required dependencies are installed.")
        
    except AssertionError as e:
        print(f"\n✗ ERROR: Assertion failed - {e}")
        print("Data validation failed. Please check the data format and content.")
        
    except KeyboardInterrupt:
        print(f"\n⚠️ Step {step_num} interrupted by user.")
        raise
        
    except Exception as e:
        print(f"\n✗ ERROR: {type(e).__name__} - {e}")
        print("\nFull traceback:")
        traceback.print_exc()
    
    elapsed = time.time() - start_time
    print(f"\n✗ Step {step_num} failed after {elapsed:.2f} seconds")
    return False


def main():
    """Main execution function."""
    print_header("LSTM VNINDEX PREDICTION PIPELINE")
    print("Starting full pipeline execution...")
    print("=" * 80)
    
    # Tìm data root (thư mục chứa data)
    try:
        data_root = find_data_root()
        data_dir = data_root / "data"
        print(f"✓ Found data root: {data_root}")
        print(f"✓ Found data directory: {data_dir}")
        
        # Hiển thị cấu trúc thư mục data
        print("\n📁 Data directory structure:")
        print(f"{data_dir}")
        print_directory_tree(data_dir, max_depth=2)
        
        # Thiết lập biến môi trường cho các step sử dụng
        os.environ["DATA_ROOT"] = str(data_root)
        os.environ["PROJECT_ROOT"] = str(Path(__file__).parent.absolute())
        
    except FileNotFoundError as e:
        print(f"✗ {e}")
        print("\nPlease ensure your directory structure looks like this:")
        print("  project_root/")
        print("  ├── data/")
        print("  │   └── processed/")
        print("  │       ├── pca/")
        print("  │       │   ├── train_pca.csv")
        print("  │       │   ├── val_pca.csv")
        print("  │       │   └── test_pca.csv")
        print("  │       ├── core/")
        print("  │       │   └── vnindex_target.csv")
        print("  │       ├── quality/")
        print("  │       ├── splits/")
        print("  │       └── ...")
        print("  └── lstm/")
        print("      ├── step/")
        print("      │   ├── step_01_imports.py")
        print("      │   ├── step_02_paths.py")
        print("      │   └── ...")
        print("      └── run_all.py")
        sys.exit(1)
    
    # Thiết lập path để import các module từ thư mục step
    try:
        step_dir = setup_path()
        print(f"\n✓ Found step directory: {step_dir}")
        print(f"✓ Added {step_dir} to Python path")
    except FileNotFoundError as e:
        print(f"✗ {e}")
        print("Please make sure the 'step' directory exists with all step files.")
        sys.exit(1)
    
    # Define all steps in order
    steps = [
        (1, "Imports and environment setup", "step_01_imports"),
        (2, "Path configuration", "step_02_paths"),
        (3, "Load PCA features and VNINDEX target", "step_03_load_data"),
        (4, "Prepare data - scaling and windowing", "step_04_prepare_data"),
        (5, "Train and evaluate LSTM models", "step_05_train_and_evaluate"),
        (6, "Export the selected LSTM model", "step_06_export_model"),
        (7, "Generate LSTM model summary report", "step_07_model_summary"),
        (8, "Export figures to logs/figures/lstm", "step_08_export_figures"),
    ]
    
    # Verify all step files exist in the step directory
    missing_files = []
    for _, _, module_name in steps:
        step_file = step_dir / f"{module_name}.py"
        if not step_file.exists():
            missing_files.append(module_name)
    
    if missing_files:
        print("\n⚠️ WARNING: The following step files are missing in the step directory:")
        for f in missing_files:
            print(f"  - step/{f}.py")
        print("\nPlease ensure all step files are in the 'step' directory.")
        response = input("Continue anyway? (y/n): ").strip().lower()
        if response != 'y':
            print("Exiting...")
            sys.exit(1)
    else:
        print(f"✓ All {len(steps)} step files found")
    
    # Kiểm tra data files
    print("\n🔍 Checking data files...")
    pca_dir = data_dir / "processed" / "pca"
    core_dir = data_dir / "processed" / "core"
    
    required_files = {
        "train_pca.csv": pca_dir / "train_pca.csv",
        "val_pca.csv": pca_dir / "val_pca.csv",
        "test_pca.csv": pca_dir / "test_pca.csv",
        "vnindex_target.csv": core_dir / "vnindex_target.csv",
    }
    
    missing_data = []
    found_files = []
    for name, file_path in required_files.items():
        if file_path.exists():
            found_files.append(name)
        else:
            missing_data.append(f"{name} ({file_path})")
    
    if missing_data:
        print("\n⚠️ WARNING: The following data files are missing:")
        for f in missing_data:
            print(f"  - {f}")
        
        print(f"\nFound {len(found_files)}/{len(required_files)} required files")
        if found_files:
            print(f"Found: {', '.join(found_files)}")
        
        response = input("\nContinue anyway? (y/n): ").strip().lower()
        if response != 'y':
            print("Exiting...")
            sys.exit(1)
    else:
        print(f"✓ All {len(required_files)} required data files found")
        print(f"  - train_pca.csv: {required_files['train_pca.csv'].stat().st_size / 1024:.1f} KB")
        print(f"  - val_pca.csv: {required_files['val_pca.csv'].stat().st_size / 1024:.1f} KB")
        print(f"  - test_pca.csv: {required_files['test_pca.csv'].stat().st_size / 1024:.1f} KB")
        print(f"  - vnindex_target.csv: {required_files['vnindex_target.csv'].stat().st_size / 1024:.1f} KB")
    
    print("\n" + "=" * 80)
    
    # Track execution
    total_start = time.time()
    completed_steps = []
    failed_steps = []
    
    # Run each step sequentially
    for step_num, step_name, module_name in steps:
        try:
            success = run_step(step_num, step_name, module_name)
            
            if success:
                completed_steps.append(step_num)
            else:
                failed_steps.append(step_num)
                print("\n" + "=" * 80)
                print("⚠️ Pipeline stopped due to failure.")
                print(f"Failed at Step {step_num}: {step_name}")
                print("=" * 80)
                
                # Ask user whether to continue or stop
                response = input("\nContinue to next step? (y/n): ").strip().lower()
                if response != 'y':
                    break
        except KeyboardInterrupt:
            print("\n\n⚠️ Pipeline interrupted by user.")
            break
    
    # Summary
    total_elapsed = time.time() - total_start
    print_header("EXECUTION SUMMARY")
    print(f"Total execution time: {total_elapsed:.2f} seconds")
    print(f"Steps completed: {len(completed_steps)}/{len(steps)}")
    
    if failed_steps:
        print(f"Failed steps: {', '.join(str(s) for s in failed_steps)}")
        print("\n⚠️ Some steps failed. Please check the error messages above.")
        
        # Show which steps were completed successfully
        if completed_steps:
            print(f"✓ Completed steps: {', '.join(str(s) for s in completed_steps)}")
    else:
        print("✅ All steps completed successfully!")
        print("\n📁 Output files can be found in:")
        lstm_dir = Path(__file__).parent.absolute()
        outputs_dir = lstm_dir / "outputs" / "lstm_vnindex_sweep"
        print(f"  - {outputs_dir}/")
        print("    ├── sweep_summary.csv")
        print("    ├── predictions_lookback_*_batch_*.csv")
        print("    └── lstm_vnindex_lb45_bs16.pkl")
    
    # Show locations
    print(f"\n📂 Project structure:")
    print(f"  Data root: {data_root}")
    print(f"  Data directory: {data_dir}")
    lstm_dir = Path(__file__).parent.absolute()
    print(f"  LSTM directory: {lstm_dir}")
    outputs_dir = lstm_dir / "outputs"
    if outputs_dir.exists():
        print(f"  Output directory: {outputs_dir}")


if __name__ == "__main__":
    main()