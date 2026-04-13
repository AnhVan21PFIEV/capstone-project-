from __future__ import annotations

import argparse
from pathlib import Path

from pca_model import run_pca_pipeline
from preprocess import preprocess_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full VNINDEX pipeline (preprocess + PCA)")
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to YAML config file",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    config_path = project_root / args.config

    print("[STEP 1/2] Running preprocessing pipeline...")
    preprocess_pipeline(project_root=project_root, config_path=config_path)

    print("[STEP 2/2] Running PCA pipeline...")
    run_pca_pipeline(project_root=project_root, config_path=config_path)

    print("[DONE] Full pipeline completed successfully.")


if __name__ == "__main__":
    main()
