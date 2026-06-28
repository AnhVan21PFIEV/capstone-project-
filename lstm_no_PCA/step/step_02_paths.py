"""Step 02: Path configuration."""
from pathlib import Path


def find_project_root() -> Path:
    """Find the project root that contains processed files."""
    expected = Path("data/processed/splits/train_scaled.csv")
    candidates = [
        Path.cwd(),
        Path.cwd().parent,
        Path.cwd().parent.parent,
        Path("/content/capstone-project-"),
        Path("/content/drive/MyDrive/capstone-project-"),
    ]

    for root in candidates:
        if (root / expected).exists():
            return root

    if Path("/content").exists():
        for p in Path("/content").rglob("train_scaled.csv"):
            if p.parent.name == "splits":
                root = p.parents[3]  # .../data/processed/splits/train_scaled.csv
                if (root / expected).exists():
                    return root

    raise FileNotFoundError(
        "Cannot find data/processed/splits/train_scaled.csv. Check the project folder location."
    )


PROJECT_ROOT = find_project_root()
SPLITS_DIR = PROJECT_ROOT / "data/processed/splits"
CORE_DIR = PROJECT_ROOT / "data/processed/core"

print("PROJECT_ROOT:", PROJECT_ROOT)
print("SPLITS_DIR exists:", SPLITS_DIR.exists())
print("CORE_DIR exists:", CORE_DIR.exists())