"""Step 02: Path configuration."""
from pathlib import Path


def find_project_root() -> Path:
    """Find the project root that contains processed PCA files."""
    expected = Path("data/processed/pca/train_pca.csv")
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
        for p in Path("/content").rglob("train_pca.csv"):
            if p.parent.name == "pca":
                root = p.parents[3]
                if (root / expected).exists():
                    return root

    raise FileNotFoundError(
        "Cannot find data/processed/pca/train_pca.csv. Put the notebook inside the project or mount the project folder in Colab."
    )


PROJECT_ROOT = find_project_root()
PCA_DIR = PROJECT_ROOT / "data/processed/pca"
CORE_DIR = PROJECT_ROOT / "data/processed/core"

print("PROJECT_ROOT:", PROJECT_ROOT)
print("PCA_DIR exists:", PCA_DIR.exists())
print("CORE_DIR exists:", CORE_DIR.exists())
