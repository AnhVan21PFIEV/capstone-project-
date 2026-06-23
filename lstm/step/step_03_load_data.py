"""Step 03: Load PCA features and VNINDEX target."""
from step_02_paths import PCA_DIR, CORE_DIR
import pandas as pd


# Load PCA features and VNINDEX target
train_pca = pd.read_csv(PCA_DIR / "train_pca.csv", parse_dates=["Ngày"]).set_index("Ngày")
val_pca = pd.read_csv(PCA_DIR / "val_pca.csv", parse_dates=["Ngày"]).set_index("Ngày")
test_pca = pd.read_csv(PCA_DIR / "test_pca.csv", parse_dates=["Ngày"]).set_index("Ngày")
vnindex = pd.read_csv(CORE_DIR / "vnindex_target.csv", parse_dates=["Ngày"]).set_index("Ngày")

train_df = train_pca.join(vnindex, how="inner")
val_df = val_pca.join(vnindex, how="inner")
test_df = test_pca.join(vnindex, how="inner")

pc_cols = [c for c in train_df.columns if c.startswith("PC")]
target_col = "VNINDEX"

print("Shapes:")
print("  train_df:", train_df.shape)
print("  val_df  :", val_df.shape)
print("  test_df :", test_df.shape)
print("  number of PCs:", len(pc_cols))

assert len(pc_cols) == 11, "This notebook expects k=11 principal components."
assert train_df.index.is_monotonic_increasing and val_df.index.is_monotonic_increasing and test_df.index.is_monotonic_increasing
assert train_df.index.intersection(val_df.index).empty
assert val_df.index.intersection(test_df.index).empty

print("Date ranges:")
print("  train:", train_df.index.min().date(), "->", train_df.index.max().date())
print("  val  :", val_df.index.min().date(), "->", val_df.index.max().date())
print("  test :", test_df.index.min().date(), "->", test_df.index.max().date())
