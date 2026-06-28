"""Step 03: Load scaled data and VNINDEX target."""
from step_02_paths import SPLITS_DIR, CORE_DIR
import pandas as pd


# Load scaled data
train_scaled = pd.read_csv(SPLITS_DIR / "train_scaled.csv", index_col=0, parse_dates=True)
val_scaled = pd.read_csv(SPLITS_DIR / "val_scaled.csv", index_col=0, parse_dates=True)
test_scaled = pd.read_csv(SPLITS_DIR / "test_scaled.csv", index_col=0, parse_dates=True)

# Load VNINDEX target
vnindex = pd.read_csv(CORE_DIR / "vnindex_target.csv", parse_dates=["Ngày"]).set_index("Ngày")

# Merge target vào từng tập
train_df = train_scaled.join(vnindex, how="inner")
val_df = val_scaled.join(vnindex, how="inner")
test_df = test_scaled.join(vnindex, how="inner")

# Lấy danh sách features (tất cả các cột trừ VNINDEX)
feature_cols = [c for c in train_df.columns if c != "VNINDEX"]
target_col = "VNINDEX"

print("Shapes:")
print("  train_df:", train_df.shape)
print("  val_df  :", val_df.shape)
print("  test_df :", test_df.shape)
print("  number of features:", len(feature_cols))

print("Date ranges:")
print("  train:", train_df.index.min().date(), "->", train_df.index.max().date())
print("  val  :", val_df.index.min().date(), "->", val_df.index.max().date())
print("  test :", test_df.index.min().date(), "->", test_df.index.max().date())