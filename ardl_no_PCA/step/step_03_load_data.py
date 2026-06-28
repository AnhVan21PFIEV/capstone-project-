from __future__ import annotations

from .common import load_inputs


def run(context: dict) -> dict:
    data = load_inputs(context["PROJECT_ROOT"])
    context.update(data)

    print("ARDL step 3: data loaded (NO PCA)")
    print("  train_df:", context["train_df"].shape)
    print("  val_df  :", context["val_df"].shape)
    print("  test_df :", context["test_df"].shape)
    print("  Features:", len(context["feature_cols"]))
    return context