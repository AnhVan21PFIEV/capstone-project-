from __future__ import annotations

from .common import load_inputs


def run(context: dict) -> dict:
    data = load_inputs(context["PROJECT_ROOT"])
    context.update(data)

    print("ARDL step 3: data loaded")
    print("  train_df:", context["train_df"].shape)
    print("  val_df  :", context["val_df"].shape)
    print("  test_df :", context["test_df"].shape)
    print("  PCs     :", len(context["pc_cols"]))
    return context
