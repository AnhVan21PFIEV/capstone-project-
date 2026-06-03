from __future__ import annotations


def run(context: dict) -> dict:
    pc_cols = context["pc_cols"]
    train_df = context["train_df"]

    is_pca_applied = bool(pc_cols) and all(train_df[pc_cols].dtypes.apply(lambda t: t.kind in "if"))
    can_use_ardl = len(pc_cols) == 11 and len(train_df) > 30

    print("ARDL step 4: PCA check")
    print("  Is PCA applied?", is_pca_applied)
    print("  Number of detected principal components k =", len(pc_cols))
    print("  Can ARDL be applied with 11 PCs?", can_use_ardl)

    if not can_use_ardl:
        raise ValueError("ARDL prerequisites not satisfied. Check k or sample size.")

    context["is_pca_applied"] = is_pca_applied
    context["can_use_ardl"] = can_use_ardl
    return context
