from __future__ import annotations


def run(context: dict) -> dict:
    feature_cols = context["feature_cols"]
    train_df = context["train_df"]

    # Kiểm tra dữ liệu đã được scale chưa
    is_scaled = all(train_df[feature_cols].dtypes.apply(lambda t: t.kind in "if"))
    can_use_ardl = len(feature_cols) > 1 and len(train_df) > 30

    print("ARDL step 4: Data validation (NO PCA)")
    print("  Is data scaled?", is_scaled)
    print("  Number of features:", len(feature_cols))
    print("  Can ARDL be applied?", can_use_ardl)

    if not can_use_ardl:
        raise ValueError("ARDL prerequisites not satisfied. Check features or sample size.")

    context["is_data_scaled"] = is_scaled
    context["can_use_ardl"] = can_use_ardl
    return context