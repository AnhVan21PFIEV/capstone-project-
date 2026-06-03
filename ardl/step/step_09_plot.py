from __future__ import annotations

import matplotlib.pyplot as plt


def run(context: dict) -> dict:
    y_test = context["y_test"]
    pred_test = context["pred_test"]
    selected_pair = context["SELECTED_PAIR"]

    plt.figure(figsize=(12, 5))
    plt.plot(y_test.index, y_test.values, label="Actual VNINDEX", linewidth=2)
    plt.plot(pred_test.index, pred_test.values, label=f"ARDL Forecast {selected_pair}", linewidth=2)
    plt.title(f"VNINDEX Forecast on Test Set (ARDL + PCA, pair={selected_pair})")
    plt.xlabel("Date")
    plt.ylabel("VNINDEX")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    print("ARDL step 9: plot shown")
    return context
