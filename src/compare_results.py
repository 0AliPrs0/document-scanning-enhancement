import os
import json
import matplotlib.pyplot as plt


def load_metrics(filepath):
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            return json.load(f)
    return None


def plot_dropout_comparison():
    no_drop = load_metrics("output/metrics_enhancement_nodrop.json")
    with_drop = load_metrics("output/metrics_enhancement_dropout.json")

    if not no_drop or not with_drop:
        return

    labels = ["Train PSNR", "Val PSNR", "Test PSNR"]

    nodrop_vals = [
        no_drop["train"]["psnr"],
        no_drop["val"]["psnr"],
        no_drop["test"]["psnr"],
    ]

    drop_vals = [
        with_drop["train"]["psnr"],
        with_drop["val"]["psnr"],
        with_drop["test"]["psnr"],
    ]

    x = range(len(labels))
    width = 0.35

    fig, ax = plt.subplots()

    ax.bar(
        [i - width / 2 for i in x],
        nodrop_vals,
        width,
        label="Without Dropout",
    )

    ax.bar(
        [i + width / 2 for i in x],
        drop_vals,
        width,
        label="With Dropout",
    )

    ax.set_ylabel("PSNR (Higher is better)")
    ax.set_title("Impact of Dropout on Overfitting (Enhancement)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    plt.savefig("output/dropout_comparison.png")
    plt.show()


def plot_corner_comparison():
    reg = load_metrics("output/metrics_reg_dropout.json")
    heat = load_metrics("output/metrics_heatmap_dropout.json")

    if not reg or not heat:
        return

    labels = ["Train Error", "Val Error", "Test Error"]

    reg_vals = [
        reg["train_error_px"],
        reg["val_error_px"],
        reg["test_error_px"],
    ]

    heat_vals = [
        heat["train_error_px"],
        heat["val_error_px"],
        heat["test_error_px"],
    ]

    x = range(len(labels))
    width = 0.35

    fig, ax = plt.subplots()

    ax.bar(
        [i - width / 2 for i in x],
        reg_vals,
        width,
        label="Approach A (Regression)",
    )

    ax.bar(
        [i + width / 2 for i in x],
        heat_vals,
        width,
        label="Approach B (Heatmap)",
    )

    ax.set_ylabel("Mean Error in Pixels (Lower is better)")
    ax.set_title("Corner Detection Methods Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    plt.savefig("output/corner_comparison.png")
    plt.show()

    print("\n" + "=" * 65)
    print(
        f"{'Split':<15} | "
        f"{'App. A (Regression)':<20} | "
        f"{'App. B (Heatmap)':<20}"
    )
    print("-" * 65)
    print(f"{'Training':<15} | " f"{reg_vals[0]:<20.2f} | " f"{heat_vals[0]:<20.2f}")
    print(f"{'Validation':<15} | " f"{reg_vals[1]:<20.2f} | " f"{heat_vals[1]:<20.2f}")
    print(f"{'Test':<15} | " f"{reg_vals[2]:<20.2f} | " f"{heat_vals[2]:<20.2f}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    plot_dropout_comparison()
    plot_corner_comparison()
