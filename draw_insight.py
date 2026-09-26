#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import os
import csv
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

font_path = "path/times.ttf"
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
plt.rcParams.update({
    "font.family": "Times New Roman",
})

scale = 1.6
plt.rcParams.update({
    "font.size": 10 * scale,  # default font
    "axes.labelsize": 10 * scale,  # x/y label
    "axes.titlesize": 12 * scale,  # title
    "xtick.labelsize": 9 * scale,  # x tick
    "ytick.labelsize": 9 * scale,  # y tick
    "legend.fontsize": 9 * scale,  # legend
    "figure.titlesize": 12 * scale,  # figure title
})


def read_csv_dict(csv_path):
    """Read CSV -> dict(str->np.array); assuming the first row is the header."""
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        cols = {h: [] for h in header}
        for row in reader:
            if not row:
                continue
            for h, v in zip(header, row):
                cols[h].append(v)

    # Convert to NumPy (use int for class_idx / count; use float for the rest)
    out = {}
    for k, arr in cols.items():
        if k in ["class_idx", "train_class_count", "test_class_count"]:
            out[k] = np.array(arr, dtype=np.int64)
        else:
            # Allow null values/NaN
            out[k] = np.array([float(x) if x != "" else np.nan for x in arr], dtype=np.float64)
    return out


def main():
    # ========== 1) Configuration ==========
    csv_path = "file_name.csv"
    out_pdf = os.path.splitext(csv_path)[0] + "_lineplot.pdf"

    left_cols = [
        "train_class_count",
        "test_class_count",
    ]
    right_cols = [
        "avg_age",
        # "avg_age_test",
        # "per_class_acc_best",
        "per_class_acc_last",
    ]

    # ========== 2) Read data ==========
    data = read_csv_dict(csv_path)

    if "class_idx" not in data:
        raise ValueError("The CSV file is missing the class_idx column.")

    x = data["class_idx"]

    # ========== 3) Drawing ==========
    fig, ax_left = plt.subplots(figsize=(10, 5))
    ax_right = ax_left.twinx()

    # Class-Balanced Loss Based on Effective Number of Samples
    color_map = {
        "train_class_count": "#1F77B4",
        "test_class_count": "#FF7E10",
        "avg_age": "#D72728",
        "avg_age_test": "#CCCCCC",
        "per_class_acc_best": "#9467BD",
        "per_class_acc_last": "#2DA02C",
    }

    label_map = {
        "train_class_count": "Training Images",
        "test_class_count": "Test Images",
        "avg_age": "AoL (Training Images)",
        "avg_age_test": "AoL (Test)",
        "per_class_acc_best": "Accuracy (Best)",
        "per_class_acc_last": "Accuracy (Test)",
    }

    # Left axis：count
    for col in left_cols:
        if col not in data:
            print(f"[WARN] skip missing column: {col}")
            continue
        ax_left.plot(
            x, data[col],
            linestyle="-",
            linewidth=1.6,
            color=color_map[col],
            label=label_map.get(col, col)
        )

    ax_left.set_xlabel("Class Index (Aligned with Training Epochs)")
    ax_left.set_ylabel("Number of Images per Class")

    ymax = 500
    pad = 0.05 * ymax  # 5% padding
    ax_left.set_ylim(-pad, ymax + pad)
    ax_left.set_yticks([0, 100, 200, 300, 400, 500])

    ax_left.grid(True, linestyle="--", alpha=0.35)

    # Right axis：Age / (accuracy)
    for col in right_cols:
        if col not in data:
            print(f"[WARN] skip missing column: {col}")
            continue
        ax_right.plot(
            x, data[col],
            linestyle="-",
            linewidth=1.6,
            color=color_map[col],
            label=label_map.get(col, col)
        )

    ax_right.set_ylabel("Accuracy (0–1) and Class AoL (1–100)")

    # --- Align y=0 tick position between left and right axes ---
    lmin, lmax = ax_left.get_ylim()  # e.g., (-25, 525)
    p = (-lmin) / (lmax - lmin)  # 0-position fraction from bottom

    y_top = 13.8  # keep your right-axis top
    y_bottom = -(p * y_top) / (1.0 - p)  # derived bottom so that 0 aligns

    ax_right.set_ylim(y_bottom, y_top)
    ax_right.set_yticks(np.arange(0, 13, 2))

    # merge legend
    lines_l, labels_l = ax_left.get_legend_handles_labels()
    lines_r, labels_r = ax_right.get_legend_handles_labels()
    # ax_left.legend(
    #     lines_l + lines_r,
    #     labels_l + labels_r,
    #     loc="upper center",
    #     bbox_to_anchor=(0.5, 1.22),
    #     ncol=3,
    #     frameon=False
    # )
    ax_left.legend(
        lines_l + lines_r,
        labels_l + labels_r,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.1),
        ncol=len(lines_l + lines_r),
        frameon=False,
        columnspacing=1.2,
        handlelength=1.8,
        handletextpad=0.4,
        borderaxespad=0.0
    )

    # plt.title("Per-Class Frequency (Left) and Class Age/Accuracy (Right) on CIFAR-100 with ResNet-34")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_pdf, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved: {out_pdf}")


if __name__ == "__main__":
    main()
