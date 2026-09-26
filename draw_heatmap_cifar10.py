#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.colors as mcolors

# =========================
# Font / style
# =========================
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

font_path = "path/times.ttf"
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)

plt.rcParams.update({
    "font.family": "Times New Roman",
})

scale = 2.0
plt.rcParams.update({
    "font.size": 10 * scale,
    "axes.labelsize": 10 * scale,
    "axes.titlesize": 12 * scale,
    "xtick.labelsize": 9 * scale,
    "ytick.labelsize": 9 * scale,
    "legend.fontsize": 9 * scale,
    "figure.titlesize": 12 * scale,
})


def read_age_csv(csv_path):
    df = pd.read_csv(csv_path)

    if "class_idx" not in df.columns:
        raise ValueError(f"{csv_path} Missing class_idx column")

    epoch_cols = [c for c in df.columns if re.fullmatch(r"epoch\d+", c)]
    epoch_cols = sorted(epoch_cols, key=lambda x: int(x.replace("epoch", "")))

    if len(epoch_cols) == 0:
        raise ValueError(f"{csv_path} No column for epoch001 was found.")

    class_idx = df["class_idx"].to_numpy(dtype=int)
    values = df[epoch_cols].to_numpy(dtype=float)

    return class_idx, epoch_cols, values


def plot_heatmap(csv_path, out_dir):
    class_idx, epoch_cols, values = read_age_csv(csv_path)

    values_plot = values

    c = len(class_idx)
    t = len(epoch_cols)

    fig, ax = plt.subplots(figsize=(16.5, 8.0))

    norm = mcolors.PowerNorm(gamma=0.35, vmin=1, vmax=100)

    im = ax.imshow(
        values_plot,
        aspect="auto",
        interpolation="nearest",
        origin="upper",
        cmap="viridis",
        norm=norm,
        extent=[0, t, c, 0],
    )

    base_name = os.path.splitext(os.path.basename(csv_path))[0]

    title_pad = 18
    label_pad = 18

    ax.set_title("Per-class Mean AoL Across Training Epochs on CIFAR-10 (IF = 200)", pad=title_pad)
    ax.set_xlabel("Epoch Index (Column)", labelpad=label_pad)
    ax.set_ylabel("Class Index (Row)", labelpad=label_pad)

    # x-axis: 0, 20, 40, ..., 200
    xticks = np.arange(0, t + 1, 20)
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(x) for x in xticks])

    # y-axis: 0, 1, 2, ..., 9
    yticks = np.arange(0, c, 1)
    ax.set_yticks(yticks)
    ax.set_yticklabels([str(y) for y in yticks])

    ax.tick_params(axis="both", which="major", length=4, width=1)

    cbar = fig.colorbar(im, ax=ax, pad=0.04)
    cbar.set_label("AoL", labelpad=label_pad)
    cbar.set_ticks([1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])

    # Head / Tail
    ax.annotate(
        "Head",
        xy=(-0.0645, 0.95),
        xytext=(-0.0645, 0.82),
        xycoords=ax.transAxes,
        textcoords=ax.transAxes,
        ha="center",
        va="center",
        rotation=90,
        fontsize=10 * scale,
        arrowprops=dict(arrowstyle="-|>", lw=1.0, color="black"),
        annotation_clip=False,
    )

    ax.annotate(
        "Tail",
        xy=(-0.0645, 0.05),
        xytext=(-0.0645, 0.18),
        xycoords=ax.transAxes,
        textcoords=ax.transAxes,
        ha="center",
        va="center",
        rotation=90,
        fontsize=10 * scale,
        arrowprops=dict(arrowstyle="-|>", lw=1.0, color="black"),
        annotation_clip=False,
    )

    plt.subplots_adjust(left=0.22, right=0.93, top=0.90, bottom=0.12)

    out_pdf = os.path.join(out_dir, base_name + "_aol_heatmap.pdf")
    out_png = os.path.join(out_dir, base_name + "_aol_heatmap.png")

    plt.savefig(out_pdf, bbox_inches="tight")
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[OK] saved heatmap: {out_pdf}")
    print(f"[OK] saved heatmap: {out_png}")


def main():
    csv_path = "file_name.csv"
    out_dir = "path"

    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"file not found: {csv_path}")

    print(f"\n[INFO] processing: {csv_path}")
    plot_heatmap(csv_path, out_dir)


if __name__ == "__main__":
    main()
