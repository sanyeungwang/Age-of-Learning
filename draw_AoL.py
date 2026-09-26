import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# ======================
# 1. Font: Times New Roman
# ======================
font_path = "path/times.ttf"
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)

plt.rcParams.update({
    "font.family": "Times New Roman",

    # Preserve $\checkmark$ and $\times$, and avoid cursive warnings.
    "font.cursive": ["Times New Roman"],

    "mathtext.fontset": "custom",
    "mathtext.rm": "Times New Roman",
    "mathtext.it": "Times New Roman:italic",
    "mathtext.bf": "Times New Roman:bold",
    "mathtext.cal": "Times New Roman",

    "axes.unicode_minus": False,
})

# ======================
# 2. Data
# Wrong Wrong Wrong Correct Wrong Wrong Correct Correct Wrong Correct Correct
# 0 = wrong, 1 = correct
# ======================
correct = np.array([0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1])
epochs = np.arange(1, len(correct) + 1)  # Epoch 1 ~ 12

# AoL: t0 ~ t12
age = 1
aol = [1]

for c in correct:
    if c == 1:
        age = 1
    else:
        age += 1
    aol.append(age)

aol = np.array(aol)
t = np.arange(0, len(aol))  # t0 ~ t12

# ======================
# 3. Figure and aligned axes
# ======================
fig = plt.figure(figsize=(14, 8.5))

left = 0.075
width = 0.88

x_min = -0.55
x_max = len(correct) + 0.95

y_min = -0.85
y_max = max(aol) + 1.00

x_range = x_max - x_min
y_range = y_max - y_min

fig_w, fig_h = fig.get_size_inches()

ax2_height = width * (fig_w / fig_h) * (y_range / x_range)
ax2_bottom = 0.145

ax1_height = 0.16
ax1_bottom = ax2_bottom + ax2_height + 0.055

ax1 = fig.add_axes([left, ax1_bottom, width, ax1_height])
ax2 = fig.add_axes([left, ax2_bottom, width, ax2_height])

# ======================
# 4. Upper figure
# ======================
ax1.set_xlim(x_min, x_max)
ax1.set_ylim(-0.30, 0.62)

main_fontsize = 24
label_right_x = -0.06

# Epoch row
for x in epochs:
    ax1.text(
        x, 0.38, str(x),
        ha="center", va="center",
        fontsize=main_fontsize,
        fontfamily="Times New Roman"
    )

ax1.text(
    label_right_x, 0.38, "Epoch",
    ha="right", va="center",
    fontsize=main_fontsize,
    fontweight="bold",
    fontfamily="Times New Roman"
)

# Prediction row
for x, c in zip(epochs, correct):
    if c == 1:
        symbol = r"$\checkmark$"
    else:
        symbol = r"$\times$"

    ax1.text(
        x, 0.03, symbol,
        ha="center", va="center",
        fontsize=main_fontsize + 5,
        color="black",
        fontfamily="Times New Roman"
    )

ax1.text(
    label_right_x, 0.03, "Prediction",
    ha="right", va="center",
    fontsize=main_fontsize,
    fontweight="bold",
    fontfamily="Times New Roman"
)

ax1.set_xticks([])
ax1.set_yticks([])

for spine in ax1.spines.values():
    spine.set_visible(False)

# ======================
# 5. Lower figure
# ======================
ax2.set_xlim(x_min, x_max)
ax2.set_ylim(y_min, y_max)

ax2.set_aspect("equal", adjustable="box")

ax2.set_xticks([])
ax2.set_yticks([])

for spine in ax2.spines.values():
    spine.set_visible(False)

# ======================
# 6. Axes arrows
# ======================
axis_lw = 1.25
arrow_ms = 17

ax2.annotate(
    "",
    xy=(len(correct) + 0.72, 0),
    xytext=(0, 0),
    arrowprops=dict(
        arrowstyle="-|>",
        lw=axis_lw,
        color="black",
        mutation_scale=arrow_ms,
        shrinkA=0,
        shrinkB=0,
        joinstyle="miter"
    ),
    zorder=2
)

ax2.annotate(
    "",
    xy=(0, y_max - 0.10),
    xytext=(0, 0),
    arrowprops=dict(
        arrowstyle="-|>",
        lw=axis_lw,
        color="black",
        mutation_scale=arrow_ms,
        shrinkA=0,
        shrinkB=0,
        joinstyle="miter"
    ),
    zorder=2
)

# ======================
# 7. Ticks on axes
# ======================
tick_lw = 1.15
tick_len = 0.08

# x-axis ticks: t0 ~ t12
for x in t:
    ax2.vlines(
        x,
        ymin=-tick_len,
        ymax=tick_len,
        colors="black",
        linewidth=tick_lw,
        zorder=4
    )

# y-axis ticks: 1, 2, 3, 4
for y in [1, 2, 3, 4]:
    ax2.hlines(
        y,
        xmin=-tick_len,
        xmax=tick_len,
        colors="black",
        linewidth=tick_lw,
        zorder=4
    )

# ======================
# 8. Dotted guide lines
# ======================
dot_style = (0, (1, 4.6))
guide_lw = 1.10

# horizontal dotted lines: y = 1, 2, 3, 4
for y in [1, 2, 3, 4]:
    ax2.hlines(
        y,
        xmin=0,
        xmax=len(correct) + 0.48,
        colors="black",
        linestyles=dot_style,
        linewidth=guide_lw,
        zorder=1
    )

# vertical dotted lines: t1 ~ t12
for x in t[1:]:
    ax2.vlines(
        x,
        ymin=0,
        ymax=y_max - 0.20,
        colors="black",
        linestyles=dot_style,
        linewidth=guide_lw,
        zorder=1
    )

# ======================
# 9. AoL step curve
# ======================
line_lw = 3.35

ax2.step(
    t,
    aol,
    where="post",
    linewidth=line_lw,
    color="black",
    solid_capstyle="butt",
    solid_joinstyle="miter",
    zorder=3
)

dot_size = 56

ax2.scatter(
    t,
    aol,
    s=dot_size,
    color="black",
    zorder=5,
    clip_on=False
)

# ======================
# 10. Axis tick labels
# ======================
coord_fontsize = main_fontsize

for x in t:
    ax2.text(
        x, -0.22,
        rf"$\mathrm{{t}}_{{{x}}}$",
        ha="center", va="top",
        fontsize=coord_fontsize,
        fontfamily="Times New Roman"
    )

ax2.text(
    len(correct) + 0.88,
    -0.22,
    r"$\mathrm{t}$",
    ha="left", va="top",
    fontsize=coord_fontsize,
    fontfamily="Times New Roman"
)

for y in [1, 2, 3, 4]:
    ax2.text(
        -0.18, y,
        str(y),
        ha="right", va="center",
        fontsize=coord_fontsize,
        fontfamily="Times New Roman"
    )

ax2.text(
    label_right_x,
    y_max - 0.08,
    "AoL",
    ha="right", va="bottom",
    fontsize=coord_fontsize,
    fontfamily="Times New Roman"
)

# ======================
# 11. Save
# ======================
plt.savefig("draw_AoL.pdf", dpi=600, bbox_inches="tight")
plt.savefig("draw_AoL.png", dpi=600, bbox_inches="tight")
plt.show()
