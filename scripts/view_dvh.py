import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator
import os

# ────────────────────────────────────────────────────────────────────────────────
#   (1) ─── SET A CONSISTENT, MINIMALIST STYLE ─────────────────────────────────
# ────────────────────────────────────────────────────────────────────────────────
try:
    plt.style.use("seaborn-v0_8-paper")
except OSError:
    plt.style.use("ggplot")

plt.rcParams["font.family"]    = "serif"
plt.rcParams["font.serif"]     = ["Nimbus Roman No9 L"]
plt.rcParams["font.size"]      = 12
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.labelsize"] = 13
plt.rcParams["legend.fontsize"]= 10
plt.rcParams["xtick.labelsize"]= 10
plt.rcParams["ytick.labelsize"]= 10

# ────────────────────────────────────────────────────────────────────────────────
#   (2) ─── PARSE THE DVH FILE FOR BOTH PLANS ────────────────────────────────────
# ────────────────────────────────────────────────────────────────────────────────
dvh_path = os.path.expanduser("~/thesis/DVH.txt")

with open(dvh_path, 'r') as f:
    lines = f.readlines()

dvh_data = {}  # store {plan_name: (dose_array, vol_array)}

i = 0
num_lines = len(lines)
while i < num_lines:
    line = lines[i].strip()
    if line.startswith("Structure:") and "GTV" in line:
        plan_name = None
        j = i + 1
        while j < num_lines:
            next_line = lines[j].strip()
            if next_line.startswith("Plan:"):
                plan_name = next_line.split("Plan:")[1].strip()
                break
            j += 1
        # Advance until data header
        while j < num_lines and not lines[j].strip().startswith("Relative dose"):
            j += 1
        j += 1
        dose_vals = []
        vol_vals = []
        while j < num_lines:
            data_line = lines[j].strip()
            if data_line == "" or data_line.startswith("Structure:"):
                break
            parts = data_line.split()
            if len(parts) >= 3:
                try:
                    dose_vals.append(float(parts[1]))
                    vol_vals.append(float(parts[2]))
                except ValueError:
                    pass
            j += 1
        if plan_name:
            dvh_data[plan_name] = (np.array(dose_vals), np.array(vol_vals))
        i = j
    else:
        i += 1

# ────────────────────────────────────────────────────────────────────────────────
#   (3) ─── PREPARE BROKEN AXIS PLOT WITH ADJUSTED RATIOS ─────────────────────────
# ────────────────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(
    1, 2, sharey=True, figsize=(10, 6),
    gridspec_kw={'width_ratios': [1, 4], 'wspace': 0.01}
)

# Set x-axis limits for left and right panels
ax1.set_xlim(0, 2)
ax2.set_xlim(2, 4)

# Plot each plan's DVH on both axes, masked by x-range
styles = {
    "Plan1":       {"color": "#1f77b4", "linestyle": "-",  "linewidth": 3.0},
    "Plan1_sCT":   {"color": "#d62728", "linestyle": "--", "linewidth": 3.0},
}

names = {
    "Plan1": "CT",
    "Plan1_sCT": "sCT",
}

for plan, (dose_vals, vol_vals) in dvh_data.items():
    style = styles.get(plan, {"color": "k", "linestyle": "-", "linewidth": 2.0})
    mask_left = dose_vals <= 2
    ax1.plot(dose_vals[mask_left], vol_vals[mask_left], label=names.get(plan), **style)
    mask_right = dose_vals >= 2
    ax2.plot(dose_vals[mask_right], vol_vals[mask_right], **style)

# Remove inner spines
ax1.spines["right"].set_visible(False)
ax2.spines["left"].set_visible(False)

# Configure y-axis ticks and labels on left only
ax1.yaxis.set_ticks_position('left')
ax1.set_yticks(np.arange(0, 101, 10))
ax1.set_ylim(0, 100)
ax1.yaxis.set_major_locator(MultipleLocator(10))
ax1.tick_params(axis='y', labelleft=True, length=5)

# For right axis: no tick marks or labels, but set positions for grid
ax2.yaxis.set_ticks_position('none')
ax2.set_yticks(np.arange(0, 101, 10))
# ax2.set_yticklabels([])
ax1.yaxis.set_major_locator(MultipleLocator(10))
ax1.set_ylim(0, 100)

# Add diagonal lines to indicate axis break
d = .015
kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)
ax1.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
kwargs.update(transform=ax2.transAxes)
ax2.plot((-d, +d), (-d, +d), **kwargs)
ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)

# Set common y-label
ax1.set_ylabel("Ratio of Total Structure Volume (%)", fontsize=16)

# Configure x-axis ticks
ax1.set_xticks([0, 1])
ax2.set_xticks(np.arange(2, 4.001, 0.2))

# Add grid lines (horizontal on both panels)
ax1.xaxis.grid(True, linestyle="--", linewidth=0.8, alpha=0.7)
ax1.yaxis.grid(True, linestyle="--", linewidth=0.8, alpha=0.7)
ax2.xaxis.grid(True, linestyle="--", linewidth=0.8, alpha=0.7)
ax2.yaxis.grid(True, linestyle="--", linewidth=0.8, alpha=0.7)

for ax in (ax1, ax2):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# Single x-axis label centered below both subplots
fig.supxlabel("Dose (Gy)", fontsize=16, y=0.05)

# Legend centered above both axes
handles, labels = ax1.get_legend_handles_labels()
fig.legend(
    handles, labels,
    loc="lower center",
    bbox_to_anchor=(0.5, 0.88),
    ncol=len(handles),
    frameon=False,
    fontsize=14
)

# Adjust layout
plt.subplots_adjust(top=0.88, bottom=0.15)

fig.tight_layout()

save_path = os.path.expanduser(f"~/thesis/figures/dvh.pdf")
fig.savefig(save_path, format="pdf", bbox_inches="tight", dpi=300)
print("Saved to:", save_path)

plt.show()
