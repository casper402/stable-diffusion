import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re

# ────────────────────────────────────────────────────────────────────────────────
# (1)  SET A CONSISTENT, MINIMALIST STYLE
# ────────────────────────────────────────────────────────────────────────────────
try:
    plt.style.use("seaborn-v0_8-paper")
except OSError:
    plt.style.use("ggplot")

plt.rcParams["font.family"]     = "serif"
plt.rcParams["font.serif"]      = ["Nimbus Roman No9 L"]
plt.rcParams["font.size"]       = 12
plt.rcParams["axes.titlesize"]  = 14
plt.rcParams["axes.labelsize"]  = 13
plt.rcParams["legend.fontsize"] = 10
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10

# ────────────────────────────────────────────────────────────────────────────────
# (2)  PARSE THE DOSE PROFILE FILE (unchanged from before)
# ────────────────────────────────────────────────────────────────────────────────
profile_path = os.path.expanduser("~/thesis/DoseProfile.txt")

with open(profile_path, 'r') as f:
    lines = f.readlines()

data_start_idx = next(
    (i for i, line in enumerate(lines) if re.match(r'^\s*\d', line)),
    None
)
if data_start_idx is None:
    raise RuntimeError(f"No numeric lines found in {profile_path!r}.")

numeric_lines = []
for line in lines[data_start_idx:]:
    if line.strip() == "":
        break
    numeric_lines.append(line.strip())

numeric_text = "\n".join(numeric_lines)
df_prof = pd.read_csv(
    pd.io.common.StringIO(numeric_text),
    sep=r'\s+',
    header=None,
    names=["Position", "CT", "sCT"]
)
df_prof = df_prof.astype({"Position": float, "CT": float, "sCT": float})

pos      = df_prof["Position"].values
ct_prof  = df_prof["CT"].values
sct_prof = df_prof["sCT"].values

# If you have more profiles, simply extend this list:
profs = ["CT", "sCT"]

# ────────────────────────────────────────────────────────────────────────────────
# (3)  CREATE A SINGLE–AXIS DOSE PROFILE PLOT WITH YOUR SPECIFICATIONS
# ────────────────────────────────────────────────────────────────────────────────
fig, ax_prof = plt.subplots(figsize=(8, 6))

styles = {
    "CT":  {"color": "#1f77b4", "linestyle": "-",  "linewidth": 2.5},
    "sCT": {"color": "#d62728", "linestyle": "--", "linewidth": 2.5},
}

# Plot each profile
ax_prof.plot(pos, ct_prof,  label="CT",  **styles["CT"])
ax_prof.plot(pos, sct_prof, label="sCT", **styles["sCT"])

# X–axis cut at 10 cm:
ax_prof.set_xlim(0, 10)

# Y–axis goes from 0 to a little above max:
ymax = np.max(np.concatenate([ct_prof, sct_prof])) * 1.05
ax_prof.set_ylim(0, ymax)

# ────────────────────────────────────────────────────────────────────────────────
# (4)  FORMAT AXES, GRID, TICKS, AND LEGEND AS REQUESTED
# ────────────────────────────────────────────────────────────────────────────────
#   (a) Ticks: every 1 cm on x, every 0.5 on y
ax_prof.xaxis.set_major_locator(plt.MultipleLocator(1.0))
ax_prof.yaxis.set_major_locator(plt.MultipleLocator(25))

#   (b) Labels
ax_prof.set_xlabel("Position (cm)", fontsize=16)
ax_prof.set_ylabel("Dose (Gy)", fontsize=16)

#   (c) Grid (dashed) on both axes
ax_prof.grid(
    which='major',
    axis='both',
    linestyle='--',
    linewidth=0.8,
    alpha=0.7
)

#   (d) Remove top/right spines for a cleaner look
ax_prof.spines['top'].set_visible(False)
ax_prof.spines['right'].set_visible(False)

#   (e) Place legend above the plot, spanning all columns:
ax_prof.legend(
    loc="lower center",
    bbox_to_anchor=(0.5, 0.97),
    ncol=len(profs),
    frameon=False,
    fontsize=14
)

# ────────────────────────────────────────────────────────────────────────────────
# (5)  SAVE AND SHOW
# ────────────────────────────────────────────────────────────────────────────────
save_dir = os.path.expanduser("~/thesis/figures")
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, "dose_profile_single_axis.pdf")

fig.savefig(save_path, format="pdf", bbox_inches="tight", dpi=300)
print("Saved to:", save_path)

plt.tight_layout()
plt.show()
