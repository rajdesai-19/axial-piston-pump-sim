"""post/style.py - Shared plot style (light mode, print): categorical slots 1-3 only, other series in
muted grey, one y-axis per panel, recessive grid, blue sequential and blue-red diverging maps."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
CONTEXT = "#c3c2b7"
BLUE_RAMP = LinearSegmentedColormap.from_list(
    "blue_seq", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
DIVERGING = LinearSegmentedColormap.from_list(
    "blue_red_div", ["#104281", "#2a78d6", "#86b6ef", "#f0efec", "#f2a3a2", "#e34948", "#9e2b2a"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "sans-serif", "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
    "axes.titlecolor": INK, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": AXIS, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.6,
    "legend.frameon": True, "legend.facecolor": SURFACE, "legend.edgecolor": SURFACE,
    "legend.framealpha": 0.92, "legend.fontsize": 8, "legend.labelcolor": INK2,
})


def colorbar(fig, pc, ax, label, **kw):
    cb = fig.colorbar(pc, ax=ax, **kw)
    cb.set_label(label, color=INK2)
    cb.outline.set_edgecolor(AXIS)
    cb.ax.tick_params(colors=MUTED, labelsize=7)
    return cb


def angle_axis(ax, x_max=1080.0):
    ax.set_xlim(0, x_max)
    ax.set_xticks(np.arange(0, x_max + 1, 90 if x_max <= 360 else 180))
    for r in np.arange(360, x_max, 360):
        ax.axvline(r, color=AXIS, lw=0.8, ls=(0, (4, 3)), zorder=0)
    ax.set_xlabel("Shaft angle φ since start [deg]  (dashed: revolution boundaries)")
