"""
post/contours.py - Pressure and leakage-flux contours of the three gaps (with-leakage run) at the
snapshot angles of the last revolution (default 0, 90, 180, 270 deg). One PNG per gap and angle,
pressure on the left, leakage flux on the right.

  slipper 1   : polar land map, theta = 0 along the sliding direction xi; pocket drawn as a disc.
                Leakage = pressure-driven radial flux -h^3/(12 mu) dp/dr (+ outward); the sliding drag
                only circulates oil round the land.
  valve plate : polar map in the PDF x-y frame (discharge port y < 0); kidneys/ports left blank.
                Leakage = radial flux (+ outward).
  piston 1    : film unwrapped over the engagement length; leakage = full axial flux in the bore
                frame, + towards the case (the drag of the sliding piston is a real leakage here).
"""
import numpy as np
import matplotlib.pyplot as plt

from ..params import DEFAULT
from .style import BLUE_RAMP, DIVERGING, GRID, MUTED, colorbar

M2S = 1e6          # m^2/s -> mm^2/s


def _radial_qr(p, h, r, mu):
    return -(h**3) / (12.0 * mu) * np.gradient(p, r[1] - r[0], axis=0)


def _polar(ax, fig, TH, R, Z, cmap, label, vmin, vmax):
    pc = ax.pcolormesh(TH, R * 1e3, Z, cmap=cmap, shading="auto", vmin=vmin, vmax=vmax)
    ax.grid(color=GRID, lw=0.5)
    ax.tick_params(colors=MUTED, labelsize=7)
    ax.set_yticklabels([])
    colorbar(fig, pc, ax, label, shrink=0.75, pad=0.10)


def _sym(v):
    lim = float(np.nanpercentile(np.abs(v), 99.5)) if np.isfinite(v).any() else 1.0
    return lim or 1.0


def slipper(F, i, path, prm=DEFAULT):
    r, th = F["s1_r"], F["s1_th"]
    R, TH = np.meshgrid(r, th, indexing="ij")
    p = F["s1_p"][i] / 1e5
    pp = F["s1_p_pocket"][i] / 1e5
    vmin = min(float(p.min()), pp, 0.0); vmax = max(float(p.max()), pp, vmin + 1e-3)
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.9), subplot_kw=dict(projection="polar"))
    rp = np.linspace(0.0, r[0], 6)
    Rp, THp = np.meshgrid(rp, th, indexing="ij")
    axs[0].pcolormesh(THp, Rp * 1e3, np.full(Rp.shape, pp), cmap=BLUE_RAMP, shading="auto", vmin=vmin, vmax=vmax)
    _polar(axs[0], fig, TH, R, p, BLUE_RAMP, "Gauge pressure [bar]", vmin, vmax)
    axs[0].set_title(f"Pressure (pocket {pp:.1f} bar)\nθ = 0: sliding direction ξ", pad=22)
    q = _radial_qr(F["s1_p"][i].astype(float), F["s1_h"][i].astype(float), r, prm.mu) * M2S
    lim = _sym(q)
    _polar(axs[1], fig, TH, R, q, DIVERGING, "Radial leakage flux [mm²/s] (+ outward)", -lim, lim)
    axs[1].set_ylim(0, r[-1] * 1e3)
    axs[1].set_title(f"Leakage flux\nfilm {F['s1_h'][i].min() * 1e6:.2f}–{F['s1_h'][i].max() * 1e6:.2f} µm", pad=22)
    _finish(fig, F, i, "Slipper 1", path)


def valveplate(F, i, path, prm=DEFAULT):
    r, th = F["vp_r"], F["vp_th"]
    R, TH = np.meshgrid(r, th, indexing="ij")
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.9), subplot_kw=dict(projection="polar"))
    p = F["vp_p"][i] / 1e5
    _polar(axs[0], fig, TH, R, p, BLUE_RAMP, "Gauge pressure [bar]", min(float(p.min()), 0.0), None)
    axs[0].set_ylim(0, r[-1] * 1e3)
    axs[0].set_title("Pressure\ndischarge port y < 0, suction y > 0", pad=22)
    q = np.ma.masked_invalid(F["vp_qr"][i] * M2S)
    lim = _sym(q.filled(np.nan))
    _polar(axs[1], fig, TH, R, q, DIVERGING, "Radial leakage flux [mm²/s] (+ outward)", -lim, lim)
    axs[1].set_ylim(0, r[-1] * 1e3)
    axs[1].set_title(f"Leakage flux (film {F['vp_h0'][i] * 1e6:.2f} µm)\nopen kidneys/ports blank", pad=22)
    _finish(fig, F, i, "Valve plate", path)


def piston(F, i, path, prm=DEFAULT):
    z, th = F["k1_z"][i], F["k1_th"]
    TH, Z = np.meshgrid(np.degrees(th), z * 1e3)
    p, h, qz = F["k1_p"][i] / 1e5, F["k1_h"][i], F["k1_qz"][i]
    fig, axs = plt.subplots(1, 2, figsize=(11.0, 4.6), sharey=True)
    pc = axs[0].pcolormesh(TH, Z, p, cmap=BLUE_RAMP, shading="auto", vmin=min(float(p.min()), 0.0))
    colorbar(fig, pc, axs[0], "Gauge pressure [bar]", shrink=0.9, pad=0.02)
    axs[0].set_title("Pressure")
    leak = -qz * M2S
    lim = float(np.abs(leak).max()) or 1.0
    pc = axs[1].pcolormesh(TH, Z, leak, cmap=DIVERGING, shading="auto", vmin=-lim, vmax=lim)
    colorbar(fig, pc, axs[1], "Axial leakage flux [mm²/s] (+ towards the case)", shrink=0.9, pad=0.02)
    axs[1].set_title(f"Leakage flux (film {h.min() * 1e6:.2f}–{h.max() * 1e6:.2f} µm)")
    for ax in axs:
        ax.set_xlabel("θ from radial direction r [deg]")
        ax.set_xticks([0, 90, 180, 270, 360])
    axs[0].set_ylabel("ζ from the end face [mm]\n(0 = chamber end, bottom = bore exit)")
    _finish(fig, F, i, f"Piston 1 / bore (engagement {(z[-1] - z[0]) * 1e3:.1f} mm)", path)


def _finish(fig, F, i, name, path):
    ang = np.degrees(F["phi"][i]) % 360
    fig.suptitle(f"{name}, shaft angle φ = {ang:.0f}°", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def all_contours(F, outdir, prm=DEFAULT):
    outdir.mkdir(parents=True, exist_ok=True)
    files = []
    for i, phi in enumerate(F["phi"]):
        tag = f"{np.degrees(phi) % 360:03.0f}deg"
        for name, fn in (("slipper1", slipper), ("valveplate", valveplate), ("piston1", piston)):
            if name == "piston1" and not (np.asarray(F.get("k1_p", [])).size):
                continue
            f = outdir / f"{name}_{tag}.png"
            fn(F, i, f, prm)
            files.append(f)
    return files
