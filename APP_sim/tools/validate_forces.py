"""
validate_forces.py - Phase 2: forces with the steady chamber pressures from Phase 1,
compared with the old macro_forces/all_forces_combined.py results.

Run:  py -m APP_sim.tools.validate_forces      (needs APP_sim/outputs/tools/chamber_cycles_1-5_full.npz from validate_chamber)
"""
from pathlib import Path
import numpy as np

from ..params import DEFAULT
from ..frame import SwashFrame
from ..forces import piston_forces, block_forces, swashplate_forces

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "tools"


def geometry_check(prm=DEFAULT):
    fr = SwashFrame(prm)
    zb_bdc, zb_tdc = fr.z_bj(fr.phi_bdc), fr.z_bj(fr.phi_tdc)
    lines = [
        f"r_shoe {prm.r_shoe * 1e3:.3f} mm, l_K {prm.piston_l_K * 1e3:.4f} mm, l_S1 {prm.piston_l_S1 * 1e3:.4f} mm, "
        f"l_SG {prm.slipper_l_SG * 1e3:.4f} mm, m_piston {prm.m_piston * 1e3:.2f} g, m_slipper {prm.m_slipper * 1e3:.2f} g",
        f"ball joint z: BDC {zb_bdc * 1e3:.2f} mm, TDC {zb_tdc * 1e3:.2f} mm (simulation frame)",
        f"guide length l_F: BDC {fr.engagement(fr.phi_bdc) * 1e3:.2f} mm, TDC {fr.engagement(fr.phi_tdc) * 1e3:.2f} mm "
        f"(flat swash plate: {prm.engagement_flat * 1e3:.4f} mm)",
        f"piston chamber-end face at TDC: z = {(zb_tdc + prm.piston_l_K) * 1e3:.2f} mm, valve plate face {prm.z_VP * 1e3:.2f} mm",
        f"block front {prm.z_block_front * 1e3:.2f} mm, bearing centre {prm.z_bearing * 1e3:.2f} mm, block length {prm.L_block * 1e3:.2f} mm",
    ]
    warn = []
    if zb_tdc + prm.piston_l_K > prm.z_VP:
        warn.append("piston end face at TDC lies beyond the valve plate face")
    if fr.engagement(fr.phi_bdc) <= 0:
        warn.append("piston leaves the bore at BDC")
    return lines, warn


def main():
    prm = DEFAULT
    d = np.load(OUT / "chamber_cycles_1-5_full.npz")
    steps = 360 * prm.steps_per_deg
    sl = slice(4 * steps, 5 * steps)                    # cycle 5 (steady, Pd drift ~0)
    every = prm.steps_per_deg                           # 1 deg resolution is plenty here
    phi = np.mod(d["phi_abs"][sl][::every], 2 * np.pi)
    p = d["p"][:, sl][:, ::every]

    pf = piston_forces(phi, p, prm)
    bf = block_forces(pf, p, prm)
    sf = swashplate_forces(pf, prm)
    Vg = prm.num_piston * prm.Ak * SwashFrame(prm).stroke
    dp_mean = d["p_d"][sl].mean() - prm.p_case

    new = {
        "Block clamping F_DB+F_TB [N]": (bf["F_DB_tot"] + bf["F_TB_tot"]).mean(),
        "Block clamping incl. spring F_Bz [N]": bf["F_Bz"].mean(),
        "Radial load |F_RB| = bearing [N]": bf["F_bearing"].mean(),
        "F_Bx mean [N]": bf["F_Bx"].mean(),
        "F_By mean [N]": bf["F_By"].mean(),
        "Drive torque M_z [N.m]": bf["M_drive"].mean(),
        "Tilting moment @VP, no bearing [N.m]": bf["M_kipp"].mean(),
        "Tilting moment @VP, with bearing [N.m]": bf["M_kipp_brg"].mean(),
        "Centre of pressure x_B [mm]": bf["x_B"].mean() * 1e3,
        "Centre of pressure y_B [mm]": bf["y_B"].mean() * 1e3,
        "Swash plate normal force F_S [N]": sf["F_S_total"].mean(),
        "Cradle torque T [N.m]": sf["T_cradle"].mean(),
    }
    old_keys = {
        "Block clamping F_DB+F_TB [N]": ("F_z_net", 1.0),
        "Radial load |F_RB| = bearing [N]": ("F_bearing_mag", 1.0),
        "F_Bx mean [N]": ("F_Bx", 1.0),
        "F_By mean [N]": ("F_By", 1.0),
        "Drive torque M_z [N.m]": ("M_z_drive", 1.0),
        "Tilting moment @VP, no bearing [N.m]": ("M_kipp_no_brg", 1.0),
        "Tilting moment @VP, with bearing [N.m]": ("M_kipp_net_VP", 1.0),
        "Centre of pressure x_B [mm]": ("x_B", 1e3),
        "Centre of pressure y_B [mm]": ("y_B", 1e3),
        "Swash plate normal force F_S [N]": ("F_S_total", 1.0),
        "Cradle torque T [N.m]": ("T_cradle", 1.0),
    }
    old_npz = ROOT / "outputs" / "all_forces_combined.npz"
    old = np.load(old_npz) if old_npz.exists() else None

    print("Geometry check (placeholders marked TO VERIFY in params.py):")
    lines, warn = geometry_check(prm)
    for l in lines:
        print("   " + l)
    for w in warn:
        print("   WARNING: " + w)

    print(f"\nPhase-1 steady pressures, cycle 5: mean Pd = {d['p_d'][sl].mean() / 1e5:.2f} bar")
    print(f"Ideal drive torque V_g*dp/2pi = {Vg * dp_mean / (2 * np.pi):.2f} N.m "
          f"(V_g = {Vg * 1e6:.2f} cc, dp = Pd - p_case)\n")
    print(f"{'quantity':42s} {'new (simulation frame)':>16s} {'old code':>12s}")
    for k, v in new.items():
        o = ""
        if old is not None and k in old_keys:
            key, s = old_keys[k]
            if key in old.files:
                o = f"{float(np.mean(old[key])) * s:12.2f}"
        print(f"{k:42s} {v:16.2f} {o}")

    bk = block_forces(pf, p, prm, centrifugal="book")
    print(f"\nBook-literal variant (F'_wK summed into F_RB): |F_RB| {bk['F_bearing'].mean():.2f} N, "
          f"F_Bx {bk['F_Bx'].mean():.2f} N, tilting moment w/o bearing {bk['M_kipp'].mean():.2f} N.m")
    print("\nCradle torque split: pressure {:.2f} N.m, inertia {:.2f} N.m (mean)".format(
        sf["T_cradle_pressure"].mean(), sf["T_cradle_inertia"].mean()))
    np.savez(OUT / "forces_cycle5.npz", phi=phi,
             **{f"piston_{k}": v for k, v in pf.items() if isinstance(v, np.ndarray)},
             **{f"block_{k}": v for k, v in bf.items() if isinstance(v, np.ndarray)},
             **{f"swash_{k}": v for k, v in sf.items() if isinstance(v, np.ndarray)})
    print(f"\nSaved {OUT / 'forces_cycle5.npz'}")


if __name__ == "__main__":
    main()
