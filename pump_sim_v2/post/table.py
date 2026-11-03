"""
post/table.py - The version-2 results table: one row every 0.5 deg of shaft rotation
(720 points per revolution), revolutions 1-3 from a cold start.

Every column is declared once with its unit, group and meaning (Col). The same list writes
the CSV and its glossary, so the two can never drift apart.

Inputs
  samples: fine-step histories of the chamber model (both runs)
           t [s], phi_abs [rad], p (z, N), pd, ps (N), q_port (z, N) flow INTO each chamber
           through its kidney, q_out (N) sum of port flows into the delivery line, q_load (N),
           q_leak (z, N) or None
  gap:     gap-step histories of the coupled solver (with-leakage run only), see solver.py
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd

from ..params import PumpParams, DEFAULT
from ..frame import SwashFrame, cradle_moment_arm
from ..forces import piston_forces, block_forces, swashplate_forces
from ..gaps.piston import PistonGap

LPM = 6.0e4
EXPORT_POINTS_PER_REV = 720


@dataclass
class Col:
    name: str
    unit: str
    group: str
    desc: str
    values: np.ndarray


# ------------------------------------------------------------------------- sampling
def sample_fine(H, t, phi_abs, steps, revs, ppr=EXPORT_POINTS_PER_REV):
    """Pick ppr points per revolution from fine-step histories (dict of arrays (..., n))."""
    every = steps // ppr
    idx = np.arange(0, revs * steps, every)
    S = {k: (None if v is None else np.asarray(v)[..., idx]) for k, v in H.items()}
    S["t"], S["phi_abs"], S["k"] = np.asarray(t)[idx], np.asarray(phi_abs)[idx], idx
    return S


def _gap_index(G, k):
    """Index of the last gap solve at or before fine step k."""
    return np.clip(np.searchsorted(G["k"], k, side="right") - 1, 0, len(G["k"]) - 1)


# ------------------------------------------------------------------------- table
def build_columns(S, prm: PumpParams = DEFAULT, G=None):
    fr = SwashFrame(prm)
    z = prm.num_piston
    phi = np.mod(S["phi_abs"], 2 * np.pi)
    p, p_d, p_s = S["p"], S["p_d"], S["p_s"]
    a, b, c = fr.abc
    gi = None if G is None else _gap_index(G, S["k"])
    hG = None if G is None else np.maximum(G["qG"][gi, :, 0].T, 0.3e-6)       # (z, N)
    pf = piston_forces(phi, p, prm, h_G=hG)
    bf = block_forces(pf, p, prm)
    sw = swashplate_forces(pf, prm)
    phj = pf["phi"]                                                             # (z, N)
    arm = cradle_moment_arm(phj, prm.Rp, prm.r_shoe, prm.beta, prm.eta, prm.ex, prm.ez)
    Tp = pf["F_DK"] / c * arm
    Ti = pf["F_aK"] / c * arm
    Q_geo = z * prm.Ak * fr.stroke * prm.rpm / 60.0
    rev = (S["k"] // (360 * prm.steps_per_deg)).astype(int) + 1
    q_pump = S["q_load"]
    eta_rev = pd.Series(q_pump).groupby(rev).transform("mean").to_numpy() / Q_geo * 100.0
    V_g = z * prm.Ak * fr.stroke                                                # displacement per rev

    C = []
    add = lambda *a_: C.append(Col(*a_))
    # ---------------------------------------------------------------- run
    add("time_s", "s", "run", "Time since the cold start.", S["t"])
    add("revolution", "-", "run", "Shaft revolution number (1 = first revolution after the cold start).", rev)
    add("phi_abs_deg", "deg", "run", "Shaft angle since the start, 0-1080 deg over three revolutions (clockwise seen from +z, PDF frame).",
        np.degrees(S["phi_abs"]))
    add("phi_deg", "deg", "run", "Shaft angle within the revolution, 0-360 deg. Piston 1 sits at this angle (phi = 0 near BDC; BDC ~351 deg, TDC ~171 deg).",
        np.degrees(phi))
    # ---------------------------------------------------------------- hydraulics
    add("P1_p_bar", "bar", "hydraulics", "Chamber pressure of piston 1 (absolute).", p[0] / 1e5)
    add("P1_Q_port_LPM", "L/min", "hydraulics",
        "Flow of chamber 1 through its valve-plate kidney, + out of the chamber into the ports (discharge), - into the chamber (suction).",
        -S["q_port"][0] * LPM)
    add("Pd_bar", "bar", "hydraulics", "Discharge (delivery line) pressure (absolute).", p_d / 1e5)
    add("Ps_bar", "bar", "hydraulics", "Suction line pressure (absolute).", p_s / 1e5)
    add("Q_discharge_LPM", "L/min", "hydraulics", "Discharge flow: sum of all kidney flows into the delivery line.", S["q_out"] * LPM)
    add("Q_pump_LPM", "L/min", "hydraulics", "Pump flow: flow through the load orifice (what the pump delivers to the system).", q_pump * LPM)
    add("Q_geometric_LPM", "L/min", "hydraulics", "Geometric flow 9 * Ak * stroke * n (no leakage, incompressible).",
        np.full(phi.size, Q_geo * LPM))
    add("eta_vol_inst_pct", "%", "hydraulics", "Instantaneous volumetric efficiency Q_pump / Q_geometric.", q_pump / Q_geo * 100.0)
    add("eta_vol_rev_pct", "%", "hydraulics", "Volumetric efficiency of the revolution: revolution-mean Q_pump / Q_geometric.", eta_rev)
    # ---------------------------------------------------------------- kinematics
    add("P1_sk_mm", "mm", "kinematics", "Piston 1 stroke position from BDC towards the valve plate (+z).", fr.sk(phj[0]) * 1e3)
    add("P1_vz_ms", "m/s", "kinematics", "Piston 1 axial velocity v_z (+ towards the valve plate = discharge stroke).", pf["vz"][0])
    add("P1_az_ms2", "m/s^2", "kinematics", "Piston 1 axial acceleration a_z.", pf["az"][0])
    add("P1_V_mm3", "mm^3", "kinematics", "Chamber 1 volume.", fr.volume(phj[0]) * 1e9)
    add("P1_lF_mm", "mm", "kinematics", "Piston 1 engagement (guide) length in the bore, block front face to piston end face.", pf["l_F"][0] * 1e3)
    add("P1_arm_mm", "mm", "kinematics", "Piston 1 cradle moment arm (PDF p.20): cradle torque = normal force * arm.", arm[0] * 1e3)
    # ---------------------------------------------------------------- torques
    add("P1_T_pressure_Nm", "N.m", "torque", "Piston 1 cradle torque from chamber pressure, F_DK / c * arm (PDF p.31 Tpp). + = CCW seen from +y.", Tp[0])
    add("P1_T_inertia_Nm", "N.m", "torque", "Piston 1 cradle torque from piston inertia, F_aK / c * arm (PDF p.31 Tpi).", Ti[0])
    add("P1_T_cradle_Nm", "N.m", "torque", "Piston 1 cradle torque, pressure + inertia.", Tp[0] + Ti[0])
    add("ALL_T_pressure_Nm", "N.m", "torque", "Cradle torque from pressure, sum of the 9 pistons.", Tp.sum(0))
    add("ALL_T_inertia_Nm", "N.m", "torque", "Cradle torque from inertia, sum of the 9 pistons.", Ti.sum(0))
    add("ALL_T_cradle_Nm", "N.m", "torque", "Total cradle torque of the 9 pistons (pressure + inertia).", Tp.sum(0) + Ti.sum(0))
    add("T_drive_Nm", "N.m", "torque", "Input shaft (drive) torque M_z from the piston radial forces.", bf["M_drive"])
    add("T_drive_ideal_Nm", "N.m", "torque", "Ideal drive torque V_g (Pd - Ps) / (2 pi).", V_g * (p_d - p_s) / (2 * np.pi))
    # ---------------------------------------------------------------- piston 1 external forces
    add("P1_F_DK_N", "N", "piston forces", "Pressure force on piston 1, -(p - p_case) Ak along z (PDF p.16).", pf["F_DK"][0])
    add("P1_F_aK_N", "N", "piston forces", "Inertia force of piston 1 + slipper, -m_k a_z along z.", pf["F_aK"][0])
    add("P1_F_AK_N", "N", "piston forces", "Total axial force F_DK + F_aK.", pf["F_AK"][0])
    add("P1_F_SK_N", "N", "piston forces", "Swash-plate normal force of piston 1, + = pressing on the plate (PDF FSK = F_AK / c, sign flipped).", -pf["F_SK_n"][0])
    add("P1_F_SKx_N", "N", "piston forces", "Swash-plate reaction on piston 1 at the ball joint, x component.", pf["F_SKx"][0])
    add("P1_F_SKy_N", "N", "piston forces", "Swash-plate reaction on piston 1 at the ball joint, y component.", pf["F_SKy"][0])
    add("P1_F_TG_N", "N", "piston forces", "Slipper viscous friction magnitude (opposes the sliding velocity).", pf["F_TG_mag"][0])
    add("P1_F_wK_N", "N", "piston forces", "Centrifugal force of piston 1 + slipper, m_k w^2 Rp (radially outward).", np.full(phi.size, pf["F_wK"]))
    add("P1_F_RKx_N", "N", "piston forces", "Radial force at the ball joint (Ivantysyn 4.79, with the equivalent centrifugal force), x component.", pf["F_RKx"][0])
    add("P1_F_RKy_N", "N", "piston forces", "Radial force at the ball joint, y component.", pf["F_RKy"][0])
    # ---------------------------------------------------------------- block / swash plate
    add("BLK_F_clamp_N", "N", "block forces", "Cylinder block clamping force onto the valve plate (+z): chamber pressure on the bore bottoms + spring.", bf["F_Bz"])
    add("BLK_F_Rx_N", "N", "block forces", "Radial force on the block from the 9 pistons, x.", bf["F_Bx"])
    add("BLK_F_Ry_N", "N", "block forces", "Radial force on the block from the 9 pistons, y.", bf["F_By"])
    add("BLK_F_bearing_N", "N", "block forces", "Radial bearing reaction magnitude.", bf["F_bearing"])
    add("BLK_M_VPx_Nm", "N.m", "block moments", "External tilting moment on the block about the valve-plate centre, x (with the bearing reaction).", bf["M_VPx_brg"])
    add("BLK_M_VPy_Nm", "N.m", "block moments", "External tilting moment on the block about the valve-plate centre, y (with the bearing reaction).", bf["M_VPy_brg"])
    add("BLK_xB_mm", "mm", "block forces", "Centre of the chamber clamping forces, x (Ivantysyn 4.110).", bf["x_B"] * 1e3)
    add("BLK_yB_mm", "mm", "block forces", "Centre of the chamber clamping forces, y.", bf["y_B"] * 1e3)
    add("SW_F_normal_N", "N", "swash plate", "Total normal force on the swash plate from the 9 pistons (+ = pressing).", sw["F_S_total"])
    add("SW_Fx_N", "N", "swash plate", "Force on the swash plate, x.", sw["F_Sx"])
    add("SW_Fy_N", "N", "swash plate", "Force on the swash plate, y.", sw["F_Sy"])
    add("SW_Fz_N", "N", "swash plate", "Force on the swash plate, z.", sw["F_Sz"])
    add("SW_Mx_Nm", "N.m", "swash plate", "Moment of the normal forces on the swash plate about the origin, x.", sw["M_Sx"])
    add("SW_My_Nm", "N.m", "swash plate", "Moment about the origin, y.", sw["M_Sy"])
    add("SW_Mz_Nm", "N.m", "swash plate", "Moment about the origin, z.", sw["M_Sz"])

    if G is None:
        return C, pf

    # ================================================================ lubricating gaps
    r_o = prm.slipper_r_out
    # ---------------------------------------------------------------- slipper
    hG1 = G["hpts_G1"][gi] * 1e6
    Ff = G["S1_F_fluid"][gi]; Qc = G["S1_Qc"][gi]; Qx = G["S1_Qext"][gi]
    QSG = G["QSG"][gi]
    add("SL1_h0_um", "um", "slipper film", "Slipper 1 film thickness at the centre of the land (h0).", G["qG"][gi, 0, 0] * 1e6)
    add("SL1_h0deg_um", "um", "slipper film", "Slipper 1 film at the land outer radius, 0 deg (sliding direction xi).", hG1[:, 0])
    add("SL1_h120deg_um", "um", "slipper film", "Slipper 1 film at the land outer radius, 120 deg (from xi towards zeta).", hG1[:, 1])
    add("SL1_h240deg_um", "um", "slipper film", "Slipper 1 film at the land outer radius, 240 deg.", hG1[:, 2])
    add("SL1_hmin_um", "um", "slipper film", "Slipper 1 minimum film (negative = inside the contact penalty).", G["hmin_G"][gi, 0] * 1e6)
    add("SL1_tilt_xi_mrad", "mrad", "slipper film", "Slipper 1 tilt about zeta (film slope along xi).", G["qG"][gi, 0, 1] * 1e3)
    add("SL1_tilt_zeta_mrad", "mrad", "slipper film", "Slipper 1 tilt about xi (film slope along zeta).", G["qG"][gi, 0, 2] * 1e3)
    add("SL1_p_pocket_bar", "bar", "slipper film", "Slipper 1 pocket pressure (absolute).", G["S1_p_pocket"][gi] / 1e5)
    add("SL_mean_h0_um", "um", "slipper film", "Mean centre film h0 of the 9 slippers.", G["qG"][gi, :, 0].mean(1) * 1e6)
    add("SL1_leak_LPM", "L/min", "slipper leakage", "Slipper 1 leakage out of chamber 1 (+ out of the chamber).", QSG[:, 0] * LPM)
    add("SL_mean_leak_LPM", "L/min", "slipper leakage", "Mean leakage per slipper (9 slippers).", QSG.mean(1) * LPM)
    add("SL_all_leak_LPM", "L/min", "slipper leakage", "Total leakage of the 9 slippers.", QSG.sum(1) * LPM)
    add("SL1_F_piston_N", "N", "slipper forces", "Normal force from piston 1 on slipper 1 (+ = pressing on the swash plate).", -G["S1_Fn"][gi])
    add("SL1_F_fluid_N", "N", "slipper forces", "Fluid film force on slipper 1 (pocket + land, incl. squeeze), + = lifting off the plate.", Ff[:, 0])
    add("SL1_F_contact_N", "N", "slipper forces", "Asperity contact force on slipper 1.", Qc[:, 0])
    add("SL1_F_retainer_N", "N", "slipper forces", "Retainer (hold-down) force on slipper 1.", G["F_retainer"][gi, 0])
    add("SL1_F_friction_N", "N", "slipper forces", "Viscous friction on the slipper 1 face.", G["S1_F_friction"][gi])
    add("SL1_M_fluid_xi_Nm", "N.m", "slipper moments", "Fluid tilting moment on slipper 1 about the ball joint, xi-tilt.", Ff[:, 1])
    add("SL1_M_fluid_zeta_Nm", "N.m", "slipper moments", "Fluid tilting moment on slipper 1 about the ball joint, zeta-tilt.", Ff[:, 2])
    add("SL1_M_contact_xi_Nm", "N.m", "slipper moments", "Contact tilting moment on slipper 1, xi-tilt.", Qc[:, 1])
    add("SL1_M_contact_zeta_Nm", "N.m", "slipper moments", "Contact tilting moment on slipper 1, zeta-tilt.", Qc[:, 2])
    add("SL1_M_ext_xi_Nm", "N.m", "slipper moments", "External tilting moment (friction + inertia) on slipper 1, xi-tilt.", Qx[:, 1])
    add("SL1_M_ext_zeta_Nm", "N.m", "slipper moments", "External tilting moment on slipper 1, zeta-tilt.", Qx[:, 2])
    # ---------------------------------------------------------------- valve plate
    hB = G["hpts_B"][gi] * 1e6
    fl = G["vp_flows"][gi] * LPM
    FfB = G["F_fluid_vp"][gi]; QcB = G["Qc_vp"][gi]
    QSB = G["QSB"][gi]
    add("VP_h0_um", "um", "valve-plate film", "Block / valve-plate film thickness at the axis (h0).", G["qB"][gi, 0] * 1e6)
    add("VP_h0deg_um", "um", "valve-plate film", "Valve-plate film at the outer land radius, 0 deg (+x).", hB[:, 0])
    add("VP_h120deg_um", "um", "valve-plate film", "Valve-plate film at the outer land radius, 120 deg (from +x towards +y).", hB[:, 1])
    add("VP_h240deg_um", "um", "valve-plate film", "Valve-plate film at the outer land radius, 240 deg.", hB[:, 2])
    add("VP_hmin_um", "um", "valve-plate film", "Minimum valve-plate film.", G["hmin_B"][gi] * 1e6)
    add("VP_K1_leak_LPM", "L/min", "valve-plate leakage", "Leakage out of kidney 1 (chamber 1) into the film.", QSB[:, 0] * LPM)
    add("VP_mean_leak_LPM", "L/min", "valve-plate leakage", "Mean kidney leakage per chamber (9 kidneys).", QSB.mean(1) * LPM)
    add("VP_all_leak_LPM", "L/min", "valve-plate leakage", "Total leakage out of the 9 kidneys into the film.", QSB.sum(1) * LPM)
    add("VP_case_leak_LPM", "L/min", "valve-plate leakage", "Valve-plate leakage reaching the case (inner + outer land edges; includes flow from the ports).", fl[:, 0] + fl[:, 1])
    add("VP_F_clamp_N", "N", "valve-plate forces", "Clamping load on the film (block external axial load).", G["F_clamp"][gi])
    add("VP_F_fluid_N", "N", "valve-plate forces", "Fluid lift force of the film (incl. squeeze).", FfB[:, 0])
    add("VP_F_contact_N", "N", "valve-plate forces", "Asperity contact force between block and valve plate.", QcB[:, 0])
    add("VP_M_ext_x_Nm", "N.m", "valve-plate moments", "External moment on the block about the valve-plate centre, x.", G["M_ext"][gi, 0])
    add("VP_M_ext_y_Nm", "N.m", "valve-plate moments", "External moment on the block about the valve-plate centre, y.", G["M_ext"][gi, 1])
    add("VP_M_fluid_x_Nm", "N.m", "valve-plate moments", "Fluid film moment about the valve-plate centre, x.", FfB[:, 1])
    add("VP_M_fluid_y_Nm", "N.m", "valve-plate moments", "Fluid film moment about the valve-plate centre, y.", FfB[:, 2])
    add("VP_M_contact_x_Nm", "N.m", "valve-plate moments", "Contact moment about the valve-plate centre, x.", QcB[:, 1])
    add("VP_M_contact_y_Nm", "N.m", "valve-plate moments", "Contact moment about the valve-plate centre, y.", QcB[:, 2])
    add("VP_M_bearing_x_Nm", "N.m", "valve-plate moments", "Moment carried by the bearing (block kept parallel to the plate), x.", G["M_bearing"][gi, 0])
    add("VP_M_bearing_y_Nm", "N.m", "valve-plate moments", "Moment carried by the bearing, y.", G["M_bearing"][gi, 1])
    # ---------------------------------------------------------------- piston / bore
    if G["QSK"].size:
        L = PistonGap(prm).L_ref
        e = G["K1_ecc"][gi] * 1e6
        hK = G["K1_hpts"][gi] * 1e6
        Qe, Qf, Qk = G["K1_Qext"][gi], G["K1_Ff4"][gi], G["K1_Qc"][gi]
        QSK = G["QSK"][gi]
        add("PB1_e_case_r_um", "um", "piston film", "Piston 1 eccentricity at the bore exit (case end), radial direction r.", e[:, 0])
        add("PB1_e_case_t_um", "um", "piston film", "Piston 1 eccentricity at the bore exit, tangential direction t = z x r.", e[:, 1])
        add("PB1_e_ch_r_um", "um", "piston film", "Piston 1 eccentricity at the piston end face (chamber end), r.", e[:, 2])
        add("PB1_e_ch_t_um", "um", "piston film", "Piston 1 eccentricity at the end face, t.", e[:, 3])
        for n, end, where in ((0, "case", "bore exit (case end)"), (1, "ch", "piston end face (chamber end)")):
            for m, ang in enumerate((0, 120, 240)):
                add(f"PB1_h_{end}_{ang}deg_um", "um", "piston film",
                    f"Piston 1 film at the {where}, {ang} deg from r towards t.", hK[:, 3 * n + m])
        add("PB1_hmin_um", "um", "piston film", "Piston 1 minimum film (radial clearance = half the diametral clearance when centred).", G["hmin_K"][gi, 0] * 1e6)
        add("PB1_leak_LPM", "L/min", "piston leakage", "Piston 1 / bore leakage out of chamber 1 (+ out; includes the drag of the sliding piston).", QSK[:, 0] * LPM)
        add("PB_mean_leak_LPM", "L/min", "piston leakage", "Mean piston / bore leakage per piston (9 pistons).", QSK.mean(1) * LPM)
        add("PB_all_leak_LPM", "L/min", "piston leakage", "Total piston / bore leakage of the 9 pistons.", QSK.sum(1) * LPM)
        for lab, Q, what in (("ext", Qe, "External lateral force (swash-plate reaction, slipper friction, centrifugal)"),
                             ("fluid", Qf, "Fluid film lateral force (incl. squeeze)"),
                             ("contact", Qk, "Bore contact lateral force")):
            add(f"PB1_F_{lab}_r_N", "N", "piston film forces", f"{what} on piston 1, radial direction r.", Q[:, 0] + Q[:, 2])
            add(f"PB1_F_{lab}_t_N", "N", "piston film forces", f"{what} on piston 1, tangential direction t.", Q[:, 1] + Q[:, 3])
        add("PB1_F_friction_N", "N", "piston film forces", "Viscous axial friction of the bore film on piston 1 (+z).", G["F_fric_K"][gi, 0])
        for lab, Q, what in (("ext", Qe, "external forces"), ("fluid", Qf, "fluid film"), ("contact", Qk, "bore contact")):
            add(f"PB1_M_{lab}_r_Nm", "N.m", "piston film moments",
                f"Moment of the {what} about the piston 1 end-face centre, from the r-forces (sum F_r * zeta).", -L * Q[:, 2])
            add(f"PB1_M_{lab}_t_Nm", "N.m", "piston film moments",
                f"Moment of the {what} about the end-face centre, from the t-forces (sum F_t * zeta).", -L * Q[:, 3])
    # ---------------------------------------------------------------- totals
    total = QSG.sum(1) + QSB.sum(1) + (G["QSK"][gi].sum(1) if G["QSK"].size else 0.0)
    add("LEAK_P1_LPM", "L/min", "total leakage", "Total leakage out of chamber 1 (slipper + kidney + piston bore).",
        (QSG[:, 0] + QSB[:, 0] + (G["QSK"][gi, 0] if G["QSK"].size else 0.0)) * LPM)
    add("LEAK_all_LPM", "L/min", "total leakage", "Total leakage out of the 9 chambers (all gaps).", total * LPM)
    return C, pf


def to_frame(cols):
    return pd.DataFrame({c.name: np.asarray(c.values) for c in cols})


def write_csv_and_glossary(cols, csv_path, title):
    df = to_frame(cols)
    df.to_csv(csv_path, index=False, float_format="%.6g")
    write_glossary(cols, csv_path.with_name(csv_path.stem + "_glossary.md"), title)
    return df


PREFIXES = [
    ("P1_", "Piston / chamber 1 (piston 1 is at the shaft angle phi; piston j trails it by (j-1)*40 deg)."),
    ("ALL_", "Sum over the 9 pistons."),
    ("Pd, Ps", "Discharge (delivery line) and suction line pressure."),
    ("Q_", "Pump-level flows."),
    ("eta_vol", "Volumetric efficiency."),
    ("T_", "Shaft torques."),
    ("BLK_", "Cylinder block, external loads (from the pistons, spring, bearing)."),
    ("SW_", "Swash plate."),
    ("SL1_ / SL_", "Slipper 1 / all 9 slippers (slipper-swash plate gap)."),
    ("VP_", "Cylinder block - valve plate gap. VP_K1 = kidney of chamber 1."),
    ("PB1_ / PB_", "Piston 1 / all 9 pistons, piston-bore gap."),
    ("LEAK_", "Leakage summed over all gaps."),
]
SUFFIXES = [
    ("_s", "seconds"), ("_deg", "degrees"), ("_bar", "bar (absolute pressure)"), ("_LPM", "litres per minute"),
    ("_pct", "percent"), ("_mm", "millimetres"), ("_mm3", "cubic millimetres"), ("_um", "micrometres"),
    ("_mrad", "milliradians"), ("_ms", "metres per second"), ("_ms2", "metres per second squared"),
    ("_N", "newtons"), ("_Nm", "newton-metres"),
]
TERMS = [
    ("F_DK", "Pressure force on the piston (Ivantysyn D = Druck, K = Kolben/piston)."),
    ("F_aK", "Inertia force of the reciprocating piston + slipper."),
    ("F_AK", "Total axial force on the piston, F_DK + F_aK."),
    ("F_SK", "Swash-plate (S) force on the piston (K) along the plate normal n."),
    ("F_TG", "Friction (T) at the slipper (G = Gleitschuh)."),
    ("F_wK", "Centrifugal force (omega) of the piston + slipper."),
    ("F_RK", "Radial force on the piston at the ball joint."),
    ("arm", "Cradle moment arm (PDF p.20): y-component of (contact point - cradle axis) x n."),
    ("h0", "Film thickness at the centre (slipper, valve plate)."),
    ("hmin", "Minimum film thickness over the gap."),
    ("xi, zeta", "Slipper frame: xi = sliding direction, zeta = n x xi."),
    ("r, t", "Piston frame: r = radially outward from the shaft axis, t = z x r."),
    ("case / ch", "Piston-bore film ends: bore exit at the block front (case pressure) / piston end face (chamber)."),
    ("zeta (piston)", "Axial coordinate along the piston, 0 at the end face, negative towards the bore exit."),
    ("leak", "Leakage, + = out of the chamber, in L/min (LPM)."),
    ("ext / fluid / contact", "External load / fluid film force (incl. squeeze) / asperity contact penalty force."),
]


def write_glossary(cols, path, title):
    lines = [f"# {title} - CSV glossary", "",
             "One row every 0.5 deg of shaft rotation (720 points per revolution), revolutions 1-3 from a cold start "
             "(p_initial everywhere). Frame and signs: v5 Cradle Torque Eq.pdf (+z towards the valve plate, shaft angle "
             "clockwise seen from +z). Pressures in bar absolute unless stated; leakage + = out of the chamber.", "",
             "## Prefixes", "", "| Prefix | Meaning |", "|---|---|"]
    lines += [f"| `{p}` | {d} |" for p, d in PREFIXES]
    lines += ["", "## Unit suffixes", "", "| Suffix | Unit |", "|---|---|"]
    lines += [f"| `{s}` | {u} |" for s, u in SUFFIXES]
    lines += ["", "## Short terms", "", "| Term | Meaning |", "|---|---|"]
    lines += [f"| `{t}` | {d} |" for t, d in TERMS]
    lines += ["", "## Columns", ""]
    group = None
    for i, c in enumerate(cols, 1):
        if c.group != group:
            group = c.group
            lines += ["", f"### {group}", "", "| # | Column | Unit | Meaning |", "|---|---|---|---|"]
        lines.append(f"| {i} | `{c.name}` | {c.unit} | {c.desc} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    pd.DataFrame([(c.name, c.unit, c.group, c.desc) for c in cols],
                 columns=["column", "unit", "group", "meaning"]).to_csv(path.with_suffix(".csv"), index=False)
    return path


def rewrite_glossary(glossary_csv, title):
    """Regenerate the Markdown glossary from a saved glossary CSV (column, unit, group, meaning)."""
    g = pd.read_csv(glossary_csv)
    cols = [Col(r.column, r.unit, r.group, r.meaning, None) for r in g.itertuples()]
    return write_glossary(cols, glossary_csv.with_suffix(".md"), title)
