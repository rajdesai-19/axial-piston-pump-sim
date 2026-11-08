"""
post/figures.py - Result figures of both runs, laid out as dashboards:
chamber pressures and flows, kinematics, piston / block forces, swash plate and torques, a summary
dashboard, and (with-leakage run) slipper, valve-plate, piston-bore and leakage-breakdown figures.
All panels are plotted over the exported revolutions 1-3 against the shaft angle since the start.
"""
import numpy as np
import matplotlib.pyplot as plt

from ..params import DEFAULT
from .style import S1, S2, S3, INK, CONTEXT, MUTED, angle_axis

K = INK  # 4th series (totals / reference) in ink


def _panel(ax, x, series, title, ylabel, legend_loc="best", x_max=None):
    for s in series:
        y, lab, col = s[:3]
        kw = s[3] if len(s) > 3 else {}
        ax.plot(x, y, color=col, label=lab, **kw)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    angle_axis(ax, x_max or float(np.ceil(x.max() / 360.0) * 360.0))
    if len(series) > 1:
        ax.legend(loc=legend_loc)


def _save(fig, path, title):
    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def _m(df, c):
    """Mean over the last exported revolution (for labels)."""
    last = df["revolution"] == df["revolution"].max()
    return df.loc[last, c].mean()


# ------------------------------------------------------------------------- common figures
def fig_chamber(df, S, path, title, prm=DEFAULT):
    x = df["phi_abs_deg"].to_numpy()
    fig, axs = plt.subplots(2, 2, figsize=(14, 8.5))
    ax = axs[0, 0]
    for j in range(1, prm.num_piston):
        ax.plot(x, S["p"][j] / 1e5, color=CONTEXT, lw=0.7, zorder=1)
    ax.plot([], [], color=CONTEXT, lw=0.7, label="Chambers 2–9")
    _panel(ax, x, [(df["P1_p_bar"], "Chamber 1", S1, dict(zorder=3)), (df["Pd_bar"], "Discharge line Pd", S2),
                   (df["Ps_bar"], "Suction line Ps", S3)], "Chamber and line pressures", "Pressure [bar]")
    _panel(axs[0, 1], x, [(df["P1_Q_port_LPM"], "Chamber 1 port flow", S1)],
           "Piston 1 flow through its kidney (+ out of the chamber)", "Flow [L/min]")
    _panel(axs[1, 0], x, [(df["Q_discharge_LPM"], f"Discharge flow (last rev mean {_m(df, 'Q_discharge_LPM'):.2f})", S1),
                          (df["Q_pump_LPM"], f"Pump flow, load orifice ({_m(df, 'Q_pump_LPM'):.2f})", S2),
                          (df["Q_geometric_LPM"], f"Geometric flow ({df['Q_geometric_LPM'].iloc[0]:.2f})", K, dict(ls="--", lw=1.0))],
           "Discharge and pump flow", "Flow [L/min]")
    _panel(axs[1, 1], x, [(df["eta_vol_inst_pct"], "Instantaneous", CONTEXT, dict(lw=0.8)),
                          (df["eta_vol_rev_pct"], f"Revolution mean (last {_m(df, 'eta_vol_rev_pct'):.2f} %)", S1)],
           "Volumetric efficiency Q_pump / Q_geometric", "η_vol [%]")
    return _save(fig, path, title + " — chamber pressures and flows")


def fig_kinematics(df, path, title):
    x = df["phi_abs_deg"].to_numpy()
    fig, axs = plt.subplots(3, 2, figsize=(14, 10))
    _panel(axs[0, 0], x, [(df["P1_sk_mm"], "", S1)], "Piston 1 stroke position from BDC", "s_k [mm]")
    _panel(axs[0, 1], x, [(df["P1_vz_ms"], "", S1)], "Piston 1 axial velocity (+ discharge stroke)", "v_z [m/s]")
    _panel(axs[1, 0], x, [(df["P1_az_ms2"], "", S1)], "Piston 1 axial acceleration", "a_z [m/s²]")
    _panel(axs[1, 1], x, [(df["P1_arm_mm"], "", S1)], "Piston 1 cradle moment arm", "Arm [mm]")
    _panel(axs[2, 0], x, [(df["P1_V_mm3"], "", S1)], "Chamber 1 volume", "V [mm³]")
    _panel(axs[2, 1], x, [(df["P1_lF_mm"], "", S1)], "Piston 1 engagement length in the bore", "l_F [mm]")
    return _save(fig, path, title + " — piston 1 kinematics")


def fig_forces(df, path, title):
    x = df["phi_abs_deg"].to_numpy()
    fig, axs = plt.subplots(4, 2, figsize=(14, 15))
    _panel(axs[0, 0], x, [(df["P1_F_DK_N"], "Pressure F_DK", S1), (df["P1_F_aK_N"], "Inertia F_aK", S2),
                          (df["P1_F_AK_N"], "Total axial F_AK", K, dict(lw=1.0))], "Piston 1 axial forces (z)", "Force [N]")
    _panel(axs[0, 1], x, [(df["P1_F_SKx_N"], "Swash-plate reaction F_SKx", S1), (df["P1_F_RKx_N"], "Radial at ball F_RKx", S2)],
           "Piston 1 radial forces along x", "Force [N]")
    _panel(axs[1, 0], x, [(df["P1_F_SKy_N"], "Swash-plate reaction F_SKy", S1), (df["P1_F_RKy_N"], "Radial at ball F_RKy", S2)],
           "Piston 1 radial forces along y (secondary angle η)", "Force [N]")
    _panel(axs[1, 1], x, [(df["P1_F_SK_N"], "Normal force F_SK (+ pressing)", S1)],
           f"Piston 1 swash-plate normal force (slipper friction {_m(df, 'P1_F_TG_N'):.1f} N)", "Force [N]")
    _panel(axs[2, 0], x, [(df["BLK_F_clamp_N"], f"Clamping F_Bz (mean {_m(df, 'BLK_F_clamp_N'):.0f} N)", S1)],
           "Cylinder block clamping force onto the valve plate", "Force [N]")
    _panel(axs[2, 1], x, [(df["BLK_F_Rx_N"], "Block F_Bx", S1), (df["BLK_F_Ry_N"], "Block F_By", S2),
                          (df["BLK_F_bearing_N"], "Bearing reaction |F|", K, dict(lw=1.0))],
           "Block radial loads and bearing reaction", "Force [N]")
    _panel(axs[3, 0], x, [(df["T_drive_Nm"], f"Drive torque (mean {_m(df, 'T_drive_Nm'):.2f} N·m)", S1),
                          (df["T_drive_ideal_Nm"], f"Ideal V_g Δp / 2π ({_m(df, 'T_drive_ideal_Nm'):.2f})", K, dict(ls="--", lw=1.0))],
           "Input shaft drive torque", "Torque [N·m]")
    _panel(axs[3, 1], x, [(df["BLK_M_VPx_Nm"], "M_x", S1), (df["BLK_M_VPy_Nm"], "M_y", S2)],
           "Block tilting moments about the valve-plate centre (with bearing)", "Moment [N·m]")
    return _save(fig, path, title + " — piston and cylinder-block forces")


def fig_swash(df, path, title):
    x = df["phi_abs_deg"].to_numpy()
    fig, axs = plt.subplots(3, 2, figsize=(14, 11))
    _panel(axs[0, 0], x, [(df["SW_F_normal_N"] / 1e3, f"Total (mean {_m(df, 'SW_F_normal_N') / 1e3:.2f} kN)", S1)],
           "Total normal force on the swash plate", "Force [kN]")
    _panel(axs[0, 1], x, [(df["SW_Fx_N"] / 1e3, "F_x", S1), (df["SW_Fy_N"] / 1e3, "F_y", S2), (df["SW_Fz_N"] / 1e3, "F_z", S3)],
           "Swash-plate force components", "Force [kN]")
    _panel(axs[1, 0], x, [(df["P1_T_pressure_Nm"], "Pressure", S1), (df["P1_T_inertia_Nm"], "Inertia", S2),
                          (df["P1_T_cradle_Nm"], "Total", K, dict(lw=1.0))], "Piston 1 cradle torque (+ CCW seen from +y)", "Torque [N·m]")
    _panel(axs[1, 1], x, [(df["ALL_T_pressure_Nm"], f"Pressure (mean {_m(df, 'ALL_T_pressure_Nm'):.1f})", S1),
                          (df["ALL_T_inertia_Nm"], f"Inertia ({_m(df, 'ALL_T_inertia_Nm'):.1f})", S2),
                          (df["ALL_T_cradle_Nm"], f"Total ({_m(df, 'ALL_T_cradle_Nm'):.1f})", K, dict(lw=1.0))],
           "Cradle torque of the 9 pistons", "Torque [N·m]")
    _panel(axs[2, 0], x, [(df["SW_Mx_Nm"], "M_x", S1), (df["SW_My_Nm"], "M_y", S2), (df["SW_Mz_Nm"], "M_z", S3)],
           "Swash-plate moments about the origin", "Moment [N·m]")
    _panel(axs[2, 1], x, [(df["P1_arm_mm"], "", S1)], "Piston 1 cradle moment arm", "Arm [mm]")
    return _save(fig, path, title + " — swash plate and cradle torque")


def fig_dashboard(df, path, title, leak):
    x = df["phi_abs_deg"].to_numpy()
    fig, axs = plt.subplots(4, 2, figsize=(14, 15))
    _panel(axs[0, 0], x, [(df["P1_p_bar"], "Chamber 1", S1), (df["Pd_bar"], "Discharge Pd", S2), (df["Ps_bar"], "Suction Ps", S3)],
           "Pressures", "Pressure [bar]")
    _panel(axs[0, 1], x, [(df["Q_discharge_LPM"], "Discharge", S1), (df["Q_pump_LPM"], "Pump (load orifice)", S2)],
           f"Flows (η_vol last rev {_m(df, 'eta_vol_rev_pct'):.2f} %)", "Flow [L/min]")
    _panel(axs[1, 0], x, [(df["P1_F_AK_N"], "Axial F_AK", S1), (df["P1_F_SK_N"], "Swash-plate normal F_SK", S2)],
           "Piston 1 axial and swash-plate forces", "Force [N]")
    if leak:
        _panel(axs[1, 1], x, [(df["SL_all_leak_LPM"], "9 slippers", S1), (df["VP_all_leak_LPM"], "Valve plate (kidneys)", S2),
                              *([(df["PB_all_leak_LPM"], "9 piston bores", S3)] if "PB_all_leak_LPM" in df else []),
                              (df["LEAK_all_LPM"], "Total", K, dict(lw=1.0))],
               f"Leakage (last rev mean total {_m(df, 'LEAK_all_LPM'):.3f} L/min)", "Leakage [L/min]")
        _panel(axs[2, 0], x, [(df["SL1_h0_um"], "Slipper 1 h0", S1), (df["VP_h0_um"], "Valve plate h0", S2),
                              *([(df["PB1_hmin_um"], "Piston 1 h_min", S3)] if "PB1_hmin_um" in df else [])],
               "Film thickness", "Film [µm]")
    else:
        _panel(axs[1, 1], x, [(df["eta_vol_inst_pct"], "Instantaneous", CONTEXT, dict(lw=0.8)),
                              (df["eta_vol_rev_pct"], "Revolution mean", S1)], "Volumetric efficiency (no leakage)", "η_vol [%]")
        _panel(axs[2, 0], x, [(df["P1_vz_ms"], "", S1)], "Piston 1 axial velocity", "v_z [m/s]")
    _panel(axs[2, 1], x, [(df["BLK_F_bearing_N"], "", S1)], "Bearing radial reaction", "Force [N]")
    _panel(axs[3, 0], x, [(df["BLK_F_clamp_N"], "Clamping force", S1)], "Cylinder block clamping", "Force [N]")
    _panel(axs[3, 1], x, [(df["ALL_T_cradle_Nm"], "Cradle torque (9 pistons)", S1), (df["T_drive_Nm"], "Drive torque", S2)],
           "Cradle and drive torque", "Torque [N·m]")
    return _save(fig, path, title + " — dashboard")


# ------------------------------------------------------------------------- with-leakage figures
def fig_slipper(df, G, gi, path, title, prm=DEFAULT):
    x = df["phi_abs_deg"].to_numpy()
    fig, axs = plt.subplots(4, 2, figsize=(14, 15))
    _panel(axs[0, 0], x, [(df["SL1_h0deg_um"], "h at 0° (ξ)", S1), (df["SL1_h120deg_um"], "h at 120°", S2),
                          (df["SL1_h240deg_um"], "h at 240°", S3), (df["SL1_h0_um"], f"centre h0 (last rev {_m(df, 'SL1_h0_um'):.2f})", K, dict(lw=1.0))],
           "Slipper 1 film: centre and 3 points at the land outer radius", "Film [µm]")
    ax = axs[0, 1]
    h0 = G["qG"][gi, :, 0] * 1e6
    for j in range(1, prm.num_piston):
        ax.plot(x, h0[:, j], color=CONTEXT, lw=0.7)
    ax.plot([], [], color=CONTEXT, lw=0.7, label="Slippers 2–9")
    _panel(ax, x, [(h0[:, 0], "Slipper 1", S1)], "Centre film h0 of all 9 slippers", "h0 [µm]")
    _panel(axs[1, 0], x, [(df["SL1_tilt_xi_mrad"], "ξ-tilt", S1), (df["SL1_tilt_zeta_mrad"], "ζ-tilt", S2)],
           "Slipper 1 tilts", "Tilt [mrad]")
    _panel(axs[1, 1], x, [(df["P1_p_bar"], "Chamber 1", S1), (df["SL1_p_pocket_bar"], "Slipper 1 pocket", S2)],
           "Chamber and pocket pressure", "Pressure [bar]")
    _panel(axs[2, 0], x, [(df["SL1_leak_LPM"], f"Slipper 1 (last rev mean {_m(df, 'SL1_leak_LPM'):.4f})", S1),
                          (df["SL_mean_leak_LPM"], f"Mean per slipper ({_m(df, 'SL_mean_leak_LPM'):.4f})", S2)],
           "Slipper leakage, single", "Leakage [L/min]")
    _panel(axs[2, 1], x, [(df["SL_all_leak_LPM"], f"9 slippers (last rev mean {_m(df, 'SL_all_leak_LPM'):.4f})", S1)],
           "Slipper leakage, total", "Leakage [L/min]")
    _panel(axs[3, 0], x, [(df["SL1_F_piston_N"], "From piston (+ pressing)", S1), (df["SL1_F_fluid_N"], "Fluid film", S2),
                          (df["SL1_F_contact_N"], "Contact", S3), (df["SL1_F_retainer_N"], "Retainer", K, dict(lw=1.0))],
           "Slipper 1 normal forces", "Force [N]")
    _panel(axs[3, 1], x, [(df["SL1_M_fluid_xi_Nm"], "Fluid, ξ-tilt", S1), (df["SL1_M_ext_xi_Nm"], "External, ξ-tilt", S2),
                          (df["SL1_M_fluid_zeta_Nm"], "Fluid, ζ-tilt", S3), (df["SL1_M_ext_zeta_Nm"], "External, ζ-tilt", K, dict(lw=1.0))],
           "Slipper 1 tilting moments about the ball joint", "Moment [N·m]")
    return _save(fig, path, title + " — slipper / swash-plate gap")


def fig_valveplate(df, path, title):
    x = df["phi_abs_deg"].to_numpy()
    fig, axs = plt.subplots(3, 2, figsize=(14, 11))
    _panel(axs[0, 0], x, [(df["VP_h0deg_um"], "h at 0° (+x)", S1), (df["VP_h120deg_um"], "h at 120°", S2),
                          (df["VP_h240deg_um"], "h at 240°", S3), (df["VP_h0_um"], f"axis h0 ({_m(df, 'VP_h0_um'):.3f})", K, dict(lw=1.0, ls="--"))],
           "Valve-plate film: axis and 3 points at the outer land radius", "Film [µm]")
    _panel(axs[0, 1], x, [(df["VP_K1_leak_LPM"], f"Kidney 1 ({_m(df, 'VP_K1_leak_LPM'):.5f})", S1),
                          (df["VP_mean_leak_LPM"], f"Mean per kidney ({_m(df, 'VP_mean_leak_LPM'):.5f})", S2)],
           "Valve-plate leakage, single", "Leakage [L/min]")
    _panel(axs[1, 0], x, [(df["VP_all_leak_LPM"], f"Out of the 9 kidneys ({_m(df, 'VP_all_leak_LPM'):.4f})", S1),
                          (df["VP_case_leak_LPM"], f"To the case ({_m(df, 'VP_case_leak_LPM'):.4f})", S2)],
           "Valve-plate leakage, total", "Leakage [L/min]")
    _panel(axs[1, 1], x, [(df["VP_F_clamp_N"], "Clamping load", S1), (df["VP_F_fluid_N"], "Fluid lift", S2),
                          (df["VP_F_contact_N"], "Contact", S3)], "Valve-plate axial forces", "Force [N]")
    _panel(axs[2, 0], x, [(df["VP_M_ext_x_Nm"], "External", S1), (df["VP_M_fluid_x_Nm"], "Fluid", S2),
                          (df["VP_M_bearing_x_Nm"], "Bearing", S3)], "Moments about the valve-plate centre, x", "Moment [N·m]")
    _panel(axs[2, 1], x, [(df["VP_M_ext_y_Nm"], "External", S1), (df["VP_M_fluid_y_Nm"], "Fluid", S2),
                          (df["VP_M_bearing_y_Nm"], "Bearing", S3)], "Moments about the valve-plate centre, y", "Moment [N·m]")
    return _save(fig, path, title + " — cylinder block / valve-plate gap")


def fig_piston_gap(df, path, title):
    x = df["phi_abs_deg"].to_numpy()
    fig, axs = plt.subplots(4, 2, figsize=(14, 15))
    _panel(axs[0, 0], x, [(df["PB1_e_case_r_um"], "Bore exit, r", S1), (df["PB1_e_case_t_um"], "Bore exit, t", S2),
                          (df["PB1_e_ch_r_um"], "End face, r", S3), (df["PB1_e_ch_t_um"], "End face, t", K, dict(lw=1.0))],
           f"Piston 1 eccentricity (clearance {DEFAULT.piston_clearance * 1e6:.0f} µm)", "Eccentricity [µm]")
    _panel(axs[0, 1], x, [(df["PB1_hmin_um"], f"h_min (min {df['PB1_hmin_um'].min():.2f} µm)", S1)], "Piston 1 minimum film", "Film [µm]")
    _panel(axs[1, 0], x, [(df[f"PB1_h_case_{a}deg_um"], f"{a}°", c) for a, c in ((0, S1), (120, S2), (240, S3))],
           "Piston 1 film at the bore exit (case end)", "Film [µm]")
    _panel(axs[1, 1], x, [(df[f"PB1_h_ch_{a}deg_um"], f"{a}°", c) for a, c in ((0, S1), (120, S2), (240, S3))],
           "Piston 1 film at the end face (chamber end)", "Film [µm]")
    _panel(axs[2, 0], x, [(df["PB1_leak_LPM"], f"Piston 1 ({_m(df, 'PB1_leak_LPM'):.4f})", S1),
                          (df["PB_mean_leak_LPM"], f"Mean per piston ({_m(df, 'PB_mean_leak_LPM'):.4f})", S2)],
           "Piston-bore leakage, single (+ out of the chamber)", "Leakage [L/min]")
    _panel(axs[2, 1], x, [(df["PB_all_leak_LPM"], f"9 pistons (last rev mean {_m(df, 'PB_all_leak_LPM'):.4f})", S1)],
           "Piston-bore leakage, total", "Leakage [L/min]")
    _panel(axs[3, 0], x, [(np.hypot(df["PB1_F_ext_r_N"], df["PB1_F_ext_t_N"]), "External |F|", S1),
                          (np.hypot(df["PB1_F_fluid_r_N"], df["PB1_F_fluid_t_N"]), "Fluid |F|", S2),
                          (np.hypot(df["PB1_F_contact_r_N"], df["PB1_F_contact_t_N"]), "Contact |F|", S3)],
           "Piston 1 lateral forces", "Force [N]")
    _panel(axs[3, 1], x, [(np.hypot(df["PB1_M_ext_r_Nm"], df["PB1_M_ext_t_Nm"]), "External |M|", S1),
                          (np.hypot(df["PB1_M_fluid_r_Nm"], df["PB1_M_fluid_t_Nm"]), "Fluid |M|", S2),
                          (np.hypot(df["PB1_M_contact_r_Nm"], df["PB1_M_contact_t_Nm"]), "Contact |M|", S3)],
           "Piston 1 moments about the end-face centre", "Moment [N·m]")
    return _save(fig, path, title + " — piston / bore gap")


def fig_leakage(df, path, title):
    x = df["phi_abs_deg"].to_numpy()
    fig, axs = plt.subplots(2, 2, figsize=(14, 8.5))
    _panel(axs[0, 0], x, [(df["SL_all_leak_LPM"], "9 slippers", S1), (df["VP_all_leak_LPM"], "Valve plate (9 kidneys)", S2),
                          *([(df["PB_all_leak_LPM"], "9 piston bores", S3)] if "PB_all_leak_LPM" in df else []),
                          (df["LEAK_all_LPM"], "Total", K, dict(lw=1.0))],
           "Leakage of all chambers", "Leakage [L/min]")
    _panel(axs[0, 1], x, [(df["SL1_leak_LPM"], "Slipper 1", S1), (df["VP_K1_leak_LPM"], "Kidney 1", S2),
                          *([(df["PB1_leak_LPM"], "Piston 1 bore", S3)] if "PB1_leak_LPM" in df else []),
                          (df["LEAK_P1_LPM"], "Total chamber 1", K, dict(lw=1.0))],
           "Leakage of chamber 1", "Leakage [L/min]")
    ax = axs[1, 0]
    series = [("SL_all_leak_LPM", "Slippers", S1), ("VP_all_leak_LPM", "Valve plate", S2)]
    if "PB_all_leak_LPM" in df:
        series.append(("PB_all_leak_LPM", "Piston bores", S3))
    series.append(("LEAK_all_LPM", "Total", K))
    g = df.groupby("revolution")[[c for c, _, _ in series]].mean()
    w = 0.2
    for k, (c, lab, col) in enumerate(series):
        bars = ax.bar(g.index + (k - (len(series) - 1) / 2) * w, g[c], w, color=col, label=lab)
        ax.bar_label(bars, fmt="%.3f", fontsize=7, color=MUTED)
    ax.set_xticks(g.index); ax.set_xlabel("Revolution"); ax.set_ylabel("Mean leakage [L/min]")
    ax.set_title("Mean leakage per revolution"); ax.legend()
    _panel(axs[1, 1], x, [(df["eta_vol_inst_pct"], "Instantaneous", CONTEXT, dict(lw=0.8)),
                          (df["eta_vol_rev_pct"], f"Revolution mean (last {_m(df, 'eta_vol_rev_pct'):.2f} %)", S1)],
           "Volumetric efficiency", "η_vol [%]")
    return _save(fig, path, title + " — leakage breakdown")


def all_figures(df, S, outdir, title, G=None, gi=None):
    outdir.mkdir(parents=True, exist_ok=True)
    leak = G is not None
    files = [fig_dashboard(df, outdir / "00_dashboard.png", title, leak),
             fig_chamber(df, S, outdir / "01_chamber_pressures_flows.png", title),
             fig_kinematics(df, outdir / "02_kinematics.png", title),
             fig_forces(df, outdir / "03_piston_block_forces.png", title),
             fig_swash(df, outdir / "04_swashplate_cradle_torque.png", title)]
    if leak:
        files += [fig_slipper(df, G, gi, outdir / "05_slipper_gap.png", title),
                  fig_valveplate(df, outdir / "06_valveplate_gap.png", title)]
        if "PB1_leak_LPM" in df:
            files.append(fig_piston_gap(df, outdir / "07_piston_bore_gap.png", title))
        files.append(fig_leakage(df, outdir / "08_leakage_breakdown.png", title))
    return files
