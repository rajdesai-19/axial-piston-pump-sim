"""
chamber.py - Displacement chamber pressure model (isothermal), PDF frame.

Physics:
  dp_j/dt  = K / V_j * (Q_d,j + Q_s,j - Q_leak,j - dV_j/dt)
  dp_d/dt  = K / V_d * (Q_pump_out - Q_load)
  dp_s/dt  = K / V_s * (Q_inlet - sum_j Q_s,j)
with turbulent orifice flows through the valve-plate kidneys, the load orifice
and the suction inlet.

Changes vs the old file (see plan, Phase 1):
  * kinematics/volume from frame.py (v5 Cradle PDF frame, phi = 0 at BDC,
    clockwise); the port table is looked up at the piston angle directly
    (the old "+180 deg" is gone because the table is already in this frame);
  * uniform grid without duplicated 0/360 sample, dt = 2*pi / (steps * omega);
  * every flow sample is stored (old code left the last sample of each rev at 0);
  * all parameters come from params.py (rho = 870, load orifice -> reservoir).

Sign convention for flows: Q_d, Q_s > 0 means flow INTO the chamber.
Q_leak > 0 means flow OUT of the chamber (slipper + valve plate + piston gaps).
"""

from dataclasses import dataclass
from pathlib import Path
import csv
import numpy as np

from .params import PumpParams, DEFAULT
from .frame import SwashFrame


# [angle_deg, suction_percent, discharge_percent] -- valve-plate port opening in
# the PDF frame (discharge 0..180 deg, suction 180..360 deg). Generated from the overlap of the
# barrel kidney with the ports in cad_data/ (demo geometry), 100 % = A_port_max.
AREA_PROFILE_POINTS = np.array(
    [
        [0.0, 0.000, 0.000], [1.5, 0.000, 0.000], [2.0, 0.000, 0.016], [2.5, 0.000, 0.431],
        [3.0, 0.000, 1.113], [3.5, 0.000, 1.972], [4.0, 0.000, 2.970], [4.5, 0.000, 4.087],
        [5.0, 0.000, 5.306], [5.5, 0.000, 6.613], [6.0, 0.000, 7.999], [6.5, 0.000, 9.454],
        [7.0, 0.000, 10.972], [7.5, 0.000, 12.547], [8.0, 0.000, 14.173], [8.5, 0.000, 15.845],
        [9.0, 0.000, 17.554], [9.5, 0.000, 19.300], [10.0, 0.000, 21.075], [10.5, 0.000, 22.879],
        [11.0, 0.000, 24.700], [11.5, 0.000, 26.538], [12.0, 0.000, 28.377], [12.5, 0.000, 30.221],
        [13.0, 0.000, 32.064], [13.5, 0.000, 33.906], [14.0, 0.000, 35.749], [14.5, 0.000, 37.592],
        [15.0, 0.000, 39.436], [15.5, 0.000, 41.280], [16.0, 0.000, 43.122], [16.5, 0.000, 44.965],
        [17.0, 0.000, 46.806], [17.5, 0.000, 48.649], [18.0, 0.000, 50.492], [18.5, 0.000, 52.334],
        [19.0, 0.000, 54.177], [19.5, 0.000, 56.021], [20.0, 0.000, 57.864], [20.5, 0.000, 59.708],
        [21.0, 0.000, 61.551], [21.5, 0.000, 63.391], [22.0, 0.000, 65.237], [22.5, 0.000, 67.080],
        [23.0, 0.000, 68.924], [23.5, 0.000, 70.766], [24.0, 0.000, 72.606], [24.5, 0.000, 74.450],
        [25.0, 0.000, 76.292], [25.5, 0.000, 78.135], [26.0, 0.000, 79.977], [26.5, 0.000, 81.820],
        [27.0, 0.000, 83.665], [27.5, 0.000, 85.507], [28.0, 0.000, 87.350], [28.5, 0.000, 89.193],
        [29.0, 0.000, 91.035], [29.5, 0.000, 92.876], [30.0, 0.000, 94.721], [30.5, 0.000, 96.554],
        [31.0, 0.000, 98.341], [31.5, 0.000, 99.873], [32.0, 0.000, 100.000], [148.0, 0.000, 100.000],
        [148.5, 0.000, 99.873], [149.0, 0.000, 98.341], [149.5, 0.000, 96.554], [150.0, 0.000, 94.721],
        [150.5, 0.000, 92.876], [151.0, 0.000, 91.035], [151.5, 0.000, 89.193], [152.0, 0.000, 87.350],
        [152.5, 0.000, 85.507], [153.0, 0.000, 83.665], [153.5, 0.000, 81.820], [154.0, 0.000, 79.977],
        [154.5, 0.000, 78.135], [155.0, 0.000, 76.292], [155.5, 0.000, 74.450], [156.0, 0.000, 72.606],
        [156.5, 0.000, 70.766], [157.0, 0.000, 68.924], [157.5, 0.000, 67.080], [158.0, 0.000, 65.237],
        [158.5, 0.000, 63.391], [159.0, 0.000, 61.551], [159.5, 0.000, 59.708], [160.0, 0.000, 57.864],
        [160.5, 0.000, 56.021], [161.0, 0.000, 54.177], [161.5, 0.000, 52.334], [162.0, 0.000, 50.492],
        [162.5, 0.000, 48.649], [163.0, 0.000, 46.806], [163.5, 0.000, 44.965], [164.0, 0.000, 43.122],
        [164.5, 0.000, 41.280], [165.0, 0.000, 39.436], [165.5, 0.000, 37.592], [166.0, 0.000, 35.749],
        [166.5, 0.000, 33.906], [167.0, 0.000, 32.064], [167.5, 0.000, 30.221], [168.0, 0.000, 28.377],
        [168.5, 0.000, 26.538], [169.0, 0.000, 24.700], [169.5, 0.000, 22.879], [170.0, 0.000, 21.075],
        [170.5, 0.000, 19.300], [171.0, 0.000, 17.554], [171.5, 0.000, 15.845], [172.0, 0.000, 14.173],
        [172.5, 0.000, 12.547], [173.0, 0.000, 10.972], [173.5, 0.000, 9.454], [174.0, 0.000, 7.999],
        [174.5, 0.000, 6.613], [175.0, 0.000, 5.306], [175.5, 0.000, 4.087], [176.0, 0.000, 2.970],
        [176.5, 0.000, 1.972], [177.0, 0.000, 1.113], [177.5, 0.000, 0.431], [178.0, 0.000, 0.016],
        [178.5, 0.000, 0.000], [181.5, 0.000, 0.000], [182.0, 0.016, 0.000], [182.5, 0.431, 0.000],
        [183.0, 1.113, 0.000], [183.5, 1.972, 0.000], [184.0, 2.970, 0.000], [184.5, 4.087, 0.000],
        [185.0, 5.306, 0.000], [185.5, 6.613, 0.000], [186.0, 7.999, 0.000], [186.5, 9.454, 0.000],
        [187.0, 10.972, 0.000], [187.5, 12.547, 0.000], [188.0, 14.173, 0.000], [188.5, 15.845, 0.000],
        [189.0, 17.554, 0.000], [189.5, 19.300, 0.000], [190.0, 21.075, 0.000], [190.5, 22.879, 0.000],
        [191.0, 24.700, 0.000], [191.5, 26.538, 0.000], [192.0, 28.377, 0.000], [192.5, 30.221, 0.000],
        [193.0, 32.064, 0.000], [193.5, 33.906, 0.000], [194.0, 35.749, 0.000], [194.5, 37.592, 0.000],
        [195.0, 39.436, 0.000], [195.5, 41.280, 0.000], [196.0, 43.122, 0.000], [196.5, 44.965, 0.000],
        [197.0, 46.806, 0.000], [197.5, 48.649, 0.000], [198.0, 50.492, 0.000], [198.5, 52.334, 0.000],
        [199.0, 54.177, 0.000], [199.5, 56.021, 0.000], [200.0, 57.864, 0.000], [200.5, 59.708, 0.000],
        [201.0, 61.551, 0.000], [201.5, 63.391, 0.000], [202.0, 65.237, 0.000], [202.5, 67.080, 0.000],
        [203.0, 68.924, 0.000], [203.5, 70.766, 0.000], [204.0, 72.606, 0.000], [204.5, 74.450, 0.000],
        [205.0, 76.292, 0.000], [205.5, 78.135, 0.000], [206.0, 79.977, 0.000], [206.5, 81.820, 0.000],
        [207.0, 83.665, 0.000], [207.5, 85.507, 0.000], [208.0, 87.350, 0.000], [208.5, 89.193, 0.000],
        [209.0, 91.035, 0.000], [209.5, 92.876, 0.000], [210.0, 94.721, 0.000], [210.5, 96.554, 0.000],
        [211.0, 98.341, 0.000], [211.5, 99.873, 0.000], [212.0, 100.000, 0.000], [328.0, 100.000, 0.000],
        [328.5, 99.873, 0.000], [329.0, 98.341, 0.000], [329.5, 96.554, 0.000], [330.0, 94.721, 0.000],
        [330.5, 92.876, 0.000], [331.0, 91.035, 0.000], [331.5, 89.193, 0.000], [332.0, 87.350, 0.000],
        [332.5, 85.507, 0.000], [333.0, 83.665, 0.000], [333.5, 81.820, 0.000], [334.0, 79.977, 0.000],
        [334.5, 78.135, 0.000], [335.0, 76.292, 0.000], [335.5, 74.450, 0.000], [336.0, 72.606, 0.000],
        [336.5, 70.766, 0.000], [337.0, 68.924, 0.000], [337.5, 67.080, 0.000], [338.0, 65.237, 0.000],
        [338.5, 63.391, 0.000], [339.0, 61.551, 0.000], [339.5, 59.708, 0.000], [340.0, 57.864, 0.000],
        [340.5, 56.021, 0.000], [341.0, 54.177, 0.000], [341.5, 52.334, 0.000], [342.0, 50.492, 0.000],
        [342.5, 48.649, 0.000], [343.0, 46.806, 0.000], [343.5, 44.965, 0.000], [344.0, 43.122, 0.000],
        [344.5, 41.280, 0.000], [345.0, 39.436, 0.000], [345.5, 37.592, 0.000], [346.0, 35.749, 0.000],
        [346.5, 33.906, 0.000], [347.0, 32.064, 0.000], [347.5, 30.221, 0.000], [348.0, 28.377, 0.000],
        [348.5, 26.538, 0.000], [349.0, 24.700, 0.000], [349.5, 22.879, 0.000], [350.0, 21.075, 0.000],
        [350.5, 19.300, 0.000], [351.0, 17.554, 0.000], [351.5, 15.845, 0.000], [352.0, 14.173, 0.000],
        [352.5, 12.547, 0.000], [353.0, 10.972, 0.000], [353.5, 9.454, 0.000], [354.0, 7.999, 0.000],
        [354.5, 6.613, 0.000], [355.0, 5.306, 0.000], [355.5, 4.087, 0.000], [356.0, 2.970, 0.000],
        [356.5, 1.972, 0.000], [357.0, 1.113, 0.000], [357.5, 0.431, 0.000], [358.0, 0.016, 0.000],
        [358.5, 0.000, 0.000], [360.0, 0.000, 0.000],
    ],
    dtype=float,
)


def port_areas(phi_deg, a_max: float):
    """Discharge and suction kidney areas [m^2] at piston angle phi (deg, PDF frame)."""
    ph = np.mod(np.asarray(phi_deg, dtype=float), 360.0)
    ang = AREA_PROFILE_POINTS[:, 0]
    A_d = a_max * np.interp(ph, ang, AREA_PROFILE_POINTS[:, 2]) / 100.0
    A_s = a_max * np.interp(ph, ang, AREA_PROFILE_POINTS[:, 1]) / 100.0
    return A_d, A_s


def orifice_flow(p_up, p_down, cd: float, area, rho: float):
    """Signed turbulent orifice flow from p_up to p_down [m^3/s]."""
    dp = p_up - p_down
    return np.sign(dp) * cd * area * np.sqrt(2.0 * np.abs(dp) / rho)


@dataclass
class ChamberGrid:
    """Precomputed per-step kinematics and port areas for one revolution."""
    phi: np.ndarray        # shaft angle [rad], shape (steps,)
    dt: float
    V: np.ndarray          # (z, steps)
    dVdt: np.ndarray       # (z, steps)
    A_d: np.ndarray        # (z, steps)
    A_s: np.ndarray        # (z, steps)
    sk: np.ndarray         # (z, steps) displacement from BDC
    vz: np.ndarray         # (z, steps)
    lk: np.ndarray         # (z, steps) engagement length


def build_grid(prm: PumpParams = DEFAULT) -> ChamberGrid:
    steps = 360 * prm.steps_per_deg
    phi = np.linspace(0.0, 2.0 * np.pi, steps, endpoint=False)
    dt = (2.0 * np.pi / steps) / prm.omega
    fr = SwashFrame(prm)
    phj = fr.piston_phi(phi)                       # (z, steps)
    A_d, A_s = port_areas(fr.port_angle_deg(phj), prm.A_port_max)
    return ChamberGrid(phi=phi, dt=dt, V=fr.volume(phj), dVdt=fr.dV_dt(phj),
                       A_d=A_d, A_s=A_s, sk=fr.sk(phj), vz=fr.vz(phj), lk=fr.engagement(phj))


def rhs(p, p_d, p_s, V, dVdt, A_d, A_s, prm: PumpParams = DEFAULT, q_leak=0.0):
    """
    Time derivatives of the hydraulic state at one instant.
    p, V, dVdt, A_d, A_s, q_leak: arrays over pistons.
    Returns (dp_dt, dpd_dt, dps_dt, flows) with flows a dict of the
    individual flows [m^3/s].
    """
    q_d = orifice_flow(p_d, p, prm.cd_discharge_port, A_d, prm.rho)
    q_s = orifice_flow(p_s, p, prm.cd_suction_port, A_s, prm.rho)
    dp_dt = (prm.K_bulk / V) * (q_d + q_s - q_leak - dVdt)

    q_out = -np.sum(q_d)
    q_load = prm.cd_load * prm.A_load * np.sqrt(2.0 * max(p_d - prm.p_load_downstream, 0.0) / prm.rho)
    dpd_dt = (prm.K_bulk / prm.V_delivery_line) * (q_out - q_load)

    q_in = float(orifice_flow(prm.p_reservoir, p_s, prm.cd_suction_inlet, prm.A_suction_inlet, prm.rho))
    q_s_tot = float(np.sum(q_s))
    dps_dt = (prm.K_bulk / prm.V_suction_line) * (q_in - q_s_tot)

    return dp_dt, dpd_dt, dps_dt, dict(q_d=q_d, q_s=q_s, q_out=q_out, q_load=q_load,
                                       q_inlet=q_in, q_suction_ports=q_s_tot)


P_FLOOR = 0.1e5   # numerical floor on chamber / suction pressure [Pa] (as in the old model)


def simulate(prm: PumpParams = DEFAULT, q_leak=None, revolutions=None, verbose=True):
    """
    Explicit-Euler time march over `revolutions` shaft revolutions.

    q_leak: None, or array (z, steps) of leakage out of each chamber over one
            revolution (repeated every revolution).
    Returns a dict of full-resolution histories, shape (..., revs*steps).
    """
    g = build_grid(prm)
    z = prm.num_piston
    steps = g.phi.size
    revs = prm.total_revolutions if revolutions is None else revolutions
    n = revs * steps
    ql = np.zeros((z, steps)) if q_leak is None else np.asarray(q_leak, dtype=float)

    p_hist = np.empty((z, n)); pd_hist = np.empty(n); ps_hist = np.empty(n)
    qd_hist = np.empty((z, n)); qs_hist = np.empty((z, n))
    qout_hist = np.empty(n); qload_hist = np.empty(n); qin_hist = np.empty(n)

    p = np.full(z, prm.p_initial); p_d = prm.p_initial; p_s = prm.p_initial
    dt = g.dt
    for k in range(n):
        i = k % steps
        dp_dt, dpd_dt, dps_dt, f = rhs(p, p_d, p_s, g.V[:, i], g.dVdt[:, i],
                                       g.A_d[:, i], g.A_s[:, i], prm, ql[:, i])
        # store the state and the flows evaluated at the same instant
        p_hist[:, k] = p; pd_hist[k] = p_d; ps_hist[k] = p_s
        qd_hist[:, k] = f["q_d"]; qs_hist[:, k] = f["q_s"]
        qout_hist[k] = f["q_out"]; qload_hist[k] = f["q_load"]; qin_hist[k] = f["q_inlet"]
        # Euler update
        p = np.maximum(p + dp_dt * dt, P_FLOOR)
        p_d = p_d + dpd_dt * dt
        p_s = max(p_s + dps_dt * dt, P_FLOOR)

        if verbose and i == steps - 1:
            r = k // steps
            sl = slice(r * steps, (r + 1) * steps)
            print(f"  rev {r + 1:2d}/{revs}: Pd={pd_hist[sl].mean() / 1e5:7.2f} bar  "
                  f"Ps={ps_hist[sl].mean() / 1e5:6.2f} bar  "
                  f"Q_out={qout_hist[sl].mean() * 6e4:7.2f} LPM  Q_load={qload_hist[sl].mean() * 6e4:7.2f} LPM")

    return dict(grid=g, steps=steps, revs=revs, dt=dt,
                time=np.arange(n) * dt, phi_abs=np.arange(n) * (2.0 * np.pi / steps),
                p=p_hist, p_d=pd_hist, p_s=ps_hist, q_d=qd_hist, q_s=qs_hist,
                q_out=qout_hist, q_load=qload_hist, q_inlet=qin_hist)


def cycle_slice(res, rev: int) -> slice:
    """Slice of revolution `rev` (0-based; negative counts from the end)."""
    revs, steps = res["revs"], res["steps"]
    r = rev % revs
    return slice(r * steps, (r + 1) * steps)


def export_cycles_csv(res, path, prm: PumpParams = DEFAULT, n_cycles=None, points_per_rev=None):
    """
    CSV of cycles 1..n_cycles (same column layout as the old
    export_data_full_APP.csv). Kinematic columns are in the PDF frame:
    Y = sk = displacement from BDC towards the valve plate [mm],
    vk = v_z [m/s], lk = engagement length [mm].
    """
    n_cycles = prm.n_export_cycles if n_cycles is None else n_cycles
    ppr = prm.n_export_points_per_rev if points_per_rev is None else points_per_rev
    g, steps = res["grid"], res["steps"]
    idx = np.linspace(0, steps, ppr, endpoint=False).astype(int)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        header = ["cycle_number", "absolute_theta_deg", "time_s", "Pd_bar", "Ps_bar"]
        for j in range(prm.num_piston):
            header += [f"P{j + 1}_bar", f"Y{j + 1}_mm", f"vk{j + 1}_ms", f"lk{j + 1}_mm"]
        w.writerow(header)
        for c in range(min(n_cycles, res["revs"])):
            for i in idx:
                k = c * steps + i
                row = [c + 1, np.degrees(res["phi_abs"][k]), res["time"][k],
                       res["p_d"][k] / 1e5, res["p_s"][k] / 1e5]
                for j in range(prm.num_piston):
                    row += [res["p"][j, k] / 1e5, g.sk[j, i] * 1e3, g.vz[j, i], g.lk[j, i] * 1e3]
                w.writerow(row)
    return path


def export_cycles_npz(res, path, prm: PumpParams = DEFAULT, n_cycles=None):
    """Full-resolution histories of cycles 1..n_cycles (compressed NPZ)."""
    n_cycles = prm.n_export_cycles if n_cycles is None else n_cycles
    sl = slice(0, min(n_cycles, res["revs"]) * res["steps"])
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, time=res["time"][sl], phi_abs=res["phi_abs"][sl],
                        p=res["p"][:, sl], p_d=res["p_d"][sl], p_s=res["p_s"][sl],
                        q_d=res["q_d"][:, sl], q_s=res["q_s"][:, sl],
                        q_out=res["q_out"][sl], q_load=res["q_load"][sl], q_inlet=res["q_inlet"][sl])
    return path


if __name__ == "__main__":
    import time
    t0 = time.time()
    res = simulate(DEFAULT)
    out = Path(__file__).resolve().parent / "outputs" / "tools"
    export_cycles_csv(res, out / "chamber_cycles_1-5.csv")
    export_cycles_npz(res, out / "chamber_cycles_1-5_full.npz")
    sl = cycle_slice(res, -1)
    print(f"done in {time.time() - t0:.1f} s; last rev Pd={res['p_d'][sl].mean() / 1e5:.2f} bar, "
          f"Q_out={res['q_out'][sl].mean() * 6e4:.2f} LPM")
