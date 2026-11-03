"""
validate_gaps.py - Phase 3 stand-alone check of the gap solvers with the real loads.

For each shaft angle (cycle-5 chamber pressures from Phase 1, external loads from Phase 2)
the static film equilibrium  F_fluid(q) + Q_contact(q) + Q_ext(q) = 0  (qdot = 0) is found
by damped Newton iteration. This shows whether a lubricating film carries the load
everywhere, without the coupled time march of Phase 4.

The piston/bore gap is checked separately (one-way time march, see piston()).

Run:  py -m pump_sim_v2.tools.validate_gaps            (valve plate + slipper)
      py -m pump_sim_v2.tools.validate_gaps piston     (piston / bore gap)
"""
from pathlib import Path
import time
import numpy as np

from ..params import DEFAULT
from ..forces import piston_forces, block_forces
from ..gaps.valveplate import ValvePlateGap, block_external_load
from ..gaps.slipper import SlipperGap

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "tools"
LPM = 6.0e4


def newton(residual, q0, steps, lo, iters=40, tol=1e-6):
    """Damped Newton with forward-difference Jacobian; returns q, residual norm, iterations."""
    q = np.array(q0, float)
    r = residual(q)
    scale = np.maximum(np.abs(r), 1.0)
    for it in range(iters):
        J = np.empty((3, 3))
        for k in range(3):
            dq = np.zeros(3); dq[k] = steps[k]
            J[:, k] = (residual(q + dq) - r) / steps[k]
        dq = np.linalg.solve(J, -r)
        lam = 1.0
        while lam > 1e-3:
            qn = q + lam * dq
            qn[0] = max(qn[0], lo)
            rn = residual(qn)
            if np.linalg.norm(rn / scale) < np.linalg.norm(r / scale):
                break
            lam *= 0.5
        q, r = qn, rn
        if np.linalg.norm(r / scale) < tol:
            break
    return q, np.linalg.norm(r / scale), it + 1


def main(step_deg=10):
    prm = DEFAULT
    d = np.load(OUT / "chamber_cycles_1-5_full.npz")
    steps = 360 * prm.steps_per_deg
    sl = slice(4 * steps, 5 * steps)
    idx = np.arange(0, steps, prm.steps_per_deg * step_deg)
    phi = np.mod(d["phi_abs"][sl][idx], 2 * np.pi)
    p = d["p"][:, sl][:, idx]
    pd, ps = d["p_d"][sl][idx], d["p_s"][sl][idx]
    pf = piston_forces(phi, p, prm)
    bf = block_forces(pf, p, prm)

    # ---------------- valve plate ----------------
    # For a parallel film with Dirichlet boundaries the pressure field does not depend on h0,
    # so the hydrostatic lift has no h0-stiffness and a static Newton solve is singular.
    # Report the load balance instead (flat film, h0 = 5 um).
    vp = ValvePlateGap(prm)
    rows = []
    t0 = time.time()
    for k in range(phi.size):
        Qe = block_external_load(bf, k)
        st = vp.evaluate(phi[k], np.array([5e-6, 0.0, 0.0]), p[:, k], pd[k], ps[k])
        f = vp.flows(st)
        net = st.F_base + Qe
        rows.append((np.degrees(phi[k]), st.F_base[0], -Qe[0], st.F_base[0] / -Qe[0],
                     st.F_base[1], Qe[1], st.F_base[2], Qe[2], net[1], net[2],
                     (f["case_inner"] + f["case_outer"]) * LPM))
    t_vp = time.time() - t0
    R = np.array(rows)
    print("VALVE PLATE load balance, flat film h0 = 5 um (cycle-5 loads)")
    print(" phi  F_lift[N] F_clamp[N] lift/clamp  Mx_fluid  Mx_ext  My_fluid  My_ext  net_Mx  net_My  Q_case@5um[LPM]")
    for r in R:
        print(f"{r[0]:4.0f} {r[1]:9.0f} {r[2]:9.0f} {r[3]:10.3f} {r[4]:9.1f} {r[5]:7.1f} {r[6]:9.1f} {r[7]:7.1f} "
              f"{r[8]:7.1f} {r[9]:7.1f} {r[10]:10.3f}")
    print(f"mean lift/clamp {R[:, 3].mean():.3f}; mean unbalanced moments Mx {R[:, 8].mean():.1f} N.m, "
          f"My {R[:, 9].mean():.1f} N.m; required centre-of-pressure shift "
          f"{np.hypot(R[:, 8], R[:, 9]).mean() / R[:, 2].mean() * 1e3:.1f} mm  ({t_vp:.0f} s)\n")

    # ---------------- slipper of piston 1 ----------------
    sg = SlipperGap(prm)
    rows = []
    q = np.array([prm.slipper_h0_init, 0.0, 0.0])
    t0 = time.time()
    for k in range(phi.size):
        phj = pf["phi"][0, k]
        _, _, _, v = sg.local_axes(phj)

        def res(qq):
            st = sg.evaluate(qq, p[0, k], v)
            return st.F_base + st.Q_contact + sg.external_load(phj, pf["F_SK_n"][0, k], st.F_friction) + sg.retainer(qq)
        q, rn, it = newton(res, q, (2e-8, 1e-5, 1e-5), lo=0.3e-6)
        st = sg.evaluate(q, p[0, k], v)
        hmin = q[0] - np.hypot(q[1], q[2]) * prm.slipper_r_out
        rows.append((np.degrees(phj), p[0, k] / 1e5, q[0] * 1e6, q[1] * 1e3, q[2] * 1e3, hmin * 1e6,
                     st.p_pocket / 1e5, sg.leakage(st) * LPM, st.Q_contact[0], rn, it))
    t_sg = time.time() - t0
    S = np.array(rows)
    print("SLIPPER (piston 1) static equilibrium")
    print(" phi_j  p[bar]  h0[um] g_xi[mrad] g_zeta[mrad] hmin[um] p_pock[bar] Q[LPM] Fcont[N] resid  it")
    for r in S:
        print(f"{r[0]:6.0f} {r[1]:7.1f} {r[2]:7.3f} {r[3]:10.4f} {r[4]:11.4f} {r[5]:8.3f} {r[6]:10.1f} {r[7]:7.4f} {r[8]:8.1f} {r[9]:.1e} {int(r[10]):3d}")
    print(f"mean h0 {S[:, 2].mean():.2f} um, min film {S[:, 5].min():.2f} um, mean leakage per slipper {S[:, 7].mean():.4f} LPM "
          f"(x9 = {9 * S[:, 7].mean():.3f} LPM), contact at {np.sum(S[:, 8] > 0)} of {len(S)} angles  ({t_sg:.0f} s)")
    np.savez(OUT / "gaps_static_equilibrium.npz", vp=R, slipper=S)


def piston(step_deg=0.5, revs=2):
    """
    Piston 1 / bore gap with the real loads, one-way (Phase-1 cycle-5 chamber pressures, no
    leakage feedback). A static solve (qdot = 0) is not used: the hydrostatic film stiffness of a
    tilted piston is indefinite (a piston tilted with its narrow side at the chamber end is pushed
    further over), so there is no unique static film equilibrium during discharge. The squeeze film
    decides the position, exactly as in the coupled march: backward Euler at the gap step.
    """
    from ..gaps.piston import PistonGap
    from ..solver import CoupledSolver
    prm = DEFAULT
    d = np.load(OUT / "chamber_cycles_1-5_full.npz")
    steps = 360 * prm.steps_per_deg
    sl = slice(4 * steps, 5 * steps)
    kk = int(round(step_deg * prm.steps_per_deg))
    idx = np.arange(0, steps, kk)
    phi = np.mod(d["phi_abs"][sl][idx], 2 * np.pi)
    p = d["p"][:, sl][:, idx]
    pf = piston_forces(phi, p, prm)
    pg = PistonGap(prm)
    solver = CoupledSolver.__new__(CoupledSolver)
    solver.prm, solver.pg = prm, pg
    dt_k = kk * 2 * np.pi / (steps * prm.omega)
    q = np.zeros(4)
    rows = []
    t0 = time.time()
    for r in range(revs):
        for k in range(phi.size):
            phj = pf["phi"][0, k]
            F_ball = (pf["F_SKx"][0, k] + pf["F_TGx"][0, k], pf["F_SKy"][0, k] + pf["F_TGy"][0, k])
            q, st, qd, Qe = solver._piston_step(phj, F_ball, p[0, k], q, dt_k)
            if r == revs - 1:
                eC, eD = pg.eccentricity_ends(q, phj)
                rows.append((np.degrees(phj), p[0, k] / 1e5, st.vz, Qe[0] + Qe[2], Qe[1] + Qe[3],
                             *(eC * 1e6), *(eD * 1e6), pg.h_min(q, phj) * 1e6, st.F_contact,
                             pg.leakage(st, p[0, k], qd) * LPM, st.F_friction))
    R = np.array(rows)
    print("PISTON 1 / BORE gap, one-way time march (last of %d revolutions, %.2f deg step)" % (revs, step_deg))
    print(" phi_j  p[bar]  vz[m/s] Fr_ext[N] Ft_ext[N]  eC_r   eC_t   eD_r   eD_t [um] hmin[um] Fcont[N]  Q[LPM]  Ffric[N]")
    for r in R[:: max(1, int(round(10 / step_deg)))]:
        print(f"{r[0]:6.0f} {r[1]:7.1f} {r[2]:7.2f} {r[3]:9.0f} {r[4]:9.0f} {r[5]:6.2f} {r[6]:6.2f} {r[7]:6.2f} {r[8]:6.2f} "
              f"{r[9]:8.2f} {r[10]:8.0f} {r[11]:8.4f} {r[12]:8.2f}")
    print(f"mean leakage per piston {R[:, 11].mean():.4f} LPM (x9 = {9 * R[:, 11].mean():.3f} LPM), min film "
          f"{R[:, 9].min():.2f} um, mean contact {R[:, 10].mean():.0f} N  ({time.time() - t0:.0f} s)")
    np.save(OUT / "gaps_piston_oneway.npy", R)
    return R


if __name__ == "__main__":
    import sys
    if "piston" in sys.argv[1:]:
        piston()
    else:
        main()
