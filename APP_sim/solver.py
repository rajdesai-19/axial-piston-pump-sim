"""
solver.py - Coupled isothermal time march.

Per time step:
  1. state: chamber pressures p_j, delivery p_d, suction p_s; block gap q_B = (h0, th_x, th_y);
     slipper gaps q_G,j = (h0, g_xi, g_zeta); piston eccentricities q_K,j = (eA_r, eA_t, eB_r, eB_t)
  2. external forces from forces/ (piston, block) at the current phi and p
  3. each gap: fluid force + squeeze matrix (gaps/), balance of forces
         F_fluid(q) + Q_contact(q) + Q_ext + C qdot = 0
     integrated with backward Euler:  (C/dt_k + J) dq = -(F_fluid + Q_contact + Q_ext)
  4. leakages Q_SK,j (piston/bore) + Q_SG,j (slipper orifice flow) + Q_SB,j (flow out of kidney j)
         dp_j/dt = K/V_j (Q_r - Q_SK - Q_SG - Q_SB - dV/dt)
  5. Euler update, t += dt, until the requested number of revolutions.

Multi-rate: the chamber ODE runs at the fine step dt (steps_per_deg); the gap solves run every
gap_step_deg. Between gap solves the slipper leakage is scaled with the instantaneous chamber
pressure (Q ~ p_j - p_c), the piston leakage uses its exact linear form Q_D (p_j - p_c) + Q_rest
(film frozen), and the valve-plate kidney leakage is held. The valve-plate fluid Jacobian is
refreshed every vp_jac_deg degrees (its contact part is analytic and updated every gap step);
the slipper and piston Jacobians are recomputed at every gap step (cheap).
"""
from dataclasses import dataclass, field
from pathlib import Path
import time
import numpy as np

from .params import PumpParams, DEFAULT
from .chamber import build_grid, rhs, P_FLOOR
from .forces import piston_forces, block_forces
from .gaps.valveplate import ValvePlateGap, block_external_load, vp_fields
from .gaps.slipper import SlipperGap
from .gaps.piston import PistonGap

LPM = 6.0e4


class _Reduced:
    """Gap state restricted to the solved dofs (keeps the full state in .full)."""
    def __init__(self, full, C):
        self.full, self.C = full, C


@dataclass
class SolverOptions:
    revolutions: int = 3
    gap_step_deg: float = 0.5
    vp_jac_deg: float = 5.0
    init_from: str = "cold"              # "cold" (p_initial everywhere) or "phase1" (tools/validate_chamber npz)
    max_dh_frac: float = 0.5             # limit on |dh0| per gap step as a fraction of h0
    verbose: bool = True
    record_fields: bool = True           # store pressure/leakage-flux field snapshots in the last revolution
    field_angles_deg: tuple = (0.0, 90.0, 180.0, 270.0)   # shaft angles of the snapshots


class CoupledSolver:
    def __init__(self, prm: PumpParams = DEFAULT, opt: SolverOptions = SolverOptions()):
        self.prm, self.opt = prm, opt
        self.grid = build_grid(prm)
        self.vp = ValvePlateGap(prm)
        self.sg = SlipperGap(prm)
        self.pg = PistonGap(prm) if prm.piston_gap else None
        self.k_gap = max(int(round(opt.gap_step_deg * prm.steps_per_deg)), 1)
        self.k_jac = max(int(round(opt.vp_jac_deg / opt.gap_step_deg)), 1)
        self.vp_dofs = [0, 1, 2] if prm.vp_tilt_free else [0]

    # ------------------------------------------------------------------ helpers
    def _initial_hydraulics(self):
        prm, z = self.prm, self.prm.num_piston
        if self.opt.init_from == "phase1":
            f = Path(__file__).resolve().parent / "outputs" / "tools" / "chamber_cycles_1-5_full.npz"
            d = np.load(f)
            n = d["p_d"].size
            steps = self.grid.phi.size
            if n % steps == 0 and d["p"].shape[0] == z:
                return d["p"][:, -1].copy(), float(d["p_d"][-1]), float(d["p_s"][-1])
        return np.full(z, prm.p_initial), prm.p_initial, prm.p_initial

    def _vp_fluid(self, phi, q, p, pd, ps):
        st = self.vp.evaluate(phi, q, p, pd, ps)
        return st, st.F_base

    def _vp_contact_jac(self, q):
        prm, g = self.prm, self.vp.g
        h = q[0] - q[1] * g.Y + q[2] * g.X
        pen = np.clip((prm.vp_h_contact - h) / prm.vp_h_contact, 0.0, None)
        dp = -2.0 * prm.vp_p_contact * pen / prm.vp_h_contact
        W = self.vp.W
        wA = g.dA                       # contact on the whole land (kidneys are a small share)
        return np.array([[np.sum(dp * W[i] * W[j] * wA) for j in range(3)] for i in range(3)])

    def _slipper_residual(self, q, p_j, v, phj, F_n):
        st = self.sg.evaluate(q, p_j, v)
        R = st.F_base + st.Q_contact + self.sg.external_load(phj, F_n, st.F_friction) + self.sg.retainer(q)
        return st, R

    def _piston_step(self, phj, F_ball, p_j, q, dt_k):
        """One backward-Euler step of the piston eccentricities (FD Jacobian at the start of the step)."""
        pg, prm = self.pg, self.prm
        Qe = pg.external_load(phj, F_ball)

        def res(qq):
            st = pg.evaluate(phj, qq, p_j)
            return st, st.F_base + st.Q_contact + Qe
        st0, R0 = res(q)
        J = np.empty((4, 4))
        for c in range(4):
            dq = np.zeros(4); dq[c] = 1e-9
            J[:, c] = (res(q + dq)[1] - R0) / 1e-9
        hm = max(pg.h_min(q, phj), prm.piston_h_contact)
        q_new, st, qd, _, _ = self.implicit_step(res, q, dt_k, lambda qq, s: J, np.full(4, 0.5 * hm), np.ones(4))
        return q_new, st, qd, Qe

    @staticmethod
    def implicit_step(resfun, q_old, dt_k, jacfun, dq_max, w, max_iter=10, tol=1e-6):
        """
        Backward Euler for C(q) qdot + R(q) = 0:  G(q) = R(q) + C(q) (q - q_old)/dt_k = 0,
        damped Newton with per-component step limits dq_max and backtracking on |w*G|.
        resfun(q) -> (state, R) with state.C; jacfun(q, state) -> approx dR/dq.
        Returns (q_new, state at q_new, qdot, iterations, |w*G|).
        """
        q = q_old.copy()
        st, R = resfun(q)
        G = R + st.C @ (q - q_old) / dt_k
        g0 = np.linalg.norm(w * G)
        scale = max(g0, 1.0)
        for it in range(1, max_iter + 1):
            if np.linalg.norm(w * G) <= tol * scale:
                break
            A = st.C / dt_k + jacfun(q, st)
            dq = np.linalg.solve(A, -G)
            f = np.max(np.abs(dq) / dq_max)
            if f > 1.0:
                dq /= f
            lam, gn = 1.0, np.linalg.norm(w * G)
            while True:
                qn = q + lam * dq
                stn, Rn = resfun(qn)
                Gn = Rn + stn.C @ (qn - q_old) / dt_k
                if np.linalg.norm(w * Gn) < gn or lam < 1.0 / 16:
                    break
                lam *= 0.5
            q, st, R, G = qn, stn, Rn, Gn
        return q, st, (q - q_old) / dt_k, it, np.linalg.norm(w * G) / scale

    # ------------------------------------------------------------------ main loop
    def run(self):
        prm, opt, g = self.prm, self.opt, self.grid
        z, steps = prm.num_piston, g.phi.size
        if not prm.slipper_use_retainer:
            print("  NOTE: slipper retainer disabled - during suction nothing holds the slipper on the plate")
        n = opt.revolutions * steps
        dt = g.dt
        dt_k = self.k_gap * dt

        p, pd, ps = self._initial_hydraulics()
        qB = np.array([prm.vp_h0_init, 0.0, 0.0])
        qG = np.tile([prm.slipper_h0_init, 0.0, 0.0], (z, 1))
        J_vp = None

        # fine-step histories
        H = dict(p=np.empty((z, n)), p_d=np.empty(n), p_s=np.empty(n), q_out=np.empty(n),
                 q_load=np.empty(n), q_leak=np.empty((z, n)), q_port=np.empty((z, n)))
        # gap-step histories
        G = {k: [] for k in ("k", "qB", "qBdot", "qG", "QSB", "QSG", "Q_case_vp", "F_lift", "F_clamp",
                             "F_contact_vp", "M_ext", "M_fluid", "M_bearing", "F_retainer", "p_pocket",
                             "F_contact_G", "hmin_B", "hmin_G",
                             # extra detail for the CSV export
                             "qGdot", "hpts_G1", "hpts_B", "F_fluid_vp", "Qc_vp", "vp_flows",
                             "S1_Fn", "S1_F_fluid", "S1_Qc", "S1_Qext", "S1_F_friction", "S1_p_ch",
                             "S1_p_pocket", "S1_v", "S1_phi", "pd", "ps",
                             # piston / bore gaps
                             "qK", "qKdot", "QSK", "hmin_K", "F_contact_K", "F_fric_K",
                             "K1_ecc", "K1_hpts", "K1_lF", "K1_vz", "K1_Fext", "K1_Ffluid", "K1_Qext",
                             "K1_Qc", "K1_Ff4")}
        # field snapshots (last revolution)
        FLD = {k: [] for k in ("phi", "vp_p", "vp_qr", "vp_qt", "vp_h0",
                               "s1_p", "s1_h", "s1_qr", "s1_qt", "s1_p_pocket", "s1_phi_piston",
                               "k1_p", "k1_h", "k1_qz", "k1_qz_p", "k1_z")}
        k_last = (opt.revolutions - 1) * steps
        k_gap_dt = self.k_gap
        k_fields = {k_last + int(round(a * prm.steps_per_deg / k_gap_dt)) * k_gap_dt
                    for a in opt.field_angles_deg}

        Q_SG = np.zeros(z); Q_SB = np.zeros(z)
        p_ref_G = np.zeros(z)
        qGdot = np.zeros((z, 3))
        qK = np.zeros((z, 4)); qKdot = np.zeros((z, 4))
        K_D = np.zeros(z); K_rest = np.zeros(z)          # Q_SK,j = K_D (p_j - p_c) + K_rest between gap steps
        t0 = time.time()
        for k in range(n):
            i = k % steps
            phi = g.phi[i]
            if k % self.k_gap == 0:
                # ---------- external forces at this instant
                pf = piston_forces(np.array([phi]), p[:, None], prm, h_G=np.maximum(qG[:, :1], 0.3e-6))
                bf = block_forces(pf, p[:, None], prm)
                Qe = block_external_load(bf, 0)

                # ---------- cylinder block gap (implicit balance, iterated)
                # dofs: h0 only when the bearing carries the tilt (vp_tilt_free = False)
                dofs = self.vp_dofs
                ix = np.ix_(dofs, dofs)
                if J_vp is None or (k // self.k_gap) % self.k_jac == 0:
                    st0 = self.vp.evaluate(phi, qB, p, pd, ps)
                    J_vp = np.zeros((3, 3))
                    for c in dofs:
                        s = (2e-8, 1e-6, 1e-6)[c]
                        dq = np.zeros(3); dq[c] = s
                        J_vp[:, c] = (self.vp.evaluate(phi, qB + dq, p, pd, ps).F_base - st0.F_base) / s

                def resB(qr, _q=qB.copy()):
                    qq = _q.copy(); qq[dofs] = qr
                    st = self.vp.evaluate(phi, qq, p, pd, ps)
                    return _Reduced(st, st.C[ix]), (st.F_base + st.Q_contact + Qe)[dofs]

                def jacB(qr, st, _q=qB.copy()):
                    qq = _q.copy(); qq[dofs] = qr
                    return (J_vp + self._vp_contact_jac(qq))[ix]

                hB = max(qB[0], prm.vp_h_contact)
                lim = np.array([0.5 * hB, 0.5 * hB / prm.vp_r_out, 0.5 * hB / prm.vp_r_out])[dofs]
                wts = np.array([1.0, 1.0 / prm.vp_r_out, 1.0 / prm.vp_r_out])[dofs]
                qr, red, qrdot, itB, _ = self.implicit_step(resB, qB[dofs], dt_k, jacB, lim, wts)
                qB[dofs] = qr
                qBdot = np.zeros(3); qBdot[dofs] = qrdot
                stB = red.full
                fl = self.vp.flows(stB, qBdot)
                Q_SB = fl["kidney"]
                M_brg = -(stB.F_base + stB.Q_contact + Qe)[1:]     # moment the bearing must carry

                # ---------- slipper gaps (implicit balance, FD Jacobian at the start of each step)
                for j in range(z):
                    phj = pf["phi"][j, 0]
                    _, _, _, v = self.sg.local_axes(phj)
                    F_n = pf["F_SK_n"][j, 0]
                    res = lambda qq: self._slipper_residual(qq, p[j], v, phj, F_n)
                    st0, R0 = res(qG[j])
                    J = np.empty((3, 3))
                    for c, s in enumerate((2e-9, 1e-6, 1e-6)):
                        dq = np.zeros(3); dq[c] = s
                        J[:, c] = (res(qG[j] + dq)[1] - R0) / s
                    hG = max(qG[j, 0], prm.slipper_h_contact)
                    r_o = prm.slipper_r_out
                    qG[j], stG, qd, _, _ = self.implicit_step(
                        res, qG[j], dt_k, lambda qq, st: J, np.array([0.5 * hG, 0.5 * hG / r_o, 0.5 * hG / r_o]),
                        np.array([1.0, 1.0 / r_o, 1.0 / r_o]))
                    Q_SG[j] = self.sg.leakage(stG, qd)          # signed: < 0 = inflow from the case
                    p_ref_G[j] = p[j] - prm.p_case
                    qGdot[j] = qd
                    G["p_pocket"].append(stG.p_pocket + stG.dp_pocket @ qd + prm.p_case)
                    G["F_contact_G"].append(stG.Q_contact[0])
                    G["hmin_G"].append(qG[j, 0] - np.hypot(qG[j, 1], qG[j, 2]) * prm.slipper_r_out)
                    if j == 0:
                        G["S1_Fn"].append(F_n)
                        G["S1_F_fluid"].append(stG.F_base + stG.C @ qd)
                        G["S1_Qc"].append(stG.Q_contact.copy())
                        G["S1_Qext"].append(self.sg.external_load(phj, F_n, stG.F_friction))
                        G["S1_F_friction"].append(stG.F_friction)
                        G["S1_p_ch"].append(p[j]); G["S1_v"].append(v); G["S1_phi"].append(phj)
                        G["S1_p_pocket"].append(stG.p_pocket + stG.dp_pocket @ qd + prm.p_case)
                        G["hpts_G1"].append(self.sg.film_points(qG[j]))
                        st_s1, qd_s1 = stG, qd.copy()

                # ---------- piston / bore gaps (implicit balance, FD Jacobian at the start of each step)
                if self.pg is not None:
                    pg = self.pg
                    hmK, FcK, FfK = np.zeros(z), np.zeros(z), np.zeros(z)
                    for j in range(z):
                        phj = pf["phi"][j, 0]
                        F_ball = (pf["F_SKx"][j, 0] + pf["F_TGx"][j, 0], pf["F_SKy"][j, 0] + pf["F_TGy"][j, 0])
                        qK[j], stK, qd, QeK = self._piston_step(phj, F_ball, p[j], qK[j], dt_k)
                        qKdot[j] = qd
                        K_D[j] = stK.Q_D
                        K_rest[j] = stK.Q_w + stK.dQ @ qd
                        hmK[j] = pg.h_min(qK[j], phj); FcK[j] = stK.F_contact; FfK[j] = stK.F_friction
                        if j == 0:
                            Ff = stK.F_base + stK.C @ qd
                            G["K1_ecc"].append(np.concatenate(pg.eccentricity_ends(qK[j], phj)))
                            G["K1_hpts"].append(pg.film_points(qK[j], phj, (0.0, 120.0, 240.0)).ravel())
                            G["K1_lF"].append(stK.g.z1 - stK.g.z0); G["K1_vz"].append(stK.vz)
                            G["K1_Fext"].append([QeK[0] + QeK[2], QeK[1] + QeK[3]])
                            G["K1_Ffluid"].append([Ff[0] + Ff[2], Ff[1] + Ff[3]])
                            G["K1_Qext"].append(QeK.copy())
                            G["K1_Qc"].append(stK.Q_contact.copy()); G["K1_Ff4"].append(Ff.copy())
                            st_k1, qd_k1 = stK, qd.copy()
                    G["qK"].append(qK.copy()); G["qKdot"].append(qKdot.copy())
                    G["QSK"].append(K_D * (p - prm.p_case) + K_rest)
                    G["hmin_K"].append(hmK); G["F_contact_K"].append(FcK); G["F_fric_K"].append(FfK)

                G["k"].append(k); G["qB"].append(qB.copy()); G["qBdot"].append(qBdot.copy())
                G["qG"].append(qG.copy()); G["QSB"].append(Q_SB.copy()); G["QSG"].append(Q_SG.copy())
                G["Q_case_vp"].append(fl["case_inner"] + fl["case_outer"])
                G["F_lift"].append(stB.F_base[0]); G["F_clamp"].append(-Qe[0])
                G["F_contact_vp"].append(stB.Q_contact[0]); G["M_ext"].append(Qe[1:].copy())
                G["M_fluid"].append(stB.F_base[1:].copy())
                G["M_bearing"].append(M_brg)
                G["F_retainer"].append([-self.sg.retainer(qG[j])[0] for j in range(z)])
                G["hmin_B"].append(qB[0] - np.hypot(qB[1], qB[2]) * prm.vp_r_out)
                G["qGdot"].append(qGdot.copy()); G["pd"].append(pd); G["ps"].append(ps)
                G["hpts_B"].append(self.vp.film_points(qB))
                G["F_fluid_vp"].append(stB.F_base + stB.C @ qBdot)
                G["Qc_vp"].append(stB.Q_contact.copy())
                G["vp_flows"].append([fl["case_inner"], fl["case_outer"], fl["kidney"].sum(),
                                      fl["discharge"], fl["suction"]])

                # ---------- field snapshots over the last revolution
                if opt.record_fields and k in k_fields:
                    pv, _, qrv, qtv = vp_fields(self.vp, stB, qBdot)
                    ps1, hs1, qrs1, qts1, pps1 = self.sg.fields(st_s1, qd_s1)
                    f32 = lambda a: np.asarray(a, dtype=np.float32)
                    FLD["phi"].append(phi); FLD["vp_h0"].append(qB[0])
                    FLD["vp_p"].append(f32(pv)); FLD["vp_qr"].append(f32(qrv)); FLD["vp_qt"].append(f32(qtv))
                    FLD["s1_p"].append(f32(ps1)); FLD["s1_h"].append(f32(hs1))
                    FLD["s1_qr"].append(f32(qrs1)); FLD["s1_qt"].append(f32(qts1))
                    FLD["s1_p_pocket"].append(pps1); FLD["s1_phi_piston"].append(pf["phi"][0, 0])
                    if self.pg is not None:
                        pk, hk, qzk, qzpk, zk = self.pg.fields(st_k1, qd_k1)
                        FLD["k1_p"].append(f32(pk)); FLD["k1_h"].append(f32(hk))
                        FLD["k1_qz"].append(f32(qzk)); FLD["k1_qz_p"].append(f32(qzpk)); FLD["k1_z"].append(zk)

            # ---------- chamber ODE with leakage (slipper scaled with chamber pressure)
            # slipper leakage follows the instantaneous pressure difference to the case
            # (held where the reference difference is below 1 bar)
            big = np.abs(p_ref_G) > 1e5
            scale = np.where(big, np.clip((p - prm.p_case) / np.where(big, p_ref_G, 1.0), -2.0, 2.0), 1.0)
            q_leak = Q_SG * scale + Q_SB + K_D * (p - prm.p_case) + K_rest
            dp_dt, dpd_dt, dps_dt, f = rhs(p, pd, ps, g.V[:, i], g.dVdt[:, i], g.A_d[:, i], g.A_s[:, i], prm, q_leak)
            H["p"][:, k] = p; H["p_d"][k] = pd; H["p_s"][k] = ps
            H["q_out"][k] = f["q_out"]; H["q_load"][k] = f["q_load"]; H["q_leak"][:, k] = q_leak
            H["q_port"][:, k] = f["q_d"] + f["q_s"]
            p = np.maximum(p + dp_dt * dt, P_FLOOR)
            pd = pd + dpd_dt * dt
            ps = max(ps + dps_dt * dt, P_FLOOR)

            if opt.verbose and i == steps - 1:
                r = k // steps
                sl = slice(r * steps, (r + 1) * steps)
                n_rev = int(round(360 / opt.gap_step_deg))
                qBs = np.array(G["qB"][-n_rev:])
                qGs = np.array(G["qG"][-n_rev:])
                QK = np.array(G["QSK"][-n_rev:]).sum(1).mean() * LPM if G["QSK"] else 0.0
                print(f"  rev {r + 1}/{opt.revolutions}: Pd={H['p_d'][sl].mean() / 1e5:7.2f} bar  "
                      f"Q_out={H['q_out'][sl].mean() * LPM:7.2f} LPM  leak={H['q_leak'][:, sl].sum(0).mean() * LPM:6.3f} LPM "
                      f"(piston {QK:6.3f})  h0_B={qBs[:, 0].mean() * 1e6:6.3f} um  h0_G1={qGs[:, 0, 0].mean() * 1e6:6.2f} um  "
                      f"({time.time() - t0:.0f} s)", flush=True)

        G = {k: np.array(v) for k, v in G.items()}
        G["p_pocket"] = G["p_pocket"].reshape(-1, z); G["F_contact_G"] = G["F_contact_G"].reshape(-1, z)
        G["hmin_G"] = G["hmin_G"].reshape(-1, z)
        FLD = {k: np.array(v) for k, v in FLD.items()}
        FLD.update(vp_r=self.vp.g.r, vp_th=self.vp.g.th, s1_r=self.sg.g.r, s1_th=self.sg.g.th,
                   vp_open=np.asarray(self.vp.rings | self.vp.discharge | self.vp.suction))
        if self.pg is not None:
            FLD["k1_th"] = np.linspace(0.0, 2.0 * np.pi, prm.piston_Nth, endpoint=False)
        return dict(grid=g, steps=steps, revs=opt.revolutions, dt=dt, fine=H, gap=G, fields=FLD,
                    phi_gap=g.phi[np.mod(G["k"], steps)], opt=opt, prm=prm)


def summarize(res):
    prm, steps, H, G = res["prm"], res["steps"], res["fine"], res["gap"]
    sl = slice((res["revs"] - 1) * steps, res["revs"] * steps)
    mk = G["k"] >= sl.start
    Q_SG = G["QSG"][mk].sum(1).mean() * LPM
    Q_SB = G["QSB"][mk].sum(1).mean() * LPM
    out = dict(
        Pd=H["p_d"][sl].mean() / 1e5, Q_out=H["q_out"][sl].mean() * LPM, Q_load=H["q_load"][sl].mean() * LPM,
        leak_total=H["q_leak"][:, sl].sum(0).mean() * LPM, Q_SG=Q_SG, Q_SB=Q_SB,
        Q_case_vp=G["Q_case_vp"][mk].mean() * LPM,
        h0_B=G["qB"][mk, 0].mean() * 1e6, hmin_B=G["hmin_B"][mk].min() * 1e6,
        thx_B=G["qB"][mk, 1].mean() * 1e3, thy_B=G["qB"][mk, 2].mean() * 1e3,
        F_contact_vp=G["F_contact_vp"][mk].mean(), lift_over_clamp=(G["F_lift"][mk] / G["F_clamp"][mk]).mean(),
        h0_G1=G["qG"][mk, 0, 0].mean() * 1e6, hmin_G=G["hmin_G"][mk].min() * 1e6,
        F_contact_G=G["F_contact_G"][mk].mean(),
        M_bearing=np.hypot(*G["M_bearing"][mk].T).mean(),
        F_retainer_max=G["F_retainer"][mk].max(),
    )
    if G["QSK"].size:
        out.update(Q_SK=G["QSK"][mk].sum(1).mean() * LPM, hmin_K=G["hmin_K"][mk].min() * 1e6,
                   F_contact_K=G["F_contact_K"][mk].mean(), F_fric_K=np.abs(G["F_fric_K"][mk]).mean())
    return out
