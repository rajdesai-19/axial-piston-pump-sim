"""
gaps/slipper.py - Slipper / swash plate lubricating gap (isothermal), PDF frame.

Local frame on the swash plate plane for each slipper:
  n    = swash plate normal (a, b, c), pointing from the plate towards the slipper/valve plate
  xi   = unit sliding direction of the slipper (ball-joint velocity, PDF p.14; lies in the plane)
  zeta = n x xi
Polar grid on the sealing land r_in..r_out with th measured from xi towards zeta.
Slipper state q = (h0, g_xi, g_zeta):  h = h0 + g_xi * xi + g_zeta * zeta,  W = (1, xi, zeta).
The slipper rotates about the ball-joint centre B (height H = r_shoe above the running face);
a tilt (g_xi, g_zeta) moves the face centre in-plane by H (g_xi xi + g_zeta zeta) and the CoM
(l_SG below B) by l_SG (g_xi xi + g_zeta zeta). Virtual work gives
  fluid    Q_i = int (p - p_c) W_i dA over land + pocket
  external Q  = ( F_n,  H F_face.xi + l_SG F_cm.xi,  H F_face.zeta + l_SG F_cm.zeta )
with F_n the normal force from the piston on the slipper (PDF FSK, negative = towards the
plate), F_face the in-plane viscous friction at the face, F_cm the slipper inertia
(-m_s a, includes the centrifugal force) at the CoM.
Reynolds: the plate moves at -v xi relative to the slipper, mean surface velocity -v/2 xi.
Pocket: fed from the chamber through the capillary + neck orifice in series,
  p_D - p_pocket = R_lam Q + K_turb Q^2,  Q = flow into the land + A_pocket * dh0/dt,
linearised in qdot (equivalent conductance K_eq) for the squeeze matrix.
Leakage of chamber j (Q_SG,j) = orifice flow Q.
"""
from dataclasses import dataclass
import numpy as np

from ..params import PumpParams, DEFAULT
from ..frame import SwashFrame
from .polar import PolarGrid, ReynoldsOperator


@dataclass
class SlipperState:
    op: ReynoldsOperator
    h: np.ndarray
    v: float
    p_pocket: float        # gauge pocket pressure for qdot = 0
    dp_pocket: np.ndarray  # (3,) d p_pocket / d qdot
    p_base: np.ndarray     # gauge land pressure for qdot = 0
    p_sq: np.ndarray       # (3, Nr, Nth) land squeeze fields incl. pocket response
    F_base: np.ndarray
    C: np.ndarray
    Q_contact: np.ndarray
    Q0: float              # orifice flow for qdot = 0 [m^3/s]
    dQ: np.ndarray         # (3,) d Q / d qdot
    F_friction: float      # viscous friction magnitude on the slipper face [N]


class SlipperGap:
    def __init__(self, prm: PumpParams = DEFAULT):
        self.prm = prm
        self.fr = SwashFrame(prm)
        self.g = PolarGrid(prm.slipper_r_in, prm.slipper_r_out, prm.slipper_Nr, prm.slipper_Nphi)
        g = self.g
        shp = (g.Nr, g.Nth)
        self.dir = np.zeros(shp, bool); self.dir[0] = self.dir[-1] = True
        self.inner = np.zeros(shp, bool); self.inner[0] = True
        self.W = np.stack([np.ones(shp), g.X, g.Y])
        self.A_pocket = np.pi * prm.slipper_r_in**2
        A1 = np.pi / 4 * prm.slipper_d1**2
        A2 = np.pi / 4 * prm.slipper_d2**2
        A_eq = 1.0 / np.sqrt(1.0 / A1**2 + 1.0 / A2**2)
        self.K_turb = prm.rho / (2.0 * (prm.slipper_cd * A_eq) ** 2)
        self.R_lam = 128.0 * prm.mu / np.pi * (prm.slipper_l1 / prm.slipper_d1**4 + prm.slipper_l2 / prm.slipper_d2**4)
        th = g.TH
        self.U_r_face = lambda v: -0.5 * v * np.cos(th)                    # at +r faces
        self.U_t_face = lambda v: 0.5 * v * np.sin(th + 0.5 * g.dth)        # at +theta faces

    # ------------------------------------------------------------- local frame
    def local_axes(self, phi_j):
        """(n, xi, zeta) unit vectors and sliding speed for piston angle phi_j."""
        a, b, c = self.fr.abc
        n = np.array([a, b, c])
        vx, vy = self.fr.vxy(phi_j)
        v = np.array([vx, vy, self.fr.vz(phi_j)])
        speed = np.linalg.norm(v)
        xi = v / speed
        zeta = np.cross(n, xi)
        return n, xi, zeta, speed

    def film(self, q):
        g = self.g
        return np.maximum(q[0] + q[1] * g.X + q[2] * g.Y, self.prm.slipper_h_min)

    def _flows(self, op, p, v=None):
        if v is None:
            q_r, q_t = op.face_flows(p, self.prm.mu)
        else:
            q_r, q_t = op.face_flows(p, self.prm.mu, U_r=self.U_r_face(v), U_t=self.U_t_face(v))
        return op.region_outflow(self.inner, q_r, q_t)

    # ------------------------------------------------------------- solve
    def evaluate(self, q, p_ch, speed) -> SlipperState:
        prm, g = self.prm, self.g
        mu = prm.mu
        h = self.film(q)
        op = ReynoldsOperator(g, h, self.dir)
        zeros = np.zeros_like(h)
        unit_in = zeros.copy(); unit_in[0] = 1.0

        # dh/dxi in polar form, mean velocity -v/2 along xi
        dhdr = np.gradient(h, g.dr, axis=0)
        dhdth = (np.roll(h, -1, axis=1) - np.roll(h, 1, axis=1)) / (2 * g.dth)
        dhdxi = np.cos(g.TH) * dhdr - np.sin(g.TH) / g.R * dhdth
        p_A = op.solve(zeros, unit_in)
        p_w = op.solve(12.0 * mu * (-0.5 * speed) * dhdxi, zeros)
        p_s = np.stack([op.solve(12.0 * mu * self.W[i], zeros) for i in range(3)])

        Q_A = self._flows(op, p_A)
        Q_w = self._flows(op, p_w, speed)
        Q_s = np.array([self._flows(op, p_s[i]) for i in range(3)])
        Q_s[0] += self.A_pocket                       # pocket volume change

        # pocket pressure (gauge), signed orifice law (flow reverses when the chamber is below case):
        #   dp - p = R_lam Q + K_turb Q|Q|,   Q = Q_A p + Q_w   (f(p) is strictly decreasing in p)
        dp = p_ch - prm.p_case
        f = lambda pp: dp - pp - self.R_lam * (Q_A * pp + Q_w) - self.K_turb * (Q_A * pp + Q_w) * abs(Q_A * pp + Q_w)
        p_zero = -Q_w / Q_A                            # pocket pressure with no orifice flow
        lo, hi = min(dp, p_zero), max(dp, p_zero)
        for _ in range(100):                           # bisection: robust for both flow directions
            mid = 0.5 * (lo + hi)
            if f(mid) > 0.0:
                lo = mid
            else:
                hi = mid
            if hi - lo < 1e-5:
                break
        pp = 0.5 * (lo + hi)
        Q0 = Q_A * pp + Q_w
        K_eq = 1.0 / (self.R_lam + 2.0 * self.K_turb * abs(Q0))
        dpp = -Q_s / (Q_A + K_eq)                      # d p_pocket / d qdot
        dQ = -K_eq * dpp                               # orifice flow change

        p_base = pp * p_A + p_w
        p_sq = p_s + dpp[:, None, None] * p_A
        wA = g.dA
        p_clip = np.maximum(p_base, prm.p_cavitation - prm.p_case)
        pocket_W = np.array([self.A_pocket, 0.0, 0.0])
        F_base = np.array([np.sum(p_clip * self.W[i] * wA) for i in range(3)]) + pp * pocket_W
        C = np.array([[np.sum(p_sq[j] * self.W[i] * wA) for j in range(3)] for i in range(3)])
        C[0, :] += self.A_pocket * dpp

        h_true = q[0] + q[1] * g.X + q[2] * g.Y
        pen = np.clip((prm.slipper_h_contact - h_true) / prm.slipper_h_contact, 0.0, None)
        p_cont = prm.slipper_p_contact * pen**2
        Q_contact = np.array([np.sum(p_cont * self.W[i] * wA) for i in range(3)])
        F_fric = float(np.sum(mu * speed / h * wA))
        return SlipperState(op, h, speed, pp, dpp, p_base, p_sq, F_base, C, Q_contact, Q0, dQ, F_fric)

    def external_load(self, phi_j, F_n, F_friction):
        """
        Generalized external load on q for one slipper.
        F_n: normal force from the piston on the slipper along n (PDF FSK = F_AK / c).
        F_friction: viscous friction magnitude (acts along -xi at the face).
        """
        prm = self.prm
        n, xi, zeta, _ = self.local_axes(phi_j)
        ax, ay = self.fr.axy(phi_j)
        acc = np.array([ax, ay, self.fr.az(phi_j)])
        F_cm = -prm.m_slipper * acc
        F_face = -F_friction * xi
        H, l_SG = prm.r_shoe, prm.slipper_l_SG
        return np.array([F_n,
                         H * F_face @ xi + l_SG * F_cm @ xi,
                         H * F_face @ zeta + l_SG * F_cm @ zeta])

    def retainer(self, q):
        """Hold-down plate: quadratic penalty pushing the slipper back when h0 > clearance
        (only if prm.slipper_use_retainer)."""
        if not self.prm.slipper_use_retainer:
            return np.zeros(3)
        d = q[0] - self.prm.slipper_retainer_clearance
        return np.array([-self.prm.slipper_K_retainer * d * d if d > 0 else 0.0, 0.0, 0.0])

    @staticmethod
    def velocities(st: SlipperState, Q_ext):
        return np.linalg.solve(-st.C, st.F_base + st.Q_contact + np.asarray(Q_ext))

    def leakage(self, st: SlipperState, qdot=np.zeros(3)):
        """Orifice (= chamber) leakage for the given qdot [m^3/s]."""
        return float(st.Q0 + st.dQ @ qdot)

    def fields(self, st: SlipperState, qdot=np.zeros(3)):
        """
        Land fields in the slipper frame (x = xi, y = zeta): gauge pressure [Pa], film [m],
        local leakage flux per unit width q = -h^3/(12 mu) grad p + h U_mean [m^2/s]
        (U_mean = -v/2 xi), and the gauge pocket pressure [Pa].
        """
        g, mu = self.g, self.prm.mu
        p = st.p_base + np.tensordot(qdot, st.p_sq, axes=1)
        p = np.maximum(p, self.prm.p_cavitation - self.prm.p_case)     # half-Sommerfeld cavitation limit
        p_pocket = st.p_pocket + st.dp_pocket @ qdot
        h = st.h
        dpdr = np.gradient(p, g.dr, axis=0)
        dpdth = (np.roll(p, -1, axis=1) - np.roll(p, 1, axis=1)) / (2 * g.dth)
        k = h**3 / (12.0 * mu)
        q_r = -k * dpdr - 0.5 * st.v * np.cos(g.TH) * h
        q_t = -k * dpdth / g.R + 0.5 * st.v * np.sin(g.TH) * h
        return p, h, q_r, q_t, float(p_pocket)

    def film_points(self, q, angles_deg=(0.0, 120.0, 240.0)):
        """Film thickness at the land outer radius at the given angles (from xi towards zeta) [m]."""
        r = self.prm.slipper_r_out
        a = np.radians(angles_deg)
        return q[0] + q[1] * r * np.cos(a) + q[2] * r * np.sin(a)
