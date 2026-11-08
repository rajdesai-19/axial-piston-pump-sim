"""
gaps/piston.py - Piston / bore lubricating gap (isothermal), simulation frame.

The film is unwrapped on the piston surface over the instantaneous engagement length
l_F(phi) = z_bj + l_K - z_block_front (frame.engagement), from the bore exit at the block front
(case pressure) to the piston end face (chamber pressure). Piston-fixed axial coordinate
  zeta = z - z_P in [-l_F, 0],   z_P = z_bj + l_K (piston end face), +zeta towards the chamber.
Local radial frame of piston j (rotates with the block; right-handed with +z):
  e_r = (cos phi, -sin phi, 0)  radially outward,  e_t = z x e_r = (sin phi, cos phi, 0),
th measured from e_r towards e_t, n(th) = cos th e_r + sin th e_t.

State q = (eA_r, eA_t, eB_r, eB_t): lateral displacement of the piston axis at two piston-fixed
points, A = end face (zeta = 0) and B = zeta = -L_ref (L_ref = shortest engagement, at BDC):
  u(zeta) = eA (1 + zeta/L_ref) - eB zeta/L_ref,     h = c - u(zeta) . n(th),
  W = dh/dq = ( -cos th (1 + zeta/L_ref), -sin th (1 + zeta/L_ref), cos th zeta/L_ref, sin th zeta/L_ref ).
Reynolds in the piston frame: the bore slides at -v_z, mean velocity U_z = -v_z/2, dh/dt|zeta = W . qdot
(the tilted piston sliding in the bore gives the wedge term d(h U_z)/dzeta = (v_z/2) gamma . n).
Virtual work (rigid piston, ball joint transmits forces only):
  fluid     Q_i = int (p - p_c) W_i dA + friction moment of the axial wall shear
            tau = -mu v_z / h - (h/2) dp/dzeta  (on the piston, +z): a tilt d gamma moves a
            surface point axially by -r n . d gamma  ->  Q_A += -int tau r n / L_ref dA,  Q_B = -that
  external  Q = sum_k F_k,lat . du(zeta_k)/dq for the lateral forces on the piston:
            at the ball joint (zeta = -l_K): F_SK,xy + F_TG,xy + slipper centrifugal m_s w^2 Rp e_r
            at the piston CoM:                piston centrifugal m_p w^2 Rp e_r
  contact   quadratic penalty below piston_h_contact
Squeeze matrix C_ij = int p_sq,j W_i dA (symmetric, negative definite).

Leakage of chamber j (Q_SK,j), as seen by the chamber model (which uses the piston area Ak):
  Q_SK = (flow from the chamber into the film across the moving end face, piston frame)
         - pi d c v_z   (bore annulus above the end face, not part of Ak * stroke)
       = bore-frame flux  -int (-h^3/12mu dp/dz + h v_z/2) ds  at the end face.
It is linear in the chamber pressure for a given film:  Q_SK = Q_D (p - p_c) + Q_w + dQ . qdot.
"""
from dataclasses import dataclass
import numpy as np

from ..params import PumpParams, DEFAULT
from ..frame import SwashFrame
from .cylinder import CylGrid, CylReynolds


@dataclass
class PistonState:
    op: CylReynolds
    g: CylGrid
    h: np.ndarray
    vz: float
    p_base: np.ndarray     # gauge film pressure for qdot = 0 (unclipped)
    p_sq: np.ndarray       # (4, Nz, Nth) squeeze basis fields
    F_base: np.ndarray     # (4,) fluid generalized force for qdot = 0 (cavitation-clipped) incl. friction moment
    C: np.ndarray          # (4, 4) squeeze matrix
    Q_contact: np.ndarray  # (4,) contact generalized force
    F_contact: float       # magnitude of the summed contact force [N]
    Q_D: float             # leakage per Pa of chamber gauge pressure [m^3/s/Pa]
    Q_w: float             # pressure-independent leakage (Couette, wedge) [m^3/s]
    dQ: np.ndarray         # (4,) d Q_SK / d qdot
    Q_case: float          # flow into the case at the bore exit for qdot = 0 (piston frame) [m^3/s]
    F_friction: float      # viscous axial friction on the piston (+z) for qdot = 0 [N]
    F_lat: np.ndarray      # (2,) fluid lateral force on the piston (e_r, e_t) for qdot = 0 [N]


class PistonGap:
    def __init__(self, prm: PumpParams = DEFAULT):
        self.prm = prm
        self.fr = SwashFrame(prm)
        self.r = 0.5 * prm.dk
        self.c = prm.piston_clearance
        self.L_ref = float(self.fr.engagement(self.fr.phi_bdc))
        self.zeta_ball = -prm.piston_l_K
        self.zeta_cg = (prm.z_cg_piston_drawing - prm.z_ball_drawing) - prm.piston_l_K

    # ------------------------------------------------------------- geometry
    def local_axes(self, phi_j):
        e_r = np.array([np.cos(phi_j), -np.sin(phi_j), 0.0])
        e_t = np.array([np.sin(phi_j), np.cos(phi_j), 0.0])
        return e_r, e_t

    def grid(self, phi_j):
        L = float(self.fr.engagement(phi_j))
        return CylGrid(self.r, -L, 0.0, self.prm.piston_Nz, self.prm.piston_Nth)

    def shape(self, zeta):
        """Axis displacement weights (a, b): u(zeta) = a eA + b eB."""
        return 1.0 + zeta / self.L_ref, -zeta / self.L_ref

    def W(self, g: CylGrid):
        a, b = self.shape(g.Z)
        c, s = np.cos(g.TH), np.sin(g.TH)
        return np.stack([-c * a, -s * a, -c * b, -s * b])

    def axis_offset(self, q, zeta):
        """Lateral axis displacement (u_r, u_t) at piston-fixed zeta [m]."""
        a, b = self.shape(zeta)
        return a * q[0] + b * q[2], a * q[1] + b * q[3]

    def film_true(self, q, g: CylGrid):
        ur, ut = self.axis_offset(q, g.Z)
        return self.c - ur * np.cos(g.TH) - ut * np.sin(g.TH)

    def film(self, q, g: CylGrid):
        return np.maximum(self.film_true(q, g), self.prm.piston_h_min)

    def eccentricity_ends(self, q, phi_j):
        """Eccentricity vectors (e_r, e_t) at the bore exit (case end) and at the end face [m]."""
        L = float(self.fr.engagement(phi_j))
        return np.array(self.axis_offset(q, -L)), np.array(self.axis_offset(q, 0.0))

    def film_points(self, q, phi_j, angles_deg=(0.0, 90.0, 180.0, 270.0)):
        """Film at the case end and at the chamber end at the given th (from e_r towards e_t) [m], (2, n)."""
        a = np.radians(angles_deg)
        eC, eD = self.eccentricity_ends(q, phi_j)
        return np.array([self.c - e[0] * np.cos(a) - e[1] * np.sin(a) for e in (eC, eD)])

    def h_min(self, q, phi_j):
        eC, eD = self.eccentricity_ends(q, phi_j)
        return self.c - max(np.hypot(*eC), np.hypot(*eD))

    # ------------------------------------------------------------- solve
    def evaluate(self, phi_j, q, p_ch) -> PistonState:
        prm = self.prm
        mu = prm.mu
        g = self.grid(phi_j)
        h = self.film(q, g)
        op = CylReynolds(g, h)
        vz = float(self.fr.vz(phi_j))
        U = -0.5 * vz
        W = self.W(g)
        zeros = np.zeros_like(h)

        p_D = op.solve(zeros, 0.0, 1.0)                               # unit chamber pressure
        p_w = op.solve(12.0 * mu * op.couette_rhs(U))                 # wedge (sliding of the tilted piston)
        p_s = np.stack([op.solve(12.0 * mu * W[i]) for i in range(4)])

        dp = p_ch - prm.p_case
        p_base = dp * p_D + p_w
        wA = g.dA
        p_clip = np.maximum(p_base, prm.p_cavitation - prm.p_case)
        F_p = np.array([np.sum(p_clip * W[i] * wA) for i in range(4)])
        C = np.array([[np.sum(p_sq_j * W[i] * wA) for p_sq_j in p_s] for i in range(4)])

        # axial wall shear on the piston and its tilting (friction) moment
        tau = -mu * vz / h - 0.5 * h * np.gradient(p_clip, g.dz, axis=0)
        F_fric = float(np.sum(tau * wA))
        mA = -self.r / self.L_ref * np.array([np.sum(tau * np.cos(g.TH) * wA), np.sum(tau * np.sin(g.TH) * wA)])
        F_base = F_p + np.concatenate([mA, -mA])

        # leakage out of the chamber (bore-frame), linear in dp and qdot
        corr = -np.pi * prm.dk * self.c * vz
        Q_D = op.boundary_flows(p_D, mu)[0]
        Qw_ch, _ = op.boundary_flows(p_w, mu, U)
        Q_w = Qw_ch + corr
        dQ = np.array([op.boundary_flows(p_s[i], mu)[0] for i in range(4)])
        Q_case = op.boundary_flows(p_base, mu, U)[1]

        # asperity contact (quadratic penalty)
        h_true = self.film_true(q, g)
        pen = np.clip((prm.piston_h_contact - h_true) / prm.piston_h_contact, 0.0, None)
        p_cont = prm.piston_p_contact * pen**2
        Q_contact = np.array([np.sum(p_cont * W[i] * wA) for i in range(4)])
        F_cont = float(np.hypot(Q_contact[0] + Q_contact[2], Q_contact[1] + Q_contact[3]))

        F_lat = np.array([F_p[0] + F_p[2], F_p[1] + F_p[3]])
        return PistonState(op, g, h, vz, p_base, p_s, F_base, C, Q_contact, F_cont,
                           Q_D, Q_w, dQ, Q_case, F_fric, F_lat)

    # ------------------------------------------------------------- loads
    def lateral_loads(self, phi_j, F_ball_xy):
        """
        Lateral loads on the piston in (e_r, e_t): [(F_r, F_t, zeta), ...].
        F_ball_xy: global (x, y) force at the ball joint from the slipper (F_SK + F_TG).
        """
        prm = self.prm
        e_r, e_t = self.local_axes(phi_j)
        Fb = np.array([F_ball_xy[0], F_ball_xy[1], 0.0])
        F_cf = prm.omega**2 * prm.Rp
        return [(Fb @ e_r + prm.m_slipper * F_cf, Fb @ e_t, self.zeta_ball),
                (prm.m_piston * F_cf, 0.0, self.zeta_cg)]

    def external_load(self, phi_j, F_ball_xy):
        Q = np.zeros(4)
        for Fr, Ft, zk in self.lateral_loads(phi_j, F_ball_xy):
            a, b = self.shape(zk)
            Q += np.array([a * Fr, a * Ft, b * Fr, b * Ft])
        return Q

    def leakage(self, st: PistonState, p_ch, qdot=np.zeros(4)):
        """Chamber leakage Q_SK [m^3/s] (> 0 out of the chamber) at chamber pressure p_ch."""
        return float(st.Q_D * (p_ch - self.prm.p_case) + st.Q_w + st.dQ @ qdot)

    def fields(self, st: PistonState, qdot=np.zeros(4)):
        """
        Film fields on the unwrapped piston (rows: zeta from the bore exit to the end face,
        columns: th from e_r): gauge pressure [Pa] (cavitation-clipped), film [m], axial flux per
        unit width q_z = -h^3/(12 mu) dp/dz + h U (bore frame, U = v_z/2) [m^2/s] (+ towards the
        chamber), pressure-driven part, and the axial coordinate zeta [m].
        """
        g, mu = st.g, self.prm.mu
        p = st.p_base + np.tensordot(qdot, st.p_sq, axes=1)
        p = np.maximum(p, self.prm.p_cavitation - self.prm.p_case)
        q_p = -st.h**3 / (12.0 * mu) * np.gradient(p, g.dz, axis=0)
        return p, st.h, q_p + 0.5 * st.vz * st.h, q_p, g.z
