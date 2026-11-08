"""
gaps/valveplate.py - Cylinder block / valve plate lubricating gap (isothermal), simulation frame.

Film on the annulus r_in..r_out of the valve plate face, stationary polar grid
(x = r cos th, y = r sin th; simulation frame). Block state q = (h0, theta_x, theta_y):
  h0      : film thickness at the axis
  theta_x : small block rotation about +x through the valve-plate centre
  theta_y : small block rotation about +y through the valve-plate centre
Rotating the block about +x moves a face point by dz = theta_x * y; +z closes the gap, so
  h(x, y) = h0 - theta_x * y + theta_y * x,   W = dh/dq = (1, -y, x).
Virtual work then gives consistent generalized forces:
  fluid    Q_i = int (p - p_c) W_i dA over the solid block face   (pressure pushes the block -z)
  external Q  = (-F_Bz, M_VPx, M_VPy)  (block clamping force, moments about the VP centre)
and the squeeze matrix C_ij = int p_sq,j W_i dA is symmetric negative definite.
Force equilibrium:  Q_fluid(q) + C qdot + Q_contact + Q_ext = 0.

Boundary conditions (Zecchi masking, gauge pressure w.r.t. p_c):
  inner/outer rings -> 0 (case), kidney j -> p_j - p_c, discharge port -> p_d - p_c,
  suction port -> p_s - p_c. The block rotates clockwise: U_block = -omega r e_theta,
  mean surface velocity -omega r / 2 e_theta (Couette).
Leakage of chamber j (Q_SB,j) is the net flow leaving kidney j into the film.
"""
from dataclasses import dataclass
import numpy as np
from matplotlib.path import Path

from ..params import PumpParams, DEFAULT
from ..frame import SwashFrame
from .polar import PolarGrid, ReynoldsOperator
from .cad import load_cad


@dataclass
class VPState:
    """Result of one gap evaluation at fixed geometry (q) and boundary pressures."""
    op: ReynoldsOperator
    h: np.ndarray
    kidneys: list          # bool masks per piston
    ports: tuple           # (discharge mask, suction mask)
    solid: np.ndarray      # block solid face (pressure acts on the block)
    p_base: np.ndarray     # gauge pressure for qdot = 0
    p_sq: np.ndarray       # (3, Nr, Nth) unit squeeze fields
    F_base: np.ndarray     # (3,) generalized fluid force for qdot = 0 (cavitation-clipped)
    C: np.ndarray          # (3, 3) squeeze matrix
    Q_contact: np.ndarray  # (3,) asperity contact generalized force


class ValvePlateGap:
    def __init__(self, prm: PumpParams = DEFAULT, cad=None):
        self.prm = prm
        self.fr = SwashFrame(prm)
        self.cad = load_cad() if cad is None else cad
        self.g = PolarGrid(prm.vp_r_in, prm.vp_r_out, prm.vp_Nr, prm.vp_Ntheta)
        g = self.g
        self.pts = np.column_stack([g.X.ravel(), g.Y.ravel()])
        shp = (g.Nr, g.Nth)
        self.discharge = self.cad.discharge.contains_points(self.pts).reshape(shp)
        self.suction = self.cad.suction.contains_points(self.pts).reshape(shp)
        self.rings = np.zeros(shp, bool)
        self.rings[0] = self.rings[-1] = True
        self.W = np.stack([np.ones(shp), -g.Y, g.X])
        self.U_t = -0.5 * prm.omega * g.R             # mean tangential surface velocity

    # ------------------------------------------------------------- geometry
    def film(self, q):
        g = self.g
        h = q[0] - q[1] * g.Y + q[2] * g.X
        return np.maximum(h, self.prm.vp_h_min)

    def kidney_masks(self, phi_shaft):
        g = self.g
        X, Y = g.X.ravel(), g.Y.ravel()
        masks = []
        for phj in np.atleast_1d(self.fr.piston_phi(phi_shaft)).ravel():
            k = self.cad.kidney_at(-phj)
            lo, hi = k.min(axis=0) - 1e-4, k.max(axis=0) + 1e-4
            cand = (X >= lo[0]) & (X <= hi[0]) & (Y >= lo[1]) & (Y <= hi[1])
            m = np.zeros(X.size, bool)
            m[cand] = Path(k).contains_points(self.pts[cand])
            masks.append(m.reshape(g.Nr, g.Nth))
        return masks

    # ------------------------------------------------------------- solve
    def evaluate(self, phi_shaft, q, p_ch, p_d, p_s) -> VPState:
        prm, g = self.prm, self.g
        pc = prm.p_case
        h = self.film(q)
        kid = self.kidney_masks(phi_shaft)
        any_k = np.zeros_like(self.rings)
        vals = np.zeros_like(h)
        vals[self.discharge] = p_d - pc
        vals[self.suction] = p_s - pc
        for j, m in enumerate(kid):
            vals[m] = p_ch[j] - pc
            any_k |= m
        vals[self.rings] = 0.0
        dirichlet = self.rings | self.discharge | self.suction | any_k
        op = ReynoldsOperator(g, h, dirichlet)

        dhdth = (np.roll(h, -1, axis=1) - np.roll(h, 1, axis=1)) / (2 * g.dth)
        rhs_w = 12.0 * prm.mu * self.U_t * dhdth / g.R
        p_base = op.solve(rhs_w, vals)
        p_sq = np.stack([op.solve(12.0 * prm.mu * self.W[i], np.zeros_like(h)) for i in range(3)])

        solid = ~any_k
        wA = solid * g.dA
        p_clip = np.maximum(p_base, prm.p_cavitation - pc)
        F_base = np.array([np.sum(p_clip * self.W[i] * wA) for i in range(3)])
        C = np.array([[np.sum(p_sq[j] * self.W[i] * wA) for j in range(3)] for i in range(3)])

        pen = np.clip((prm.vp_h_contact - (q[0] - q[1] * g.Y + q[2] * g.X)) / prm.vp_h_contact, 0.0, None)
        p_cont = prm.vp_p_contact * pen**2
        Q_contact = np.array([np.sum(p_cont * self.W[i] * wA) for i in range(3)])
        return VPState(op, h, kid, (self.discharge, self.suction), solid, p_base, p_sq, F_base, C, Q_contact)

    def film_points(self, q, angles_deg=(0.0, 120.0, 240.0)):
        """Film thickness at the outer land radius at the given angles (simulation x-y frame) [m]."""
        r = self.prm.vp_r_out
        a = np.radians(angles_deg)
        return q[0] - q[1] * r * np.sin(a) + q[2] * r * np.cos(a)

    @staticmethod
    def velocities(st: VPState, Q_ext):
        """Solve the quasi-static balance F_base + Q_contact + Q_ext + C qdot = 0 for qdot."""
        return np.linalg.solve(-st.C, st.F_base + st.Q_contact + np.asarray(Q_ext))

    def flows(self, st: VPState, qdot=np.zeros(3)):
        """Leakage flows [m^3/s] for the given qdot: per kidney (out of chamber), ports, case."""
        p = st.p_base + np.tensordot(qdot, st.p_sq, axes=1)
        q_r, q_t = st.op.face_flows(p, self.prm.mu, U_t=self.U_t)
        kid = np.array([st.op.region_outflow(m, q_r, q_t) for m in st.kidneys])
        inner = np.zeros_like(self.rings); inner[0] = True
        outer = np.zeros_like(self.rings); outer[-1] = True
        return dict(
            kidney=kid,
            discharge=st.op.region_outflow(st.ports[0] & ~np.any(st.kidneys, axis=0), q_r, q_t),
            suction=st.op.region_outflow(st.ports[1] & ~np.any(st.kidneys, axis=0), q_r, q_t),
            case_inner=-st.op.region_outflow(inner, q_r, q_t),
            case_outer=-st.op.region_outflow(outer, q_r, q_t),
        )


def vp_fields(gap: "ValvePlateGap", st: VPState, qdot=np.zeros(3)):
    """
    Node fields for plotting / export: gauge pressure [Pa] and local leakage flux per unit
    width q = -h^3/(12 mu) grad p + h U_mean  [m^2/s] (radial and tangential components).
    Flux is NaN on open regions (kidneys, ports, rings), where there is no film.
    """
    g, mu = gap.g, gap.prm.mu
    p = st.p_base + np.tensordot(qdot, st.p_sq, axes=1)
    p = np.maximum(p, gap.prm.p_cavitation - gap.prm.p_case)          # half-Sommerfeld cavitation limit
    h = st.h
    dpdr = np.gradient(p, g.dr, axis=0)
    dpdth = (np.roll(p, -1, axis=1) - np.roll(p, 1, axis=1)) / (2 * g.dth)
    k = h**3 / (12.0 * mu)
    q_r = -k * dpdr
    q_t = -k * dpdth / g.R + gap.U_t * h
    film = ~st.op.dir
    q_r = np.where(film, q_r, np.nan); q_t = np.where(film, q_t, np.nan)
    return p, h, q_r, q_t


def block_external_load(bf: dict, k: int):
    """Generalized external load on q = (h0, theta_x, theta_y) from forces/block.py at sample k."""
    return np.array([-bf["F_Bz"][k], bf["M_VPx_brg"][k], bf["M_VPy_brg"][k]])
