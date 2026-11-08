"""
gaps/polar.py - Isothermal 2D Reynolds solver on an annulus (polar grid), shared by
the slipper and the cylinder-block / valve-plate gaps.

  d/dr( r h^3 dp/dr )/r + d/dth( h^3 dp/dth )/r^2 = 12 mu ( U_mean . grad h + dh/dt )

U_mean is the mean velocity of the two surfaces (Couette term). Nodes in `dirichlet`
have prescribed pressure (rings, pockets, ports, kidneys); all other nodes follow the PDE.
Periodic in theta; the first and last radial rows must be Dirichlet.

Everything is linear in the RHS, so the caller builds pressure fields by superposition of
basis solutions with one LU factorisation per film geometry. Flows between nodes are
evaluated with the same finite-volume fluxes as the operator, so mass is conserved
discretely (Poiseuille + Couette across every cell face).
"""
from dataclasses import dataclass
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu


@dataclass
class PolarGrid:
    r_in: float
    r_out: float
    Nr: int
    Nth: int

    def __post_init__(self):
        self.r = np.linspace(self.r_in, self.r_out, self.Nr)
        self.th = np.linspace(0.0, 2.0 * np.pi, self.Nth, endpoint=False)
        self.dr = (self.r_out - self.r_in) / (self.Nr - 1)
        self.dth = 2.0 * np.pi / self.Nth
        self.R, self.TH = np.meshgrid(self.r, self.th, indexing="ij")
        self.X = self.R * np.cos(self.TH)
        self.Y = self.R * np.sin(self.TH)
        wr = np.full(self.Nr, self.dr)
        wr[0] = wr[-1] = 0.5 * self.dr
        self.dA = self.R * wr[:, None] * self.dth          # trapezoidal area weights


class ReynoldsOperator:
    """LU-factorised operator for one film thickness field h and one Dirichlet set."""

    def __init__(self, g: PolarGrid, h: np.ndarray, dirichlet: np.ndarray):
        if not (dirichlet[0].all() and dirichlet[-1].all()):
            raise ValueError("first and last radial rows must be Dirichlet")
        self.g, self.h, self.dir = g, h, dirichlet
        Nr, Nt = g.Nr, g.Nth
        R = g.R
        h3 = h ** 3
        ip = np.minimum(np.arange(Nr) + 1, Nr - 1)
        im = np.maximum(np.arange(Nr) - 1, 0)
        jp = (np.arange(Nt) + 1) % Nt
        jm = (np.arange(Nt) - 1) % Nt
        # face values (radial faces at r +/- dr/2, theta faces at th +/- dth/2)
        self.h3_rp = 0.5 * (h3 + h3[ip, :]); self.h3_rm = 0.5 * (h3 + h3[im, :])
        self.h3_tp = 0.5 * (h3 + h3[:, jp]); self.h3_tm = 0.5 * (h3 + h3[:, jm])
        self.r_p = 0.5 * (R + R[ip, :]); self.r_m = 0.5 * (R + R[im, :])
        self.cr_p = self.r_p * self.h3_rp / (R * g.dr**2)
        self.cr_m = self.r_m * self.h3_rm / (R * g.dr**2)
        self.ct_p = self.h3_tp / (R**2 * g.dth**2)
        self.ct_m = self.h3_tm / (R**2 * g.dth**2)
        self._ip, self._im, self._jp, self._jm = ip, im, jp, jm

        idx = np.arange(Nr * Nt).reshape(Nr, Nt)
        inner = ~dirichlet
        rows = [idx.ravel()]
        cols = [idx.ravel()]
        diag = np.where(dirichlet, 1.0, -(self.cr_p + self.cr_m + self.ct_p + self.ct_m))
        data = [diag.ravel()]
        for coef, nb in ((self.cr_p, idx[ip, :]), (self.cr_m, idx[im, :]),
                         (self.ct_p, idx[:, jp]), (self.ct_m, idx[:, jm])):
            rows.append(idx[inner]); cols.append(nb[inner]); data.append(coef[inner])
        A = coo_matrix((np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
                       shape=(Nr * Nt, Nr * Nt)).tocsc()
        self.lu = splu(A)

    def solve(self, rhs_interior, dirichlet_values):
        """rhs_interior: (Nr, Nth) RHS of the PDE (12 mu (...)); dirichlet_values: (Nr, Nth)."""
        b = np.where(self.dir, dirichlet_values, rhs_interior)
        return self.lu.solve(b.ravel()).reshape(self.g.Nr, self.g.Nth)

    # ------------------------------------------------------------------ fluxes
    def face_flows(self, p, mu, U_r=None, U_t=None):
        """
        Volume flow through the +r and +theta faces of every cell [m^3/s]
        (positive outward in +r / +theta). Poiseuille plus Couette with mean
        surface velocity components U_r, U_t (arrays or None) at the faces.
        """
        g = self.g
        ip, jp = self._ip, self._jp
        q_r = -(self.r_p * g.dth) * self.h3_rp / (12.0 * mu) * (p[ip, :] - p) / g.dr
        q_t = -(g.dr) * self.h3_tp / (12.0 * mu) * (p[:, jp] - p) / (g.R * g.dth)
        if U_r is not None:
            h_rp = 0.5 * (self.h + self.h[ip, :])
            q_r = q_r + U_r * h_rp * self.r_p * g.dth
        if U_t is not None:
            h_tp = 0.5 * (self.h + self.h[:, jp])
            q_t = q_t + U_t * h_tp * g.dr
        q_r[-1, :] = 0.0                                    # no face beyond the outer row
        return q_r, q_t

    def region_outflow(self, region, q_r, q_t, into=None):
        """
        Net flow leaving the node set `region` (bool, Nr x Nth) across faces whose
        neighbour is outside the region and inside `into` (default: the PDE nodes,
        i.e. flow into the lubricating film only; region-to-region contacts such as
        kidney/port overlaps are open passages handled by the orifice model).
        """
        into = ~self.dir if into is None else into
        ip, im, jp, jm = self._ip, self._im, self._jp, self._jm
        nb = ~region & into
        out = 0.0
        m = region & nb[ip, :]; m[-1, :] = False          # +r faces
        out += q_r[m].sum()
        m = region & nb[im, :]; m[0, :] = False           # -r faces (flow i -> i-1 = -q_r[i-1])
        out -= q_r[im, :][m].sum()
        m = region & nb[:, jp]                            # +theta faces
        out += q_t[m].sum()
        m = region & nb[:, jm]                            # -theta faces
        out -= q_t[:, jm][m].sum()
        return float(out)
