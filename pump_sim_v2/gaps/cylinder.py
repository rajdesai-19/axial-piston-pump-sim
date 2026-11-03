"""
gaps/cylinder.py - Isothermal 2D Reynolds solver on an unwrapped cylindrical film
(piston / bore gap), finite-volume form:

  d/dz( h^3 dp/dz ) + d/ds( h^3 dp/ds ) = 12 mu ( d(h U_z)/dz + dh/dt ),   s = r * th

z is the axial coordinate along the film (first row = case end, last row = chamber end, both
Dirichlet), th is periodic. U_z is the mean axial surface velocity (Couette); the Couette term
is discretised as the divergence of the face fluxes h_face * U_z, so the flows evaluated with
face_flows conserve mass exactly (Poiseuille + Couette across every cell face).

As in polar.py the problem is linear in the RHS and the Dirichlet values: the caller builds
fields by superposition of basis solutions with one LU factorisation per film geometry.
"""
from dataclasses import dataclass
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu


@dataclass
class CylGrid:
    radius: float          # film radius (piston radius) [m]
    z0: float              # case-end axial coordinate [m]
    z1: float              # chamber-end axial coordinate [m]
    Nz: int
    Nth: int

    def __post_init__(self):
        self.z = np.linspace(self.z0, self.z1, self.Nz)
        self.th = np.linspace(0.0, 2.0 * np.pi, self.Nth, endpoint=False)
        self.dz = (self.z1 - self.z0) / (self.Nz - 1)
        self.dth = 2.0 * np.pi / self.Nth
        self.ds = self.radius * self.dth
        self.Z, self.TH = np.meshgrid(self.z, self.th, indexing="ij")
        wz = np.full(self.Nz, self.dz)
        wz[0] = wz[-1] = 0.5 * self.dz
        self.dA = wz[:, None] * self.ds * np.ones((1, self.Nth))      # trapezoidal area weights


class CylReynolds:
    """LU-factorised operator for one film field h (Nz, Nth); rows 0 and Nz-1 are Dirichlet."""

    def __init__(self, g: CylGrid, h: np.ndarray):
        self.g, self.h = g, h
        Nz, Nt = g.Nz, g.Nth
        h3 = h ** 3
        jp = (np.arange(Nt) + 1) % Nt
        jm = (np.arange(Nt) - 1) % Nt
        self._jp, self._jm = jp, jm
        # face values: axial faces between rows i and i+1 (Nz-1, Nt); circumferential faces j+1/2 (Nz, Nt)
        self.h3_z = 0.5 * (h3[1:] + h3[:-1])
        self.h_z = 0.5 * (h[1:] + h[:-1])
        self.h3_t = 0.5 * (h3 + h3[:, jp])
        cz = self.h3_z / g.dz**2
        ct = self.h3_t / g.ds**2
        idx = np.arange(Nz * Nt).reshape(Nz, Nt)
        I = idx[1:-1]
        czp, czm = cz[1:], cz[:-1]                    # (Nz-2, Nt) faces above / below interior rows
        ctp, ctm = ct[1:-1], ct[1:-1][:, jm]
        rows = [idx[[0, -1]].ravel(), I.ravel()]
        cols = [idx[[0, -1]].ravel(), I.ravel()]
        data = [np.ones(2 * Nt), -(czp + czm + ctp + ctm).ravel()]
        for coef, nb in ((czp, idx[2:]), (czm, idx[:-2]), (ctp, I[:, jp]), (ctm, I[:, jm])):
            rows.append(I.ravel()); cols.append(nb.ravel()); data.append(coef.ravel())
        A = coo_matrix((np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
                       shape=(Nz * Nt, Nz * Nt)).tocsc()
        self.lu = splu(A)

    def couette_rhs(self, U_z):
        """12 mu factor excluded: discrete d(h U_z)/dz at the interior rows (zero on the boundary rows)."""
        out = np.zeros_like(self.h)
        out[1:-1] = U_z * (self.h_z[1:] - self.h_z[:-1]) / self.g.dz
        return out

    def solve(self, rhs, p_case_end=0.0, p_chamber_end=0.0):
        """rhs: (Nz, Nth) PDE right-hand side (12 mu (...)); boundary pressures (scalars or (Nth,))."""
        b = rhs.copy()
        b[0] = p_case_end
        b[-1] = p_chamber_end
        return self.lu.solve(b.ravel()).reshape(self.h.shape)

    def axial_face_flow(self, p, mu, U_z=0.0):
        """Volume flow through every axial face (between rows i and i+1) of width ds, +z [m^3/s], (Nz-1, Nth)."""
        g = self.g
        return (-self.h3_z / (12.0 * mu) * (p[1:] - p[:-1]) / g.dz + self.h_z * U_z) * g.ds

    def boundary_flows(self, p, mu, U_z=0.0):
        """(flow out of the chamber end into the film, flow out of the film into the case end) [m^3/s]."""
        q = self.axial_face_flow(p, mu, U_z)
        return float(-q[-1].sum()), float(-q[0].sum())
