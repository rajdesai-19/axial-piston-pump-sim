import numpy as np
import pytest

from pump_sim_v2.gaps.polar import PolarGrid, ReynoldsOperator

MU = 0.0174


def _ring_dirichlet(g):
    d = np.zeros((g.Nr, g.Nth), bool)
    d[0] = d[-1] = True
    return d


def test_annulus_poiseuille_leakage():
    g = PolarGrid(7.0e-3, 9.0e-3, 41, 48)
    h = np.full((g.Nr, g.Nth), 10e-6)
    op = ReynoldsOperator(g, h, _ring_dirichlet(g))
    dv = np.zeros_like(h); dv[0] = 100e5
    p = op.solve(np.zeros_like(h), dv)
    q_r, q_t = op.face_flows(p, MU)
    inner = np.zeros_like(h, bool); inner[0] = True
    Q = op.region_outflow(inner, q_r, q_t)
    Q_exact = np.pi * 10e-6**3 * 100e5 / (6 * MU * np.log(9.0 / 7.0))
    assert Q == pytest.approx(Q_exact, rel=2e-3)
    outer = np.zeros_like(h, bool); outer[-1] = True
    assert op.region_outflow(outer, q_r, q_t) == pytest.approx(-Q, rel=1e-9)   # conservation


def test_annulus_squeeze_force():
    ri, ro, h0, hdot = 7.0e-3, 9.0e-3, 10e-6, 1e-3
    g = PolarGrid(ri, ro, 81, 24)
    h = np.full((g.Nr, g.Nth), h0)
    op = ReynoldsOperator(g, h, _ring_dirichlet(g))
    p = op.solve(np.full_like(h, 12 * MU * hdot), np.zeros_like(h))
    F = np.sum(p * g.dA)
    # analytic: p = s r^2/4 + A ln r + B, s = 12 mu hdot / h^3, p(ri) = p(ro) = 0
    s = 12 * MU * hdot / h0**3
    A = -s * (ro**2 - ri**2) / (4 * np.log(ro / ri))
    B = -s * ri**2 / 4 - A * np.log(ri)
    r = np.linspace(ri, ro, 20001)
    F_exact = np.trapezoid((s * r**2 / 4 + A * np.log(r) + B) * 2 * np.pi * r, r)
    assert F == pytest.approx(F_exact, rel=2e-3)
    assert F < 0                              # separating surfaces -> suction
