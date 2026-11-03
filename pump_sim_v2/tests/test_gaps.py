import numpy as np
import pytest

from pump_sim_v2.params import DEFAULT
from pump_sim_v2.gaps.valveplate import ValvePlateGap
from pump_sim_v2.gaps.slipper import SlipperGap

PRM = DEFAULT.with_(vp_Nr=21, vp_Ntheta=180)          # coarse grid: fast tests


@pytest.fixture(scope="module")
def vp():
    return ValvePlateGap(PRM)


def _vp_state(vp, q=(6e-6, 1e-5, -2e-5), phi=0.3):
    phj = vp.fr.piston_phi(phi)
    p_ch = np.where(np.mod(phj, 2 * np.pi) < np.pi, 350e5, 10e5)
    return vp.evaluate(phi, np.array(q), p_ch, 352e5, 9.9e5)


def test_vp_squeeze_matrix_symmetric_negative_definite(vp):
    st = _vp_state(vp)
    C = st.C
    assert np.allclose(C, C.T, rtol=1e-3, atol=1e-6 * np.abs(C).max())
    assert np.all(np.linalg.eigvalsh(-0.5 * (C + C.T)) > 0)


def test_vp_mass_conservation(vp):
    st = _vp_state(vp)
    f = vp.flows(st)
    out = f["kidney"].sum() + f["discharge"] + f["suction"]
    case = f["case_inner"] + f["case_outer"]
    assert out == pytest.approx(case, rel=1e-6, abs=1e-12)
    assert case > 0


def test_vp_lift_positive_and_gauge(vp):
    st = _vp_state(vp)
    assert st.F_base[0] > 0
    # all boundaries at case pressure, no tilt, no motion -> no load at all (gauge integration)
    st0 = vp.evaluate(0.0, np.array([6e-6, 0, 0]), np.full(9, PRM.p_case), PRM.p_case, PRM.p_case)
    assert np.allclose(st0.F_base, 0.0, atol=1e-6)


@pytest.fixture(scope="module")
def sg():
    return SlipperGap(DEFAULT)


def test_slipper_hydrostatic_balance(sg):
    h0, p_ch = 10e-6, 300e5
    st = sg.evaluate(np.array([h0, 0, 0]), p_ch, 0.0)
    prm = DEFAULT
    G = np.pi * h0**3 / (6 * prm.mu * np.log(prm.slipper_r_out / prm.slipper_r_in))
    assert st.Q0 == pytest.approx(G * st.p_pocket, rel=2e-3)                     # land flow
    dp = p_ch - prm.p_case
    assert dp - st.p_pocket == pytest.approx(sg.R_lam * st.Q0 + sg.K_turb * st.Q0**2, rel=1e-9)
    # lift = pocket + land with log pressure profile
    ri, ro = prm.slipper_r_in, prm.slipper_r_out
    F_exact = st.p_pocket * np.pi * (ro**2 - ri**2) / (2 * np.log(ro / ri))
    assert st.F_base[0] == pytest.approx(F_exact, rel=2e-3)


def test_slipper_reverse_leakage_below_case(sg):
    st = sg.evaluate(np.array([10e-6, 0, 0]), 0.5e5, 0.0)          # chamber below case pressure
    assert st.Q0 < 0                                                 # oil flows from the case into the chamber
    assert DEFAULT.p_case + st.p_pocket < DEFAULT.p_case


def test_slipper_squeeze_matrix(sg):
    st = sg.evaluate(np.array([10e-6, 2e-4, -1e-4]), 300e5, 15.0)
    C = st.C
    assert np.all(np.linalg.eigvalsh(-0.5 * (C + C.T)) > 0)
    assert np.allclose(C, C.T, rtol=2e-2, atol=1e-3 * np.abs(C).max())


def test_three_point_films(sg, vp):
    q = np.array([8e-6, 2e-4, -1e-4])
    h = sg.film_points(q)
    assert h.mean() == pytest.approx(q[0])                       # mean of 0/120/240 deg = centre film
    assert h[0] == pytest.approx(q[0] + q[1] * DEFAULT.slipper_r_out)
    hB = vp.film_points(np.array([3e-6, 1e-5, -2e-5]))
    assert hB.mean() == pytest.approx(3e-6)


def test_field_flux_matches_leakage(sg):
    # concentric slipper at rest: radial flux integrated round the land = orifice/land leakage
    st = sg.evaluate(np.array([10e-6, 0, 0]), 300e5, 0.0)
    p, h, q_r, q_t, pp = sg.fields(st)
    g = sg.g
    i = g.Nr // 2
    Q = np.sum(q_r[i] * g.r[i] * g.dth)
    assert Q == pytest.approx(st.Q0, rel=5e-3)
    assert np.allclose(q_t, 0.0, atol=1e-12)


def test_slipper_friction_tips_leading_edge_down(sg):
    phi_j = 1.0
    Q = sg.external_load(phi_j, -4000.0, 20.0)
    Q_nofric = sg.external_load(phi_j, -4000.0, 0.0)
    assert Q[1] - Q_nofric[1] == pytest.approx(-DEFAULT.r_shoe * 20.0)
    assert Q[2] == pytest.approx(Q_nofric[2])
    n, xi, zeta, v = sg.local_axes(phi_j)
    assert abs(n @ xi) < 1e-12 and abs(np.linalg.norm(zeta) - 1) < 1e-12   # xi lies in the plane
