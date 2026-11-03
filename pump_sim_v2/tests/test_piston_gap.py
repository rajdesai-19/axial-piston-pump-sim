"""Formula verification of the piston / bore gap against closed-form annular-gap solutions."""
import numpy as np
import pytest

from pump_sim_v2.params import DEFAULT
from pump_sim_v2.gaps.piston import PistonGap

PRM = DEFAULT
MU, C0, D = PRM.mu, PRM.piston_clearance, PRM.dk


@pytest.fixture(scope="module")
def pg():
    return PistonGap(PRM)


def test_concentric_poiseuille(pg):
    phi = pg.fr.phi_bdc                                   # v_z = 0 at a dead centre
    st = pg.evaluate(phi, np.zeros(4), 300e5)
    assert abs(st.vz) < 1e-9
    L = pg.L_ref
    Q = np.pi * D * C0**3 * (300e5 - PRM.p_case) / (12 * MU * L)
    assert pg.leakage(st, 300e5) == pytest.approx(Q, rel=1e-9)
    assert st.Q_case == pytest.approx(Q, rel=1e-9)         # no squeeze: what enters leaves to the case


@pytest.mark.parametrize("eps", [0.3, 0.6, 0.9])
def test_eccentric_poiseuille(pg, eps):
    phi = pg.fr.phi_bdc
    q = np.array([eps * C0, 0.0, eps * C0, 0.0])         # parallel eccentric piston
    st = pg.evaluate(phi, q, 300e5)
    Q = np.pi * D * C0**3 * (300e5 - PRM.p_case) / (12 * MU * pg.L_ref) * (1 + 1.5 * eps**2)
    assert pg.leakage(st, 300e5) == pytest.approx(Q, rel=1e-9)
    # linear axial pressure, independent of th: no lateral pressure force
    assert np.abs(st.F_lat).max() < 1e-9 * (300e5 * D * pg.L_ref)


def test_concentric_couette(pg):
    phi = np.radians(80.0)
    st = pg.evaluate(phi, np.zeros(4), PRM.p_case)          # no pressure difference
    vz = st.vz
    assert vz > 1.0
    assert pg.leakage(st, PRM.p_case) == pytest.approx(-np.pi * D * C0 * vz / 2, rel=1e-9)
    L = st.g.z1 - st.g.z0
    assert st.F_friction == pytest.approx(-MU * vz / C0 * np.pi * D * L, rel=1e-9)      # Petroff shear


def test_squeeze_translation_finite_length(pg):
    """Concentric piston translating sideways, finite length, zero pressure at both ends:
    p = A(z) cos th,  A'' - A/r^2 = -12 mu e'/c^3  ->  C = -12 pi mu r^3/c^3 (L - 2 r tanh(L/(2r)))."""
    phi = pg.fr.phi_bdc
    st = pg.evaluate(phi, np.zeros(4), PRM.p_case)
    v = np.array([1.0, 0.0, 1.0, 0.0])
    r, L = 0.5 * D, pg.L_ref
    C_exact = -12 * np.pi * MU * r**3 / C0**3 * (L - 2 * r * np.tanh(L / (2 * r)))
    assert v @ st.C @ v == pytest.approx(C_exact, rel=1e-2)


def test_squeeze_matrix_symmetric_negative_definite(pg):
    q = np.array([4e-6, -2e-6, -5e-6, 3e-6])
    st = pg.evaluate(np.radians(60.0), q, 320e5)
    C = st.C
    assert np.allclose(C, C.T, rtol=1e-6, atol=1e-9 * np.abs(C).max())
    assert np.all(np.linalg.eigvalsh(-0.5 * (C + C.T)) > 0)


def test_mass_conservation_with_motion(pg):
    q = np.array([4e-6, -2e-6, -5e-6, 3e-6])
    qd = np.array([0.02, -0.01, 0.03, 0.005])               # [m/s]
    st = pg.evaluate(np.radians(60.0), q, 320e5)
    g, op = st.g, st.op
    p = st.p_base + np.tensordot(qd, st.p_sq, axes=1)
    U = -0.5 * st.vz
    Q_in, Q_out = op.boundary_flows(p, MU, U)
    dhdt = np.tensordot(qd, pg.W(g), axes=1)
    dV = np.sum(dhdt[1:-1]) * g.dz * g.ds                   # film volume change of the PDE cells
    assert Q_in - Q_out == pytest.approx(dV, rel=1e-8, abs=1e-15)
    # linear leakage decomposition reproduces the direct flux (plus the bore-annulus correction)
    corr = -np.pi * D * C0 * st.vz
    assert pg.leakage(st, 320e5, qd) == pytest.approx(Q_in + corr, rel=1e-9)


def test_external_load_is_rigid_body_resultant(pg):
    phi = np.radians(40.0)
    Fb = np.array([1500.0, -300.0])
    Q = pg.external_load(phi, Fb)
    loads = pg.lateral_loads(phi, Fb)
    Fr = sum(l[0] for l in loads); Ft = sum(l[1] for l in loads)
    assert Q[0] + Q[2] == pytest.approx(Fr) and Q[1] + Q[3] == pytest.approx(Ft)
    # moment about the end face: sum F zeta = -L_ref * Q_B
    assert -pg.L_ref * Q[2] == pytest.approx(sum(l[0] * l[2] for l in loads))
    # total = ball force + centrifugal of piston and slipper
    e_r, e_t = pg.local_axes(phi)
    F_cf = PRM.m_k * PRM.omega**2 * PRM.Rp
    assert Fr == pytest.approx(Fb @ e_r[:2] + F_cf) and Ft == pytest.approx(Fb @ e_t[:2])


def test_tilted_piston_wedge_pressure(pg):
    """Concentric at the end face, displaced at B: sliding builds pressure in the converging film
    (sign check of the wedge term) and generates a restoring lateral force."""
    phi = np.radians(80.0)                                  # v_z > 0: piston moves into the bore
    q = np.array([0.0, 0.0, 6e-6, 0.0])                     # axis tilted, film closes at th = 0 towards the case end
    st = pg.evaluate(phi, q, PRM.p_case)
    assert st.vz > 0
    # bore slides to -zeta relative to the piston: at th = 0 the film converges in -zeta -> p > 0
    j0 = 0
    assert st.p_base[1:-1, j0].max() > 0
    assert st.p_base[1:-1, st.g.Nth // 2].min() < 0
