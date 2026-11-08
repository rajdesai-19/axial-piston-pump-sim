import numpy as np
import pytest

from APP_sim.params import DEFAULT
from APP_sim.chamber import port_areas, orifice_flow, rhs, build_grid


def test_port_table_windows():
    A_d, A_s = port_areas(np.array([90.0, 270.0, 0.0, 180.0, 360.0]), 1.0)
    assert A_d[0] == pytest.approx(1.0) and A_s[0] == 0.0          # discharge fully open
    assert A_s[1] == pytest.approx(1.0) and A_d[1] == 0.0          # suction fully open
    assert np.all(A_d[2:] == 0.0) and np.all(A_s[2:] == 0.0)       # dead centres closed
    ang = np.linspace(0, 360, 3601)
    A_d, A_s = port_areas(ang, 1.0)
    assert np.all(A_d * A_s == 0.0)                                 # never both open


def test_orifice_flow_sign():
    assert orifice_flow(2e5, 1e5, 0.7, 1e-6, 870.0) > 0
    assert orifice_flow(1e5, 2e5, 0.7, 1e-6, 870.0) < 0
    assert orifice_flow(1e5, 1e5, 0.7, 1e-6, 870.0) == 0


def test_rhs_conservation_and_rest_state():
    z = DEFAULT.num_piston
    p = np.full(z, 50e5)
    V = np.full(z, 1e-6); A = np.full(z, 1e-5)
    dp, dpd, dps, f = rhs(p, 50e5, 50e5, V, np.zeros(z), A, A, DEFAULT)
    assert np.allclose(dp, 0.0)                     # no pressure difference, no motion -> no change
    assert f["q_out"] == pytest.approx(-np.sum(f["q_d"]))
    # moving pistons only: dp/dt = -K/V dV/dt
    dVdt = np.linspace(-1e-4, 1e-4, z)
    dp, *_ = rhs(p, 50e5, 50e5, V, dVdt, np.zeros(z), np.zeros(z), DEFAULT)
    assert np.allclose(dp, -DEFAULT.K_bulk / V * dVdt)


def test_grid_no_duplicate_endpoint():
    g = build_grid(DEFAULT.with_(steps_per_deg=2))
    assert g.phi.size == 720 and g.phi[-1] < 2 * np.pi
    assert g.dt == pytest.approx(2 * np.pi / 720 / DEFAULT.omega)
    assert np.all(g.V > 0)
