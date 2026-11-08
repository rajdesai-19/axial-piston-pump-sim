"""
Regression tests for APP_sim/frame.py using fixed numerical cases and analytic identities.
Run:  py -m pytest APP_sim/tests -q
"""
import math
import numpy as np
import pytest

from APP_sim.frame import (
    plane_abc, d_swashplate, d_balljoint, cradle_moment_arm, pdf_single_piston, SwashFrame,
)
from APP_sim.params import DEFAULT


# --- single-piston numerical case (regression values of the reference formulas, arbitrary inputs) ---
P31 = dict(eta=-0.04, beta=0.29, Ak=math.pi * 0.016**2 / 4, Rp=0.028,
           ex=-0.0010, ez=0.015, r_shoe=0.008, omega=520.0,
           phi=40 * math.pi / 180, m_k=0.050, p=28.0e6, p_c=0.20e6)


def test_p31_numerical_case():
    t = P31
    r = pdf_single_piston(t["phi"], t["p"], t["p_c"], t["Ak"], t["m_k"], t["omega"],
                          t["Rp"], t["r_shoe"], t["beta"], t["eta"], t["ex"], t["ez"])
    # pressure force check by hand: -(p - p_c) Ak / c
    assert r["FDK_n"] == pytest.approx(-(t["p"] - t["p_c"]) * t["Ak"] / plane_abc(t["beta"], t["eta"])[2], rel=1e-9)
    assert r["FDK_n"] == pytest.approx(-5837.76, abs=0.02)
    assert r["FaK_n"] == pytest.approx(-79.77, abs=0.01)
    assert r["mom_arm"] == pytest.approx(-0.0255, abs=1e-4)
    assert r["Tp"] == pytest.approx(148.73, abs=0.01)
    assert r["Ti"] == pytest.approx(2.03, abs=0.01)


def test_unit_normal():
    a, b, c = plane_abc(0.3, -0.05)
    assert a * a + b * b + c * c == pytest.approx(1.0)


def test_d_bj_is_plane_shift_by_r_shoe():
    # Shifting a unit-normal plane by r_shoe along n adds exactly r_shoe to d.
    for beta, eta in [(0.3, -0.05), (0.2, 0.04), (0.0, 0.0)]:
        assert d_balljoint(beta, eta, -1.0e-3, 0.015, 0.008) == pytest.approx(
            d_swashplate(beta, eta, -1.0e-3, 0.015) + 0.008)


@pytest.mark.parametrize("phi_deg", [0, 25, 60, 90, 135, 200, 300])
def test_p21_moment_arm_regression(phi_deg):
    # eta = 0, ez = 0, r_shoe = 0  ->  arm = -(Rp cos(phi) - ex) / cos(beta)
    beta, Rp, ex = 0.3, 0.0276, -0.0015
    phi = math.radians(phi_deg)
    arm = cradle_moment_arm(phi, Rp, 0.0, beta, 0.0, ex, 0.0)
    assert arm == pytest.approx(-(Rp * math.cos(phi) - ex) / math.cos(beta), rel=1e-12)


def test_p24_sign_of_pressure_torque():
    # ex = ez = eta = r_shoe = 0: discharge phi in (0, 90) deg gives CCW (+) torque.
    for phi_deg in [5, 30, 60, 85]:
        r = pdf_single_piston(math.radians(phi_deg), 280e5, 9.5e5, 1.54e-4, 0.0, 628.3,
                              0.0276, 0.0, 0.3, 0.0, 0.0, 0.0)
        assert r["Tp"] > 0.0


def test_kinematics_derivatives_consistent():
    f = SwashFrame(DEFAULT)
    phi = np.linspace(0, 2 * np.pi, 3601)
    h = 1e-6
    dz = (f.z_bj(phi + h) - f.z_bj(phi - h)) / (2 * h) * DEFAULT.omega
    np.testing.assert_allclose(f.vz(phi), dz, rtol=1e-6, atol=1e-9)
    dv = (f.vz(phi + h) - f.vz(phi - h)) / (2 * h) * DEFAULT.omega
    np.testing.assert_allclose(f.az(phi), dv, rtol=1e-5, atol=1e-6)


def test_bdc_is_minimum_and_discharge_compresses():
    f = SwashFrame(DEFAULT)
    phi = np.linspace(0, 2 * np.pi, 36001, endpoint=False)
    z = f.z_bj(phi)
    assert f.z_bj(f.phi_bdc) == pytest.approx(z.min(), abs=1e-9)
    assert f.z_bj(f.phi_tdc) == pytest.approx(z.max(), abs=1e-9)
    # eta < 0 moves BDC before phi = 0: tan(phi_bdc) = tan(eta) / sin(beta) (kinematic extremum)
    bdc = 360.0 + math.degrees(math.atan(math.tan(DEFAULT.eta) / math.sin(DEFAULT.beta)))
    assert math.degrees(f.phi_bdc) == pytest.approx(bdc, abs=0.05)
    # Compression stroke BDC -> TDC (BDC + 180 deg): piston moves towards the valve plate
    assert math.degrees(f.phi_tdc) == pytest.approx(bdc - 180.0, abs=0.05)
    lo, hi = int(math.ceil(bdc - 360.0)) + 1, int(bdc - 180.0) - 1
    assert np.all(f.vz(np.radians(np.arange(lo, hi))) > 0)
    assert np.all(f.vz(np.radians(np.arange(hi + 2, lo + 359))) < 0)


def test_stroke_and_volume():
    f = SwashFrame(DEFAULT)
    a, b, c = f.abc
    assert f.stroke == pytest.approx(2 * DEFAULT.Rp * math.hypot(a, b) / c, rel=1e-12)
    phi = np.linspace(0, 2 * np.pi, 3601)
    V = f.volume(phi)
    assert V.min() == pytest.approx(f.V_tdc, rel=1e-6)
    assert V.max() - V.min() == pytest.approx(DEFAULT.Ak * f.stroke, rel=1e-6)


def test_reference_geometry_dead_volume():
    p = DEFAULT.with_(eta_deg=0.0, ex=0.0, ez=0.0)
    assert SwashFrame(p).V_tdc == pytest.approx(p.V_dead_ref, rel=1e-12)


def test_piston_phase_ordering():
    f = SwashFrame(DEFAULT)
    phis = f.piston_phi(np.array([0.0, 0.5]))
    assert phis.shape == (DEFAULT.num_piston, 2)
    # piston 1 reaches piston 0's position 40 deg of shaft rotation later
    assert f.piston_phi(math.radians(40.0), 1) == pytest.approx(f.piston_phi(0.0, 0), abs=1e-12)
