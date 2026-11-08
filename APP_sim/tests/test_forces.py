"""
Phase 2 checks for APP_sim/forces.
Reference-equation terms are checked against the reference formulas; Ivantysyn-only terms with
limit cases and physical direction checks.
"""
import numpy as np
import pytest

from APP_sim.params import DEFAULT
from APP_sim.frame import SwashFrame, pdf_single_piston
from APP_sim.forces import piston_forces, block_forces, swashplate_forces

PHI = np.linspace(0.0, 2 * np.pi, 3600, endpoint=False)


def square_pressure(prm, p_high=350e5):
    """High pressure on the discharge half (piston angle 0..180 deg), case pressure elsewhere."""
    phj = SwashFrame(prm).piston_phi(PHI)
    return np.where(phj < np.pi, p_high, prm.p_case)


def test_cradle_torque_matches_pdf_formula():
    prm = DEFAULT
    p = square_pressure(prm)
    pf = piston_forces(PHI, p, prm)
    sf = swashplate_forces(pf, prm)
    ref = pdf_single_piston(pf["phi"], p, prm.p_case, prm.Ak, prm.m_k, prm.omega, prm.Rp,
                            prm.r_shoe, prm.beta, prm.eta, prm.ex, prm.ez)
    np.testing.assert_allclose(sf["T_i"], ref["Tp"] + ref["Ti"], rtol=1e-12, atol=1e-9)
    np.testing.assert_allclose(sf["F_n"], ref["FDK_n"] + ref["FaK_n"], rtol=1e-12)


def _ideal(prm):
    # single-angle book case: no eta / ex, no inertia, no centrifugal, no friction
    return prm.with_(eta_deg=0.0, ex=0.0, ez=0.0, m_k=0.0, mu=0.0)


def test_ivantysyn_4_124_radial_force():
    prm = _ideal(DEFAULT)
    p = square_pressure(prm)
    pf = piston_forces(PHI, p, prm)
    expected = (p - prm.p_case) * prm.Ak * np.tan(prm.beta)       # IV 4.124 magnitude
    np.testing.assert_allclose(pf["F_RK"], expected, rtol=1e-12, atol=1e-9)
    # direction: towards +x (the low side of the swash plate, BDC side) when pressurised
    assert np.all(pf["F_RKx"][p > prm.p_case] > 0)
    np.testing.assert_allclose(pf["F_RKy"], 0.0, atol=1e-9)


def test_drive_torque_equals_displacement_times_pressure():
    prm = _ideal(DEFAULT)
    dp = 350e5 - prm.p_case
    p = square_pressure(prm)
    pf = piston_forces(PHI, p, prm)
    bf = block_forces(pf, p, prm)
    Vg = prm.num_piston * prm.Ak * SwashFrame(prm).stroke
    assert bf["M_drive"].mean() == pytest.approx(Vg * dp / (2 * np.pi), rel=2e-3)
    assert np.all(bf["M_drive"] > 0)           # resists the clockwise rotation


def test_friction_opposes_motion():
    prm = DEFAULT
    p = square_pressure(prm)
    pf = piston_forces(PHI, p, prm, include_bore_friction_in_axial=True)
    assert np.all(pf["F_TGx"] * pf["vx"] + pf["F_TGy"] * pf["vy"] < 0)
    assert np.all(pf["F_TK"] * pf["vz"] <= 0)


def test_swashplate_reaction_balances_axial_load():
    prm = DEFAULT
    p = square_pressure(prm)
    pf = piston_forces(PHI, p, prm)
    a, b, c = SwashFrame(prm).abc
    # force on piston = -(force on swash plate); z-component cancels F_AK
    np.testing.assert_allclose(pf["F_SKz"] + pf["F_AK"], 0.0, atol=1e-9)
    np.testing.assert_allclose(np.hypot(pf["F_SKx"], np.hypot(pf["F_SKy"], pf["F_SKz"])),
                               np.abs(pf["F_SK_n"]), rtol=1e-12)


def test_exact_centrifugal_cancels_in_resultant():
    prm = DEFAULT
    p = square_pressure(prm)
    pf = piston_forces(PHI, p, prm)
    np.testing.assert_allclose(pf["F_wx_act"].sum(axis=0), 0.0, atol=1e-6)
    np.testing.assert_allclose(pf["F_wy_act"].sum(axis=0), 0.0, atol=1e-6)
    ex = block_forces(pf, p, prm, centrifugal="exact")
    bk = block_forces(pf, p, prm, centrifugal="book")
    diff = (pf["F_wx"] - pf["F_wx_act"]).sum(axis=0)
    np.testing.assert_allclose(bk["F_Bx"] - ex["F_Bx"], diff, atol=1e-6)


def test_cross_angle_side_force_sign():
    # eta < 0 (discharge side raised): b > 0, pressurised pistons are pushed towards +y
    prm = DEFAULT.with_(m_k=0.0, mu=0.0)
    p = square_pressure(prm)
    pf = piston_forces(PHI, p, prm)
    assert np.all(pf["F_SKy"][p > prm.p_case] > 0)
