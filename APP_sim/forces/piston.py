"""
forces/piston.py - Forces on each piston-slipper assembly, simulation frame.

Equations and frame mapping (IV = Ivantysyn & Ivantysynova):
  F_DK  = -(p - p_c) Ak  along z                    IV 4.57
  F_aK  = -m_k a_z       along z                    IV 4.60
  F_TK  = f_K |F_RK| sgn(-v_z) along z              IV 4.80  (opposes piston axial velocity)
  F_AK  = F_DK + F_aK [+ F_TK]                      IV 4.69  (reference model neglects F_TK -> switch, default off)
  swash plate reaction ON the piston, at the ball joint:
        F_SK = -(F_AK / c) * n,  n = (a, b, c)       z-component = -F_AK balances the axial load
        (F_SK = F_AK / c is the force ON the swash plate along n)
  slipper friction (viscous), at the ball joint:
        |F_TG| = mu * omega * Rp / h_G * pi (r_o^2 - r_i^2)    IV 4.72
        direction opposite to the slipper velocity in xy: v = omega Rp (-sin phi, -cos phi)
        -> F_TG = |F_TG| (sin phi, cos phi)           (IV 4.73 rotated into the simulation frame; z part neglected as in IV)
  centrifugal force, replaced by its equivalent at the ball joint:
        F_wK  = m_k Rp omega^2, radially outward (cos phi, -sin phi)      IV 4.63-4.64
        F'_wK = F_wK (l_S1 - l_F/2) / l_KM,  l_KM = l_K - l_F/2           IV 4.76-4.78
  radial force on the piston at the ball joint (reacted by the bore):
        F_RK = F_SK,xy + F'_wK + F_TG                 IV 4.79 / 4.94
  F'_wK is moment-equivalent about the guide centre M only (it gives the bore contact
  load, hence F_TK). It is NOT force-equivalent: the real centrifugal forces of all
  pistons cancel in the sum, the F'_wK do not (l_F differs per piston). Block resultants
  therefore use the actual F_wK at the CoM (see forces/block.py, centrifugal="exact").
All arrays have shape (z, N) for N shaft angles.
"""
import numpy as np

from ..params import PumpParams, DEFAULT
from ..frame import SwashFrame


def piston_forces(phi_shaft, p, prm: PumpParams = DEFAULT, h_G=None,
                  include_bore_friction_in_axial: bool = False, n_iter: int = 3):
    """
    phi_shaft: (N,) shaft angles [rad]; p: (z, N) chamber pressures [Pa].
    h_G: slipper gap [m] for viscous friction, scalar or (z, N); default slipper_h0_init.
    """
    fr = SwashFrame(prm)
    phi = fr.piston_phi(np.asarray(phi_shaft, dtype=float))     # (z, N)
    p = np.asarray(p, dtype=float)
    a, b, c = fr.abc
    x, y = fr.xy(phi)
    z_bj = fr.z_bj(phi)
    vz, az = fr.vz(phi), fr.az(phi)
    vx, vy = fr.vxy(phi)

    F_DK = -(p - prm.p_case) * prm.Ak
    F_aK = -prm.m_k * az

    hG = prm.slipper_h0_init if h_G is None else h_G
    A_land = np.pi * (prm.slipper_r_out**2 - prm.slipper_r_in**2)
    F_TG_mag = prm.mu * prm.omega * prm.Rp / hG * A_land * np.ones_like(phi)
    v_xy = np.hypot(vx, vy)
    F_TGx = -F_TG_mag * vx / v_xy
    F_TGy = -F_TG_mag * vy / v_xy

    l_F = fr.engagement(phi)
    l_KM = prm.piston_l_K - l_F / 2.0
    F_wK = prm.m_k * prm.Rp * prm.omega**2
    F_wK_eq = F_wK * (prm.piston_l_S1 - l_F / 2.0) / l_KM
    F_wx = F_wK_eq * np.cos(phi)
    F_wy = -F_wK_eq * np.sin(phi)
    # actual centrifugal force at the piston+slipper CoM (for exact rigid-body resultants)
    z_com = z_bj + prm.piston_l_K - prm.piston_l_S1
    F_wx_act = F_wK * np.cos(phi)
    F_wy_act = -F_wK * np.sin(phi)

    F_TK = np.zeros_like(phi)
    for _ in range(n_iter if include_bore_friction_in_axial else 1):
        F_AK = F_DK + F_aK + (F_TK if include_bore_friction_in_axial else 0.0)
        F_SKx = -(F_AK / c) * a
        F_SKy = -(F_AK / c) * b
        F_RKx = F_SKx + F_wx + F_TGx
        F_RKy = F_SKy + F_wy + F_TGy
        F_RK = np.hypot(F_RKx, F_RKy)
        F_TK = prm.f_bore * F_RK * np.sign(-vz)

    return dict(
        phi=phi, x=x, y=y, z_bj=z_bj, vz=vz, az=az, vx=vx, vy=vy, l_F=l_F,
        F_DK=F_DK, F_aK=F_aK, F_TK=F_TK, F_AK=F_AK,
        F_SKx=F_SKx, F_SKy=F_SKy, F_SKz=-F_AK,
        F_SK_n=F_AK / c,                       # F_SK: force ON the swash plate along n (negative = pressing)
        F_TGx=F_TGx, F_TGy=F_TGy, F_TG_mag=F_TG_mag,
        F_wK=F_wK, F_wx=F_wx, F_wy=F_wy, F_wx_act=F_wx_act, F_wy_act=F_wy_act, z_com=z_com,
        F_RKx=F_RKx, F_RKy=F_RKy, F_RK=F_RK,
    )
