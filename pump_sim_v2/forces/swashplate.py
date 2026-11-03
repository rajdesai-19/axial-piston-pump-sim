"""
forces/swashplate.py - Loads on the swash plate and cradle torque, PDF frame.

PDF terms (v5 Cradle Torque Eq.pdf p.16-21):
  normal force ON the swash plate from piston i:  F_n,i = F_AK,i / c  along n = (a, b, c)
  (negative = pressing the plate, PDF sign), applied at the contact-ellipse point (p.10)
  cradle torque about the cradle axis (|| y through (ex, 0, ez)):
        T_i = F_n,i * moment_arm(phi_i),  positive CCW seen from +y      p.20-21, p.24
Ivantysyn-only term (not in the PDF, reported separately, not in T_cradle):
  slipper friction ON the swash plate = -(friction on the slipper), in the xy plane.
"""
import numpy as np

from ..params import PumpParams, DEFAULT
from ..frame import SwashFrame, contact_xyz, cradle_moment_arm


def swashplate_forces(pf: dict, prm: PumpParams = DEFAULT):
    fr = SwashFrame(prm)
    a, b, c = fr.abc
    phi = pf["phi"]
    F_n = pf["F_SK_n"]                                   # (z, N)
    Fx, Fy, Fz = F_n * a, F_n * b, F_n * c
    xs, ys, zs = contact_xyz(phi, prm.Rp, prm.r_shoe, prm.beta, prm.eta, prm.ex, prm.ez)
    arm = cradle_moment_arm(phi, prm.Rp, prm.r_shoe, prm.beta, prm.eta, prm.ex, prm.ez)
    T_i = F_n * arm

    # friction on the swash plate (reaction of the slipper friction), Ivantysyn only
    Ftx, Fty = -pf["F_TGx"], -pf["F_TGy"]

    M_Sx = (ys * Fz - zs * Fy).sum(axis=0)
    M_Sy = (zs * Fx - xs * Fz).sum(axis=0)
    M_Sz = (xs * Fy - ys * Fx).sum(axis=0)
    M_fric_z = (xs * Fty - ys * Ftx).sum(axis=0)

    return dict(
        F_n=F_n, F_S_total=-F_n.sum(axis=0),             # total pressing force (positive)
        F_Sx=Fx.sum(axis=0), F_Sy=Fy.sum(axis=0), F_Sz=Fz.sum(axis=0),
        arm=arm, T_i=T_i, T_cradle=T_i.sum(axis=0),
        T_cradle_pressure=(pf["F_DK"] / c * arm).sum(axis=0),
        T_cradle_inertia=(pf["F_aK"] / c * arm).sum(axis=0),
        M_Sx=M_Sx, M_Sy=M_Sy, M_Sz=M_Sz, M_friction_z=M_fric_z,
        F_friction_x=Ftx.sum(axis=0), F_friction_y=Fty.sum(axis=0),
        contact=(xs, ys, zs),
    )
