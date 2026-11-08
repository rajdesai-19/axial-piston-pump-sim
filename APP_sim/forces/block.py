"""
forces/block.py - Resultant loads on the cylinder block, simulation frame (+z towards the valve plate).

Ivantysyn & Ivantysynova Sect. 4.2.2(b), Eqs. 4.85-4.111, with directions mapped:
  F_DB,i = (p_i - p_c) A_D  along +z (chamber pressure on the bore bottom pushes the block
           onto the valve plate), A_D = Ak - A_kidney, on the cylinder axis      IV 4.85-4.88
  F_TB,i = -F_TK,i  along z (reaction of the piston bore friction)               IV 4.92-4.93
  radial loads transmitted by piston i:
     centrifugal="exact" (default): F_SK,xy + F_TG at the ball joint (x, y, z_bj) and the
        actual centrifugal force F_wK at the piston CoM (x, y, z_com). This is the exact
        rigid-body resultant of the external forces on the piston.
     centrifugal="book": F_RB,i = F_RK,i (with the equivalent F'_wK) at the ball joint,
        literally IV 4.94-4.95. F'_wK is only moment-equivalent about the guide centre,
        so the summed radial force differs from the exact one.
  F_FB   = spring preload along +z on the axis                                   IV 4.99
  resultants (IV 4.96-4.100): F = sum F_j, M_O = sum r_j x F_j about the origin
Moments are also given about the valve-plate centre (0, 0, z_VP), with and without the
radial bearing reaction R = -(F_Bx, F_By) applied at (0, 0, z_bearing).
Drive torque: the block turns clockwise seen from +z (omega vector = -omega z), so the
resisting moment of the pistons is M_Oz > 0 and the input power is M_Oz * omega.
Pressures are gauge w.r.t. case pressure (the case acts on all outer block surfaces).
"""
import numpy as np

from ..params import PumpParams, DEFAULT


def block_forces(pf: dict, p, prm: PumpParams = DEFAULT, centrifugal: str = "exact"):
    """pf: output of piston_forces; p: (z, N) chamber pressures [Pa]."""
    x, y, zb = pf["x"], pf["y"], pf["z_bj"]
    A_D = prm.Ak - prm.A_kidney
    F_DB = (np.asarray(p) - prm.p_case) * A_D          # (z, N), +z
    F_TB = -pf["F_TK"]                                  # (z, N), z

    # radial force groups: list of (Fx, Fy, z_application)
    if centrifugal == "exact":
        groups = [(pf["F_SKx"] + pf["F_TGx"], pf["F_SKy"] + pf["F_TGy"], zb),
                  (pf["F_wx_act"], pf["F_wy_act"], pf["z_com"])]
    elif centrifugal == "book":
        groups = [(pf["F_RKx"], pf["F_RKy"], zb)]
    else:
        raise ValueError("centrifugal must be 'exact' or 'book'")

    F_RBx = sum(g[0] for g in groups)                   # (z, N) per piston
    F_RBy = sum(g[1] for g in groups)
    F_Bx = F_RBx.sum(axis=0)
    F_By = F_RBy.sum(axis=0)
    F_DB_tot = F_DB.sum(axis=0)
    F_TB_tot = F_TB.sum(axis=0)
    F_Bz = F_DB_tot + F_TB_tot + prm.F_spring           # clamping onto the valve plate (+z)

    Fz_i = F_DB + F_TB
    # M = sum r x F about the origin (IV 4.98)
    M_Ox = (Fz_i * y).sum(axis=0) - sum((g[1] * g[2]).sum(axis=0) for g in groups)
    M_Oy = sum((g[0] * g[2]).sum(axis=0) for g in groups) - (Fz_i * x).sum(axis=0)
    M_Oz = (F_RBy * x - F_RBx * y).sum(axis=0)

    # about the valve-plate centre (0, 0, z_VP):  M_VP = M_O - r_VP x F
    zVP = prm.z_VP
    M_VPx = M_Ox + zVP * F_By
    M_VPy = M_Oy - zVP * F_Bx
    # radial bearing reaction at (0, 0, z_bearing)
    R_x, R_y = -F_Bx, -F_By
    dz = prm.z_bearing - zVP
    M_VPx_brg = M_VPx - dz * R_y
    M_VPy_brg = M_VPy + dz * R_x

    x_B = (F_DB * x).sum(axis=0) / F_DB_tot             # IV 4.110-4.111
    y_B = (F_DB * y).sum(axis=0) / F_DB_tot

    return dict(
        A_D=A_D, F_DB=F_DB, F_TB=F_TB, F_DB_tot=F_DB_tot, F_TB_tot=F_TB_tot,
        F_RBx=F_RBx, F_RBy=F_RBy,
        F_Bx=F_Bx, F_By=F_By, F_RB=np.hypot(F_Bx, F_By), F_Bz=F_Bz,
        M_Ox=M_Ox, M_Oy=M_Oy, M_Oz=M_Oz, M_drive=M_Oz,
        M_VPx=M_VPx, M_VPy=M_VPy, M_kipp=np.hypot(M_VPx, M_VPy),
        M_VPx_brg=M_VPx_brg, M_VPy_brg=M_VPy_brg, M_kipp_brg=np.hypot(M_VPx_brg, M_VPy_brg),
        F_bearing_x=R_x, F_bearing_y=R_y, F_bearing=np.hypot(R_x, R_y),
        x_B=x_B, y_B=y_B,
    )
