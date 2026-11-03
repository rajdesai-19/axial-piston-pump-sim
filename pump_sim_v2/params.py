"""
params.py - Single source of truth for every pump parameter (SI units).

Frame and sign conventions follow the v5 Cradle Torque Eq. document:
right-handed, +z towards the valve plate, shaft angle phi clockwise,
piston at (Rp*cos(phi), -Rp*sin(phi)), phi = 0 at BDC, discharge 0..180 deg.

NOTE: the values below describe a fictitious DEMO pump (about 29 cm^3/rev). They are
not the data of any real product. Replace them (and cad_data/*.dxf plus the port-area
table in chamber.py) with your own pump.
"""

from dataclasses import dataclass, replace
import math


@dataclass(frozen=True)
class PumpParams:
    # --- Pump architecture ---
    num_piston: int = 9
    rpm: float = 5000.0
    dk: float = 15.8e-3                 # piston diameter [m]
    Rp: float = 27.6e-3                 # pitch circle radius [m] (matches cad_data/*.dxf)

    # --- Swash plate / cradle ---
    beta_deg: float = 16.5              # primary swash plate angle (about y)
    eta_deg: float = -2.5               # secondary swash plate angle (about x)
    ex: float = -1.5e-3                 # cradle axis offset along x [m]
    ez: float = 0.0                     # cradle axis offset along z [m]; 0: axis fixed in z

    # --- Drawing geometry, global z with a FLAT swash plate (beta = eta = 0) ---
    # The swash plate face lies at z = z_swash_face_drawing in the drawing; in the PDF frame (ez = 0)
    # it passes through z = 0 at beta = 0, so PDF z = drawing z - z_swash_face_drawing.
    z_swash_face_drawing: float = 11.3e-3     # slipper running face / swash plate face
    z_ball_drawing: float = 19.38e-3          # piston ball centre
    ball_radius: float = 4.8e-3
    z_block_front_drawing: float = 35.76e-3   # cylinder block face towards the swash plate
    z_VP_drawing: float = 79.21e-3            # cylinder block face on the valve plate
    engagement_flat: float = 28.20e-3         # piston-bore engagement with a flat swash plate
    z_cg_slipper_drawing: float = 14.88e-3
    z_cg_piston_drawing: float = 40.83e-3
    z_cg_assembly_drawing: float = 35.73e-3   # piston + slipper

    # --- Masses ---
    m_k: float = 0.0500                 # piston + slipper reciprocating mass [kg]
    m_block: float = 0.780              # cylinder block [kg]

    # --- Fluid (isothermal, ISO VG 46 at 60 C) ---
    rho: float = 870.0                  # [kg/m^3]
    mu: float = 870.0 * 20.0e-6         # [Pa.s] = 0.0174
    K_bulk: float = 1.4e9               # [Pa]

    # --- Pressures ---
    p_case: float = 2.0e5               # case / drain [Pa]
    p_reservoir: float = 5.0e5          # suction boost reservoir [Pa]
    p_initial: float = 5.0e5            # initial fill of chambers and lines [Pa]
    p_load_downstream: float = 5.0e5    # load orifice discharges to the reservoir [Pa]
    pd_nominal: float = 300.0e5         # reference only (plots) [Pa]

    # --- Hydraulic lines & porting ---
    d_load: float = 4.05e-3             # load orifice diameter [m]
    cd_load: float = 0.70
    cd_discharge_port: float = 0.70     # valve-plate discharge kidney
    cd_suction_port: float = 0.70       # valve-plate suction kidney
    cd_suction_inlet: float = 0.70      # reservoir -> suction line
    r_suction_inlet: float = 18.0e-3    # [m]
    V_delivery_line: float = 60.0e-6    # [m^3] delivery line volume
    V_suction_line: float = 130.0e-6    # [m^3] suction line volume
    A_port_max: float = 73.19e-6        # max kidney opening area [m^2] (kidney area in barrel_top.dxf)

    # --- Chamber dead volume ---
    # Volume at TDC measured on the geometry with ex = ez = 0 (and eta = 0), same beta and r_shoe.
    # The actual TDC volume is corrected from z_bj in frame.py.
    V_dead_ref: float = 1050.0e-9       # [m^3]

    # --- Slipper ---
    slipper_r_in: float = 7.05e-3       # pocket radius [m]
    slipper_r_out: float = 9.15e-3      # land outer radius [m]
    slipper_d1: float = 1.40e-3         # piston capillary diameter [m]
    slipper_l1: float = 16.2e-3         # piston capillary length [m]
    slipper_d2: float = 1.00e-3         # slipper neck orifice diameter [m]
    slipper_l2: float = 2.80e-3         # slipper neck orifice length [m]
    slipper_cd: float = 0.65
    slipper_h0_init: float = 10.0e-6    # initial gap [m] (solved afterwards)
    slipper_retainer_clearance: float = 20.0e-6
    slipper_max_tilt: float = 1.5e-3    # [rad]
    slipper_h_min: float = 0.5e-6       # film clamp inside the Reynolds solver [m]
    slipper_h_contact: float = 2.5e-6   # asperity contact threshold [m]
    slipper_p_contact: float = 1.0e8    # contact pressure at h = 0 [Pa] (quadratic penalty)
    slipper_K_retainer: float = 1.0e13  # retainer quadratic stiffness [N/m^2]
    slipper_use_retainer: bool = True   # retainer (hold-down) stop at the retainer clearance;
                                        # force reported, not judged for failure
    # slipper_max_tilt is informational only (no hard stop applied).
    slipper_Nr: int = 21
    slipper_Nphi: int = 48

    # --- Valve plate / cylinder block gap ---
    vp_r_in: float = 20.5e-3            # inner sealing land radius [m]
    vp_r_out: float = 34.5e-3           # outer sealing land radius [m]
    vp_h0_init: float = 4.0e-6          # [m]
    vp_h_contact: float = 2.2e-6        # asperity contact threshold [m]
    vp_h_max: float = 20.0e-6           # [m]
    vp_h_min: float = 0.5e-6            # film clamp inside the Reynolds solver [m]
    vp_tilt_free: bool = False          # False: bearing/shaft carry the block tilting moment,
                                        # the gap solves h0 only with the block parallel to the valve plate
    vp_p_contact: float = 1.0e8         # contact pressure at h = 0 [Pa] (quadratic penalty)
    p_cavitation: float = 0.0           # absolute pressure floor in the films (half-Sommerfeld) [Pa]
    vp_Nr: int = 36
    vp_Ntheta: int = 480

    # --- Piston / bore gap ---
    piston_gap: bool = True             # solve the piston-bore film and feed Q_SK to the chambers
    piston_clearance_diametral: float = 20.0e-6   # bore - piston diameter [m]
    piston_h_min: float = 0.5e-6        # film clamp inside the Reynolds solver [m] (as slipper / valve plate)
    piston_h_contact: float = 1.0e-6    # asperity contact threshold [m] (assumed, ~3x combined roughness)
    piston_p_contact: float = 1.0e10    # contact pressure at h = 0 [Pa] (quadratic penalty). Stiffer than the
                                        # flat gaps because the bore-edge contact is a narrow line
    piston_Nz: int = 25                 # axial nodes over the engagement length
    piston_Nth: int = 48                # circumferential nodes

    # --- Cylinder block & spring ---
    A_kidney: float = 73.19e-6          # barrel kidney area used for shoulder A_D [m^2]
    F_spring: float = 600.0             # [N]
    k_spring: float = 95.0e3            # [N/m]
    x0_spring: float = 3.5e-3           # [m]
    z_CG_block: float = 20.9e-3         # from valve plate face [m]

    # --- Bearing (outer race at the swash-plate end of the block) ---
    bearing_ir: float = 38.0e-3
    bearing_or: float = 40.4e-3
    bearing_width: float = 16.0e-3
    bearing_z_rollers: int = 38
    bearing_stribeck: float = 4.37
    bearing_contact_length: float = 11.5e-3

    # --- Friction ---
    f_bore: float = 0.03
    f_slipper: float = 0.01

    # --- Numerics ---
    steps_per_deg: int = 200            # chamber ODE resolution (multiplier)
    total_revolutions: int = 15
    n_export_cycles: int = 5
    n_export_points_per_rev: int = 720

    # ----- derived quantities -----
    @property
    def omega(self) -> float:
        return self.rpm * math.pi / 30.0

    @property
    def beta(self) -> float:
        return math.radians(self.beta_deg)

    @property
    def eta(self) -> float:
        return math.radians(self.eta_deg)

    @property
    def Ak(self) -> float:
        return math.pi * self.dk ** 2 / 4.0

    @property
    def piston_clearance(self) -> float:
        """Radial piston-bore clearance (concentric film thickness)."""
        return 0.5 * self.piston_clearance_diametral

    @property
    def A_load(self) -> float:
        return math.pi * self.d_load ** 2 / 4.0

    @property
    def A_suction_inlet(self) -> float:
        return math.pi * self.r_suction_inlet ** 2

    # --- geometry derived from the flat-swash-plate drawing (PDF frame: z - z_swash_face) ---
    @property
    def r_shoe(self) -> float:
        """Ball centre to slipper running face."""
        return self.z_ball_drawing - self.z_swash_face_drawing

    @property
    def z_VP(self) -> float:
        return self.z_VP_drawing - self.z_swash_face_drawing

    @property
    def z_block_front(self) -> float:
        return self.z_block_front_drawing - self.z_swash_face_drawing

    @property
    def z_bearing(self) -> float:
        """Bearing centre: half the outer-race width from the block front face."""
        return self.z_block_front + 0.5 * self.bearing_width

    @property
    def L_block(self) -> float:
        return self.z_VP_drawing - self.z_block_front_drawing

    @property
    def piston_l_K(self) -> float:
        """Ball centre to piston chamber-end face (book l_K): block front + flat engagement - ball."""
        return self.z_block_front_drawing + self.engagement_flat - self.z_ball_drawing

    @property
    def piston_l_S1(self) -> float:
        """Piston chamber-end face to piston+slipper CoM (book l_S1)."""
        return self.piston_l_K - (self.z_cg_assembly_drawing - self.z_ball_drawing)

    @property
    def slipper_l_SG(self) -> float:
        """Ball centre to slipper CoM (towards the running face)."""
        return self.z_ball_drawing - self.z_cg_slipper_drawing

    @property
    def m_piston(self) -> float:
        """Piston mass consistent with the three CoM positions and m_k."""
        zs, zp, za = self.z_cg_slipper_drawing, self.z_cg_piston_drawing, self.z_cg_assembly_drawing
        return self.m_k * (za - zs) / (zp - zs)

    @property
    def m_slipper(self) -> float:
        return self.m_k - self.m_piston

    def with_(self, **changes) -> "PumpParams":
        """Return a copy with some fields changed (dataclass is frozen)."""
        return replace(self, **changes)


DEFAULT = PumpParams()
