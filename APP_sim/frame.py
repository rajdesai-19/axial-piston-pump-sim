"""
frame.py - Geometry and kinematics in the simulation coordinate frame.

Geometry, kinematics and force conventions:
  * right-handed frame, +z towards the valve plate
  * shaft / piston angle phi is CLOCKWISE seen from +z:
        x = Rp*cos(phi),  y = -Rp*sin(phi)
  * swash plate unit normal:  n = (a, b, c) = (cos(eta)sin(beta), -sin(eta), cos(eta)cos(beta))
  * swash plate plane  a x + b y + c z = d_sp
  * ball-joint plane   a x + b y + c z = d_bj
  * ball-joint ellipse z_bj(phi) = d_bj/c - (Rp/c)(a cos(phi) - b sin(phi))
  * BDC = minimum z at phi_bdc = atan2(-b, a)
  * v_z, a_z
  * F_DK = -(p - p_c) Ak along z, F_aK = -m_k a_z, normal force = F_z / c
  * cradle moment arm, torque positive CCW seen from +y

All angles in radians, lengths in metres. Functions are numpy-vectorised in phi.
"""

from dataclasses import dataclass
import numpy as np

from .params import PumpParams, DEFAULT


# ----------------------------------------------------------------------------
# Plane coefficients
# ----------------------------------------------------------------------------
def plane_abc(beta: float, eta: float):
    """Swash plate unit normal components."""
    a = np.cos(eta) * np.sin(beta)
    b = -np.sin(eta)
    c = np.cos(eta) * np.cos(beta)
    return a, b, c


def d_swashplate(beta: float, eta: float, ex: float, ez: float) -> float:
    """Plane constant of the swash plate running face."""
    a, _, c = plane_abc(beta, eta)
    return a * (ex - ez * np.sin(beta)) + c * (ez - ez * np.cos(beta))


def d_balljoint(beta: float, eta: float, ex: float, ez: float, r_shoe: float) -> float:
    """Plane constant of the ball-joint centre plane."""
    a, b, c = plane_abc(beta, eta)
    return (a * (ex - ez * np.sin(beta) + r_shoe * np.cos(eta) * np.sin(beta))
            - b * r_shoe * np.sin(eta)
            + c * (ez - ez * np.cos(beta) + r_shoe * np.cos(eta) * np.cos(beta)))


# ----------------------------------------------------------------------------
# Positions
# ----------------------------------------------------------------------------
def pitch_xy(phi, Rp: float):
    """Piston (ball-joint) x, y on the pitch circle, clockwise phi."""
    return Rp * np.cos(phi), -Rp * np.sin(phi)


def z_on_plane(phi, Rp: float, a: float, b: float, c: float, d: float):
    """z of the plane a x + b y + c z = d above the pitch-circle point."""
    return d / c - (Rp / c) * (a * np.cos(phi) - b * np.sin(phi))


def contact_xyz(phi, Rp, r_shoe, beta, eta, ex, ez):
    """Shoe / swash plate contact point on the contact ellipse."""
    a, b, c = plane_abc(beta, eta)
    d_sp = d_swashplate(beta, eta, ex, ez)
    x = Rp * np.cos(phi) - r_shoe * np.cos(eta) * np.sin(beta)
    y = -Rp * np.sin(phi) + r_shoe * np.sin(eta)
    z = z_on_plane(phi, Rp, a, b, c, d_sp)
    return x, y, z


def cradle_moment_arm(phi, Rp, r_shoe, beta, eta, ex, ez):
    """
    y-component of (r_ce - r_ca) x n. Multiplied by the normal
    force it gives the single-piston cradle torque, positive CCW seen from +y.
    Note: the model uses z of the swash plate plane (d_sp), not d_bj.
    """
    a, b, c = plane_abc(beta, eta)
    d_sp = d_swashplate(beta, eta, ex, ez)
    X = Rp * np.cos(phi) - r_shoe * np.cos(eta) * np.sin(beta) - ex
    Z = (d_sp / c) - (Rp / c) * (a * np.cos(phi) - b * np.sin(phi)) - ez
    return -c * X + a * Z


# ----------------------------------------------------------------------------
# Single-piston forces exactly as written in the reference model
# ----------------------------------------------------------------------------
def pdf_single_piston(phi, p, p_c, Ak, m_k, omega, Rp, r_shoe, beta, eta, ex, ez):
    """
    Returns dict with FDK_z, FaK_z, FDK_n, FaK_n, mom_arm, Tp, Ti
    (pressure and inertia cradle torques) from the pressure force, inertia force and moment arm.
    """
    a, b, c = plane_abc(beta, eta)
    FDK_z = -(p - p_c) * Ak
    FaK_z = -m_k * omega**2 * (Rp / c) * (a * np.cos(phi) - b * np.sin(phi))
    FDK_n = FDK_z / c
    FaK_n = FaK_z / c
    arm = cradle_moment_arm(phi, Rp, r_shoe, beta, eta, ex, ez)
    return dict(FDK_z=FDK_z, FaK_z=FaK_z, FDK_n=FDK_n, FaK_n=FaK_n,
                mom_arm=arm, Tp=FDK_n * arm, Ti=FaK_n * arm)


# ----------------------------------------------------------------------------
# Convenience object bound to a parameter set
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class SwashFrame:
    """Kinematics of all pistons for one PumpParams set, in the simulation frame."""
    prm: PumpParams = DEFAULT

    # plane coefficients
    @property
    def abc(self):
        return plane_abc(self.prm.beta, self.prm.eta)

    @property
    def d_sp(self) -> float:
        p = self.prm
        return d_swashplate(p.beta, p.eta, p.ex, p.ez)

    @property
    def d_bj(self) -> float:
        p = self.prm
        return d_balljoint(p.beta, p.eta, p.ex, p.ez, p.r_shoe)

    @property
    def phi_bdc(self) -> float:
        """Angle of minimum z (BDC), wrapped to [0, 2pi)."""
        a, b, _ = self.abc
        return float(np.mod(np.arctan2(-b, a), 2.0 * np.pi))

    @property
    def phi_tdc(self) -> float:
        return float(np.mod(self.phi_bdc + np.pi, 2.0 * np.pi))

    # piston phase
    def piston_phi(self, phi_shaft, j=None):
        """
        Angle of piston j when the shaft is at phi_shaft (all clockwise).
        Piston j trails piston 0 by j*360/z degrees, i.e. it reaches any given
        position j*40 deg of shaft rotation later. With j=None returns an
        array of shape (z, ...) for all pistons.
        """
        z = self.prm.num_piston
        phi = np.asarray(phi_shaft, dtype=float)
        if j is None:
            offs = np.arange(z) * 2.0 * np.pi / z
            return np.mod(phi[None, ...] - offs.reshape((z,) + (1,) * phi.ndim), 2.0 * np.pi)
        return np.mod(phi - j * 2.0 * np.pi / z, 2.0 * np.pi)

    # ball-joint kinematics
    def z_bj(self, phi):
        a, b, c = self.abc
        return z_on_plane(phi, self.prm.Rp, a, b, c, self.d_bj)

    def xy(self, phi):
        return pitch_xy(phi, self.prm.Rp)

    def sk(self, phi):
        """Displacement from BDC in +z (>= 0)."""
        return self.z_bj(phi) - self.z_bj(self.phi_bdc)

    def vz(self, phi):
        a, b, c = self.abc
        return self.prm.omega * (self.prm.Rp / c) * (a * np.sin(phi) + b * np.cos(phi))

    def az(self, phi):
        a, b, c = self.abc
        return self.prm.omega**2 * (self.prm.Rp / c) * (a * np.cos(phi) - b * np.sin(phi))

    def vxy(self, phi):
        w, R = self.prm.omega, self.prm.Rp
        return -w * R * np.sin(phi), -w * R * np.cos(phi)

    def axy(self, phi):
        w, R = self.prm.omega, self.prm.Rp
        return -w * w * R * np.cos(phi), w * w * R * np.sin(phi)

    @property
    def stroke(self) -> float:
        return float(self.z_bj(self.phi_tdc) - self.z_bj(self.phi_bdc))

    # chamber volume
    @property
    def V_tdc(self) -> float:
        """
        Dead volume at TDC. V_dead_ref is defined for the reference geometry
        (eta = ex = ez = 0, same beta and r_shoe). Any shift of the TDC ball-joint
        height relative to that reference changes the dead volume by Ak * dz.
        """
        p = self.prm
        ref = SwashFrame(p.with_(eta_deg=0.0, ex=0.0, ez=0.0))
        dz = ref.z_bj(ref.phi_tdc) - self.z_bj(self.phi_tdc)
        return p.V_dead_ref + p.Ak * dz

    def volume(self, phi):
        """Chamber volume: larger z (towards valve plate) means smaller volume."""
        return self.V_tdc + self.prm.Ak * (self.z_bj(self.phi_tdc) - self.z_bj(phi))

    def dV_dt(self, phi):
        return -self.prm.Ak * self.vz(phi)

    def engagement(self, phi):
        """
        Piston guide length l_F (Ivantysyn Eq. 4.77) from the axial geometry:
        piston end face (ball centre + l_K) minus the block front face. Shortest at BDC.
        """
        return self.z_bj(phi) + self.prm.piston_l_K - self.prm.z_block_front

    def port_angle_deg(self, phi):
        """
        Angle used to look up the valve-plate port area table. The table
        (discharge 0..180, suction 180..360) is already written in the shaft-angle convention.
        """
        return np.degrees(np.mod(phi, 2.0 * np.pi))
