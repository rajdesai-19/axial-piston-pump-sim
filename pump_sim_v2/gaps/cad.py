"""
gaps/cad.py - Valve plate and barrel geometry from the CAD drawings.

vp.dxf / barrel_top.dxf (pump_sim_v2/cad_data/) are used in the PDF frame as drawn, with the
kidney of a piston at angle phi located at mathematical angle -phi (clockwise rotation).
The DXF files shipped here are a synthetic DEMO geometry (annular ports and kidneys with round
end caps), not a real valve plate. The two ports are mirror-symmetric about x; the port at y < 0
is the DISCHARGE port because the pistons compress on that side (kinematics).
The kidney/port overlap reproduces the chamber model's port-area table (chamber.py).
"""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import matplotlib.path as mpath

CAD_DIR = Path(__file__).resolve().parents[1] / "cad_data"


def _read_loops(path):
    """Stitch LINE/ARC entities into closed polygons [m]; also return CIRCLE radii [m]."""
    import ezdxf
    msp = ezdxf.readfile(str(path)).modelspace()
    segs, circles = [], []
    for e in msp:
        t = e.dxftype()
        if t == "CIRCLE":
            circles.append(float(e.dxf.radius) * 1e-3)
        elif t == "LINE":
            p1 = np.array([e.dxf.start.x, e.dxf.start.y]); p2 = np.array([e.dxf.end.x, e.dxf.end.y])
            segs.append([p1, p2])
        elif t == "ARC":
            c = np.array([e.dxf.center.x, e.dxf.center.y]); r = float(e.dxf.radius)
            sa, ea = np.radians(e.dxf.start_angle), np.radians(e.dxf.end_angle)
            if ea < sa:
                ea += 2 * np.pi
            th = np.linspace(sa, ea, max(15, int(np.degrees(ea - sa) * 2)))
            segs.append([c + r * np.array([np.cos(a), np.sin(a)]) for a in th])
    loops, unused = [], list(range(len(segs)))
    while unused:
        loop = list(segs[unused.pop(0)])
        while True:
            end, hit = loop[-1], None
            for k in unused:
                s = segs[k]
                if np.linalg.norm(end - s[0]) < 0.05:
                    hit = s[1:]
                elif np.linalg.norm(end - s[-1]) < 0.05:
                    hit = s[::-1][1:]
                if hit is not None:
                    unused.remove(k)
                    break
            if hit is None:
                break
            loop.extend(hit)
        if np.linalg.norm(loop[-1] - loop[0]) < 0.1 and len(loop) > 3:
            loops.append(np.array(loop) * 1e-3)
    return loops, circles


@dataclass
class ValvePlateCAD:
    discharge: mpath.Path      # port at y < 0
    suction: mpath.Path        # port at y > 0
    kidney: np.ndarray         # one barrel kidney polygon (N, 2) [m]
    kidney_angle: float        # its centroid angle [rad]
    r_in: float                # inner land radius from the drawing [m]
    r_out: float

    def kidney_at(self, theta):
        """Kidney polygon rotated so that its centroid sits at mathematical angle theta."""
        d = theta - self.kidney_angle
        c, s = np.cos(d), np.sin(d)
        return self.kidney @ np.array([[c, -s], [s, c]]).T


def load_cad(cad_dir=CAD_DIR) -> ValvePlateCAD:
    vp_loops, circles = _read_loops(Path(cad_dir) / "vp.dxf")
    ports = sorted(vp_loops, key=lambda lp: lp[:, 1].mean())
    discharge, suction = ports[0], ports[-1]
    if not (discharge[:, 1].mean() < 0 < suction[:, 1].mean()):
        raise RuntimeError("unexpected port layout in vp.dxf")
    b_loops, _ = _read_loops(Path(cad_dir) / "barrel_top.dxf")
    kidneys = [lp for lp in b_loops if np.mean(np.hypot(lp[:, 0], lp[:, 1])) > 20e-3]
    k = kidneys[0]
    small = [r for r in circles if r < 25e-3]
    big = [r for r in circles if r >= 25e-3]
    return ValvePlateCAD(
        discharge=mpath.Path(discharge), suction=mpath.Path(suction), kidney=k,
        kidney_angle=float(np.arctan2(k[:, 1].mean(), k[:, 0].mean())),
        r_in=max(small) if small else np.nan, r_out=min(big) if big else np.nan)
