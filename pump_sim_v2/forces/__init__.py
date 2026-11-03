"""External forces and moments on pistons, cylinder block and swash plate (PDF frame)."""
from .piston import piston_forces
from .block import block_forces
from .swashplate import swashplate_forces


def all_forces(phi_shaft, p, prm=None, **kw):
    from ..params import DEFAULT
    prm = DEFAULT if prm is None else prm
    pf = piston_forces(phi_shaft, p, prm, **kw)
    return pf, block_forces(pf, p, prm), swashplate_forces(pf, prm)
