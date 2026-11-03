"""
convergence_gapstep.py - Gap-update interval study for the coupled solver (Phase 4 accuracy check).
Coarse valve-plate grid, 2 revolutions each; compares the last-revolution summaries.

Run:  py -m pump_sim_v2.tools.convergence_gapstep
"""
import json
from pathlib import Path

from ..params import DEFAULT
from ..solver import CoupledSolver, SolverOptions, summarize

OUT = Path(__file__).resolve().parents[1] / "outputs" / "tools"


def main(steps_deg=(1.0, 0.5)):
    prm = DEFAULT.with_(vp_Nr=21, vp_Ntheta=180)
    rows = {}
    for s in steps_deg:
        print(f"--- gap step {s} deg ---", flush=True)
        rows[s] = summarize(CoupledSolver(prm, SolverOptions(revolutions=2, gap_step_deg=s)).run())
    keys = list(rows[steps_deg[0]].keys())
    print(f"\n{'quantity':18s}" + "".join(f"{s:>12.2f}deg" for s in steps_deg) + "   rel.diff")
    for k in keys:
        a, b = rows[steps_deg[0]][k], rows[steps_deg[-1]][k]
        rel = abs(a - b) / max(abs(b), 1e-12)
        print(f"{k:18s}" + "".join(f"{rows[s][k]:15.4f}" for s in steps_deg) + f"   {rel * 100:7.2f} %")
    (OUT / "convergence_gapstep.json").write_text(json.dumps({str(k): v for k, v in rows.items()}, indent=1))


if __name__ == "__main__":
    main()
