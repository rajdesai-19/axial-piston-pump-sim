"""
run_with_leakage.py - Run 2 of 2: coupled solver WITH leakage (slipper, valve-plate and
piston-bore films).

Cold start (p_initial everywhere), 3 revolutions, gap solves every 0.5 deg, exported at 720 points per
revolution. Takes about 2 h.

Outputs (APP_sim/outputs/with_leakage/):
  results.csv, results_glossary.md / .csv, figures/*.png,
  contours/{slipper1,valveplate,piston1}_{000,090,180,270}deg.png  (last revolution)

Run:  py -m APP_sim.run_with_leakage
"""
from pathlib import Path
import time
import numpy as np

from .params import DEFAULT
from .solver import CoupledSolver, SolverOptions, summarize
from .post.table import sample_fine, build_columns, write_csv_and_glossary, _gap_index
from .post.figures import all_figures
from .post.contours import all_contours

OUT = Path(__file__).resolve().parent / "outputs" / "with_leakage"
REVOLUTIONS = 3
TITLE = "With leakage"


def main(prm=DEFAULT, opt=None, out=OUT):
    t0 = time.time()
    out.mkdir(parents=True, exist_ok=True)
    opt = opt or SolverOptions(revolutions=REVOLUTIONS, init_from="cold")
    res = CoupledSolver(prm, opt).run()
    steps, n = res["steps"], res["revs"] * res["steps"]
    H = res["fine"]
    S = sample_fine(dict(p=H["p"], p_d=H["p_d"], p_s=H["p_s"], q_port=H["q_port"], q_out=H["q_out"],
                         q_load=H["q_load"], q_leak=H["q_leak"]),
                    np.arange(n) * res["dt"], np.arange(n) * 2 * np.pi / steps, steps, res["revs"])
    G = res["gap"]
    cols, _ = build_columns(S, prm, G)
    df = write_csv_and_glossary(cols, out / "results.csv", "APP_sim " + TITLE)
    files = all_figures(df, S, out / "figures", TITLE, G, _gap_index(G, S["k"]))
    cont = all_contours(res["fields"], out / "contours", prm)
    s = summarize(res)
    with open(out / "summary_last_revolution.txt", "w") as f:
        for k, v in s.items():
            f.write(f"{k:18s} {v:12.4f}\n")
    print(f"wrote {out / 'results.csv'} ({len(df)} rows, {df.shape[1]} columns), glossary, {len(files)} figures, "
          f"{len(cont)} contour plots ({time.time() - t0:.0f} s)")
    return df


if __name__ == "__main__":
    main()
