"""
run_no_leakage.py - Run 1 of 2: chamber pressure model WITHOUT leakage (no gap films).

Cold start (p_initial everywhere), 3 revolutions, exported at 720 points per revolution.
External forces, torques and kinematics are evaluated from the chamber pressures.

Outputs (APP_sim/outputs/no_leakage/):
  results.csv, results_glossary.md / .csv, figures/*.png

Run:  py -m APP_sim.run_no_leakage
"""
from pathlib import Path
import time

from .params import DEFAULT
from .chamber import simulate
from .post.table import sample_fine, build_columns, write_csv_and_glossary
from .post.figures import all_figures

OUT = Path(__file__).resolve().parent / "outputs" / "no_leakage"
REVOLUTIONS = 3
TITLE = "Without leakage"


def main(prm=DEFAULT, revolutions=REVOLUTIONS, out=OUT):
    t0 = time.time()
    out.mkdir(parents=True, exist_ok=True)
    res = simulate(prm, revolutions=revolutions)
    H = dict(p=res["p"], p_d=res["p_d"], p_s=res["p_s"], q_port=res["q_d"] + res["q_s"],
             q_out=res["q_out"], q_load=res["q_load"])
    S = sample_fine(H, res["time"], res["phi_abs"], res["steps"], revolutions)
    cols, _ = build_columns(S, prm)
    df = write_csv_and_glossary(cols, out / "results.csv", "APP_sim " + TITLE)
    files = all_figures(df, S, out / "figures", TITLE)
    print(f"wrote {out / 'results.csv'} ({len(df)} rows, {df.shape[1]} columns), glossary, {len(files)} figures "
          f"({time.time() - t0:.0f} s)")
    return df


if __name__ == "__main__":
    main()
