"""
validate_chamber.py - Phase 1 checks for APP_sim/chamber.py.

  1. steady-state volume balance (pump out = load, inlet = suction ports)
  2. delivered flow vs geometric displacement, with the loss breakdown
  3. time-step convergence (steps_per_deg 100 / 200 / 400)
Writes the cycles 1-5 export of the baseline run to APP_sim/outputs/tools/.

Run:  py -m APP_sim.tools.validate_chamber
"""
from pathlib import Path
import numpy as np
import pandas as pd

from ..params import DEFAULT
from ..frame import SwashFrame
from ..chamber import simulate, cycle_slice, export_cycles_csv, export_cycles_npz

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "tools"
LPM = 6.0e4


def steady_stats(res, prm):
    sl = cycle_slice(res, -1)
    dt = res["dt"]
    q_d = res["q_d"][:, sl]
    q_s = res["q_s"][:, sl]
    st = dict(
        Pd=res["p_d"][sl].mean() / 1e5,
        Pd_pp=(res["p_d"][sl].max() - res["p_d"][sl].min()) / 1e5,
        Ps=res["p_s"][sl].mean() / 1e5,
        Q_out=res["q_out"][sl].mean() * LPM,
        Q_load=res["q_load"][sl].mean() * LPM,
        Q_inlet=res["q_inlet"][sl].mean() * LPM,
        Q_suction_ports=q_s.sum(axis=0).mean() * LPM,
        # reverse flows: delivery -> chamber (q_d > 0) and chamber -> suction (q_s < 0)
        backflow_delivery=q_d.clip(min=0).sum(axis=0).mean() * LPM,
        backflow_suction=(-q_s).clip(min=0).sum(axis=0).mean() * LPM,
        P1_max=res["p"][0, sl].max() / 1e5, P1_min=res["p"][0, sl].min() / 1e5,
        dPd_last_two=(res["p_d"][sl].mean() - res["p_d"][cycle_slice(res, -2)].mean()) / 1e5,
    )
    return st


def main():
    prm = DEFAULT
    fr = SwashFrame(prm)
    Q_th = prm.num_piston * prm.Ak * fr.stroke * prm.rpm * 1e3
    print(f"Geometric displacement flow Q_th = {Q_th:.2f} LPM, V_tdc = {fr.V_tdc * 1e9:.1f} mm^3, "
          f"BDC = {np.degrees(fr.phi_bdc):.2f} deg, TDC = {np.degrees(fr.phi_tdc):.2f} deg\n")

    runs = {}
    for spd in (200, 100, 400):
        print(f"--- steps_per_deg = {spd} ---")
        p = prm.with_(steps_per_deg=spd)
        res = simulate(p, verbose=False)
        runs[spd] = steady_stats(res, p)
        if spd == 200:
            export_cycles_csv(res, OUT / "chamber_cycles_1-5.csv", p)
            export_cycles_npz(res, OUT / "chamber_cycles_1-5_full.npz", p)
            base = res
        print({k: round(v, 4) for k, v in runs[spd].items()})

    print("--- eta = 0 (no dead-centre / port-timing mismatch) ---")
    p0 = prm.with_(eta_deg=0.0)
    s0 = steady_stats(simulate(p0, verbose=False), p0)
    Q_th0 = prm.num_piston * prm.Ak * SwashFrame(p0).stroke * prm.rpm * 1e3
    print({k: round(v, 4) for k, v in s0.items()}, f"Q_th={Q_th0:.2f}")

    s = runs[200]
    print("\n================ PHASE 1 CHECKS ================")
    bal = abs(s["Q_out"] - s["Q_load"]) / s["Q_load"] * 100
    sbal = abs(s["Q_inlet"] - s["Q_suction_ports"]) / s["Q_inlet"] * 100
    print(f"[1] steady volume balance  out vs load : {bal:.3f} %   (limit 0.1 %)  {'PASS' if bal < 0.1 else 'FAIL'}")
    print(f"    suction inlet vs ports            : {sbal:.3f} %   (limit 0.1 %)  {'PASS' if sbal < 0.1 else 'FAIL'}")
    print(f"    Pd drift last two revs            : {s['dPd_last_two']:+.3f} bar")
    loss = Q_th - s["Q_out"]
    print(f"[2] Q_out = {s['Q_out']:.2f} LPM vs Q_th = {Q_th:.2f} LPM  ->  deficit {loss:.2f} LPM ({loss / Q_th * 100:.2f} %)")
    print(f"    reverse flow delivery->chamber    : {s['backflow_delivery']:.2f} LPM")
    print(f"    reverse flow chamber->suction     : {s['backflow_suction']:.2f} LPM")
    print(f"    same with eta = 0: deficit {Q_th0 - s0['Q_out']:.2f} LPM ({(Q_th0 - s0['Q_out']) / Q_th0 * 100:.2f} %)")
    d1 = abs(runs[100]["Pd"] - runs[200]["Pd"]); d2 = abs(runs[400]["Pd"] - runs[200]["Pd"])
    print(f"[3] dt convergence mean Pd: 100->{runs[100]['Pd']:.3f}, 200->{runs[200]['Pd']:.3f}, 400->{runs[400]['Pd']:.3f} bar "
          f"(max diff {max(d1, d2):.3f} bar, limit 0.1)  {'PASS' if max(d1, d2) < 0.1 else 'FAIL'}")
    print(f"    P1 max: 100->{runs[100]['P1_max']:.2f}, 200->{runs[200]['P1_max']:.2f}, 400->{runs[400]['P1_max']:.2f} bar")

    print(f"\nExports: {OUT / 'chamber_cycles_1-5.csv'}\n         {OUT / 'chamber_cycles_1-5_full.npz'}")


if __name__ == "__main__":
    main()
