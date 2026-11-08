"""Regression checks for plots with and without the optional piston-bore film."""
import numpy as np
import pandas as pd
import pytest

from APP_sim.post.figures import fig_dashboard, fig_leakage


@pytest.mark.parametrize("piston_gap", [False, True])
def test_leakage_plots_support_optional_piston_gap(tmp_path, piston_gap):
    columns = """
        P1_p_bar Pd_bar Ps_bar Q_discharge_LPM Q_pump_LPM eta_vol_rev_pct
        P1_F_AK_N P1_F_SK_N SL_all_leak_LPM VP_all_leak_LPM LEAK_all_LPM
        SL1_h0_um VP_h0_um BLK_F_bearing_N BLK_F_clamp_N ALL_T_cradle_Nm
        T_drive_Nm SL1_leak_LPM VP_K1_leak_LPM LEAK_P1_LPM eta_vol_inst_pct
    """.split()
    if piston_gap:
        columns += ["PB_all_leak_LPM", "PB1_leak_LPM", "PB1_hmin_um"]
    df = pd.DataFrame({c: np.ones(8) for c in columns})
    df["phi_abs_deg"] = np.arange(8) * 90.0
    df["revolution"] = np.repeat([1, 2], 4)

    dashboard = tmp_path / "dashboard.png"
    leakage = tmp_path / "leakage.png"
    fig_dashboard(df, dashboard, "With leakage", leak=True)
    fig_leakage(df, leakage, "With leakage")

    for path in (dashboard, leakage):
        assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
