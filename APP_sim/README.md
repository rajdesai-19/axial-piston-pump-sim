# APP_sim: coupled axial piston pump solver

> **Demo data.** All pump parameters (`params.py`), the valve-plate and barrel geometry (`cad_data/*.dxf`) and the port-area table (`chamber.py`) describe a fictitious ~29 cm³/rev pump. They are not the data of any real product. Replace them with your own pump.

Coupled solver for a 9-piston swash-plate pump: chamber pressures, piston / block / swash-plate forces, and the slipper, valve-plate and piston-bore lubricating films (2D Reynolds) with their leakage.

- **Two entry scripts:** one run without leakage and one with leakage.
- **Standard export:** revolutions 1–3 from a cold start, 720 points per revolution, so one row every 0.5°.
- **One CSV per run,** with a glossary generated from the same column list.
- **Figures:** dashboards for chamber, kinematics, forces, swash plate and the three gaps.
- **Contours:** pressure and leakage for all three gaps at 0/90/180/270°.
- **No NPZ files from the two entry scripts.** Optional validation tools write intermediate NPZ files.

The simulation uses a right-handed coordinate frame: +z points toward the valve plate, and the shaft angle φ is measured clockwise seen from +z. φ = 0 is near BDC (BDC ≈ 351°, TDC ≈ 171° for the demo swash angles), and discharge runs over 0–180°.

The equations and coordinate conventions are documented in [`frame.py`](frame.py); the coupled solution procedure is described in [`solver.py`](solver.py). No external PDF or flowchart is required. Progress is printed to the terminal.

## Run

Run from the folder that **contains** `APP_sim/` (not from inside it). Use Python 3.13 in an activated virtual environment (see the [setup instructions](../README.md)). Install the pinned requirements first:

```bash
python -m pip install -r APP_sim/requirements.txt
```

The run without leakage uses the chamber model only and takes about 1 min:

```bash
python -m APP_sim.run_no_leakage
```

The run with leakage solves the slipper, valve-plate and piston–bore films, and takes about 2 h:

```bash
python -m APP_sim.run_with_leakage
```

Both runs start cold, at p_initial everywhere, and simulate 3 revolutions. Revolution 1 contains the start-up transient; revolutions 2–3 are at steady state.

Unit tests:

```bash
python -m pytest APP_sim/tests
```

## Outputs (`APP_sim/outputs/`)

The folder is created when you run a script (it is not in the repository).

| Path | Content |
|---|---|
| `no_leakage/results.csv` | 2160 rows: hydraulics, kinematics, torques, and the external forces on the piston, block and swash plate. |
| `with_leakage/results.csv` | The same columns, plus the slipper, valve-plate and piston–bore films, leakage, forces and moments, and the total leakage. |
| `*/results_glossary.md`, `*/results_glossary.csv` | Every column with its unit and meaning, plus the prefixes, unit suffixes and short terms. |
| `*/figures/00_dashboard.png` | Summary dashboard. |
| `*/figures/01_chamber_pressures_flows.png` | Chamber and line pressures, piston 1 port flow, discharge and pump flow, volumetric efficiency. |
| `*/figures/02_kinematics.png` | Piston 1 stroke, velocity, acceleration, moment arm, volume and engagement length. |
| `*/figures/03_piston_block_forces.png` | Piston 1 axial and radial forces, block clamping, radial load and bearing, drive torque, block tilting moments. |
| `*/figures/04_swashplate_cradle_torque.png` | Swash-plate forces and moments; cradle torque of piston 1 and of all 9 pistons (pressure and inertia). |
| `with_leakage/figures/05_slipper_gap.png` | Slipper films, tilts, pocket pressure, leakage, forces and moments. |
| `with_leakage/figures/06_valveplate_gap.png` | Valve-plate films, leakage, forces and moments. |
| `with_leakage/figures/07_piston_bore_gap.png` | Piston eccentricity, films, leakage, forces and moments. |
| `with_leakage/figures/08_leakage_breakdown.png` | Leakage by gap, for chamber 1 and for all chambers; mean per revolution; volumetric efficiency. |
| `with_leakage/contours/{slipper1,valveplate,piston1}_{000,090,180,270}deg.png` | Pressure and leakage-flux contours in revolution 3. |
| `with_leakage/summary_last_revolution.txt` | Mean values of revolution 3. |

### Flow definitions
- **`P1_Q_port_LPM`:** chamber 1's flow through its valve-plate kidney. Positive means out of the chamber.
- **`Q_discharge_LPM`:** the sum of all kidney flows into the delivery line.
- **`Q_pump_LPM`:** the flow through the load orifice, i.e. what the pump delivers.
- **`eta_vol_*`:** `Q_pump` divided by the geometric flow 9·Ak·stroke·n. It is given both instantaneously and as a revolution mean.

## Parameters

All parameters live in `params.py`. To change one for a single run:

```python
from APP_sim.params import DEFAULT
from APP_sim import run_no_leakage
run_no_leakage.main(DEFAULT.with_(rpm=4000))
```

For the run with leakage, `run_with_leakage.main(prm, SolverOptions(...))` also accepts `gap_step_deg` and `field_angles_deg`.

## Layout

```
APP_sim/
  run_no_leakage.py    entry script 1
  run_with_leakage.py  entry script 2
  params.py  frame.py  chamber.py  solver.py
  forces/              piston.py, block.py, swashplate.py
  gaps/                polar.py, cylinder.py (Reynolds solvers), slipper.py, valveplate.py, piston.py, cad.py
  post/                table.py (CSV + glossary), figures.py, contours.py, style.py
  tools/               phase checks (optional), run with python -m APP_sim.tools.<name>:
                       validate_chamber, validate_forces, validate_gaps [piston], convergence_gapstep
  tests/               unit tests (formula regression cases, analytic gap solutions)
  cad_data/            vp.dxf, barrel_top.dxf (synthetic demo geometry)
  outputs/             results (created by the runs)
  CSV_GLOSSARY.md      glossary of all CSV columns
  requirements.txt
```

`tools/validate_chamber` writes the chamber npz file that `validate_forces` and `validate_gaps` read, into `APP_sim/outputs/tools/`.

## Assumptions

- The films are isothermal.
- The bearing carries the block's tilting moment.
- Film cavitation is limited at 0 bar absolute.
- The chamber pressure has a floor at 0.1 bar.
- Contact is a quadratic penalty, so the contact forces depend on the penalty values. The leakage does not.
