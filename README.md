# Axial piston pump simulation (APP_sim)

Coupled simulation of a 9-piston swash-plate axial piston pump in Python:

- chamber pressures and port flows (lumped compressible chambers, orifice flows);
- piston, cylinder-block and swash-plate forces, drive and cradle torques;
- the three lubricating gaps (slipper / swash plate, cylinder block / valve plate, piston / bore),
  each solved with a 2D Reynolds equation, coupled to the body dynamics, with their leakage.

> **Demo data.** The pump parameters, CAD geometry and port-area table in this repository describe a
> fictitious pump (about 29 cm³/rev, 5000 rpm, 300 bar). They are not the data of any real product.

## Quick start

Use Python 3.13. The requirements pin the dependency versions used for validation.
Create and activate a virtual environment before installing them:

```powershell
py -3.13 -m venv .venv
.venv\Scripts\Activate.ps1
```

On Linux / macOS: `python3.13 -m venv .venv`, then `source .venv/bin/activate`.
With the environment active, run:

```bash
python -m pip install -r APP_sim/requirements.txt
python -m pytest APP_sim/tests
python -m APP_sim.run_no_leakage      # about 1 min
python -m APP_sim.run_with_leakage    # about 2 h
```

Run the commands from this folder (the one that contains `APP_sim/`).
Results go to `APP_sim/outputs/`.

GitHub Actions runs the tests on Linux and Windows with Python 3.13 for pushes and pull requests.

See [APP_sim/README.md](APP_sim/README.md) for outputs, flow definitions, parameters and assumptions,
and [APP_sim/CSV_GLOSSARY.md](APP_sim/CSV_GLOSSARY.md) for every CSV column.

## License

MIT, see [LICENSE](LICENSE).
