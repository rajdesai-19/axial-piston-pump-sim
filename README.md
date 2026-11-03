# Axial piston pump simulation (pump_sim_v2)

Coupled simulation of a 9-piston swash-plate axial piston pump in Python:

- chamber pressures and port flows (lumped compressible chambers, orifice flows);
- piston, cylinder-block and swash-plate forces, drive and cradle torques;
- the three lubricating gaps (slipper / swash plate, cylinder block / valve plate, piston / bore),
  each solved with a 2D Reynolds equation, coupled to the body dynamics, with their leakage.

> **Demo data.** The pump parameters, CAD geometry and port-area table in this repository describe a
> fictitious pump (about 29 cm³/rev, 5000 rpm, 300 bar). They are not the data of any real product.

## Quick start

```bash
pip install -r pump_sim_v2/requirements.txt
py -m pytest pump_sim_v2/tests
py -m pump_sim_v2.run_no_leakage      # about 1 min
py -m pump_sim_v2.run_with_leakage    # about 2 h
```

Run the commands from this folder (the one that contains `pump_sim_v2/`). On Linux / macOS use `python` instead of `py`.
Results go to `pump_sim_v2/outputs/`.

See [pump_sim_v2/README.md](pump_sim_v2/README.md) for outputs, flow definitions, parameters and assumptions,
and [pump_sim_v2/CSV_GLOSSARY.md](pump_sim_v2/CSV_GLOSSARY.md) for every CSV column.

## License

MIT, see [LICENSE](LICENSE).
