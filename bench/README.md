## Benchmarks

This directory contains the benchmark harness and artifacts used to compare `riv2` against reference codecs.

- **Sequences**: pre‑converted PNG image sequences live in `image_sequences/` (run `convert.py` to generate them from `.y4m` sources).
- **Runner**: `run.py` encodes and decodes with multiple codecs, then computes per‑sequence **PSNR** and **SSIM**.
- **Results**: per‑sequence JSON files like `results_foreman_cif.json` are written next to this README.
- **Plots**: `plot.py` turns result JSON into `psnr_*.png` and `ssim_*.png` rate–distortion curves.

### Running benchmarks

- **All codecs, one sequence (full sweep)**:

```bash
python3 bench/run.py --seq foreman_cif
```

- **Single codec** (e.g. only `riv2`):

```bash
python3 bench/run.py --seq foreman_cif --codec riv2
```

- **Quick subset of quality points**:

```bash
python3 bench/run.py --seq foreman_cif --quick
```

- **Regenerate plots for all sequences**:

```bash
python3 bench/plot.py
```

