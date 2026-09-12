# XiChen: A global weather observation-to-forecast machine learning system via four-dimensional variational gradient-guided flexible assimilation

Official **inference-only** release of **XiChen**, an end-to-end AI weather
system that couples a medium-range global forecaster with learned radiance
observation operators and a cascade of 4DVar-gradient-conditioned data
assimilation (DA) networks — all at 1.0° (181×360, 69 channels).

> Manuscript: *XiChen: A global weather observation-to-forecast machine learning system via four-dimensional variational gradient-guided flexible assimilation* (under revision).
> Checkpoints & demo data: **Zenodo DOI `<10.5281/zenodo.XXXXXXX>`**
> (placeholder, filled in at upload).

With the released checkpoints you can:

1. **Forecast** — run a 10-day autoregressive global forecast from an ERA5 initial condition (five cascaded sub-models, 24/12/6/3/1 h steps).
2. **Assimilate** — cycle the cascade DA (12 h windows, 6 h re-analyses at 00/12 UTC) over six observation streams: ATMS, AMSU-A, MHS, HRS4 (through learned observation operators) + prepbufr, SATWND (identity H).
3. **Chain them** — launch the 10-day forecast *from the DA analysis*, the full ERA5 + observations → analysis → forecast loop of the paper.

## Installation

```bash
git clone <repo-url> XiChen_1p0deg_public
cd XiChen_1p0deg_public
pip install -e .            # core library + CLI
pip install -e .[notebooks] # + jupyter / cartopy for the demo notebooks
```

Requirements: Python ≥ 3.10, PyTorch ≥ 2.0, and the runtime dependency `omegaconf` (a CUDA GPU is strongly recommended; CPU works but is very slow). The released checkpoints embed the training-time OmegaConf (Hydra) config, so they load via full deserialization (`weights_only=False`) — only load checkpoints whose source you trust.

## Download weights and demo data

Everything is on Zenodo (DOI placeholder above). Two archives:

| Archive                         | Contents                                                   | Unpacks to |
| ------------------------------- | ---------------------------------------------------------- | ---------- |
| `xichen_1p0deg_ckpts_v1.tar.gz` | 6 checkpoints (see table below)                            | `ckpts/`   |
| `xichen_1p0deg_data_v1.tar.gz`  | demo ERA5 + observations + norms/climatology + R estimates | `data/`    |

| Checkpoint                                                            | Role                                              |
| --------------------------------------------------------------------- | ------------------------------------------------- |
| `xichen_state_forecast_ar15.ckpt`                                     | state forecaster (AR-15 fine-tuned)               |
| `xichen_obsop_{atms,amsua,mhs,hrs4}.ckpt`                             | 4 radiance observation operators                  |
| `xichen_cascade_da_randombg_atms_amsua_mhs_hrs4_prepbufr_satwnd.ckpt` | cascade DA (6 per-stream DA networks in one file) |

Either download manually from the Zenodo page and unpack at the repo root,
or use the bundled helper (pure stdlib, verifies SHA-256):

```bash
python scripts/download.py                # both archives
python scripts/download.py --ckpts-only   # just checkpoints
python scripts/download.py --data-only    # just demo data
```

> **Note:** until the Zenodo record is published, `download.py` exits with a
> "placeholder URL/checksum" error — the DOI and SHA-256 digests are filled in
> at upload time. Download the two archives manually until then.

Expected layout after unpacking:

```
XiChen_1p0deg_public/
├── ckpts/                 # 6 *.ckpt
└── data/
    ├── era5/              # 1.0° npy archive (see Data formats)
    │   ├── normalized_mean_std/normalize_{mean,std}.npz
    │   ├── climatology_np181x360_2010_2021/<MM-DD>/<var>.npy
    │   └── 2023/2023-01-*/HH:MM:SS.npy
    └── observation/
        ├── 1b{atms,amsua,mhs,hrs4}_merged_npy_1.0deg/   # + avg_obs_error.npz, scalers, schema
        ├── GDAS_prepbufr_merged_npy_1.0deg/             # + obs_sigma.npz
        └── satwnd_merged_npy_1.0deg/                    # + obs_sigma.npz
```

## Quick start

Run from the repository root. Paths are resolved relative to the current
working directory (configs use repo-relative paths), so invoking from a
different `cwd` will not find the data.

```bash
# 1) 10-day forecast from an ERA5 initial condition
xichen-forecast \
  --ckpt ckpts/xichen_state_forecast_ar15.ckpt \
  --era5_lr_dir data/era5 \
  --init_times_json configs/demo_init_times.json \
  --output_dir outputs/era5_init_forecast

# 2) cascade DA demo: 3 cycles from a climatology cold start
xichen-dacycle --config configs/dacycle_demo.json

# 3) 10-day forecast from the DA analysis produced in step 2
xichen-da-forecast \
  --ic_dir outputs/demo_dacycle/ic_6h \
  --init_times_json configs/demo_init_times.json \
  --ckpt ckpts/xichen_state_forecast_ar15.ckpt \
  --era5_lr_dir data/era5 \
  --output_dir outputs/da_init_forecast

# (optional) re-estimate observation errors (R) with an observation operator
xichen-obsop --obs_name atms --era5_dir data/era5 --obs_dir data/observation \
  --ckpt ckpts/xichen_obsop_atms.ckpt --debug
```

> `xichen-obsop` writes the re-estimated `avg_obs_error.npz` into its plot
> output directory (under `--save_dir`), not into the observation stream dir
> that the DA cycle reads from. To feed a re-estimated R into
> `xichen-dacycle`, copy it into `data/observation/1b<sat>_merged_npy_1.0deg/`
> manually.

## Demo notebooks

See `notebooks/` — run them in order; each is self-contained and plots its
results inline:

| Notebook                         | Shows                                                                                   | Runtime (1 GPU) |
| -------------------------------- | --------------------------------------------------------------------------------------- | --------------- |
| `01_medium_range_forecast.ipynb` | 10-day ERA5-init forecast: RMSE/ACC curves + z-500 field/error maps                     | ~2–5 min        |
| `02_dacycle.ipynb`               | 3× cascade DA cycles from climatology: analysis increments, xb-vs-xa RMSE, obs coverage | ~5–10 min       |
| `03_da_init_forecast.ipynb`      | 10-day forecast from the `ic_6h` analysis, vs the ERA5-init baseline                    | ~3–5 min        |

Notebook 03 reads the analysis written by notebook 02 and the baseline
metrics saved by notebook 01, so run 01 → 02 → 03.

## Repository structure

```
├── xichen/                  # the package (pure PyTorch, no training code)
│   ├── layers/              # Swin-Transformer V2 building blocks
│   ├── models/              # forecast.py / obsoperator.py / da.py / cascade.py (Solver)
│   │                        #   varcost.py (4DVar cost) / ar.py (trajectory rollout)
│   ├── data.py              # VARIABLES (69 channels), normalization, obs readers
│   ├── nwp.py               # NWP-Benchmark NetCDF I/O
│   ├── ckpt.py              # checkpoint loading (flat .ckpt or run dir)
│   ├── metrics.py           # latitude-weighted RMSE / ACC / activity
│   ├── forecast_eval.py     # eval_forecast + AR rollout + init-time loaders
│   ├── obsop_eval.py        # observation-operator OMB evaluation
│   ├── dacycle/             # common.py (loop, models, obs window) + det.py (one cycle)
│   └── cli/                 # xichen-{forecast,dacycle,da-forecast,obsop} entry points
├── configs/                 # model hyper-parameters + demo / one-year DA cycle configs
├── notebooks/               # the three demos above
├── scripts/download.py      # Zenodo downloader (stdlib only)
└── scripts/check_imports.py # import hygiene check
```

## Data formats

**ERA5 (state) archive** — one float32 npy per validity time, shape
`(69, 181, 360)` in physical units, channel order `xichen.data.VARIABLES`
(4 surface + 13 pressure levels × {z, u, v, t, q}):

```
data/era5/<YYYY>/<YYYY-MM-DD>/<HH:MM:SS>.npy
```

Normalization constants `data/era5/normalized_mean_std/normalize_{mean,std}.npz`
are per-variable dicts (`np.load(path)[var]`), used as `(x − mean) / std`.
Climatology for ACC lives in
`data/era5/climatology_np181x360_2010_2021/<MM-DD>/<var>.npy`.

**Observations** — per 3 h slot, per stream:

```
data/observation/1b<sat>_merged_npy_1.0deg/<YYYY>/<YYYY-MM-DD>/<HH:MM:SS>-{auxiliary_value,<radiance>,mask}.npy
data/observation/GDAS_prepbufr_merged_npy_1.0deg/...  (same time layout)
data/observation/satwnd_merged_npy_1.0deg/...
```

Per-stream radiance file name (the `<radiance>` token differs by instrument):
`atms` uses `brightness_temperature_value.npy`; `amsua` / `mhs` / `hrs4` use
`tmbrs_value.npy`. Both share the same time prefix. Conventional streams
(`prepbufr`, `satwnd`) read `-obs_value.npy` plus an external `_qcmask.npy`
quality mask.

Each stream directory also carries its scalers/schema
(`*_scaler.npz`, `<sat>_1.0deg_schema.json`, `normalize_{mean,std}.npz`) and
the DA observation-error estimates (`avg_obs_error.npz` for microwave,
`obs_sigma.npz` for conventional). Missing time slots are handled gracefully
(zero-filled with zero mask).

The demo dataset covers 2023-01-05 00 UTC → 2023-01-06 00 UTC for cycling
(obs at every 3 h) plus ERA5 truth out to 2023-01-16 00 UTC for the 10-day
forecasts. Note the last demo cycle (2023-01-06T00) opens a 12 h window that
extends past the archived obs coverage — those later slots are zero-masked,
so that cycle's analysis is effectively background-limited. To run your own
period, mirror this layout with the full ERA5 / observation archives and
point the configs at it.

## Reproducing paper results

The one-year cycling experiment uses `configs/dacycle_oneyear.json`, whose
paths are environment-variable templates:

```bash
export XICHEN_DATA_DIR=/path/to/full/data     # contains era5/ and observation/
export XICHEN_CKPT_DIR=/path/to/ckpts
xichen-dacycle --config configs/dacycle_oneyear.json \
  --output_root /path/to/results --task_name_out cycle_da_2023
```

Reference cost: roughly a day on a single modern GPU for the full archive
(784 cycles; `configs/dacycle_oneyear.json` spans 2022-12-05T00 →
2023-12-31T12 at 12-h steps). Medium-range scores vs GFS/IFS-HRES, observation-impact
ablations and the TC case study are evaluated with the same CLIs over the
full archives (see the paper for protocol details).

## License & Citation

Code: [Apache-2.0](LICENSE). Please check the Zenodo record for the license
of the checkpoints and data.

```bibtex
@misc{wang2026xichen,
      title={XiChen: A global weather observation-to-forecast machine learning system via four-dimensional variational gradient-guided flexible assimilation}, 
      author={Wuxin Wang and Weicheng Ni and Lilan Huang and Tao Hao and Ben Fei and Shuo Ma and Taikang Yuan and Yanlai Zhao and Kefeng Deng and Xiaoyong Li and Hongze Leng and Boheng Duan and Lei Bai and Weimin Zhang and Junqiang Song and Kaijun Ren},
      year={2026},
      eprint={2507.09202},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2507.09202}, 
}
```
