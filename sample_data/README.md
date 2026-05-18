# Sample data (not included in this repository)

This folder is intentionally **empty of real experiment files**.

The dashboard loads data through the **sidebar file uploaders** in the web app. Do not commit campaign CSV/XLSX files to GitHub.

## Files you upload at runtime

| Upload slot | Required | Formats | Purpose |
|-------------|----------|---------|---------|
| **Explora file** | Yes | `.csv`, `.xlsx`, `.xlsm`, `.xls` | Sensor CO₂ and temperature time series (columns such as `timestamp`, `co2`, `temperature`, `sensor_number`, `wall`) |
| **Experiment log / stage file** | No | `.xlsx`, `.xlsm`, `.xls` | Stage windows (sheet `Summary Experiment Stages`) |
| **MFC file** | No | `.csv` | Mass-flow controller release trace (`Timestamp`, `Fsetpoint`, `Fmeasure`) |

## Local testing

Keep copies of anonymised or synthetic files on your machine **outside** this repo, or in a local folder that is listed in `.gitignore` (for example `data/` or `raw/`).

## Optional demo files

If the team later needs a tiny public demo dataset, add only **non-sensitive, synthetic** files here and update `.gitignore` with explicit `!sample_data/demo_*.csv` exceptions. Until then, this directory contains documentation only.
