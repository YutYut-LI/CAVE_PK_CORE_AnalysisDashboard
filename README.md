# CAVE–PK CO₂ Analysis Dashboard

Interactive [Streamlit](https://streamlit.io) dashboard for **CAVE and PK CO₂ and temperature campaign analysis**. Upload experiment exports in the browser, configure thresholds and exclusions, and generate time-series metrics, zone summaries, infiltration indicators, and publication-style plots.

This repository contains **code only** — no raw experiment files. After deployment, open the shared Streamlit Cloud link and upload Explora, stage log, and MFC files in the sidebar.

---

## What this app does

- Loads **Explora** sensor exports (CO₂, temperature, wall, sensor ID, optional height `z`)
- Optionally aligns analysis to **experiment stages** from an Excel log
- Optionally overlays **MFC release** timing for infiltration / excess CO₂ metrics
- Produces CAVE vs PK views: overall metrics, zone CO₂/temperature, wall profiles, scatter and I/O ratios, downloadable CSV summaries

All inputs are provided via **`st.file_uploader`** — there are no hard-coded local file paths.

---

## Project layout (commit these to GitHub)

```
.
├── app.py                 # Streamlit entry point (was CAVE_PK_CO2_Temp_Metrics.py)
├── requirements.txt       # Python dependencies for local run & Streamlit Cloud
├── README.md
├── .gitignore
└── sample_data/
    └── README.md          # Describes upload formats; no real datasets
```

**Do not commit:** `.venv/`, `__pycache__/`, `.streamlit/cache/`, raw campaign folders, or any `*.csv` / `*.xlsx` / `*.xls` / `*.xlsm` data files (see `.gitignore`).

---

## Local setup

### 1. Clone and enter the repo

```bash
git clone <your-github-repo-url>
cd CAVE_PK_CORE_AnalysisDashboard
```

### 2. Create a virtual environment (recommended)

**Windows (PowerShell):**

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Run the dashboard

```bash
streamlit run app.py
```

Open the URL shown in the terminal (usually `http://localhost:8501`). Upload your Explora file in the sidebar, adjust settings, and click **Run analysis**.

---

## Deploy to Streamlit Community Cloud

1. Push this repository to **GitHub** (private or public). Confirm no data files are tracked: `git status` should not list `.csv` / `.xlsx` under the repo.
2. Sign in at [share.streamlit.io](https://share.streamlit.io) with your GitHub account.
3. **Create app** → select the repository and branch.
4. Set **Main file path** to: `app.py`
5. Leave **Requirements file** as: `requirements.txt` (auto-detected).
6. Deploy. Share the public URL with your group (e.g. `https://<app-name>.streamlit.app`).

Users open the link, upload files in the sidebar, and run analysis — no data is stored in the GitHub repo.

### Security reminder

- Do **not** upload raw experimental datasets to GitHub.
- Do **not** put credentials in the repo; use [Streamlit secrets](https://docs.streamlit.io/develop/concepts/connections/secrets-management) only if you add external services later (`.streamlit/secrets.toml` is gitignored).

---

## Requirements

- Python **3.10+** (3.11 or 3.12 recommended; 3.13 supported with recent package versions)
- Dependencies: see `requirements.txt` (`streamlit`, `pandas`, `numpy`, `matplotlib`, `plotly`, `openpyxl`, `pyarrow`)

---

## Support

For analysis parameters and file column expectations, see `sample_data/README.md` and the in-app sidebar labels.
