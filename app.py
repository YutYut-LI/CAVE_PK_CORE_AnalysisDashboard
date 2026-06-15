# CAVE–PK CO2 / temperature analysis dashboard (Streamlit entry point).
# Former script name: CAVE_PK_CO2_Temp_Metrics.py
# Run locally: streamlit run app.py

from __future__ import annotations

import html
import hashlib
import io
import re
import traceback
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Any

import numpy as np
import pandas as pd
import streamlit as st
import datetime as _dt
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch

try:
    import plotly.colors as pc
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except Exception:  # pragma: no cover
    pc = None  # type: ignore
    go = None  # type: ignore
    make_subplots = None  # type: ignore

# Plotly default qualitative palette (matches unset trace colours in charts).
_PLOTLY_SERIES_COLORS = (
    list(pc.qualitative.Plotly)
    if pc is not None
    else [
        "#636EFA",
        "#EF553B",
        "#00CC96",
        "#AB63FA",
        "#FFA15A",
        "#19D3F3",
        "#FF6692",
        "#B6E880",
        "#FF97FF",
        "#FECB52",
    ]
)


# =========================================================
# Page config
# =========================================================
st.set_page_config(
    page_title="CAVE–PK CO2 Analysis Dashboard",
    layout="wide",
)

# Plotly charts: fill Streamlit column width on resize (laptops / narrow windows).
st.markdown(
    """
    <style>
    div[data-testid="stPlotlyChart"] { width: 100% !important; max-width: 100%; }
    div[data-testid="stPlotlyChart"] .js-plotly-plot,
    div[data-testid="stPlotlyChart"] .plotly,
    div[data-testid="stPlotlyChart"] .plot-container.plotly {
        width: 100% !important;
        max-width: 100%;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

PLOTLY_CHART_CONFIG = {
    "responsive": True,
    "displayModeBar": True,
    "displaylogo": False,
}


# =========================================================
# Config dataclass
# =========================================================
@dataclass
class AppConfig:
    exp_code: str = "Experiment"
    align_to: str = "10s"
    min_sensors: int = 3
    coverage_factor: float = 1.20

    apply_cave_exclusions: bool = True
    exclude_fixtures: Tuple[str, ...] = ("supply", "extract")
    exclude_z_levels: Tuple[str, ...] = ("z1",)  # labels like z1 → levels from raw z (m)
    exclude_sensors: Tuple[int, ...] = (24, 25, 26)

    cave_z_low_min: float = 0.0
    cave_z_low_max: float = 2.0
    cave_z_high_min: float = 8.0
    cave_z_high_max: float = 10.0

    pk_low_z_levels: Tuple[str, ...] = ("z1", "z2")
    pk_high_z_levels: Tuple[str, ...] = ("z6", "z7")

    cave_walls_to_plot: Tuple[str, ...] = ("North", "East", "South (RSD)", "South", "West")

    plot_pre_min: int = 0
    use_fixed_ylims: bool = True

    abs_ex_thresh: float = 50.0
    baseline_fallback_minutes: int = 10
    flow_on_th: float = 0.2

    ylims: Dict[str, Tuple[float, float]] = None


def default_ylims():
    return {
        "co2_mean": (350, 1300),
        "co2_std": (0, 300),
        "co2_cv": (0.00, 0.60),
        "co2_mi": (0, 1.00),
        "co2_coverage": (0, 110),
        "temp_mean": (8, 30),
        "temp_std": (0.0, 5.0),
        "temp_deltaT": (-5, 15.0),
        "temp_r2": (0.0, 1.0),
        "temp_mi": (0, 1.00),
        "temp_pk_minus_cave": (0, 25),
        "zone_cave_co2": (350, 1300),
        "zone_pk_co2": (350, 1300),
        "rh_mean": (0, 100),
        "rh_std": (0, 25),
        "zone_cave_rh": (0, 100),
        "zone_pk_rh": (0, 100),
        "io_ex": (0.0, 1.00),
        "scatter_cave_ex": (0, 600),
        "scatter_pk_ex": (0, 300),
    }


def apply_matplotlib_publication_rc(base_pt: float) -> None:
    """Set matplotlib rcParams for dashboard + export (publication-friendly)."""
    b = float(base_pt)
    plt.rcParams.update(
        {
            "font.size": b,
            "axes.labelsize": b,
            "axes.titlesize": b + 1,
            "xtick.labelsize": max(b - 1, 8),
            "ytick.labelsize": max(b - 1, 8),
            "legend.fontsize": max(b - 2, 7),
            "figure.titlesize": b + 2,
            "axes.linewidth": 1.0,
            "axes.edgecolor": "black",
            "axes.labelcolor": "black",
            "xtick.color": "black",
            "ytick.color": "black",
            "text.color": "black",
            "legend.frameon": True,
            "legend.facecolor": "white",
            "legend.edgecolor": "black",
        }
    )


# =========================================================
# Helpers
# =========================================================
def parse_csv_or_excel(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    elif name.endswith((".xlsx", ".xlsm", ".xls")):
        return pd.read_excel(uploaded_file)
    else:
        raise ValueError(f"Unsupported file type: {uploaded_file.name}")


@st.cache_data(show_spinner=False)
def load_explora_any(file_bytes: bytes, filename: str) -> pd.DataFrame:
    bio = io.BytesIO(file_bytes)
    if filename.lower().endswith(".csv"):
        df = pd.read_csv(bio)
    elif filename.lower().endswith((".xlsx", ".xlsm", ".xls")):
        xl = pd.ExcelFile(bio)
        sheet = "Full" if "Full" in xl.sheet_names else xl.sheet_names[0]
        df = pd.read_excel(io.BytesIO(file_bytes), sheet_name=sheet)
    else:
        raise ValueError(f"Unsupported Explora file type: {filename}")

    df.columns = [str(c).strip().lower() for c in df.columns]

    time_candidates = ["timestamp", "time_europe_london", "time", "datetime", "date_time"]
    time_col = next((c for c in time_candidates if c in df.columns), None)
    if time_col is None:
        raise ValueError(f"Explora missing time column. Columns: {list(df.columns)}")

    required = ["co2", "temperature", "sensor_number", "wall"]
    for r in required:
        if r not in df.columns:
            raise ValueError(f"Explora missing '{r}'. Columns: {list(df.columns)}")

    df["time"] = pd.to_datetime(df[time_col], errors="coerce", dayfirst=True)
    df["co2"] = pd.to_numeric(df["co2"], errors="coerce")
    df["temperature"] = pd.to_numeric(df["temperature"], errors="coerce")
    df["sensor_number"] = pd.to_numeric(df["sensor_number"], errors="coerce").astype("Int64")

    hum_col = _detect_humidity_column(list(df.columns))
    if hum_col is not None:
        df["humidity"] = pd.to_numeric(df[hum_col], errors="coerce")
        if hum_col != "humidity":
            df.attrs["humidity_source_col"] = hum_col

    if "z" in df.columns:
        df["z"] = pd.to_numeric(df["z"], errors="coerce")

    for col in ["wall", "fixture", "structure"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    # Normalize wall names so downstream plots use consistent labels
    # (e.g., historic "South (MidWall)" should now be treated as "South (RSD)")
    if "wall" in df.columns:
        w = df["wall"].astype(str).str.strip()
        w_norm = w.str.replace(r"\s+", " ", regex=True)
        w_norm_lower = w_norm.str.lower()
        is_midwall = w_norm_lower.isin({"south (midwall)", "south(midwall)", "south (mid wall)", "south(mid wall)"})
        df.loc[is_midwall, "wall"] = "South (RSD)"

    df = df.dropna(subset=["time", "co2"]).copy()
    df = df.sort_values(["time", "sensor_number"]).reset_index(drop=True)
    return df


def _detect_humidity_column(columns) -> Optional[str]:
    """Return Explora humidity column name (lowercase headers) if present."""
    cols = [str(c).strip().lower() for c in columns]
    lookup = {c: c for c in cols}
    for key in (
        "humidity",
        "rh",
        "relative humidity",
        "relative_humidity",
        "humidity_%",
        "rh_%",
        "humidity percent",
        "humidity_percent",
    ):
        if key in lookup:
            return lookup[key]
    for c in cols:
        if "humid" in c or c == "rh" or c.startswith("rh_") or c.endswith("_rh"):
            return c
    return None


def humidity_has_data(df: Optional[pd.DataFrame]) -> bool:
    if df is None or df.empty or "humidity" not in df.columns:
        return False
    return bool(df["humidity"].notna().any())


@st.cache_data(show_spinner=False)
def load_stages_from_log(file_bytes: bytes, filename: str, sheet="Summary Experiment Stages"):
    bio = io.BytesIO(file_bytes)
    raw = pd.read_excel(bio, sheet_name=sheet, header=None)

    rows = []
    for i in range(len(raw)):
        s = raw.iloc[i, 1]
        if isinstance(s, str) and s.strip().lower().startswith("stage"):
            note = raw.iloc[i, 3]
            stt = pd.to_datetime(raw.iloc[i, 4], errors="coerce")
            ett = pd.to_datetime(raw.iloc[i, 5], errors="coerce")
            if pd.notna(stt) and pd.notna(ett):
                rows.append((str(note).strip(), stt, ett))
    return rows


def _clean_mfc_column_name(name: str) -> str:
    return str(name).strip().lstrip("\ufeff").strip()


def _find_mfc_column(columns, *candidates: str) -> Optional[str]:
    """Case-insensitive MFC column lookup (handles BOM / stray spaces)."""
    lookup = {_clean_mfc_column_name(c).lower(): c for c in columns}
    for cand in candidates:
        key = _clean_mfc_column_name(cand).lower()
        if key in lookup:
            return lookup[key]
    return None


def _normalize_mfc_col_key(name: str) -> str:
    s = _clean_mfc_column_name(name).lower()
    s = re.sub(r"\([^)]*\)|\[[^\]]*\]", "", s)
    s = re.sub(r"[\s_\-.°]+", "", s)
    return re.sub(r"[^a-z0-9]", "", s)


_MFC_TEMP_EXCLUDE_NORM = frozenset(
    {
        "timestamp",
        "time",
        "datetime",
        "date",
        "fsetpoint",
        "fmeasure",
        "fset",
        "fmeas",
        "flow",
        "flowrate",
        "flowsetpoint",
        "flowmeasure",
        "setpoint",
        "measure",
    }
)


def _detect_mfc_temperature_column(columns) -> Optional[str]:
    """Return the raw MFC CSV column name for temperature, if present."""
    cols = list(columns)
    lower = {_clean_mfc_column_name(c).lower(): c for c in cols}
    for key in (
        "temperature",
        "temp",
        "mfc_temperature",
        "mfc temp",
        "t_mfc",
        "t_c",
        "gas temperature",
        "gas temp",
        "cylinder temperature",
        "cylinder temp",
    ):
        if key in lower:
            return lower[key]

    norm_map: Dict[str, str] = {}
    for c in cols:
        nk = _normalize_mfc_col_key(c)
        if nk and nk not in norm_map:
            norm_map[nk] = c

    for key in (
        "temperature",
        "temp",
        "mfctemperature",
        "gastemperature",
        "gastemp",
        "cylindertemperature",
        "cylindertemp",
        "tmfc",
        "tc",
    ):
        if key in norm_map:
            return norm_map[key]

    for nk, orig in norm_map.items():
        if nk in _MFC_TEMP_EXCLUDE_NORM:
            continue
        if "temp" in nk or nk in ("t", "tc", "tmfc"):
            return orig
    return None


def _parse_mfc_numeric_series(series: pd.Series) -> pd.Series:
    """Parse numeric MFC fields; tolerate unit suffixes like '25.3 C' or '25,3'."""
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    as_str = series.astype(str).str.strip()
    as_str = as_str.str.replace(",", ".", regex=False)
    extracted = as_str.str.extract(r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", expand=False)
    return pd.to_numeric(extracted, errors="coerce")


def mfc_has_temperature(mfc_df: Optional[pd.DataFrame]) -> bool:
    if mfc_df is None or mfc_df.empty or "T" not in mfc_df.columns:
        return False
    return bool(mfc_df["T"].notna().any())


@st.cache_data(show_spinner=False)
def load_mfc_csv(file_bytes: bytes, filename: str) -> pd.DataFrame:
    bio = io.BytesIO(file_bytes)
    dfm = pd.read_csv(bio)
    dfm.columns = [_clean_mfc_column_name(c) for c in dfm.columns]

    ts_col = _find_mfc_column(dfm.columns, "Timestamp", "Time", "DateTime", "Date time")
    fset_col = _find_mfc_column(dfm.columns, "Fsetpoint", "F setpoint", "Flow setpoint")
    fmeas_col = _find_mfc_column(dfm.columns, "Fmeasure", "F measure", "Flow measure")
    missing = [
        label
        for label, col in (("Timestamp", ts_col), ("Fsetpoint", fset_col), ("Fmeasure", fmeas_col))
        if col is None
    ]
    if missing:
        raise ValueError(f"MFC missing {missing}. Columns: {list(dfm.columns)}")

    dfm["t"] = pd.to_datetime(dfm[ts_col], errors="coerce", dayfirst=True)
    dfm["Fset"] = _parse_mfc_numeric_series(dfm[fset_col])
    dfm["Fmeas"] = _parse_mfc_numeric_series(dfm[fmeas_col])
    temp_col = _detect_mfc_temperature_column(dfm.columns)
    if temp_col is not None:
        dfm["T"] = _parse_mfc_numeric_series(dfm[temp_col])
    dfm = (
        dfm.dropna(subset=["t"])
        .sort_values("t")
        .drop_duplicates(subset=["t"], keep="last")
        .reset_index(drop=True)
    )
    dfm["F"] = dfm["Fmeas"].fillna(dfm["Fset"])
    if temp_col is not None:
        dfm.attrs["temp_source_col"] = temp_col
    return dfm


def add_z_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Continuous height z (m) from the raw Explora `z` column only."""
    out = df.copy()
    z_num = pd.Series(np.nan, index=out.index, dtype=float)
    if "z" in out.columns:
        z_num = pd.to_numeric(out["z"], errors="coerce")
    out["z_maybe"] = z_num
    return out


def classify_regions(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "structure" in out.columns:
        s = out["structure"].astype(str).str.strip().str.upper()
        out["region"] = np.where(s == "PK", "PK", "CAVE")
    else:
        cave_walls = {
            "North", "East", "South", "West", "Ceiling",
            "South (RSD)", "FFE", "GFE"
        }
        out["region"] = np.where(out["wall"].isin(cave_walls), "CAVE", "PK")
    return out


def apply_cave_exclusions(df: pd.DataFrame, cfg: AppConfig) -> pd.DataFrame:
    if not cfg.apply_cave_exclusions:
        return df.copy()

    out = df.copy()
    df_cave_tmp = out[out["region"] == "CAVE"].copy()

    fixture_ok = "fixture" in df_cave_tmp.columns
    fixture = (
        df_cave_tmp["fixture"].astype(str).str.strip().str.lower()
        if fixture_ok else pd.Series([""] * len(df_cave_tmp), index=df_cave_tmp.index)
    )
    exclude_levels = _parse_z_level_labels(cfg.exclude_z_levels)
    z_level = _assign_z_level(df_cave_tmp)

    mask_excl = (
        (fixture_ok & fixture.isin([x.lower() for x in cfg.exclude_fixtures])) |
        (df_cave_tmp["sensor_number"].isin(cfg.exclude_sensors))
    )
    if exclude_levels:
        mask_excl = mask_excl | z_level.isin(exclude_levels)

    keep_cave_idx = df_cave_tmp.loc[~mask_excl].index
    out = pd.concat(
        [out.loc[keep_cave_idx], out[out["region"] == "PK"]],
        axis=0
    ).sort_values(["time", "sensor_number"]).reset_index(drop=True)
    return out


def compute_co2_metrics(df_region: pd.DataFrame, align_to: str, min_sensors: int, coverage_factor: float):
    d = df_region.dropna(subset=["time", "co2"]).copy()
    d["tbin"] = d["time"].dt.floor(align_to)

    g = d.groupby("tbin")
    n = g["sensor_number"].nunique()

    mean = g["co2"].mean()
    std = g["co2"].std()
    cv = std / mean
    mi = 1 - cv

    ok = n >= min_sensors
    mean = mean.where(ok)
    std = std.where(ok)
    cv = cv.where(ok)
    mi = mi.where(ok)

    mean_valid = mean.dropna()
    baseline = float(mean_valid.iloc[0]) if len(mean_valid) else np.nan
    threshold = baseline * coverage_factor if np.isfinite(baseline) else np.nan

    tmp = d.copy()
    tmp["covered"] = tmp["co2"] >= threshold
    coverage = tmp.groupby("tbin")["covered"].mean()
    coverage = (coverage.where(ok) * 100.0)

    return {
        "n": n,
        "mean": mean,
        "std": std,
        "cv": cv,
        "mi": mi,
        "coverage": coverage,
        "baseline": baseline,
        "threshold": threshold,
    }


def compute_temp_metrics(df_region: pd.DataFrame, align_to: str, min_sensors: int, high_selector, low_selector):
    d = df_region.dropna(subset=["time", "temperature"]).copy()
    d["tbin"] = d["time"].dt.floor(align_to)
    d = add_z_numeric(d)

    nT = d.groupby("tbin")["sensor_number"].nunique()
    okT = nT >= min_sensors

    gT = d.groupby("tbin")["temperature"]
    mean_T = gT.mean().where(okT)
    std_T = gT.std().where(okT)
    cv_T = std_T / mean_T
    mi_T = (1 - cv_T).where(okT)

    def delta_onebin(subdf):
        hi = subdf.loc[high_selector(subdf), "temperature"].dropna()
        lo = subdf.loc[low_selector(subdf), "temperature"].dropna()
        if len(hi) == 0 or len(lo) == 0:
            return np.nan
        return float(hi.mean() - lo.mean())

    deltaT = d.groupby("tbin").apply(delta_onebin).where(okT)

    def r2_linear_fit(z, t):
        ok = np.isfinite(z) & np.isfinite(t)
        z = z[ok]
        t = t[ok]
        if len(t) < 2:
            return np.nan
        p = np.polyfit(z, t, 1)
        t_hat = p[0] * z + p[1]
        ss_res = np.sum((t - t_hat) ** 2)
        ss_tot = np.sum((t - np.mean(t)) ** 2)
        if ss_tot == 0:
            return np.nan
        return float(1 - ss_res / ss_tot)

    def r2_onebin(subdf):
        z = subdf["z_maybe"].to_numpy(dtype=float)
        t = subdf["temperature"].to_numpy(dtype=float)
        return r2_linear_fit(z, t)

    r2_Tz = d.groupby("tbin").apply(r2_onebin).where(okT)

    return {
        "n": nT,
        "mean_T": mean_T,
        "std_T": std_T,
        "deltaT": deltaT,
        "r2_Tz": r2_Tz,
        "mi_T": mi_T,
    }


def compute_humidity_metrics(df_region: pd.DataFrame, align_to: str, min_sensors: int):
    """Region-level humidity time series (mean, std, CV, mixing index)."""
    if not humidity_has_data(df_region):
        empty = pd.Series(dtype=float)
        return {"n": empty, "mean": empty, "std": empty, "cv": empty, "mi": empty}

    d = df_region.dropna(subset=["time", "humidity"]).copy()
    d["tbin"] = d["time"].dt.floor(align_to)
    g = d.groupby("tbin")
    n = g["sensor_number"].nunique()
    mean = g["humidity"].mean()
    std = g["humidity"].std()
    cv = std / mean
    mi = 1 - cv
    ok = n >= min_sensors
    mean = mean.where(ok)
    std = std.where(ok)
    cv = cv.where(ok)
    mi = mi.where(ok)
    return {"n": n, "mean": mean, "std": std, "cv": cv, "mi": mi}


def zone_mean_timeseries(df_region: pd.DataFrame, zone_col: str, zones, value_col: str, align_to: str, min_sensors: int):
    d = df_region.dropna(subset=["time", value_col]).copy()
    d["tbin"] = d["time"].dt.floor(align_to)
    d[zone_col] = d[zone_col].astype(str).str.strip()

    if zones is not None:
        d = d[d[zone_col].isin(zones)].copy()

    g = d.groupby(["tbin", zone_col])
    mu = g[value_col].mean()
    n = g["sensor_number"].nunique()
    mu = mu.where(n >= min_sensors)

    out = mu.unstack(zone_col).sort_index()
    return out


def sensor_catalog(df_region: pd.DataFrame) -> pd.DataFrame:
    """One row per sensor_number with wall (zone) and median z for selection UI."""
    if df_region is None or len(df_region) == 0:
        return pd.DataFrame(columns=["sensor_number", "wall", "z_median"])

    d = df_region.dropna(subset=["sensor_number", "wall"]).copy()
    d["wall"] = d["wall"].astype(str).str.strip()
    d["sensor_number"] = pd.to_numeric(d["sensor_number"], errors="coerce")
    d = d.dropna(subset=["sensor_number"])
    d["sensor_number"] = d["sensor_number"].astype(int)

    rows = []
    for sid, g in d.groupby("sensor_number"):
        wall = g["wall"].mode()
        w = str(wall.iloc[0]) if len(wall) else "?"
        z_med = np.nan
        if "z" in g.columns:
            z_med = pd.to_numeric(g["z"], errors="coerce").median()
        rows.append({"sensor_number": int(sid), "wall": w, "z_median": z_med})
    out = pd.DataFrame(rows).sort_values(["wall", "sensor_number"]).reset_index(drop=True)
    return out


def _sensor_series_label(sensor_number: int, wall: str, z_median: float = np.nan) -> str:
    z_part = f", z={z_median:.2f}m" if np.isfinite(z_median) else ""
    return f"S{int(sensor_number)} ({wall}{z_part})"


def sensors_in_walls(catalog: pd.DataFrame, walls: List[str]) -> List[int]:
    if catalog is None or len(catalog) == 0 or not walls:
        return []
    wset = {str(w).strip() for w in walls}
    return sorted(catalog.loc[catalog["wall"].isin(wset), "sensor_number"].astype(int).tolist())


def sensor_value_timeseries(
    df_region: pd.DataFrame,
    sensor_numbers: List[int],
    align_to: str,
    value_col: str,
    catalog: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Per-sensor mean by time bin (columns = readable sensor labels)."""
    if df_region is None or len(df_region) == 0 or not sensor_numbers or value_col not in df_region.columns:
        return pd.DataFrame()

    cat = catalog if catalog is not None else sensor_catalog(df_region)
    label_by_id = {
        int(r["sensor_number"]): _sensor_series_label(
            int(r["sensor_number"]), str(r["wall"]), float(r["z_median"]) if pd.notna(r["z_median"]) else np.nan
        )
        for _, r in cat.iterrows()
    }

    d = df_region.dropna(subset=["time", value_col]).copy()
    d["sensor_number"] = pd.to_numeric(d["sensor_number"], errors="coerce")
    d = d.dropna(subset=["sensor_number"])
    d["sensor_number"] = d["sensor_number"].astype(int)
    d = d[d["sensor_number"].isin([int(s) for s in sensor_numbers])].copy()
    if len(d) == 0:
        return pd.DataFrame()

    d["tbin"] = d["time"].dt.floor(align_to)
    mu = d.groupby(["tbin", "sensor_number"])[value_col].mean()
    out = mu.unstack("sensor_number").sort_index()
    out.columns = [label_by_id.get(int(c), f"S{int(c)}") for c in out.columns]
    return out


def sensor_co2_timeseries(
    df_region: pd.DataFrame,
    sensor_numbers: List[int],
    align_to: str,
    catalog: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    return sensor_value_timeseries(df_region, sensor_numbers, align_to, "co2", catalog=catalog)


def add_stage_shading(ax, stage_defs, stage_patches):
    for (name, stt, ett, col) in stage_defs:
        ax.axvspan(stt, ett, color=col, alpha=0.10, zorder=0)
        if name not in [p.get_label() for p in stage_patches]:
            stage_patches.append(Patch(facecolor=col, alpha=0.40, label=name))


def add_plotly_stage_vrects(
    fig,
    stage_defs,
    fill_opacity: float = 0.08,
    row: Optional[Any] = None,
    col: Optional[int] = None,
) -> None:
    """Shaded vertical bands for experiment stages (all panels or one subplot)."""
    if not stage_defs:
        return
    for (_name, stt, ett, colr) in stage_defs:
        kw = dict(x0=stt, x1=ett, fillcolor=colr, opacity=fill_opacity, line_width=0)
        if row is not None:
            fig.add_vrect(**kw, row=row, col=col if col is not None else 1)
        else:
            fig.add_vrect(**kw)


def _stage_legend_items(stage_defs) -> List[Tuple[str, str, Any, Any]]:
    """Unique (name, colour, start, end) for external stage legend."""
    seen: set = set()
    items: List[Tuple[str, str, Any, Any]] = []
    for (name, stt, ett, colr) in stage_defs or []:
        lab = str(name)
        if lab in seen:
            continue
        seen.add(lab)
        items.append((lab, str(colr), stt, ett))
    return items


def _series_color_for_index(index: int) -> str:
    return _PLOTLY_SERIES_COLORS[index % len(_PLOTLY_SERIES_COLORS)]


def _trace_legend_color(trace, index: int = 0, fig=None) -> str:
    """Line/marker colour for an external legend swatch (must match trace styling)."""
    try:
        if getattr(trace, "line", None) and trace.line.color:
            c = trace.line.color
            if c and str(c).strip().lower() not in ("", "auto"):
                return str(c)
        if getattr(trace, "marker", None) and trace.marker.color:
            c = trace.marker.color
            if c and str(c).strip().lower() not in ("", "auto"):
                return str(c)
        if fig is not None and getattr(fig.layout, "colorway", None):
            cw = list(fig.layout.colorway)
            if cw:
                return str(cw[index % len(cw)])
    except Exception:
        pass
    return _series_color_for_index(index)


def render_series_legend_outside(fig, *, title: str = "Sensors / series") -> None:
    """Series key below the plot (for long legends on sensor compare charts)."""
    if fig is None or not getattr(fig, "data", None):
        return
    chips = []
    series_idx = 0
    for tr in fig.data:
        name = getattr(tr, "name", None)
        if not name:
            continue
        colr = _trace_legend_color(tr, series_idx, fig)
        series_idx += 1
        safe_name = html.escape(str(name))
        chips.append(
            f'<span style="display:inline-flex;align-items:center;margin:3px 12px 3px 0;max-width:100%;">'
            f'<span style="display:inline-block;width:22px;height:3px;background:{colr};'
            f'border-radius:1px;margin-right:6px;flex-shrink:0;"></span>'
            f'<span style="font-size:0.85rem;line-height:1.3;">{safe_name}</span></span>'
        )
    if not chips:
        return
    safe_title = html.escape(str(title))
    st.markdown(
        '<div class="series-legend-outside" style="display:block;margin:0.35rem 0 1.1rem 0;padding:10px 14px;'
        'border:1px solid #d0d0d0;border-radius:8px;background:#fafafa;max-height:220px;overflow-y:auto;">'
        f'<span style="font-weight:600;font-size:0.92rem;margin-right:12px;">{safe_title}</span>'
        '<span style="display:inline-flex;flex-wrap:wrap;align-items:center;vertical-align:middle;">'
        + "".join(chips)
        + "</span></div>",
        unsafe_allow_html=True,
    )


def render_stage_legend_outside(stage_defs, *, swatch_opacity: float = 0.38) -> None:
    """Stage colour key below the plot (separate from the in-figure CAVE/PK legend)."""
    items = _stage_legend_items(stage_defs)
    if not items:
        return
    chips = []
    for lab, colr, stt, ett in items:
        t0 = pd.Timestamp(stt).strftime("%H:%M") if pd.notna(stt) else "?"
        t1 = pd.Timestamp(ett).strftime("%H:%M") if pd.notna(ett) else "?"
        chips.append(
            f'<span style="display:inline-flex;align-items:center;margin:4px 14px 4px 0;white-space:nowrap;">'
            f'<span style="display:inline-block;width:15px;height:15px;background:{colr};opacity:{swatch_opacity};'
            f'border:1px solid rgba(0,0,0,0.45);margin-right:7px;flex-shrink:0;"></span>'
            f'<span style="font-size:0.9rem;"><b>{lab}</b>'
            f'<span style="color:#555;font-weight:normal;"> ({t0}–{t1})</span></span></span>'
        )
    st.markdown(
        '<div class="stage-legend-outside" style="display:block;margin:0.35rem 0 1rem 0;padding:10px 14px;'
        'border:1px solid #d0d0d0;border-radius:8px;background:#fafafa;">'
        '<span style="font-weight:600;font-size:0.92rem;margin-right:12px;">Experiment stages</span>'
        '<span style="display:inline-flex;flex-wrap:wrap;align-items:center;vertical-align:middle;">'
        + "".join(chips)
        + "</span></div>",
        unsafe_allow_html=True,
    )


def prepare_stage_defs(stage_rows):
    if not stage_rows:
        return []
    colors = ["orange", "skyblue", "red", "cyan", "brown", "green", "magenta"]
    return [(n, stt, ett, colors[i % len(colors)]) for i, (n, stt, ett) in enumerate(stage_rows)]


def find_release_window(stage_defs):
    if not stage_defs:
        return None, None, "no stage_defs"
    for (name, stt, ett, col) in stage_defs:
        if isinstance(name, str) and ("release" in name.lower()):
            return pd.Timestamp(stt), pd.Timestamp(ett), f"stage: {name}"
    if len(stage_defs) >= 2:
        name, stt, ett, col = stage_defs[1]
        return pd.Timestamp(stt), pd.Timestamp(ett), f"stage2 fallback: {name}"
    name, stt, ett, col = stage_defs[0]
    return pd.Timestamp(stt), pd.Timestamp(ett), f"only stage available: {name}"


def find_baseline_window(stage_defs):
    if stage_defs:
        for (name, stt, ett, col) in stage_defs:
            if isinstance(name, str) and ("baseline" in name.lower()):
                return pd.Timestamp(stt), pd.Timestamp(ett), f"stage: {name}"
    return None, None, "no baseline stage"


def snap_to_index(series, t):
    s = series.dropna()
    if len(s) == 0:
        return pd.Timestamp(t)
    idx = s.index
    t = pd.Timestamp(t)
    try:
        pos = idx.get_indexer([t], method="nearest")[0]
        if pos >= 0:
            return idx[pos]
    except Exception:
        pass
    return t


def mean_in_window(series, t0, t1):
    s = series.dropna()
    s = s[(s.index >= t0) & (s.index <= t1)]
    return float(s.mean()) if len(s) else np.nan


def build_summary_df(summary_dict: dict) -> pd.DataFrame:
    return pd.DataFrame({"metric": list(summary_dict.keys()), "value": list(summary_dict.values())})


def find_stage_by_keyword(stage_defs, keyword: str):
    kw = str(keyword).strip().lower()
    for (name, stt, ett, col) in stage_defs or []:
        if isinstance(name, str) and (kw in name.strip().lower()):
            return (name, pd.Timestamp(stt), pd.Timestamp(ett), col)
    return None


def split_time_range(t0: pd.Timestamp, t1: pd.Timestamp, n: int):
    t0 = pd.Timestamp(t0)
    t1 = pd.Timestamp(t1)
    if n <= 0:
        raise ValueError("n must be > 0")
    if pd.isna(t0) or pd.isna(t1) or t1 <= t0:
        return []
    dt = (t1 - t0) / n
    out = []
    for i in range(n):
        a = t0 + i * dt
        b = t0 + (i + 1) * dt
        out.append((a, b))
    return out


def _parse_z_level_labels(labels: Tuple[str, ...]) -> frozenset[float]:
    """Parse sidebar labels (z1, z2, …) into discrete level numbers."""
    levels: List[float] = []
    for lab in labels:
        s = str(lab).strip().lower()
        parsed = pd.Series([s]).str.extract(r"([0-9]+(?:\.[0-9]+)?)", expand=False).iloc[0]
        if pd.notna(parsed):
            levels.append(float(parsed))
    return frozenset(levels)


def _z_level_label(level: float) -> str:
    return f"z{int(level)}" if np.isfinite(level) else ""


def _z_coord_to_level(z_series: pd.Series) -> pd.Series:
    """
    Map continuous z (m) to discrete height level:
      z in [0, 1] -> level 1, (1, 2] -> level 2, (2, 3] -> level 3, ...
    e.g. z=1.0 -> 1, z=2.0 -> 2.
    """
    z = pd.to_numeric(z_series, errors="coerce")
    ok = np.isfinite(z) & (z >= 0)
    level = np.ceil(z).astype(float)
    level = np.maximum(level, 1.0)  # z=0 -> level 1
    return pd.Series(level, index=z_series.index).where(ok)


def _assign_z_level(d: pd.DataFrame) -> pd.Series:
    """Discrete height level per row from raw Explora `z` (m) only."""
    if "z" not in d.columns:
        return pd.Series(np.nan, index=d.index, dtype=float)
    return _z_coord_to_level(d["z"])


def _rows_in_z_levels(subdf: pd.DataFrame, levels: frozenset[float]) -> pd.Series:
    if not levels:
        return pd.Series(False, index=subdf.index)
    return _assign_z_level(subdf).isin(levels)


def sensors_by_z_level(df_region: pd.DataFrame) -> pd.DataFrame:
    """Unique sensor_number at each z level (matches vertical profile binning)."""
    empty = pd.DataFrame(columns=["z_level", "z_label", "sensor_numbers"])
    if df_region is None or len(df_region) == 0:
        return empty

    d = df_region.copy()
    d["z_level"] = _assign_z_level(d)
    d = d.dropna(subset=["z_level", "sensor_number"]).copy()
    if len(d) == 0:
        return empty

    rows = []
    for zl, g in d.groupby("z_level", sort=True):
        sensors = sorted(int(s) for s in g["sensor_number"].dropna().unique())
        rows.append(
            {
                "z_level": zl,
                "z_label": _z_level_label(zl),
                "sensor_numbers": sensors,
            }
        )
    return pd.DataFrame(rows).sort_values("z_level").reset_index(drop=True)


def format_z_level_sensor_map(df_region: pd.DataFrame, region_label: str) -> str:
    tbl = sensors_by_z_level(df_region)
    if tbl.empty:
        return f"**{region_label}:** no sensors with assignable z level in the loaded data."
    lines = [f"**{region_label}** — sensor numbers per z level:"]
    for _, row in tbl.iterrows():
        sns = ", ".join(str(s) for s in row["sensor_numbers"])
        lines.append(f"- **{row['z_label']}**: {sns}")
    return "\n".join(lines)


def vertical_profile_means(df_region: pd.DataFrame, t0, t1, value_col: str) -> pd.DataFrame:
    if df_region is None or len(df_region) == 0:
        return pd.DataFrame(columns=["z_level", "z_label", "mean"])

    d = df_region.copy()
    d = d[(d["time"] >= pd.Timestamp(t0)) & (d["time"] <= pd.Timestamp(t1))].copy()
    d[value_col] = pd.to_numeric(d[value_col], errors="coerce")
    d["z_level"] = _assign_z_level(d)

    d = d.dropna(subset=[value_col, "z_level"]).copy()
    if len(d) == 0:
        return pd.DataFrame(columns=["z_level", "z_label", "mean"])

    out = d.groupby("z_level")[value_col].mean().rename("mean").reset_index()
    out["z_label"] = out["z_level"].apply(_z_level_label)
    out = out.sort_values("z_level").reset_index(drop=True)
    return out


def _vertical_profile_title(line_prefix: str) -> str:
    """Two-line figure title for vertical profile panels (Matplotlib: \\n; Plotly converts to <br>)."""
    return f"{line_prefix}\nvertical profile"


def plot_vertical_profiles_matplotlib(
    profiles,
    title: str,
    x_label: str,
    x_range=None,
    y_range=(0, 10),
    *,
    show_legend: bool = True,
    line_width: float = 2.0,
    marker_size: float = 6.0,
    legend_fontsize: int = 9,
):
    # Figure aspect: vertical (y) : horizontal (x) = 2 : 1 → figsize (w, h) with h/w = 2
    fig, ax = plt.subplots(figsize=(3.5, 7.0))
    for label, dfp in profiles:
        if dfp is None or len(dfp) == 0:
            continue
        ax.plot(
            dfp["mean"].values,
            dfp["z_level"].values,
            marker="o",
            linewidth=float(line_width),
            markersize=float(marker_size),
            label=label,
        )
    ax.grid(True)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel("z slice")
    if x_range is not None:
        ax.set_xlim(x_range[0], x_range[1])
    if y_range is not None:
        ax.set_ylim(y_range[0], y_range[1])
    ax.set_yticks(list(range(0, 11)))
    ax.set_yticklabels([f"z{i}" if i > 0 else "z0" for i in range(0, 11)])
    if show_legend:
        ax.legend(loc="best", fontsize=max(5, min(24, int(legend_fontsize))))
    return fig


def plot_vertical_profiles_plotly(
    profiles,
    title: str,
    x_label: str,
    x_range=None,
    y_range=(0, 10),
    *,
    line_width: float = 2.0,
    marker_size: float = 6.0,
):
    _require_plotly()
    lw = max(0.25, float(line_width))
    ms = max(1.0, float(marker_size))
    fig = go.Figure()
    for label, dfp in profiles:
        if dfp is None or len(dfp) == 0:
            continue
        fig.add_trace(
            go.Scatter(
                x=dfp["mean"].values,
                y=dfp["z_level"].values,
                mode="lines+markers",
                name=label,
                line=dict(width=lw),
                marker=dict(size=ms),
            )
        )
    # Tall profile panels; width follows container (responsive).
    _ph = 640
    _title = str(title).strip().replace("\n", "<br>")
    _tm = 78 if "<br>" in _title else 60
    fig.update_layout(
        title=dict(text=_title, x=0.5, xref="paper", xanchor="center"),
        xaxis_title=x_label,
        yaxis_title="z level",
        height=_ph,
        autosize=True,
        showlegend=True,
        template="plotly_white",
        margin=dict(l=58, r=24, t=_tm, b=52),
    )
    fig.update_yaxes(automargin=True, title_standoff=10)
    fig.update_xaxes(automargin=True)
    if x_range is not None:
        fig.update_xaxes(range=list(x_range))
    if y_range is not None:
        fig.update_yaxes(range=list(y_range))
    return fig


# =========================================================
# Plotting
# =========================================================
def plot_overall_metrics(
    co2_cave,
    co2_pk,
    temp_cave,
    temp_pk,
    deltaT_pk_minus_cave,
    stage_defs,
    cfg: AppConfig,
    plot_start,
    plot_end,
    *,
    line_width: float = 2.0,
    legend_fontsize: int = 9,
):
    lw_c = float(line_width) * 1.5
    lw_p = float(line_width) * 1.0
    lw_dt = float(line_width) * 1.0
    fig, axs = plt.subplots(5, 2, figsize=(18, 16), sharex=True)

    titles_co2 = ["Mean CO₂", "Std CO₂", "CV (CO₂)", "Mixing Index (CO₂)", f"Coverage (CO₂ ≥ baseline×{cfg.coverage_factor:.2f})"]
    titles_T = ["Mean T (°C)", "Std T (°C)", "ΔT(high-low) (°C)", "R²(T~z)", "Mixing Index (T)"]

    axs[0, 0].plot(co2_cave["mean"].index, co2_cave["mean"].values, linewidth=lw_c, label="CAVE mean")
    axs[0, 0].plot(co2_pk["mean"].index, co2_pk["mean"].values, linewidth=lw_p, linestyle="--", label="PK mean")

    axs[1, 0].plot(co2_cave["std"].index, co2_cave["std"].values, linewidth=lw_c, label="CAVE std")
    axs[1, 0].plot(co2_pk["std"].index, co2_pk["std"].values, linewidth=lw_p, linestyle="--", label="PK std")

    axs[2, 0].plot(co2_cave["cv"].index, co2_cave["cv"].values, linewidth=lw_c, label="CAVE CV")
    axs[2, 0].plot(co2_pk["cv"].index, co2_pk["cv"].values, linewidth=lw_p, linestyle="--", label="PK CV")

    axs[3, 0].plot(co2_cave["mi"].index, co2_cave["mi"].values, linewidth=lw_c, label="CAVE MI")
    axs[3, 0].plot(co2_pk["mi"].index, co2_pk["mi"].values, linewidth=lw_p, linestyle="--", label="PK MI")

    axs[4, 0].plot(co2_cave["coverage"].index, co2_cave["coverage"].values, linewidth=lw_c, label="CAVE coverage")
    axs[4, 0].plot(co2_pk["coverage"].index, co2_pk["coverage"].values, linewidth=lw_p, linestyle="--", label="PK coverage")

    axs[0, 1].plot(temp_cave["mean_T"].index, temp_cave["mean_T"].values, linewidth=lw_c, label="CAVE mean T")
    axs[0, 1].plot(temp_pk["mean_T"].index, temp_pk["mean_T"].values, linewidth=lw_p, linestyle="--", label="PK mean T")

    ax_dt = axs[0, 1].twinx()
    ax_dt.plot(deltaT_pk_minus_cave.index, deltaT_pk_minus_cave.values, linewidth=lw_dt, linestyle=":", label="ΔT (PK − CAVE)")
    ax_dt.set_ylabel("ΔT (°C)", fontsize=11, fontweight="bold")
    axs[0, 1]._ax_dt = ax_dt

    axs[1, 1].plot(temp_cave["std_T"].index, temp_cave["std_T"].values, linewidth=lw_c, label="CAVE std T")
    axs[1, 1].plot(temp_pk["std_T"].index, temp_pk["std_T"].values, linewidth=lw_p, linestyle="--", label="PK std T")

    axs[2, 1].plot(temp_cave["deltaT"].index, temp_cave["deltaT"].values, linewidth=lw_c, label="CAVE ΔT(H-L)")
    axs[2, 1].plot(temp_pk["deltaT"].index, temp_pk["deltaT"].values, linewidth=lw_p, linestyle="--", label="PK ΔT(H-L)")

    axs[3, 1].plot(temp_cave["r2_Tz"].index, temp_cave["r2_Tz"].values, linewidth=lw_c, label="CAVE R²")
    axs[3, 1].plot(temp_pk["r2_Tz"].index, temp_pk["r2_Tz"].values, linewidth=lw_p, linestyle="--", label="PK R²")

    axs[4, 1].plot(temp_cave["mi_T"].index, temp_cave["mi_T"].values, linewidth=lw_c, label="CAVE MI(T)")
    axs[4, 1].plot(temp_pk["mi_T"].index, temp_pk["mi_T"].values, linewidth=lw_p, linestyle="--", label="PK MI(T)")

    stage_patches = []
    for r in range(5):
        for c in range(2):
            ax = axs[r, c]
            ax.grid(True)
            if stage_defs:
                add_stage_shading(ax, stage_defs, stage_patches)

    for i in range(5):
        axs[i, 0].set_ylabel(titles_co2[i], fontsize=12, fontweight="bold")
        axs[i, 1].set_ylabel(titles_T[i], fontsize=12, fontweight="bold")

    for ax in axs[-1, :]:
        ax.set_xlabel("Time", fontsize=12, fontweight="bold")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

    if plot_start is not None and plot_end is not None:
        for ax_row in axs:
            for ax in ax_row:
                ax.set_xlim(plot_start, plot_end)

    plt.setp(axs[-1, 0].get_xticklabels(), rotation=45)
    plt.setp(axs[-1, 1].get_xticklabels(), rotation=45)

    leg_fs = max(5, min(24, int(legend_fontsize)))

    h1, l1 = axs[0, 1].get_legend_handles_labels()
    h2, l2 = axs[0, 1]._ax_dt.get_legend_handles_labels()
    axs[0, 1].legend(h1 + h2, l1 + l2, loc="upper right", frameon=True, fontsize=leg_fs)
    axs[0, 0].legend(loc="upper right", frameon=True, fontsize=leg_fs)

    if cfg.use_fixed_ylims:
        y = cfg.ylims
        axs[0, 0].set_ylim(*y["co2_mean"])
        axs[1, 0].set_ylim(*y["co2_std"])
        axs[2, 0].set_ylim(*y["co2_cv"])
        axs[3, 0].set_ylim(*y["co2_mi"])
        axs[4, 0].set_ylim(*y["co2_coverage"])

        axs[0, 1].set_ylim(*y["temp_mean"])
        axs[1, 1].set_ylim(*y["temp_std"])
        axs[2, 1].set_ylim(*y["temp_deltaT"])
        axs[3, 1].set_ylim(*y["temp_r2"])
        axs[4, 1].set_ylim(*y["temp_mi"])
        axs[0, 1]._ax_dt.set_ylim(*y["temp_pk_minus_cave"])

    plt.suptitle(f"{cfg.exp_code} — Overall metrics (CAVE vs PK)", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def plot_zone_co2(
    cave_zone_co2,
    pk_zone_co2,
    stage_defs,
    cfg: AppConfig,
    plot_start,
    plot_end,
    *,
    cave_line_width: float = 2.5,
    pk_line_width: float = 2.0,
    cave_legend_fs: int = 9,
    pk_legend_fs: int = 9,
):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    lfc = max(5, min(24, int(cave_legend_fs)))
    lfp = max(5, min(24, int(pk_legend_fs)))

    for col in cave_zone_co2.columns:
        ax1.plot(cave_zone_co2.index, cave_zone_co2[col].values, linewidth=cave_line_width, label=col)

    if stage_defs:
        stage_patches_c = []
        add_stage_shading(ax1, stage_defs, stage_patches_c)

    ax1.set_title(f"{cfg.exp_code} — CAVE selected walls mean CO₂", fontsize=13, fontweight="bold")
    ax1.set_ylabel("CO₂ (ppm)", fontsize=12, fontweight="bold")
    ax1.grid(True)
    ax1.legend(fontsize=lfc, frameon=True, loc="upper right")

    for col in pk_zone_co2.columns:
        ax2.plot(pk_zone_co2.index, pk_zone_co2[col].values, linewidth=pk_line_width, label=col)

    if stage_defs:
        stage_patches_b = []
        add_stage_shading(ax2, stage_defs, stage_patches_b)

    ax2.set_title(f"{cfg.exp_code} — PK zones mean CO₂ (by wall)", fontsize=13, fontweight="bold")
    ax2.set_ylabel("CO₂ (ppm)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Time", fontsize=12, fontweight="bold")
    ax2.grid(True)
    ax2.legend(ncol=4, fontsize=lfp, frameon=True, loc="upper right")

    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    plt.xticks(rotation=45)

    if plot_start is not None and plot_end is not None:
        ax1.set_xlim(plot_start, plot_end)

    if cfg.use_fixed_ylims:
        ax1.set_ylim(*cfg.ylims["zone_cave_co2"])
        ax2.set_ylim(*cfg.ylims["zone_pk_co2"])

    plt.tight_layout()
    return fig


def plot_zone_temp(
    cave_zone_temp,
    pk_zone_temp,
    stage_defs,
    cfg: AppConfig,
    plot_start,
    plot_end,
    *,
    cave_line_width: float = 2.5,
    pk_line_width: float = 2.0,
    cave_legend_fs: int = 9,
    pk_legend_fs: int = 9,
):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    lfc = max(5, min(24, int(cave_legend_fs)))
    lfp = max(5, min(24, int(pk_legend_fs)))

    for col in cave_zone_temp.columns:
        ax1.plot(cave_zone_temp.index, cave_zone_temp[col].values, linewidth=cave_line_width, label=col)

    if stage_defs:
        stage_patches_c = []
        add_stage_shading(ax1, stage_defs, stage_patches_c)

    ax1.set_title(f"{cfg.exp_code} — CAVE selected walls mean temperature", fontsize=13, fontweight="bold")
    ax1.set_ylabel("Temperature (°C)", fontsize=12, fontweight="bold")
    ax1.grid(True)
    ax1.legend(fontsize=lfc, frameon=True, loc="upper right")

    for col in pk_zone_temp.columns:
        ax2.plot(pk_zone_temp.index, pk_zone_temp[col].values, linewidth=pk_line_width, label=col)

    if stage_defs:
        stage_patches_b = []
        add_stage_shading(ax2, stage_defs, stage_patches_b)

    ax2.set_title(f"{cfg.exp_code} — PK zones mean temperature (by wall)", fontsize=13, fontweight="bold")
    ax2.set_ylabel("Temperature (°C)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Time", fontsize=12, fontweight="bold")
    ax2.grid(True)
    ax2.legend(ncol=4, fontsize=lfp, frameon=True, loc="upper right")

    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    plt.xticks(rotation=45)

    if plot_start is not None and plot_end is not None:
        ax1.set_xlim(plot_start, plot_end)

    plt.tight_layout()
    return fig


def plot_mfc(mfc_df, t_on, t_off, t_rel0, t_rel1, cfg: AppConfig, *, line_width: float = 2.2, legend_fontsize: int = 10):
    lw = float(line_width)
    leg_fs = max(5, min(24, int(legend_fontsize)))
    has_temp = mfc_has_temperature(mfc_df)
    fig, ax = plt.subplots(figsize=(14, 4.8))
    ax.plot(
        mfc_df["t"],
        mfc_df["F"],
        linewidth=lw,
        color="#1f77b4",
        label="MFC flow (Fmeas if available else Fset)",
    )
    ax.axhline(cfg.flow_on_th, linestyle=":", linewidth=max(1.0, lw * 0.85), color="#444444", label=f"FLOW_ON_TH={cfg.flow_on_th}")

    ax2 = None
    if has_temp:
        ax2 = ax.twinx()
        ax2.plot(
            mfc_df["t"],
            mfc_df["T"],
            linewidth=lw,
            color="#d62728",
            linestyle="-",
            label="Temperature (°C)",
        )
        ax2.set_ylabel("Temperature (°C)", fontsize=12, fontweight="bold", color="#d62728")
        ax2.tick_params(axis="y", labelcolor="#d62728")

    if (t_on is not None) and (t_off is not None):
        ax.axvspan(t_on, t_off, alpha=0.15, color="green", label="Detected release (F>TH)")

    if (t_rel0 is not None) and (t_rel1 is not None):
        ax.axvspan(t_rel0, t_rel1, alpha=0.10, color="orange", label="Stage2 (Release)")

    title = f"{cfg.exp_code} — MFC Release Quicklook"
    if has_temp:
        title += " (flow + temperature)"
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_ylabel("Flow (MFC units)", fontsize=12, fontweight="bold", color="#1f77b4")
    ax.tick_params(axis="y", labelcolor="#1f77b4")
    ax.set_xlabel("Time", fontsize=12, fontweight="bold")
    ax.grid(True)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    plt.xticks(rotation=45)

    if (t_rel0 is not None) and (t_rel1 is not None):
        ax.set_xlim(t_rel0, t_rel1)

    h1, l1 = ax.get_legend_handles_labels()
    if ax2 is not None:
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, frameon=True, fontsize=leg_fs, loc="upper right")
    else:
        ax.legend(frameon=True, fontsize=leg_fs, loc="upper right")
    plt.tight_layout()
    return fig


def plot_io_ratio(io_ex, infiltration_factor, t_rel0, t_rel1, t_base0, t_base1, ex_thresh, cfg: AppConfig):
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(io_ex.index, io_ex.values, linewidth=2.0, label="I/O_ex(t) = PK_ex / CAVE_ex (thresholded)")
    ax.axvspan(t_rel0, t_rel1, alpha=0.15, label="Release window (Stage2)")

    if np.isfinite(infiltration_factor):
        ax.axhline(infiltration_factor, linestyle="--", linewidth=2.0, label=f"mean(I/O_ex) in Release = {infiltration_factor:.3f}")

    ax.axvspan(t_base0, t_base1, alpha=0.08, label="Baseline window")
    ax.text(0.01, 0.02, f"Threshold: CAVE_ex > {ex_thresh:.1f} ppm", transform=ax.transAxes, fontsize=9, va="bottom", ha="left")

    ax.set_title(f"{cfg.exp_code} — Excess I/O ratio", fontsize=12, fontweight="bold")
    ax.set_ylabel("I/O_ex (-)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Time", fontsize=12, fontweight="bold")
    ax.grid(True)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.set_xlim(t_rel0, t_rel1)

    if cfg.use_fixed_ylims:
        ax.set_ylim(*cfg.ylims["io_ex"])

    plt.xticks(rotation=45)
    ax.legend(frameon=True, fontsize=10, loc="upper left")
    plt.tight_layout()
    return fig


def plot_scatter(df_sc, slope, intercept, r2, cfg: AppConfig):
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.scatter(df_sc["cave_ex"].values, df_sc["pk_ex"].values, s=25, alpha=0.8, label="Release points (thresholded)")

    if np.isfinite(slope):
        xline = np.array([df_sc["cave_ex"].min(), df_sc["cave_ex"].max()])
        yline = intercept + slope * xline
        ax.plot(xline, yline, linewidth=2.0, linestyle="--", label=f"Fit: slope={slope:.3f}, R²={r2:.3f}")

    ax.set_title(f"{cfg.exp_code} — PK_ex vs CAVE_ex (Release only)", fontsize=12, fontweight="bold")
    ax.set_xlabel("CAVE_ex (ppm)", fontsize=12, fontweight="bold")
    ax.set_ylabel("PK_ex (ppm)", fontsize=12, fontweight="bold")
    ax.grid(True)
    ax.legend(frameon=True, fontsize=9, loc="upper left")

    if cfg.use_fixed_ylims:
        ax.set_xlim(*cfg.ylims["scatter_cave_ex"])
        ax.set_ylim(*cfg.ylims["scatter_pk_ex"])

    plt.tight_layout()
    return fig


# ---- Plotly per-page UI defaults & helpers ---------------------------------

LEGEND_POSITION_LABELS = [
    "Top-right (inside)",
    "Top-left (inside)",
    "Bottom-right (inside)",
    "Bottom-left (inside)",
]


def _legend_layout_from_style(style: Dict[str, Any]) -> Dict[str, Any]:
    pos = str(style.get("legend_pos", LEGEND_POSITION_LABELS[1]))
    ncol = max(1, min(int(style.get("legend_ncol", 1)), 6))
    raw_leg = style.get("legend_fs")
    if raw_leg is not None:
        leg_font = max(5, min(24, int(raw_leg)))
    else:
        fs = int(style.get("tick_fs", 11))
        leg_font = max(6, fs - 4)

    pos_map = {
        "Top-right (inside)": dict(x=0.98, y=0.98, xanchor="right", yanchor="top"),
        "Top-left (inside)": dict(x=0.02, y=0.98, xanchor="left", yanchor="top"),
        "Bottom-right (inside)": dict(x=0.98, y=0.02, xanchor="right", yanchor="bottom"),
        "Bottom-left (inside)": dict(x=0.02, y=0.02, xanchor="left", yanchor="bottom"),
    }
    xy = pos_map.get(pos, pos_map["Top-right (inside)"])

    bold_leg = bool(style.get("legend_bold", False))
    fam_leg = "Arial Black" if bold_leg else "Arial"

    _isz = str(style.get("legend_itemsizing", "constant")).lower()
    if _isz not in ("constant", "trace"):
        _isz = "constant"
    common = dict(
        bgcolor="rgba(255,255,255,0.75)",
        bordercolor="black",
        borderwidth=0.5,
        font=dict(size=leg_font, color="black", family=fam_leg),
        itemsizing=_isz,
    )
    if ncol <= 1:
        return dict(orientation="v", tracegroupgap=2, **xy, **common)
    return dict(
        orientation="h",
        x=xy["x"],
        y=xy["y"],
        xanchor=xy["xanchor"],
        yanchor=xy["yanchor"],
        entrywidthmode="fraction",
        entrywidth=float(1.0 / ncol),
        **common,
    )


def _plotly_layout_meta(fig) -> Dict[str, Any]:
    """Inspect figure layout for responsive margin sizing."""
    y_title_lens: List[int] = []
    has_secondary_y = False
    has_x_title = False
    layout = fig.layout
    for key in layout:
        ks = str(key)
        if ks.startswith("yaxis"):
            ax = layout[key]
            if ax is None:
                continue
            title = ""
            if getattr(ax, "title", None) and getattr(ax.title, "text", None):
                title = str(ax.title.text)
            if title:
                plain = title.replace("<br>", " ").replace("<br/>", " ")
                y_title_lens.append(len(plain))
            if getattr(ax, "overlaying", None) or (
                getattr(ax, "side", None) == "right" and ks != "yaxis"
            ):
                has_secondary_y = True
        elif ks.startswith("xaxis"):
            ax = layout[key]
            if ax is not None and getattr(ax, "title", None) and getattr(ax.title, "text", None):
                has_x_title = True
    return {
        "n_yaxes": max(1, len(y_title_lens)),
        "n_traces": len(fig.data or []),
        "max_y_title_len": max(y_title_lens) if y_title_lens else 0,
        "has_secondary_y": has_secondary_y,
        "has_x_title": has_x_title,
    }


def apply_responsive_plotly_layout(fig, style: Optional[Dict[str, Any]] = None) -> Any:
    """Fit Plotly figures to container width; reduce label/legend clipping on narrow screens."""
    if fig is None:
        return fig
    try:
        meta = _plotly_layout_meta(fig)
        style = style or {}
        ncol = max(1, int(style.get("legend_ncol", 1)))
        show_leg = bool(style.get("show_legend", True))

        left = max(58, min(128, 48 + meta["max_y_title_len"] * 4))
        if meta["n_yaxes"] >= 5:
            left = max(left, 74)
        elif meta["n_yaxes"] >= 3:
            left = max(left, 66)

        bottom = 54
        if meta["has_x_title"]:
            bottom += 6
        if show_leg and ncol > 1:
            bottom = max(bottom, 58 + 14 * min(ncol, 6))
        elif show_leg and meta["n_traces"] > 10:
            bottom = max(bottom, 64)

        top = 64
        if getattr(fig.layout, "title", None) and getattr(fig.layout.title, "text", None):
            tit = str(fig.layout.title.text)
            top = 90 if "<br" in tit.lower() else 72

        right = 30
        if meta["has_secondary_y"]:
            right = max(right, 52)
        if show_leg and meta["n_traces"] > 14:
            right = max(right, 38)

        fig.update_layout(
            autosize=True,
            width=None,
            margin=dict(l=left, r=right, t=top, b=bottom, pad=2),
        )
        fig.update_yaxes(automargin=True, title_standoff=12)
        fig.update_xaxes(automargin=True)
    except Exception:
        pass
    return fig


def show_plotly_chart(
    fig,
    stage_defs=None,
    *,
    show_stage_legend: bool = True,
    external_series_legend: bool = False,
    series_legend_title: str = "Sensors / series",
) -> None:
    """Display a Plotly figure at full column width with responsive resize."""
    if fig is None:
        return
    st.plotly_chart(fig, width="stretch", config=PLOTLY_CHART_CONFIG)
    if external_series_legend:
        render_series_legend_outside(fig, title=series_legend_title)
    if show_stage_legend and stage_defs:
        render_stage_legend_outside(stage_defs)


def show_matplotlib_fig(fig, stage_defs=None, *, show_stage_legend: bool = True) -> None:
    """Display a Matplotlib figure with tight layout for narrow viewports."""
    if fig is None:
        return
    try:
        fig.tight_layout()
    except Exception:
        pass
    st.pyplot(fig, width="stretch")
    if show_stage_legend and stage_defs:
        render_stage_legend_outside(stage_defs)


def apply_plotly_style(fig, style: Dict[str, Any]) -> Any:
    """Apply fonts, grid, legend from a style dict (from per-page options)."""
    if fig is None:
        return fig
    base_style: Dict[str, Any] = {
        "title_fs": 18,
        "title_bold": True,
        "axis_title_fs": 18,
        "axis_title_bold": True,
        "tick_fs": 16,
        "tick_bold": True,
        "legend_ncol": 1,
        "legend_pos": LEGEND_POSITION_LABELS[1],
        "legend_bold": True,
        "legend_fs": 12,
        "show_legend": True,
    }
    style = {**base_style, **(style or {})}
    try:
        fs_t = int(style.get("title_fs", 18))
        bold_t = bool(style.get("title_bold", True))
        fs_at = int(style.get("axis_title_fs", 13))
        bold_at = bool(style.get("axis_title_bold", False))
        fs_tick = int(style.get("tick_fs", 11))
        bold_tick = bool(style.get("tick_bold", False))
        show_leg = bool(style.get("show_legend", True))

        fam_axis = "Arial Black" if bold_at else "Arial"
        fam_tick = "Arial Black" if bold_tick else "Arial"

        layout_kw: Dict[str, Any] = dict(
            template="none",
            paper_bgcolor="white",
            plot_bgcolor="white",
            font=dict(size=fs_tick, color="black"),
        )
        if show_leg:
            layout_kw["legend"] = _legend_layout_from_style(style)
        else:
            layout_kw["showlegend"] = False
        fig.update_layout(**layout_kw)

        if getattr(fig.layout, "title", None) and getattr(fig.layout.title, "text", None):
            t = str(fig.layout.title.text).replace("<b>", "").replace("</b>", "")
            if bold_t:
                fig.update_layout(title=dict(text=f"<b>{t}</b>", font=dict(size=fs_t, color="black")))
            else:
                fig.update_layout(title=dict(text=t, font=dict(size=fs_t, color="black")))

        if fig.layout.annotations:
            for ann in fig.layout.annotations:
                if getattr(ann, "text", None):
                    txt = str(ann.text).replace("<b>", "").replace("</b>", "")
                    if bold_at:
                        ann.text = f"<b>{txt}</b>"
                    else:
                        ann.text = txt
                    ann.font = dict(size=fs_at, color="black", family=fam_axis)

        grid_col = "rgba(0,0,0,0.18)"
        axis_common = dict(
            showline=True,
            linecolor="black",
            linewidth=1,
            mirror=True,
            ticks="outside",
            tickcolor="black",
            ticklen=5,
            tickwidth=1,
            showgrid=True,
            gridcolor=grid_col,
            zeroline=False,
            tickfont=dict(size=fs_tick, color="black", family=fam_tick),
            title_font=dict(size=fs_at, color="black", family=fam_axis),
            automargin=True,
        )
        fig.update_xaxes(**axis_common)
        yaxis2 = getattr(fig.layout, "yaxis2", None)
        has_overlay_y2 = bool(
            yaxis2 is not None and getattr(yaxis2, "overlaying", None)
        )
        if has_overlay_y2:
            y2_grid = dict(showgrid=False)
            fig.update_yaxes(**axis_common)
            fig.update_layout(yaxis2=y2_grid)
        else:
            fig.update_yaxes(**axis_common)
        apply_responsive_plotly_layout(fig, style)
    except Exception:
        pass
    return fig


def _ensure_widget_defaults(prefix: str, defaults: Dict[str, Any]) -> None:
    snap = st.session_state.get(f"{prefix}__USER_SNAPSHOT")
    base = {**defaults, **(snap or {})}
    force = bool(st.session_state.get("__force_defaults_from_upload", False))
    for k, v in base.items():
        sk = f"{prefix}__{k}"
        if force or (sk not in st.session_state):
            st.session_state[sk] = v


def _reset_widgets(prefix: str, values: Dict[str, Any]) -> None:
    for k, v in values.items():
        st.session_state[f"{prefix}__{k}"] = v


def _merged_defaults_with_snapshot(prefix: str, defaults: Dict[str, Any]) -> Dict[str, Any]:
    """Built-in defaults overridden by user snapshot (only keys that exist in defaults)."""
    snap = st.session_state.get(f"{prefix}__USER_SNAPSHOT") or {}
    out = {**defaults}
    for k, v in snap.items():
        if k in defaults:
            out[k] = v
    # Global publication style: keep legend text bold by default across all pages.
    # Users can still turn it off in the current run, but we don't persist "non-bold" as a default.
    if "legend_bold" in out:
        out["legend_bold"] = True
    return out


def _style_from_prefix(prefix: str) -> Dict[str, Any]:
    keys = [
        "title_fs",
        "title_bold",
        "axis_title_fs",
        "axis_title_bold",
        "tick_fs",
        "tick_bold",
        "legend_ncol",
        "legend_pos",
        "legend_bold",
        "legend_fs",
        "show_legend",
        "legend_itemsizing",
    ]
    out: Dict[str, Any] = {}
    for k in keys:
        v = st.session_state.get(f"{prefix}__{k}")
        if v is not None:
            out[k] = v
    return out


def _collect_ylims_from_prefix(prefix: str, ykeys: List[Tuple[str, str]], fallback: Dict[str, Tuple[float, float]]) -> Dict[str, Tuple[float, float]]:
    out = {}
    for key, _ in ykeys:
        lo = float(st.session_state.get(f"{prefix}__y_{key}_min", fallback[key][0]))
        hi = float(st.session_state.get(f"{prefix}__y_{key}_max", fallback[key][1]))
        if lo >= hi:
            lo, hi = fallback[key]
        out[key] = (lo, hi)
    return out


def _x_mode_option_list(stage_defs) -> List[str]:
    opts = ["Full data (+ pre-minutes)", "Manual (time slider)"]
    if stage_defs:
        for (name, _, _, _) in stage_defs:
            opts.append(f"Stage — {name}")
    return opts


def render_x_mode_widgets(prefix: str, t0, t1, stage_defs) -> None:
    """Render x-axis mode widgets (must run before render_x_controls)."""
    if t0 is None or pd.isna(t0) or t1 is None or pd.isna(t1):
        return
    t0p, t1p = pd.Timestamp(t0), pd.Timestamp(t1)
    opts = _x_mode_option_list(stage_defs)
    cur = st.session_state.get(f"{prefix}__x_mode")
    if cur not in opts:
        st.session_state[f"{prefix}__x_mode"] = opts[0]
    st.selectbox("X-axis window", options=opts, key=f"{prefix}__x_mode")
    st.number_input("Pre-minutes before data start (full-data mode)", min_value=0, max_value=24 * 60, step=5, key=f"{prefix}__pre_min")

    mode = st.session_state.get(f"{prefix}__x_mode", opts[0])
    if mode == "Manual (time slider)":
        k = f"{prefix}__x_manual"
        if k not in st.session_state:
            st.session_state[k] = (t0p.to_pydatetime(), t1p.to_pydatetime())
        st.slider(
            "Manual time range",
            min_value=t0p.to_pydatetime(),
            max_value=t1p.to_pydatetime(),
            key=k,
            help="Only used when X-axis window is Manual.",
        )


def render_x_controls(prefix: str, t0, t1, stage_defs) -> Tuple[Any, Any]:
    """Returns (x_start, x_end) as Timestamps or None."""
    if t0 is None or pd.isna(t0) or t1 is None or pd.isna(t1):
        return None, None
    t0p, t1p = pd.Timestamp(t0), pd.Timestamp(t1)
    mode = st.session_state.get(f"{prefix}__x_mode", "Full data (+ pre-minutes)")
    pre = int(st.session_state.get(f"{prefix}__pre_min", 0))

    if mode == "Manual (time slider)":
        pair = st.session_state.get(f"{prefix}__x_manual")
        if pair is not None:
            return pd.Timestamp(pair[0]), pd.Timestamp(pair[1])
        return t0p - pd.Timedelta(minutes=pre), t1p

    if mode.startswith("Stage — ") and stage_defs:
        label = mode.replace("Stage — ", "", 1)
        for (name, stt, ett, _) in stage_defs:
            if str(name) == label:
                return pd.Timestamp(stt), pd.Timestamp(ett)
        return t0p, t1p

    return t0p - pd.Timedelta(minutes=pre), t1p


def series_mean_in_window(series: pd.Series, x0, x1) -> float:
    """Mean of a time-indexed series within [x0, x1] (inclusive)."""
    if series is None or len(series) == 0:
        return float("nan")
    s = series.dropna()
    if len(s) == 0:
        return float("nan")
    if x0 is not None and x1 is not None:
        x0p, x1p = pd.Timestamp(x0), pd.Timestamp(x1)
        s = s[(s.index >= x0p) & (s.index <= x1p)]
    if len(s) == 0:
        return float("nan")
    return float(s.mean())


def render_font_legend_widgets(prefix: str, show_legend_toggle: bool = False) -> None:
    st.markdown("**Fonts**")
    c1, c2 = st.columns(2)
    with c1:
        st.slider("Figure title size", 10, 32, key=f"{prefix}__title_fs")
        st.checkbox("Bold figure title", key=f"{prefix}__title_bold")
    with c2:
        st.slider("Axis title size", 8, 24, key=f"{prefix}__axis_title_fs")
        st.checkbox("Bold axis titles", key=f"{prefix}__axis_title_bold")
    c3, c4 = st.columns(2)
    with c3:
        st.slider("Tick / axis label size", 8, 22, key=f"{prefix}__tick_fs")
    with c4:
        st.checkbox("Bold tick labels", key=f"{prefix}__tick_bold")
    st.markdown("**Legend**")
    if show_legend_toggle:
        st.checkbox("Show legend", key=f"{prefix}__show_legend")
    lc1, lc2 = st.columns(2)
    with lc1:
        st.number_input("Legend columns", min_value=1, max_value=6, step=1, key=f"{prefix}__legend_ncol")
    with lc2:
        st.selectbox("Legend position", options=LEGEND_POSITION_LABELS, key=f"{prefix}__legend_pos")
    st.slider("Legend text size", min_value=5, max_value=24, step=1, key=f"{prefix}__legend_fs")
    st.checkbox("Bold legend text", key=f"{prefix}__legend_bold")


def render_series_line_marker_widgets(prefix: str) -> None:
    st.markdown("**Series (plot & legend sample)**")
    st.caption('Line width and marker size apply to the chart; legend icons follow these when "Match plot" is selected.')
    c1, c2 = st.columns(2)
    with c1:
        st.slider("Line width", min_value=0.25, max_value=6.0, step=0.25, key=f"{prefix}__line_width")
    with c2:
        st.slider("Marker size", min_value=2, max_value=24, step=1, key=f"{prefix}__marker_size")
    st.selectbox(
        "Legend icon sizing",
        options=["trace", "constant"],
        format_func=lambda s: "Match plot (line & markers)" if s == "trace" else "Compact (fixed size)",
        key=f"{prefix}__legend_itemsizing",
    )


def render_save_reset_row(prefix: str, defaults: Dict[str, Any]) -> None:
    c1, c2, c3 = st.columns(3)
    with c1:

        def _do_reset():
            merged = _merged_defaults_with_snapshot(prefix, defaults)
            _reset_widgets(prefix, merged)

        st.button(
            "Reset defaults",
            key=f"{prefix}__btn_reset",
            on_click=_do_reset,
            help="Built-in suggested values, then your saved values (from Save) override where you have saved.",
        )
    with c2:

        def _do_save():
            snap = {
                k: st.session_state[f"{prefix}__{k}"]
                for k in defaults
                if f"{prefix}__{k}" in st.session_state
            }
            st.session_state[f"{prefix}__USER_SNAPSHOT"] = snap

        st.button(
            "Save current as my defaults",
            key=f"{prefix}__btn_save",
            on_click=_do_save,
            help="Stores current settings. Use Reset to apply them; new widget keys pick them up on first run.",
        )
    with c3:

        def _do_clear_save():
            st.session_state.pop(f"{prefix}__USER_SNAPSHOT", None)
            _reset_widgets(prefix, defaults)

        st.button(
            "Clear my save",
            key=f"{prefix}__btn_clear_save",
            on_click=_do_clear_save,
            help="Remove saved defaults and reset this section to built-in suggested values only.",
        )


def _y_pair_from_prefix(prefix: str, fb_lo: float, fb_hi: float) -> Tuple[float, float]:
    lo = float(st.session_state.get(f"{prefix}__y_min", fb_lo))
    hi = float(st.session_state.get(f"{prefix}__y_max", fb_hi))
    if lo >= hi:
        return fb_lo, fb_hi
    return lo, hi


def _prof_x_range(prefix: str) -> Optional[Tuple[float, float]]:
    if not st.session_state.get(f"{prefix}__x_use_manual", False):
        return None
    lo = float(st.session_state.get(f"{prefix}__x_vmin", 0.0))
    hi = float(st.session_state.get(f"{prefix}__x_vmax", 1.0))
    if lo >= hi:
        return None
    return (lo, hi)


def _prof_yz_range(prefix: str) -> Optional[Tuple[float, float]]:
    if not st.session_state.get(f"{prefix}__use_fixed_y_z", True):
        return None
    lo = float(st.session_state.get(f"{prefix}__y_z_min", 0.0))
    hi = float(st.session_state.get(f"{prefix}__y_z_max", 10.0))
    if lo >= hi:
        return None
    return (lo, hi)


def _line_marker_from_prefix(prefix: str) -> Tuple[float, float]:
    lw = float(st.session_state.get(f"{prefix}__line_width", 2.0))
    ms = float(st.session_state.get(f"{prefix}__marker_size", 6.0))
    return max(0.25, lw), max(1.0, ms)


def _legend_fs_from_prefix(prefix: str) -> int:
    v = st.session_state.get(f"{prefix}__legend_fs")
    if v is not None:
        return max(5, min(24, int(v)))
    t = int(st.session_state.get(f"{prefix}__tick_fs", 10))
    return max(6, t - 4)


def render_prof_panel_options(prefix: str, defaults: Dict[str, Any]) -> None:
    _ensure_widget_defaults(prefix, defaults)
    render_save_reset_row(prefix, defaults)
    render_font_legend_widgets(prefix, show_legend_toggle=True)
    render_series_line_marker_widgets(prefix)
    st.checkbox("Manual x-axis limits (mean value axis)", key=f"{prefix}__x_use_manual")
    c1, c2 = st.columns(2)
    with c1:
        st.number_input("X min", key=f"{prefix}__x_vmin", format="%.4g")
    with c2:
        st.number_input("X max", key=f"{prefix}__x_vmax", format="%.4g")
    st.checkbox("Fix z-axis extent", key=f"{prefix}__use_fixed_y_z")
    c3, c4 = st.columns(2)
    with c3:
        st.number_input("Z min", key=f"{prefix}__y_z_min", format="%.4g")
    with c4:
        st.number_input("Z max", key=f"{prefix}__y_z_max", format="%.4g")


OVERALL_Y_KEYS = [
    ("co2_mean", "CO₂ mean"),
    ("co2_std", "CO₂ std"),
    ("co2_cv", "CO₂ CV"),
    ("co2_mi", "CO₂ MI"),
    ("co2_coverage", "CO₂ coverage %"),
    ("temp_mean", "Temp mean"),
    ("temp_std", "Temp std"),
    ("temp_deltaT", "Temp ΔT(H-L)"),
    ("temp_r2", "Temp R²"),
    ("temp_mi", "Temp MI"),
    ("temp_pk_minus_cave", "ΔT PK−CAVE"),
]

RH_OVERVIEW_Y_KEYS = [
    ("rh_mean", "Mean RH (%)"),
    ("rh_std", "Std RH (%)"),
]

OVERALL_WIDGET_DEFAULTS: Dict[str, Any] = {
    "title_fs": 18,
    "title_bold": True,
    "axis_title_fs": 18,
    "axis_title_bold": True,
    "tick_fs": 16,
    "tick_bold": True,
    "legend_ncol": 1,
    "legend_pos": LEGEND_POSITION_LABELS[1],
    "legend_bold": True,
    "legend_fs": 12,
    "show_subplot_titles": False,
    "use_fixed_y": True,
    "pre_min": 0,
    "x_mode": "Full data (+ pre-minutes)",
    "line_width": 3.0,
    "marker_size": 10,
    "legend_itemsizing": "constant",
}

ZONE_WIDGET_DEFAULTS: Dict[str, Any] = {
    **{k: v for k, v in OVERALL_WIDGET_DEFAULTS.items() if k not in ("use_fixed_y",)},
    "use_fixed_y": True,
    "pre_min": 0,
    "x_mode": "Full data (+ pre-minutes)",
}

PROF_WIDGET_DEFAULTS: Dict[str, Any] = {
    "title_fs": 18,
    "title_bold": True,
    "axis_title_fs": 18,
    "axis_title_bold": True,
    "tick_fs": 16,
    "tick_bold": True,
    "legend_ncol": 1,
    "legend_pos": LEGEND_POSITION_LABELS[0],
    "legend_bold": True,
    "legend_fs": 12,
    "show_legend": True,
    "line_width": 3.0,
    "marker_size": 10,
    "legend_itemsizing": "trace",
    "x_use_manual": False,
    "x_vmin": 0.0,
    "x_vmax": 1.0,
    "y_z_min": 0.5,
    "y_z_max": 10.5,
    "use_fixed_y_z": True,
}

MFC_WIDGET_DEFAULTS: Dict[str, Any] = {
    **{k: v for k, v in ZONE_WIDGET_DEFAULTS.items()},
    "lock_x_release": True,
    "y_min": 0.0,
    "y_max": 1.0,
    "use_custom_y": False,
    "line_width": 3.0,
}

RH_PAGE_DEFAULTS: Dict[str, Any] = {
    **ZONE_WIDGET_DEFAULTS,
    **{f"y_{k}_min": default_ylims()[k][0] for k, _ in RH_OVERVIEW_Y_KEYS},
    **{f"y_{k}_max": default_ylims()[k][1] for k, _ in RH_OVERVIEW_Y_KEYS},
}

_ylim0 = default_ylims()
OVERALL_PAGE_DEFAULTS: Dict[str, Any] = {
    **OVERALL_WIDGET_DEFAULTS,
    **{f"y_{k}_min": _ylim0[k][0] for k, _ in OVERALL_Y_KEYS},
    **{f"y_{k}_max": _ylim0[k][1] for k, _ in OVERALL_Y_KEYS},
}


def _export_plotly_style() -> Dict[str, Any]:
    return {
        k: OVERALL_WIDGET_DEFAULTS[k]
        for k in ("title_fs", "title_bold", "axis_title_fs", "axis_title_bold", "tick_fs", "tick_bold", "legend_ncol", "legend_pos", "legend_bold")
    }


def zone_ts_page_defaults(ylims: Dict[str, Tuple[float, float]], ykey: str) -> Dict[str, Any]:
    lo, hi = ylims[ykey]
    return {**ZONE_WIDGET_DEFAULTS, "y_min": float(lo), "y_max": float(hi), "show_markers": False}


# =========================================================
# Plotly (interactive hover)
# =========================================================
def _require_plotly():
    if go is None:
        raise RuntimeError("Plotly is not installed. Please run: pip install plotly")


def plot_zone_single_plotly(
    zone_df: pd.DataFrame,
    title: str,
    y_title: str,
    stage_defs,
    plot_start,
    plot_end,
    y_range=None,
    show_markers: bool = False,
    line_width: float = 2.0,
    marker_size: float = 6.0,
    legend_in_plot: bool = True,
):
    _require_plotly()
    fig = go.Figure()
    if zone_df is None or zone_df.empty:
        fig.update_layout(title=title)
        return fig

    lw = max(0.25, float(line_width))
    ms = max(1.0, float(marker_size))
    mode = "lines+markers" if show_markers else "lines"
    fig.update_layout(colorway=list(_PLOTLY_SERIES_COLORS))
    for i, col in enumerate(zone_df.columns):
        s = zone_df[col].dropna()
        color = _series_color_for_index(i)
        fig.add_trace(
            go.Scatter(
                x=s.index,
                y=s.values,
                mode=mode,
                name=str(col),
                line=dict(width=lw, color=color),
                marker=dict(size=ms, color=color),
                showlegend=legend_in_plot,
            )
        )

    add_plotly_stage_vrects(fig, stage_defs)

    if plot_start is not None and plot_end is not None:
        fig.update_xaxes(range=[plot_start, plot_end])
    if y_range is not None:
        fig.update_yaxes(range=list(y_range))

    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title=y_title,
        height=520,
        showlegend=legend_in_plot,
    )
    return fig


def plot_io_ratio_plotly(io_ex, infiltration_factor, t_rel0, t_rel1, t_base0, t_base1, ex_thresh, cfg: AppConfig):
    _require_plotly()
    fig = go.Figure()

    s = io_ex.dropna()
    fig.add_trace(
        go.Scatter(
            x=s.index,
            y=s.values,
            mode="lines+markers",
            name="I/O_ex(t) = PK_ex / CAVE_ex (thresholded)",
            line=dict(width=2),
            marker=dict(size=5, opacity=0.6),
            hovertemplate="t=%{x|%Y-%m-%d %H:%M:%S}<br>I/O_ex=%{y:.4f}<extra></extra>",
        )
    )

    # Shaded windows
    if (t_rel0 is not None) and (t_rel1 is not None):
        fig.add_vrect(x0=t_rel0, x1=t_rel1, fillcolor="orange", opacity=0.15, line_width=0, annotation_text="Release", annotation_position="top left")
    if (t_base0 is not None) and (t_base1 is not None):
        fig.add_vrect(x0=t_base0, x1=t_base1, fillcolor="gray", opacity=0.10, line_width=0, annotation_text="Baseline", annotation_position="bottom left")

    # Mean line during release
    if np.isfinite(infiltration_factor):
        fig.add_hline(y=infiltration_factor, line_dash="dash", line_width=2, annotation_text=f"mean={infiltration_factor:.3f}", annotation_position="top right")

    fig.update_layout(
        title=f"{cfg.exp_code} — Excess I/O ratio",
        xaxis_title="Time",
        yaxis_title="I/O_ex (-)",
        template="plotly_white",
        height=420,
        margin=dict(l=40, r=20, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        clickmode="event+select",
    )

    if (t_rel0 is not None) and (t_rel1 is not None):
        fig.update_xaxes(range=[t_rel0, t_rel1])

    if cfg.use_fixed_ylims:
        fig.update_yaxes(range=list(cfg.ylims["io_ex"]))

    # Threshold note (as annotation)
    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0.01,
        y=0.02,
        showarrow=False,
        text=f"Threshold: CAVE_ex > {ex_thresh:.1f} ppm",
        font=dict(size=11),
        align="left",
    )

    return fig


def plot_scatter_plotly(df_sc, slope, intercept, r2, cfg: AppConfig):
    _require_plotly()
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df_sc["cave_ex"],
            y=df_sc["pk_ex"],
            mode="markers",
            name="Release points (thresholded)",
            marker=dict(size=7, opacity=0.8),
            hovertemplate="CAVE_ex=%{x:.2f} ppm<br>PK_ex=%{y:.2f} ppm<extra></extra>",
        )
    )

    if np.isfinite(slope) and len(df_sc) >= 2:
        x0 = float(df_sc["cave_ex"].min())
        x1 = float(df_sc["cave_ex"].max())
        xs = np.array([x0, x1])
        ys = intercept + slope * xs
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                name=f"Fit slope={slope:.3f}, R²={r2:.3f}",
                line=dict(dash="dash", width=2),
                hoverinfo="skip",
            )
        )

    fig.update_layout(
        title=f"{cfg.exp_code} — PK_ex vs CAVE_ex (Release only)",
        xaxis_title="CAVE_ex (ppm)",
        yaxis_title="PK_ex (ppm)",
        template="plotly_white",
        height=520,
        margin=dict(l=40, r=20, t=60, b=40),
        clickmode="event+select",
    )

    if cfg.use_fixed_ylims:
        fig.update_xaxes(range=list(cfg.ylims["scatter_cave_ex"]))
        fig.update_yaxes(range=list(cfg.ylims["scatter_pk_ex"]))

    return fig


def plot_overall_metrics_plotly(
    co2_cave,
    co2_pk,
    temp_cave,
    temp_pk,
    deltaT_pk_minus_cave,
    stage_defs,
    cfg: AppConfig,
    plot_start,
    plot_end,
    ylims_src: Optional[Dict[str, Tuple[float, float]]] = None,
    use_fixed_y: Optional[bool] = None,
    show_subplot_titles: bool = False,
    line_width: float = 2.0,
    marker_size: float = 6.0,
):
    _require_plotly()
    if make_subplots is None:
        raise RuntimeError("Plotly subplots not available")

    lw_c = max(0.25, float(line_width) * 1.5)
    lw_p = max(0.25, float(line_width) * 1.0)
    lw_d = max(0.25, float(line_width) * 1.0)
    _ms = max(1.0, float(marker_size))

    cave_color = "#1f77b4"
    pk_color = "#ff7f0e"

    cov_detail = f"(CO₂ ≥ baseline×{cfg.coverage_factor:.2f})"
    title_cov_banner = f"Coverage {cov_detail}"
    title_cov_yaxis = f"Coverage<br>{cov_detail}"

    titles_co2_banner = ["Mean CO₂", "Std CO₂", "CV (CO₂)", "Mixing Index (CO₂)", title_cov_banner]
    titles_co2_yaxis = [
        "Mean CO₂",
        "Std CO₂",
        "CV (CO₂)",
        "Mixing Index<br>(CO₂)",
        title_cov_yaxis,
    ]
    titles_T = ["Mean T (°C)", "Std T (°C)", "ΔT(high-low) (°C)", "R²(T~z)", "Mixing Index (T)"]

    # subplot_titles are assigned in row-major order: (r1,c1), (r1,c2), (r2,c1), ...
    subplot_titles_rowmajor = [t for i in range(5) for t in (titles_co2_banner[i], titles_T[i])]

    fig = make_subplots(
        rows=5,
        cols=2,
        shared_xaxes=True,
        vertical_spacing=0.04,
        horizontal_spacing=0.11,
        row_heights=[0.22, 0.20, 0.20, 0.19, 0.19],
        specs=[
            [{"secondary_y": False}, {"secondary_y": True}],
            [{"secondary_y": False}, {"secondary_y": False}],
            [{"secondary_y": False}, {"secondary_y": False}],
            [{"secondary_y": False}, {"secondary_y": False}],
            [{"secondary_y": False}, {"secondary_y": False}],
        ],
        **({"subplot_titles": subplot_titles_rowmajor} if show_subplot_titles else {}),
    )

    # Left: CO2
    metrics_co2 = ["mean", "std", "cv", "mi", "coverage"]
    for i, m in enumerate(metrics_co2, start=1):
        s_c = co2_cave[m].dropna()
        s_p = co2_pk[m].dropna()
        # Only show legend once for repeated CAVE/PK traces (single global legend, inside the figure)
        show_leg = (i == 1)
        fig.add_trace(
            go.Scatter(
                x=s_c.index,
                y=s_c.values,
                mode="lines",
                name="CAVE",
                line=dict(width=lw_c, color=cave_color),
                marker=dict(size=_ms),
                legendgroup="CAVE",
                showlegend=show_leg,
            ),
            row=i, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=s_p.index,
                y=s_p.values,
                mode="lines",
                name="PK",
                line=dict(width=lw_p, dash="dash", color=pk_color),
                marker=dict(size=_ms),
                legendgroup="PK",
                showlegend=show_leg,
            ),
            row=i, col=1,
        )

    # Right: Temperature
    metrics_T = ["mean_T", "std_T", "deltaT", "r2_Tz", "mi_T"]
    for i, m in enumerate(metrics_T, start=1):
        s_c = temp_cave[m].dropna()
        s_p = temp_pk[m].dropna()
        fig.add_trace(
            go.Scatter(
                x=s_c.index,
                y=s_c.values,
                mode="lines",
                name="CAVE",
                line=dict(width=lw_c, color=cave_color),
                marker=dict(size=_ms),
                legendgroup="CAVE",
                showlegend=False,
            ),
            row=i, col=2,
        )
        fig.add_trace(
            go.Scatter(
                x=s_p.index,
                y=s_p.values,
                mode="lines",
                name="PK",
                line=dict(width=lw_p, dash="dash", color=pk_color),
                marker=dict(size=_ms),
                legendgroup="PK",
                showlegend=False,
            ),
            row=i, col=2,
        )

    # Extra ΔT (PK-CAVE) in first temp subplot
    s_dt = deltaT_pk_minus_cave.dropna()
    fig.add_trace(
        go.Scatter(
            x=s_dt.index,
            y=s_dt.values,
            mode="lines",
            name="ΔT (PK − CAVE)",
            line=dict(width=lw_d, dash="dot"),
            marker=dict(size=_ms),
            showlegend=True,
        ),
        row=1,
        col=2,
        secondary_y=True,
    )

    # Stage shading on all panels (legend rendered below chart via render_stage_legend_outside)
    add_plotly_stage_vrects(fig, stage_defs)

    # Axes and ranges
    for i in range(1, 6):
        fig.update_yaxes(title_text=titles_co2_yaxis[i - 1], row=i, col=1)
        fig.update_yaxes(title_text=titles_T[i - 1], row=i, col=2, secondary_y=False)
    # Right axis label for ΔT(PK-CAVE)
    fig.update_yaxes(title_text="ΔT (PK − CAVE) (°C)", row=1, col=2, secondary_y=True)

    if plot_start is not None and plot_end is not None:
        fig.update_xaxes(range=[plot_start, plot_end])

    apply_y = cfg.use_fixed_ylims if use_fixed_y is None else bool(use_fixed_y)
    yref = ylims_src if ylims_src is not None else cfg.ylims
    if apply_y and yref is not None:
        y = yref
        fig.update_yaxes(range=list(y["co2_mean"]), row=1, col=1)
        fig.update_yaxes(range=list(y["co2_std"]), row=2, col=1)
        fig.update_yaxes(range=list(y["co2_cv"]), row=3, col=1)
        fig.update_yaxes(range=list(y["co2_mi"]), row=4, col=1)
        fig.update_yaxes(range=list(y["co2_coverage"]), row=5, col=1)

        fig.update_yaxes(range=list(y["temp_mean"]), row=1, col=2, secondary_y=False)
        fig.update_yaxes(range=list(y["temp_std"]), row=2, col=2)
        fig.update_yaxes(range=list(y["temp_deltaT"]), row=3, col=2)
        fig.update_yaxes(range=list(y["temp_r2"]), row=4, col=2)
        fig.update_yaxes(range=list(y["temp_mi"]), row=5, col=2)
        fig.update_yaxes(range=list(y["temp_pk_minus_cave"]), row=1, col=2, secondary_y=True)

    fig.update_layout(
        height=1100,
        title=f"{cfg.exp_code} — Overall metrics (CAVE vs PK)",
        # Style (axes/legend/fonts) is applied in apply_plotly_style()
        showlegend=True,
    )

    # X-axis: show clock time only (no calendar date on ticks); full stamp still in hover
    fig.update_xaxes(tickformat="%H:%M", hoverformat="%Y-%m-%d %H:%M:%S")

    return fig


def plot_zone_co2_plotly(cave_zone_co2, pk_zone_co2, stage_defs, cfg: AppConfig, plot_start, plot_end):
    _require_plotly()
    if make_subplots is None:
        raise RuntimeError("Plotly subplots not available")

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.07,
        row_heights=[0.5, 0.5],
    )

    # CAVE
    for j, col in enumerate(cave_zone_co2.columns):
        s = cave_zone_co2[col].dropna()
        fig.add_trace(
            go.Scatter(
                x=s.index,
                y=s.values,
                mode="lines+markers",
                name=str(col),
                legendgroup="CAVE_WALLS",
                legendgrouptitle_text="CAVE walls",
                showlegend=True,
            ),
            row=1,
            col=1,
        )

    # PK
    for j, col in enumerate(pk_zone_co2.columns):
        s = pk_zone_co2[col].dropna()
        fig.add_trace(
            go.Scatter(
                x=s.index,
                y=s.values,
                mode="lines+markers",
                name=str(col),
                legendgroup="PK_WALLS",
                legendgrouptitle_text="PK walls",
                showlegend=True,
            ),
            row=2,
            col=1,
        )

    # Stage shading
    if stage_defs:
        for (name, stt, ett, colr) in stage_defs:
            fig.add_vrect(x0=stt, x1=ett, fillcolor=colr, opacity=0.08, line_width=0, row="all", col=1)

    if plot_start is not None and plot_end is not None:
        fig.update_xaxes(range=[plot_start, plot_end])

    if cfg.use_fixed_ylims:
        fig.update_yaxes(range=list(cfg.ylims["zone_cave_co2"]), row=1, col=1)
        fig.update_yaxes(range=list(cfg.ylims["zone_pk_co2"]), row=2, col=1)

    fig.update_yaxes(title_text="CO₂ (ppm)", row=1, col=1)
    fig.update_yaxes(title_text="CO₂ (ppm)", row=2, col=1)
    fig.update_xaxes(title_text="Time", row=2, col=1)

    fig.update_layout(
        height=850,
        title_text=f"{cfg.exp_code} — Zone CO₂ (CAVE walls & PK walls)",
        template="plotly_white",
    )

    return fig


def plot_zone_temp_plotly(cave_zone_temp, pk_zone_temp, stage_defs, cfg: AppConfig, plot_start, plot_end):
    _require_plotly()
    if make_subplots is None:
        raise RuntimeError("Plotly subplots not available")

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.07,
        row_heights=[0.5, 0.5],
    )

    # CAVE
    for j, col in enumerate(cave_zone_temp.columns):
        s = cave_zone_temp[col].dropna()
        fig.add_trace(
            go.Scatter(
                x=s.index,
                y=s.values,
                mode="lines+markers",
                name=str(col),
                legendgroup="CAVE_WALLS_T",
                legendgrouptitle_text="CAVE walls (T)",
                showlegend=True,
            ),
            row=1,
            col=1,
        )

    # PK
    for j, col in enumerate(pk_zone_temp.columns):
        s = pk_zone_temp[col].dropna()
        fig.add_trace(
            go.Scatter(
                x=s.index,
                y=s.values,
                mode="lines+markers",
                name=str(col),
                legendgroup="PK_WALLS_T",
                legendgrouptitle_text="PK walls (T)",
                showlegend=True,
            ),
            row=2,
            col=1,
        )

    # Stage shading
    if stage_defs:
        for (name, stt, ett, colr) in stage_defs:
            fig.add_vrect(x0=stt, x1=ett, fillcolor=colr, opacity=0.08, line_width=0, row="all", col=1)

    if plot_start is not None and plot_end is not None:
        fig.update_xaxes(range=[plot_start, plot_end])

    fig.update_yaxes(title_text="Temperature (°C)", row=1, col=1)
    fig.update_yaxes(title_text="Temperature (°C)", row=2, col=1)
    fig.update_xaxes(title_text="Time", row=2, col=1)

    fig.update_layout(
        height=850,
        title_text=f"{cfg.exp_code} — Zone temperature (CAVE walls & PK walls)",
        template="plotly_white",
    )

    return fig


def plot_humidity_overview_plotly(
    rh_cave,
    rh_pk,
    stage_defs,
    cfg: AppConfig,
    plot_start,
    plot_end,
    ylims_src: Optional[Dict[str, Tuple[float, float]]] = None,
    use_fixed_y: bool = True,
    line_width: float = 2.0,
):
    _require_plotly()
    if make_subplots is None:
        raise RuntimeError("Plotly subplots not available")

    lw_c = max(0.25, float(line_width) * 1.5)
    lw_p = max(0.25, float(line_width) * 1.0)
    cave_color = "#1f77b4"
    pk_color = "#ff7f0e"

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.10,
        subplot_titles=("Mean relative humidity (%)", "Std relative humidity (%)"),
    )

    panels = [("mean", 1), ("std", 2)]
    for metric, row in panels:
        s_c = rh_cave[metric].dropna()
        s_p = rh_pk[metric].dropna()
        fig.add_trace(
            go.Scatter(
                x=s_c.index, y=s_c.values, mode="lines", name="CAVE",
                line=dict(width=lw_c, color=cave_color), legendgroup="CAVE", showlegend=(row == 1),
            ),
            row=row, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=s_p.index, y=s_p.values, mode="lines", name="PK",
                line=dict(width=lw_p, dash="dash", color=pk_color), legendgroup="PK", showlegend=(row == 1),
            ),
            row=row, col=1,
        )
        fig.update_yaxes(title_text="RH (%)" if metric == "mean" else "Std (%)", row=row, col=1)

    add_plotly_stage_vrects(fig, stage_defs)
    if plot_start is not None and plot_end is not None:
        fig.update_xaxes(range=[plot_start, plot_end])
    yref = ylims_src if ylims_src is not None else cfg.ylims
    if use_fixed_y and yref is not None:
        fig.update_yaxes(range=list(yref["rh_mean"]), row=1, col=1)
        fig.update_yaxes(range=list(yref["rh_std"]), row=2, col=1)

    fig.update_layout(
        height=620,
        title=f"{cfg.exp_code} — Humidity overview (CAVE vs PK)",
        showlegend=True,
    )
    fig.update_xaxes(tickformat="%H:%M", hoverformat="%Y-%m-%d %H:%M:%S", row=2, col=1)
    return fig


def plot_mfc_plotly(
    mfc_df,
    t_on,
    t_off,
    t_rel0,
    t_rel1,
    cfg: AppConfig,
    x_start=None,
    x_end=None,
    lock_x_release: bool = True,
    y_range=None,
    line_width: float = 2.2,
):
    _require_plotly()
    if mfc_df is None or mfc_df.empty:
        return None

    lw = max(0.25, float(line_width))
    has_temp = mfc_has_temperature(mfc_df)
    flow_color = "#1f77b4"
    temp_color = "#d62728"

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=mfc_df["t"],
            y=mfc_df["F"],
            mode="lines",
            name="MFC flow (Fmeas if available else Fset)",
            line=dict(width=lw, color=flow_color),
            yaxis="y",
        )
    )

    if has_temp:
        fig.add_trace(
            go.Scatter(
                x=mfc_df["t"],
                y=mfc_df["T"],
                mode="lines",
                name="Temperature (°C)",
                line=dict(width=lw, color=temp_color),
                yaxis="y2",
            )
        )

    fig.add_hline(
        y=cfg.flow_on_th,
        line_dash="dot",
        line_width=max(1.0, lw * 0.85),
        line_color="#444444",
        annotation_text=f"FLOW_ON_TH={cfg.flow_on_th}",
    )

    if (t_on is not None) and (t_off is not None):
        fig.add_vrect(x0=t_on, x1=t_off, fillcolor="green", opacity=0.15, line_width=0)

    if (t_rel0 is not None) and (t_rel1 is not None):
        fig.add_vrect(x0=t_rel0, x1=t_rel1, fillcolor="orange", opacity=0.10, line_width=0)

    if x_start is not None and x_end is not None:
        fig.update_xaxes(range=[x_start, x_end])
    elif lock_x_release and (t_rel0 is not None) and (t_rel1 is not None):
        fig.update_xaxes(range=[t_rel0, t_rel1])

    layout_kw: Dict[str, Any] = dict(
        title=f"{cfg.exp_code} — MFC Release Quicklook" + (" (flow + temperature)" if has_temp else ""),
        xaxis_title="Time",
        yaxis=dict(
            title=dict(text="Flow (MFC units)", font=dict(color=flow_color)),
            tickfont=dict(color=flow_color),
            side="left",
        ),
        template="plotly_white",
        height=520,
    )
    if has_temp:
        t_valid = mfc_df["T"].dropna()
        t_pad = (float(t_valid.max()) - float(t_valid.min())) * 0.05 if len(t_valid) else 1.0
        if not np.isfinite(t_pad) or t_pad <= 0:
            t_pad = 1.0
        layout_kw["yaxis2"] = dict(
            title=dict(text="Temperature (°C)", font=dict(color=temp_color)),
            tickfont=dict(color=temp_color),
            overlaying="y",
            side="right",
            showgrid=False,
            range=[float(t_valid.min()) - t_pad, float(t_valid.max()) + t_pad] if len(t_valid) else None,
        )
    fig.update_layout(**layout_kw)

    if y_range is not None:
        fig.update_layout(yaxis=dict(range=list(y_range)))

    return fig
# =========================================================
# Main title
# =========================================================
st.title("CAVE–PK CO₂ Analysis Dashboard")
st.caption("Upload experiment files, configure parameters, and run a repeatable analysis workflow.")


# =========================================================
# Sidebar
# =========================================================
st.sidebar.header("1) Upload files")

explora_file = st.sidebar.file_uploader(
    "Explora file (required)",
    type=["csv", "xlsx", "xlsm", "xls"]
)

stage_file = st.sidebar.file_uploader(
    "Experiment log / stage file (optional)",
    type=["xlsx", "xlsm", "xls"]
)

mfc_file = st.sidebar.file_uploader(
    "MFC file (optional)",
    type=["csv"]
)

def _upload_signature(file_obj) -> str:
    if file_obj is None:
        return ""
    try:
        b = file_obj.getvalue()
        h = hashlib.md5(b).hexdigest()
        return f"{getattr(file_obj, 'name', '')}|{len(b)}|{h}"
    except Exception:
        return f"{getattr(file_obj, 'name', '')}|na|na"


_sig = "|".join([_upload_signature(explora_file), _upload_signature(stage_file), _upload_signature(mfc_file)])
_prev_sig = st.session_state.get("__last_upload_signature", "")
if _sig and (_sig != _prev_sig):
    # When the user uploads new files, force all plot widgets to re-seed from built-in defaults
    # (or their saved snapshot, if present) so they don't need to "apply/reset" per page.
    st.session_state["__force_defaults_from_upload"] = True
    st.session_state["__last_upload_signature"] = _sig

st.sidebar.header("2) Analysis settings")

exp_code = st.sidebar.text_input("Experiment code", value="Experiment")
align_to = st.sidebar.text_input("Align to", value="10s")
min_sensors = st.sidebar.number_input("Min sensors", min_value=1, max_value=50, value=3, step=1)
coverage_factor = st.sidebar.number_input("Coverage factor", min_value=1.0, max_value=5.0, value=1.20, step=0.05)

apply_cave_exclusions_flag = st.sidebar.checkbox("Apply CAVE exclusions", value=True)
exclude_fixtures = st.sidebar.text_input("Exclude fixtures (comma-separated)", value="supply,extract")
exclude_z_levels = st.sidebar.text_input(
    "Exclude z levels from CAVE (comma-separated, from raw z in m)",
    value="z1",
)
exclude_sensors = st.sidebar.text_input("Exclude sensors (comma-separated)", value="24,25,26")

st.sidebar.header("3) Temperature stratification")

c1, c2 = st.sidebar.columns(2)
with c1:
    cave_z_low_min = st.number_input("CAVE low z min", value=0.0)
    cave_z_high_min = st.number_input("CAVE high z min", value=8.0)
with c2:
    cave_z_low_max = st.number_input("CAVE low z max", value=2.0)
    cave_z_high_max = st.number_input("CAVE high z max", value=10.0)

pk_low_z_levels = st.sidebar.text_input("PK low z levels (from raw z in m)", value="z1,z2")
pk_high_z_levels = st.sidebar.text_input("PK high z levels (from raw z in m)", value="z6,z7")

st.sidebar.header("4) Infiltration / MFC")
abs_ex_thresh = st.sidebar.number_input("Absolute excess threshold (ppm)", min_value=0.0, value=50.0, step=5.0)
baseline_fallback_minutes = st.sidebar.number_input("Fallback baseline minutes", min_value=1, value=10, step=1)
flow_on_th = st.sidebar.number_input("MFC flow-on threshold", min_value=0.0, value=0.2, step=0.1)

if "run_analysis" not in st.session_state:
    st.session_state.run_analysis = False


def _set_run_analysis_true():
    st.session_state.run_analysis = True


def _reset_run_analysis():
    st.session_state.run_analysis = False


st.sidebar.button("Run analysis", type="primary", on_click=_set_run_analysis_true)
st.sidebar.button("Reset", on_click=_reset_run_analysis)


# =========================================================
# Build config
# =========================================================
def split_str_list(s: str) -> Tuple[str, ...]:
    vals = [x.strip() for x in s.split(",") if x.strip()]
    return tuple(vals)

def split_int_list(s: str) -> Tuple[int, ...]:
    vals = []
    for x in s.split(","):
        x = x.strip()
        if x:
            try:
                vals.append(int(x))
            except ValueError:
                pass
    return tuple(vals)

cfg = AppConfig(
    exp_code=exp_code,
    align_to=align_to,
    min_sensors=int(min_sensors),
    coverage_factor=float(coverage_factor),
    apply_cave_exclusions=apply_cave_exclusions_flag,
    exclude_fixtures=split_str_list(exclude_fixtures),
    exclude_z_levels=split_str_list(exclude_z_levels),
    exclude_sensors=split_int_list(exclude_sensors),
    cave_z_low_min=float(cave_z_low_min),
    cave_z_low_max=float(cave_z_low_max),
    cave_z_high_min=float(cave_z_high_min),
    cave_z_high_max=float(cave_z_high_max),
    pk_low_z_levels=split_str_list(pk_low_z_levels),
    pk_high_z_levels=split_str_list(pk_high_z_levels),
    plot_pre_min=0,
    use_fixed_ylims=True,
    abs_ex_thresh=float(abs_ex_thresh),
    baseline_fallback_minutes=int(baseline_fallback_minutes),
    flow_on_th=float(flow_on_th),
    ylims=default_ylims(),
)


# =========================================================
# Main app
# =========================================================
if not explora_file:
    st.info("Please upload an Explora file to begin.")
    st.stop()

if not st.session_state.run_analysis:
    st.warning("Set parameters in the sidebar, then click 'Run analysis'.")
    st.stop()

try:
    with st.spinner("Loading files and running analysis..."):
        # -----------------------------
        # Load files
        # -----------------------------
        df = load_explora_any(explora_file.getvalue(), explora_file.name)

        stage_rows = []
        if stage_file is not None:
            try:
                stage_rows = load_stages_from_log(stage_file.getvalue(), stage_file.name)
            except Exception as e:
                st.warning(f"Could not read stage log: {e}")
                stage_rows = []

        stage_defs = prepare_stage_defs(stage_rows)

        mfc_df = None
        if mfc_file is not None:
            try:
                mfc_df = load_mfc_csv(mfc_file.getvalue(), mfc_file.name)
            except Exception as e:
                st.warning(f"Could not read MFC file: {e}")
                mfc_df = None

        # -----------------------------
        # Classify + exclusions
        # -----------------------------
        df = classify_regions(df)
        _z_usable = "z" in df.columns and pd.to_numeric(df["z"], errors="coerce").notna().any()
        if not _z_usable:
            st.warning(
                "Explora data has no usable **`z`** column (height in m). "
                "Vertical profiles, z-level exclusions, and PK stratification by z level will not work."
            )
        df = apply_cave_exclusions(df, cfg)

        df_cave = df[df["region"] == "CAVE"].copy()
        df_pk = df[df["region"] == "PK"].copy()

        # -----------------------------
        # Metrics
        # -----------------------------
        co2_cave = compute_co2_metrics(df_cave, cfg.align_to, cfg.min_sensors, cfg.coverage_factor)
        co2_pk = compute_co2_metrics(df_pk, cfg.align_to, cfg.min_sensors, cfg.coverage_factor)

        def cave_high(subdf):
            z = subdf["z_maybe"]
            return (z >= cfg.cave_z_high_min) & (z <= cfg.cave_z_high_max)

        def cave_low(subdf):
            z = subdf["z_maybe"]
            return (z >= cfg.cave_z_low_min) & (z <= cfg.cave_z_low_max)

        _pk_high_levels = _parse_z_level_labels(cfg.pk_high_z_levels) or frozenset({6.0, 7.0})
        _pk_low_levels = _parse_z_level_labels(cfg.pk_low_z_levels) or frozenset({1.0, 2.0})

        def pk_high(subdf):
            return _rows_in_z_levels(subdf, _pk_high_levels)

        def pk_low(subdf):
            return _rows_in_z_levels(subdf, _pk_low_levels)

        temp_cave = compute_temp_metrics(df_cave, cfg.align_to, cfg.min_sensors, cave_high, cave_low)
        temp_pk = compute_temp_metrics(df_pk, cfg.align_to, cfg.min_sensors, pk_high, pk_low)

        deltaT_pk_minus_cave = temp_pk["mean_T"].reindex(temp_cave["mean_T"].index) - temp_cave["mean_T"]

        t0 = df["time"].min()
        t1 = df["time"].max()
        plot_start = t0 if pd.notna(t0) else None
        plot_end = t1

        pk_zones_auto = sorted(df_pk["wall"].dropna().astype(str).str.strip().unique())
        cave_zone_co2 = zone_mean_timeseries(
            df_cave,
            zone_col="wall",
            zones=list(cfg.cave_walls_to_plot),
            value_col="co2",
            align_to=cfg.align_to,
            min_sensors=1,
        )
        pk_zone_co2 = zone_mean_timeseries(
            df_pk,
            zone_col="wall",
            zones=pk_zones_auto,
            value_col="co2",
            align_to=cfg.align_to,
            min_sensors=1,
        )

        # -----------------------------
        # Zone temperature (no infiltration analysis)
        # -----------------------------
        cave_zone_temp = zone_mean_timeseries(
            df_cave,
            zone_col="wall",
            zones=list(cfg.cave_walls_to_plot),
            value_col="temperature",
            align_to=cfg.align_to,
            min_sensors=1,
        )
        pk_zone_temp = zone_mean_timeseries(
            df_pk,
            zone_col="wall",
            zones=pk_zones_auto,
            value_col="temperature",
            align_to=cfg.align_to,
            min_sensors=1,
        )

        has_rh_data = humidity_has_data(df)
        rh_cave = rh_pk = None
        cave_zone_rh = pk_zone_rh = None
        if has_rh_data:
            rh_cave = compute_humidity_metrics(df_cave, cfg.align_to, cfg.min_sensors)
            rh_pk = compute_humidity_metrics(df_pk, cfg.align_to, cfg.min_sensors)
            cave_zone_rh = zone_mean_timeseries(
                df_cave,
                zone_col="wall",
                zones=list(cfg.cave_walls_to_plot),
                value_col="humidity",
                align_to=cfg.align_to,
                min_sensors=1,
            )
            pk_zone_rh = zone_mean_timeseries(
                df_pk,
                zone_col="wall",
                zones=pk_zones_auto,
                value_col="humidity",
                align_to=cfg.align_to,
                min_sensors=1,
            )

        # Simple release window based on stages (for MFC quicklook only)
        t_rel0 = t_rel1 = None
        rel_note = "no release stage"
        if stage_defs:
            t_rel0, t_rel1, rel_note = find_release_window(stage_defs)

        if (t_rel0 is None or t_rel1 is None) and pd.notna(t0) and pd.notna(t1):
            t_rel0, t_rel1 = t0, t1
            rel_note = "fallback: full available time"

        # -----------------------------
        # MFC summary
        # -----------------------------
        mfc_summary = None
        t_on = t_off = None

        if mfc_df is not None:
            mask_on = mfc_df["F"] > cfg.flow_on_th
            t_on = mfc_df.loc[mask_on, "t"].min() if mask_on.any() else None
            t_off = mfc_df.loc[mask_on, "t"].max() if mask_on.any() else None

            if mask_on.any():
                df_on = mfc_df.loc[mask_on].copy()
                dur_s = (t_off - t_on).total_seconds()
                dur_min = dur_s / 60.0

                f_mean = float(df_on["F"].mean())
                f_std = float(df_on["F"].std(ddof=1)) if len(df_on) > 1 else np.nan
                f_min = float(df_on["F"].min())
                f_max = float(df_on["F"].max())

                dt_min = df_on["t"].diff().dt.total_seconds().fillna(0) / 60.0
                total_l = float((df_on["F"] * dt_min).sum())
                f_cv = (f_std / f_mean) if (np.isfinite(f_std) and f_mean > 0) else np.nan

                mfc_summary = {
                    "mfc_start": t_on,
                    "mfc_end": t_off,
                    "mfc_duration_min": dur_min,
                    "flow_mean": f_mean,
                    "flow_std": f_std,
                    "flow_cv": f_cv,
                    "flow_min": f_min,
                    "flow_max": f_max,
                    "total_released_volume": total_l,
                }

        # -----------------------------
        # Figures (overall + zones + optional MFC)
        # -----------------------------
        lw_overall, _ = _line_marker_from_prefix("overall")
        leg_overall = _legend_fs_from_prefix("overall")
        fig_overall = plot_overall_metrics(
            co2_cave, co2_pk, temp_cave, temp_pk, deltaT_pk_minus_cave,
            stage_defs, cfg, plot_start, plot_end,
            line_width=lw_overall,
            legend_fontsize=leg_overall,
        )
        lw_zc, _ = _line_marker_from_prefix("zco2_cave")
        lw_zp, _ = _line_marker_from_prefix("zco2_pk")
        leg_zc = _legend_fs_from_prefix("zco2_cave")
        leg_zp = _legend_fs_from_prefix("zco2_pk")
        fig_zone = plot_zone_co2(
            cave_zone_co2, pk_zone_co2, stage_defs, cfg, plot_start, plot_end,
            cave_line_width=lw_zc * 1.25,
            pk_line_width=lw_zp * 1.0,
            cave_legend_fs=leg_zc,
            pk_legend_fs=leg_zp,
        )
        lw_tc, _ = _line_marker_from_prefix("zt_cave")
        lw_tp, _ = _line_marker_from_prefix("zt_pk")
        leg_tc = _legend_fs_from_prefix("zt_cave")
        leg_tp = _legend_fs_from_prefix("zt_pk")
        fig_zone_T = plot_zone_temp(
            cave_zone_temp, pk_zone_temp, stage_defs, cfg, plot_start, plot_end,
            cave_line_width=lw_tc * 1.25,
            pk_line_width=lw_tp * 1.0,
            cave_legend_fs=leg_tc,
            pk_legend_fs=leg_tp,
        )
        lw_mfc, _ = _line_marker_from_prefix("mfc")
        leg_mfc = _legend_fs_from_prefix("mfc")
        fig_mfc = (
            plot_mfc(mfc_df, t_on, t_off, t_rel0, t_rel1, cfg, line_width=lw_mfc, legend_fontsize=leg_mfc)
            if mfc_df is not None
            else None
        )

        # -----------------------------
        # Summary table (no infiltration-specific metrics)
        # -----------------------------
        summary = {
            "exp_code": cfg.exp_code,
            "explora_rows": len(df),
            "cave_rows": len(df_cave),
            "pk_rows": len(df_pk),
            "time_start": t0,
            "time_end": t1,
            "co2_cave_baseline": co2_cave["baseline"],
            "co2_pk_baseline": co2_pk["baseline"],
            "release_window_start": t_rel0,
            "release_window_end": t_rel1,
            "release_window_note": rel_note,
        }

        summary_df = build_summary_df(summary)

except Exception as e:
    st.error(f"Analysis failed: {type(e).__name__}: {e}")
    st.code(traceback.format_exc())
    st.stop()


# =========================================================
# Tabs
# =========================================================
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(
    [
        "Data Preview",
        "Overall Metrics",
        "Zone CO₂ & Temperature",
        "Sensor CO₂ & Temp",
        "Humidity",
        "Vertical Profiles (Decay)",
        "MFC (optional)",
        "Export",
    ]
)

with tab1:
    st.subheader("Input summary")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Sensors", f"{df['sensor_number'].nunique():,}")
    c3.metric("Walls", f"{df['wall'].nunique():,}")
    c4.metric("Regions", f"{df['region'].nunique():,}")

    st.write("**Time range**")
    st.write(f"{df['time'].min()} → {df['time'].max()}")

    st.write("**Columns**")
    st.write(list(df.columns))
    if humidity_has_data(df):
        src = df.attrs.get("humidity_source_col", "humidity")
        n_rh = int(df["humidity"].notna().sum())
        st.caption(f"Humidity data available — source column **{src}** ({n_rh:,} valid readings). See **Humidity** tab.")

    if stage_defs:
        st.write("**Detected stages**")
        stage_table = pd.DataFrame(
            [{"stage_name": n, "start": stt, "end": ett} for (n, stt, ett, _) in stage_defs]
        )
        st.dataframe(stage_table, use_container_width=True)

    st.write("**Explora preview**")
    st.dataframe(df.head(50), use_container_width=True)

with tab2:
    st.subheader("Overall metrics")

    c1, c2, c3 = st.columns(3)
    c1.metric("CAVE baseline", f"{co2_cave['baseline']:.2f}" if np.isfinite(co2_cave["baseline"]) else "NA")
    c2.metric("PK baseline", f"{co2_pk['baseline']:.2f}" if np.isfinite(co2_pk["baseline"]) else "NA")
    c3.metric("Coverage factor", f"{cfg.coverage_factor:.2f}")

    if go is None or make_subplots is None:
        st.warning("Plotly not installed; showing static matplotlib figure. To enable hover, run: pip install plotly")
        show_matplotlib_fig(fig_overall, stage_defs)
    else:
        with st.expander("Plot options — Overall metrics", expanded=False):
            _ensure_widget_defaults("overall", OVERALL_PAGE_DEFAULTS)
            render_save_reset_row("overall", OVERALL_PAGE_DEFAULTS)
            render_font_legend_widgets("overall")
            render_series_line_marker_widgets("overall")
            st.checkbox("Show subplot titles (panel headers)", key="overall__show_subplot_titles")
            st.checkbox("Use fixed y-limits (all panels)", key="overall__use_fixed_y")
            with st.expander("Y-axis limits (per panel)", expanded=False):
                for key, label in OVERALL_Y_KEYS:
                    c1, c2 = st.columns(2)
                    with c1:
                        st.number_input(f"{label} — min", key=f"overall__y_{key}_min")
                    with c2:
                        st.number_input(f"{label} — max", key=f"overall__y_{key}_max")
            st.markdown("**X-axis (time)**")
            render_x_mode_widgets("overall", t0, t1, stage_defs)

        x0, x1 = render_x_controls("overall", t0, t1, stage_defs)
        y_fb = default_ylims()
        y_merged = _collect_ylims_from_prefix("overall", OVERALL_Y_KEYS, y_fb)
        use_fy = bool(st.session_state.get("overall__use_fixed_y", True))
        show_panels = bool(st.session_state.get("overall__show_subplot_titles", False))
        lw_ov, ms_ov = _line_marker_from_prefix("overall")
        fig_overall_p = plot_overall_metrics_plotly(
            co2_cave,
            co2_pk,
            temp_cave,
            temp_pk,
            deltaT_pk_minus_cave,
            stage_defs,
            cfg,
            x0,
            x1,
            ylims_src=y_merged,
            use_fixed_y=use_fy,
            show_subplot_titles=show_panels,
            line_width=lw_ov,
            marker_size=ms_ov,
        )
        apply_plotly_style(fig_overall_p, _style_from_prefix("overall"))
        fig_overall_p.update_xaxes(tickformat="%H:%M", hoverformat="%Y-%m-%d %H:%M:%S")
        show_plotly_chart(fig_overall_p, stage_defs)

    st.write("---")
    st.write("**Overall metrics table (for export)**")
    overall_metrics = {
        "time_start": df["time"].min(),
        "time_end": df["time"].max(),
        "align_to": cfg.align_to,
        "min_sensors": cfg.min_sensors,
        "coverage_factor": cfg.coverage_factor,
        "co2_cave_baseline": co2_cave["baseline"],
        "co2_cave_threshold": co2_cave["threshold"],
        "co2_pk_baseline": co2_pk["baseline"],
        "co2_pk_threshold": co2_pk["threshold"],
        "co2_cave_mean_avg": float(co2_cave["mean"].mean(skipna=True)),
        "co2_pk_mean_avg": float(co2_pk["mean"].mean(skipna=True)),
        "co2_cave_coverage_avg_pct": float(co2_cave["coverage"].mean(skipna=True)),
        "co2_pk_coverage_avg_pct": float(co2_pk["coverage"].mean(skipna=True)),
        "temp_cave_mean_avg": float(temp_cave["mean_T"].mean(skipna=True)),
        "temp_pk_mean_avg": float(temp_pk["mean_T"].mean(skipna=True)),
        "temp_cave_deltaT_avg": float(temp_cave["deltaT"].mean(skipna=True)),
        "temp_pk_deltaT_avg": float(temp_pk["deltaT"].mean(skipna=True)),
        "temp_cave_r2_avg": float(temp_cave["r2_Tz"].mean(skipna=True)),
        "temp_pk_r2_avg": float(temp_pk["r2_Tz"].mean(skipna=True)),
        "temp_pk_minus_cave_avg": float(deltaT_pk_minus_cave.mean(skipna=True)),
    }

    overall_metrics_df = build_summary_df(overall_metrics)
    st.dataframe(overall_metrics_df, use_container_width=True)

    overall_csv_bytes = overall_metrics_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download overall metrics CSV",
        data=overall_csv_bytes,
        file_name=f"{cfg.exp_code}_overall_metrics_values.csv",
        mime="text/csv",
    )

with tab3:
    st.subheader("Zone mean CO₂ & Temperature")
    st.write("This section compares selected CAVE walls against PK wall-level response (CO₂ and temperature).")
    if go is None or make_subplots is None:
        st.warning("Plotly not installed; showing static matplotlib figure. To enable hover, run: pip install plotly")
        show_matplotlib_fig(fig_zone, stage_defs)
    else:
        dc_cave_co2 = zone_ts_page_defaults(cfg.ylims, "zone_cave_co2")
        dc_pk_co2 = zone_ts_page_defaults(cfg.ylims, "zone_pk_co2")
        dc_cave_t = {**ZONE_WIDGET_DEFAULTS, "y_min": 8.0, "y_max": 30.0, "use_fixed_y": False, "show_markers": False}
        dc_pk_t = {**ZONE_WIDGET_DEFAULTS, "y_min": 8.0, "y_max": 30.0, "use_fixed_y": False, "show_markers": False}

        with st.expander("Plot options — CAVE zone CO₂", expanded=False):
            _ensure_widget_defaults("zco2_cave", dc_cave_co2)
            render_save_reset_row("zco2_cave", dc_cave_co2)
            render_font_legend_widgets("zco2_cave")
            render_series_line_marker_widgets("zco2_cave")
            st.checkbox("Use fixed y-limits", key="zco2_cave__use_fixed_y")
            c1, c2 = st.columns(2)
            with c1:
                st.number_input("Y min", key="zco2_cave__y_min")
            with c2:
                st.number_input("Y max", key="zco2_cave__y_max")
            st.checkbox("Show markers", key="zco2_cave__show_markers")
            st.markdown("**X-axis (time)**")
            render_x_mode_widgets("zco2_cave", t0, t1, stage_defs)

        with st.expander("Plot options — PK zone CO₂", expanded=False):
            _ensure_widget_defaults("zco2_pk", dc_pk_co2)
            render_save_reset_row("zco2_pk", dc_pk_co2)
            render_font_legend_widgets("zco2_pk")
            render_series_line_marker_widgets("zco2_pk")
            st.checkbox("Use fixed y-limits", key="zco2_pk__use_fixed_y")
            c1, c2 = st.columns(2)
            with c1:
                st.number_input("Y min", key="zco2_pk__y_min")
            with c2:
                st.number_input("Y max", key="zco2_pk__y_max")
            st.checkbox("Show markers", key="zco2_pk__show_markers")
            st.markdown("**X-axis (time)**")
            render_x_mode_widgets("zco2_pk", t0, t1, stage_defs)

        xa_c, xa_c1 = render_x_controls("zco2_cave", t0, t1, stage_defs)
        xa_p, xa_p1 = render_x_controls("zco2_pk", t0, t1, stage_defs)
        uy_c = bool(st.session_state.get("zco2_cave__use_fixed_y", True))
        uy_p = bool(st.session_state.get("zco2_pk__use_fixed_y", True))
        ylo_c, yhi_c = _y_pair_from_prefix("zco2_cave", cfg.ylims["zone_cave_co2"][0], cfg.ylims["zone_cave_co2"][1])
        ylo_p, yhi_p = _y_pair_from_prefix("zco2_pk", cfg.ylims["zone_pk_co2"][0], cfg.ylims["zone_pk_co2"][1])
        y_rc = (ylo_c, yhi_c) if uy_c else None
        y_rp = (ylo_p, yhi_p) if uy_p else None
        mk_c = bool(st.session_state.get("zco2_cave__show_markers", False))
        mk_p = bool(st.session_state.get("zco2_pk__show_markers", False))
        lw_zcc, ms_zcc = _line_marker_from_prefix("zco2_cave")
        lw_zcp, ms_zcp = _line_marker_from_prefix("zco2_pk")

        fig_cave_zone_co2 = plot_zone_single_plotly(
            cave_zone_co2,
            title=f"{cfg.exp_code} — CAVE selected walls mean CO₂",
            y_title="CO₂ (ppm)",
            stage_defs=stage_defs,
            plot_start=xa_c,
            plot_end=xa_c1,
            y_range=y_rc,
            show_markers=mk_c,
            line_width=lw_zcc,
            marker_size=ms_zcc,
        )
        fig_pk_zone_co2 = plot_zone_single_plotly(
            pk_zone_co2,
            title=f"{cfg.exp_code} — PK zones mean CO₂ (by wall)",
            y_title="CO₂ (ppm)",
            stage_defs=stage_defs,
            plot_start=xa_p,
            plot_end=xa_p1,
            y_range=y_rp,
            show_markers=mk_p,
            line_width=lw_zcp,
            marker_size=ms_zcp,
        )
        apply_plotly_style(fig_cave_zone_co2, _style_from_prefix("zco2_cave"))
        apply_plotly_style(fig_pk_zone_co2, _style_from_prefix("zco2_pk"))
        show_plotly_chart(fig_cave_zone_co2, stage_defs, show_stage_legend=False)
        show_plotly_chart(fig_pk_zone_co2, stage_defs)

    st.write("**CAVE zone mean preview**")
    st.dataframe(cave_zone_co2.head(20), use_container_width=True)

    st.write("**PK zone mean preview**")
    st.dataframe(pk_zone_co2.head(20), use_container_width=True)

    st.write("---")
    st.write("### Zone temperature")

    if go is None or make_subplots is None:
        st.warning("Plotly not installed; showing static matplotlib figure for temperature. To enable hover, run: pip install plotly")
        show_matplotlib_fig(fig_zone_T, stage_defs)
    else:
        with st.expander("Plot options — CAVE zone temperature", expanded=False):
            _ensure_widget_defaults("zt_cave", dc_cave_t)
            render_save_reset_row("zt_cave", dc_cave_t)
            render_font_legend_widgets("zt_cave")
            render_series_line_marker_widgets("zt_cave")
            st.checkbox("Use fixed y-limits", key="zt_cave__use_fixed_y")
            c1, c2 = st.columns(2)
            with c1:
                st.number_input("Y min", key="zt_cave__y_min")
            with c2:
                st.number_input("Y max", key="zt_cave__y_max")
            st.checkbox("Show markers", key="zt_cave__show_markers")
            st.markdown("**X-axis (time)**")
            render_x_mode_widgets("zt_cave", t0, t1, stage_defs)

        with st.expander("Plot options — PK zone temperature", expanded=False):
            _ensure_widget_defaults("zt_pk", dc_pk_t)
            render_save_reset_row("zt_pk", dc_pk_t)
            render_font_legend_widgets("zt_pk")
            render_series_line_marker_widgets("zt_pk")
            st.checkbox("Use fixed y-limits", key="zt_pk__use_fixed_y")
            c1, c2 = st.columns(2)
            with c1:
                st.number_input("Y min", key="zt_pk__y_min")
            with c2:
                st.number_input("Y max", key="zt_pk__y_max")
            st.checkbox("Show markers", key="zt_pk__show_markers")
            st.markdown("**X-axis (time)**")
            render_x_mode_widgets("zt_pk", t0, t1, stage_defs)

        xtc0, xtc1 = render_x_controls("zt_cave", t0, t1, stage_defs)
        xtp0, xtp1 = render_x_controls("zt_pk", t0, t1, stage_defs)
        uy_tc = bool(st.session_state.get("zt_cave__use_fixed_y", False))
        uy_tp = bool(st.session_state.get("zt_pk__use_fixed_y", False))
        ytc_lo, ytc_hi = _y_pair_from_prefix("zt_cave", 8.0, 30.0)
        ytp_lo, ytp_hi = _y_pair_from_prefix("zt_pk", 8.0, 30.0)
        y_rtc = (ytc_lo, ytc_hi) if uy_tc else None
        y_rtp = (ytp_lo, ytp_hi) if uy_tp else None
        mktc = bool(st.session_state.get("zt_cave__show_markers", False))
        mktp = bool(st.session_state.get("zt_pk__show_markers", False))
        lw_ztc, ms_ztc = _line_marker_from_prefix("zt_cave")
        lw_ztp, ms_ztp = _line_marker_from_prefix("zt_pk")

        fig_cave_zone_temp = plot_zone_single_plotly(
            cave_zone_temp,
            title=f"{cfg.exp_code} — CAVE selected walls mean temperature",
            y_title="Temperature (°C)",
            stage_defs=stage_defs,
            plot_start=xtc0,
            plot_end=xtc1,
            y_range=y_rtc,
            show_markers=mktc,
            line_width=lw_ztc,
            marker_size=ms_ztc,
        )
        fig_pk_zone_temp = plot_zone_single_plotly(
            pk_zone_temp,
            title=f"{cfg.exp_code} — PK zones mean temperature (by wall)",
            y_title="Temperature (°C)",
            stage_defs=stage_defs,
            plot_start=xtp0,
            plot_end=xtp1,
            y_range=y_rtp,
            show_markers=mktp,
            line_width=lw_ztp,
            marker_size=ms_ztp,
        )
        apply_plotly_style(fig_cave_zone_temp, _style_from_prefix("zt_cave"))
        apply_plotly_style(fig_pk_zone_temp, _style_from_prefix("zt_pk"))
        show_plotly_chart(fig_cave_zone_temp, stage_defs, show_stage_legend=False)
        show_plotly_chart(fig_pk_zone_temp, stage_defs)

    st.write("**CAVE zone temperature preview**")
    st.dataframe(cave_zone_temp.head(20), use_container_width=True)

    st.write("**PK zone temperature preview**")
    st.dataframe(pk_zone_temp.head(20), use_container_width=True)

with tab4:
    st.subheader("Sensor CO₂ & temperature compare")
    st.write(
        "Plot **CO₂** and/or **temperature vs time** for individual sensors. Use **zones (walls)** to quickly "
        "add every sensor on selected walls, then refine the sensor list or compare multiple zones on one chart."
    )
    show_co2 = st.checkbox("Show CO₂", value=True, key="scmp_show_co2")
    show_temp = st.checkbox("Show temperature", value=True, key="scmp_show_temp")
    if not show_co2 and not show_temp:
        st.warning("Enable at least one of **Show CO₂** or **Show temperature**.")

    cave_cat = sensor_catalog(df_cave)
    pk_cat = sensor_catalog(df_pk)

    if len(cave_cat) == 0 and len(pk_cat) == 0:
        st.warning("No sensors available after filtering — check Explora upload and CAVE exclusions.")
    else:
        layout_mode = st.radio(
            "Chart layout",
            options=["All selected sensors on one chart", "One chart per zone (wall)"],
            horizontal=True,
            key="scmp_layout_mode",
        )
        one_per_zone = layout_mode.startswith("One chart")

        def _render_sensor_compare_block(
            region_label: str,
            df_region: pd.DataFrame,
            catalog: pd.DataFrame,
            plot_prefix: str,
            default_walls: Tuple[str, ...],
        ):
            if len(catalog) == 0:
                st.info(f"No {region_label} sensors in the current dataset.")
                return

            st.markdown(f"### {region_label}")
            if stage_defs:
                render_stage_legend_outside(stage_defs)
            walls_avail = sorted(catalog["wall"].unique().tolist())
            default_wall_pick = [w for w in default_walls if w in walls_avail]
            if not default_wall_pick and walls_avail:
                default_wall_pick = walls_avail[: min(2, len(walls_avail))]

            zc1, zc2, zc3 = st.columns([2, 1, 1])
            with zc1:
                picked_walls = st.multiselect(
                    f"{region_label} — zones (walls)",
                    options=walls_avail,
                    default=default_wall_pick,
                    key=f"{plot_prefix}__walls",
                    help="Select one or more walls; use the button to add all their sensors.",
                )
            with zc2:
                st.write("")
                st.write("")
                ms_key = f"{plot_prefix}__sensor_ms"
                if st.button(f"Add sensors from zones", key=f"{plot_prefix}__add_zone_sensors"):
                    zone_sns = sensors_in_walls(catalog, picked_walls)
                    cur = set(int(x) for x in st.session_state.get(ms_key, []))
                    st.session_state[ms_key] = sorted(cur | set(zone_sns))
            with zc3:
                st.write("")
                st.write("")
                if st.button(f"Clear selection", key=f"{plot_prefix}__clear_sensors"):
                    st.session_state[f"{plot_prefix}__sensor_ms"] = []

            sensor_options = catalog["sensor_number"].astype(int).tolist()
            opt_labels = {
                int(r["sensor_number"]): _sensor_series_label(
                    int(r["sensor_number"]),
                    str(r["wall"]),
                    float(r["z_median"]) if pd.notna(r["z_median"]) else np.nan,
                )
                for _, r in catalog.iterrows()
            }
            ms_key = f"{plot_prefix}__sensor_ms"
            if ms_key not in st.session_state:
                st.session_state[ms_key] = []
            picked_sensors = st.multiselect(
                f"{region_label} — sensors",
                options=sensor_options,
                format_func=lambda sid: opt_labels.get(int(sid), f"S{sid}"),
                key=ms_key,
            )
            picked_sensors = [int(s) for s in picked_sensors]

            if picked_walls:
                zone_sns = sensors_in_walls(catalog, picked_walls)
                st.caption(
                    f"Selected zones **{', '.join(picked_walls)}** → sensor numbers: "
                    f"{', '.join(str(s) for s in zone_sns) if zone_sns else '—'}"
                )

            if not picked_sensors:
                st.info(f"Select at least one {region_label} sensor (or use **Add sensors from zones**).")
                return

            if go is None or make_subplots is None:
                st.warning("Plotly not installed. Run: pip install plotly")
                return

            var_specs = []
            if show_co2:
                yk = "zone_cave_co2" if region_label == "CAVE" else "zone_pk_co2"
                var_specs.append(
                    (
                        "co2",
                        "CO₂",
                        "CO₂ (ppm)",
                        yk,
                        plot_prefix,
                        f"{plot_prefix}__dl_co2_csv",
                        "co2",
                    )
                )
            if show_temp:
                var_specs.append(
                    (
                        "temperature",
                        "Temperature",
                        "Temperature (°C)",
                        "temp_mean",
                        f"{plot_prefix}_t",
                        f"{plot_prefix}__dl_temp_csv",
                        "temperature",
                    )
                )

            for value_col, var_label, y_title, ykey, pfx, dl_key, file_tag in var_specs:
                st.markdown(f"#### {var_label}")
                if ykey in ("zone_cave_co2", "zone_pk_co2"):
                    dc = zone_ts_page_defaults(cfg.ylims, ykey)
                else:
                    lo, hi = cfg.ylims["temp_mean"]
                    dc = {**ZONE_WIDGET_DEFAULTS, "y_min": float(lo), "y_max": float(hi), "use_fixed_y": False, "show_markers": False}

                with st.expander(f"Plot options — {region_label} sensor {var_label}", expanded=False):
                    _ensure_widget_defaults(pfx, dc)
                    render_save_reset_row(pfx, dc)
                    render_font_legend_widgets(pfx)
                    render_series_line_marker_widgets(pfx)
                    st.checkbox("Use fixed y-limits", key=f"{pfx}__use_fixed_y")
                    cya, cyb = st.columns(2)
                    with cya:
                        st.number_input("Y min", key=f"{pfx}__y_min")
                    with cyb:
                        st.number_input("Y max", key=f"{pfx}__y_max")
                    st.checkbox("Show markers", key=f"{pfx}__show_markers")
                    st.markdown("**X-axis (time)**")
                    render_x_mode_widgets(pfx, t0, t1, stage_defs)

                xa0, xa1 = render_x_controls(pfx, t0, t1, stage_defs)
                uy = bool(st.session_state.get(f"{pfx}__use_fixed_y", dc.get("use_fixed_y", False)))
                ydef = cfg.ylims[ykey] if ykey in cfg.ylims else (dc["y_min"], dc["y_max"])
                ylo, yhi = _y_pair_from_prefix(pfx, ydef[0], ydef[1])
                y_r = (ylo, yhi) if uy else None
                mk = bool(st.session_state.get(f"{pfx}__show_markers", False))
                lw_s, ms_s = _line_marker_from_prefix(pfx)

                def _plot_sensor_ts(ts_df: pd.DataFrame, chart_title: str, _y_title=y_title, _pfx=pfx):
                    fig = plot_zone_single_plotly(
                        ts_df,
                        title=chart_title,
                        y_title=_y_title,
                        stage_defs=stage_defs,
                        plot_start=xa0,
                        plot_end=xa1,
                        y_range=y_r,
                        show_markers=mk,
                        line_width=lw_s,
                        marker_size=ms_s,
                        legend_in_plot=False,
                    )
                    _style = {**_style_from_prefix(_pfx), "show_legend": False}
                    apply_plotly_style(fig, _style)
                    show_plotly_chart(
                        fig,
                        stage_defs=None,
                        show_stage_legend=False,
                        external_series_legend=True,
                        series_legend_title="Sensors",
                    )

                if one_per_zone:
                    walls_to_plot = picked_walls if picked_walls else sorted(
                        catalog.loc[catalog["sensor_number"].isin(picked_sensors), "wall"].unique().tolist()
                    )
                    for wall in walls_to_plot:
                        sns_wall = [
                            int(s)
                            for s in picked_sensors
                            if int(s) in set(sensors_in_walls(catalog, [wall]))
                        ]
                        if not sns_wall:
                            continue
                        ts_w = sensor_value_timeseries(
                            df_region, sns_wall, cfg.align_to, value_col, catalog=catalog
                        )
                        if ts_w.empty:
                            st.caption(f"No {var_label} data for **{wall}** in the selected sensors.")
                            continue
                        _plot_sensor_ts(
                            ts_w,
                            f"{cfg.exp_code} — {region_label} — {wall} — {var_label} (individual sensors)",
                        )
                else:
                    ts_all = sensor_value_timeseries(
                        df_region, picked_sensors, cfg.align_to, value_col, catalog=catalog
                    )
                    if ts_all.empty:
                        st.caption(f"No {var_label} data for the selected sensors.")
                        continue
                    _plot_sensor_ts(
                        ts_all,
                        f"{cfg.exp_code} — {region_label} — selected sensors ({var_label} vs time)",
                    )
                    st.download_button(
                        label=f"Download {region_label} sensor {var_label} CSV",
                        data=ts_all.to_csv().encode("utf-8"),
                        file_name=f"{cfg.exp_code}_{region_label}_sensor_{file_tag}.csv",
                        mime="text/csv",
                        key=dl_key,
                    )

        if show_co2 or show_temp:
            _render_sensor_compare_block(
                "CAVE",
                df_cave,
                cave_cat,
                "scmp_cave",
                tuple(cfg.cave_walls_to_plot),
            )
            st.write("---")
            _render_sensor_compare_block(
                "PK",
                df_pk,
                pk_cat,
                "scmp_pk",
                tuple(pk_zones_auto) if pk_zones_auto else (),
            )

with tab5:
    st.subheader("Humidity analysis")
    st.write(
        "Relative humidity (**RH**) from the Explora upload when a humidity column is present. "
        "Use the sections below for region overview, wall-level zones, and individual sensors."
    )

    if not has_rh_data:
        st.warning(
            "No humidity column found in the Explora file. "
            "Expected headers such as **humidity**, **rh**, or **relative humidity**. "
            "Other tabs are unchanged."
        )
    else:
        rh_src = df.attrs.get("humidity_source_col", "humidity")
        st.caption(f"Using Explora column **{rh_src}** ({int(df['humidity'].notna().sum()):,} valid readings).")

        rh_overview_tab, rh_zone_tab, rh_sensor_tab = st.tabs(
            ["Overview", "Zone analysis", "Sensor level"]
        )

        with rh_overview_tab:
            st.markdown("### CAVE vs PK — regional humidity")

            rh_def = {**RH_PAGE_DEFAULTS, "use_fixed_y": True}
            plot_opts_label = (
                "Plot options — Humidity overview"
                if go is not None and make_subplots is not None
                else "Time window"
            )
            with st.expander(plot_opts_label, expanded=False):
                if go is not None and make_subplots is not None:
                    _ensure_widget_defaults("rh_ov", rh_def)
                    render_save_reset_row("rh_ov", rh_def)
                    render_font_legend_widgets("rh_ov")
                    render_series_line_marker_widgets("rh_ov")
                    st.checkbox("Use fixed y-limits", key="rh_ov__use_fixed_y")
                    with st.expander("Y-axis limits (per panel)", expanded=False):
                        for key, label in RH_OVERVIEW_Y_KEYS:
                            c1, c2 = st.columns(2)
                            with c1:
                                st.number_input(f"{label} — min", key=f"rh_ov__y_{key}_min")
                            with c2:
                                st.number_input(f"{label} — max", key=f"rh_ov__y_{key}_max")
                else:
                    _ensure_widget_defaults("rh_ov", rh_def)
                st.markdown("**X-axis (time)**")
                render_x_mode_widgets("rh_ov", t0, t1, stage_defs)

            x0, x1 = render_x_controls("rh_ov", t0, t1, stage_defs)
            cave_rh_period = series_mean_in_window(
                rh_cave["mean"] if rh_cave is not None else pd.Series(dtype=float), x0, x1
            )
            pk_rh_period = series_mean_in_window(
                rh_pk["mean"] if rh_pk is not None else pd.Series(dtype=float), x0, x1
            )
            cave_rh_std_period = series_mean_in_window(
                rh_cave["std"] if rh_cave is not None else pd.Series(dtype=float), x0, x1
            )
            pk_rh_std_period = series_mean_in_window(
                rh_pk["std"] if rh_pk is not None else pd.Series(dtype=float), x0, x1
            )
            if x0 is not None and x1 is not None:
                st.caption(
                    f"Period statistics for **{pd.Timestamp(x0):%Y-%m-%d %H:%M}** → "
                    f"**{pd.Timestamp(x1):%Y-%m-%d %H:%M}** (same window as the chart). "
                    f"Min sensors per bin: **{cfg.min_sensors}**."
                )

            c1, c2, c3, c4 = st.columns(4)
            c1.metric(
                "CAVE mean RH (period)",
                f"{cave_rh_period:.1f} %" if np.isfinite(cave_rh_period) else "NA",
            )
            c2.metric(
                "CAVE std RH (period)",
                f"{cave_rh_std_period:.2f} %" if np.isfinite(cave_rh_std_period) else "NA",
                help="Average of regional RH standard deviation within the selected time window.",
            )
            c3.metric(
                "PK mean RH (period)",
                f"{pk_rh_period:.1f} %" if np.isfinite(pk_rh_period) else "NA",
            )
            c4.metric(
                "PK std RH (period)",
                f"{pk_rh_std_period:.2f} %" if np.isfinite(pk_rh_std_period) else "NA",
                help="Average of regional RH standard deviation within the selected time window.",
            )

            if go is None or make_subplots is None:
                st.warning("Plotly not installed; humidity overview requires Plotly.")
            else:
                y_merged = _collect_ylims_from_prefix("rh_ov", RH_OVERVIEW_Y_KEYS, default_ylims())
                use_fy = bool(st.session_state.get("rh_ov__use_fixed_y", True))
                lw_rh, _ = _line_marker_from_prefix("rh_ov")
                fig_rh_ov = plot_humidity_overview_plotly(
                    rh_cave,
                    rh_pk,
                    stage_defs,
                    cfg,
                    x0,
                    x1,
                    ylims_src=y_merged,
                    use_fixed_y=use_fy,
                    line_width=lw_rh,
                )
                apply_plotly_style(fig_rh_ov, _style_from_prefix("rh_ov"))
                show_plotly_chart(fig_rh_ov, stage_defs)

            st.write("**Summary table (selected period)**")
            rh_summary = {
                "window_start": pd.Timestamp(x0) if x0 is not None else None,
                "window_end": pd.Timestamp(x1) if x1 is not None else None,
                "rh_cave_mean_period_pct": cave_rh_period,
                "rh_cave_std_period_pct": cave_rh_std_period,
                "rh_pk_mean_period_pct": pk_rh_period,
                "rh_pk_std_period_pct": pk_rh_std_period,
            }
            st.dataframe(build_summary_df(rh_summary), use_container_width=True)

        with rh_zone_tab:
            st.markdown("### Wall / zone mean relative humidity")
            if go is None:
                st.warning("Plotly not installed.")
            else:
                dc_cave_rh = zone_ts_page_defaults(cfg.ylims, "zone_cave_rh")
                dc_pk_rh = zone_ts_page_defaults(cfg.ylims, "zone_pk_rh")

                with st.expander("Plot options — CAVE zone RH", expanded=False):
                    _ensure_widget_defaults("rhz_cave", dc_cave_rh)
                    render_save_reset_row("rhz_cave", dc_cave_rh)
                    render_font_legend_widgets("rhz_cave")
                    render_series_line_marker_widgets("rhz_cave")
                    st.checkbox("Use fixed y-limits", key="rhz_cave__use_fixed_y")
                    c1, c2 = st.columns(2)
                    with c1:
                        st.number_input("Y min", key="rhz_cave__y_min")
                    with c2:
                        st.number_input("Y max", key="rhz_cave__y_max")
                    st.checkbox("Show markers", key="rhz_cave__show_markers")
                    render_x_mode_widgets("rhz_cave", t0, t1, stage_defs)

                with st.expander("Plot options — PK zone RH", expanded=False):
                    _ensure_widget_defaults("rhz_pk", dc_pk_rh)
                    render_save_reset_row("rhz_pk", dc_pk_rh)
                    render_font_legend_widgets("rhz_pk")
                    render_series_line_marker_widgets("rhz_pk")
                    st.checkbox("Use fixed y-limits", key="rhz_pk__use_fixed_y")
                    c1, c2 = st.columns(2)
                    with c1:
                        st.number_input("Y min", key="rhz_pk__y_min")
                    with c2:
                        st.number_input("Y max", key="rhz_pk__y_max")
                    st.checkbox("Show markers", key="rhz_pk__show_markers")
                    render_x_mode_widgets("rhz_pk", t0, t1, stage_defs)

                xa_c0, xa_c1 = render_x_controls("rhz_cave", t0, t1, stage_defs)
                xa_p0, xa_p1 = render_x_controls("rhz_pk", t0, t1, stage_defs)
                lw_c, ms_c = _line_marker_from_prefix("rhz_cave")
                lw_p, ms_p = _line_marker_from_prefix("rhz_pk")

                def _zone_rh_plot(zone_df, title, pfx, xa0, xa1, ykey):
                    uy = bool(st.session_state.get(f"{pfx}__use_fixed_y", True))
                    ylo, yhi = _y_pair_from_prefix(pfx, cfg.ylims[ykey][0], cfg.ylims[ykey][1])
                    y_r = (ylo, yhi) if uy else None
                    mk = bool(st.session_state.get(f"{pfx}__show_markers", False))
                    lw, ms = _line_marker_from_prefix(pfx)
                    fig = plot_zone_single_plotly(
                        zone_df,
                        title=title,
                        y_title="Relative humidity (%)",
                        stage_defs=stage_defs,
                        plot_start=xa0,
                        plot_end=xa1,
                        y_range=y_r,
                        show_markers=mk,
                        line_width=lw,
                        marker_size=ms,
                        legend_in_plot=False,
                    )
                    apply_plotly_style(fig, {**_style_from_prefix(pfx), "show_legend": False})
                    show_plotly_chart(
                        fig,
                        stage_defs,
                        show_stage_legend=False,
                        external_series_legend=True,
                        series_legend_title="Zones / walls",
                    )

                if stage_defs:
                    render_stage_legend_outside(stage_defs)
                _zone_rh_plot(
                    cave_zone_rh,
                    f"{cfg.exp_code} — CAVE zone RH",
                    "rhz_cave",
                    xa_c0,
                    xa_c1,
                    "zone_cave_rh",
                )
                _zone_rh_plot(
                    pk_zone_rh,
                    f"{cfg.exp_code} — PK zone RH",
                    "rhz_pk",
                    xa_p0,
                    xa_p1,
                    "zone_pk_rh",
                )

            st.write("**Zone RH preview**")
            c1, c2 = st.columns(2)
            with c1:
                st.caption("CAVE")
                st.dataframe(cave_zone_rh.head(20), use_container_width=True)
            with c2:
                st.caption("PK")
                st.dataframe(pk_zone_rh.head(20), use_container_width=True)

        with rh_sensor_tab:
            st.markdown("### Sensor-level relative humidity")

            def _render_rh_sensor_block(region_label, df_region, catalog, plot_prefix, default_walls):
                if len(catalog) == 0:
                    st.info(f"No {region_label} sensors in the current dataset.")
                    return
                st.markdown(f"#### {region_label}")
                if stage_defs:
                    render_stage_legend_outside(stage_defs)
                walls_avail = sorted(catalog["wall"].unique().tolist())
                default_wall_pick = [w for w in default_walls if w in walls_avail]
                if not default_wall_pick and walls_avail:
                    default_wall_pick = walls_avail[: min(2, len(walls_avail))]

                zc1, zc2, zc3 = st.columns([2, 1, 1])
                with zc1:
                    picked_walls = st.multiselect(
                        f"{region_label} — zones (walls)",
                        options=walls_avail,
                        default=default_wall_pick,
                        key=f"{plot_prefix}__walls",
                    )
                with zc2:
                    st.write("")
                    st.write("")
                    if st.button("Add sensors from zones", key=f"{plot_prefix}__add_zone_sensors"):
                        zone_sns = sensors_in_walls(catalog, picked_walls)
                        cur = set(int(x) for x in st.session_state.get(f"{plot_prefix}__sensor_ms", []))
                        st.session_state[f"{plot_prefix}__sensor_ms"] = sorted(cur | set(zone_sns))
                with zc3:
                    st.write("")
                    st.write("")
                    if st.button("Clear selection", key=f"{plot_prefix}__clear_sensors"):
                        st.session_state[f"{plot_prefix}__sensor_ms"] = []

                sensor_options = catalog["sensor_number"].astype(int).tolist()
                opt_labels = {
                    int(r["sensor_number"]): _sensor_series_label(
                        int(r["sensor_number"]),
                        str(r["wall"]),
                        float(r["z_median"]) if pd.notna(r["z_median"]) else np.nan,
                    )
                    for _, r in catalog.iterrows()
                }
                ms_key = f"{plot_prefix}__sensor_ms"
                if ms_key not in st.session_state:
                    st.session_state[ms_key] = []
                picked_sensors = [
                    int(s)
                    for s in st.multiselect(
                        f"{region_label} — sensors",
                        options=sensor_options,
                        format_func=lambda sid: opt_labels.get(int(sid), f"S{sid}"),
                        key=ms_key,
                    )
                ]
                if not picked_sensors:
                    st.info(f"Select at least one {region_label} sensor.")
                    return

                pfx = f"{plot_prefix}_rh"
                dc = {**ZONE_WIDGET_DEFAULTS, "y_min": 0.0, "y_max": 100.0, "use_fixed_y": False, "show_markers": False}
                with st.expander(f"Plot options — {region_label} sensor RH", expanded=False):
                    _ensure_widget_defaults(pfx, dc)
                    render_save_reset_row(pfx, dc)
                    render_font_legend_widgets(pfx)
                    render_series_line_marker_widgets(pfx)
                    st.checkbox("Use fixed y-limits", key=f"{pfx}__use_fixed_y")
                    cya, cyb = st.columns(2)
                    with cya:
                        st.number_input("Y min", key=f"{pfx}__y_min")
                    with cyb:
                        st.number_input("Y max", key=f"{pfx}__y_max")
                    st.checkbox("Show markers", key=f"{pfx}__show_markers")
                    render_x_mode_widgets(pfx, t0, t1, stage_defs)

                xa0, xa1 = render_x_controls(pfx, t0, t1, stage_defs)
                uy = bool(st.session_state.get(f"{pfx}__use_fixed_y", False))
                ylo, yhi = _y_pair_from_prefix(pfx, 0.0, 100.0)
                y_r = (ylo, yhi) if uy else None
                mk = bool(st.session_state.get(f"{pfx}__show_markers", False))
                lw_s, ms_s = _line_marker_from_prefix(pfx)

                layout_mode = st.session_state.get("rh_scmp_layout_mode", "All selected sensors on one chart")
                one_per_zone = str(layout_mode).startswith("One chart")

                if one_per_zone:
                    walls_to_plot = picked_walls if picked_walls else sorted(
                        catalog.loc[catalog["sensor_number"].isin(picked_sensors), "wall"].unique().tolist()
                    )
                    for wall in walls_to_plot:
                        sns_wall = [
                            int(s)
                            for s in picked_sensors
                            if int(s) in set(sensors_in_walls(catalog, [wall]))
                        ]
                        if not sns_wall:
                            continue
                        ts_w = sensor_value_timeseries(
                            df_region, sns_wall, cfg.align_to, "humidity", catalog=catalog
                        )
                        if ts_w.empty:
                            continue
                        fig = plot_zone_single_plotly(
                            ts_w,
                            title=f"{cfg.exp_code} — {region_label} — {wall} — RH",
                            y_title="Relative humidity (%)",
                            stage_defs=stage_defs,
                            plot_start=xa0,
                            plot_end=xa1,
                            y_range=y_r,
                            show_markers=mk,
                            line_width=lw_s,
                            marker_size=ms_s,
                            legend_in_plot=False,
                        )
                        apply_plotly_style(fig, {**_style_from_prefix(pfx), "show_legend": False})
                        show_plotly_chart(
                            fig, None, show_stage_legend=False,
                            external_series_legend=True, series_legend_title="Sensors",
                        )
                else:
                    ts_all = sensor_value_timeseries(
                        df_region, picked_sensors, cfg.align_to, "humidity", catalog=catalog
                    )
                    if ts_all.empty:
                        st.caption("No humidity data for the selected sensors.")
                        return
                    fig = plot_zone_single_plotly(
                        ts_all,
                        title=f"{cfg.exp_code} — {region_label} — selected sensors (RH)",
                        y_title="Relative humidity (%)",
                        stage_defs=stage_defs,
                        plot_start=xa0,
                        plot_end=xa1,
                        y_range=y_r,
                        show_markers=mk,
                        line_width=lw_s,
                        marker_size=ms_s,
                        legend_in_plot=False,
                    )
                    apply_plotly_style(fig, {**_style_from_prefix(pfx), "show_legend": False})
                    show_plotly_chart(
                        fig, None, show_stage_legend=False,
                        external_series_legend=True, series_legend_title="Sensors",
                    )
                    st.download_button(
                        label=f"Download {region_label} sensor RH CSV",
                        data=ts_all.to_csv().encode("utf-8"),
                        file_name=f"{cfg.exp_code}_{region_label}_sensor_humidity.csv",
                        mime="text/csv",
                        key=f"{plot_prefix}__dl_rh_csv",
                    )

            st.radio(
                "Chart layout",
                options=["All selected sensors on one chart", "One chart per zone (wall)"],
                horizontal=True,
                key="rh_scmp_layout_mode",
            )
            cave_rh_cat = sensor_catalog(df_cave[df_cave["humidity"].notna()] if "humidity" in df_cave.columns else df_cave)
            pk_rh_cat = sensor_catalog(df_pk[df_pk["humidity"].notna()] if "humidity" in df_pk.columns else df_pk)
            _render_rh_sensor_block("CAVE", df_cave, cave_rh_cat, "rhscmp_cave", tuple(cfg.cave_walls_to_plot))
            st.write("---")
            _render_rh_sensor_block("PK", df_pk, pk_rh_cat, "rhscmp_pk", tuple(pk_zones_auto) if pk_zones_auto else ())

with tab6:
    st.subheader("Vertical Profiles (Decay)")
    st.write(
        "Select a **stage** from the experiment log. That stage’s start–end time is divided into **5 equal sub-windows**; "
        "each coloured line on the plots is the **vertical mean** (by height level) of all Explora readings whose timestamp "
        "falls inside that sub-window."
    )
    st.markdown(
        "**Legend: W1–W5** — window index within the selected stage (not a sensor ID). "
        "**W1** = earliest fifth of the stage, **W5** = latest fifth. "
        "For stage duration *T*, each window spans *T*/5; labels are assigned in chronological order."
    )
    st.caption(
        "Height bins use only raw **`z`** (m): [0, 1] → z1, (1, 2] → z2, … (z=1.0 → z1, z=2.0 → z2). "
        "The Explora **`z_slice`** column is not read anywhere in this dashboard."
    )
    st.markdown(
        format_z_level_sensor_map(df_cave, "CAVE")
        + "\n\n"
        + format_z_level_sensor_map(df_pk, "PK")
        + "\n\n"
        "_Lists are from the loaded Explora file (after region split and CAVE exclusions); "
        "each vertical mean at a height averages all readings from those sensors in that z bin._"
    )

    if not stage_defs:
        st.warning("No stages detected. Please upload a stage file to enable stage selection.")
    else:
        stage_names = [str(n) for (n, _, _, _) in stage_defs]
        default_stage = find_stage_by_keyword(stage_defs, "decay")
        default_idx = 0
        if default_stage is not None:
            try:
                default_idx = stage_names.index(str(default_stage[0]))
            except Exception:
                default_idx = 0

        chosen = st.selectbox(
            "Stage to analyze (profiles are computed only within this stage)",
            options=stage_names,
            index=default_idx,
        )

        chosen_stage = next(((n, stt, ett, col) for (n, stt, ett, col) in stage_defs if str(n) == str(chosen)), None)
        if chosen_stage is None:
            st.warning("Selected stage not found.")
        else:
            stage_name, stage_start, stage_end, _ = chosen_stage
            st.write(f"**Selected stage**: {stage_name}")
            st.write(f"**Time range**: {pd.Timestamp(stage_start)} → {pd.Timestamp(stage_end)}")

            windows = split_time_range(stage_start, stage_end, 5)
            if not windows:
                st.warning("Stage time range is invalid or too short.")
            else:
                labels = []
                for i, (a, b) in enumerate(windows, start=1):
                    labels.append((f"W{i}", a, b))

                _win_tbl = pd.DataFrame(
                    [
                        {
                            "Legend": lab,
                            "Window": f"{i} of 5",
                            "Start (inclusive)": pd.Timestamp(a),
                            "End (inclusive)": pd.Timestamp(b),
                        }
                        for i, (lab, a, b) in enumerate(labels, start=1)
                    ]
                )
                st.write("**Time windows for W1–W5** (each line on the plots uses only data in that interval)")
                st.dataframe(_win_tbl, use_container_width=True, hide_index=True)

                pk_co2_profiles = [(lab, vertical_profile_means(df_pk, a, b, "co2")) for (lab, a, b) in labels]
                cave_co2_profiles = [(lab, vertical_profile_means(df_cave, a, b, "co2")) for (lab, a, b) in labels]
                pk_T_profiles = [(lab, vertical_profile_means(df_pk, a, b, "temperature")) for (lab, a, b) in labels]
                cave_T_profiles = [(lab, vertical_profile_means(df_cave, a, b, "temperature")) for (lab, a, b) in labels]

                _co2_parts = [dfp[["mean"]] for _, dfp in (cave_co2_profiles + pk_co2_profiles) if dfp is not None and len(dfp)]
                co2_all = (
                    pd.concat(_co2_parts, axis=0, ignore_index=True) if _co2_parts else pd.DataFrame({"mean": []})
                )
                _t_parts = [dfp[["mean"]] for _, dfp in (cave_T_profiles + pk_T_profiles) if dfp is not None and len(dfp)]
                t_all = pd.concat(_t_parts, axis=0, ignore_index=True) if _t_parts else pd.DataFrame({"mean": []})

                # Built-in suggested defaults for profile panels (user can override and save per-panel).
                # These are used as the default manual x-limits and fixed z extents.
                co2_min_default = 350.0
                co2_max_default = 1190.0
                t_min_default = 10.5
                t_max_default = 32.0
                z_min_default = 0.5
                z_max_default = 10.5

                def _prof_nonempty(profile_pairs):
                    return sum(1 for _, dfp in profile_pairs if dfp is not None and len(dfp))

                n_prof_traces = (
                    _prof_nonempty(cave_co2_profiles)
                    + _prof_nonempty(cave_T_profiles)
                    + _prof_nonempty(pk_co2_profiles)
                    + _prof_nonempty(pk_T_profiles)
                )
                if n_prof_traces == 0:
                    st.warning(
                        "This stage produced **no drawable profile lines** (all windows empty). "
                        "Usually the stage times do not overlap Explora data, or **`z`** (m) is missing/invalid "
                        "so height levels cannot be assigned. See **Diagnostics** below."
                    )
                with st.expander("Diagnostics — vertical profile data", expanded=(n_prof_traces == 0)):
                    st.write(f"**Selected stage window**: {pd.Timestamp(stage_start)} → {pd.Timestamp(stage_end)}")
                    if len(df_cave) == 0 and len(df_pk) == 0:
                        st.write("No CAVE or PK rows in Explora after filtering — check upload and `region`/`wall` logic.")
                    else:
                        tcomb = pd.concat([df_cave["time"], df_pk["time"]], ignore_index=True)
                        tmin_d, tmax_d = pd.Timestamp(tcomb.min()), pd.Timestamp(tcomb.max())
                        st.write(f"**Explora time span (CAVE+PK)**: {tmin_d} → {tmax_d}")
                        ov = (pd.Timestamp(stage_end) >= tmin_d) and (pd.Timestamp(stage_start) <= tmax_d)
                        st.write(f"**Stage overlaps Explora times**: {'yes' if ov else 'no — profiles will be empty'}")
                    for label, dreg in ("CAVE", df_cave), ("PK", df_pk):
                        dw = dreg[(dreg["time"] >= pd.Timestamp(stage_start)) & (dreg["time"] <= pd.Timestamp(stage_end))]
                        st.write(
                            f"**{label}** rows in stage window: **{len(dw)}** "
                            f"(CO₂+temp not NaN: **{dw.dropna(subset=['co2', 'temperature']).shape[0]}**)"
                        )
                        if len(dw) > 0:
                            z_samp = (
                                list(pd.to_numeric(dw["z"], errors="coerce").dropna().head(5))
                                if "z" in dw.columns
                                else []
                            )
                            z_lvl_samp = (
                                list(_z_coord_to_level(pd.to_numeric(dw["z"], errors="coerce")).dropna().unique()[:5])
                                if "z" in dw.columns and len(z_samp)
                                else []
                            )
                            st.caption(
                                f"Has usable `z` (m): **{'z' in dw.columns and len(z_samp) > 0}** "
                                f"(sample z: {z_samp or '—'}; mapped levels: {z_lvl_samp or '—'})."
                            )
                    st.caption(
                        "If plots look empty but traces exist, check each panel’s **Manual x-axis limits** — "
                        "wrong min/max can hide all lines."
                    )

                prof_dc_cc = {
                    **PROF_WIDGET_DEFAULTS,
                    "x_use_manual": True,
                    "x_vmin": co2_min_default,
                    "x_vmax": co2_max_default,
                    "use_fixed_y_z": True,
                    "y_z_min": z_min_default,
                    "y_z_max": z_max_default,
                    "show_legend": False,
                }
                prof_dc_ct = {
                    **PROF_WIDGET_DEFAULTS,
                    "x_use_manual": True,
                    "x_vmin": t_min_default,
                    "x_vmax": t_max_default,
                    "use_fixed_y_z": True,
                    "y_z_min": z_min_default,
                    "y_z_max": z_max_default,
                    "show_legend": False,
                }
                prof_dc_pc = {
                    **PROF_WIDGET_DEFAULTS,
                    "x_use_manual": True,
                    "x_vmin": co2_min_default,
                    "x_vmax": co2_max_default,
                    "use_fixed_y_z": True,
                    "y_z_min": z_min_default,
                    "y_z_max": z_max_default,
                    "show_legend": False,
                }
                prof_dc_pt = {
                    **PROF_WIDGET_DEFAULTS,
                    "x_use_manual": True,
                    "x_vmin": t_min_default,
                    "x_vmax": t_max_default,
                    "use_fixed_y_z": True,
                    "y_z_min": z_min_default,
                    "y_z_max": z_max_default,
                    "show_legend": True,
                }

                st.write("**Plot options (vertical profiles)**")
                with st.expander("CAVE — CO₂", expanded=False):
                    render_prof_panel_options("prof_cc", prof_dc_cc)
                with st.expander("CAVE — Temperature", expanded=False):
                    render_prof_panel_options("prof_ct", prof_dc_ct)
                with st.expander("PK — CO₂", expanded=False):
                    render_prof_panel_options("prof_pc", prof_dc_pc)
                with st.expander("PK — Temperature", expanded=False):
                    render_prof_panel_options("prof_pt", prof_dc_pt)

                co2_xrange_c = _prof_x_range("prof_cc")
                co2_xrange_p = _prof_x_range("prof_pc")
                t_xrange_c = _prof_x_range("prof_ct")
                t_xrange_p = _prof_x_range("prof_pt")
                yz_cc = _prof_yz_range("prof_cc")
                yz_pc = _prof_yz_range("prof_pc")
                yz_ct = _prof_yz_range("prof_ct")
                yz_pt = _prof_yz_range("prof_pt")

                lw_cc, ms_cc = _line_marker_from_prefix("prof_cc")
                lw_ct, ms_ct = _line_marker_from_prefix("prof_ct")
                lw_pc, ms_pc = _line_marker_from_prefix("prof_pc")
                lw_pt, ms_pt = _line_marker_from_prefix("prof_pt")
                leg_cc = _legend_fs_from_prefix("prof_cc")
                leg_ct = _legend_fs_from_prefix("prof_ct")
                leg_pc = _legend_fs_from_prefix("prof_pc")
                leg_pt = _legend_fs_from_prefix("prof_pt")

                c1, c2, c3, c4 = st.columns(4)

                if go is None or make_subplots is None:
                    with c1:
                        st.write("**CAVE — CO₂**")
                        show_matplotlib_fig(
                            plot_vertical_profiles_matplotlib(
                                cave_co2_profiles,
                                _vertical_profile_title("CAVE — CO₂"),
                                "Mean CO₂",
                                x_range=co2_xrange_c,
                                y_range=yz_cc,
                                show_legend=bool(st.session_state.get("prof_cc__show_legend", True)),
                                line_width=lw_cc,
                                marker_size=ms_cc,
                                legend_fontsize=leg_cc,
                            )
                        )
                    with c2:
                        st.write("**CAVE — Temperature**")
                        show_matplotlib_fig(
                            plot_vertical_profiles_matplotlib(
                                cave_T_profiles,
                                _vertical_profile_title("CAVE — Temperature"),
                                "Mean Temperature (°C)",
                                x_range=t_xrange_c,
                                y_range=yz_ct,
                                show_legend=bool(st.session_state.get("prof_ct__show_legend", True)),
                                line_width=lw_ct,
                                marker_size=ms_ct,
                                legend_fontsize=leg_ct,
                            )
                        )
                    with c3:
                        st.write("**PK — CO₂**")
                        show_matplotlib_fig(
                            plot_vertical_profiles_matplotlib(
                                pk_co2_profiles,
                                _vertical_profile_title("PK — CO₂"),
                                "Mean CO₂",
                                x_range=co2_xrange_p,
                                y_range=yz_pc,
                                show_legend=bool(st.session_state.get("prof_pc__show_legend", True)),
                                line_width=lw_pc,
                                marker_size=ms_pc,
                                legend_fontsize=leg_pc,
                            )
                        )
                    with c4:
                        st.write("**PK — Temperature**")
                        show_matplotlib_fig(
                            plot_vertical_profiles_matplotlib(
                                pk_T_profiles,
                                _vertical_profile_title("PK — Temperature"),
                                "Mean Temperature (°C)",
                                x_range=t_xrange_p,
                                y_range=yz_pt,
                                show_legend=bool(st.session_state.get("prof_pt__show_legend", True)),
                                line_width=lw_pt,
                                marker_size=ms_pt,
                                legend_fontsize=leg_pt,
                            )
                        )
                else:
                    fig_p_cc = plot_vertical_profiles_plotly(
                        cave_co2_profiles,
                        _vertical_profile_title("CAVE — CO₂"),
                        "Mean CO₂",
                        x_range=co2_xrange_c,
                        y_range=yz_cc,
                        line_width=lw_cc,
                        marker_size=ms_cc,
                    )
                    fig_p_ct = plot_vertical_profiles_plotly(
                        cave_T_profiles,
                        _vertical_profile_title("CAVE — Temperature"),
                        "Mean Temperature (°C)",
                        x_range=t_xrange_c,
                        y_range=yz_ct,
                        line_width=lw_ct,
                        marker_size=ms_ct,
                    )
                    fig_p_pc = plot_vertical_profiles_plotly(
                        pk_co2_profiles,
                        _vertical_profile_title("PK — CO₂"),
                        "Mean CO₂",
                        x_range=co2_xrange_p,
                        y_range=yz_pc,
                        line_width=lw_pc,
                        marker_size=ms_pc,
                    )
                    fig_p_pt = plot_vertical_profiles_plotly(
                        pk_T_profiles,
                        _vertical_profile_title("PK — Temperature"),
                        "Mean Temperature (°C)",
                        x_range=t_xrange_p,
                        y_range=yz_pt,
                        line_width=lw_pt,
                        marker_size=ms_pt,
                    )
                    apply_plotly_style(fig_p_cc, _style_from_prefix("prof_cc"))
                    apply_plotly_style(fig_p_ct, _style_from_prefix("prof_ct"))
                    apply_plotly_style(fig_p_pc, _style_from_prefix("prof_pc"))
                    apply_plotly_style(fig_p_pt, _style_from_prefix("prof_pt"))
                    with c1:
                        st.write("**CAVE — CO₂**")
                        show_plotly_chart(fig_p_cc)
                    with c2:
                        st.write("**CAVE — Temperature**")
                        show_plotly_chart(fig_p_ct)
                    with c3:
                        st.write("**PK — CO₂**")
                        show_plotly_chart(fig_p_pc)
                    with c4:
                        st.write("**PK — Temperature**")
                        show_plotly_chart(fig_p_pt)

                st.write("---")
                st.write("**Download profile data**")
                rows = []
                for win_idx, (win_label, a, b) in enumerate(labels, start=1):
                    for region, var, plist in [
                        ("CAVE", "co2", cave_co2_profiles),
                        ("CAVE", "temperature", cave_T_profiles),
                        ("PK", "co2", pk_co2_profiles),
                        ("PK", "temperature", pk_T_profiles),
                    ]:
                        dfp = dict(plist).get(win_label, pd.DataFrame())
                        if dfp is None or len(dfp) == 0:
                            continue
                        tmp = dfp.copy()
                        tmp["stage_name"] = str(stage_name)
                        tmp["stage_start"] = pd.Timestamp(stage_start)
                        tmp["stage_end"] = pd.Timestamp(stage_end)
                        tmp["window_idx"] = win_idx
                        tmp["window_label"] = win_label
                        tmp["window_start"] = pd.Timestamp(a)
                        tmp["window_end"] = pd.Timestamp(b)
                        tmp["region"] = region
                        tmp["variable"] = var
                        rows.append(tmp[["stage_name", "stage_start", "stage_end", "window_idx", "window_label", "window_start", "window_end", "region", "variable", "z_level", "z_label", "mean"]])

                profiles_df = pd.concat(rows, axis=0, ignore_index=True) if rows else pd.DataFrame(
                    columns=["stage_name", "stage_start", "stage_end", "window_idx", "window_label", "window_start", "window_end", "region", "variable", "z_level", "z_label", "mean"]
                )

                st.dataframe(profiles_df.head(50), use_container_width=True)
                profiles_csv = profiles_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Download vertical profile data (CSV)",
                    data=profiles_csv,
                    file_name=f"{cfg.exp_code}_{str(stage_name).replace(' ', '_')}_vertical_profiles.csv",
                    mime="text/csv",
                )

with tab7:
    st.subheader("MFC (optional)")
    st.write(
        "If an MFC file is uploaded, this tab shows a quicklook of **flow rate** (left axis) and, when a "
        "**temperature** column is present in the CSV, **temperature** (right axis) on the same chart."
    )

    if fig_mfc is not None:
        st.write("**MFC quicklook**")
        if mfc_has_temperature(mfc_df):
            src = mfc_df.attrs.get("temp_source_col", "T")
            n_t = int(mfc_df["T"].notna().sum())
            st.caption(f"Temperature column **{src}** detected ({n_t:,} valid points) — dual y-axis plot enabled.")
        else:
            st.caption("No usable temperature column found. Flow only.")
            with st.expander("Why is temperature missing?", expanded=True):
                st.write("**MFC file columns:**", list(mfc_df.columns))
                guess = _detect_mfc_temperature_column(mfc_df.columns)
                if guess:
                    preview = _parse_mfc_numeric_series(mfc_df[guess])
                    st.warning(
                        f"Column **{guess}** looks like temperature but has "
                        f"**{int(preview.notna().sum())}** parseable numeric values. "
                        "Check for non-numeric formatting in that column."
                    )
                else:
                    st.info(
                        "Expected a column named like **Temperature**, **Temp**, **Gas temperature**, etc. "
                        "Rename the column or tell the team the exact header to add support."
                    )
        if go is None:
            show_matplotlib_fig(fig_mfc)
        else:
            f_hi = float(mfc_df["F"].max()) if len(mfc_df) else 1.0
            mfc_def = {**MFC_WIDGET_DEFAULTS, "y_min": 0.0, "y_max": max(1.0, f_hi * 1.08)}
            with st.expander("Plot options — MFC", expanded=False):
                _ensure_widget_defaults("mfc", mfc_def)
                render_save_reset_row("mfc", mfc_def)
                render_font_legend_widgets("mfc")
                render_series_line_marker_widgets("mfc")
                st.checkbox("Lock x-axis to release window", key="mfc__lock_x_release")
                st.checkbox("Custom y-axis limits", key="mfc__use_custom_y")
                ym1, ym2 = st.columns(2)
                with ym1:
                    st.number_input("Y min", key="mfc__y_min")
                with ym2:
                    st.number_input("Y max", key="mfc__y_max")
                st.markdown("**X-axis (time) — when release lock is off**")
                if not bool(st.session_state.get("mfc__lock_x_release", True)):
                    render_x_mode_widgets("mfc", t0, t1, stage_defs)

            lock_rx = bool(st.session_state.get("mfc__lock_x_release", True))
            if lock_rx:
                xs, xe = t_rel0, t_rel1
            else:
                xs, xe = render_x_controls("mfc", t0, t1, stage_defs)
            y_r = None
            if st.session_state.get("mfc__use_custom_y", False):
                y_r = _y_pair_from_prefix("mfc", 0.0, max(1.0, f_hi * 1.08))

            lw_mfc, _ = _line_marker_from_prefix("mfc")
            fig_mfc_p = plot_mfc_plotly(
                mfc_df,
                t_on,
                t_off,
                t_rel0,
                t_rel1,
                cfg,
                x_start=xs,
                x_end=xe,
                lock_x_release=False,
                y_range=y_r,
                line_width=lw_mfc,
            )
            if fig_mfc_p is not None:
                apply_plotly_style(fig_mfc_p, _style_from_prefix("mfc"))
                show_plotly_chart(fig_mfc_p)

    if mfc_summary is not None:
        st.write("**MFC summary**")
        st.dataframe(build_summary_df(mfc_summary), use_container_width=True)

with tab8:
    st.subheader("Export")

    st.write("**Summary table**")
    st.dataframe(summary_df, use_container_width=True)

    csv_bytes = summary_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download summary CSV",
        data=csv_bytes,
        file_name=f"{cfg.exp_code}_summary.csv",
        mime="text/csv",
    )

    st.write("**Download figures (PNG)**")
    with st.expander("Figures", expanded=False):
        st.caption("Matplotlib-rendered PNGs. Interactive Plotly charts are on the other tabs.")

        # Overall metrics
        buf_overall_png = io.BytesIO()
        fig_overall.savefig(buf_overall_png, format="png", bbox_inches="tight")
        buf_overall_png.seek(0)
        st.download_button(
            label="Download overall metrics (PNG)",
            data=buf_overall_png,
            file_name=f"{cfg.exp_code}_overall_metrics.png",
            mime="image/png",
        )

        st.markdown("---")

        # Zone CO2
        buf_zone_png = io.BytesIO()
        fig_zone.savefig(buf_zone_png, format="png", bbox_inches="tight")
        buf_zone_png.seek(0)
        st.download_button(
            label="Download zone CO₂ (PNG)",
            data=buf_zone_png,
            file_name=f"{cfg.exp_code}_zone_co2.png",
            mime="image/png",
        )

        st.markdown("---")

        # Zone temperature
        buf_zoneT_png = io.BytesIO()
        fig_zone_T.savefig(buf_zoneT_png, format="png", bbox_inches="tight")
        buf_zoneT_png.seek(0)
        st.download_button(
            label="Download zone temperature (PNG)",
            data=buf_zoneT_png,
            file_name=f"{cfg.exp_code}_zone_temperature.png",
            mime="image/png",
        )

        if fig_mfc is not None:
            st.markdown("---")
            buf_mfc_png = io.BytesIO()
            fig_mfc.savefig(buf_mfc_png, format="png", bbox_inches="tight")
            buf_mfc_png.seek(0)
            st.download_button(
                label="Download MFC quicklook (PNG)",
                data=buf_mfc_png,
                file_name=f"{cfg.exp_code}_mfc.png",
                mime="image/png",
            )

# If we forced defaults due to a new upload, clear the flag after widgets have been created.
if st.session_state.get("__force_defaults_from_upload", False):
    st.session_state["__force_defaults_from_upload"] = False