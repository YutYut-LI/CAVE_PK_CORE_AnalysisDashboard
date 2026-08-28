# CAVE–PK CO2 / temperature analysis dashboard (Streamlit entry point).
# Former script name: CAVE_PK_CO2_Temp_Metrics.py
# Run locally: streamlit run app.py

from __future__ import annotations

import colorsys
from cycler import cycler
import html
import io
import os
import re
import traceback
import zipfile
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Any

import numpy as np
import pandas as pd
import streamlit as st
import datetime as _dt
import matplotlib as mpl
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
# Okabe-Ito qualitative palette — colour-blind-safe, the muted-but-distinct
# categorical scheme widely recommended (Nature Methods "Points of view")
# and used across many journal figures, in place of Matplotlib's default
# high-saturation tab10 cycle. Used for export figures only.
_JOURNAL_PALETTE = ["#0072B2", "#D55E00", "#009E73", "#E69F00", "#56B4E9", "#CC79A7", "#F0E442", "#000000"]


def _journal_colors(n: int) -> List[str]:
    """n colours in the journal palette; extends past 8 categories with
    evenly-spaced hues at a fixed muted saturation/value, so added colours
    read as the same family instead of tab20's jarring vivid/pastel pairs."""
    if n <= len(_JOURNAL_PALETTE):
        return _JOURNAL_PALETTE[:n]
    extra_n = n - len(_JOURNAL_PALETTE)
    extra = [
        mpl.colors.to_hex(colorsys.hsv_to_rgb(i / extra_n, 0.55, 0.75))
        for i in range(extra_n)
    ]
    return list(_JOURNAL_PALETTE) + extra


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
    # Fix the toolbar camera-icon "download as PNG" to a constant size/scale so
    # the exported image no longer depends on how wide the browser window
    # happens to be when you click it (fonts/lines are re-laid-out to fit
    # this size, not scaled from the on-screen render).
    "toImageButtonOptions": {
        "format": "png",
        "width": 1600,
        "height": 900,
        "scale": 2,
    },
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

    # ---- Air exchange (PK <-> CAVE) -------------------------------------
    # Zone volumes (m3). "Effective" CAVE volume excludes the space occupied
    # by PK, so it is the air volume actually taking part in the exchange.
    v_pk: float = 455.67
    v_cave_gross: float = 1917.49
    v_cave_effective: float = 1461.82
    use_effective_cave_volume: bool = True

    # Per-sensor baseline (increment) settings
    baseline_min_samples: int = 5   # sensors with fewer baseline points are dropped
    noise_sigma_k: float = 5.0      # excess threshold safeguard = k * sigma(baseline)

    # CAVE-side sensors mounted on the PK exterior wall. They belong to CAVE but
    # read the concentration at the envelope, not the room bulk, so they are kept
    # out of the CAVE bulk mean and offered separately as a driving concentration.
    envelope_walls: Tuple[str, ...] = ("FFE", "GFE")
    exclude_envelope_from_bulk: bool = True

    # Exchange-rate window / fit settings
    dc_min_ppm: float = 100.0       # continuous |dC| criterion from window start
    lam_win_min: int = 15           # sliding-window length (minutes)
    lam_step_min: int = 5           # sliding-window step (minutes)
    lam_min_pts_win: int = 10
    lam_min_pts_full: int = 20
    lam_min_pts_int: int = 10
    force_zero_intercept: bool = True
    lambda_ext: float = 0.0         # CAVE<->outdoor ACH, used only when CAVE receives

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
        "lam_window": (0.0, 0.80),
    }


def _auto_ylim(*series_list, pad_frac: float = 0.08) -> Optional[Tuple[float, float]]:
    """Data-driven y-limits (min/max across all given series, plus padding).

    Used for export-figure panels whose value range is experiment-specific
    (CO2/temperature levels, deltas) — as opposed to panels with a natural
    fixed domain (ratios like Mixing Index/R2/Coverage/RH/CV), which keep
    using the fixed defaults in default_ylims().
    """
    chunks = []
    for s in series_list:
        if s is None:
            continue
        arr = np.asarray(s, dtype=float).ravel()
        arr = arr[np.isfinite(arr)]
        if arr.size:
            chunks.append(arr)
    if not chunks:
        return None
    allv = np.concatenate(chunks)
    lo, hi = float(allv.min()), float(allv.max())
    if lo == hi:
        pad = max(abs(lo) * pad_frac, 1.0)
    else:
        pad = (hi - lo) * pad_frac
    return (lo - pad, hi + pad)


def _data_date_prefix(df: Optional[pd.DataFrame]) -> str:
    """YYYYMMDD from the data's own earliest timestamp, for export filenames."""
    try:
        t0 = pd.Timestamp(df["time"].min())
        if pd.notna(t0):
            return t0.strftime("%Y%m%d")
    except Exception:
        pass
    return "export"


def _export_figlegend(fig, handles, labels, *, where: str = "bottom", fontsize: int = 9,
                       anchor_ax=None) -> None:
    """Figure-level legend anchored just outside one axes' own top/bottom
    edge — never wedged between subplots regardless of how many panels
    there are, and wrapped into columns so it's never truncated regardless
    of series count. Anchoring in that axes' own coordinate space (rather
    than the whole figure's 0..1) means the gap is a small constant offset
    from the real edge, not compounded with tight_layout's own margins —
    which is what left a big blank band before. Use one call per figure
    when every panel shares the same series (e.g. CAVE/PK/stage colour
    coding repeated across all panels); use two calls — one 'top', one
    'bottom', each anchored to its own axes — when two panels show
    genuinely different series and a single shared legend would be
    ambiguous."""
    n = max(1, len(labels))
    axes_list = anchor_ax if anchor_ax is not None else fig.axes[0]
    if not isinstance(axes_list, (list, tuple)):
        axes_list = [axes_list]
    # Read the REAL position after layout (call tight_layout/etc. before
    # this). When multiple axes share the same row (e.g. both columns of a
    # 5x2 grid), take the union of their boxes — their tick-label widths can
    # differ enough that one column's "Time" label sits lower than the
    # other's, and anchoring off just one column would clip into it.
    fig.canvas.draw()
    boxes = [a.get_position() for a in axes_list]
    y0 = min(b.y0 for b in boxes)
    y1 = max(b.y1 for b in boxes)
    x0 = min(b.x0 for b in boxes)
    x1 = max(b.x1 for b in boxes)

    class _Pos:
        pass

    pos = _Pos()
    pos.y0, pos.y1, pos.x0, pos.x1 = y0, y1, x0, x1
    pos.width = x1 - x0
    fig_h = max(float(fig.get_figheight()), 1.0)
    fig_w = max(float(fig.get_figwidth()), 1.0)
    # Cap columns so the legend's estimated width never exceeds the anchor
    # axes' own width — otherwise a legend with few, short entries (e.g. 5
    # walls + 3 stages) renders wider than the plot it sits above/below,
    # which looks unbalanced even though nothing gets cut off.
    handlelen, handlepad, colspace = 1.8, 0.6, 1.4
    fs_pt = max(5, min(24, int(fontsize)))
    avg_chars = sum(len(str(l)) for l in labels) / n
    entry_w_in = ((handlelen + handlepad + colspace) + avg_chars * 0.58) * fs_pt / 72.0
    avail_w_in = max(pos.width * fig_w, entry_w_in)
    ncol = max(1, min(n, 6, int(avail_w_in / max(entry_w_in, 0.01))))
    n_rows = -(-n // ncol)  # ceil
    if where == "top":
        gap_in = 0.10 + 0.16 * (n_rows - 1)
        loc, anchor_y = "lower center", pos.y1 + gap_in / fig_h
    else:
        # 'bottom' has to clear that axes' rotated tick labels + the bold
        # "Time" x-label sitting just below it. Measure their real rendered
        # extent instead of guessing a fixed inch offset — a flat constant
        # drifted out of sync once export mode switched to larger fonts
        # (taller tick/axis labels) and started overlapping the legend.
        renderer = fig.canvas.get_renderer()
        content_bottoms = [pos.y0]
        for a in axes_list:
            xlabel_bbox = a.xaxis.label.get_window_extent(renderer)
            content_bottoms.append(xlabel_bbox.transformed(fig.transFigure.inverted()).y0)
            for tl in a.get_xticklabels():
                tb = tl.get_window_extent(renderer)
                if tb.width > 0 or tb.height > 0:
                    content_bottoms.append(tb.transformed(fig.transFigure.inverted()).y0)
        content_bottom = min(content_bottoms)
        buffer_in = 0.14 + 0.16 * (n_rows - 1)
        loc, anchor_y = "upper center", content_bottom - buffer_in / fig_h
    fig.legend(
        handles, labels, loc=loc, bbox_to_anchor=(0.5, anchor_y), bbox_transform=fig.transFigure,
        ncol=ncol, frameon=True, fontsize=fs_pt,
        columnspacing=colspace, handletextpad=handlepad, handlelength=handlelen, labelspacing=0.5,
        edgecolor="0.75", facecolor="white",
    )


def _publication_rc_dict(base_pt: float) -> Dict[str, Any]:
    b = float(base_pt)
    return {
        "font.size": b,
        "axes.labelsize": b,
        "axes.titlesize": b + 1,
        "xtick.labelsize": max(b - 1, 8),
        "ytick.labelsize": max(b - 1, 8),
        "legend.fontsize": max(b - 2, 7),
        "figure.titlesize": b + 2,
        "axes.prop_cycle": cycler(color=_JOURNAL_PALETTE),
        "axes.linewidth": 0.9,
        "axes.edgecolor": "0.25",
        "axes.labelcolor": "black",
        "xtick.color": "0.25",
        "ytick.color": "0.25",
        "text.color": "black",
        "legend.frameon": True,
        "legend.facecolor": "white",
        "legend.edgecolor": "0.75",
    }


def apply_matplotlib_publication_rc(base_pt: float) -> None:
    """Set matplotlib rcParams for dashboard + export (publication-friendly)."""
    plt.rcParams.update(_publication_rc_dict(base_pt))


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

    # Detect the temperature column from the *raw* CSV columns only, before adding
    # our own derived "t"/"Fset"/"Fmeas" columns below — otherwise the fallback
    # matcher can mistake our derived "t" (time) column for a temperature column
    # on older MFC files that don't record temperature at all.
    temp_col = _detect_mfc_temperature_column(dfm.columns)

    dfm["t"] = pd.to_datetime(dfm[ts_col], errors="coerce", dayfirst=True)
    dfm["Fset"] = _parse_mfc_numeric_series(dfm[fset_col])
    dfm["Fmeas"] = _parse_mfc_numeric_series(dfm[fmeas_col])
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


# =========================================================
# Air exchange (PK <-> CAVE): per-sensor baselines / increments
# =========================================================
# Every Explora sensor carries its own zero-point calibration offset. Measured
# over a baseline stage, a single sensor is stable to ~0.5 ppm while different
# sensors disagree by 40-70 ppm, so the between-sensor spread is calibration,
# not mixing. Subtracting a per-sensor baseline ("increment") removes that
# offset and, unlike a region-level baseline, stays correct when the set of
# reporting sensors changes over time.
def per_sensor_baselines(df_region: pd.DataFrame, t0, t1, min_samples: int):
    """Mean CO2 per sensor over [t0, t1]; sensors with too few points are dropped."""
    empty = (pd.Series(dtype=float), pd.DataFrame(columns=["sensor_number", "wall", "baseline", "n", "kept"]))
    if df_region is None or df_region.empty or t0 is None or t1 is None:
        return empty

    d = df_region.dropna(subset=["time", "co2"]).copy()
    d = d[(d["time"] >= pd.Timestamp(t0)) & (d["time"] <= pd.Timestamp(t1))]
    if d.empty:
        return empty

    g = d.groupby("sensor_number")
    info = pd.DataFrame({
        "baseline": g["co2"].mean(),
        "n": g["co2"].size(),
        "wall": g["wall"].first().astype(str),
    }).reset_index()
    info["kept"] = info["n"] >= int(min_samples)

    kept = info[info["kept"]]
    baselines = pd.Series(kept["baseline"].values, index=kept["sensor_number"].values, dtype=float)

    ref = float(kept["baseline"].mean()) if len(kept) else np.nan
    info["offset"] = info["baseline"] - ref
    return baselines, info.sort_values("sensor_number").reset_index(drop=True)


def add_increment_column(df: pd.DataFrame, baselines: pd.Series) -> pd.DataFrame:
    """Attach co2_inc = co2 - per-sensor baseline. Sensors without a baseline are dropped."""
    out = df.copy()
    b = out["sensor_number"].map(baselines)
    out = out[b.notna()].copy()
    out["co2_inc"] = out["co2"] - b[b.notna()]
    return out


def excess_mean_series(df_region: pd.DataFrame, align_to: str, min_sensors: int, value_col: str = "co2_inc"):
    """Region-mean excess (increment) time series, gated on sensor count per bin."""
    if df_region is None or df_region.empty or value_col not in df_region.columns:
        return pd.Series(dtype=float)
    d = df_region.dropna(subset=["time", value_col]).copy()
    if d.empty:
        return pd.Series(dtype=float)
    d["tbin"] = d["time"].dt.floor(align_to)
    g = d.groupby("tbin")
    n = g["sensor_number"].nunique()
    return g[value_col].mean().where(n >= min_sensors)


def excess_noise_stats(df_region: pd.DataFrame, ex_series: pd.Series, t0, t1, align_to: str) -> Dict[str, float]:
    """Noise floor of the debiased data over the baseline window.

    sigma_sensor : between-sensor spread of increments (calibration removed)
    sigma_mean   : noise of the region MEAN, i.e. sigma_sensor / sqrt(n)
    sd_series    : temporal sd of the region-mean excess (used for the threshold)
    """
    out = {"sigma_sensor": np.nan, "sigma_mean": np.nan, "sd_series": np.nan, "n_sensors": 0}
    if df_region is None or df_region.empty or t0 is None or t1 is None:
        return out

    d = df_region.dropna(subset=["time", "co2_inc"]).copy()
    d = d[(d["time"] >= pd.Timestamp(t0)) & (d["time"] <= pd.Timestamp(t1))]
    if not d.empty:
        d["tbin"] = d["time"].dt.floor(align_to)
        out["sigma_sensor"] = float(d.groupby("tbin")["co2_inc"].std().mean())
        out["n_sensors"] = int(d["sensor_number"].nunique())
        if out["n_sensors"] > 0 and np.isfinite(out["sigma_sensor"]):
            out["sigma_mean"] = out["sigma_sensor"] / np.sqrt(out["n_sensors"])

    s = ex_series.dropna() if ex_series is not None else pd.Series(dtype=float)
    s = s[(s.index >= pd.Timestamp(t0)) & (s.index <= pd.Timestamp(t1))]
    if len(s) > 1:
        out["sd_series"] = float(s.std(ddof=1))
    return out


# =========================================================
# Air exchange: direction, window selection, estimators
# =========================================================
DIR_CAVE_TO_PK = "CAVE → PK"
DIR_PK_TO_CAVE = "PK → CAVE"


def detect_exchange_direction(ex_cave: pd.Series, ex_pk: pd.Series, t0, t1) -> Tuple[str, float, float]:
    """Source zone = whichever region carries the larger excess over [t0, t1]."""
    c = mean_in_window(ex_cave, t0, t1) if (ex_cave is not None and len(ex_cave)) else np.nan
    p = mean_in_window(ex_pk, t0, t1) if (ex_pk is not None and len(ex_pk)) else np.nan
    if np.isfinite(c) and np.isfinite(p) and p > c:
        return DIR_PK_TO_CAVE, c, p
    return DIR_CAVE_TO_PK, c, p


def select_exchange_window(ex_other: pd.Series, ex_solve: pd.Series, t_start, t_end, dc_min: float):
    """Longest unbroken stretch in [t_start, t_end] with |dC| >= dc_min and a constant sign.

    dC = other zone - solved zone. Its sign depends on which way the gradient runs:
    positive while the solved zone is filling, negative while it is emptying. Both
    are valid — the through-origin regression flips X and Y together — so the
    magnitude sets the signal-to-noise floor and the sign must merely stay constant.
    A sign change mid-window means the gradient reversed and the single well-mixed
    two-zone model no longer describes that stretch, so runs are cut there.

    Points are never filtered individually; the fit always runs on one continuous
    segment. Taking the longest run covers both window shapes: a decay stage where
    |dC| starts large and shrinks, and a rising release stage where it must first
    climb past the threshold.
    """
    fail = (pd.DatetimeIndex([]), pd.Series(dtype=float), "no overlapping data for the two zones")
    if ex_other is None or ex_solve is None or len(ex_other) == 0 or len(ex_solve) == 0:
        return fail

    idx = ex_other.dropna().index.intersection(ex_solve.dropna().index)
    if t_start is not None:
        idx = idx[idx >= pd.Timestamp(t_start)]
    if t_end is not None:
        idx = idx[idx <= pd.Timestamp(t_end)]
    if len(idx) == 0:
        return fail

    dC = (ex_other.reindex(idx) - ex_solve.reindex(idx)).astype(float)
    vals = dC.to_numpy(dtype=float)
    ok = np.isfinite(vals) & (np.abs(vals) >= float(dc_min))

    if not ok.any():
        peak = np.nanmax(np.abs(vals)) if np.isfinite(vals).any() else np.nan
        return (pd.DatetimeIndex([]), dC,
                f"|ΔC| never reaches {dc_min:.0f} ppm in this stage (peak {peak:.0f} ppm)")

    # Maximal runs that stay above the threshold AND keep one sign.
    sgn = np.sign(vals)
    runs, start = [], None
    for i, flag in enumerate(ok):
        if not flag:
            if start is not None:
                runs.append((start, i)); start = None
        elif start is None:
            start = i
        elif sgn[i] != sgn[start]:
            runs.append((start, i)); start = i
    if start is not None:
        runs.append((start, len(ok)))

    a, b = max(runs, key=lambda r: r[1] - r[0])
    keep = idx[a:b]
    direction_word = "into" if sgn[a] > 0 else "out of"

    bits = [f"ΔC {'positive' if sgn[a] > 0 else 'negative'} (tracer moving {direction_word} the solved zone)"]
    if a > 0:
        bits.append(f"started at {idx[a]:%H:%M:%S}")
    if b < len(ok):
        bits.append(f"ended at {idx[b]:%H:%M:%S} when |ΔC| dropped below {dc_min:.0f} ppm or the gradient reversed")
    if a == 0 and b == len(ok):
        bits.append(f"|ΔC| stayed above {dc_min:.0f} ppm for the whole stage")
    if len(runs) > 1:
        bits.append(f"longest of {len(runs)} qualifying segments")

    return keep, dC.loc[keep], "; ".join(bits)


def _cumtrapz_seconds(values: np.ndarray, t_sec: np.ndarray) -> np.ndarray:
    """Cumulative trapezoidal integral, first element 0."""
    out = np.zeros_like(values, dtype=float)
    if len(values) < 2:
        return out
    out[1:] = np.cumsum(0.5 * (values[1:] + values[:-1]) * np.diff(t_sec))
    return out


def lambda_from_XY(X, Y) -> Tuple[float, float, int]:
    """Through-origin least squares  lambda = sum(X*Y) / sum(X^2)  [units of Y/X]."""
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    ok = np.isfinite(X) & np.isfinite(Y)
    X, Y = X[ok], Y[ok]
    if len(X) < 3:
        return np.nan, np.nan, len(X)
    denom = float(np.dot(X, X))
    if denom <= 0:
        return np.nan, np.nan, len(X)
    lam = float(np.dot(X, Y) / denom)
    ss_res = float(np.sum((Y - lam * X) ** 2))
    ss_tot = float(np.sum((Y - np.mean(Y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return lam, r2, len(X)


def lambda_integrated(rcv: pd.Series, dC: pd.Series, force_zero_intercept: bool, lambda_ext_per_s: float = 0.0):
    """Integrated method:  y = C_rcv(t) - C_rcv(t0) [+ lam_ext * int C_rcv]  vs  x = int dC dt."""
    out = {"lam_h": np.nan, "r2": np.nan, "n": 0, "x": np.array([]), "y": np.array([]), "intercept": 0.0}
    r = rcv.dropna()
    idx = r.index.intersection(dC.dropna().index)
    if len(idx) < 2:
        return out

    r = r.reindex(idx).astype(float)
    d = dC.reindex(idx).astype(float)
    t_sec = (idx - idx[0]).total_seconds().to_numpy(dtype=float)

    y = (r - float(r.iloc[0])).to_numpy(dtype=float)
    if lambda_ext_per_s:
        # Receiver also loses to outdoors: move that known sink to the left-hand side.
        y = y + lambda_ext_per_s * _cumtrapz_seconds(r.to_numpy(dtype=float), t_sec)
    x = _cumtrapz_seconds(d.to_numpy(dtype=float), t_sec)

    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 2:
        return out

    if force_zero_intercept:
        lam, r2, n = lambda_from_XY(x, y)
        b = 0.0
    else:
        lam, b = np.polyfit(x, y, 1)
        y_hat = lam * x + b
        ss_res = float(np.sum((y - y_hat) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
        n = len(x)

    out.update({"lam_h": lam * 3600.0, "r2": r2, "n": n, "x": x, "y": y, "intercept": float(b)})
    return out


def lambda_differential(rcv: pd.Series, dC: pd.Series, dc_min: float, lambda_ext_per_s: float = 0.0):
    """Discrete forward-difference form:  Y = dC_rcv/dt [+ lam_ext * C_rcv],  X = dC."""
    out = {"X": np.array([]), "Y": np.array([]), "t_mid": pd.DatetimeIndex([]),
           "lam_h": np.nan, "r2": np.nan, "n": 0}
    r = rcv.dropna()
    idx = r.index.intersection(dC.dropna().index)
    if len(idx) < 3:
        return out

    r = r.reindex(idx).astype(float)
    d = dC.reindex(idx).astype(float)
    t_sec = (idx - idx[0]).total_seconds().to_numpy(dtype=float)

    dt = np.diff(t_sec)
    dr = np.diff(r.to_numpy(dtype=float))
    Y = np.full_like(dt, np.nan, dtype=float)
    ok_dt = dt > 0
    Y[ok_dt] = dr[ok_dt] / dt[ok_dt]
    if lambda_ext_per_s:
        Y = Y + lambda_ext_per_s * r.to_numpy(dtype=float)[:-1]

    X = d.to_numpy(dtype=float)[:-1]
    t_mid = idx[:-1]

    # Magnitude, not sign: dC is negative whenever the solved zone is emptying.
    ok = np.isfinite(X) & np.isfinite(Y) & (np.abs(X) >= float(dc_min))
    X, Y, t_mid = X[ok], Y[ok], t_mid[ok]

    lam, r2, n = lambda_from_XY(X, Y)
    out.update({"X": X, "Y": Y, "t_mid": t_mid, "lam_h": lam * 3600.0, "r2": r2, "n": n})
    return out


def lambda_sliding(X, Y, t_mid, win_min: int, step_min: int, min_pts: int):
    """Sliding-window through-origin fits; returns window-centre times and lambda in 1/h."""
    out = {"times": pd.DatetimeIndex([]), "lam_h": np.array([]), "r2": np.array([]),
           "mean_h": np.nan, "median_h": np.nan}
    t_mid = pd.DatetimeIndex(t_mid)
    if len(t_mid) < min_pts:
        return out

    win = pd.Timedelta(minutes=int(win_min))
    step = pd.Timedelta(minutes=int(step_min))
    t_cur, t_end = t_mid.min(), t_mid.max()

    times, lams, r2s = [], [], []
    while t_cur + win <= t_end:
        m = (t_mid >= t_cur) & (t_mid < t_cur + win)
        if int(np.sum(m)) >= int(min_pts):
            lam, r2, _ = lambda_from_XY(np.asarray(X)[m], np.asarray(Y)[m])
            if np.isfinite(lam):
                times.append(t_cur + win / 2)
                lams.append(lam * 3600.0)
                r2s.append(r2)
        t_cur = t_cur + step

    if not times:
        return out
    lam_arr = np.asarray(lams, dtype=float)
    out.update({
        "times": pd.to_datetime(times),
        "lam_h": lam_arr,
        "r2": np.asarray(r2s, dtype=float),
        "mean_h": float(np.nanmean(lam_arr)),
        "median_h": float(np.nanmedian(lam_arr)),
    })
    return out


# =========================================================
# Air exchange: transfer (I/O) ratio and excess scatter
# =========================================================
def compute_transfer_ratio(src_ex: pd.Series, rcv_ex: pd.Series, thresh: float, t_rel0, t_rel1):
    """receiver/source excess ratio, gated on the DENOMINATOR only.

    The numerator is never clipped: a negative receiver excess early in the
    release is real (the receiver has not responded yet) and clipping it would
    bias the release-window mean upward.
    """
    out = {"io_ex": pd.Series(dtype=float), "factor": np.nan, "sd": np.nan, "n": 0, "n_gated": 0}
    if src_ex is None or rcv_ex is None or len(src_ex) == 0 or len(rcv_ex) == 0:
        return out

    idx = src_ex.dropna().index.intersection(rcv_ex.dropna().index)
    if len(idx) == 0:
        return out

    s = src_ex.reindex(idx).astype(float)
    r = rcv_ex.reindex(idx).astype(float)
    gate = s > float(thresh)
    io_ex = (r / s).where(gate)

    rel = io_ex
    if t_rel0 is not None and t_rel1 is not None:
        rel = io_ex[(io_ex.index >= pd.Timestamp(t_rel0)) & (io_ex.index <= pd.Timestamp(t_rel1))]
        n_gated = int((~gate[(gate.index >= pd.Timestamp(t_rel0)) & (gate.index <= pd.Timestamp(t_rel1))]).sum())
    else:
        n_gated = int((~gate).sum())
    rel = rel.dropna()

    out.update({
        "io_ex": io_ex,
        "factor": float(rel.mean()) if len(rel) else np.nan,
        "sd": float(rel.std(ddof=1)) if len(rel) > 1 else np.nan,
        "n": int(len(rel)),
        "n_gated": n_gated,
    })
    return out


def fit_excess_scatter(src_ex: pd.Series, rcv_ex: pd.Series, thresh: float, t_rel0, t_rel1):
    """OLS of receiver excess on source excess over the release window."""
    empty = pd.DataFrame(columns=["cave_ex", "pk_ex"])
    if src_ex is None or rcv_ex is None or len(src_ex) == 0 or len(rcv_ex) == 0:
        return empty, np.nan, np.nan, np.nan

    idx = src_ex.dropna().index.intersection(rcv_ex.dropna().index)
    if t_rel0 is not None and t_rel1 is not None:
        idx = idx[(idx >= pd.Timestamp(t_rel0)) & (idx <= pd.Timestamp(t_rel1))]
    if len(idx) == 0:
        return empty, np.nan, np.nan, np.nan

    # Column names kept as cave_ex/pk_ex so the existing scatter plotters work unchanged.
    df_sc = pd.DataFrame({"cave_ex": src_ex.reindex(idx), "pk_ex": rcv_ex.reindex(idx)}).dropna()
    df_sc = df_sc[df_sc["cave_ex"] > float(thresh)]
    if len(df_sc) < 2:
        return df_sc, np.nan, np.nan, np.nan

    x = df_sc["cave_ex"].to_numpy(dtype=float)
    y = df_sc["pk_ex"].to_numpy(dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    y_hat = slope * x + intercept
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return df_sc, float(slope), float(intercept), float(r2)


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
    label_fn=None,
) -> pd.DataFrame:
    """Per-sensor mean by time bin (columns = readable sensor labels).
    label_fn(sensor_number, wall, z_median) overrides the default "S### (wall,
    z=X.XXm)" label — e.g. the PK per-room view drops the wall since it's
    already implied by which room's panel the sensor is plotted on."""
    if df_region is None or len(df_region) == 0 or not sensor_numbers or value_col not in df_region.columns:
        return pd.DataFrame()

    cat = catalog if catalog is not None else sensor_catalog(df_region)
    _label = label_fn if label_fn is not None else _sensor_series_label
    label_by_id = {
        int(r["sensor_number"]): _label(
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
    label_fn=None,
) -> pd.DataFrame:
    return sensor_value_timeseries(df_region, sensor_numbers, align_to, "co2", catalog=catalog, label_fn=label_fn)


# =========================================================
# PK — per-room floor-plan view
# =========================================================
# Room adjacency matches the reference poster exactly: the floor plan
# shares its column with FF03 (above) and FF05 (below) on FF, and with
# GFS (below) on GF — same as those rooms' real physical position next to
# it — rather than being pulled into its own column. That column is wider
# than the room-only columns (so the floor plan itself can be big, close
# to its own true aspect ratio, without distorting it), but FF03/FF05/GFS
# are re-centered back down to the same width as every other room in the
# export "template" composite (see the nested subgridspec in
# plot_pk_floorplan_export) — otherwise sharing a wider column would make
# them wider than the other 6 rooms too. Row alignment between columns is
# not attempted (by request — the floor-plan:room-chart size ratio in the
# reference matters more than every column bottoming out at the same row).
PK_FLOORPLAN_LAYOUT = {
    "FF": [
        {"width": 1.0, "items": ["FF01", "__GAP__", "FF02", "__GAP__", "FF04"]},
        {"width": 1.4663, "items": ["FF03", "__GAP__", "__FLOORPLAN__", "__GAP__", "FF05"]},
        {"width": 1.0, "items": ["FF06", "__GAP__", "FFC", "__GAP__", "FFS"]},
    ],
    "GF": [
        {"width": 1.0, "items": ["GF01"]},
        {"width": 1.381, "items": ["__FLOORPLAN__", "__GAP__", "GFS"]},
    ],
}
PK_FLOORPLAN_IMAGES = {"FF": "PK_FirstFloorPlan.png", "GF": "PK_GroundFloorPlan.png"}
# Every room chart is exactly PK_FLOORPLAN_ROOM_UNIT grid rows tall,
# regardless of column — same height for every room, always. The floor
# plan gets FP_SPAN rows instead. Both floors' base numbers are measured
# directly from the reference file's own slides — FF from slide 15
# (floor plan 5.591x2.863in vs. every room chart 3.813x1.985in, ratios
# 1.466 wide/1.442 tall) and GF from its paired slide 16 (floor plan
# 5.981x3.629in vs. every room chart 5.413x2.818in, ratios 1.105 wide/
# 1.288 tall) — not a guess either way. GF01/GFE/GFS are the same size
# as each other there too; GFE itself is left out of PK_FLOORPLAN_LAYOUT
# on purpose (classify_regions() puts it in CAVE, not PK — see
# _pk_room_group's docstring), so only the floor-plan:room size ratio
# from that slide carries over, not its exact room list. GF's measured
# ratio read as noticeably smaller than FF's once actually rendered, so
# by request it's scaled up 1.25x from that measured baseline (1.105->
# 1.381 width, 129->161 span) rather than left at the literal slide value.
PK_FLOORPLAN_ROOM_WIDTH = 1.0
PK_FLOORPLAN_ROOM_UNIT = 100
# Real blank space (in the same units) between two rooms stacked in the
# same column — see the "hspace=0.0" note above for why this is a real
# grid span rather than GridSpec's hspace.
PK_FLOORPLAN_GAP_SPAN = 28
PK_FLOORPLAN_FP_SPAN = {"FF": 144, "GF": 161}
# GridSpec's column-to-column gap (a fraction of average column width).
# GF's columns hold plain, un-nested cells on both sides (GF01 | floor
# plan), so this *is* the real visible gap between them — measured at
# 2.844in with the old shared 0.30 versus the 0.797in vertical gap
# between the floor plan and GFS below it (the "__GAP__" spans, tuned
# separately), so it's cut down here to make the two roughly match. FF
# keeps the old 0.30: its side-column rooms sit right up against a
# *nested* nested cell (FF03/FF05's own margin already adds real blank
# space), so the same wspace produces a different-looking gap there.
PK_FLOORPLAN_WSPACE = {"FF": 0.30, "GF": -0.074}


def _pk_room_group(wall: str) -> str:
    """Collapse duct/perimeter sensor tags into the parent room box drawn on
    the floor plan (e.g. FF01_Supply -> FF01) rather than the finer-grained
    wall/zone tags used for the Zone CO2/Temp tab. GFE/FFE never reach here
    at all — classify_regions() already routes them to CAVE, not PK (they're
    CAVE's exterior/perimeter sensors despite the "GF"/"FF" naming), so this
    tab (which only ever looks at df_pk) correctly never shows them. Still
    normalized to upper case so a raw tag written as e.g. "ff01_supply"
    matches the (upper-case) room names in PK_FLOORPLAN_LAYOUT."""
    w = str(wall).strip().upper()
    for suffix in ("_SUPPLY", "_EXTRACT"):
        if w.endswith(suffix):
            return w[: -len(suffix)]
    return w


def _room_sensor_label(sensor_number: int, wall: str, z_median: float = np.nan) -> str:
    """Legend label for the PK per-room view: just the sensor + height, no
    wall repeated (the room is already implied by which panel it's on) —
    matches the reference poster's "S251 (z=6.00 m)" style."""
    z_part = f"z={z_median:.2f} m" if np.isfinite(z_median) else "z=?"
    return f"S{int(sensor_number)} ({z_part})"


# Bump whenever plot_room_sensors_matplotlib's drawing logic changes.
# plot_pk_floorplan_export is @st.cache_resource'd, but Streamlit's cache
# only rehashes *that* function's own bytecode on an edit — it doesn't
# recurse into plot_room_sensors_matplotlib (a plain module-level function
# it calls, not a closure), so an edit here alone would keep serving a
# stale cached composite. Passing this constant into plot_pk_floorplan_export
# as a real (hashed) argument forces the cache to invalidate whenever it's
# bumped, regardless of that limitation.
PK_ROOM_PLOT_VERSION = 2


def plot_room_sensors_matplotlib(
    ts_df: pd.DataFrame,
    room_label: str,
    stage_defs,
    plot_start,
    plot_end,
    *,
    y_range: Optional[Tuple[float, float]] = None,
    line_width: float = 1.3,
    legend_fontsize: int = 8,
    figsize: Tuple[float, float] = (5.2, 3.6),
    ax: Optional[Any] = None,
):
    """One small-multiple panel: every individual sensor in a single PK room,
    CO2 vs time — matplotlib (not Plotly) so the legend can wrap into a
    compact multi-column grid like the reference poster, scaling the column
    count with how many sensors are in the room.

    Pass an existing `ax` to draw into that axes instead of creating a new
    standalone figure — this is how the export "template" composite (whole
    floor, one image) reuses the exact same per-room drawing/legend logic
    as the on-screen single-room panels."""
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    n = max(1, len(ts_df.columns))
    # Real per-bin dropouts (a sensor occasionally missing a reading) leave
    # scattered single-point NaNs in this data — plotted as-is, matplotlib
    # breaks the line at every one of them, which for a handful of scattered
    # gaps reads as a dashed/broken line rather than a clean solid one.
    # Linear-interpolating those small gaps (not the underlying data table,
    # only this plotting copy) draws one continuous line without pretending
    # a real, longer outage didn't happen.
    ts_plot = ts_df.interpolate(method="linear", limit_direction="both")
    for col in ts_plot.columns:
        ax.plot(ts_plot.index, ts_plot[col].values, linewidth=line_width, label=col)

    stage_patches: list = []
    if stage_defs:
        add_stage_shading(ax, stage_defs, stage_patches)

    ax.set_title(f"{room_label} — CO₂ concentration over time", fontsize=10, fontweight="bold")
    ax.set_ylabel("CO₂ (ppm)", fontsize=9, fontweight="bold")
    ax.set_xlabel("Time", fontsize=9, fontweight="bold")
    ax.grid(True, color="0.85", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    plt.setp(ax.get_xticklabels(), rotation=45, fontsize=7)
    plt.setp(ax.get_yticklabels(), fontsize=7)

    if plot_start is not None and plot_end is not None:
        ax.set_xlim(plot_start, plot_end)
    if y_range is not None:
        ax.set_ylim(*y_range)
    else:
        auto = _auto_ylim(*[ts_df[c] for c in ts_df.columns])
        if auto:
            ax.set_ylim(*auto)

    # Scale legend columns with sensor count so a 20+-sensor room (e.g. GF01)
    # stays a compact block instead of one very tall single column — matching
    # the reference poster's layout, which does the same.
    ncol = max(1, min(4, -(-n // 6)))
    # fig.tight_layout() (not the pyplot-level plt.tight_layout()) so this
    # always targets this specific figure, regardless of matplotlib's
    # notion of the "current" figure — matters once this is called
    # repeatedly for several rooms sharing one composite figure.
    fig.tight_layout()
    if n <= 8:
        # Small enough to tuck into an empty corner of the plot itself
        # (matplotlib picks the least-busy spot) without covering data.
        ax.legend(fontsize=legend_fontsize, ncol=ncol, loc="best", frameon=True, framealpha=0.9)
    else:
        # Too many entries to fit inside the axes without covering data —
        # drop it below instead, like the reference poster's GF01 panel.
        # Measure the real rendered bottom edge of the x-tick labels/xlabel
        # (their height depends on font size and rotation, not on the
        # axes' own size) and anchor a small fixed buffer below THAT,
        # rather than guessing an axes-fraction offset — a flat fraction
        # constant left short single-row axes (e.g. FF04/FFS here) with too
        # little real clearance while over-spacing taller ones (e.g. GF01,
        # which spans two grid rows in the export "template" composite).
        # st.pyplot()/savefig(bbox_inches="tight") expands the saved image
        # to include it, so it never gets clipped.
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        content_bottoms_px = [ax.xaxis.label.get_window_extent(renderer).y0]
        for tl in ax.get_xticklabels():
            tb = tl.get_window_extent(renderer)
            if tb.width > 0 or tb.height > 0:
                content_bottoms_px.append(tb.y0)
        buffer_px = 0.12 * fig.dpi
        target_top_px = min(content_bottoms_px) - buffer_px
        target_axes_frac = ax.transAxes.inverted().transform((0.0, target_top_px))[1]
        ax.legend(
            fontsize=legend_fontsize, ncol=ncol, loc="upper center",
            bbox_to_anchor=(0.5, target_axes_frac),
            frameon=True, framealpha=0.9,
        )
    return fig if standalone else ax


@st.cache_resource(show_spinner=False)
def plot_pk_floorplan_export(
    floor_key: str,
    pk_cat: pd.DataFrame,
    df_pk: pd.DataFrame,
    align_to: str,
    stage_defs,
    plot_start,
    plot_end,
    fp_img_path: Optional[str],
    *,
    y_range: Optional[Tuple[float, float]] = None,
    cache_version: int = PK_ROOM_PLOT_VERSION,
):
    """One fixed, report-ready composite for a whole floor: every room's
    sensor-level CO2 panel plus the real floor plan, laid out exactly like
    PK_FLOORPLAN_LAYOUT — the downloadable "template" version of the
    on-screen PK Rooms tab, all as a single PNG/SVG."""
    columns = PK_FLOORPLAN_LAYOUT[floor_key]
    col_widths = [c["width"] for c in columns]
    room_unit = PK_FLOORPLAN_ROOM_UNIT
    fp_span = PK_FLOORPLAN_FP_SPAN.get(floor_key, 2 * room_unit)

    def _item_span(it: str) -> int:
        if it == "__FLOORPLAN__":
            return fp_span
        if it == "__GAP__":
            return PK_FLOORPLAN_GAP_SPAN
        return room_unit

    def _col_row_units(items) -> int:
        return sum(_item_span(it) for it in items)

    n_rows = max(_col_row_units(c["items"]) for c in columns)
    # A slim strip across the top, reserved for a shared stage legend
    # (Baseline/Release/Decay swatches) — one legend covers every panel
    # since they all use the same stage shading, rather than repeating it
    # per room. Counted in the same row units as everything else so it's
    # baked into the fixed 16:9 canvas below, not an extra bleed added on
    # top of it.
    legend_rows = max(1, round(n_rows * 0.07)) if stage_defs else 0
    # A little clearance between the legend strip and the first room's own
    # title right below it — without this they touch (same gap value
    # already proven sufficient between two stacked rooms, so reused here).
    legend_gap = PK_FLOORPLAN_GAP_SPAN if legend_rows else 0
    content_row0 = legend_rows + legend_gap
    total_rows = content_row0 + n_rows
    # 3.7in is "one room's height" — rows are in fine-grained room_unit
    # subdivisions now (so FP_SPAN can approximate a precisely-measured
    # fractional multiple of a room's height, e.g. 1.44x), not literally
    # one GridSpec row per room, so this scales each row by 1/room_unit
    # rather than treating a row as a whole room.
    natural_w, natural_h = 5.0 * sum(col_widths), (3.7 / room_unit) * total_rows
    # Force the overall canvas to 16:9, by *growing* whichever dimension
    # the natural layout comes up short on — never shrinking either one —
    # so every room chart (which fills a fixed share of that canvas) ends
    # up as large as this aspect ratio allows, not smaller.
    target_ratio = 16.0 / 9.0
    if natural_w / natural_h < target_ratio:
        fig_w, fig_h = natural_h * target_ratio, natural_h
    else:
        fig_w, fig_h = natural_w, natural_w / target_ratio
    fig = plt.figure(figsize=(fig_w, fig_h))
    # hspace=0: with rooms now spanning many fine-grained rows (room_unit
    # of them) instead of exactly one, matplotlib's hspace fraction gets
    # applied at *every* row boundary the GridSpec has, including the ones
    # buried inside a single room's own span — a nonzero hspace here
    # doesn't just add a gap between rooms, it bloats each room's own
    # multi-row cell by (room_unit - 1) copies of that gap. Real breathing
    # room between rooms is inserted explicitly as "__GAP__" entries in
    # PK_FLOORPLAN_LAYOUT instead (a real, deliberately-sized blank grid
    # span), which sidesteps this entirely.
    gs = fig.add_gridspec(
        total_rows, len(columns), width_ratios=col_widths,
        hspace=0.0, wspace=PK_FLOORPLAN_WSPACE.get(floor_key, 0.30),
    )

    if legend_rows:
        seen_names: set = set()
        stage_patches = []
        for (name, _stt, _ett, col) in stage_defs:
            if name in seen_names:
                continue
            seen_names.add(name)
            stage_patches.append(Patch(facecolor=col, alpha=0.40, label=name))
        legend_ax = fig.add_subplot(gs[0:legend_rows, :])
        legend_ax.axis("off")
        legend_ax.legend(
            handles=stage_patches, loc="center", ncol=len(stage_patches),
            frameon=False, fontsize=13,
        )

    for col_idx, col_spec in enumerate(columns):
        items = col_spec["items"]
        col_w = col_spec["width"]
        has_fp = "__FLOORPLAN__" in items
        # Columns shared with the floor plan are wider than a room's own
        # width so the floor plan can be big — but its room(s) still need
        # to render at the normal room width, not stretched to match. This
        # nests a normal-width, centered sub-cell for them inside the
        # wider outer cell (the floor plan itself uses the full outer
        # cell, no nesting).
        needs_nesting = has_fp and col_w > PK_FLOORPLAN_ROOM_WIDTH
        margin = (col_w - PK_FLOORPLAN_ROOM_WIDTH) / 2 if needs_nesting else None

        # Below the legend strip, top-aligned — matching how the reference
        # poster's own rooms sit (e.g. GF01 starts almost level with the
        # floor plan's own top, not centered against the whole column).
        # A short column (like GF's lone GF01) just leaves its own leftover
        # space at the bottom instead of floating in the middle.
        row_cursor = content_row0
        for item in items:
            span = _item_span(item)
            if item == "__GAP__":
                row_cursor += span
                continue
            cell = gs[row_cursor : row_cursor + span, col_idx]
            row_cursor += span

            if item == "__FLOORPLAN__":
                ax = fig.add_subplot(cell)
                ax.axis("off")
                # A visible (opaque white) axes patch here would paint over
                # any neighboring room's tick labels that happen to dip
                # into this cell's bounding box (e.g. FF03's rotated x-tick
                # labels, which extend a bit below FF03's own axes) —
                # transparent so that can never happen even if the
                # "__GAP__" clearance above/below ever turns out too small.
                ax.patch.set_alpha(0.0)
                if fp_img_path and os.path.exists(fp_img_path):
                    # Default aspect ("equal") — the floor plan's true
                    # proportions, not stretched/skewed to fill the cell.
                    ax.imshow(plt.imread(fp_img_path))
                else:
                    ax.text(0.5, 0.5, "Floor plan not found", ha="center", va="center")
                continue

            if margin is not None:
                # wspace=0: subgridspec's own default (~0.2) would eat into
                # the margin/room split, rendering this room narrower than
                # the width_ratios alone say — which is exactly why FF03/
                # FF05 came out ~12% narrower than the other 6 rooms.
                sub = cell.subgridspec(1, 3, width_ratios=[margin, PK_FLOORPLAN_ROOM_WIDTH, margin], wspace=0.0)
                ax = fig.add_subplot(sub[0, 1])
            else:
                ax = fig.add_subplot(cell)

            sensor_ids = sorted(
                pk_cat.loc[pk_cat["room_group"] == item, "sensor_number"].astype(int).tolist()
            )
            ts_room = (
                sensor_co2_timeseries(df_pk, sensor_ids, align_to, catalog=pk_cat, label_fn=_room_sensor_label)
                if sensor_ids else pd.DataFrame()
            )
            if len(ts_room) == 0:
                ax.axis("off")
                ax.text(0.5, 0.5, f"{item}\n(no data)", ha="center", va="center", fontsize=11, fontweight="bold")
                continue
            plot_room_sensors_matplotlib(
                ts_room, item, stage_defs, plot_start, plot_end, y_range=y_range, ax=ax,
            )

    return fig


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


def vertical_profile_means(df_region: pd.DataFrame, t0, t1, value_col: str, *, inclusive_end: bool = True) -> pd.DataFrame:
    if df_region is None or len(df_region) == 0:
        return pd.DataFrame(columns=["z_level", "z_label", "mean"])

    d = df_region.copy()
    end_mask = (d["time"] <= pd.Timestamp(t1)) if inclusive_end else (d["time"] < pd.Timestamp(t1))
    d = d[(d["time"] >= pd.Timestamp(t0)) & end_mask].copy()
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
# These Matplotlib figures are expensive to rebuild (they redraw every point
# in the aligned time series) and are only ever displayed when Plotly is
# unavailable or the user opens the Export tab — but Streamlit reruns this
# whole script on *every* widget interaction, anywhere on any tab. Caching
# them means a plot-option tweak on an unrelated tab no longer silently
# rebuilds all four figures in the background on every rerun.
@st.cache_resource(show_spinner=False)
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
    export_mode: bool = False,
    y_overrides: Optional[Dict[str, Tuple[float, float]]] = None,
    use_fixed_y: Optional[bool] = None,
):
    lw_c = float(line_width) * 1.5
    lw_p = float(line_width) * 1.0
    lw_dt = float(line_width) * 1.0
    # Export figures use larger, fixed font sizes regardless of whatever the
    # interactive dashboard's font widgets happen to be set to — legibility
    # in a PPT slide shouldn't depend on an on-screen setting.
    fs_axis = 15 if export_mode else 12
    fs_dt = 14 if export_mode else 11
    fig, axs = plt.subplots(
        5, 2, figsize=(18, 15.5), sharex=True,
        gridspec_kw={"hspace": 0.32, "wspace": 0.22} if export_mode else {},
    )

    # The Coverage label carries the exact threshold formula on-screen (the
    # rest of the dashboard has room for it); the export version keeps just
    # "Coverage (%)" so the rotated y-label isn't so long it collides with
    # the row above/below it — the threshold factor is still in the sidebar
    # and summary table.
    coverage_label = "Coverage (%)" if export_mode else f"Coverage (CO₂ ≥ baseline×{cfg.coverage_factor:.2f})"
    titles_co2 = ["Mean CO₂", "Std CO₂", "CV (CO₂)", "Mixing Index (CO₂)", coverage_label]
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
    ax_dt.set_ylabel("ΔT (°C)", fontsize=fs_dt, fontweight="bold")
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
            if export_mode:
                ax.grid(True, color="0.85", linewidth=0.6)
                ax.set_axisbelow(True)
            else:
                ax.grid(True)
            if stage_defs:
                add_stage_shading(ax, stage_defs, stage_patches)

    for i in range(5):
        axs[i, 0].set_ylabel(titles_co2[i], fontsize=fs_axis, fontweight="bold")
        axs[i, 1].set_ylabel(titles_T[i], fontsize=fs_axis, fontweight="bold")

    for ax in axs[-1, :]:
        ax.set_xlabel("Time", fontsize=fs_axis, fontweight="bold")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

    if plot_start is not None and plot_end is not None:
        for ax_row in axs:
            for ax in ax_row:
                ax.set_xlim(plot_start, plot_end)

    plt.setp(axs[-1, 0].get_xticklabels(), rotation=45)
    plt.setp(axs[-1, 1].get_xticklabels(), rotation=45)

    leg_fs = 16 if export_mode else max(5, min(24, int(legend_fontsize)))

    # R²/Coverage/Temp-MI have a genuine fixed domain in practice: R² of an
    # OLS fit on its own training data is always in [0,1]; Coverage is a %
    # of samples; Temp Mixing Index (unlike CO2 MI) stays comfortably inside
    # [0,1] for real data (temperature is far more spatially uniform than
    # CO2 during a release, so its CV never approaches 1). These three
    # always default to fixed 0-1/0-100, independent of the master "Use
    # fixed y-limits" toggle below — still overridable by typing different
    # min/max into that panel's own Y-axis limits box.
    #
    # CO2 Mixing Index = 1 - CV does NOT get this treatment: CO2's CV
    # regularly exceeds 1 right as a release starts (real PK MI here ranges
    # -1.29 to 0.98), so it follows the master toggle/auto-fit like every
    # other concentration/temperature/delta panel — a fixed [0,1] clamp
    # would silently clip it.
    y = cfg.ylims
    _always_fixed = {"co2_coverage", "temp_r2", "temp_mi"}

    def _panel_ylim(key: str, *series) -> Tuple[float, float]:
        if key in _always_fixed:
            src = y_overrides if (export_mode and y_overrides) else y
            return src.get(key, y[key])
        if export_mode and use_fixed_y is not None:
            if use_fixed_y:
                src = y_overrides if y_overrides else y
                return src.get(key, y[key])
            return _auto_ylim(*series) or y[key]
        return _auto_ylim(*series) or y[key]

    axs[0, 0].set_ylim(*_panel_ylim("co2_mean", co2_cave["mean"], co2_pk["mean"]))
    axs[1, 0].set_ylim(*_panel_ylim("co2_std", co2_cave["std"], co2_pk["std"]))
    axs[2, 0].set_ylim(*_panel_ylim("co2_cv", co2_cave["cv"], co2_pk["cv"]))
    axs[3, 0].set_ylim(*_panel_ylim("co2_mi", co2_cave["mi"], co2_pk["mi"]))
    axs[4, 0].set_ylim(*_panel_ylim("co2_coverage", co2_cave["coverage"], co2_pk["coverage"]))

    axs[0, 1].set_ylim(*_panel_ylim("temp_mean", temp_cave["mean_T"], temp_pk["mean_T"]))
    axs[1, 1].set_ylim(*_panel_ylim("temp_std", temp_cave["std_T"], temp_pk["std_T"]))
    axs[2, 1].set_ylim(*_panel_ylim("temp_deltaT", temp_cave["deltaT"], temp_pk["deltaT"]))
    axs[3, 1].set_ylim(*_panel_ylim("temp_r2", temp_cave["r2_Tz"], temp_pk["r2_Tz"]))
    axs[4, 1].set_ylim(*_panel_ylim("temp_mi", temp_cave["mi_T"], temp_pk["mi_T"]))
    axs[0, 1]._ax_dt.set_ylim(*_panel_ylim("temp_pk_minus_cave", deltaT_pk_minus_cave))

    # Finalize the layout *before* placing the figure-level legend, so its
    # anchor (computed from each axes' real post-layout position) is correct.
    if export_mode:
        plt.tight_layout()
    else:
        plt.suptitle(f"{cfg.exp_code} — Overall metrics (CAVE vs PK)", fontsize=14, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.96])

    h1, l1 = axs[0, 1].get_legend_handles_labels()
    h2, l2 = axs[0, 1]._ax_dt.get_legend_handles_labels()
    if export_mode:
        # Every one of the 10 panels reuses the same colour/linestyle
        # convention (solid = CAVE, dashed = PK), plus one ΔT series and the
        # stage shading — so one shared legend for the whole figure covers
        # it, instead of a separate box per panel.
        h_cave, h_pk = axs[0, 0].get_legend_handles_labels()[0][:2]
        all_handles = [h_cave, h_pk] + h2 + list(stage_patches)
        all_labels = ["CAVE", "PK", "ΔT (PK − CAVE)"] + [p.get_label() for p in stage_patches]
        _export_figlegend(fig, all_handles, all_labels, where="bottom", fontsize=leg_fs, anchor_ax=[axs[-1, 0], axs[-1, 1]])
    else:
        axs[0, 1].legend(h1 + h2, l1 + l2, loc="upper right", frameon=True, fontsize=leg_fs)
        axs[0, 0].legend(loc="upper right", frameon=True, fontsize=leg_fs)

    return fig


@st.cache_resource(show_spinner=False)
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
    export_mode: bool = False,
    cave_y_range: Optional[Tuple[float, float]] = None,
    pk_y_range: Optional[Tuple[float, float]] = None,
):
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 9), sharex=True,
        gridspec_kw={"hspace": 0.12 if export_mode else None},
    )
    fs_axis = 18 if export_mode else 12
    lfc = 15 if export_mode else max(5, min(24, int(cave_legend_fs)))
    lfp = 15 if export_mode else max(5, min(24, int(pk_legend_fs)))

    if export_mode:
        ax1.set_prop_cycle(color=_journal_colors(max(len(cave_zone_co2.columns), 1)))
        ax2.set_prop_cycle(color=_journal_colors(max(len(pk_zone_co2.columns), 1)))

    stage_patches_c: list = []
    for col in cave_zone_co2.columns:
        ax1.plot(cave_zone_co2.index, cave_zone_co2[col].values, linewidth=cave_line_width, label=col)

    if stage_defs:
        add_stage_shading(ax1, stage_defs, stage_patches_c)

    if not export_mode:
        ax1.set_title(f"{cfg.exp_code} — CAVE selected walls mean CO₂", fontsize=13, fontweight="bold")
    ax1.set_ylabel("CO₂ (ppm)", fontsize=fs_axis, fontweight="bold")
    ax1.grid(True, color="0.85", linewidth=0.6)
    ax1.set_axisbelow(True)
    if not export_mode:
        ax1.legend(fontsize=lfc, frameon=True, loc="upper right")

    stage_patches_b: list = []
    for col in pk_zone_co2.columns:
        ax2.plot(pk_zone_co2.index, pk_zone_co2[col].values, linewidth=pk_line_width, label=col)

    if stage_defs:
        add_stage_shading(ax2, stage_defs, stage_patches_b)

    if not export_mode:
        ax2.set_title(f"{cfg.exp_code} — PK zones mean CO₂ (by wall)", fontsize=13, fontweight="bold")
    ax2.set_ylabel("CO₂ (ppm)", fontsize=fs_axis, fontweight="bold")
    ax2.set_xlabel("Time", fontsize=fs_axis, fontweight="bold")
    ax2.grid(True, color="0.85", linewidth=0.6)
    ax2.set_axisbelow(True)
    if not export_mode:
        ax2.legend(ncol=4, fontsize=lfp, frameon=True, loc="upper right")

    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    plt.xticks(rotation=45)

    if plot_start is not None and plot_end is not None:
        ax1.set_xlim(plot_start, plot_end)

    # In export_mode, mirror whatever the matching interactive panel is
    # currently showing (its "use fixed y-limits" state), rather than always
    # recomputing independently — so if you've zoomed/customised on screen,
    # the downloaded figure matches what you're looking at.
    ax1.set_ylim(*(cave_y_range if export_mode else None) or (_auto_ylim(*[cave_zone_co2[c] for c in cave_zone_co2.columns]) or cfg.ylims["zone_cave_co2"]))
    ax2.set_ylim(*(pk_y_range if export_mode else None) or (_auto_ylim(*[pk_zone_co2[c] for c in pk_zone_co2.columns]) or cfg.ylims["zone_pk_co2"]))

    # Finalize layout *before* placing the figure-level legends, so their
    # anchors (computed from each axes' real post-layout position) are correct.
    plt.tight_layout()

    if export_mode:
        # Different series per panel (CAVE walls vs PK sensors) — keep two
        # separate legends so it's unambiguous which belongs to which chart:
        # top panel's legend at the very top of the figure, bottom panel's
        # legend at the very bottom. Stage shading is common to both panels,
        # so it's repeated in both legends rather than only the top one.
        h_top, l_top = ax1.get_legend_handles_labels()
        h_top = h_top + list(stage_patches_c)
        l_top = l_top + [p.get_label() for p in stage_patches_c]
        _export_figlegend(fig, h_top, l_top, where="top", fontsize=lfc, anchor_ax=ax1)
        h_bot, l_bot = ax2.get_legend_handles_labels()
        h_bot = h_bot + list(stage_patches_b)
        l_bot = l_bot + [p.get_label() for p in stage_patches_b]
        _export_figlegend(fig, h_bot, l_bot, where="bottom", fontsize=lfp, anchor_ax=ax2)

    return fig


@st.cache_resource(show_spinner=False)
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
    export_mode: bool = False,
    cave_y_range: Optional[Tuple[float, float]] = None,
    pk_y_range: Optional[Tuple[float, float]] = None,
):
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 9), sharex=True,
        gridspec_kw={"hspace": 0.12 if export_mode else None},
    )
    fs_axis = 18 if export_mode else 12
    lfc = 15 if export_mode else max(5, min(24, int(cave_legend_fs)))
    lfp = 15 if export_mode else max(5, min(24, int(pk_legend_fs)))

    if export_mode:
        ax1.set_prop_cycle(color=_journal_colors(max(len(cave_zone_temp.columns), 1)))
        ax2.set_prop_cycle(color=_journal_colors(max(len(pk_zone_temp.columns), 1)))

    stage_patches_c: list = []
    for col in cave_zone_temp.columns:
        ax1.plot(cave_zone_temp.index, cave_zone_temp[col].values, linewidth=cave_line_width, label=col)

    if stage_defs:
        add_stage_shading(ax1, stage_defs, stage_patches_c)

    if not export_mode:
        ax1.set_title(f"{cfg.exp_code} — CAVE selected walls mean temperature", fontsize=13, fontweight="bold")
    ax1.set_ylabel("Temperature (°C)", fontsize=fs_axis, fontweight="bold")
    if export_mode:
        ax1.grid(True, color="0.85", linewidth=0.6)
        ax1.set_axisbelow(True)
    else:
        ax1.grid(True)
    if not export_mode:
        ax1.legend(fontsize=lfc, frameon=True, loc="upper right")

    stage_patches_b: list = []
    for col in pk_zone_temp.columns:
        ax2.plot(pk_zone_temp.index, pk_zone_temp[col].values, linewidth=pk_line_width, label=col)

    if stage_defs:
        add_stage_shading(ax2, stage_defs, stage_patches_b)

    if not export_mode:
        ax2.set_title(f"{cfg.exp_code} — PK zones mean temperature (by wall)", fontsize=13, fontweight="bold")
    ax2.set_ylabel("Temperature (°C)", fontsize=fs_axis, fontweight="bold")
    ax2.set_xlabel("Time", fontsize=fs_axis, fontweight="bold")
    if export_mode:
        ax2.grid(True, color="0.85", linewidth=0.6)
        ax2.set_axisbelow(True)
    else:
        ax2.grid(True)
    if not export_mode:
        ax2.legend(ncol=4, fontsize=lfp, frameon=True, loc="upper right")

    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    plt.xticks(rotation=45)

    if plot_start is not None and plot_end is not None:
        ax1.set_xlim(plot_start, plot_end)

    # In export_mode, mirror whatever the matching interactive panel is
    # currently showing, rather than always auto-fitting independently.
    if export_mode:
        ax1.set_ylim(*(cave_y_range or _auto_ylim(*[cave_zone_temp[c] for c in cave_zone_temp.columns]) or (0.0, 1.0)))
        ax2.set_ylim(*(pk_y_range or _auto_ylim(*[pk_zone_temp[c] for c in pk_zone_temp.columns]) or (0.0, 1.0)))

    # Finalize layout *before* placing the figure-level legends, so their
    # anchors (computed from each axes' real post-layout position) are correct.
    plt.tight_layout()

    if export_mode:
        h_top, l_top = ax1.get_legend_handles_labels()
        h_top = h_top + list(stage_patches_c)
        l_top = l_top + [p.get_label() for p in stage_patches_c]
        _export_figlegend(fig, h_top, l_top, where="top", fontsize=lfc, anchor_ax=ax1)
        h_bot, l_bot = ax2.get_legend_handles_labels()
        h_bot = h_bot + list(stage_patches_b)
        l_bot = l_bot + [p.get_label() for p in stage_patches_b]
        _export_figlegend(fig, h_bot, l_bot, where="bottom", fontsize=lfp, anchor_ax=ax2)

    return fig


@st.cache_resource(show_spinner=False)
def plot_mfc(mfc_df, t_on, t_off, t_rel0, t_rel1, cfg: AppConfig, *, line_width: float = 2.2, legend_fontsize: int = 10, export_mode: bool = False, x_range: Optional[Tuple[Any, Any]] = None, y_range: Optional[Tuple[float, float]] = None):
    lw = float(line_width)
    leg_fs = 15 if export_mode else max(5, min(24, int(legend_fontsize)))
    fs_axis = 18 if export_mode else 12
    has_temp = mfc_has_temperature(mfc_df)
    fig, ax = plt.subplots(figsize=(14, 5))
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
        ax2.set_ylabel("Temperature (°C)", fontsize=fs_axis, fontweight="bold", color="#d62728")
        ax2.tick_params(axis="y", labelcolor="#d62728")

    if (t_on is not None) and (t_off is not None):
        ax.axvspan(t_on, t_off, alpha=0.15, color="green", label="Detected release (F>TH)")

    if (t_rel0 is not None) and (t_rel1 is not None):
        ax.axvspan(t_rel0, t_rel1, alpha=0.10, color="orange", label="Stage2 (Release)")

    if not export_mode:
        title = f"{cfg.exp_code} — MFC Release Quicklook"
        if has_temp:
            title += " (flow + temperature)"
        ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_ylabel("Flow (MFC units)", fontsize=fs_axis, fontweight="bold", color="#1f77b4")
    ax.tick_params(axis="y", labelcolor="#1f77b4")
    ax.set_xlabel("Time", fontsize=fs_axis, fontweight="bold")
    if export_mode:
        ax.grid(True, color="0.85", linewidth=0.6)
        ax.set_axisbelow(True)
    else:
        ax.grid(True)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    plt.xticks(rotation=45)

    # In export_mode, mirror whatever the interactive MFC panel is currently
    # showing (its lock-to-release / custom-y-limits state) rather than
    # always locking to the release window.
    if export_mode and x_range is not None and x_range[0] is not None and x_range[1] is not None:
        ax.set_xlim(*x_range)
    elif (t_rel0 is not None) and (t_rel1 is not None):
        ax.set_xlim(t_rel0, t_rel1)
    if export_mode and y_range is not None:
        ax.set_ylim(*y_range)

    # Finalize layout *before* placing the figure-level legend, so its
    # anchor (computed from the axes' real post-layout position) is correct.
    plt.tight_layout()

    h1, l1 = ax.get_legend_handles_labels()
    if ax2 is not None:
        h2, l2 = ax2.get_legend_handles_labels()
    else:
        h2, l2 = [], []
    if export_mode:
        _export_figlegend(fig, h1 + h2, l1 + l2, where="bottom", fontsize=leg_fs, anchor_ax=ax)
    elif ax2 is not None:
        ax.legend(h1 + h2, l1 + l2, frameon=True, fontsize=leg_fs, loc="upper right")
    else:
        ax.legend(frameon=True, fontsize=leg_fs, loc="upper right")
    return fig


@st.cache_resource(show_spinner=False)
def plot_humidity_export(
    rh_cave,
    rh_pk,
    stage_defs,
    cfg: AppConfig,
    plot_start,
    plot_end,
    *,
    line_width: float = 2.0,
    mean_y_range: Optional[Tuple[float, float]] = None,
    std_y_range: Optional[Tuple[float, float]] = None,
):
    """Report-ready CAVE vs PK humidity overview: no title, external legend,
    fonts/colours matching the other export figures. Export-only (the
    interactive on-screen chart is plot_humidity_overview_plotly), so unlike
    plot_overall_metrics etc. there's no export_mode switch here."""
    lw_c = float(line_width) * 1.5
    lw_p = float(line_width) * 1.0
    fs_axis = 16
    fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"hspace": 0.15})

    axs[0].plot(rh_cave["mean"].index, rh_cave["mean"].values, linewidth=lw_c, label="CAVE mean RH")
    axs[0].plot(rh_pk["mean"].index, rh_pk["mean"].values, linewidth=lw_p, linestyle="--", label="PK mean RH")
    axs[1].plot(rh_cave["std"].index, rh_cave["std"].values, linewidth=lw_c, label="CAVE std RH")
    axs[1].plot(rh_pk["std"].index, rh_pk["std"].values, linewidth=lw_p, linestyle="--", label="PK std RH")

    stage_patches: list = []
    for ax in axs:
        ax.grid(True, color="0.85", linewidth=0.6)
        ax.set_axisbelow(True)
        if stage_defs:
            add_stage_shading(ax, stage_defs, stage_patches)

    axs[0].set_ylabel("Mean RH (%)", fontsize=fs_axis, fontweight="bold")
    axs[1].set_ylabel("Std RH (%)", fontsize=fs_axis, fontweight="bold")
    axs[1].set_xlabel("Time", fontsize=fs_axis, fontweight="bold")
    axs[1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    plt.setp(axs[1].get_xticklabels(), rotation=45)

    if plot_start is not None and plot_end is not None:
        for ax in axs:
            ax.set_xlim(plot_start, plot_end)

    axs[0].set_ylim(*(mean_y_range or _auto_ylim(rh_cave["mean"], rh_pk["mean"]) or cfg.ylims["rh_mean"]))
    axs[1].set_ylim(*(std_y_range or _auto_ylim(rh_cave["std"], rh_pk["std"]) or cfg.ylims["rh_std"]))

    # Finalize layout *before* placing the figure-level legend, so its
    # anchor (computed from the axes' real post-layout position) is correct.
    plt.tight_layout()

    h_cave, h_pk = axs[0].get_legend_handles_labels()[0][:2]
    all_handles = [h_cave, h_pk] + list(stage_patches)
    all_labels = ["CAVE", "PK"] + [p.get_label() for p in stage_patches]
    _export_figlegend(fig, all_handles, all_labels, where="bottom", fontsize=14, anchor_ax=axs[1])
    return fig


@st.cache_resource(show_spinner=False)
def plot_vertical_profiles_export(
    cave_profiles,
    pk_profiles,
    x_label: str,
    cave_x_range,
    pk_x_range,
    cave_y_range,
    pk_y_range,
    *,
    line_width: float = 2.2,
    marker_size: float = 7.0,
):
    """Report-ready CAVE|PK vertical profile pair (W1-W5 within the selected
    stage), no title, one shared external legend — mirrors the two-panel
    layout of plot_zone_co2/plot_zone_temp. Export-only: the on-screen
    version is plot_vertical_profiles_plotly (or the matplotlib fallback),
    each drawn as 4 separate single-region panels."""
    # Tall/narrow panels, matching the on-screen profile charts' own shape —
    # each of the two columns keeps a 1:2.5 width:height ratio.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 9.0))
    fs_axis = 16
    colors = _journal_colors(max(len(cave_profiles), len(pk_profiles), 1))
    z_ticks = list(range(0, 11))
    z_labels = [f"z{i}" if i > 0 else "z0" for i in range(0, 11)]

    for ax, profiles, xr, yr in [
        (ax1, cave_profiles, cave_x_range, cave_y_range),
        (ax2, pk_profiles, pk_x_range, pk_y_range),
    ]:
        ax.set_prop_cycle(color=colors)
        for label, dfp in profiles:
            if dfp is None or len(dfp) == 0:
                continue
            ax.plot(
                dfp["mean"].values, dfp["z_level"].values, marker="o",
                linewidth=float(line_width), markersize=float(marker_size), label=label,
            )
        ax.grid(True, color="0.85", linewidth=0.6)
        ax.set_axisbelow(True)
        ax.set_xlabel(x_label, fontsize=fs_axis, fontweight="bold")
        if xr is not None:
            ax.set_xlim(xr[0], xr[1])
        ax.set_yticks(z_ticks)
        ax.set_yticklabels(z_labels)
        if yr is not None:
            ax.set_ylim(yr[0], yr[1])

    ax1.set_ylabel("z slice", fontsize=fs_axis, fontweight="bold")

    # Finalize layout *before* placing the figure-level legend, so its
    # anchor (computed from each axes' real post-layout position) is correct.
    plt.tight_layout()

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    seen: set = set()
    h_all, l_all = [], []
    for hh, ll in list(zip(h1, l1)) + list(zip(h2, l2)):
        if ll not in seen:
            seen.add(ll)
            h_all.append(hh)
            l_all.append(ll)
    _export_figlegend(fig, h_all, l_all, where="bottom", fontsize=13, anchor_ax=[ax1, ax2])
    return fig


def plot_io_ratio(io_ex, infiltration_factor, t_rel0, t_rel1, t_base0, t_base1, ex_thresh, cfg: AppConfig,
                  *, src_label: str = "CAVE", rcv_label: str = "PK",
                  window_label: str = "Release window", export_mode: bool = False):
    fig, ax = plt.subplots(figsize=(12, 5) if export_mode else (14, 5))
    ax.plot(io_ex.index, io_ex.values, linewidth=2.0,
            label=f"ratio(t) = {rcv_label}_ex / {src_label}_ex (thresholded)")
    ax.axvspan(t_rel0, t_rel1, alpha=0.15, label=window_label)

    if np.isfinite(infiltration_factor):
        ax.axhline(infiltration_factor, linestyle="--", linewidth=2.0, label=f"mean = {infiltration_factor:.3f}")

    ax.axvspan(t_base0, t_base1, alpha=0.08, label="Baseline window")
    ax.text(0.01, 0.02, f"Threshold: {src_label}_ex > {ex_thresh:.1f} ppm", transform=ax.transAxes, fontsize=9, va="bottom", ha="left")

    if not export_mode:
        ax.set_title(f"{cfg.exp_code} — Excess transfer ratio ({src_label} → {rcv_label})", fontsize=12, fontweight="bold")
    _fs = 16 if export_mode else 12
    ax.set_ylabel("Transfer ratio (-)", fontsize=_fs, fontweight="bold")
    ax.set_xlabel("Time", fontsize=_fs, fontweight="bold")
    ax.grid(True, color="0.85", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.set_xlim(t_rel0, t_rel1)

    if cfg.use_fixed_ylims:
        ax.set_ylim(*cfg.ylims["io_ex"])

    plt.xticks(rotation=45)
    if export_mode:
        plt.tight_layout()
        h, l = ax.get_legend_handles_labels()
        _export_figlegend(fig, h, l, where="bottom", fontsize=13, anchor_ax=ax)
    else:
        ax.legend(frameon=True, fontsize=10, loc="upper left")
        plt.tight_layout()
    return fig


def plot_scatter(df_sc, slope, intercept, r2, cfg: AppConfig,
                 *, src_label: str = "CAVE", rcv_label: str = "PK", export_mode: bool = False):
    fig, ax = plt.subplots(figsize=(8.0, 6.5) if export_mode else (6.5, 5.5))
    ax.scatter(df_sc["cave_ex"].values, df_sc["pk_ex"].values, s=25, alpha=0.8, label="Release points (thresholded)")

    if np.isfinite(slope):
        xline = np.array([df_sc["cave_ex"].min(), df_sc["cave_ex"].max()])
        yline = intercept + slope * xline
        ax.plot(xline, yline, linewidth=2.0, linestyle="--", label=f"Fit: slope={slope:.3f}, R²={r2:.3f}")

    if not export_mode:
        ax.set_title(f"{cfg.exp_code} — {rcv_label}_ex vs {src_label}_ex (Release only)", fontsize=12, fontweight="bold")
    _fs = 16 if export_mode else 12
    ax.set_xlabel(f"{src_label}_ex (ppm)", fontsize=_fs, fontweight="bold")
    ax.set_ylabel(f"{rcv_label}_ex (ppm)", fontsize=_fs, fontweight="bold")
    ax.grid(True, color="0.85", linewidth=0.6)
    ax.set_axisbelow(True)

    if cfg.use_fixed_ylims and not export_mode:
        ax.set_xlim(*cfg.ylims["scatter_cave_ex"])
        ax.set_ylim(*cfg.ylims["scatter_pk_ex"])

    if export_mode:
        plt.tight_layout()
        h, l = ax.get_legend_handles_labels()
        _export_figlegend(fig, h, l, where="bottom", fontsize=13, anchor_ax=ax)
    else:
        ax.legend(frameon=True, fontsize=9, loc="upper left")
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


def render_window_picker(prefix: str, stage_defs, t_lo, t_hi, default_keywords: Tuple[str, ...],
                         *, stage_help: str = "") -> Tuple[Any, Any, str]:
    """Time-window chooser: pick a logged stage, or set the window by hand.

    Stage boundaries in the experiment log are nominal. The moment a decay
    really starts — mixing switched off, release valve closed — often sits
    minutes away from the logged time and has to be read off the curves, so
    manual mode seeds itself from the chosen stage and lets it be nudged.

    Returns (t0, t1, note).
    """
    stage_names = [str(n) for (n, _, _, _) in stage_defs] if stage_defs else []

    def _stage_by_name(name):
        return next(((n, pd.Timestamp(s), pd.Timestamp(e), c)
                     for (n, s, e, c) in (stage_defs or []) if str(n) == str(name)), None)

    default_stage = None
    for kw in default_keywords:
        default_stage = find_stage_by_keyword(stage_defs, kw)
        if default_stage is not None:
            break
    if default_stage is None and stage_defs:
        n, s, e, c = stage_defs[0]
        default_stage = (n, pd.Timestamp(s), pd.Timestamp(e), c)

    modes = ["Stage", "Manual"] if stage_names else ["Manual"]
    mode = st.radio("Window", options=modes, index=0, horizontal=True, key=f"{prefix}__win_mode")

    if mode == "Stage":
        idx = 0
        if default_stage is not None:
            try:
                idx = stage_names.index(str(default_stage[0]))
            except ValueError:
                idx = 0
        pick = st.selectbox("Stage", options=stage_names, index=idx,
                            key=f"{prefix}__stage", help=stage_help)
        chosen = _stage_by_name(pick)
        if chosen is None:
            return None, None, "stage not found"
        return chosen[1], chosen[2], f"stage: {chosen[0]}"

    # ---- Manual ----------------------------------------------------------
    lo = pd.Timestamp(t_lo).to_pydatetime()
    hi = pd.Timestamp(t_hi).to_pydatetime()

    # Seed from whichever stage is currently selected, so switching to Manual
    # starts from where you were rather than jumping somewhere unrelated.
    seed = _stage_by_name(st.session_state.get(f"{prefix}__stage")) or default_stage
    if seed is not None:
        a = max(lo, seed[1].to_pydatetime())
        b = min(hi, seed[2].to_pydatetime())
    else:
        a, b = lo, hi
    if b <= a:
        a, b = lo, hi

    typed = st.checkbox("Type exact timestamps instead of dragging", key=f"{prefix}__typed")
    if typed:
        # Carry over wherever the slider was left, so switching to typed entry
        # fine-tunes the current window instead of resetting it to the stage.
        dragged = st.session_state.get(f"{prefix}__manual")
        if isinstance(dragged, (tuple, list)) and len(dragged) == 2:
            a, b = dragged[0], dragged[1]
    if typed:
        c1, c2 = st.columns(2)
        with c1:
            s_txt = st.text_input("Start", value=f"{a:%Y-%m-%d %H:%M:%S}", key=f"{prefix}__t0_txt")
        with c2:
            e_txt = st.text_input("End", value=f"{b:%Y-%m-%d %H:%M:%S}", key=f"{prefix}__t1_txt")
        t0 = pd.to_datetime(s_txt, errors="coerce")
        t1 = pd.to_datetime(e_txt, errors="coerce")
        if pd.isna(t0) or pd.isna(t1) or t1 <= t0:
            st.error("Could not read those timestamps (expected `YYYY-MM-DD HH:MM:SS`, end after start) — falling back to the stage window.")
            return pd.Timestamp(a), pd.Timestamp(b), "manual (invalid input, stage window used)"
        return t0, t1, "manual (typed)"

    span = hi - lo
    step = _dt.timedelta(minutes=1) if span > _dt.timedelta(hours=2) else _dt.timedelta(seconds=10)
    sel = st.slider(
        "Drag the ends to set the window",
        min_value=lo, max_value=hi, value=(a, b), step=step,
        format="YYYY-MM-DD HH:mm", key=f"{prefix}__manual",
    )
    return pd.Timestamp(sel[0]), pd.Timestamp(sel[1]), "manual (slider)"


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
    # Default to auto-scaled y, not the fixed cfg.ylims defaults — those are
    # generic guesses and can silently clip a real release (e.g. CO2 mean
    # above 1300 ppm). The export PNG now mirrors this panel's own setting,
    # so a fresh session's downloaded figure should default to "safe" too;
    # still fully overridable on screen if you want a fixed scale.
    "use_fixed_y": False,
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


def plot_io_ratio_plotly(io_ex, infiltration_factor, t_rel0, t_rel1, t_base0, t_base1, ex_thresh, cfg: AppConfig,
                         *, src_label: str = "CAVE", rcv_label: str = "PK",
                         window_label: str = "Release",
                         x_range: Optional[Tuple[Any, Any]] = None,
                         y_range: Optional[Tuple[float, float]] = None):
    _require_plotly()
    fig = go.Figure()

    s = io_ex.dropna()
    fig.add_trace(
        go.Scatter(
            x=s.index,
            y=s.values,
            mode="lines+markers",
            name=f"ratio(t) = {rcv_label}_ex / {src_label}_ex (thresholded)",
            line=dict(width=2),
            marker=dict(size=5, opacity=0.6),
            hovertemplate="t=%{x|%Y-%m-%d %H:%M:%S}<br>ratio=%{y:.4f}<extra></extra>",
        )
    )

    # Shaded windows
    if (t_rel0 is not None) and (t_rel1 is not None):
        fig.add_vrect(x0=t_rel0, x1=t_rel1, fillcolor="orange", opacity=0.15, line_width=0, annotation_text=window_label, annotation_position="top left")
    if (t_base0 is not None) and (t_base1 is not None):
        fig.add_vrect(x0=t_base0, x1=t_base1, fillcolor="gray", opacity=0.10, line_width=0, annotation_text="Baseline", annotation_position="bottom left")

    # Mean line during release
    if np.isfinite(infiltration_factor):
        fig.add_hline(y=infiltration_factor, line_dash="dash", line_width=2, annotation_text=f"mean={infiltration_factor:.3f}", annotation_position="top right")

    fig.update_layout(
        title=f"{cfg.exp_code} — Excess transfer ratio ({src_label} → {rcv_label})",
        xaxis_title="Time",
        yaxis_title="Transfer ratio (-)",
        template="plotly_white",
        height=420,
        margin=dict(l=40, r=20, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        clickmode="event+select",
    )

    if x_range is not None:
        fig.update_xaxes(range=[x_range[0], x_range[1]])
    elif (t_rel0 is not None) and (t_rel1 is not None):
        fig.update_xaxes(range=[t_rel0, t_rel1])

    if y_range is not None:
        fig.update_yaxes(range=list(y_range))
    elif cfg.use_fixed_ylims:
        fig.update_yaxes(range=list(cfg.ylims["io_ex"]))

    # Threshold note (as annotation)
    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0.01,
        y=0.02,
        showarrow=False,
        text=f"Threshold: {src_label}_ex > {ex_thresh:.1f} ppm",
        font=dict(size=11),
        align="left",
    )

    return fig


def plot_scatter_plotly(df_sc, slope, intercept, r2, cfg: AppConfig,
                        *, src_label: str = "CAVE", rcv_label: str = "PK",
                        auto_range: bool = False):
    _require_plotly()
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df_sc["cave_ex"],
            y=df_sc["pk_ex"],
            mode="markers",
            name="Release points (thresholded)",
            marker=dict(size=7, opacity=0.8),
            hovertemplate=f"{src_label}_ex=%{{x:.2f}} ppm<br>{rcv_label}_ex=%{{y:.2f}} ppm<extra></extra>",
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
        title=f"{cfg.exp_code} — {rcv_label}_ex vs {src_label}_ex (Release only)",
        xaxis_title=f"{src_label}_ex (ppm)",
        yaxis_title=f"{rcv_label}_ex (ppm)",
        template="plotly_white",
        height=520,
        margin=dict(l=40, r=20, t=60, b=40),
        clickmode="event+select",
    )

    if cfg.use_fixed_ylims and not auto_range:
        fig.update_xaxes(range=list(cfg.ylims["scatter_cave_ex"]))
        fig.update_yaxes(range=list(cfg.ylims["scatter_pk_ex"]))

    return fig


# ---- Air-exchange rate (lambda) figures ------------------------------------
def _lam_titles(cfg: AppConfig, src_label: str, rcv_label: str) -> Tuple[str, str, str]:
    # src/rcv here are the driving and solved zones, not the release direction —
    # spelled out so a PK-release run does not look like it is claiming CAVE → PK.
    who = f"λ_{rcv_label}, driven by {src_label}"
    return (
        f"{cfg.exp_code} — {who} | integrated",
        f"{cfg.exp_code} — {who} | full regression",
        f"{cfg.exp_code} — {who} | sliding window",
    )


def plot_lambda_integrated_plotly(res, cfg: AppConfig, src_label: str, rcv_label: str):
    _require_plotly()
    t_int, _, _ = _lam_titles(cfg, src_label, rcv_label)
    fig = go.Figure()
    x, y = res["x"], res["y"]
    fig.add_trace(go.Scatter(
        x=x, y=y, mode="markers", name="Data",
        marker=dict(size=6, opacity=0.8),
        hovertemplate="x=%{x:.4g} ppm·s<br>y=%{y:.2f} ppm<extra></extra>",
    ))
    if np.isfinite(res["lam_h"]) and len(x):
        xs = np.array([float(np.min(x)), float(np.max(x))])
        ys = (res["lam_h"] / 3600.0) * xs + res.get("intercept", 0.0)
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines", hoverinfo="skip",
            name=f"Fit λ={res['lam_h']:.3f} 1/h, R²={res['r2']:.3f}",
            line=dict(width=2, dash="dash"),
        ))
    fig.update_layout(
        title=t_int,
        xaxis_title=f"x = ∫({src_label}_ex − {rcv_label}_ex) dt  [ppm·s]",
        yaxis_title=f"y = {rcv_label}_ex(t) − {rcv_label}_ex(t₀)  [ppm]",
        template="plotly_white", height=480,
        margin=dict(l=50, r=20, t=60, b=45),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


def plot_lambda_full_plotly(res, cfg: AppConfig, src_label: str, rcv_label: str):
    _require_plotly()
    _, t_full, _ = _lam_titles(cfg, src_label, rcv_label)
    fig = go.Figure()
    X, Y = res["X"], res["Y"]
    fig.add_trace(go.Scatter(
        x=X, y=Y, mode="markers", name="Data",
        marker=dict(size=6, opacity=0.7),
        hovertemplate="ΔC=%{x:.1f} ppm<br>dC/dt=%{y:.5f} ppm/s<extra></extra>",
    ))
    if np.isfinite(res["lam_h"]) and len(X):
        xs = np.array([float(np.min(X)), float(np.max(X))])
        fig.add_trace(go.Scatter(
            x=xs, y=(res["lam_h"] / 3600.0) * xs, mode="lines", hoverinfo="skip",
            name=f"Fit λ={res['lam_h']:.3f} 1/h, R²={res['r2']:.3f}",
            line=dict(width=2),
        ))
    fig.update_layout(
        title=t_full,
        xaxis_title=f"X = {src_label}_ex − {rcv_label}_ex  [ppm]",
        yaxis_title=f"Y = d{rcv_label}_ex/dt  [ppm/s]",
        template="plotly_white", height=480,
        margin=dict(l=55, r=20, t=60, b=45),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


def plot_lambda_window_plotly(res, cfg: AppConfig, src_label: str, rcv_label: str,
                              win_min: int, step_min: int,
                              y_range: Optional[Tuple[float, float]] = None):
    _require_plotly()
    _, _, t_win = _lam_titles(cfg, src_label, rcv_label)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=res["times"], y=res["lam_h"], mode="lines+markers", name="λ_window(t)",
        line=dict(width=2.5), marker=dict(size=6),
        hovertemplate="t=%{x|%H:%M}<br>λ=%{y:.4f} 1/h<extra></extra>",
    ))
    if np.isfinite(res["mean_h"]):
        fig.add_hline(y=res["mean_h"], line_dash="dash", line_width=2,
                      annotation_text=f"mean={res['mean_h']:.3f}", annotation_position="top right")
    if np.isfinite(res["median_h"]):
        fig.add_hline(y=res["median_h"], line_dash="dot", line_width=2,
                      annotation_text=f"median={res['median_h']:.3f}", annotation_position="bottom right")
    fig.update_layout(
        title=f"{t_win} | win={win_min} min, step={step_min} min",
        xaxis_title="Time", yaxis_title="λ_window (1/h)",
        template="plotly_white", height=420,
        margin=dict(l=45, r=20, t=60, b=45),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    if y_range is not None:
        fig.update_yaxes(range=list(y_range))
    return fig


def _lam_export_axes(figsize=(9.0, 6.0)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(True, color="0.85", linewidth=0.6)
    ax.set_axisbelow(True)
    return fig, ax


def plot_lambda_integrated_export(res, cfg: AppConfig, other_label: str, solve_label: str):
    """Report-ready integrated-method figure: no title, external legend."""
    fs_axis = 16
    fig, ax = _lam_export_axes()
    ax.plot(res["x"], res["y"], "o", ms=5, alpha=0.85, label="Data")
    if np.isfinite(res["lam_h"]) and len(res["x"]):
        xs = np.array([float(np.min(res["x"])), float(np.max(res["x"]))])
        ax.plot(xs, (res["lam_h"] / 3600.0) * xs + res.get("intercept", 0.0), "-", lw=2.2,
                label=f"Fit: λ = {res['lam_h']:.3f} 1/h, R² = {res['r2']:.3f}")
    ax.set_xlabel(f"∫({other_label}_ex − {solve_label}_ex) dt  [ppm·s]",
                  fontsize=fs_axis, fontweight="bold")
    ax.set_ylabel(f"{solve_label}_ex(t) − {solve_label}_ex(t₀)  [ppm]",
                  fontsize=fs_axis, fontweight="bold")
    plt.tight_layout()
    h, l = ax.get_legend_handles_labels()
    _export_figlegend(fig, h, l, where="bottom", fontsize=13, anchor_ax=ax)
    return fig


def plot_lambda_regression_export(res, cfg: AppConfig, other_label: str, solve_label: str):
    """Report-ready differential full-regression figure."""
    fs_axis = 16
    fig, ax = _lam_export_axes()
    ax.plot(res["X"], res["Y"], "o", ms=5, alpha=0.65, label="Data")
    if np.isfinite(res["lam_h"]) and len(res["X"]):
        xs = np.array([float(np.min(res["X"])), float(np.max(res["X"]))])
        ax.plot(xs, (res["lam_h"] / 3600.0) * xs, "-", lw=2.2,
                label=f"Fit: λ = {res['lam_h']:.3f} 1/h, R² = {res['r2']:.3f}")
    ax.set_xlabel(f"ΔC = {other_label}_ex − {solve_label}_ex  [ppm]",
                  fontsize=fs_axis, fontweight="bold")
    ax.set_ylabel(f"d{solve_label}_ex/dt  [ppm/s]", fontsize=fs_axis, fontweight="bold")
    plt.tight_layout()
    h, l = ax.get_legend_handles_labels()
    _export_figlegend(fig, h, l, where="bottom", fontsize=13, anchor_ax=ax)
    return fig


def plot_lambda_window_export(res, cfg: AppConfig, other_label: str, solve_label: str,
                              win_min: int, step_min: int, y_range=None):
    """Report-ready sliding-window figure."""
    fs_axis = 16
    fig, ax = _lam_export_axes(figsize=(12.0, 5.0))
    ax.plot(res["times"], res["lam_h"], "-o", ms=5, lw=2.2,
            label=f"λ over a {win_min} min window, stepped {step_min} min")
    if np.isfinite(res["mean_h"]):
        ax.axhline(res["mean_h"], ls="--", lw=1.8, label=f"Mean = {res['mean_h']:.3f} 1/h")
    if np.isfinite(res["median_h"]):
        ax.axhline(res["median_h"], ls=":", lw=2.2, label=f"Median = {res['median_h']:.3f} 1/h")
    ax.set_xlabel("Time", fontsize=fs_axis, fontweight="bold")
    ax.set_ylabel(f"λ_{solve_label} (1/h)", fontsize=fs_axis, fontweight="bold")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    plt.setp(ax.get_xticklabels(), rotation=45)
    if y_range is not None:
        ax.set_ylim(*y_range)
    plt.tight_layout()
    h, l = ax.get_legend_handles_labels()
    _export_figlegend(fig, h, l, where="bottom", fontsize=13, anchor_ax=ax)
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

        fig.update_yaxes(range=list(y["temp_mean"]), row=1, col=2, secondary_y=False)
        fig.update_yaxes(range=list(y["temp_std"]), row=2, col=2)
        fig.update_yaxes(range=list(y["temp_deltaT"]), row=3, col=2)
        fig.update_yaxes(range=list(y["temp_pk_minus_cave"]), row=1, col=2, secondary_y=True)

    # Coverage/R²/Temp-MI have a genuine fixed domain (0-100 or 0-1) and stay
    # on that fixed scale regardless of the "Use fixed y-limits" toggle above
    # — CO2 Mixing Index doesn't get this, since MI = 1 - CV and CO2's CV
    # regularly exceeds 1 right as a release starts (real data here goes as
    # low as MI = -1.29), so a [0,1] clamp would silently clip it.
    y_always = yref if yref is not None else cfg.ylims
    fig.update_yaxes(range=list(y_always["co2_coverage"]), row=5, col=1)
    fig.update_yaxes(range=list(y_always["temp_r2"]), row=4, col=2)
    fig.update_yaxes(range=list(y_always["temp_mi"]), row=5, col=2)

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
    """Cheap identity check for 'did the upload change' — avoids hashing the
    full file on every Streamlit rerun. Streamlit's UploadedFile carries a
    stable file_id per upload; name+size is a good enough fallback."""
    if file_obj is None:
        return ""
    file_id = getattr(file_obj, "file_id", None)
    if file_id:
        return f"{getattr(file_obj, 'name', '')}|{file_id}"
    try:
        return f"{getattr(file_obj, 'name', '')}|{file_obj.size}"
    except Exception:
        return f"{getattr(file_obj, 'name', '')}|na"


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

st.sidebar.header("5) Air exchange (PK ↔ CAVE)")

baseline_min_samples = st.sidebar.number_input(
    "Min baseline samples per sensor",
    min_value=1, value=5, step=1,
    help="Sensors with fewer valid points in the baseline window get no per-sensor "
         "baseline and are dropped from the air-exchange analysis (they are NOT "
         "fallen back to the region baseline, which would reintroduce the offset).",
)
noise_sigma_k = st.sidebar.number_input(
    "Noise safeguard k (× σ)", min_value=0.0, value=5.0, step=1.0,
    help="Excess threshold used = max(absolute threshold, k × σ of the baseline).",
)

envelope_walls = st.sidebar.text_input(
    "PK-envelope sensor groups (in CAVE)", value="FFE,GFE",
    help="CAVE-side sensors mounted on the PK exterior wall. They read the interface, "
         "not the room bulk.",
)
exclude_envelope_from_bulk = st.sidebar.checkbox(
    "Keep envelope sensors out of the CAVE bulk", value=True,
    help="Leaving them in both distorts the CAVE bulk mean and correlates it with the "
         "interface series it gets compared against.",
)
dc_min_ppm = st.sidebar.number_input(
    "ΔC threshold (ppm)", min_value=0.0, value=100.0, step=10.0,
    help="The fit uses the longest unbroken stretch where |ΔC| (the gradient across the "
         "two zones) stays above this value and keeps one sign.",
)

_c1, _c2 = st.sidebar.columns(2)
with _c1:
    lam_win_min = st.number_input("λ window (min)", min_value=1, value=15, step=1)
with _c2:
    lam_step_min = st.number_input("λ step (min)", min_value=1, value=5, step=1)

force_zero_intercept = st.sidebar.checkbox(
    "Force zero intercept (integrated)", value=True,
    help="The model y = λx has no constant term; leave on unless diagnosing an offset.",
)

st.sidebar.markdown("**Zone volumes (m³)**")
v_pk = st.sidebar.number_input("V_PK", min_value=0.0, value=455.67, step=1.0)
_vc1, _vc2 = st.sidebar.columns(2)
with _vc1:
    v_cave_gross = st.number_input("V_CAVE gross", min_value=0.0, value=1917.49, step=1.0)
with _vc2:
    v_cave_effective = st.number_input("V_CAVE eff.", min_value=0.0, value=1461.82, step=1.0)
use_effective_cave_volume = st.sidebar.checkbox(
    "Use effective CAVE volume", value=True,
    help="Effective = gross minus the volume occupied by PK. This is the air volume "
         "that actually takes part in the exchange, so it is the correct choice for λ_CAVE.",
)

lambda_ext = st.sidebar.number_input(
    "λ_ext — CAVE ↔ outdoor (1/h)", min_value=0.0, value=0.0, step=0.01, format="%.3f",
    help="Only used when you solve CAVE's mass balance instead of PK's. CAVE also loses "
         "tracer to outdoors and that sink cannot be fitted alongside the PK exchange — the "
         "two are nearly collinear — so it has to be supplied here, from a companion "
         "experiment at a comparable ΔT. Solving PK needs no such term.",
)

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
    v_pk=float(v_pk),
    v_cave_gross=float(v_cave_gross),
    v_cave_effective=float(v_cave_effective),
    use_effective_cave_volume=bool(use_effective_cave_volume),
    baseline_min_samples=int(baseline_min_samples),
    noise_sigma_k=float(noise_sigma_k),
    envelope_walls=split_str_list(envelope_walls),
    exclude_envelope_from_bulk=bool(exclude_envelope_from_bulk),
    dc_min_ppm=float(dc_min_ppm),
    lam_win_min=int(lam_win_min),
    lam_step_min=int(lam_step_min),
    force_zero_intercept=bool(force_zero_intercept),
    lambda_ext=float(lambda_ext),
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
        # Air exchange: per-sensor baselines and excess series
        # -----------------------------
        # Isolated from the rest of the pipeline: if this fails the other tabs
        # must still render, so the error is captured rather than raised.
        ae: Dict[str, Any] = {"ok": False, "error": None}
        try:
            t_base0, t_base1, base_note = find_baseline_window(stage_defs)
            if t_base0 is None or t_base1 is None:
                t_base1 = pd.Timestamp(t_rel0)
                t_base0 = t_base1 - pd.Timedelta(minutes=cfg.baseline_fallback_minutes)
                base_note = f"fallback: {cfg.baseline_fallback_minutes} min before release start"

            base_cave, info_cave = per_sensor_baselines(df_cave, t_base0, t_base1, cfg.baseline_min_samples)
            base_pk, info_pk = per_sensor_baselines(df_pk, t_base0, t_base1, cfg.baseline_min_samples)

            if len(base_cave) == 0 or len(base_pk) == 0:
                raise ValueError(
                    "No sensor has enough samples in the baseline window "
                    f"({t_base0} → {t_base1}). Check the stage log or lower "
                    "'Min baseline samples per sensor'."
                )

            df_cave_inc = add_increment_column(df_cave, base_cave)
            df_pk_inc = add_increment_column(df_pk, base_pk)

            # The PK-envelope sensors sit in CAVE but read the interface, so they are
            # held out of the CAVE bulk mean; keeping them in would both distort the
            # bulk and correlate it with the interface series it is compared against.
            _env = {w.strip().upper() for w in cfg.envelope_walls}
            _is_env = df_cave_inc["wall"].astype(str).str.strip().str.upper().isin(_env)
            n_env_sensors = int(df_cave_inc.loc[_is_env, "sensor_number"].nunique())
            df_cave_bulk = df_cave_inc[~_is_env] if (cfg.exclude_envelope_from_bulk and _is_env.any()) else df_cave_inc

            ex_cave = excess_mean_series(df_cave_bulk, cfg.align_to, cfg.min_sensors)
            ex_pk = excess_mean_series(df_pk_inc, cfg.align_to, cfg.min_sensors)

            noise_cave = excess_noise_stats(df_cave_bulk, ex_cave, t_base0, t_base1, cfg.align_to)
            noise_pk = excess_noise_stats(df_pk_inc, ex_pk, t_base0, t_base1, cfg.align_to)

            direction, ex_cave_rel, ex_pk_rel = detect_exchange_direction(ex_cave, ex_pk, t_rel0, t_rel1)

            ae.update({
                "ok": True,
                "t_base0": t_base0, "t_base1": t_base1, "base_note": base_note,
                "base_cave": base_cave, "base_pk": base_pk,
                "info_cave": info_cave, "info_pk": info_pk,
                "df_cave_inc": df_cave_inc, "df_pk_inc": df_pk_inc,
                "ex_cave": ex_cave, "ex_pk": ex_pk,
                "noise_cave": noise_cave, "noise_pk": noise_pk,
                "direction_auto": direction,
                "ex_cave_rel": ex_cave_rel, "ex_pk_rel": ex_pk_rel,
                "n_env_sensors": n_env_sensors,
                "env_excluded": bool(cfg.exclude_envelope_from_bulk and n_env_sensors),
                "cave_base_mean": float(base_cave.mean()) if len(base_cave) else np.nan,
                "pk_base_mean": float(base_pk.mean()) if len(base_pk) else np.nan,
            })
        except Exception as _ae_err:  # noqa: BLE001 - surfaced in the Air Exchange tab
            ae["error"] = f"{type(_ae_err).__name__}: {_ae_err}"

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
# Note: the tab *variable* names below are out of numerical order on purpose
# — tab9 is the PK Rooms view and tab_ae is Air Exchange; both are defined
# far below, and this ordering lets each render in its intended place without
# relocating those blocks. Streamlit only cares about the position of each
# variable in this list, not what it is named.
tab1, tab2, tab3, tab9, tab4, tab5, tab6, tab_ae, tab7, tab8 = st.tabs(
    [
        "Data Preview",
        "Overall Metrics",
        "Zone CO₂ & Temperature",
        "PK Rooms (Floor Plan)",
        "Sensor CO₂ & Temp",
        "Humidity",
        "Vertical Profiles (Decay)",
        "Air Exchange",
        "MFC (optional)",
        "Export",
    ]
)

# Populated by the Air Exchange tab; the Export tab reads it further down.
ae_export: Optional[Dict[str, Any]] = None

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
        # Default to auto-scaled y (not the fixed 350-1300 default) so a
        # fresh session — and its exported PNG, which now mirrors this panel
        # — doesn't silently clip a release that goes higher, the way the
        # old fixed default did. Still fully overridable on screen.
        dc_cave_co2 = {**zone_ts_page_defaults(cfg.ylims, "zone_cave_co2"), "use_fixed_y": False}
        dc_pk_co2 = {**zone_ts_page_defaults(cfg.ylims, "zone_pk_co2"), "use_fixed_y": False}
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
            legend_in_plot=False,
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
            legend_in_plot=False,
        )
        # Legend moves below the chart (as chips, wrapping/scrolling if long)
        # instead of Plotly's default inside-the-plot legend — with up to 19
        # PK sensors, an in-plot legend covers real data.
        apply_plotly_style(fig_cave_zone_co2, {**_style_from_prefix("zco2_cave"), "show_legend": False})
        apply_plotly_style(fig_pk_zone_co2, {**_style_from_prefix("zco2_pk"), "show_legend": False})
        show_plotly_chart(
            fig_cave_zone_co2, stage_defs, show_stage_legend=False,
            external_series_legend=True, series_legend_title="CAVE walls",
        )
        show_plotly_chart(
            fig_pk_zone_co2, stage_defs,
            external_series_legend=True, series_legend_title="PK zones (by wall)",
        )

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
            legend_in_plot=False,
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
            legend_in_plot=False,
        )
        apply_plotly_style(fig_cave_zone_temp, {**_style_from_prefix("zt_cave"), "show_legend": False})
        apply_plotly_style(fig_pk_zone_temp, {**_style_from_prefix("zt_pk"), "show_legend": False})
        show_plotly_chart(
            fig_cave_zone_temp, stage_defs, show_stage_legend=False,
            external_series_legend=True, series_legend_title="CAVE walls",
        )
        show_plotly_chart(
            fig_pk_zone_temp, stage_defs,
            external_series_legend=True, series_legend_title="PK zones (by wall)",
        )

    st.write("**CAVE zone temperature preview**")
    st.dataframe(cave_zone_temp.head(20), use_container_width=True)

    st.write("**PK zone temperature preview**")
    st.dataframe(pk_zone_temp.head(20), use_container_width=True)

with tab9:
    st.subheader("PK — Rooms by floor plan (sensor-level CO₂)")
    st.write(
        "Every individual CO₂ sensor in a room, plotted on its own (not the zone mean) — laid out "
        "around the real floor plan, matching each room's actual position on it."
    )

    if len(df_pk) == 0:
        st.info("No PK data in the current upload.")
    else:
        pk_cat_fp = pk_cat if "pk_cat" in dir() else sensor_catalog(df_pk)
        pk_cat_fp = pk_cat_fp.copy()
        pk_cat_fp["room_group"] = pk_cat_fp["wall"].apply(_pk_room_group)

        floor_choice = st.radio(
            "Floor", options=["FF — upper floor", "GF — ground floor"],
            horizontal=True, key="pkfp__floor",
        )
        floor_key = "FF" if floor_choice.startswith("FF") else "GF"
        floor_columns = PK_FLOORPLAN_LAYOUT[floor_key]
        fp_img_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "assets", "floorplans", PK_FLOORPLAN_IMAGES[floor_key]
        )

        with st.expander("Plot options (applies to every room on this floor)", expanded=False):
            st.markdown("**X-axis (time)**")
            render_x_mode_widgets("pkfp", t0, t1, stage_defs)
            st.checkbox("Use fixed y-limits (all rooms)", key="pkfp__use_fixed_y", value=False)
            c1, c2 = st.columns(2)
            with c1:
                st.number_input("Y min", key="pkfp__y_min", value=350.0)
            with c2:
                st.number_input("Y max", key="pkfp__y_max", value=2000.0)

        x0_fp, x1_fp = render_x_controls("pkfp", t0, t1, stage_defs)
        use_fy_fp = bool(st.session_state.get("pkfp__use_fixed_y", False))
        y_range_fp = None
        if use_fy_fp:
            y_range_fp = (
                float(st.session_state.get("pkfp__y_min", 350.0)),
                float(st.session_state.get("pkfp__y_max", 2000.0)),
            )

        if stage_defs:
            render_stage_legend_outside(stage_defs)

        def _render_room_chart(room: str) -> None:
            sensor_ids = sorted(
                pk_cat_fp.loc[pk_cat_fp["room_group"] == room, "sensor_number"].astype(int).tolist()
            )
            if not sensor_ids:
                st.info(f"**{room}**: no sensors found.")
                return
            ts_room = sensor_co2_timeseries(
                df_pk, sensor_ids, cfg.align_to, catalog=pk_cat_fp, label_fn=_room_sensor_label,
            )
            if len(ts_room) == 0:
                st.info(f"**{room}**: no data in range.")
                return
            fig_room = plot_room_sensors_matplotlib(
                ts_room, room, stage_defs, x0_fp, x1_fp, y_range=y_range_fp,
            )
            # Every column is the same width (see PK_FLOORPLAN_LAYOUT) and
            # every room chart uses the same figsize, so stretching to fill
            # the column renders all rooms at the same final size.
            st.pyplot(fig_room, use_container_width=True)
            plt.close(fig_room)

        cols = st.columns([c["width"] for c in floor_columns])
        for col_widget, col_spec in zip(cols, floor_columns):
            # A column shared with the floor plan (e.g. FF's middle column,
            # width 1.4663) is wider than a plain room column (width 1.0).
            # use_container_width=True would stretch that column's room
            # charts (FF03/FF05) to the full, wider column — larger than
            # every other room. Nest matching narrow sub-columns, mirroring
            # the download composite's subgridspec margin trick, so a room
            # chart here renders at the same width as any other room.
            col_w = col_spec["width"]
            margin = (col_w - PK_FLOORPLAN_ROOM_WIDTH) / 2
            needs_margin = margin > 1e-6
            with col_widget:
                for item in col_spec["items"]:
                    if item == "__GAP__":
                        # Blank spacer, export-composite-only (keeps rooms
                        # in the same column from crowding each other and
                        # the floor plan there) — nothing to render on
                        # screen, where Streamlit just stacks content with
                        # its own natural spacing.
                        continue
                    if item == "__FLOORPLAN__":
                        if os.path.exists(fp_img_path):
                            st.image(fp_img_path, use_container_width=True)
                        else:
                            st.warning(f"Floor plan image not found at `{fp_img_path}`.")
                    elif needs_margin:
                        _, room_sub, _ = st.columns([margin, PK_FLOORPLAN_ROOM_WIDTH, margin])
                        with room_sub:
                            _render_room_chart(item)
                    else:
                        _render_room_chart(item)

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
    # Set once the profile data + per-panel x/y ranges are actually computed
    # below — the Export tab reads this rather than the profile variables
    # directly, since they're only assigned inside the "stage selected and
    # windows valid" branch further down (same script run, so plain globals
    # are visible there, but only once that branch has actually executed).
    vertical_profiles_ready = False
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
                    labels.append((f"W{i}", a, b, i == len(windows)))

                _win_tbl = pd.DataFrame(
                    [
                        {
                            "Legend": lab,
                            "Window": f"{i} of 5",
                            "Start (inclusive)": pd.Timestamp(a),
                            "End": f"{pd.Timestamp(b)}" + (" (inclusive)" if last else " (exclusive)"),
                        }
                        for i, (lab, a, b, last) in enumerate(labels, start=1)
                    ]
                )
                st.write(
                    "**Time windows for W1–W5** (each line on the plots uses only data in that interval; "
                    "windows are back-to-back, so each window's end is exclusive except the final one — "
                    "a reading exactly on a boundary is never double-counted)"
                )
                st.dataframe(_win_tbl, use_container_width=True, hide_index=True)

                pk_co2_profiles = [(lab, vertical_profile_means(df_pk, a, b, "co2", inclusive_end=last)) for (lab, a, b, last) in labels]
                cave_co2_profiles = [(lab, vertical_profile_means(df_cave, a, b, "co2", inclusive_end=last)) for (lab, a, b, last) in labels]
                pk_T_profiles = [(lab, vertical_profile_means(df_pk, a, b, "temperature", inclusive_end=last)) for (lab, a, b, last) in labels]
                cave_T_profiles = [(lab, vertical_profile_means(df_cave, a, b, "temperature", inclusive_end=last)) for (lab, a, b, last) in labels]

                _co2_parts = [dfp[["mean"]] for _, dfp in (cave_co2_profiles + pk_co2_profiles) if dfp is not None and len(dfp)]
                co2_all = (
                    pd.concat(_co2_parts, axis=0, ignore_index=True) if _co2_parts else pd.DataFrame({"mean": []})
                )
                _t_parts = [dfp[["mean"]] for _, dfp in (cave_T_profiles + pk_T_profiles) if dfp is not None and len(dfp)]
                t_all = pd.concat(_t_parts, axis=0, ignore_index=True) if _t_parts else pd.DataFrame({"mean": []})

                # Suggested defaults for profile panels (user can still override
                # and save per-panel). The value-axis (x) range is computed from
                # this stage's *actual* CAVE+PK data — not a hardcoded guess —
                # so a high release never gets silently clipped off the chart
                # (the old fixed 350-1190 for CO2 did exactly that here: real
                # PK profile means reach ~1960 ppm in the later windows). Using
                # the *combined* CAVE+PK range for both regions' panels (rather
                # than auto-fitting each independently) also means CAVE and PK
                # start on the same x-scale, so the two are directly comparable.
                _co2_auto = _auto_ylim(co2_all["mean"]) if len(co2_all) else None
                co2_min_default, co2_max_default = _co2_auto or (350.0, 1190.0)
                _t_auto = _auto_ylim(t_all["mean"]) if len(t_all) else None
                t_min_default, t_max_default = _t_auto or (10.5, 32.0)
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
                vertical_profiles_ready = True

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
                for win_idx, (win_label, a, b, _last) in enumerate(labels, start=1):
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

with tab_ae:
    st.subheader("Air Exchange (PK ↔ CAVE)")
    st.write(
        "Quantifies how much tracer moves between the two zones, using **per-sensor "
        "increments** rather than raw concentrations. Each sensor is referenced to its own "
        "baseline, which removes its individual calibration offset — without that, the "
        "baseline difference between the two regions sits inside every ΔC and biases the "
        "exchange rate."
    )

    if not ae.get("ok"):
        st.error(f"Air-exchange analysis unavailable — {ae.get('error')}")
    else:
        _AE_DEFAULTS = {**ZONE_WIDGET_DEFAULTS}
        _ensure_widget_defaults("ae", _AE_DEFAULTS)

        # ----------------------------------------------------------------
        # 1) Baseline / calibration diagnostics
        # ----------------------------------------------------------------
        st.markdown("### 1 · Baseline & sensor calibration")

        _nc, _np_ = ae["noise_cave"], ae["noise_pk"]
        _ic, _ip = ae["info_cave"], ae["info_pk"]
        st.caption(
            f"Baseline window **{pd.Timestamp(ae['t_base0'])} → {pd.Timestamp(ae['t_base1'])}** "
            f"({ae['base_note']})"
        )

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("CAVE sensors used", f"{int(_ic['kept'].sum())}",
                  delta=f"-{int((~_ic['kept']).sum())} dropped" if (~_ic["kept"]).any() else None,
                  delta_color="off")
        m2.metric("PK sensors used", f"{int(_ip['kept'].sum())}",
                  delta=f"-{int((~_ip['kept']).sum())} dropped" if (~_ip["kept"]).any() else None,
                  delta_color="off")
        m3.metric("CAVE − PK baseline", f"{ae['cave_base_mean'] - ae['pk_base_mean']:+.2f} ppm")
        m4.metric("Region-mean noise floor",
                  f"{max(_nc.get('sigma_mean', np.nan), _np_.get('sigma_mean', np.nan)):.3f} ppm")

        _spread_c = float(_ic.loc[_ic["kept"], "baseline"].max() - _ic.loc[_ic["kept"], "baseline"].min()) if _ic["kept"].any() else np.nan
        _spread_p = float(_ip.loc[_ip["kept"], "baseline"].max() - _ip.loc[_ip["kept"], "baseline"].min()) if _ip["kept"].any() else np.nan
        st.caption(
            f"Between-sensor baseline spread — CAVE **{_spread_c:.1f} ppm**, PK **{_spread_p:.1f} ppm**; "
            f"single-sensor noise σ — CAVE **{_nc.get('sigma_sensor', np.nan):.2f} ppm**, "
            f"PK **{_np_.get('sigma_sensor', np.nan):.2f} ppm**. "
            "A spread far larger than σ means the disagreement is calibration, not mixing."
        )

        if abs(ae["cave_base_mean"] - ae["pk_base_mean"]) > 5.0:
            st.info(
                f"The two regions differ by **{ae['cave_base_mean'] - ae['pk_base_mean']:+.2f} ppm** before "
                "release. Working in increments removes this from ΔC. It is most likely calibration "
                "scatter, but a genuine pre-release gradient would look identical — worth a glance "
                "against the per-sensor spread above before quoting λ to three decimals."
            )

        st.markdown("**What `CAVE_ex` and `PK_ex` mean everywhere below**")
        st.latex(
            r"b_i = \overline{C_i}\big|_{\text{baseline window}} \qquad"
            r"\text{CAVE\_ex}(t) = \frac{1}{N_{\text{CAVE}}}\sum_{i \in \text{CAVE}} \big(C_i(t) - b_i\big)"
            r" \qquad \text{PK\_ex}(t) = \frac{1}{N_{\text{PK}}}\sum_{i \in \text{PK}} \big(C_i(t) - b_i\big)"
        )
        st.markdown(
            "Each sensor is referenced to **its own** baseline mean `b_i`, then the region average is "
            "taken over the debiased values — that is the *excess*, or increment, in ppm. It is zero "
            "during the baseline window by construction, so roughly half the baseline points are "
            "negative; that is noise around zero and is never clipped, because clipping a symmetric "
            "noise would bias the mean upward.\n\n"
            "Everything downstream is built from these two series: the ratio in section 3, and "
            "ΔC in section 4. Note that per-sensor and per-region baselines are equivalent as long as "
            "the same sensors report throughout — the per-sensor form is used because it also survives "
            "sensors dropping in and out."
        )

        with st.expander("Per-sensor baselines and offsets", expanded=False):
            for _lab, _inf in (("CAVE", _ic), ("PK", _ip)):
                st.markdown(f"**{_lab}**")
                _show = _inf.rename(columns={
                    "sensor_number": "Sensor", "wall": "Wall", "baseline": "Baseline (ppm)",
                    "offset": "Offset (ppm)", "n": "Baseline pts", "kept": "Used",
                })
                st.dataframe(
                    _show[["Sensor", "Wall", "Baseline (ppm)", "Offset (ppm)", "Baseline pts", "Used"]]
                    .style.format({"Baseline (ppm)": "{:.1f}", "Offset (ppm)": "{:+.1f}"}),
                    use_container_width=True, hide_index=True, height=240,
                )
                _out = _inf.loc[_inf["kept"]].reindex(
                    _inf.loc[_inf["kept"], "offset"].abs().sort_values(ascending=False).index
                ).head(5)
                if len(_out):
                    st.caption(
                        "Largest offsets: "
                        + ", ".join(f"S{int(r.sensor_number)} ({r.offset:+.0f} ppm)" for r in _out.itertuples())
                    )

        st.markdown("---")

        # ----------------------------------------------------------------
        # 2) Direction / roles
        # ----------------------------------------------------------------
        st.markdown("### 2 · Release direction and the zone being solved")

        _dir_choice = st.radio(
            "Release direction (labels the transfer ratio; does not choose the fit)",
            options=["Auto-detect", DIR_CAVE_TO_PK, DIR_PK_TO_CAVE],
            index=0, horizontal=True, key="ae__direction",
        )
        direction = ae["direction_auto"] if _dir_choice == "Auto-detect" else _dir_choice
        st.caption(
            f"Auto-detected **{ae['direction_auto']}** from release-window excess "
            f"(CAVE {ae['ex_cave_rel']:.1f} ppm vs PK {ae['ex_pk_rel']:.1f} ppm)."
        )

        if direction == DIR_CAVE_TO_PK:
            src_label, rcv_label = "CAVE", "PK"
            ex_src_bulk, ex_rcv = ae["ex_cave"], ae["ex_pk"]
            noise_src = ae["noise_cave"]
            st.success(
                "**Tracer released into CAVE — this is an infiltration experiment.** CAVE plays the "
                "part of the outdoor environment and PK the building. The question is how much of "
                "what is outside gets in, and how fast. PK fills up, so ΔC = CAVE_ex − PK_ex is "
                "**positive** and the ratio in section 3 is a genuine **infiltration factor**."
            )
        else:
            src_label, rcv_label = "PK", "CAVE"
            ex_src_bulk, ex_rcv = ae["ex_pk"], ae["ex_cave"]
            noise_src = ae["noise_pk"]
            st.success(
                "**Tracer released inside PK — this is an emission / decay experiment.** The question "
                "is the reverse one: how fast something released indoors clears out. PK empties, so "
                "ΔC = CAVE_ex − PK_ex is **negative** throughout, and section 3 measures "
                "**exfiltration / dilution — not an infiltration factor**, even though the arithmetic "
                "is identical.\n\n"
                "Two things behave differently in this direction. CAVE is 3.2× the volume and is "
                "usually ventilated, so its excess stays near zero and its response lags — CAVE is "
                "not a reliable readout here, which is why λ is still taken from PK's own balance. "
                "And during the release stage PK has an internal source and internal transport of "
                "its own, so the model does not describe that stretch; fit the decay."
            )

        st.write(
            "**Which zone's mass balance to solve is a separate choice from where the tracer "
            "was released.** PK exchanges only with CAVE, so its balance has a single unknown "
            "and λ_PK = Q/V_PK is identifiable from two-zone data alone. CAVE's balance also "
            "carries its loss to outdoors, which cannot be separated from the PK exchange "
            "without knowing λ_ext independently — the two integrals are almost perfectly "
            "collinear. Solve PK unless you have a specific reason not to."
        )

        _solve_zone = st.radio(
            "Solve λ for",
            options=["PK (recommended)", "CAVE"],
            index=0, horizontal=True, key="ae__solve_zone",
        )
        solve_pk = _solve_zone.startswith("PK")

        if solve_pk:
            solve_label, other_label = "PK", "CAVE"
            ex_solve, ex_other_default = ae["ex_pk"], ae["ex_cave"]
            df_other_inc = ae["df_cave_inc"]
            v_solve = cfg.v_pk
            v_solve_note = "V_PK"
        else:
            solve_label, other_label = "CAVE", "PK"
            ex_solve, ex_other_default = ae["ex_cave"], ae["ex_pk"]
            df_other_inc = ae["df_pk_inc"]
            v_solve = cfg.v_cave_effective if cfg.use_effective_cave_volume else cfg.v_cave_gross
            v_solve_note = "V_CAVE effective" if cfg.use_effective_cave_volume else "V_CAVE gross"

        lambda_ext_per_s = (cfg.lambda_ext / 3600.0) if (not solve_pk and cfg.lambda_ext > 0) else 0.0
        if not solve_pk:
            if lambda_ext_per_s:
                st.warning(
                    f"Solving CAVE's balance with λ_ext = **{cfg.lambda_ext:.3f} 1/h** applied. "
                    "The result is only as good as that number, and CAVE's excess is often small "
                    "enough that its own ventilation and baseline drift dominate the signal. "
                    "Cross-check against the PK solution."
                )
            else:
                st.error(
                    "Solving CAVE's balance with **λ_ext = 0** ignores CAVE's loss to outdoors "
                    "and will bias λ low — often to near zero or negative when CAVE is actively "
                    "ventilated. Either set λ_ext in the sidebar or switch to solving PK."
                )

        if ae.get("env_excluded"):
            st.caption(
                f"{ae['n_env_sensors']} PK-envelope sensors ({', '.join(cfg.envelope_walls)}) are "
                "held out of the CAVE bulk mean; they remain available below as a driving concentration."
            )

        st.markdown("---")

        # ----------------------------------------------------------------
        # 3) Transfer ratio
        # ----------------------------------------------------------------
        _ratio_name = "Infiltration factor" if direction == DIR_CAVE_TO_PK else "Exfiltration / dilution ratio"
        st.markdown(f"### 3 · {_ratio_name}  ({rcv_label}_ex / {src_label}_ex)")
        st.latex(
            r"\text{ratio}(t) = \frac{\text{" + rcv_label + r"\_ex}(t)}{\text{" + src_label
            + r"\_ex}(t)} \quad\text{for}\quad \text{" + src_label + r"\_ex}(t) > \varepsilon"
        )
        if direction == DIR_CAVE_TO_PK:
            st.markdown(
                "**How much of the environment's excess is present inside the building.** 0 means PK "
                "is unaffected, 1 means it has fully caught up with CAVE. Since CO₂ is inert and does "
                "not deposit, the equilibrium value is 1 — anything below that means the release "
                "simply has not run long enough, not that the envelope filters anything out. This is "
                "the quantity the infiltration literature calls the infiltration factor."
            )
        else:
            st.markdown(
                "**How much of what was released inside PK shows up in CAVE.** This is a dilution "
                "measure, not a penetration measure: CAVE is the larger, ventilated zone, so the "
                "ratio stays small because the tracer is diluted and vented, not because anything is "
                "blocking it. **Do not quote this as an infiltration factor** — the arithmetic matches "
                "but the physics is the reverse one."
            )

        _sd_series = noise_src.get("sd_series", np.nan)
        _rel_thresh = cfg.noise_sigma_k * _sd_series if np.isfinite(_sd_series) else 0.0
        ex_thresh = max(cfg.abs_ex_thresh, _rel_thresh)

        tr_t0, tr_t1, tr_note = render_window_picker(
            "ae_tr", stage_defs, t0, t1, ("release",),
            stage_help="The ratio is conventionally taken over the release stage, but any "
                       "stage works — and Manual lets you set the window by eye.",
        )
        if tr_t0 is None or tr_t1 is None:
            tr_t0, tr_t1, tr_note = t_rel0, t_rel1, "fallback: release window"
        st.caption(f"Ratio window **{pd.Timestamp(tr_t0):%Y-%m-%d %H:%M:%S} → {pd.Timestamp(tr_t1):%H:%M:%S}** ({tr_note}).")

        tr = compute_transfer_ratio(ex_src_bulk, ex_rcv, ex_thresh, tr_t0, tr_t1)
        df_sc, sc_slope, sc_intercept, sc_r2 = fit_excess_scatter(ex_src_bulk, ex_rcv, ex_thresh, tr_t0, tr_t1)

        r1, r2c, r3, r4 = st.columns(4)
        r1.metric("Mean ratio (window)", f"{tr['factor']:.3f}" if np.isfinite(tr["factor"]) else "n/a",
                  delta=f"± {tr['sd']:.3f}" if np.isfinite(tr["sd"]) else None, delta_color="off")
        r2c.metric("Scatter slope", f"{sc_slope:.3f}" if np.isfinite(sc_slope) else "n/a",
                   delta=f"R² = {sc_r2:.3f}" if np.isfinite(sc_r2) else None, delta_color="off")
        r3.metric("Points used", f"{tr['n']}")
        r4.metric("Threshold", f"{ex_thresh:.1f} ppm")

        st.caption(
            f"Gate is on the **denominator only** ({src_label}_ex > {ex_thresh:.1f} ppm = "
            f"max({cfg.abs_ex_thresh:.0f}, {cfg.noise_sigma_k:.0f}σ)); "
            f"{tr['n_gated']} bins in the window fell below it. A negative numerator is kept "
            "as-is — clipping it would bias the mean upward."
        )

        if tr["n"] < 10:
            st.warning(
                f"Only **{tr['n']}** points survive the threshold inside this window. "
                "For a short pulse release the receiving zone has barely started to respond, so "
                "the ratio says almost nothing about the exchange — read λ in section 4 instead."
            )

        with st.expander("Plot options — transfer ratio", expanded=False):
            render_save_reset_row("ae", _AE_DEFAULTS)
            render_font_legend_widgets("ae")
            st.checkbox("Show full experiment (not just the release window)", key="ae__full_x")
            st.checkbox("Auto-scale axes", value=True, key="ae__auto_y")

        _full_x = bool(st.session_state.get("ae__full_x", False))
        _auto_y = bool(st.session_state.get("ae__auto_y", True))
        _style_ae = _style_from_prefix("ae")

        if go is None:
            show_matplotlib_fig(plot_io_ratio(
                tr["io_ex"], tr["factor"], tr_t0, tr_t1, ae["t_base0"], ae["t_base1"],
                ex_thresh, cfg, src_label=src_label, rcv_label=rcv_label,
                window_label="Analysis window"))
            show_matplotlib_fig(plot_scatter(
                df_sc, sc_slope, sc_intercept, sc_r2, cfg,
                src_label=src_label, rcv_label=rcv_label))
        else:
            fig_io_p = plot_io_ratio_plotly(
                tr["io_ex"], tr["factor"], tr_t0, tr_t1, ae["t_base0"], ae["t_base1"],
                ex_thresh, cfg, src_label=src_label, rcv_label=rcv_label,
                window_label="Analysis window",
                x_range=(t0, t1) if _full_x else None,
                y_range=None if _auto_y else cfg.ylims["io_ex"],
            )
            apply_plotly_style(fig_io_p, _style_ae)
            show_plotly_chart(fig_io_p)

            fig_sc_p = plot_scatter_plotly(
                df_sc, sc_slope, sc_intercept, sc_r2, cfg,
                src_label=src_label, rcv_label=rcv_label, auto_range=_auto_y,
            )
            apply_plotly_style(fig_sc_p, _style_ae)
            show_plotly_chart(fig_sc_p)

        st.markdown("---")

        # ----------------------------------------------------------------
        # 4) Exchange rate lambda
        # ----------------------------------------------------------------
        st.markdown(f"### 4 · Exchange rate λ_{solve_label}")
        st.latex(
            r"\frac{d\,\text{" + solve_label + r"\_ex}}{dt} = \lambda \cdot \Delta C, \qquad"
            r"\Delta C = \text{" + other_label + r"\_ex} - \text{" + solve_label + r"\_ex}, \qquad"
            r"\lambda = \frac{Q}{V_{" + solve_label + r"}}"
        )

        _filling = (direction == DIR_CAVE_TO_PK) == solve_pk
        st.markdown(
            f"The rate at which **{solve_label}**'s concentration changes is proportional to the "
            f"gradient across the two zones. λ is that constant of proportionality, in 1/h: "
            f"λ = 0.4 means {solve_label} exchanges 40 % of its own volume with {other_label} per hour."
            + (
                f"\n\nHere ΔC is **positive** — {solve_label} is filling, tracer moving in — so "
                f"{solve_label}\\_ex rises and both axes of the fit are positive."
                if _filling else
                f"\n\nHere ΔC is **negative** — {solve_label} is emptying, tracer moving out — so "
                f"{solve_label}\\_ex falls. Both x and y of the fit go negative together, which leaves "
                "λ positive and unchanged: the same equation describes filling and emptying, only the "
                "sign of the gradient flips. That is why the window rule tests **|ΔC|** and only "
                "requires the sign to stay constant."
            )
        )

        cA, cB = st.columns([1, 1])
        with cA:
            _fit_t0, _fit_t1, _fit_note = render_window_picker(
                "ae_lam", stage_defs, t0, t1, ("decay", "release"),
                stage_help="Decay for a short pulse release; Release for a long continuous "
                           "one. The model only requires that the solved zone has no internal "
                           "source, so it works on a rise and on a decay alike.",
            )
        with cB:
            _drive_mode = st.radio(
                "Driving concentration",
                options=[f"Bulk (all {other_label} sensors)", "Selected sensor groups"],
                index=0, key="ae__drive_mode",
                help="What drives the solved zone is the concentration at its envelope, "
                     "which need not equal the other zone's bulk mean when that zone is "
                     "not well mixed.",
            )

        _walls_avail = sorted(df_other_inc["wall"].dropna().astype(str).str.strip().unique())
        _env_set = {w.strip().upper() for w in cfg.envelope_walls}
        _iface_default = [w for w in _walls_avail if w.upper() in _env_set]
        _drive_walls: List[str] = []
        if not _drive_mode.startswith("Bulk"):
            _drive_walls = st.multiselect(
                f"{other_label} sensor groups used as the driving concentration",
                options=_walls_avail,
                default=_iface_default or _walls_avail,
                key="ae__drive_walls",
            )
            if _iface_default:
                st.caption(
                    f"Default **{' + '.join(_iface_default)}** — CAVE-side sensors on the PK "
                    "exterior wall, so they read the concentration right at the envelope."
                )

        _fit_ok = False
        if _fit_t0 is None or _fit_t1 is None:
            st.warning(
                "No fitting window. Upload a stage log, or switch the window control to "
                "**Manual** and set the start and end by hand."
            )
        else:
            _sname, _sstart, _send = _fit_note, _fit_t0, _fit_t1

            if _drive_mode.startswith("Bulk") or not _drive_walls:
                ex_drive = ex_other_default
                drive_note = f"{other_label} bulk mean"
            else:
                _sub = df_other_inc[df_other_inc["wall"].astype(str).str.strip().isin(_drive_walls)]
                _n_sub = int(_sub["sensor_number"].nunique())
                ex_drive = excess_mean_series(_sub, cfg.align_to, max(1, min(cfg.min_sensors, _n_sub)))
                drive_note = f"{other_label}: {', '.join(_drive_walls)} ({_n_sub} sensors)"

            idx_fit, dC_fit, end_reason = select_exchange_window(
                ex_drive, ex_solve, _sstart, _send, cfg.dc_min_ppm
            )

            if len(idx_fit) < cfg.lam_min_pts_int:
                st.warning(
                    f"Only {len(idx_fit)} usable points in **{_sname}** — need "
                    f"{cfg.lam_min_pts_int}. {end_reason}. Lower the ΔC threshold or pick "
                    "another stage."
                )
            else:
                solve_fit = ex_solve.reindex(idx_fit).astype(float)
                res_int = lambda_integrated(solve_fit, dC_fit, cfg.force_zero_intercept, lambda_ext_per_s)
                res_full = lambda_differential(solve_fit, dC_fit, cfg.dc_min_ppm, lambda_ext_per_s)
                res_win = lambda_sliding(
                    res_full["X"], res_full["Y"], res_full["t_mid"],
                    cfg.lam_win_min, cfg.lam_step_min, cfg.lam_min_pts_win,
                )
                _fit_ok = True

                st.caption(
                    f"Window **{idx_fit[0]:%H:%M:%S} → {idx_fit[-1]:%H:%M:%S}** "
                    f"({len(idx_fit)} points) — {end_reason}. Driving concentration: {drive_note}."
                )

                _sv = solve_fit.dropna()
                _sv_range = float(_sv.max() - _sv.min()) if len(_sv) else np.nan
                _dc_med = float(np.nanmedian(np.abs(dC_fit.to_numpy(dtype=float))))
                st.caption(
                    f"Over this window {solve_label}'s own excess moves through "
                    f"**{_sv_range:.1f} ppm** against a median |ΔC| of **{_dc_med:.0f} ppm**. "
                    "λ is only identifiable when the solved zone actually responds — a zone "
                    "that barely moves while the gradient is large is being governed by "
                    "something other than this exchange."
                )

                if np.isfinite(res_int["lam_h"]) and res_int["lam_h"] <= 0:
                    st.error(
                        f"λ came out **non-positive ({res_int['lam_h']:.4f} 1/h)**, which is "
                        "unphysical. **The fitting window is the most likely cause** — set "
                        "**Window → Manual** above and place the start by hand, then check these "
                        "in order:\n\n"
                        f"1. **Does the window include a stage where {solve_label} had its own "
                        f"source?** The model assumes {solve_label} is only fed by exchange. If the "
                        f"release went into {solve_label}, the release stage has to be excluded: "
                        "drag the start to where the decay actually begins. Logged stage boundaries "
                        "are nominal and often sit minutes away from it.\n"
                        f"2. **Is t₀ in the right place?** The fit is anchored at "
                        f"y = {solve_label}\\_ex(t) − {solve_label}\\_ex(t₀) and forced through the "
                        "origin, so starting before the zone was loaded makes y keep one sign while "
                        "ΔC keeps the other, and λ comes out negative.\n"
                        f"3. **Is {solve_label} actually responding to the other zone**, rather than "
                        "to its own ventilation or baseline drift? If not, solving the other zone may "
                        "be the better route."
                    )
                elif np.isfinite(res_int["r2"]) and res_int["r2"] < 0.5:
                    st.warning(
                        f"The integrated fit reaches only R² = {res_int['r2']:.3f}. The two-zone "
                        "model is not describing this window well; treat λ as indicative only."
                    )

                q1, q2, q3 = st.columns(3)
                q1.metric("λ integrated", f"{res_int['lam_h']:.4f} 1/h" if np.isfinite(res_int["lam_h"]) else "n/a",
                          delta=f"R² = {res_int['r2']:.3f}" if np.isfinite(res_int["r2"]) else None,
                          delta_color="off")
                q2.metric("λ full regression", f"{res_full['lam_h']:.4f} 1/h" if np.isfinite(res_full["lam_h"]) else "n/a",
                          delta=f"R² = {res_full['r2']:.3f}" if np.isfinite(res_full["r2"]) else None,
                          delta_color="off")
                q3.metric("λ sliding window", f"{res_win['mean_h']:.4f} 1/h" if np.isfinite(res_win["mean_h"]) else "n/a",
                          delta=f"median = {res_win['median_h']:.3f}" if np.isfinite(res_win["median_h"]) else None,
                          delta_color="off")

                st.caption(
                    "The integrated method usually shows the higher R² because integration "
                    "smooths the noise; the differential methods reveal whether λ drifts during "
                    "the window. They are cross-checks, not alternatives."
                )

                if go is None:
                    show_matplotlib_fig(plot_lambda_panel_matplotlib(
                        res_int, res_full, res_win, cfg, other_label, solve_label,
                        cfg.lam_win_min, cfg.lam_step_min))
                else:
                    gc1, gc2 = st.columns(2)
                    with gc1:
                        f1 = plot_lambda_integrated_plotly(res_int, cfg, other_label, solve_label)
                        apply_plotly_style(f1, _style_ae)
                        show_plotly_chart(f1)
                    with gc2:
                        f2 = plot_lambda_full_plotly(res_full, cfg, other_label, solve_label)
                        apply_plotly_style(f2, _style_ae)
                        show_plotly_chart(f2)
                    f3 = plot_lambda_window_plotly(
                        res_win, cfg, other_label, solve_label, cfg.lam_win_min, cfg.lam_step_min,
                        y_range=None if _auto_y else cfg.ylims["lam_window"],
                    )
                    apply_plotly_style(f3, _style_ae)
                    show_plotly_chart(f3)

                # --------------------------------------------------------
                # 5) Q conversion and equilibrium check
                # --------------------------------------------------------
                st.markdown("---")
                st.markdown("### 5 · Exchange flow Q and result summary")

                lam_ref = res_int["lam_h"]
                Q = lam_ref * v_solve if np.isfinite(lam_ref) else np.nan
                tau_h = (1.0 / lam_ref) if (np.isfinite(lam_ref) and lam_ref > 0) else np.nan

                s1, s2, s3 = st.columns(3)
                s1.metric("Q (exchange flow)", f"{Q:,.1f} m³/h" if np.isfinite(Q) else "n/a",
                          delta=f"{v_solve_note} = {v_solve:,.2f} m³", delta_color="off")
                s2.metric("τ = 1/λ", f"{tau_h:.2f} h" if np.isfinite(tau_h) else "n/a")

                _rel_h = ((pd.Timestamp(t_rel1) - pd.Timestamp(t_rel0)).total_seconds() / 3600.0
                          if (t_rel0 is not None and t_rel1 is not None) else np.nan)
                # Step-response bound: valid only if the source held a constant level for
                # the whole release. A ramping source leaves the receiver further behind,
                # so this is an upper bound on how equilibrated the receiver really is.
                _equil = (1.0 - np.exp(-_rel_h / tau_h)) if (np.isfinite(_rel_h) and np.isfinite(tau_h) and tau_h > 0) else np.nan
                s3.metric("Release / τ", f"{_rel_h / tau_h:.2f}" if np.isfinite(_rel_h) and np.isfinite(tau_h) else "n/a",
                          delta=f"≤{100 * _equil:.0f}% equilibrated (step bound)" if np.isfinite(_equil) else None,
                          delta_color="off")

                st.caption(
                    f"Q is the quantity that does not depend on which zone was solved: λ is Q "
                    f"divided by that zone's volume, so λ_PK and λ_CAVE differ by "
                    f"{(cfg.v_cave_effective / cfg.v_pk):.2f}× for one and the same airflow. "
                    "Compare experiments on Q whenever the solved zone or the release "
                    "direction differs between them."
                )

                _obs_final = np.nan
                _io_rel = tr["io_ex"].dropna()
                if t_rel0 is not None and t_rel1 is not None:
                    _io_rel = _io_rel[(_io_rel.index >= pd.Timestamp(t_rel0)) & (_io_rel.index <= pd.Timestamp(t_rel1))]
                if len(_io_rel):
                    _obs_final = float(_io_rel.iloc[-1])

                if np.isfinite(_equil) and _equil < 0.95:
                    _msg = (
                        f"The release lasted **{_rel_h / tau_h:.2f} τ**, so the transfer ratio in "
                        "section 3 is a **transient** value that still depends on release duration. "
                        "It is not a steady-state penetration factor and is only comparable across "
                        "experiments of equal release duration."
                    )
                    if np.isfinite(_obs_final):
                        _msg += (
                            f" The ratio actually reached **{_obs_final:.3f}** by the end of the "
                            f"release, against a step-response bound of {_equil:.3f}. The step bound "
                            "assumes the source held a constant level; a source that ramps up leaves "
                            "the receiver further behind, so a sizeable gap between the two is "
                            "expected for a continuous release rather than a sign of error."
                        )
                    st.info(_msg)

                _dT = series_mean_in_window(deltaT_pk_minus_cave, idx_fit[0], idx_fit[-1])
                ae_summary = {
                    "exp_code": cfg.exp_code,
                    "direction": direction,
                    "release_source_zone": src_label,
                    "release_receiver_zone": rcv_label,
                    "solved_zone": solve_label,
                    "driving_zone": other_label,
                    "baseline_window_start": ae["t_base0"],
                    "baseline_window_end": ae["t_base1"],
                    "baseline_note": ae["base_note"],
                    "cave_baseline_mean_ppm": ae["cave_base_mean"],
                    "pk_baseline_mean_ppm": ae["pk_base_mean"],
                    "cave_minus_pk_baseline_ppm": ae["cave_base_mean"] - ae["pk_base_mean"],
                    "cave_sensors_used": int(_ic["kept"].sum()),
                    "pk_sensors_used": int(_ip["kept"].sum()),
                    "sigma_sensor_source_ppm": noise_src.get("sigma_sensor", np.nan),
                    "sigma_regionmean_source_ppm": noise_src.get("sigma_mean", np.nan),
                    "excess_threshold_ppm": ex_thresh,
                    "transfer_ratio_mean": tr["factor"],
                    "transfer_ratio_sd": tr["sd"],
                    "transfer_ratio_n": tr["n"],
                    "transfer_ratio_final": _obs_final,
                    "ratio_window_start": pd.Timestamp(tr_t0),
                    "ratio_window_end": pd.Timestamp(tr_t1),
                    "ratio_window_source": tr_note,
                    "scatter_slope": sc_slope,
                    "scatter_intercept": sc_intercept,
                    "scatter_r2": sc_r2,
                    "fit_window_source": _sname,
                    "drive_mode": drive_note,
                    "fit_window_start": idx_fit[0],
                    "fit_window_end": idx_fit[-1],
                    "fit_n_points": int(len(idx_fit)),
                    "fit_end_reason": end_reason,
                    "dc_threshold_ppm": cfg.dc_min_ppm,
                    "lambda_integrated_1ph": res_int["lam_h"],
                    "lambda_integrated_r2": res_int["r2"],
                    "lambda_full_1ph": res_full["lam_h"],
                    "lambda_full_r2": res_full["r2"],
                    "lambda_window_mean_1ph": res_win["mean_h"],
                    "lambda_window_median_1ph": res_win["median_h"],
                    "lambda_window_min": cfg.lam_win_min,
                    "lambda_step_min": cfg.lam_step_min,
                    "lambda_ext_applied_1ph": cfg.lambda_ext if lambda_ext_per_s else 0.0,
                    "V_receiver_m3": v_solve,
                    "V_receiver_basis": v_solve_note,
                    "Q_m3ph": Q,
                    "tau_h": tau_h,
                    "release_over_tau": (_rel_h / tau_h) if (np.isfinite(_rel_h) and np.isfinite(tau_h)) else np.nan,
                    "equilibrated_fraction_step_bound": _equil,
                    "deltaT_pk_minus_cave_mean": _dT,
                }
                ae_summary_df = build_summary_df(ae_summary)
                st.dataframe(ae_summary_df, use_container_width=True, hide_index=True)

                ae_export = {
                    "summary": ae_summary,
                    "summary_df": ae_summary_df,
                    "res_int": res_int,
                    "res_full": res_full,
                    "res_win": res_win,
                    "tr": tr,
                    "tr_window": (tr_t0, tr_t1),
                    "df_sc": df_sc,
                    "sc": (sc_slope, sc_intercept, sc_r2),
                    "labels": (src_label, rcv_label),
                    "lam_labels": (other_label, solve_label),
                    "ex_thresh": ex_thresh,
                }

        if not _fit_ok:
            st.caption(
                "λ results appear here once the window above contains a usable stretch. "
                "Switch the window control to **Manual** if the logged stage boundaries do not "
                "match where the exchange actually starts."
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

    exp_date = _data_date_prefix(df)
    st.caption(
        f"Filenames below start with the experiment's own data date (**{exp_date}**), "
        "not today's date — so files stay identifiable regardless of when you download them."
    )

    st.write("**Summary table**")
    st.dataframe(summary_df, use_container_width=True)

    csv_bytes = summary_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download summary CSV",
        data=csv_bytes,
        file_name=f"{exp_date}_summary.csv",
        mime="text/csv",
    )

    st.markdown("---")
    st.write("**Air exchange (PK ↔ CAVE)**")
    if ae_export is None:
        st.caption(
            "Nothing to export yet — open the **Air Exchange** tab and select a stage with a "
            "usable fitting window first."
        )
    else:
        _ae_src, _ae_rcv = ae_export["labels"]
        _ae_other, _ae_solve = ae_export["lam_labels"]
        st.caption(f"Release direction: **{_ae_src} → {_ae_rcv}**  ·  λ solved for **{_ae_solve}**")
        st.dataframe(ae_export["summary_df"], use_container_width=True, hide_index=True)

        st.download_button(
            label="Download air-exchange summary CSV",
            data=ae_export["summary_df"].to_csv(index=False).encode("utf-8"),
            file_name=f"{cfg.exp_code}_air_exchange_summary.csv",
            mime="text/csv",
        )

        # One row per experiment: convenient to concatenate across runs later.
        _wide = pd.DataFrame([ae_export["summary"]])
        st.download_button(
            label="Download air-exchange result row (wide CSV)",
            data=_wide.to_csv(index=False).encode("utf-8"),
            file_name=f"{cfg.exp_code}_air_exchange_row.csv",
            mime="text/csv",
        )

        st.caption(
            "The air-exchange figures themselves are in **Download figures** below, "
            "alongside every other page's, in PNG and SVG and in the bundled ZIP."
        )

    st.markdown("---")
    st.write("**Download figures**")
    with st.expander("Figures", expanded=False):
        st.caption(
            "Report-ready versions of the charts: no title (so they drop straight into a PPT slide "
            "with your own caption), full legend placed outside the plot so it's never cropped or "
            "overlapping data, and y-axis limits auto-fit to this experiment's data (ratios like "
            "Mixing Index / R² / Coverage / RH / CV keep their fixed 0–1 or 0–100 scale). "
            "PNG for quick use, SVG if you want a lossless vector version you can still edit or "
            "scale up without pixelating."
        )

        # A placeholder written into further down, once every figure below
        # has actually been built — Streamlit renders whatever a container
        # holds at its *position* in the layout, not the order it was filled,
        # so this "download everything" row can sit above the individual
        # figures even though the ZIPs themselves aren't assembled until
        # after the last one (PK Rooms) is generated at the bottom.
        download_all_slot = st.container()
        st.markdown("---")

        # (keyword, png_bytes, svg_bytes) for every figure below — collected
        # here so "download everything" doesn't need to re-render any figure,
        # just re-package the same bytes already handed to each individual
        # download button.
        _export_figs: List[Tuple[str, bytes, bytes]] = []

        def _download_row(fig, keyword: str, label: str):
            png_buf = io.BytesIO()
            fig.savefig(png_buf, format="png", bbox_inches="tight", dpi=200)
            png_buf.seek(0)
            svg_buf = io.BytesIO()
            fig.savefig(svg_buf, format="svg", bbox_inches="tight")
            svg_buf.seek(0)
            png_bytes, svg_bytes = png_buf.getvalue(), svg_buf.getvalue()
            _export_figs.append((keyword, png_bytes, svg_bytes))
            c1, c2 = st.columns(2)
            with c1:
                st.download_button(
                    label=f"Download {label} (PNG)",
                    data=png_bytes,
                    file_name=f"{exp_date}_{keyword}.png",
                    mime="image/png",
                    key=f"dl_{keyword}_png",
                )
            with c2:
                st.download_button(
                    label=f"Download {label} (SVG, vector)",
                    data=svg_bytes,
                    file_name=f"{exp_date}_{keyword}.svg",
                    mime="image/svg+xml",
                    key=f"dl_{keyword}_svg",
                )

        # Mirror each chart's own interactive "Plot options" state (x-axis
        # window, fixed/auto y-limits) so the downloaded figure matches
        # whatever you're currently looking at on screen — these helpers
        # only *read* session_state (the widgets themselves were already
        # created when that tab's code ran above), so calling them again
        # here is safe and doesn't touch the UI.
        x0_ov, x1_ov = render_x_controls("overall", t0, t1, stage_defs)
        y_merged_ov = _collect_ylims_from_prefix("overall", OVERALL_Y_KEYS, default_ylims())
        use_fy_ov = bool(st.session_state.get("overall__use_fixed_y", True))

        xa_c_exp, xa_c1_exp = render_x_controls("zco2_cave", t0, t1, stage_defs)
        uy_c_exp = bool(st.session_state.get("zco2_cave__use_fixed_y", True))
        uy_p_exp = bool(st.session_state.get("zco2_pk__use_fixed_y", True))
        ylo_c_exp, yhi_c_exp = _y_pair_from_prefix("zco2_cave", cfg.ylims["zone_cave_co2"][0], cfg.ylims["zone_cave_co2"][1])
        ylo_p_exp, yhi_p_exp = _y_pair_from_prefix("zco2_pk", cfg.ylims["zone_pk_co2"][0], cfg.ylims["zone_pk_co2"][1])
        y_rc_exp = (ylo_c_exp, yhi_c_exp) if uy_c_exp else None
        y_rp_exp = (ylo_p_exp, yhi_p_exp) if uy_p_exp else None

        xtc0_exp, xtc1_exp = render_x_controls("zt_cave", t0, t1, stage_defs)
        uy_tc_exp = bool(st.session_state.get("zt_cave__use_fixed_y", False))
        uy_tp_exp = bool(st.session_state.get("zt_pk__use_fixed_y", False))
        ytc_lo_exp, ytc_hi_exp = _y_pair_from_prefix("zt_cave", 8.0, 30.0)
        ytp_lo_exp, ytp_hi_exp = _y_pair_from_prefix("zt_pk", 8.0, 30.0)
        y_rtc_exp = (ytc_lo_exp, ytc_hi_exp) if uy_tc_exp else None
        y_rtp_exp = (ytp_lo_exp, ytp_hi_exp) if uy_tp_exp else None

        lock_rx_exp = bool(st.session_state.get("mfc__lock_x_release", True))
        if lock_rx_exp:
            xs_exp, xe_exp = t_rel0, t_rel1
        else:
            xs_exp, xe_exp = render_x_controls("mfc", t0, t1, stage_defs)
        y_r_exp = None
        if mfc_df is not None and st.session_state.get("mfc__use_custom_y", False):
            f_hi_exp = float(mfc_df["F"].max()) if len(mfc_df) else 1.0
            y_r_exp = _y_pair_from_prefix("mfc", 0.0, max(1.0, f_hi_exp * 1.08))

        # Larger, publication-style fonts for exported figures only — scoped
        # with rc_context so it never leaks into other charts or sessions.
        with plt.rc_context(_publication_rc_dict(16.0)):
            fig_overall_export = plot_overall_metrics(
                co2_cave, co2_pk, temp_cave, temp_pk, deltaT_pk_minus_cave,
                stage_defs, cfg, x0_ov, x1_ov,
                line_width=lw_overall, legend_fontsize=leg_overall, export_mode=True,
                y_overrides=y_merged_ov, use_fixed_y=use_fy_ov,
            )
            _download_row(fig_overall_export, "overall_metrics", "overall metrics")

            st.markdown("---")

            fig_zone_export = plot_zone_co2(
                cave_zone_co2, pk_zone_co2, stage_defs, cfg, xa_c_exp, xa_c1_exp,
                cave_line_width=lw_zc * 1.25, pk_line_width=lw_zp * 1.0,
                cave_legend_fs=leg_zc, pk_legend_fs=leg_zp, export_mode=True,
                cave_y_range=y_rc_exp, pk_y_range=y_rp_exp,
            )
            _download_row(fig_zone_export, "zone_co2", "zone CO₂")

            st.markdown("---")

            fig_zone_T_export = plot_zone_temp(
                cave_zone_temp, pk_zone_temp, stage_defs, cfg, xtc0_exp, xtc1_exp,
                cave_line_width=lw_tc * 1.25, pk_line_width=lw_tp * 1.0,
                cave_legend_fs=leg_tc, pk_legend_fs=leg_tp, export_mode=True,
                cave_y_range=y_rtc_exp, pk_y_range=y_rtp_exp,
            )
            _download_row(fig_zone_T_export, "zone_temperature", "zone temperature")

            if mfc_df is not None:
                st.markdown("---")
                fig_mfc_export = plot_mfc(
                    mfc_df, t_on, t_off, t_rel0, t_rel1, cfg,
                    line_width=lw_mfc, legend_fontsize=leg_mfc, export_mode=True,
                    x_range=(xs_exp, xe_exp), y_range=y_r_exp,
                )
                _download_row(fig_mfc_export, "mfc_quicklook", "MFC quicklook")

            if has_rh_data and rh_cave is not None and rh_pk is not None:
                st.markdown("---")
                x0_rh_exp, x1_rh_exp = render_x_controls("rh_ov", t0, t1, stage_defs)
                y_merged_rh_exp = _collect_ylims_from_prefix("rh_ov", RH_OVERVIEW_Y_KEYS, default_ylims())
                use_fy_rh_exp = bool(st.session_state.get("rh_ov__use_fixed_y", True))
                mean_yr_exp = y_merged_rh_exp["rh_mean"] if use_fy_rh_exp else None
                std_yr_exp = y_merged_rh_exp["rh_std"] if use_fy_rh_exp else None
                fig_rh_export = plot_humidity_export(
                    rh_cave, rh_pk, stage_defs, cfg, x0_rh_exp, x1_rh_exp,
                    mean_y_range=mean_yr_exp, std_y_range=std_yr_exp,
                )
                _download_row(fig_rh_export, "humidity_overview", "humidity overview")

            if vertical_profiles_ready:
                st.markdown("---")
                _prof_stage_slug = re.sub(r"[^A-Za-z0-9]+", "_", str(stage_name)).strip("_")

                fig_vp_co2_export = plot_vertical_profiles_export(
                    cave_co2_profiles, pk_co2_profiles, "Mean CO₂ (ppm)",
                    co2_xrange_c, co2_xrange_p, yz_cc, yz_pc,
                    line_width=max(lw_cc, lw_pc),
                )
                _download_row(
                    fig_vp_co2_export, f"vertical_profile_co2_{_prof_stage_slug}",
                    f"vertical profile CO₂ ({stage_name})",
                )

                st.markdown("---")

                fig_vp_temp_export = plot_vertical_profiles_export(
                    cave_T_profiles, pk_T_profiles, "Mean Temperature (°C)",
                    t_xrange_c, t_xrange_p, yz_ct, yz_pt,
                    line_width=max(lw_ct, lw_pt),
                )
                _download_row(
                    fig_vp_temp_export, f"vertical_profile_temperature_{_prof_stage_slug}",
                    f"vertical profile temperature ({stage_name})",
                )

            if len(df_pk) > 0:
                st.markdown("---")
                # Same fixed template as the on-screen PK Rooms tab (one
                # composite image per floor: every room's sensor-level CO2
                # panel plus the real floor plan), mirroring that tab's own
                # x-window / y-limit widget state ("pkfp" prefix) — this
                # only *reads* session_state, so calling it again here is
                # safe (its widgets were already created when that tab ran).
                pk_cat_exp = pk_cat if "pk_cat" in dir() else sensor_catalog(df_pk)
                pk_cat_exp = pk_cat_exp.copy()
                pk_cat_exp["room_group"] = pk_cat_exp["wall"].apply(_pk_room_group)

                x0_pkfp_exp, x1_pkfp_exp = render_x_controls("pkfp", t0, t1, stage_defs)
                use_fy_pkfp_exp = bool(st.session_state.get("pkfp__use_fixed_y", False))
                y_range_pkfp_exp = None
                if use_fy_pkfp_exp:
                    y_range_pkfp_exp = (
                        float(st.session_state.get("pkfp__y_min", 350.0)),
                        float(st.session_state.get("pkfp__y_max", 2000.0)),
                    )

                for floor_key_exp, floor_label_exp in [("FF", "upper floor"), ("GF", "ground floor")]:
                    fp_img_path_exp = os.path.join(
                        os.path.dirname(os.path.abspath(__file__)), "assets", "floorplans",
                        PK_FLOORPLAN_IMAGES[floor_key_exp],
                    )
                    fig_pkrooms_exp = plot_pk_floorplan_export(
                        floor_key_exp, pk_cat_exp, df_pk, cfg.align_to, stage_defs,
                        x0_pkfp_exp, x1_pkfp_exp, fp_img_path_exp, y_range=y_range_pkfp_exp,
                    )
                    _download_row(
                        fig_pkrooms_exp, f"pk_rooms_{floor_key_exp.lower()}",
                        f"PK rooms — {floor_label_exp}",
                    )
                    st.markdown("---")

        # ---- Air exchange (PK <-> CAVE) ----------------------------------
        if ae_export is not None:
            _ae_other, _ae_solve = ae_export["lam_labels"]
            _ae_src, _ae_rcv = ae_export["labels"]
            _w0, _w1 = ae_export["tr_window"]

            _download_row(
                plot_io_ratio(
                    ae_export["tr"]["io_ex"], ae_export["tr"]["factor"], _w0, _w1,
                    ae["t_base0"], ae["t_base1"], ae_export["ex_thresh"], cfg,
                    src_label=_ae_src, rcv_label=_ae_rcv,
                    window_label="Analysis window", export_mode=True,
                ),
                "transfer_ratio", "transfer ratio",
            )
            st.markdown("---")

            _sl, _ic_, _r2_ = ae_export["sc"]
            _download_row(
                plot_scatter(ae_export["df_sc"], _sl, _ic_, _r2_, cfg,
                             src_label=_ae_src, rcv_label=_ae_rcv, export_mode=True),
                "excess_scatter", "excess scatter",
            )
            st.markdown("---")

            _download_row(
                plot_lambda_integrated_export(ae_export["res_int"], cfg, _ae_other, _ae_solve),
                "lambda_integrated", f"λ integrated ({_ae_solve})",
            )
            st.markdown("---")

            _download_row(
                plot_lambda_regression_export(ae_export["res_full"], cfg, _ae_other, _ae_solve),
                "lambda_regression", f"λ full regression ({_ae_solve})",
            )
            st.markdown("---")

            _download_row(
                plot_lambda_window_export(ae_export["res_win"], cfg, _ae_other, _ae_solve,
                                          cfg.lam_win_min, cfg.lam_step_min),
                "lambda_sliding_window", f"λ sliding window ({_ae_solve})",
            )
            st.markdown("---")

        # Now that every figure above has been built and its PNG/SVG bytes
        # collected, fill in the placeholder reserved at the top of this
        # expander with one-click "everything at once" downloads.
        if _export_figs:
            zip_png_buf = io.BytesIO()
            with zipfile.ZipFile(zip_png_buf, "w", zipfile.ZIP_DEFLATED) as zf:
                for keyword, png_bytes, _svg_bytes in _export_figs:
                    zf.writestr(f"{exp_date}_{keyword}.png", png_bytes)
            zip_png_buf.seek(0)

            zip_svg_buf = io.BytesIO()
            with zipfile.ZipFile(zip_svg_buf, "w", zipfile.ZIP_DEFLATED) as zf:
                for keyword, _png_bytes, svg_bytes in _export_figs:
                    zf.writestr(f"{exp_date}_{keyword}.svg", svg_bytes)
            zip_svg_buf.seek(0)

            with download_all_slot:
                st.write(f"**Download all {len(_export_figs)} figures at once**")
                ca1, ca2 = st.columns(2)
                with ca1:
                    st.download_button(
                        label="⬇️ Download ALL figures (ZIP of PNGs)",
                        data=zip_png_buf,
                        file_name=f"{exp_date}_all_figures_png.zip",
                        mime="application/zip",
                        key="dl_all_png_zip",
                    )
                with ca2:
                    st.download_button(
                        label="⬇️ Download ALL figures (ZIP of SVGs, vector)",
                        data=zip_svg_buf,
                        file_name=f"{exp_date}_all_figures_svg.zip",
                        mime="application/zip",
                        key="dl_all_svg_zip",
                    )

# If we forced defaults due to a new upload, clear the flag after widgets have been created.
if st.session_state.get("__force_defaults_from_upload", False):
    st.session_state["__force_defaults_from_upload"] = False