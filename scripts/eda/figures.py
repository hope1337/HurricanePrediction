"""Publication-ready figures for the IBTrACS-WP EDA (Agg backend, >=200 DPI)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm

plt.rcParams.update({
    "figure.dpi": 200, "savefig.dpi": 200, "savefig.bbox": "tight",
    "axes.grid": True, "grid.alpha": 0.3, "axes.axisbelow": True,
    "font.size": 10, "axes.titlesize": 12, "axes.labelsize": 11,
})

MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def _save(fig, path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_storm_count_by_year(df, path, start=1980, end=2024) -> None:
    sub = df[(df["SEASON"] >= start) & (df["SEASON"] <= end)]
    per_year = sub.groupby("SEASON").agg(n_storms=("SID", "nunique"), n_obs=("SID", "size")).reset_index()
    per_year["SEASON"] = per_year["SEASON"].astype(int)
    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.bar(per_year["SEASON"].to_numpy(), per_year["n_storms"], color="steelblue", label="Unique storms")
    ax1.set_xlabel("Season (year storm began)")
    ax1.set_ylabel("Unique storms", color="steelblue")
    ax1.set_xticks(per_year["SEASON"].to_numpy()[::2])
    ax1.tick_params(axis="x", rotation=90, labelsize=7)
    ax1.set_title(f"Storm count and observation count by season ({start}–{end}, WP file)")
    ax2 = ax1.twinx()
    ax2.plot(per_year["SEASON"].to_numpy(), per_year["n_obs"], color="darkorange",
             marker="o", ms=3, label="Observations")
    ax2.set_ylabel("Observations (3-hourly + off-synoptic)", color="darkorange")
    fig.legend(loc="upper left", bbox_to_anchor=(0.08, 0.95))
    _save(fig, path)


def fig_observations_per_storm(df, path, start=1980, end=2024) -> None:
    sub = df[(df["SEASON"] >= start) & (df["SEASON"] <= end)]
    n = sub.groupby("SID").size().to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].hist(n, bins=60, color="steelblue", edgecolor="white")
    axes[0].set_xlabel("Observations per storm (all timestamps)")
    axes[0].set_ylabel("Storms")
    axes[0].set_title("Distribution of observations per storm")
    axes[0].axvline(np.median(n), color="red", ls="--", label=f"median={np.median(n):.0f}")
    axes[0].legend()
    axes[1].hist(n, bins=60, color="steelblue", edgecolor="white", log=True)
    axes[1].set_xlabel("Observations per storm (all timestamps)")
    axes[1].set_ylabel("Storms (log scale)")
    axes[1].set_title("Same distribution, log y-scale")
    _save(fig, path)


def fig_missing_values(df, path) -> None:
    cols = ["LAT", "LON", "STORM_SPEED", "STORM_DIR", "WMO_WIND", "WMO_PRES",
            "BASIN", "TIME", "DIST2LAND", "LANDFALL", "NATURE", "WMO_AGENCY", "IFLAG"]
    cols = [c for c in cols if c in df.columns]
    pct = {c: float(pd.Series(df[c]).isna().mean() * 100) if c != "IFLAG"
           else float((df[c].astype(str) == "").mean() * 100) for c in cols}
    order = sorted(pct, key=pct.get, reverse=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(order, [pct[c] for c in order], color="firebrick")
    ax.set_xlabel("Missing / empty (%)")
    ax.set_title("Missing-value percentage by variable (raw records, all years)")
    for i, c in enumerate(order):
        ax.text(pct[c] + 0.2, i, f"{pct[c]:.1f}%", va="center", fontsize=9)
    ax.set_xlim(0, max(max(pct.values()) * 1.25, 5))
    _save(fig, path)


def fig_latitude_distribution(df, path, start=1980, end=2024) -> None:
    sub = df[(df["SEASON"] >= start) & (df["SEASON"] <= end)]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(sub["LAT"].dropna(), bins=70, color="seagreen", edgecolor="white")
    ax.set_xlabel("Latitude (degrees_north)")
    ax.set_ylabel("Observations")
    ax.set_title(f"Latitude distribution ({start}–{end})")
    _save(fig, path)


def fig_longitude_distribution(df, path, start=1980, end=2024) -> None:
    sub = df[(df["SEASON"] >= start) & (df["SEASON"] <= end)]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    axes[0].hist(sub["LON"].dropna(), bins=80, color="darkslateblue", edgecolor="white")
    axes[0].set_xlabel("Longitude (degrees_east, file convention 0–360)")
    axes[0].set_ylabel("Observations")
    axes[0].set_title("As stored (0–360°E, no dateline jump)")
    wrapped = ((sub["LON"].dropna() + 180.0) % 360.0) - 180.0
    axes[1].hist(wrapped, bins=80, color="darkslateblue", edgecolor="white")
    axes[1].set_xlabel("Longitude (degrees, wrapped to −180…180)")
    axes[1].set_title("Wrapped view (antimeridian jump is a display artefact)")
    fig.suptitle(f"Longitude distribution ({start}–{end}) — file uses 0–360°E", y=1.02)
    _save(fig, path)


def fig_speed_distribution(df, path, start=1980, end=2024) -> None:
    sub = df[(df["SEASON"] >= start) & (df["SEASON"] <= end)]
    v = pd.to_numeric(sub["STORM_SPEED"], errors="coerce").dropna()
    v = v[(v >= 0) & (v <= 998)]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(v, bins=80, color="teal", edgecolor="white")
    ax.set_xlabel("STORM_SPEED (kts)")
    ax.set_ylabel("Observations")
    ax.set_title(f"Translation speed distribution ({start}–{end}) — file STORM_SPEED")
    ax.axvline(v.median(), color="red", ls="--", label=f"median={v.median():.1f} kts")
    ax.legend()
    _save(fig, path)


def fig_direction_distribution(df, path, start=1980, end=2024) -> None:
    sub = df[(df["SEASON"] >= start) & (df["SEASON"] <= end)]
    v = pd.to_numeric(sub["STORM_DIR"], errors="coerce").dropna().to_numpy(dtype=float)
    v = v[(v >= 0) & (v <= 360)]
    theta = np.deg2rad(v)
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="polar")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    hist, edges = np.histogram(theta, bins=36, range=(0, 2 * np.pi))
    ax.bar(edges[:-1], hist, width=np.diff(edges), align="edge", color="teal", edgecolor="white")
    ax.set_title("Translation direction (STORM_DIR, degrees east of north;\n0° = moving north)", pad=20)
    _save(fig, path)


def fig_sample_tracks(df, path, start=1980, end=2024, n=12, seed=42) -> None:
    rng = np.random.default_rng(seed)
    sub = df[(df["SEASON"] >= start) & (df["SEASON"] <= end) & (df["TRACK_TYPE"] == "main")].copy()
    storms = sorted(sub["SID"].unique().tolist())
    pick = sorted(rng.choice(storms, size=min(n, len(storms)), replace=False).tolist())
    from .io import unwrap_longitude_per_storm
    fig, ax = plt.subplots(figsize=(12, 6))
    for sid in pick:
        g = sub[sub["SID"] == sid].sort_values("TIME")
        lon = unwrap_longitude_per_storm(g, "LON").to_numpy(dtype=float)
        # Break the line wherever the time gap exceeds 6h (different segments).
        dt = g["TIME"].diff().dt.total_seconds().to_numpy(dtype=float) / 3600.0
        lon_p, lat_p = lon.copy(), g["LAT"].to_numpy(dtype=float)
        lon_p[1:][dt[1:] > 6.5] = np.nan
        ax.plot(lon_p, lat_p, lw=1.2, alpha=0.85, label=str(sid))
    ax.set_xlabel("Longitude (degrees_east, unwrapped — no dateline artefacts)")
    ax.set_ylabel("Latitude (degrees_north)")
    ax.set_title(f"Sample of {len(pick)} main-track storms ({start}–{end}, deterministic seed={seed})")
    ax.legend(fontsize=7, ncol=2)
    _save(fig, path)


def fig_time_gaps(df, path, start=1980, end=2024) -> None:
    sub = df[(df["SEASON"] >= start) & (df["SEASON"] <= end)].sort_values(["SID", "TIME"])
    gaps = sub.groupby("SID")["TIME"].diff().dt.total_seconds().to_numpy(dtype=float) / 3600.0
    gaps = gaps[np.isfinite(gaps)]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].hist(gaps, bins=np.arange(0, 13) - 0.5, color="steelblue", edgecolor="white")
    axes[0].set_xlim(-0.75, 12.5)
    axes[0].set_xlabel("Gap between consecutive obs (hours)")
    axes[0].set_ylabel("Observation pairs")
    axes[0].set_title("Time gaps 0–12 h (linear)")
    axes[1].hist(gaps, bins=np.arange(0, int(np.nanmax(gaps)) + 2) - 0.5,
                 color="steelblue", edgecolor="white", log=True)
    axes[1].set_xlabel("Gap between consecutive obs (hours)")
    axes[1].set_ylabel("Pairs (log scale)")
    axes[1].set_title("Full gap range (log y)")
    _save(fig, path)


def fig_duration_distribution(df, path, start=1980, end=2024) -> None:
    sub = df[(df["SEASON"] >= start) & (df["SEASON"] <= end)]
    dur = sub.groupby("SID")["TIME"].agg(lambda s: (s.max() - s.min()).total_seconds() / 86400.0)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(dur.dropna(), bins=60, color="saddlebrown", edgecolor="white")
    ax.set_xlabel("Storm duration (days, last − first timestamp)")
    ax.set_ylabel("Storms")
    ax.set_title(f"Storm duration distribution ({start}–{end})")
    ax.axvline(dur.median(), color="red", ls="--", label=f"median={dur.median():.1f} d")
    ax.legend()
    _save(fig, path)


def fig_monthly_frequency(df, path, start=1980, end=2024) -> None:
    sub = df[(df["SEASON"] >= start) & (df["SEASON"] <= end)].copy()
    sub["MONTH"] = sub["TIME"].dt.month
    by_month_obs = sub.groupby("MONTH").size().reindex(range(1, 13), fill_value=0)
    by_month_storms = sub.groupby("MONTH")["SID"].nunique().reindex(range(1, 13), fill_value=0)
    x = np.arange(12)
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x, by_month_obs.values, color="steelblue", label="Observations")
    ax.set_xticks(x, MONTH_NAMES)
    ax.set_ylabel("Observations")
    ax2 = ax.twinx()
    ax2.plot(x, by_month_storms.values, color="darkorange", marker="o", label="Active storm-months")
    ax2.set_ylabel("Storms active in month")
    ax.set_title(f"Monthly frequency ({start}–{end}) — observations vs storms active")
    fig.legend(loc="upper left", bbox_to_anchor=(0.08, 0.95))
    _save(fig, path)


def fig_track_type(df, path, start=1980, end=2024) -> None:
    sub = df[(df["SEASON"] >= start) & (df["SEASON"] <= end)]
    c = sub["TRACK_TYPE"].fillna("(missing)").value_counts()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(c.index.astype(str), c.values, color="slategray")
    ax.set_xlabel("TRACK_TYPE")
    ax.set_ylabel("Observations")
    ax.set_title(f"Track-type distribution ({start}–{end}, per observation)")
    for i, v in enumerate(c.values):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=9)
    _save(fig, path)


def fig_basin(df, path, start=1980, end=2024) -> None:
    sub = df[(df["SEASON"] >= start) & (df["SEASON"] <= end)]
    c = sub["BASIN"].fillna("(missing)").value_counts()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(c.index.astype(str), c.values, color="darkgreen")
    ax.set_xlabel("Per-observation BASIN")
    ax.set_ylabel("Observations")
    ax.set_title(f"Basin distribution ({start}–{end}, per observation)")
    for i, v in enumerate(c.values):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=9)
    _save(fig, path)


def fig_spatial_density(df, path, start=1980, end=2024) -> None:
    sub = df[(df["SEASON"] >= start) & (df["SEASON"] <= end)].copy()
    sub["LONW"] = ((pd.to_numeric(sub["LON"], errors="coerce") + 180.0) % 360.0) - 180.0
    ok = sub.dropna(subset=["LONW", "LAT"])
    fig, ax = plt.subplots(figsize=(12, 5.5))
    hb = ax.hexbin(ok["LONW"], ok["LAT"], gridsize=90, cmap="YlOrRd", norm=LogNorm(), mincnt=1)
    fig.colorbar(hb, ax=ax, label="Observations per hex cell (log scale)")
    ax.set_xlabel("Longitude (wrapped to −180…180 for display)")
    ax.set_ylabel("Latitude (degrees_north)")
    ax.set_title(f"Spatial density of storm positions ({start}–{end}, log scale)")
    _save(fig, path)


def fig_speed_comparison(df_motion, path, start=1980, end=2024) -> None:
    sub = df_motion[(df_motion["SEASON"] >= start) & (df_motion["SEASON"] <= end)].copy()
    rep = pd.to_numeric(sub["STORM_SPEED"], errors="coerce")
    rec = pd.to_numeric(sub["RECOMPUTED_SPEED"], errors="coerce")
    ok = sub[rep.notna() & rec.notna() & (rep <= 998)].copy()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].hexbin(ok["RECOMPUTED_SPEED"], pd.to_numeric(ok["STORM_SPEED"]),
                   gridsize=60, cmap="Blues", norm=LogNorm(), mincnt=1)
    lim = [0, max(60, float(np.nanpercentile(ok['RECOMPUTED_SPEED'], 99.5)))]
    axes[0].plot(lim, lim, "r--", lw=1, label="equality")
    axes[0].set_xlim(lim)
    axes[0].set_ylim(lim)
    axes[0].set_xlabel("Past-only recomputed speed (kts, backward difference)")
    axes[0].set_ylabel("File STORM_SPEED (kts)")
    axes[0].set_title("Reported vs past-only speed")
    axes[0].legend()
    resid = pd.to_numeric(ok["STORM_SPEED"], errors="coerce") - ok["RECOMPUTED_SPEED"].to_numpy(dtype=float)
    resid = resid[np.isfinite(resid)]
    p = np.nanpercentile(resid, [1, 99])
    axes[1].hist(resid[(resid >= p[0]) & (resid <= p[1])], bins=80, color="purple", edgecolor="white")
    axes[1].axvline(0, color="red", ls="--")
    axes[1].set_xlabel("STORM_SPEED − past-only speed (kts; 1–99 pct range)")
    axes[1].set_ylabel("Observations")
    axes[1].set_title("Residual distribution (tests causal-safety assumption)")
    _save(fig, path)


def fig_length_vs_duration(df_motion, path, start=1980, end=2024) -> None:
    sub = df_motion[(df_motion["SEASON"] >= start) & (df_motion["SEASON"] <= end)]
    per = sub.groupby("SID").agg(
        duration_d=("TIME", lambda s: (s.max() - s.min()).total_seconds() / 86400.0),
        path_km=("DISPLACEMENT_KM", "sum"), n_obs=("SID", "size"))
    per = per[np.isfinite(per["duration_d"]) & np.isfinite(per["path_km"])]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.hexbin(per["duration_d"], per["path_km"], gridsize=50, cmap="Greens", norm=LogNorm(), mincnt=1)
    ax.set_xlabel("Storm duration (days)")
    ax.set_ylabel("Trajectory path length (km, summed haversine steps)")
    ax.set_title(f"Duration vs trajectory length ({start}–{end})")
    _save(fig, path)
