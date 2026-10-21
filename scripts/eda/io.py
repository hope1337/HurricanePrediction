"""Loading utilities for IBTrACS WP data (NetCDF and CSV).

NetCDF handling notes
---------------------
* The file stores storms on a fixed ``(storm, date_time)`` grid; each storm
  only uses its first ``numobs`` positions. Only those cells are materialised
  into the long-format table (padding cells are never counted as observations).
* Fill values (``-9999`` / ``-9999000.0``) and NetCDF masked arrays are mapped
  to ``NaN``/``NaT``/empty-string so downstream statistics treat them as
  missing instead of real measurements.
* Byte-string variables (``|S1``) carry trailing NUL/space padding which is
  stripped on decode.
* ``time`` units are ``days since 1858-11-17 00:00:00`` (standard calendar).
* Only the variables needed for EDA are read into memory; agency-specific
  sub-tracks (usa_*, tokyo_*, cma_*, ...) are intentionally skipped.

CSV handling notes
------------------
* Official IBTrACS CSV exports contain a *units row* directly under the
  header (e.g. ``Year``, ``degrees_north``, ...). It is detected and skipped
  so it is never parsed as data.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

TIME_FILL = -9999000.0
TIME_UNITS = "days since 1858-11-17 00:00:00"
TIME_EPOCH = pd.Timestamp("1858-11-17 00:00:00", tz="UTC")

# Tokens that identify the IBTrACS CSV units row (case-insensitive match on a
# few distinctive entries is enough; the full row is then skipped).
_UNITS_TOKENS = {
    "year", "degrees_north", "degrees_east", "kts", "mb", "degrees",
    "1", "km", "nmile", "second", "year-month-day",
}


def _decode_bytes(arr: np.ndarray) -> np.ndarray:
    """Decode a trailing-dimension ``|S1`` array to stripped unicode strings."""
    raw = np.asarray(arr)
    if raw.dtype.kind != "S":
        return raw
    flat = raw.reshape(-1, raw.shape[-1])
    out = np.empty(flat.shape[0], dtype=object)
    for i, row in enumerate(flat):
        out[i] = b"".join(row).decode("utf-8", errors="ignore").replace("\x00", "").strip()
    return out.reshape(raw.shape[:-1])


def _clean_float(values: np.ndarray, fill: float = -9999.0) -> np.ndarray:
    out = np.asarray(values, dtype=float)
    out[out == fill] = np.nan
    return out


def _clean_int(values: np.ndarray, fill: int = -9999) -> np.ndarray:
    out = np.asarray(values, dtype=float)
    out[out == fill] = np.nan
    return out


def load_netcdf(path: str | Path) -> tuple[pd.DataFrame, dict]:
    """Load an IBTrACS ``.nc`` file into a long-format observation table.

    Returns
    -------
    df : DataFrame
        One row per valid storm observation (``numobs`` cells only).
    meta : dict
        Dataset metadata (dims, global attrs, variable attrs, time range).
    """
    import xarray as xr

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    log.info("Opening NetCDF: %s", path)
    ds = xr.open_dataset(path, decode_times=False, decode_cf=False, mask_and_scale=False)

    meta: dict = {
        "source_file": str(path),
        "dims": {k: int(v) for k, v in ds.sizes.items()},
        "global_attrs": {k: str(v) for k, v in ds.attrs.items()},
        "variable_attrs": {},
    }
    for name in ("sid", "season", "basin", "iso_time", "lat", "lon",
                 "storm_speed", "storm_dir", "wmo_wind", "wmo_pres",
                 "track_type", "iflag", "time", "nature", "dist2land", "landfall"):
        if name in ds:
            meta["variable_attrs"][name] = {k: str(v) for k, v in ds[name].attrs.items()}

    n_storm = ds.sizes["storm"]
    numobs = np.asarray(ds["numobs"].values, dtype=int)
    numobs = np.where(numobs < 0, 0, numobs)

    sid = _decode_bytes(ds["sid"].values)
    season = _clean_int(ds["season"].values).astype("float64")
    number = _clean_int(ds["number"].values) if "number" in ds else np.full(n_storm, np.nan)
    track_type = _decode_bytes(ds["track_type"].values)
    main_sid = _decode_bytes(ds["main_track_sid"].values) if "main_track_sid" in ds else np.full(n_storm, "")
    name = _decode_bytes(ds["name"].values) if "name" in ds else np.full(n_storm, "")

    basin = _decode_bytes(ds["basin"].values)
    subbasin = _decode_bytes(ds["subbasin"].values) if "subbasin" in ds else None
    nature = _decode_bytes(ds["nature"].values) if "nature" in ds else None
    iso_raw = _decode_bytes(ds["iso_time"].values)
    iflag = _decode_bytes(ds["iflag"].values) if "iflag" in ds else None
    wmo_agency = _decode_bytes(ds["wmo_agency"].values) if "wmo_agency" in ds else None

    lat = _clean_float(np.asarray(ds["lat"].values, dtype=float))
    lon = _clean_float(np.asarray(ds["lon"].values, dtype=float))
    time_days = np.asarray(ds["time"].values, dtype=float)
    time_days[time_days == TIME_FILL] = np.nan
    sspd = _clean_int(np.asarray(ds["storm_speed"].values, dtype=float))
    sdir = _clean_int(np.asarray(ds["storm_dir"].values, dtype=float))
    wwind = _clean_int(np.asarray(ds["wmo_wind"].values, dtype=float))
    wpres = _clean_int(np.asarray(ds["wmo_pres"].values, dtype=float))
    d2l = _clean_int(np.asarray(ds["dist2land"].values, dtype=float)) if "dist2land" in ds else None
    landfall = _clean_int(np.asarray(ds["landfall"].values, dtype=float)) if "landfall" in ds else None

    records: dict[str, list] = {
        "SID": [], "SEASON": [], "STORM_NUMBER": [], "TRACK_TYPE": [],
        "MAIN_TRACK_SID": [], "NAME": [], "OBS_IDX": [], "TIME_DAYS": [],
        "ISO_TIME": [], "BASIN": [], "SUBBASIN": [], "NATURE": [],
        "LAT": [], "LON": [], "STORM_SPEED": [], "STORM_DIR": [],
        "WMO_WIND": [], "WMO_PRES": [], "WMO_AGENCY": [],
        "DIST2LAND": [], "LANDFALL": [], "IFLAG": [],
    }
    for i in range(n_storm):
        n = int(numobs[i])
        if n <= 0:
            continue
        n = min(n, ds.sizes["date_time"])
        for j in range(n):
            records["SID"].append(sid[i])
            records["SEASON"].append(season[i])
            records["STORM_NUMBER"].append(number[i])
            records["TRACK_TYPE"].append(track_type[i])
            records["MAIN_TRACK_SID"].append(main_sid[i])
            records["NAME"].append(name[i])
            records["OBS_IDX"].append(j)
            records["TIME_DAYS"].append(time_days[i, j])
            records["ISO_TIME"].append(iso_raw[i, j])
            records["BASIN"].append(basin[i, j])
            records["SUBBASIN"].append(subbasin[i, j] if subbasin is not None else "")
            records["NATURE"].append(nature[i, j] if nature is not None else "")
            records["LAT"].append(lat[i, j])
            records["LON"].append(lon[i, j])
            records["STORM_SPEED"].append(sspd[i, j])
            records["STORM_DIR"].append(sdir[i, j])
            records["WMO_WIND"].append(wwind[i, j])
            records["WMO_PRES"].append(wpres[i, j])
            records["WMO_AGENCY"].append(wmo_agency[i, j] if wmo_agency is not None else "")
            records["DIST2LAND"].append(d2l[i, j] if d2l is not None else np.nan)
            records["LANDFALL"].append(landfall[i, j] if landfall is not None else np.nan)
            records["IFLAG"].append(iflag[i, j] if iflag is not None else "")

    df = pd.DataFrame(records)
    # Timestamps: prefer ISO_TIME (explicit per-observation strings); fall back
    # to numeric TIME_DAYS when ISO parsing fails.
    df["ISO_TIME"] = pd.to_datetime(df["ISO_TIME"], errors="coerce", utc=True)
    time_fallback = TIME_EPOCH + pd.to_timedelta(df["TIME_DAYS"], unit="D")
    df["TIME"] = df["ISO_TIME"].fillna(time_fallback)
    df["SEASON"] = pd.to_numeric(df["SEASON"], errors="coerce").astype("Int64")
    for col in ("LAT", "LON", "STORM_SPEED", "STORM_DIR", "WMO_WIND", "WMO_PRES",
                "DIST2LAND", "LANDFALL", "TIME_DAYS"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.sort_values(["SID", "TIME", "OBS_IDX"], kind="mergesort").reset_index(drop=True)

    meta["n_storm_rows"] = int(n_storm)
    meta["n_obs_long"] = int(len(df))
    meta["time_min"] = str(df["TIME"].min())
    meta["time_max"] = str(df["TIME"].max())
    ds.close()
    log.info("Loaded %d observations from %d storm rows.", len(df), n_storm)
    return df, meta


def _looks_like_units_row(values: list[str]) -> bool:
    lowered = {str(v).strip().lower() for v in values}
    return len(lowered & _UNITS_TOKENS) >= 2


def load_csv(path: str | Path) -> tuple[pd.DataFrame, dict]:
    """Load an IBTrACS CSV export, skipping the units row when present."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        header = fh.readline().rstrip("\n").split(",")
        second = fh.readline().rstrip("\n").split(",")
    skip = 2 if _looks_like_units_row(second) else 1
    log.info("CSV header has %d columns; units row %s.", len(header), "detected" if skip == 2 else "absent")
    df = pd.read_csv(path, skiprows=list(range(1, skip)) if skip > 1 else None,
                     na_values=["", " ", "-9999", "-999.0", "MM", "NA", "N/A", "null", "NULL"],
                     low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    rename = {c: c.upper() for c in df.columns}
    df = df.rename(columns=rename)
    colmap = {"LAT": "LAT", "LON": "LON", "SEASON": "SEASON", "SID": "SID",
              "BASIN": "BASIN", "ISO_TIME": "ISO_TIME", "TRACK_TYPE": "TRACK_TYPE",
              "STORM_SPEED": "STORM_SPEED", "STORM_DIR": "STORM_DIR",
              "WMO_WIND": "WMO_WIND", "WMO_PRES": "WMO_PRES", "IFLAG": "IFLAG"}
    for want, have in list(colmap.items()):
        if have not in df.columns:
            for c in df.columns:
                if c.replace("_", "") == want.replace("_", ""):
                    colmap[want] = c
                    break
    if "ISO_TIME" in df.columns:
        df["TIME"] = pd.to_datetime(df["ISO_TIME"], errors="coerce", utc=True)
    else:
        df["TIME"] = pd.NaT
    for col in ("LAT", "LON", "STORM_SPEED", "STORM_DIR", "WMO_WIND", "WMO_PRES", "SEASON"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ("SID", "BASIN", "TRACK_TYPE", "IFLAG"):
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str)
    if "OBS_IDX" not in df.columns:
        df["OBS_IDX"] = df.groupby("SID").cumcount() if "SID" in df.columns else 0
    df = df.sort_values(["SID", "TIME"], kind="mergesort").reset_index(drop=True)
    meta = {"source_file": str(path), "units_row_skipped": skip == 2,
            "n_obs_long": int(len(df)), "time_min": str(df["TIME"].min()),
            "time_max": str(df["TIME"].max())}
    return df, meta


def load_ibtracs(path: str | Path) -> tuple[pd.DataFrame, dict]:
    """Dispatch loader by file extension (``.nc`` / ``.csv``)."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".nc":
        return load_netcdf(path)
    if suffix == ".csv":
        return load_csv(path)
    raise ValueError(f"Unsupported input extension {suffix!r}; expected .nc or .csv: {path}")


def normalise_longitude(lon: pd.Series) -> pd.Series:
    """Return longitude wrapped to ``[-180, 180)`` for display purposes."""
    return ((np.asarray(lon, dtype=float) + 180.0) % 360.0) - 180.0


def unwrap_longitude_per_storm(df: pd.DataFrame, lon_col: str = "LON") -> pd.Series:
    """Unwrap 0-360 longitudes per storm to a continuous track (radians unwrap).

    Avoids spurious straight-line jumps when a storm crosses the antimeridian
    in ``[-180, 180]`` displays and gives correct haversine displacements.
    """
    out = np.full(len(df), np.nan)
    for _, idx in df.groupby("SID", sort=False).groups.items():
        vals = pd.to_numeric(df.loc[idx, lon_col], errors="coerce").to_numpy(dtype=float).copy()
        ok = np.isfinite(vals)
        if ok.sum() >= 2:
            vals[ok] = np.unwrap(np.deg2rad(vals[ok]))
            vals[ok] = np.rad2deg(vals[ok])
        out[df.index.get_indexer(idx)] = vals
    return pd.Series(out, index=df.index)
