"""Geodesic calculations for tropical-cyclone tracks.

Primary backend is :class:`pyproj.Geod` on the WGS84 ellipsoid.  If pyproj
is unavailable, a verified spherical-haversine fallback (mean Earth radius
6371.0088 km) is used instead; the active backend is exposed as ``BACKEND``
and recorded in the dataset metadata so results stay reproducible.

Conventions
-----------
* Input/output latitudes in degrees north, longitudes in degrees east.
  Longitudes may use any convention (0–360 or −180–180); the Geod backend
  handles both, and stored ground truth uses the file convention (0–360°E).
* Bearings/azimuths are degrees east of north, normalised to [0, 360).
* ``enu_from_geodetic`` expresses a target in the local tangent plane of an
  origin: e = d·sin(az), n = d·cos(az).  This is exactly reversible with
  :func:`geodetic_from_enu` (up to floating-point precision).
* Degree differences are NEVER interpreted as distances; every distance
  goes through :func:`inverse`.
"""

from __future__ import annotations

import numpy as np

try:
    from pyproj import Geod

    _GEOD = Geod(ellps="WGS84")
    BACKEND = "pyproj.Geod(WGS84)"
except Exception:  # pragma: no cover - fallback path
    _GEOD = None
    BACKEND = "haversine-fallback(R=6371.0088km)"

R_KM = 6371.0088


def _asarray(*args) -> list[np.ndarray]:
    return [np.asarray(a, dtype=float) for a in args]


def _broadcast(*args) -> list[np.ndarray]:
    arrs = [np.asarray(a, dtype=float) for a in args]
    return [np.asarray(a, dtype=float) for a in np.broadcast_arrays(*arrs)]


def _haversine_inverse(lat1, lon1, lat2, lon2) -> tuple[np.ndarray, np.ndarray]:
    """Spherical inverse: distance (m) and forward azimuth (deg) arrays."""
    lat1, lon1, lat2, lon2 = _asarray(lat1, lon1, lat2, lon2)
    dlon = np.deg2rad((lon2 - lon1 + 540.0) % 360.0 - 180.0)
    rlat1, rlat2 = np.deg2rad(lat1), np.deg2rad(lat2)
    a = np.sin((rlat2 - rlat1) / 2.0) ** 2 + np.cos(rlat1) * np.cos(rlat2) * np.sin(dlon / 2.0) ** 2
    dist_m = 2.0 * (R_KM * 1000.0) * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    x = np.sin(dlon) * np.cos(rlat2)
    y = np.cos(rlat1) * np.sin(rlat2) - np.sin(rlat1) * np.cos(rlat2) * np.cos(dlon)
    return dist_m, (np.rad2deg(np.arctan2(x, y)) + 360.0) % 360.0


def _haversine_forward(lat, lon, az_deg, dist_m) -> tuple[np.ndarray, np.ndarray]:
    """Spherical forward computation: destination of (az, distance)."""
    lat, lon, az_deg, dist_m = _asarray(lat, lon, az_deg, dist_m)
    delta = (dist_m / 1000.0) / R_KM
    rlat, az = np.deg2rad(lat), np.deg2rad(az_deg)
    rlat2 = np.arcsin(np.sin(rlat) * np.cos(delta) + np.cos(rlat) * np.sin(delta) * np.cos(az))
    rlon2 = np.deg2rad(lon) + np.arctan2(
        np.sin(az) * np.sin(delta) * np.cos(rlat), np.cos(delta) - np.sin(rlat) * np.sin(rlat2))
    return np.rad2deg(rlat2), np.rad2deg(rlon2)


def inverse(lat1, lon1, lat2, lon2) -> tuple[np.ndarray, np.ndarray]:
    """Geodesic distance (metres) and forward azimuth (deg E of N).

    Accepts scalars or array-likes; always returns ``np.ndarray``.
    """
    lat1, lon1, lat2, lon2 = _broadcast(lat1, lon1, lat2, lon2)
    shape = lat1.shape
    if _GEOD is not None:
        az12, _, dist_m = _GEOD.inv(lon1, lat1, lon2, lat2)
        return (np.asarray(dist_m, dtype=float).reshape(shape),
                ((np.asarray(az12, dtype=float).reshape(shape) + 360.0) % 360.0))
    return _haversine_inverse(lat1, lon1, lat2, lon2)


def forward(lat, lon, az_deg, dist_m) -> tuple[np.ndarray, np.ndarray]:
    """Destination (lat, lon) of travelling (azimuth, distance) from a point."""
    lat, lon, az_deg, dist_m = _broadcast(lat, lon, az_deg, dist_m)
    shape = lat.shape
    if _GEOD is not None:
        lon2, lat2, _ = _GEOD.fwd(lon, lat, az_deg, dist_m)
        return (np.asarray(lat2, dtype=float).reshape(shape),
                np.asarray(lon2, dtype=float).reshape(shape))
    return _haversine_forward(lat, lon, az_deg, dist_m)


def normalize_lon_360(lon) -> np.ndarray:
    """Wrap longitudes to [0, 360)."""
    return np.asarray(lon, dtype=float) % 360.0


def unwrap_lon(lon_deg) -> np.ndarray:
    """Unwrap a longitude series (any convention) to a continuous track."""
    lon = np.asarray(lon_deg, dtype=float).copy()
    ok = np.isfinite(lon)
    if ok.sum() >= 2:
        unwrapped = np.unwrap(np.deg2rad(lon[ok]))
        lon[ok] = np.rad2deg(unwrapped)
    return lon


def bearing_deg(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Forward bearing in degrees east of north, normalised to [0, 360)."""
    _, az = inverse(lat1, lon1, lat2, lon2)
    return az


def haversine_distance_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Great-circle distance in kilometres (spherical Haversine, R=6371.0088 km).

    This is the canonical distance for trajectory *evaluation* (Task 4):
    unlike :func:`inverse`, it never uses the WGS84 ellipsoid, so reported
    errors are exactly reproducible from the formula in ``docs/evaluation.md``.
    Longitude differences are wrapped to [-180, 180) before use, so
    antimeridian crossings (in either lon convention) need no special casing.
    The ``haversine a`` term is clipped to [0, 1] for numerical stability.
    """
    dist_m, _ = _haversine_inverse(*_broadcast(lat1, lon1, lat2, lon2))
    return np.asarray(dist_m, dtype=float) / 1000.0


def enu_from_geodetic(lat_o, lon_o, lat_t, lon_t) -> tuple[np.ndarray, np.ndarray]:
    """East/north displacement (km) of target(s) in the origin's local frame."""
    dist_m, az = inverse(lat_o, lon_o, lat_t, lon_t)
    dist_km = np.asarray(dist_m, dtype=float) / 1000.0
    azr = np.deg2rad(np.asarray(az, dtype=float))
    return dist_km * np.sin(azr), dist_km * np.cos(azr)


def geodetic_from_enu(lat_o, lon_o, e_km, n_km) -> tuple[np.ndarray, np.ndarray]:
    """Invert :func:`enu_from_geodetic`: destination of a local (e, n) offset."""
    e_km, n_km = _asarray(e_km, n_km)
    dist_m = np.hypot(e_km, n_km) * 1000.0
    az = (np.rad2deg(np.arctan2(e_km, n_km)) + 360.0) % 360.0
    return forward(lat_o, lon_o, az, dist_m)
