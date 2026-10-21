# EDA outputs — IBTrACS Western Pacific

Reproduce everything with (from the repository root):

```bash
python -m scripts.eda.run_eda --input data/IBTrACS.WP.v04r01.nc --out EDA
```

`--input` also accepts an IBTrACS CSV export (units row auto-detected).
Requires: pandas, NumPy, Matplotlib, xarray, netCDF4 (see repo environment).

## Figures (`figures/`, all ≥200 DPI)

| File | Content |
|---|---|
| 01_storm_count_by_year.png | Unique storms + observations per season |
| 02_observations_per_storm.png | Observations-per-storm histogram (linear + log) |
| 03_missing_values.png | Missing-value percentage by variable |
| 04_latitude_distribution.png | Latitude histogram |
| 05_longitude_distribution.png | Longitude 0–360 (stored) vs wrapped −180…180 (display) |
| 06_storm_speed_distribution.png | File STORM_SPEED histogram |
| 07_storm_direction_distribution.png | STORM_DIR polar histogram |
| 08_sample_storm_tracks.png | 12 deterministic sample tracks (seed=42, gaps broken) |
| 09_observation_time_gaps.png | Consecutive-observation gaps |
| 10_storm_duration_distribution.png | Storm duration (days) |
| 11_monthly_frequency.png | Monthly observation/storm frequency |
| 12_track_type_distribution.png | TRACK_TYPE per-observation counts |
| 13_basin_distribution.png | Per-observation BASIN counts |
| 14_spatial_density.png | Log-scale hexbin position density |
| 15_speed_reported_vs_recomputed.png | Reported vs past-only speed + residuals |
| 16_duration_vs_path_length.png | Duration vs summed haversine path length |

## Tables (`tables/`)

01 overview · 02 by-year · 03 by-month · 04 obs-per-storm · 05 missing ·
06 track-type · 07 basin · 08 gaps · 09 IFLAG · 10 subset filters ·
11 continuity · 12 train/val/test splits + windows · 13 quality flags ·
14 speed agreement · 15 exclusion reasons.

## Notes

* Raw NetCDF files are never modified.
* `report.md` distinguishes confirmed findings from recommendations.
* Longitude is stored 0–360°E; display wraps are labelled wherever used.
