"""Shared constants for the IBTrACS-WP EDA pipeline."""

# Study design (university project specification).
STUDY_START_SEASON = 1980
STUDY_END_SEASON = 2024
TRAIN_RANGE = (1980, 2016)
VAL_RANGE = (2017, 2020)
TEST_RANGE = (2021, 2024)

SYNOPTIC_HOURS_6H = (0, 6, 12, 18)
SYNOPTIC_HOURS_3H = (0, 3, 6, 9, 12, 15, 18, 21)

# Forecast design for later stages (context only; no modelling here).
N_INPUT_OBS = 5
N_FUTURE_OBS = 8
WINDOW_LENGTH = N_INPUT_OBS + N_FUTURE_OBS  # 13 consecutive 6-hourly obs
HORIZON_STEPS = {6: 1, 12: 2, 24: 4, 48: 8}  # hours -> steps ahead of last input obs

# Data-quality thresholds (detection only — never used to delete data in EDA).
JUMP_KM_THRESHOLD = 1000.0  # per consecutive observation pair, flagged as suspicious
EXTREME_SPEED_KTS = 80.0  # translation speed above this is flagged as implausible/fast
LON_CONVENTION_MIN, LON_CONVENTION_MAX = 0.0, 360.0  # degrees_east per file metadata

# IFLAG agency order from the NetCDF global/variable metadata.
IFLAG_AGENCIES = [
    "USA", "JMA", "CMA", "HKO", "KMA", "NewDelhi", "LaReunion", "BoM",
    "Nadi", "Wellington", "TD9636", "TD9635", "ds824", "Neumann", "Chenoweth",
]
# IFLAG per-agency characters: O=Original, P=Position interpolated,
# I=Intensity interpolated, V=Other values interpolated (per file metadata).
IFLAG_CHARS = ("O", "P", "I", "V")

KEY_OBS_COLUMNS = [
    "SID", "SEASON", "TRACK_TYPE", "ISO_TIME", "BASIN", "LAT", "LON",
    "STORM_SPEED", "STORM_DIR", "WMO_WIND", "WMO_PRES",
]
