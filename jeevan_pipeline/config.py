"""Central configuration for the JEEVAN-DRISHTI preprocessing pipeline (Phase 2)."""
from pathlib import Path

PIPELINE_VERSION = "2.0.0"
DATASET_VERSION = "phase2-v1"

PROJ = Path(__file__).resolve().parents[1]
RAW = PROJ / "data" / "raw"
INTERIM = PROJ / "data" / "interim"
PROCESSED = PROJ / "data" / "processed"
REPORTS = PROJ / "data" / "reports"
REFERENCE = PROJ / "data" / "reference"
REGISTRY = PROJ / "model_dataset_registry.json"

HIST_RAW = RAW / "maharashtra_epiclim_filtered.csv"
IDSP_RAW = RAW / "maharashtra_weekly_outbreak_records_2023_2025_ORIGINAL_UPLOAD.csv"
RAIN_NC = RAW / "imdweb_5dgbAskl.nc"
TMIN_NC = RAW / "imdweb_Oe9H04Bg.nc"
TMAX_NC = RAW / "imdweb_tXexQbB6.nc"
RH_RAW = RAW / "humid_tel_hr_cwprs_mh_2021_2025.csv"
RAW_MANIFEST = RAW / "raw_manifest_checksums.csv"

OFFICIAL_DISTRICTS = [
    "Ahmednagar", "Akola", "Amravati", "Beed", "Bhandara", "Buldhana", "Chandrapur",
    "Chhatrapati Sambhajinagar", "Dharashiv", "Dhule", "Gadchiroli", "Gondia", "Hingoli",
    "Jalgaon", "Jalna", "Kolhapur", "Latur", "Mumbai City", "Mumbai Suburban", "Nagpur",
    "Nanded", "Nandurbar", "Nashik", "Palghar", "Parbhani", "Pune", "Raigad", "Ratnagiri",
    "Sangli", "Satara", "Sindhudurg", "Solapur", "Thane", "Wardha", "Washim", "Yavatmal",
]
assert len(OFFICIAL_DISTRICTS) == 36

# raw spelling (lower-case, single-spaced) -> canonical. Confirmed by the project owner in Phase 1 review.
DISTRICT_ALIASES = {
    "aurangabad": "Chhatrapati Sambhajinagar",
    "osmanabad": "Dharashiv",
    "osmanabad (maharashtra)": "Dharashiv",
    "nasik": "Nashik",
    "sholapur": "Solapur",
    "palaghar": "Palghar",
    "mumbai (city)": "Mumbai City",
    "ahmadnagar": "Ahmednagar",
    "ahilyanagar": "Ahmednagar",
}
# District names split across the district/disease columns by the PDF->CSV parser in the original upload.
SPLIT_DISTRICT_SUFFIX = {"mumbai": "suburban", "chhatrapati": "sambhajinagar"}

# Official district creation / renaming facts used only for boundary notes.
PALGHAR_CREATED = "2014-08-01"

TARGET_DISEASES = [
    "Dengue", "Malaria", "Chikungunya", "Acute Diarrhoeal Disease",
    "Cholera", "Leptospirosis", "Heat-related illness",
]

# Climate validation limits (physically implausible values are set to NaN and logged, never replaced)
RAIN_MAX_MM_DAY = 1500.0
TEMP_MIN_C, TEMP_MAX_C = -5.0, 55.0
RH_MIN_EXCLUSIVE, RH_MAX = 0.0, 100.0   # 0 treated as sensor failure; >100 impossible
RH_EXPECTED_OBS_PER_DAY = 288           # 5-minute telemetry
MAHARASHTRA_BBOX = (15.5, 22.1, 72.6, 80.9)  # lat_min, lat_max, lon_min, lon_max
HIGH_VALUE_QUANTILE = 0.995
