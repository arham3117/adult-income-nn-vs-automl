"""Central configuration: paths, seed, column schema, hyper-parameter grids."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"
ARTIFACTS = ROOT / "artifacts"
DEPLOY_DIR = ARTIFACTS / "deployment"
FIGURES = ROOT / "figures"
RESULTS = ROOT / "results"
REPORTS = ROOT / "reports"

for _p in (DATA_RAW, DATA_PROC, ARTIFACTS, DEPLOY_DIR, FIGURES, RESULTS, REPORTS):
    _p.mkdir(parents=True, exist_ok=True)

SEED = 42

# The 15 columns of the UCI Adult data files, in file order.
RAW_COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
    "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
    "hours-per-week", "native-country", "income",
]

TARGET = "income"

# Dropped with reasoning (see report):
#   fnlwgt    -> census sampling weight, a property of the survey design, not of the person.
#   education -> exact duplicate of the ordinal `education-num` in string form.
DROP_COLUMNS = ["fnlwgt", "education"]

NUMERIC_FEATURES = [
    "age", "education-num", "capital-gain", "capital-loss", "hours-per-week",
]
CATEGORICAL_FEATURES = [
    "workclass", "marital-status", "occupation", "relationship", "race", "sex",
    "native-country",
]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Columns that actually contain '?' in the raw files.
MISSING_TOKEN = "?"
MISSING_CATEGORY = "Missing"

TEST_SIZE = 0.20
VAL_SIZE = 0.20  # of the full dataset, carved out of the remaining 80%

UCI_URL = "https://archive.ics.uci.edu/static/public/2/adult.zip"
