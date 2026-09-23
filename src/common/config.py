from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "Data"
PROCESSED_DIR = DATA_DIR / "processed"
THREAT_INTEL_DIR = DATA_DIR / "threat_intel"
MODELS_DIR = PROJECT_ROOT / "models"

CICIDS_RAW_PATH = DATA_DIR / "archive" / "Week_filtered.csv"
UNSW_TRAIN_PATH = DATA_DIR / "UNSW_NB15_training-set.csv"
UNSW_TEST_PATH = DATA_DIR / "UNSW_NB15_testing-set.csv"

RANDOM_SEED = 42
VAL_SIZE = 0.15
TEST_SIZE = 0.15  # only used for CICIDS's from-scratch split; UNSW uses the official test file

for _d in (
    PROCESSED_DIR / "cicids",
    PROCESSED_DIR / "unsw",
    THREAT_INTEL_DIR,
    MODELS_DIR / "cicids",
    MODELS_DIR / "unsw",
):
    _d.mkdir(parents=True, exist_ok=True)
