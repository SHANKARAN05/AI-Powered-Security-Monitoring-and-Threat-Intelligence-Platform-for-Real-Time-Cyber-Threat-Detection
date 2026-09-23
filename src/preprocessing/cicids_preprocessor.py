from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.common.base_preprocessor import BasePreprocessor
from src.common.config import CICIDS_RAW_PATH, MODELS_DIR, RANDOM_SEED, TEST_SIZE, VAL_SIZE

# Raw CICIDS2017 Web Attack labels contain a mangled dash character from an
# encoding issue in the original dataset; canonicalize before use as labels.
_LABEL_CLEANUP = {
    r"Web Attack.*Brute Force": "Web Attack - Brute Force",
    r"Web Attack.*XSS": "Web Attack - XSS",
    r"Web Attack.*Sql Injection": "Web Attack - Sql Injection",
}

# Known failure mode for this dataset family (not observed in Week_filtered.csv
# on inspection, but guarded defensively in case a raw daily file is used instead).
_INF_RATE_COLUMNS = ["Flow Bytes/s", "Flow Packets/s"]


class CICIDSPreprocessor(BasePreprocessor):
    name = "cicids"

    def load_and_clean(self) -> Dict[str, pd.DataFrame]:
        df = pd.read_csv(CICIDS_RAW_PATH)
        df.columns = df.columns.str.strip()

        if "Fwd Header Length.1" in df.columns:
            df = df.drop(columns=["Fwd Header Length.1"])

        df = df.drop_duplicates()

        df["Label"] = df["Label"].astype(str).str.strip()
        for pattern, canonical in _LABEL_CLEANUP.items():
            df["Label"] = df["Label"].str.replace(pattern, canonical, regex=True)

        for col in _INF_RATE_COLUMNS:
            if col in df.columns:
                df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        numeric_cols = df.select_dtypes(include="number").columns
        df[numeric_cols] = df[numeric_cols].apply(lambda s: s.fillna(s.median()))

        df[self.BINARY_LABEL_COL] = (df["Label"] != "BENIGN").astype(int)

        train_val, test = train_test_split(
            df, test_size=TEST_SIZE, stratify=df["Label"], random_state=RANDOM_SEED
        )
        train, val = train_test_split(
            train_val,
            test_size=VAL_SIZE / (1 - TEST_SIZE),
            stratify=train_val["Label"],
            random_state=RANDOM_SEED,
        )

        train, val, test = self.encode_multiclass_labels(train, val, test, raw_col="Label")
        for split in (train, val, test):
            split.drop(columns=["Label"], inplace=True)

        return {"train": train, "val": val, "test": test}

    def categorical_columns(self) -> List[str]:
        return []  # CICFlowMeter flow features are all numeric

    def log_transform_columns(self, df: pd.DataFrame) -> List[str]:
        candidates = [
            "Flow Duration",
            "Flow Bytes/s",
            "Flow Packets/s",
            "Total Length of Fwd Packets",
            "Total Length of Bwd Packets",
            "Flow IAT Mean",
            "Flow IAT Max",
            "Fwd IAT Total",
            "Bwd IAT Total",
        ]
        return [c for c in candidates if c in df.columns]


if __name__ == "__main__":
    pre = CICIDSPreprocessor(artifact_dir=MODELS_DIR / "cicids")
    splits = pre.run()
    print("train:", splits.train.shape, "val:", splits.val.shape, "test:", splits.test.shape)
    print("selected features:", splits.feature_columns)
