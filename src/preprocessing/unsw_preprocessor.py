from __future__ import annotations

from typing import Dict, List

import pandas as pd
from sklearn.model_selection import train_test_split

from src.common.base_preprocessor import BasePreprocessor
from src.common.config import MODELS_DIR, RANDOM_SEED, UNSW_TEST_PATH, UNSW_TRAIN_PATH, VAL_SIZE


class UNSWPreprocessor(BasePreprocessor):
    name = "unsw"

    def load_and_clean(self) -> Dict[str, pd.DataFrame]:
        # Respect the dataset authors' official train/test split -- do not repool.
        train_full = pd.read_csv(UNSW_TRAIN_PATH)
        test = pd.read_csv(UNSW_TEST_PATH)

        for df in (train_full, test):
            df.drop(columns=["id"], inplace=True, errors="ignore")
            df["service"] = df["service"].replace("-", "unknown")
            df.rename(columns={"label": self.BINARY_LABEL_COL}, inplace=True)

        train, val = train_test_split(
            train_full, test_size=VAL_SIZE, stratify=train_full["attack_cat"], random_state=RANDOM_SEED
        )

        train, val, test = self.encode_multiclass_labels(train, val, test, raw_col="attack_cat")
        for split in (train, val, test):
            split.drop(columns=["attack_cat"], inplace=True)

        return {"train": train, "val": val, "test": test}

    def categorical_columns(self) -> List[str]:
        return ["proto", "service", "state"]

    def log_transform_columns(self, df: pd.DataFrame) -> List[str]:
        candidates = ["dur", "sbytes", "dbytes", "sload", "dload", "sinpkt", "dinpkt", "sjit", "djit", "rate"]
        return [c for c in candidates if c in df.columns]


if __name__ == "__main__":
    pre = UNSWPreprocessor(artifact_dir=MODELS_DIR / "unsw")
    splits = pre.run()
    print("train:", splits.train.shape, "val:", splits.val.shape, "test:", splits.test.shape)
    print("selected features:", splits.feature_columns)
