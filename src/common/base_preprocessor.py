from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from src.common.config import RANDOM_SEED


@dataclass
class ProcessedSplits:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    feature_columns: List[str]
    binary_label_col: str
    multiclass_label_col: str


class BasePreprocessor(ABC):
    """Shared clean -> encode -> scale -> select-features pipeline.

    Subclasses implement only dataset-specific loading/cleaning/splitting
    (`load_and_clean`) plus column metadata. Everything downstream of the raw
    train/val/test split is identical across datasets, so the two pipelines
    (CICIDS2017, UNSW-NB15) stay structurally comparable and the fitted
    artifacts can be reused at inference time via `transform`.
    """

    name: str
    scaler_cls = StandardScaler
    BINARY_LABEL_COL = "binary_label"
    MULTICLASS_LABEL_COL = "multiclass_label"

    def __init__(self, artifact_dir: Path):
        self.artifact_dir = Path(artifact_dir)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)

        self.onehot_encoder: OneHotEncoder | None = None
        self.onehot_output_cols: List[str] = []
        self.scaler = None
        self._scaled_columns: List[str] = []
        self.variance_selector: VarianceThreshold | None = None
        self.correlated_dropped: List[str] = []
        self.selected_features: List[str] = []
        self.multiclass_encoder: LabelEncoder | None = None

    # ---- dataset-specific hooks -------------------------------------------
    @abstractmethod
    def load_and_clean(self) -> Dict[str, pd.DataFrame]:
        """Return {'train': df, 'val': df, 'test': df}, cleaned and with
        BINARY_LABEL_COL / MULTICLASS_LABEL_COL already populated (use
        `encode_multiclass_labels` below), but not yet encoded/scaled/selected.
        """

    @abstractmethod
    def categorical_columns(self) -> List[str]: ...

    @abstractmethod
    def log_transform_columns(self, df: pd.DataFrame) -> List[str]: ...

    def encode_multiclass_labels(self, train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, raw_col: str):
        """Fit a LabelEncoder on the raw multiclass string column (train only)
        and add MULTICLASS_LABEL_COL to all three splits."""
        if self.multiclass_encoder is None:
            self.multiclass_encoder = LabelEncoder()
            self.multiclass_encoder.fit(train[raw_col])
        known = set(self.multiclass_encoder.classes_)
        fallback = self.multiclass_encoder.classes_[0]

        def _map(df: pd.DataFrame) -> pd.DataFrame:
            df = df.copy()
            safe = df[raw_col].where(df[raw_col].isin(known), fallback)
            df[self.MULTICLASS_LABEL_COL] = self.multiclass_encoder.transform(safe)
            return df

        return _map(train), _map(val), _map(test)

    # ---- shared pipeline ----------------------------------------------------
    def run(self, n_features: int = 40, corr_threshold: float = 0.95) -> ProcessedSplits:
        splits = self.load_and_clean()
        train, val, test = splits["train"], splits["val"], splits["test"]

        train = self._apply_log_transform(train)
        val = self._apply_log_transform(val)
        test = self._apply_log_transform(test)

        train = self._apply_categorical(train, fit=True)
        val = self._apply_categorical(val, fit=False)
        test = self._apply_categorical(test, fit=False)

        exclude = {self.BINARY_LABEL_COL, self.MULTICLASS_LABEL_COL}
        train = self._apply_scaling(train, fit=True, exclude=exclude)
        val = self._apply_scaling(val, fit=False, exclude=exclude)
        test = self._apply_scaling(test, fit=False, exclude=exclude)

        train = self._apply_feature_selection(train, fit=True, n_features=n_features, corr_threshold=corr_threshold)
        val = self._apply_feature_selection(val, fit=False, n_features=n_features, corr_threshold=corr_threshold)
        test = self._apply_feature_selection(test, fit=False, n_features=n_features, corr_threshold=corr_threshold)

        self.save_artifacts()
        return ProcessedSplits(train, val, test, self.selected_features, self.BINARY_LABEL_COL, self.MULTICLASS_LABEL_COL)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply already-fitted transforms to new rows at inference time
        (API / replay simulator). Call `load_artifacts()` first."""
        df = self._apply_log_transform(df)
        df = self._apply_categorical(df, fit=False)
        df = self._apply_scaling(df, fit=False, exclude={self.BINARY_LABEL_COL, self.MULTICLASS_LABEL_COL})
        keep = [c for c in self.selected_features if c in df.columns]
        return df[keep]

    # ---- steps ----------------------------------------------------------------
    def _apply_log_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for c in self.log_transform_columns(df):
            df[c] = np.log1p(df[c].clip(lower=0))
        return df

    def _apply_categorical(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        cat_cols = [c for c in self.categorical_columns() if c in df.columns]
        if not cat_cols:
            return df
        if fit:
            self.onehot_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            self.onehot_encoder.fit(df[cat_cols])
            self.onehot_output_cols = list(self.onehot_encoder.get_feature_names_out(cat_cols))
        encoded = self.onehot_encoder.transform(df[cat_cols])
        encoded_df = pd.DataFrame(encoded, columns=self.onehot_output_cols, index=df.index)
        return pd.concat([df.drop(columns=cat_cols), encoded_df], axis=1)

    def _apply_scaling(self, df: pd.DataFrame, fit: bool, exclude: set) -> pd.DataFrame:
        if fit:
            self._scaled_columns = [
                c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
            ]
            self.scaler = self.scaler_cls()
            self.scaler.fit(df[self._scaled_columns])
        df = df.copy()
        cols = [c for c in self._scaled_columns if c in df.columns]
        df[cols] = self.scaler.transform(df[cols])
        return df

    def _apply_feature_selection(self, df: pd.DataFrame, fit: bool, n_features: int, corr_threshold: float) -> pd.DataFrame:
        exclude = {self.BINARY_LABEL_COL, self.MULTICLASS_LABEL_COL}
        if fit:
            candidate_cols = [c for c in df.columns if c not in exclude]

            self.variance_selector = VarianceThreshold(threshold=0.0)
            self.variance_selector.fit(df[candidate_cols])
            kept_mask = self.variance_selector.get_support()
            candidate_cols = list(np.array(candidate_cols)[kept_mask])

            corr = df[candidate_cols].corr().abs()
            upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
            self.correlated_dropped = [c for c in upper.columns if (upper[c] > corr_threshold).any()]
            candidate_cols = [c for c in candidate_cols if c not in self.correlated_dropped]

            rf = RandomForestClassifier(
                n_estimators=200, random_state=RANDOM_SEED, n_jobs=-1, class_weight="balanced"
            )
            rf.fit(df[candidate_cols], df[self.MULTICLASS_LABEL_COL])
            importances = pd.Series(rf.feature_importances_, index=candidate_cols).sort_values(ascending=False)
            self.selected_features = importances.head(n_features).index.tolist()

        keep = [c for c in self.selected_features if c in df.columns]
        keep += [c for c in (self.BINARY_LABEL_COL, self.MULTICLASS_LABEL_COL) if c in df.columns]
        return df[keep]

    # ---- persistence ------------------------------------------------------------
    def save_artifacts(self) -> None:
        joblib.dump(self.onehot_encoder, self.artifact_dir / "onehot_encoder.pkl")
        joblib.dump(self.scaler, self.artifact_dir / "scaler.pkl")
        joblib.dump(self.multiclass_encoder, self.artifact_dir / "multiclass_encoder.pkl")
        metadata = {
            "selected_features": self.selected_features,
            "correlated_dropped": self.correlated_dropped,
            "onehot_output_cols": self.onehot_output_cols,
            "scaled_columns": self._scaled_columns,
        }
        with open(self.artifact_dir / "feature_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    def load_artifacts(self) -> None:
        self.onehot_encoder = joblib.load(self.artifact_dir / "onehot_encoder.pkl")
        self.scaler = joblib.load(self.artifact_dir / "scaler.pkl")
        self.multiclass_encoder = joblib.load(self.artifact_dir / "multiclass_encoder.pkl")
        with open(self.artifact_dir / "feature_metadata.json") as f:
            metadata = json.load(f)
        self.selected_features = metadata["selected_features"]
        self.correlated_dropped = metadata["correlated_dropped"]
        self.onehot_output_cols = metadata["onehot_output_cols"]
        self._scaled_columns = metadata["scaled_columns"]
