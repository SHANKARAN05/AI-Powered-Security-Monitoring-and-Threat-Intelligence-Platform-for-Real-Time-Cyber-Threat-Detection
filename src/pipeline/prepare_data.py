from __future__ import annotations

from src.common.base_preprocessor import ProcessedSplits
from src.common.config import MODELS_DIR, PROCESSED_DIR
from src.preprocessing.cicids_preprocessor import CICIDSPreprocessor
from src.preprocessing.unsw_preprocessor import UNSWPreprocessor

PREPROCESSORS = {
    "cicids": CICIDSPreprocessor,
    "unsw": UNSWPreprocessor,
}


def _save(splits: ProcessedSplits, out_dir) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    splits.train.to_parquet(out_dir / "train.parquet", index=False)
    splits.val.to_parquet(out_dir / "val.parquet", index=False)
    splits.test.to_parquet(out_dir / "test.parquet", index=False)


def main() -> None:
    for key, preprocessor_cls in PREPROCESSORS.items():
        pre = preprocessor_cls(artifact_dir=MODELS_DIR / key)
        splits = pre.run()
        _save(splits, PROCESSED_DIR / key)
        print(f"[{key}] train={splits.train.shape} val={splits.val.shape} test={splits.test.shape}")
        print(f"[{key}] features: {splits.feature_columns}")


if __name__ == "__main__":
    main()
