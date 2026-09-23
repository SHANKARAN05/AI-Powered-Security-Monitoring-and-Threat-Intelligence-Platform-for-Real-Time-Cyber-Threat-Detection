from __future__ import annotations

import argparse
import random
import threading
import time
from typing import Optional

import pandas as pd

from src.common.base_preprocessor import BasePreprocessor
from src.common.config import PROCESSED_DIR, THREAT_INTEL_DIR
from src.pipeline.inference_pipeline import InferencePipeline

BINARY_LABEL_COL = BasePreprocessor.BINARY_LABEL_COL
MULTICLASS_LABEL_COL = BasePreprocessor.MULTICLASS_LABEL_COL


def _sample_ip(rng: random.Random, blacklist_rate: float, blacklist_pool: list) -> str:
    # Neither dataset retains real source/destination IPs (stripped upstream
    # to avoid identity-based shortcuts) -- these are synthetic demo IPs,
    # occasionally drawn from the placeholder blacklist to exercise threat-intel
    # correlation live during the replay.
    if blacklist_pool and rng.random() < blacklist_rate:
        return rng.choice(blacklist_pool)
    return f"10.{rng.randint(0, 255)}.{rng.randint(0, 255)}.{rng.randint(1, 254)}"


def replay(
    dataset: str,
    n_rows: int = 200,
    delay_seconds: float = 0.5,
    shuffle: bool = True,
    seed: int = 7,
    stop_event: Optional[threading.Event] = None,
    pipeline: Optional[InferencePipeline] = None,
) -> None:
    test = pd.read_parquet(PROCESSED_DIR / dataset / "test.parquet")
    if shuffle:
        test = test.sample(frac=1.0, random_state=seed)
    test = test.head(n_rows)

    # Callers that already maintain a pipeline registry (e.g. the Flask API)
    # should pass one in so replay reuses the same loaded models rather than
    # re-reading them from disk; standalone CLI use builds its own.
    if pipeline is None:
        pipeline = InferencePipeline(dataset=dataset)
    blacklist_pool = pd.read_csv(THREAT_INTEL_DIR / "blacklist_ips.csv")["ip"].tolist()
    rng = random.Random(seed)

    print(f"Replaying {len(test)} rows from {dataset} test set (synthetic demo IPs; delay={delay_seconds}s)...")
    for i, (_, row) in enumerate(test.iterrows()):
        if stop_event is not None and stop_event.is_set():
            print(f"Replay for {dataset} stopped early at row {i}/{len(test)}.")
            return
        feature_row = row.drop(labels=[BINARY_LABEL_COL, MULTICLASS_LABEL_COL]).to_dict()
        feature_row["src_ip"] = _sample_ip(rng, blacklist_rate=0.05, blacklist_pool=blacklist_pool)
        feature_row["dst_ip"] = _sample_ip(rng, blacklist_rate=0.0, blacklist_pool=[])
        feature_row["flow_ref"] = f"{dataset}-replay-{i}"

        result = pipeline.process(feature_row)
        true_label = pipeline.class_names[int(row[MULTICLASS_LABEL_COL])]

        print(
            f"[{i + 1}/{len(test)}] true={true_label} predicted={result['attack_label']} "
            f"flagged={result['anomaly_flagged']} severity={result['severity']} risk={result['risk_score']:.1f}"
        )
        time.sleep(delay_seconds)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay held-out test rows through the inference pipeline to simulate real-time traffic."
    )
    parser.add_argument("dataset", choices=["cicids", "unsw"])
    parser.add_argument("--n-rows", type=int, default=200)
    parser.add_argument("--delay", type=float, default=0.5)
    parser.add_argument("--no-shuffle", action="store_true")
    args = parser.parse_args()
    replay(args.dataset, n_rows=args.n_rows, delay_seconds=args.delay, shuffle=not args.no_shuffle)


if __name__ == "__main__":
    main()
