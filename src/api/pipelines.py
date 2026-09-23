from __future__ import annotations

from typing import Dict, List

from src.pipeline.inference_pipeline import InferencePipeline

_PIPELINES: Dict[str, InferencePipeline] = {}


def get_pipeline(dataset: str) -> InferencePipeline:
    if dataset not in _PIPELINES:
        _PIPELINES[dataset] = InferencePipeline(dataset=dataset)
    return _PIPELINES[dataset]


def loaded_datasets() -> List[str]:
    return list(_PIPELINES.keys())
