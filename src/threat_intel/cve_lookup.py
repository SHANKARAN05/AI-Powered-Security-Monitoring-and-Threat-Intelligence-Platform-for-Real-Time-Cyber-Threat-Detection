from __future__ import annotations

import json
from functools import lru_cache

from src.common.config import THREAT_INTEL_DIR

CVE_LOOKUP_PATH = THREAT_INTEL_DIR / "cve_lookup.json"
_FALLBACK = {"description": "Unclassified/uncommon attack category.", "cwe_refs": [], "cve_refs": []}


@lru_cache(maxsize=1)
def _load_lookup() -> dict:
    with open(CVE_LOOKUP_PATH) as f:
        return json.load(f)


def lookup(attack_label: str) -> dict:
    return _load_lookup().get(attack_label, _FALLBACK)
