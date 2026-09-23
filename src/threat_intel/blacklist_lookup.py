from __future__ import annotations

from functools import lru_cache
from typing import Optional

import pandas as pd

from src.common.config import THREAT_INTEL_DIR

# Sample/placeholder blacklist using RFC 5737 documentation IP ranges -- not
# real threat data. In production this would sync from a live feed (e.g.
# abuse.ch, AlienVault OTX) on the "continuous retraining" schedule.
BLACKLIST_PATH = THREAT_INTEL_DIR / "blacklist_ips.csv"


@lru_cache(maxsize=1)
def _load_blacklist() -> dict:
    df = pd.read_csv(BLACKLIST_PATH)
    return dict(zip(df["ip"], df["reason"]))


def check_ip(ip: str) -> Optional[dict]:
    reason = _load_blacklist().get(ip)
    return {"ip": ip, "reason": reason} if reason else None
