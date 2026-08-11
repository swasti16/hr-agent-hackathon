"""
Per-stage latency timing + CSV logging for the HR agent pipeline.
"""
import time
import csv
import os
from contextlib import contextmanager
from datetime import datetime

_LOG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "reports", "latency_log.csv"
)

_FIELDNAMES = [
    "timestamp", "query", "shield", "greeting_check",
    "intent_classification", "retrieval",
    "reasoning_classify", "reasoning_generate", "total",
]


@contextmanager
def timed_stage(name: str, timings: dict):
    """Times a block and records elapsed seconds into timings[name]."""
    start = time.perf_counter()
    try:
        yield
    finally:
        timings[name] = round(time.perf_counter() - start, 4)


def log_latency(query: str, timings: dict) -> None:
    """Appends one row to reports/latency_log.csv. Creates file+header if absent."""
    os.makedirs(os.path.dirname(_LOG_PATH), exist_ok=True)
    file_exists = os.path.isfile(_LOG_PATH)

    row = {"timestamp": datetime.now().isoformat(timespec="seconds"), "query": query}
    row.update({k: timings.get(k, "") for k in _FIELDNAMES if k not in ("timestamp", "query")})

    with open(_LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
