"""
Lightweight local storage standing in for the diagram's "Embedding store"
and the "Decision + audit log" box.

For this demo/code-video build, both are flat files under data/ -- no database
server needed to run it. Swapping these for a real DB later only touches this
file, not the Streamlit UI or the face model.
"""

import csv
import json
import os
from datetime import datetime, timezone

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
EMBEDDINGS_PATH = os.path.join(DATA_DIR, "embeddings.json")
AUDIT_LOG_PATH = os.path.join(DATA_DIR, "audit_log.csv")
AUDIT_FIELDS = ["timestamp", "order_id", "tier", "distance", "threshold", "decision"]


def _ensure_data_dir():
    os.makedirs(DATA_DIR, exist_ok=True)


def _load_all_embeddings() -> dict:
    _ensure_data_dir()
    if not os.path.exists(EMBEDDINGS_PATH):
        return {}
    with open(EMBEDDINGS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def enroll_customer(order_id: str, name: str, embedding) -> None:
    """Store a customer's reference embedding, keyed by order ID."""
    records = _load_all_embeddings()
    records[order_id] = {
        "name": name,
        "embedding": embedding.tolist(),
        "enrolled_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(EMBEDDINGS_PATH, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)


def get_enrollment(order_id: str):
    """Return {"name":..., "embedding": np.ndarray, "enrolled_at":...} or None."""
    import numpy as np

    records = _load_all_embeddings()
    rec = records.get(order_id)
    if rec is None:
        return None
    return {
        "name": rec["name"],
        "embedding": np.array(rec["embedding"]),
        "enrolled_at": rec["enrolled_at"],
    }


def list_enrollments() -> dict:
    return _load_all_embeddings()


def log_verification(order_id: str, tier: str, distance: float, threshold: float, decision: str) -> None:
    """Append one row to the audit log -- every verification attempt, accepted or not."""
    _ensure_data_dir()
    is_new = not os.path.exists(AUDIT_LOG_PATH)
    with open(AUDIT_LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=AUDIT_FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "order_id": order_id,
                "tier": tier,
                "distance": f"{distance:.4f}",
                "threshold": threshold,
                "decision": decision,
            }
        )


def read_audit_log():
    """Return the audit log as a list of dict rows (most recent last)."""
    if not os.path.exists(AUDIT_LOG_PATH):
        return []
    with open(AUDIT_LOG_PATH, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))
