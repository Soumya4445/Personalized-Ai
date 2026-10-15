

from __future__ import annotations

from datetime import datetime
from typing import Dict, List

import pandas as pd


class FeedbackLoop:
    def __init__(self):
        self.new_feedback: List[Dict] = []

    def record_feedback(self, user_id: str, media_id: str, *, liked: bool = False,
                         rating: float = None, watch_time_pct: float = 0.0,
                         skipped: bool = False, shared: bool = False,
                         comment: str = "") -> Dict:
        entry = {
            "user_id": user_id,
            "media_id": media_id,
            "timestamp": datetime.now(),
            "rating": rating,
            "liked": liked,
            "watch_time_pct": watch_time_pct,
            "skipped": skipped,
            "shared": shared,
            "comment": comment,
            "mood_input": "",
        }
        self.new_feedback.append(entry)
        return entry

    def as_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.new_feedback)

    def merge_with_history(self, interactions: pd.DataFrame) -> pd.DataFrame:
        """Appends freshly collected feedback onto the historical interaction
        log, ready to be re-fed into preprocessing + model retraining."""
        new_df = self.as_dataframe()
        if new_df.empty:
            return interactions
        merged = pd.concat([interactions, new_df], ignore_index=True, sort=False)
        self.new_feedback.clear()
        return merged

    def should_retrain(self, threshold: int = 200) -> bool:
        """Simple trigger: retrain once enough new feedback has accumulated.
        In production this could instead be a scheduled (e.g. nightly) job."""
        return len(self.new_feedback) >= threshold


if __name__ == "__main__":
    fl = FeedbackLoop()
    fl.record_feedback("U0001", "M0002", liked=True, rating=5, watch_time_pct=0.95,
                        comment="Loved it, very motivating!")
    fl.record_feedback("U0001", "M0010", liked=False, rating=2, watch_time_pct=0.1,
                        skipped=True)
    print(fl.as_dataframe())
    print("Should retrain?", fl.should_retrain(threshold=2))
