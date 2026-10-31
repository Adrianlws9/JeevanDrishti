"""Transformation log: every pipeline step appends one row."""
from datetime import datetime
import pandas as pd


class TransformLog:
    def __init__(self):
        self.rows = []

    def add(self, dataset, step, action, rows_before=None, rows_after=None, rows_affected=None, detail=""):
        self.rows.append(dict(
            step_id=len(self.rows) + 1,
            timestamp=datetime.now().isoformat(timespec="seconds"),
            dataset=dataset, step=step, action=action,
            rows_before=rows_before, rows_after=rows_after, rows_affected=rows_affected, detail=detail,
        ))

    def frame(self):
        return pd.DataFrame(self.rows)
