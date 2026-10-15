

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Optional

import pandas as pd


class RecSysDatabase:
    def __init__(self, db_path: str = "motivational_recsys.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self._init_schema()

    def _init_schema(self):
        cur = self.conn.cursor()
        cur.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            user_id TEXT PRIMARY KEY,
            age INTEGER,
            gender TEXT,
            profession TEXT,
            interests TEXT
        );
        CREATE TABLE IF NOT EXISTS media (
            media_id TEXT PRIMARY KEY,
            title TEXT,
            media_type TEXT,
            category TEXT,
            tags TEXT,
            description TEXT,
            duration_min REAL
        );
        CREATE TABLE IF NOT EXISTS interactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            media_id TEXT,
            timestamp TEXT,
            rating REAL,
            liked INTEGER,
            watch_time_pct REAL,
            skipped INTEGER,
            comment TEXT,
            mood_input TEXT
        );
        CREATE TABLE IF NOT EXISTS model_registry (
            model_name TEXT PRIMARY KEY,
            trained_at TEXT,
            metrics TEXT
        );
        """)
        self.conn.commit()

    # ------------------------------------------------------------------ #
    def save_users(self, users: pd.DataFrame):
        df = users.copy()
        df["interests"] = df["interests"].apply(json.dumps)
        df.to_sql("users", self.conn, if_exists="replace", index=False)

    def save_media(self, media: pd.DataFrame):
        df = media.copy()
        df["tags"] = df["tags"].apply(json.dumps)
        df.to_sql("media", self.conn, if_exists="replace", index=False)

    def save_interactions(self, interactions: pd.DataFrame):
        df = interactions.copy()
        for col in ("liked", "skipped"):
            if col in df.columns:
                df[col] = df[col].astype(int)
        df["timestamp"] = df["timestamp"].astype(str)
        keep_cols = ["user_id", "media_id", "timestamp", "rating", "liked",
                     "watch_time_pct", "skipped", "comment", "mood_input"]
        df = df[[c for c in keep_cols if c in df.columns]]
        df.to_sql("interactions", self.conn, if_exists="append", index=False)

    def log_model_training(self, model_name: str, metrics: dict):
        cur = self.conn.cursor()
        cur.execute(
            "INSERT OR REPLACE INTO model_registry (model_name, trained_at, metrics) "
            "VALUES (?, datetime('now'), ?)",
            (model_name, json.dumps(metrics)),
        )
        self.conn.commit()

    # ------------------------------------------------------------------ #
    def load_users(self) -> pd.DataFrame:
        df = pd.read_sql("SELECT * FROM users", self.conn)
        if not df.empty:
            df["interests"] = df["interests"].apply(json.loads)
        return df

    def load_media(self) -> pd.DataFrame:
        df = pd.read_sql("SELECT * FROM media", self.conn)
        if not df.empty:
            df["tags"] = df["tags"].apply(json.loads)
        return df

    def load_interactions(self) -> pd.DataFrame:
        return pd.read_sql("SELECT * FROM interactions", self.conn)

    def close(self):
        self.conn.close()


if __name__ == "__main__":
    from data_collection import DataCollector

    raw = DataCollector().collect_all(n_users=10, n_media=30, n_interactions=100)
    db = RecSysDatabase(":memory:")
    db.save_users(raw["users"])
    db.save_media(raw["media"])
    db.save_interactions(raw["interactions"])
    db.log_model_training("hybrid_v1", {"precision_at_10": 0.42, "rmse": 0.87})

    print("Users loaded back:", db.load_users().shape)
    print("Media loaded back:", db.load_media().shape)
    print("Interactions loaded back:", db.load_interactions().shape)
