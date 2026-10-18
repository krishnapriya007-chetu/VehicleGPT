

import sqlite3
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "vehicle_data.db"
CSV_PATH = BASE_DIR / "vehicle_data.csv"


def initialize_database():
    """Create the SQLite table from the bundled CSV if needed."""

    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {CSV_PATH}"
        )

    with sqlite3.connect(DB_PATH) as connection:
        cursor = connection.cursor()

        cursor.execute("""
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
              AND name = 'vehicle_records'
        """)

        table_exists = cursor.fetchone() is not None

        if table_exists:
            cursor.execute("SELECT COUNT(*) FROM vehicle_records")
            row_count = cursor.fetchone()[0]

            if row_count > 0:
                return

        df = pd.read_csv(CSV_PATH)

        # Remove an exported DataFrame index, if present.
        df = df.loc[
            :,
            ~df.columns.astype(str).str.match(r"^Unnamed:")
        ]

        if "timestamp" not in df.columns:
            raise ValueError(
                "The CSV must contain a timestamp column."
            )

        if df.empty:
            raise ValueError(
                "The vehicle dataset contains no records."
            )

        df.to_sql(
            "vehicle_records",
            connection,
            if_exists="replace",
            index=False,
        )

        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_vehicle_timestamp "
            "ON vehicle_records(timestamp)"
        )

        connection.commit()
