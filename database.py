
import sqlite3
import pandas as pd
from pathlib import Path

DATA_FILE = Path("data/vehicle_data.csv")
DB_FILE = Path("vehicle_data.db")


def load_and_validate_data():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}. "
            "Place vehicle_data.csv inside the data folder."
        )

    df = pd.read_csv(DATA_FILE)

    required_columns = [
        "timestamp",
        "speed_kmh",
        "soc_percent",
        "battery_temp_c",
        "energy_wh_km",
        "distance_km",
        "diagnostic_code",
        "adas_event",
        "warning",
        "data_source",
    ]

    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing columns: {sorted(missing_columns)}")

    # Parse timestamps and reject invalid or duplicate records.
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df = df.drop_duplicates(subset=["timestamp"])
    df = df.sort_values("timestamp")

    # Convert measurement columns to numeric values.
    numeric_columns = [
        "speed_kmh",
        "soc_percent",
        "battery_temp_c",
        "energy_wh_km",
        "distance_km",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    # Keep records with valid essential measurements.
    df = df.dropna(subset=numeric_columns)

    # Basic plausibility checks.
    df = df[
        df["speed_kmh"].between(0, 250)
        & df["soc_percent"].between(0, 100)
        & df["battery_temp_c"].between(-30, 100)
        & df["energy_wh_km"].between(0, 1000)
        & (df["distance_km"] >= 0)
    ].copy()

    # Store timestamps as ISO-formatted text in SQLite.
    df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")

    for column in ["diagnostic_code", "adas_event", "warning", "data_source"]:
        df[column] = df[column].fillna("")

    return df


def create_database():
    df = load_and_validate_data()

    with sqlite3.connect(DB_FILE) as connection:
        df.to_sql("vehicle_records", connection, if_exists="replace", index=False)

        connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_vehicle_timestamp
            ON vehicle_records(timestamp)
        """)

        total = connection.execute(
            "SELECT COUNT(*) FROM vehicle_records"
        ).fetchone()[0]

        warnings = connection.execute("""
            SELECT COUNT(*)
            FROM vehicle_records
            WHERE warning != ''
        """).fetchone()[0]

    print("Database created successfully!")
    print(f"Database: {DB_FILE.resolve()}")
    print(f"Valid records stored: {total}")
    print(f"Warning records: {warnings}")


if __name__ == "__main__":
    create_database()
