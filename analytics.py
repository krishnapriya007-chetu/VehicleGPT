
import sqlite3
import pandas as pd

DB_PATH = "vehicle_data.db"


def load_records():
    """Load vehicle records from SQLite and prepare their data types."""
    with sqlite3.connect(DB_PATH) as connection:
        df = pd.read_sql_query(
            "SELECT * FROM vehicle_records ORDER BY timestamp",
            connection,
        )

    if df.empty:
        return df

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(
            df["timestamp"], errors="coerce"
        )

    numeric_columns = [
        "speed_kmh",
        "soc_percent",
        "battery_temp_c",
        "energy_wh_km",
        "distance_km",
    ]

    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    return df


def compare_energy_consumption():
    """
    Compare the first seven calendar days with the last seven
    calendar days in the available dataset.
    """
    df = load_records()

    if df.empty:
        return {"error": "No vehicle records are available."}

    required = ["timestamp", "energy_wh_km"]
    if not all(column in df.columns for column in required):
        return {"error": "Timestamp or energy data is missing."}

    df = df.dropna(subset=required)

    if df.empty:
        return {"error": "No valid energy readings are available."}

    start = df["timestamp"].min().normalize()
    end = df["timestamp"].max().normalize()
    last_start = end - pd.Timedelta(days=6)

    first = df[
        (df["timestamp"] >= start)
        & (df["timestamp"] < start + pd.Timedelta(days=7))
    ]

    last = df[
        (df["timestamp"] >= last_start)
        & (df["timestamp"] < end + pd.Timedelta(days=1))
    ]

    result = {
        "data_source": "SIMULATED",
        "comparison_method": (
            "First seven versus last seven calendar days"
        ),
        "first_period": (
            f"{start.date()} to "
            f"{(start + pd.Timedelta(days=6)).date()}"
        ),
        "last_period": f"{last_start.date()} to {end.date()}",
        "first_period_record_count": len(first),
        "last_period_record_count": len(last),
    }

    def compare_column(column, output_key):
        if column not in df.columns:
            result[output_key] = {
                "status": "Column unavailable"
            }
            return

        first_values = pd.to_numeric(
            first[column], errors="coerce"
        ).dropna()

        last_values = pd.to_numeric(
            last[column], errors="coerce"
        ).dropna()

        if first_values.empty or last_values.empty:
            result[output_key] = {
                "status": "Insufficient valid measurements"
            }
            return

        old = float(first_values.mean())
        new = float(last_values.mean())
        difference = new - old

        if difference > 0:
            direction = "increased"
        elif difference < 0:
            direction = "decreased"
        else:
            direction = "unchanged"

        percentage = (
            difference / abs(old) * 100 if old != 0 else None
        )

        result[output_key] = {
            "first_period_average": round(old, 2),
            "last_period_average": round(new, 2),
            "absolute_change": round(difference, 2),
            "percentage_change": (
                round(percentage, 2)
                if percentage is not None
                else None
            ),
            "direction": direction,
            "first_valid_count": len(first_values),
            "last_valid_count": len(last_values),
        }

    compare_column("energy_wh_km", "energy_consumption_wh_km")
    compare_column("speed_kmh", "speed_kmh")
    compare_column("battery_temp_c", "battery_temp_c")
    compare_column("soc_percent", "soc_percent")

    for column in [
        "speed_kmh",
        "battery_temp_c",
        "soc_percent",
    ]:
        key = f"energy_correlation_with_{column}"

        if column not in df.columns:
            result[key] = None
            continue

        paired = df[["energy_wh_km", column]].dropna()

        if len(paired) >= 3 and paired[column].nunique() > 1:
            correlation = paired["energy_wh_km"].corr(
                paired[column]
            )
            result[key] = (
                round(float(correlation), 3)
                if pd.notna(correlation)
                else None
            )
        else:
            result[key] = None

    result["causation_note"] = (
        "Period averages and correlations are descriptive. "
        "They do not establish causation or statistical significance."
    )

    return result


def investigate_energy_change():
    return compare_energy_consumption()


def get_battery_warnings():
    df = load_records()

    if df.empty:
        return {"message": "No vehicle records are available."}

    warning_mask = pd.Series(False, index=df.index)

    if "warning" in df.columns:
        warning_mask |= (
            df["warning"]
            .fillna("")
            .astype(str)
            .str.strip()
            .ne("")
        )

    if "diagnostic_code" in df.columns:
        warning_mask |= (
            df["diagnostic_code"]
            .fillna("")
            .astype(str)
            .str.strip()
            .ne("")
        )

    columns = [
        column
        for column in [
            "timestamp",
            "battery_temp_c",
            "warning",
            "diagnostic_code",
            "data_source",
        ]
        if column in df.columns
    ]

    return {
        "warning_count": int(warning_mask.sum()),
        "records": df.loc[warning_mask, columns].to_dict(
            orient="records"
        ),
        "data_source": "SIMULATED",
    }


def get_adas_events():
    df = load_records()

    if df.empty or "adas_event" not in df.columns:
        return {"message": "No ADAS data is available."}

    mask = (
        df["adas_event"]
        .fillna("")
        .astype(str)
        .str.strip()
        .ne("")
    )

    columns = [
        column
        for column in [
            "timestamp",
            "speed_kmh",
            "adas_event",
            "warning",
            "data_source",
        ]
        if column in df.columns
    ]

    return {
        "event_count": int(mask.sum()),
        "records": df.loc[mask, columns].to_dict(
            orient="records"
        ),
        "data_source": "SIMULATED",
    }


def detect_high_energy_consumption():
    df = load_records()

    if df.empty or "energy_wh_km" not in df.columns:
        return {"message": "No energy data is available."}

    valid = df.dropna(subset=["energy_wh_km"]).copy()

    if len(valid) < 3:
        return {
            "message": "Insufficient readings for anomaly screening."
        }

    mean = valid["energy_wh_km"].mean()
    std = valid["energy_wh_km"].std()

    if pd.isna(std) or std == 0:
        return {
            "message": "No variation detected in energy readings.",
            "data_source": "SIMULATED",
        }

    threshold = mean + std
    unusual = (
        valid[valid["energy_wh_km"] > threshold]
        .sort_values("energy_wh_km", ascending=False)
        .head(10)
    )

    columns = [
        column
        for column in [
            "timestamp",
            "energy_wh_km",
            "speed_kmh",
            "battery_temp_c",
            "data_source",
        ]
        if column in unusual.columns
    ]

    return {
        "overall_mean_wh_km": round(float(mean), 2),
        "standard_deviation_wh_km": round(float(std), 2),
        "threshold_wh_km": round(float(threshold), 2),
        "flagged_count_in_returned_results": len(unusual),
        "records": unusual[columns].to_dict(orient="records"),
        "method": "Mean + one standard deviation; exploratory only",
        "data_source": "SIMULATED",
    }
