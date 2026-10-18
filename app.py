
import json
import sqlite3
from pathlib import Path

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

from database import initialize_database
from analytics import (
    load_records,
    compare_energy_consumption,
    get_battery_warnings,
    get_adas_events,
    detect_high_energy_consumption,
)

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="VehicleGPT",
    page_icon="🚗",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "llama3.2:1b"


# --------------------------------------------------
# DATABASE INITIALIZATION
# --------------------------------------------------

try:
    initialize_database()
    df = load_records()
except Exception as exc:
    st.error(f"Could not initialize or load the database: {exc}")
    st.stop()

if df.empty:
    st.error(
        "No vehicle records were found. Check vehicle_data.csv "
        "and the vehicle_records database table."
    )
    st.stop()


# --------------------------------------------------
# EVIDENCE RETRIEVAL
# --------------------------------------------------

def retrieve_evidence(question):
    q = question.lower()

    adas_words = [
        "adas", "lane departure", "lane detection",
        "braking event", "automatic braking",
        "collision", "driver assistance",
    ]

    warning_words = [
        "battery warning", "warning records",
        "diagnostic code", "fault code",
        "error code", "battery fault",
        "show warnings", "list warnings",
    ]

    anomaly_words = [
        "anomaly", "anomalies", "outlier",
        "unusual readings", "abnormal readings",
        "high consumption readings",
    ]

    energy_words = [
        "energy", "consumption", "efficiency",
        "range", "compare", "comparison",
        "increase", "increased", "decrease",
        "decreased", "speed", "correlation",
        "cause", "caused", "why", "period",
        "temperature", "soc", "state of charge",
        "battery temperature",
    ]

    if any(word in q for word in adas_words):
        return get_adas_events()

    # Explicit requests for warning records take priority.
    if any(word in q for word in warning_words):
        return get_battery_warnings()

    if any(word in q for word in anomaly_words):
        return detect_high_energy_consumption()

    if any(word in q for word in energy_words):
        return compare_energy_consumption()

    return {
        "message": (
            "I could not identify the main question. "
            "Ask about energy consumption, speed, battery temperature, "
            "SOC, warnings, anomalies, or ADAS events."
        ),
        "available_topics": [
            "Energy consumption comparison",
            "Speed and battery temperature",
            "State of charge",
            "Battery warnings and diagnostic codes",
            "High energy consumption readings",
            "ADAS events",
        ],
        "data_source": "SIMULATED",
    }


# --------------------------------------------------
# LOCAL LANGUAGE MODEL
# --------------------------------------------------

def ask_local_llm(question, evidence):
    system_prompt = """
You are VehicleGPT, an assistant for vehicle telemetry analysis.

Rules:
1. Use only the supplied evidence.
2. Never invent values, dates, events, or measurements.
3. Preserve numerical values exactly.
4. Distinguish individual warning readings from period averages.
5. Correlation does not establish causation.
6. Do not claim statistical significance or insignificance
   unless a suitable statistical test has been performed.
7. The dataset is simulated, not real vehicle telemetry.
8. State clearly when evidence is unavailable.
9. Answer the question directly.
10. Describe observed changes separately from explanations of causes.
11. Do not call a correlation weak, moderate, or strong as
    though the classification were definitive.
12. Do not claim two variables are unrelated merely because
    their averages move in opposite directions.
"""

    payload = {
        "model": MODEL_NAME,
        "stream": False,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    f"Question:\n{question}\n\n"
                    "Evidence in JSON format:\n"
                    + json.dumps(evidence, default=str, indent=2)
                ),
            },
        ],
        "options": {"temperature": 0},
    }

    try:
        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=120,
        )
        response.raise_for_status()
        return response.json()["message"]["content"]

    except requests.exceptions.ConnectionError:
        return (
            "The AI model is not reachable from this hosted app. "
            "The numerical analytics still work, but the chat requires "
            "a remotely accessible model API or a separately hosted "
            "Ollama instance. Your laptop's localhost is not accessible "
            "from Streamlit Cloud."
        )

    except requests.exceptions.Timeout:
        return "The model request timed out. Please try again."

    except (
        requests.exceptions.RequestException,
        KeyError,
        ValueError,
    ) as exc:
        return f"The AI response could not be generated: {exc}"


# --------------------------------------------------
# VERIFIED ANALYTICS DISPLAY
# --------------------------------------------------

def show_energy_comparison(metrics):
    if "error" in metrics:
        st.error(metrics["error"])
        return

    st.subheader("Verified comparison statistics")

    st.caption(
        "Values are calculated in Python from simulated vehicle records."
    )

    st.write(
        f"**Method:** {metrics.get('comparison_method', 'Unavailable')}"
    )
    st.write(
        f"**First period:** {metrics.get('first_period', 'Unavailable')}"
    )
    st.write(
        f"**Last period:** {metrics.get('last_period', 'Unavailable')}"
    )

    definitions = [
        ("Energy consumption", "energy_consumption_wh_km", "Wh/km"),
        ("Average speed", "speed_kmh", "km/h"),
        ("Battery temperature", "battery_temp_c", "°C"),
        ("Average SOC", "soc_percent", "%"),
    ]

    rows = []

    for label, key, unit in definitions:
        item = metrics.get(key, {})

        if not isinstance(item, dict):
            continue

        if "first_period_average" not in item:
            continue

        old = item["first_period_average"]
        new = item["last_period_average"]
        change = item["absolute_change"]
        pct = item["percentage_change"]

        rows.append({
            "Metric": label,
            "Unit": unit,
            "First-period average": old,
            "Last-period average": new,
            "Absolute change": change,
            "Percentage change": pct,
            "Direction": item["direction"],
        })

    if not rows:
        st.warning("No valid comparison metrics are available.")
        return

    table = pd.DataFrame(rows)

    metric_columns = st.columns(len(rows))

    for container, row in zip(metric_columns, rows):
        with container:
            pct = row["Percentage change"]
            st.metric(
                row["Metric"],
                f"{row['Last-period average']:.2f} {row['Unit']}",
                (
                    f"{pct:+.2f}%"
                    if pct is not None
                    else None
                ),
            )

    st.dataframe(
        table,
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Energy correlations")

    correlation_rows = []

    for column, label in [
        ("speed_kmh", "Speed"),
        ("battery_temp_c", "Battery temperature"),
        ("soc_percent", "State of charge"),
    ]:
        value = metrics.get(
            f"energy_correlation_with_{column}"
        )

        correlation_rows.append({
            "Variable": label,
            "Correlation with energy consumption": (
                value if value is not None else "Unavailable"
            ),
        })

    st.dataframe(
        pd.DataFrame(correlation_rows),
        use_container_width=True,
        hide_index=True,
    )

    st.info(
        metrics.get(
            "causation_note",
            "Correlation alone does not establish causation.",
        )
    )


# --------------------------------------------------
# DASHBOARD
# --------------------------------------------------

st.title("🚗 VehicleGPT")

st.markdown(
    "Natural-language vehicle analytics using Python, SQLite, "
    "Streamlit, and a language model."
)

st.warning(
    "This application uses SIMULATED vehicle data. "
    "Its findings are for demonstration and are not a diagnosis "
    "of a real vehicle."
)

st.header("Vehicle telemetry overview")

overview = st.columns(4)

overview_metrics = [
    ("Total records", f"{len(df):,}"),
    (
        "Average speed",
        (
            f"{df['speed_kmh'].mean():.2f} km/h"
            if "speed_kmh" in df.columns
            and df["speed_kmh"].notna().any()
            else "Unavailable"
        ),
    ),
    (
        "Average battery temperature",
        (
            f"{df['battery_temp_c'].mean():.2f} °C"
            if "battery_temp_c" in df.columns
            and df["battery_temp_c"].notna().any()
            else "Unavailable"
        ),
    ),
    (
        "Average energy consumption",
        (
            f"{df['energy_wh_km'].mean():.2f} Wh/km"
            if "energy_wh_km" in df.columns
            and df["energy_wh_km"].notna().any()
            else "Unavailable"
        ),
    ),
]

for container, (label, value) in zip(overview, overview_metrics):
    container.metric(label, value)


# --------------------------------------------------
# TELEMETRY CHARTS
# --------------------------------------------------

st.header("Telemetry trends")

chart_options = {
    "Energy consumption (Wh/km)": "energy_wh_km",
    "Battery temperature (°C)": "battery_temp_c",
    "State of charge (%)": "soc_percent",
    "Speed (km/h)": "speed_kmh",
}

available_charts = {
    label: column
    for label, column in chart_options.items()
    if column in df.columns and df[column].notna().any()
}

if available_charts and "timestamp" in df.columns:
    selected_label = st.selectbox(
        "Select a metric",
        list(available_charts.keys()),
    )

    selected_column = available_charts[selected_label]
    chart_df = df.dropna(
        subset=["timestamp", selected_column]
    )

    if not chart_df.empty:
        fig = px.line(
            chart_df,
            x="timestamp",
            y=selected_column,
            title=selected_label,
            labels={
                "timestamp": "Timestamp",
                selected_column: selected_label,
            },
        )
        fig.update_layout(hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No chartable telemetry data is available.")

with st.expander("View vehicle records"):
    st.dataframe(df, use_container_width=True, hide_index=True)


# --------------------------------------------------
# ENERGY COMPARISON
# --------------------------------------------------

st.divider()
st.header("📊 Energy consumption analysis")

if st.button("Run energy comparison"):
    st.session_state["energy_comparison"] = (
        compare_energy_consumption()
    )

if "energy_comparison" in st.session_state:
    show_energy_comparison(
        st.session_state["energy_comparison"]
    )


# --------------------------------------------------
# CHAT INTERFACE
# --------------------------------------------------

st.divider()
st.header("💬 Ask VehicleGPT")

st.caption(
    "Ask about energy consumption, speed, battery temperature, SOC, "
    "battery warnings, unusual readings, or ADAS events."
)

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

for message in st.session_state["chat_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input(
    "Example: Why did energy consumption increase?"
)

if question:
    st.session_state["chat_history"].append({
        "role": "user",
        "content": question,
    })

    with st.chat_message("user"):
        st.markdown(question)

    evidence = retrieve_evidence(question)

    with st.chat_message("assistant"):
        # Show calculated facts separately from AI-generated prose.
        if (
            "error" not in evidence
            and "energy_consumption_wh_km" in evidence
        ):
            show_energy_comparison(evidence)

        with st.spinner("Analyzing vehicle data..."):
            answer = ask_local_llm(question, evidence)

        st.subheader("VehicleGPT explanation")
        st.markdown(answer)

        st.caption(
            "Evidence source: simulated SQLite vehicle records. "
            "AI explanations may be unavailable if the model is not hosted."
        )

    st.session_state["chat_history"].append({
        "role": "assistant",
        "content": answer,
    })

if st.button("Clear chat history"):
    st.session_state["chat_history"] = []
    st.rerun()
