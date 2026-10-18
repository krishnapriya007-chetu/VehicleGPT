
import json
import requests
import sqlite3
import pandas as pd
import streamlit as st
import plotly.express as px

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

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "llama3.2:1b"


# --------------------------------------------------
# DATA RETRIEVAL
# --------------------------------------------------

def retrieve_evidence(question):
    """
    Choose evidence based on the user's primary intent.

    Explicit energy-comparison questions take priority over
    incidental mentions of battery temperature or speed.
    """
    q = question.lower()

    adas_words = [
        "adas",
        "lane departure",
        "lane detection",
        "braking event",
        "automatic braking",
        "collision",
        "driver assistance",
    ]

    warning_words = [
        "battery warning",
        "warnings",
        "warning records",
        "diagnostic code",
        "fault code",
        "error code",
        "battery fault",
    ]

    anomaly_words = [
        "unusual",
        "anomaly",
        "anomalies",
        "abnormal",
        "outlier",
        "high consumption readings",
    ]

    energy_words = [
        "energy",
        "consumption",
        "efficiency",
        "range",
        "compare",
        "comparison",
        "increase",
        "increased",
        "decrease",
        "decreased",
        "speed",
        "correlation",
        "cause",
        "caused",
        "why",
        "period",
        "temperature",
        "soc",
        "state of charge",
        "battery temperature",
    ]

    # Route explicit requests for particular event records first.
    if any(word in q for word in adas_words):
        return get_adas_events()

    # Explicit requests for warning or fault records.
    if any(word in q for word in warning_words):
        return get_battery_warnings()

    # Anomaly detection only when it is the main request.
    if any(word in q for word in anomaly_words):
        return detect_high_energy_consumption()

    # Energy-related questions retrieve all comparison metrics.
    if any(word in q for word in energy_words):
        return compare_energy_consumption()

    return {
        "message": (
            "I could not identify the main question. "
            "Try asking about energy consumption, battery temperature, "
            "SOC, battery warnings, anomalies, or ADAS events."
        ),
        "available_topics": [
            "Energy consumption comparison",
            "Speed and battery temperature trends",
            "State of charge (SOC)",
            "Battery warnings and diagnostic codes",
            "High energy consumption readings",
            "ADAS events",
        ],
        "data_source": "SIMULATED",
    }


# --------------------------------------------------
# LOCAL LLM
# --------------------------------------------------

def ask_local_llm(question, evidence):
    system_prompt = """
You are VehicleGPT, an assistant for vehicle telemetry analysis.

Rules:
1. Use only the evidence provided in the user message.
2. Never invent values, dates, warning counts, or measurements.
3. Preserve every numerical value exactly as supplied.
4. Clearly distinguish period averages from individual warning readings.
5. A correlation does not establish causation.
6. Do not claim statistical significance unless it was tested.
7. The dataset is simulated, not real vehicle telemetry.
8. If evidence is missing, explicitly say it is unavailable.
9. Answer the user's actual question directly.
10. If energy comparison metrics are supplied, discuss energy,
    speed, battery temperature, and SOC when relevant.
11. Do not describe a temperature decrease as an increase, or vice versa.
12. Be concise and use clear bullet points when useful.
"""

    user_content = (
        f"Question:\n{question}\n\n"
        f"Retrieved evidence (JSON):\n"
        f"{json.dumps(evidence, default=str, indent=2)}"
    )

    payload = {
        "model": MODEL_NAME,
        "stream": False,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "options": {
            "temperature": 0,
        },
    }

    try:
        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()
        return data["message"]["content"]

    except requests.exceptions.ConnectionError:
        return (
            "I could not connect to Ollama. Open a terminal and run "
            "`ollama serve`, then try again."
        )

    except requests.exceptions.Timeout:
        return (
            "The local model took too long to respond. "
            "Please try again."
        )

    except (requests.exceptions.RequestException, KeyError, ValueError) as exc:
        return f"Could not generate an answer: {exc}"


# --------------------------------------------------
# FORMATTING
# --------------------------------------------------

def get_metric(metrics, key):
    value = metrics.get(key, {})
    return value if isinstance(value, dict) else {}


def format_metric(metrics, key, unit):
    item = get_metric(metrics, key)

    if "first_period_average" not in item:
        return "Not available"

    old = item["first_period_average"]
    new = item["last_period_average"]
    change = item["absolute_change"]
    pct = item["percentage_change"]
    direction = item["direction"]

    pct_text = (
        f"{pct:+.2f}%"
        if pct is not None
        else "percentage unavailable"
    )

    return (
        f"{old:.2f} → {new:.2f} {unit}\n"
        f"{direction.capitalize()} by {change:+.2f} {unit} "
        f"({pct_text})"
    )


def show_energy_comparison(metrics):
    """Display trusted calculations independently of the LLM."""
    if "error" in metrics:
        st.error(metrics["error"])
        return

    st.subheader("Verified comparison statistics")

    st.caption(
        "Source: simulated vehicle records. "
        "Values below are calculated in Python, not generated by the LLM."
    )

    st.write(
        f"**Comparison method:** {metrics.get('comparison_method', 'Unavailable')}"
    )

    st.write(
        f"**First period:** {metrics.get('first_period', 'Unavailable')}"
    )

    st.write(
        f"**Last period:** {metrics.get('last_period', 'Unavailable')}"
    )

    energy = get_metric(metrics, "energy_consumption_wh_km")
    speed = get_metric(metrics, "speed_kmh")
    temperature = get_metric(metrics, "battery_temp_c")
    soc = get_metric(metrics, "soc_percent")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        if "first_period_average" in energy:
            st.metric(
                "Energy (Wh/km)",
                f"{energy['last_period_average']:.2f}",
                f"{energy['percentage_change']:+.2f}%"
                if energy["percentage_change"] is not None
                else None,
            )
        else:
            st.metric("Energy (Wh/km)", "Unavailable")

    with c2:
        if "first_period_average" in speed:
            st.metric(
                "Average speed (km/h)",
                f"{speed['last_period_average']:.2f}",
                f"{speed['percentage_change']:+.2f}%"
                if speed["percentage_change"] is not None
                else None,
            )
        else:
            st.metric("Average speed", "Unavailable")

    with c3:
        if "first_period_average" in temperature:
            st.metric(
                "Battery temperature (°C)",
                f"{temperature['last_period_average']:.2f}",
                f"{temperature['percentage_change']:+.2f}%"
                if temperature["percentage_change"] is not None
                else None,
            )
        else:
            st.metric("Battery temperature", "Unavailable")

    with c4:
        if "first_period_average" in soc:
            st.metric(
                "Average SOC (%)",
                f"{soc['last_period_average']:.2f}",
                f"{soc['percentage_change']:+.2f}%"
                if soc["percentage_change"] is not None
                else None,
            )
        else:
            st.metric("Average SOC", "Unavailable")

    table_rows = []

    for label, key, unit in [
        ("Energy consumption", "energy_consumption_wh_km", "Wh/km"),
        ("Average speed", "speed_kmh", "km/h"),
        ("Battery temperature", "battery_temp_c", "°C"),
        ("State of charge", "soc_percent", "%"),
    ]:
        item = get_metric(metrics, key)

        if "first_period_average" not in item:
            continue

        table_rows.append({
            "Metric": label,
            "Unit": unit,
            "First-period average": item["first_period_average"],
            "Last-period average": item["last_period_average"],
            "Absolute change": item["absolute_change"],
            "Percentage change": (
                item["percentage_change"]
            ),
            "Direction": item["direction"],
        })

    if table_rows:
        st.dataframe(
            pd.DataFrame(table_rows),
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
st.caption(
    "On-device natural-language vehicle analytics "
    "using SQLite, Python, Streamlit, and Ollama."
)

st.warning(
    "This project currently uses SIMULATED vehicle data. "
    "Its findings are for demonstration and are not a diagnosis "
    "of a real vehicle."
)

try:
    df = load_records()
except (sqlite3.Error, Exception) as exc:
    st.error(f"Could not load the vehicle database: {exc}")
    st.stop()

if df.empty:
    st.error(
        "No vehicle records were found. Check that vehicle_data.db "
        "exists and contains the vehicle_records table."
    )
    st.stop()

st.subheader("Vehicle telemetry overview")

m1, m2, m3, m4 = st.columns(4)

with m1:
    st.metric("Records", f"{len(df):,}")

with m2:
    st.metric(
        "Average speed",
        (
            f"{df['speed_kmh'].mean():.2f} km/h"
            if "speed_kmh" in df.columns
            and df["speed_kmh"].notna().any()
            else "Unavailable"
        ),
    )

with m3:
    st.metric(
        "Average battery temperature",
        (
            f"{df['battery_temp_c'].mean():.2f} °C"
            if "battery_temp_c" in df.columns
            and df["battery_temp_c"].notna().any()
            else "Unavailable"
        ),
    )

with m4:
    st.metric(
        "Average energy consumption",
        (
            f"{df['energy_wh_km'].mean():.2f} Wh/km"
            if "energy_wh_km" in df.columns
            and df["energy_wh_km"].notna().any()
            else "Unavailable"
        ),
    )

# Charts
if "timestamp" in df.columns:
    st.subheader("Telemetry trends")

    chart_options = {
        "Energy consumption (Wh/km)": "energy_wh_km",
        "Battery temperature (°C)": "battery_temp_c",
        "State of charge (%)": "soc_percent",
        "Speed (km/h)": "speed_kmh",
    }

    available_options = {
        label: column
        for label, column in chart_options.items()
        if column in df.columns and df[column].notna().any()
    }

    selected_chart = st.selectbox(
        "Choose a telemetry metric",
        list(available_options.keys()),
    )

    selected_column = available_options[selected_chart]

    chart_df = df.dropna(
        subset=["timestamp", selected_column]
    )

    if not chart_df.empty:
        fig = px.line(
            chart_df,
            x="timestamp",
            y=selected_column,
            title=selected_chart,
            labels={
                "timestamp": "Timestamp",
                selected_column: selected_chart,
            },
        )

        fig.update_layout(
            hovermode="x unified",
            xaxis_title="Time",
            yaxis_title=selected_chart,
        )

        st.plotly_chart(fig, use_container_width=True)

with st.expander("View vehicle records"):
    st.dataframe(df, use_container_width=True, hide_index=True)


# --------------------------------------------------
# ENERGY ANALYSIS
# --------------------------------------------------

st.divider()
st.header("📊 Energy consumption analysis")

if st.button("Run energy comparison", key="run_energy_comparison"):
    st.session_state["energy_comparison"] = compare_energy_consumption()

if "energy_comparison" in st.session_state:
    show_energy_comparison(st.session_state["energy_comparison"])


# --------------------------------------------------
# CHAT
# --------------------------------------------------

st.divider()
st.header("💬 Ask VehicleGPT")

st.caption(
    "Ask about energy consumption, speed, battery temperature, "
    "SOC, warning records, unusual readings, or ADAS events."
)

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

for message in st.session_state["chat_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input(
    "Example: Why did energy consumption increase? Compare speed and battery temperature."
)

if question:
    st.session_state["chat_history"].append({
        "role": "user",
        "content": question,
    })

    with st.chat_message("user"):
        st.markdown(question)

    evidence = retrieve_evidence(question)

    # Display exact comparison values before the generated explanation.
    # This avoids relying on the language model to reproduce calculations.
    is_energy_question = (
        "error" not in evidence
        and "energy_consumption_wh_km" in evidence
    )

    with st.chat_message("assistant"):
        if is_energy_question:
            show_energy_comparison(evidence)

        with st.spinner("Analyzing the retrieved evidence..."):
            answer = ask_local_llm(question, evidence)

        st.markdown("### VehicleGPT explanation")
        st.markdown(answer)

        st.caption(
            "Evidence source: simulated SQLite vehicle records. "
            "The explanation is AI-generated; verify figures against "
            "the numerical results above."
        )

    st.session_state["chat_history"].append({
        "role": "assistant",
        "content": answer,
    })

if st.button("Clear chat history"):
    st.session_state["chat_history"] = []
    st.rerun()
