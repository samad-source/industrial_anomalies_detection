
from pathlib import Path
import sys
import io
import math
import struct
import wave

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# =========================================================
# PROJECT CONFIGURATION
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.model_loader import load_models
from app.services.simulation_engine import SimulationEngine
from app.services.simulation_monitor import monitor_simulation_batch
from app.services.alarm_manager import generate_alarm_events
from app.services.alarm_database import (
    initialize_database,
    save_alarms,
    get_all_alarms,
    get_known_alarm_ids,
    acknowledge_alarm as acknowledge_alarm_in_database
)
from app.components.facility_3d import render_facility_3d


st.set_page_config(
    page_title="Industrial Process Monitoring",
    page_icon="🏭",
    layout="wide"
)

MAX_ALARM_HISTORY = 500


# =========================================================
# AUDIO GENERATION
# =========================================================

def create_alarm_sound(consensus=False):
    """
    Generate a WAV notification.

    Single-model alarm: one short tone.
    Consensus alarm: three distinct tones.
    """

    sample_rate = 44100
    amplitude = 0.35

    if consensus:
        tone_sequence = [
            (660, 0.18),
            (880, 0.18),
            (1040, 0.22)
        ]
    else:
        tone_sequence = [
            (880, 0.35)
        ]

    audio_buffer = io.BytesIO()

    with wave.open(audio_buffer, "wb") as wav_file:

        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)

        frames = []

        for frequency, duration in tone_sequence:

            for index in range(int(sample_rate * duration)):

                time_value = index / sample_rate

                envelope = min(
                    time_value / 0.02,
                    1.0,
                    (duration - time_value) / 0.04
                )

                envelope = max(0.0, envelope)

                sample = (
                    amplitude
                    * envelope
                    * math.sin(
                        2 * math.pi * frequency * time_value
                    )
                )

                frames.append(
                    struct.pack(
                        "<h",
                        int(sample * 32767)
                    )
                )

            if consensus:

                silence_frames = int(sample_rate * 0.08)

                frames.extend(
                    [struct.pack("<h", 0)] * silence_frames
                )

        wav_file.writeframes(b"".join(frames))

    return audio_buffer.getvalue()


# =========================================================
# MODEL LOADING
# =========================================================

@st.cache_resource
def get_models():
    return load_models()


# =========================================================
# SESSION STATE
# =========================================================

if "models" not in st.session_state:
    st.session_state.models = get_models()

if "engine" not in st.session_state:
    engine = SimulationEngine(batch_size=1000)
    engine.load_normal_operation()
    st.session_state.engine = engine

if "history" not in st.session_state:
    st.session_state.history = []

if "latest_results" not in st.session_state:
    st.session_state.latest_results = pd.DataFrame()

if "scenario_name" not in st.session_state:
    st.session_state.scenario_name = "Normal Operation"

if "total_processed" not in st.session_state:
    st.session_state.total_processed = 0

# Initialize SQLite and restore alarm history when the app starts.
initialize_database()

if "alarms" not in st.session_state:
    # The database returns newest-first; the dashboard stores oldest-first.
    st.session_state.alarms = list(reversed(get_all_alarms()))

if "known_alarm_ids" not in st.session_state:
    st.session_state.known_alarm_ids = get_known_alarm_ids()

# Alarms waiting for a successful database write are kept here and retried.
if "pending_alarms" not in st.session_state:
    st.session_state.pending_alarms = []

if "alarm_database_error" not in st.session_state:
    st.session_state.alarm_database_error = None

if "alarm_sound_enabled" not in st.session_state:
    st.session_state.alarm_sound_enabled = True


# =========================================================
# ALARM ACKNOWLEDGEMENT CALLBACK
# =========================================================

def acknowledge_alarm(alarm_id):
    """
    Save an alarm acknowledgement to SQLite, then update the UI record.
    """

    try:
        database_updated = acknowledge_alarm_in_database(
            alarm_id=alarm_id,
            acknowledged_by="Operator"
        )

        if not database_updated:
            return

        acknowledged_at = pd.Timestamp.now().isoformat()

        for alarm in st.session_state.alarms:
            if alarm["alarm_id"] == alarm_id:
                alarm["status"] = "Acknowledged"
                alarm["acknowledged_at"] = acknowledged_at
                alarm["acknowledged_by"] = "Operator"
                break

        st.session_state.alarm_database_error = None

    except Exception as error:
        st.session_state.alarm_database_error = str(error)


def persist_pending_alarms():
    """
    Retry saving queued alarm events.

    Alarm IDs are added to known_alarm_ids only after SQLite confirms
    that the save operation completed successfully.
    """

    pending = st.session_state.pending_alarms

    if not pending:
        return

    try:
        save_alarms(pending)

        # INSERT OR IGNORE makes retries safe if some rows were already saved.
        st.session_state.known_alarm_ids.update(
            alarm["alarm_id"] for alarm in pending
        )

        st.session_state.pending_alarms = []
        st.session_state.alarm_database_error = None

    except Exception as error:
        # Keep the queue intact so the next simulation refresh retries it.
        st.session_state.alarm_database_error = str(error)


# =========================================================
# PAGE HEADER
# =========================================================

st.title("🏭 Industrial Process Monitoring")

st.caption(
    "Machine Learning-Based Anomaly Detection "
    "Using the Tennessee Eastman Process"
)

st.divider()


# =========================================================
# SIDEBAR - SIMULATION CONTROLS
# =========================================================

with st.sidebar:

    st.header("Simulation Controls")

    scenario_options = {
        "Normal Operation": 0
    }

    scenario_options.update({
        f"Fault Scenario {i:02d}": i
        for i in range(1, 25)
    })

    selected_scenario = st.selectbox(
        "Select Simulation Scenario",
        list(scenario_options.keys())
    )

    playback_speed = st.slider(
        "Simulation Speed (seconds)",
        min_value=1,
        max_value=5,
        value=2
    )

    st.divider()

    st.subheader("Alarm Settings")

    st.session_state.alarm_sound_enabled = st.checkbox(
        "Enable alarm sound",
        value=st.session_state.alarm_sound_enabled
    )

    if st.button(
        "Load Scenario",
        width="stretch"
    ):

        engine = st.session_state.engine
        scenario_number = scenario_options[selected_scenario]

        if scenario_number == 0:
            engine.load_normal_operation()
        else:
            engine.load_fault_scenario(scenario_number)

        engine.reset()

        st.session_state.scenario_name = selected_scenario
        st.session_state.history = []
        st.session_state.latest_results = pd.DataFrame()
        st.session_state.total_processed = 0

        # Alarm history is permanent and is not cleared when
        # a different simulation scenario is loaded.

        st.success("Scenario loaded successfully.")

    st.divider()

    st.subheader("Playback")

    col1, col2 = st.columns(2)

    with col1:

        if st.button(
            "Start",
            width="stretch"
        ):

            st.session_state.engine.start()
            st.rerun()

    with col2:

        if st.button(
            "Pause",
            width="stretch"
        ):

            st.session_state.engine.pause()
            st.rerun()

    col3, col4 = st.columns(2)

    with col3:

        if st.button(
            "Resume",
            width="stretch"
        ):

            try:
                st.session_state.engine.resume()

            except RuntimeError as error:
                st.warning(str(error))

            st.rerun()

    with col4:

        if st.button(
            "Reset",
            width="stretch"
        ):

            st.session_state.engine.reset()

            st.session_state.history = []
            st.session_state.latest_results = pd.DataFrame()
            st.session_state.total_processed = 0

            # Keep all previously recorded alarms when resetting
            # the simulation itself.

            st.rerun()

    st.divider()

    st.subheader("Model Status")

    st.success("Isolation Forest - Loaded")
    st.success("One-Class SVM - Loaded")
    st.success("Autoencoder - Loaded")

    st.divider()

    st.caption("Research Prototype")
    st.caption("Dataset: Tennessee Eastman Process")


# =========================================================
# SIMULATION OVERVIEW
# =========================================================

@st.fragment(run_every=playback_speed)
def simulation_overview():

    engine = st.session_state.engine
    models = st.session_state.models

    if engine.is_running:

        batch = engine.next_batch()

        if batch is not None:

            monitored = monitor_simulation_batch(
                models,
                batch
            )

            results = monitored["results"]

            st.session_state.latest_results = results

            processed = len(results)

            st.session_state.total_processed += processed

            st.session_state.history.append({
                "Observation": st.session_state.total_processed,
                "Isolation Forest": int(
                    results["isolation_forest_prediction"].sum()
                ),
                "One-Class SVM": int(
                    results["one_class_svm_prediction"].sum()
                ),
                "Autoencoder": int(
                    results["autoencoder_prediction"].sum()
                ),
                "Ensemble": int(
                    results["ensemble_alert"].sum()
                )
            })

            # -------------------------------------------------
            # GENERATE AND PERSIST NEW ALARM EVENTS
            # -------------------------------------------------

            # Include IDs already in the database and IDs waiting in
            # the retry queue to prevent duplicate generation.
            candidate_known_ids = set(
                st.session_state.known_alarm_ids
            )
            candidate_known_ids.update(
                alarm["alarm_id"]
                for alarm in st.session_state.pending_alarms
            )

            new_alarms = generate_alarm_events(
                results=results,
                scenario_name=st.session_state.scenario_name,
                known_alarm_ids=candidate_known_ids
            )

            if new_alarms:
                # Queue first. If SQLite is temporarily unavailable,
                # these events remain available for the next retry.
                st.session_state.pending_alarms.extend(new_alarms)

                # Display generated alerts immediately.
                st.session_state.alarms.extend(new_alarms)

                # -------------------------------------------------
                # ALARM SOUND NOTIFICATION
                # -------------------------------------------------

                if st.session_state.alarm_sound_enabled:

                    consensus_detected = any(
                        alarm["alarm_type"] == "CONSENSUS"
                        for alarm in new_alarms
                    )

                    st.audio(
                        create_alarm_sound(
                            consensus=consensus_detected
                        ),
                        format="audio/wav",
                        autoplay=True
                    )

                    consensus_count = sum(
                        1
                        for alarm in new_alarms
                        if alarm["alarm_type"] == "CONSENSUS"
                    )

                    single_model_count = (
                        len(new_alarms) - consensus_count
                    )

                    st.caption(
                        f"Alarm notification: {len(new_alarms)} "
                        f"new event(s) detected | "
                        f"Consensus: {consensus_count} | "
                        f"Single-model: {single_model_count}"
                    )

    # Retry queued alarm writes on every fragment refresh.
    # This also runs when the simulation is paused.
    persist_pending_alarms()

    # -----------------------------------------------------
    # SIMULATION STATUS
    # -----------------------------------------------------

    status = engine.get_status()

    progress = status["progress"]

    if status["is_running"]:
        simulation_status = "Running"

    elif progress >= 1:
        simulation_status = "Completed"

    elif progress > 0:
        simulation_status = "Paused"

    else:
        simulation_status = "Ready"

    # -----------------------------------------------------
    # SIMULATION OVERVIEW
    # -----------------------------------------------------

    st.subheader("Simulation Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Current Scenario",
            st.session_state.scenario_name
        )

    with col2:

        st.metric(
            "Observations Processed",
            f"{st.session_state.total_processed:,}"
        )

    with col3:

        st.metric(
            "Total Observations",
            f"{status['total_observations']:,}"
        )

    with col4:

        st.metric(
            "Simulation Status",
            simulation_status
        )

    st.progress(
        progress,
        text=f"Simulation Progress: {progress * 100:.2f}%"
    )


# =========================================================
# 3D FACILITY VISUALIZATION
# =========================================================

def facility_visualization():

    st.divider()

    st.subheader("Industrial Facility Visualization")

    st.caption(
        "Conceptual onshore oil and gas production and processing "
        "facility. The Tennessee Eastman Process benchmark drives "
        "the machine-learning monitoring system; it is not actual "
        "oil and gas operational telemetry."
    )

    render_facility_3d(height=600)

    st.divider()


# =========================================================
# ANOMALY ALARM DASHBOARD
# =========================================================

@st.fragment(run_every=playback_speed)
def alarm_dashboard():

    alarms = st.session_state.alarms

    st.subheader("🚨 Anomaly Alarm System")

    st.caption(
        "Notifications generated from machine-learning anomaly "
        "predictions. These are research alerts, not certified "
        "industrial safety alarms."
    )

    if st.session_state.alarm_database_error:
        st.error(
            "Alarm database write/update issue. Pending alarm events "
            "will be retried automatically while this app session remains "
            "active. Details: "
            + st.session_state.alarm_database_error
        )

    if st.session_state.pending_alarms:
        st.warning(
            f"{len(st.session_state.pending_alarms)} alarm event(s) "
            "are waiting to be saved to the database."
        )

    active_alarms = [
        alarm for alarm in alarms
        if alarm["status"] == "Active"
    ]

    acknowledged_alarms = [
        alarm for alarm in alarms
        if alarm["status"] == "Acknowledged"
    ]

    consensus_alarms = [
        alarm for alarm in alarms
        if alarm["alarm_type"] == "CONSENSUS"
    ]

    # -----------------------------------------------------
    # ALARM SUMMARY
    # -----------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Total Alarms",
            len(alarms)
        )

    with col2:

        st.metric(
            "Active Alarms",
            len(active_alarms)
        )

    with col3:

        st.metric(
            "Acknowledged",
            len(acknowledged_alarms)
        )

    with col4:

        st.metric(
            "Consensus Alarms",
            len(consensus_alarms)
        )

    st.divider()

    # -----------------------------------------------------
    # ACTIVE ALARM NOTIFICATIONS
    # -----------------------------------------------------

    st.markdown("#### Active Alarm Notifications")

    if active_alarms:

        for alarm in reversed(active_alarms[-10:]):

            if alarm["alarm_type"] == "CONSENSUS":

                st.error(
                    f"CONSENSUS ALERT — {alarm['scenario']} | "
                    f"Fault {alarm['fault_number']} | "
                    f"Sample {alarm['sample']} | "
                    f"{alarm['models_flagged']} models flagged"
                )

            else:

                st.warning(
                    f"SINGLE-MODEL ALERT — {alarm['scenario']} | "
                    f"Fault {alarm['fault_number']} | "
                    f"Sample {alarm['sample']}"
                )

            st.caption(
                "Detecting models: "
                + ", ".join(alarm["detecting_models"])
            )

            st.caption(
                f"Detected at: {alarm['detected_at']}"
            )

            # -------------------------------------------------
            # ACKNOWLEDGE INDIVIDUAL ALARM
            # -------------------------------------------------

            st.button(
                "Acknowledge Alarm",
                key=f"ack_{alarm['alarm_id']}",
                on_click=acknowledge_alarm,
                args=(alarm["alarm_id"],)
            )

    else:

        st.success("No active alarms at this time.")

    st.divider()

    # -----------------------------------------------------
    # ALARM HISTORY
    # -----------------------------------------------------

    st.markdown("#### Alarm History")

    if alarms:

        alarm_records = []

        for alarm in reversed(alarms):

            alarm_records.append({
                "Detected At": alarm["detected_at"],
                "Scenario": alarm["scenario"],
                "Fault Number": alarm["fault_number"],
                "Simulation Run": alarm["simulation_run"],
                "Sample": alarm["sample"],
                "Alarm Type": alarm["alarm_type"],
                "Models Flagged": alarm["models_flagged"],
                "Detecting Models": ", ".join(
                    alarm["detecting_models"]
                ),
                "Status": alarm["status"],
                "Acknowledged At": alarm.get(
                    "acknowledged_at"
                ),
                "Acknowledged By": alarm.get(
                    "acknowledged_by"
                )
            })

        st.dataframe(
            pd.DataFrame(alarm_records),
            width="stretch",
            hide_index=True
        )

    else:

        st.info(
            "Alarm history will appear when the simulation "
            "detects anomalous observations."
        )

    st.caption(
        "Alarm events are generated from the three model predictions. "
        "A consensus alert means at least two models flagged the "
        "same observation."
    )


# =========================================================
# ANOMALY DETECTION RESULTS
# =========================================================

@st.fragment(run_every=playback_speed)
def anomaly_detection_dashboard():

    latest = st.session_state.latest_results

    st.subheader("Anomaly Detection Results")

    if not latest.empty:

        col1, col2, col3, col4 = st.columns(4)

        with col1:

            st.metric(
                "Isolation Forest",
                f"{int(latest['isolation_forest_prediction'].sum()):,}"
            )

        with col2:

            st.metric(
                "One-Class SVM",
                f"{int(latest['one_class_svm_prediction'].sum()):,}"
            )

        with col3:

            st.metric(
                "Autoencoder",
                f"{int(latest['autoencoder_prediction'].sum()):,}"
            )

        with col4:

            st.metric(
                "Ensemble Alerts",
                f"{int(latest['ensemble_alert'].sum()):,}"
            )

    else:

        st.info(
            "Start the simulation to display anomaly detection results."
        )

    st.divider()

    # -----------------------------------------------------
    # DETECTION HISTORY AND RECENT RECORDS
    # -----------------------------------------------------

    history = pd.DataFrame(st.session_state.history)

    left_column, right_column = st.columns([1.15, 1])

    with left_column:

        st.subheader("Anomaly Detection History")

        if not history.empty:

            fig = go.Figure()

            for column in [
                "Isolation Forest",
                "One-Class SVM",
                "Autoencoder",
                "Ensemble"
            ]:

                fig.add_trace(
                    go.Scatter(
                        x=history["Observation"],
                        y=history[column],
                        mode="lines",
                        name=column
                    )
                )

            fig.update_layout(
                height=420,
                template="plotly_white",
                xaxis_title="Observations Processed",
                yaxis_title="Anomalies Detected",
                hovermode="x unified",
                legend=dict(
                    orientation="h",
                    y=1.12
                ),
                margin=dict(
                    l=20,
                    r=20,
                    t=35,
                    b=20
                )
            )

            st.plotly_chart(
                fig,
                width="stretch"
            )

        else:

            st.info(
                "Detection history will appear when simulation starts."
            )

    with right_column:

        st.subheader("Recent Detection Records")

        if not latest.empty:

            display_columns = [
                column for column in [
                    "faultNumber",
                    "simulationRun",
                    "sample",
                    "isolation_forest_prediction",
                    "one_class_svm_prediction",
                    "autoencoder_prediction",
                    "models_flagged",
                    "ensemble_alert"
                ]
                if column in latest.columns
            ]

            st.dataframe(
                latest[display_columns].tail(20),
                width="stretch",
                hide_index=True,
                height=420
            )

        else:

            st.info(
                "Detection records will appear when simulation starts."
            )

    # -----------------------------------------------------
    # COMPLETION MESSAGE
    # -----------------------------------------------------

    status = st.session_state.engine.get_status()

    if status["progress"] >= 1:

        st.success("Simulation completed successfully.")


# =========================================================
# MAIN DASHBOARD
# =========================================================

# Section 1: Automatically refreshing simulation overview
simulation_overview()

# Section 2: Static 3D facility visualization
facility_visualization()

# Section 3: Automatically refreshing alarm dashboard
alarm_dashboard()

# Section 4: Automatically refreshing anomaly detection results
anomaly_detection_dashboard()


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Industrial Process Monitoring | "
    "Isolation Forest | One-Class SVM | Autoencoder"
)
