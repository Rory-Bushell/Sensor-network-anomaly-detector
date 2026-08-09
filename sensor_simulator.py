"""
sensor_simulator.py

Generates synthetic time-series data for a small perimeter security sensor network at a fictional critical infrastructure site:
- Fence vibration sensor   (detects climbing/cutting attempts)
- Approach motion sensor   (detects movement on the approach path)
- Cabinet temperature      (detects tamperin/equipment fault)

Each sensor produces "normal" readings with realistic noise, and a handful of anomalies are deliberately injected at known points so the detector (built in a later step) has something real to catch.
"""

import numpy as np
import pandas as pd

# Reproducible results — same anomalies appear every run
np.random.seed(42)

# one reading per simulated minute
NUM_READINGS = 500        
READING_INTERVAL_MIN = 1


def _timestamps(n=NUM_READINGS):
    return pd.date_range("2026-01-01 00:00", periods=n, freq="1min")


# Records exactly which reading indices were deliberately injected as anomalies for each sensor.
# Populated as each sensor is simulated, and used by evaluate.py to score detector performance against known ground truth, rather than just visually checking the plots.
GROUND_TRUTH_RANGES = {}


def simulate_vibration_sensor():
    """
    Fence vibration sensor. 
    Normally near to zero with small ambient noise (wind, passing traffic). 
    Someone climbing the fence produces a sharp, sustained spike.
    """
    baseline = np.random.normal(loc=0.05, scale=0.03, size=NUM_READINGS)
    baseline = np.clip(baseline, 0, None)

    df = pd.DataFrame({
        "timestamp": _timestamps(),
        "value": baseline,
    })

    # Inject someone climbing the fence: sharp spike sustained over a few readings
    climb_start = 180
    climb_end = climb_start + 4
    df.loc[climb_start:climb_end, "value"] = np.random.uniform(0.8, 1.2, size=5)
    GROUND_TRUTH_RANGES["fence_vibration"] = [(climb_start, climb_end)]

    return df


def simulate_motion_sensor():
    """
    Approach-path motion sensor. 
    Normally reports small sporadic counts(wildlife, staff patrols). 
    A sensor dropout (e.g. disconnected or disabled by an intruder) shows as a flat run of zero readings where some activity would normally be expected.
    """
    baseline = np.random.poisson(lam=1.0, size=NUM_READINGS)

    df = pd.DataFrame({
        "timestamp": _timestamps(),
        "value": baseline,
    })

    # Inject a dropout: sensor goes silent for an extended period
    dropout_start = 300
    dropout_length = 40
    dropout_end = dropout_start + dropout_length
    df.loc[dropout_start:dropout_end, "value"] = 0
    GROUND_TRUTH_RANGES["approach_motion"] = [(dropout_start, dropout_end)]

    return df


def simulate_temperature_sensor():
    """
    Equipment cabinet temperature sensor. 
    Normally stable around 20C with slow daily drift.
    A cabinet intrusion (panel opened, equipment fault) produces a rapid spike.
    """

    # slow daily warm-up
    drift = np.linspace(0, 1.5, NUM_READINGS)  
    noise = np.random.normal(loc=0, scale=0.2, size=NUM_READINGS)
    baseline = 20 + drift + noise

    df = pd.DataFrame({
        "timestamp": _timestamps(),
        "value": baseline,
    })

    # Inject a cabinet intrusion spike
    spike_start = 420
    spike_end = spike_start + 6
    df.loc[spike_start:spike_end, "value"] = np.random.uniform(28, 32, size=7)
    GROUND_TRUTH_RANGES["cabinet_temperature"] = [(spike_start, spike_end)]

    return df


def get_ground_truth_mask(sensor_name, length=NUM_READINGS):
    """
    Returns a boolean array, one entry per reading, True where that reading was a deliberately injected anomaly. 
    Requires the corresponding simulate_*_sensor() function to have been called first (simulate_all_sensors() does this).
    """
    mask = np.zeros(length, dtype=bool)
    for start, end in GROUND_TRUTH_RANGES.get(sensor_name, []):
        mask[start:end + 1] = True
    return mask


def simulate_all_sensors():
    """Returns a dictionary of sensor_name -> DataFrame."""
    return {
        "fence_vibration": simulate_vibration_sensor(),
        "approach_motion": simulate_motion_sensor(),
        "cabinet_temperature": simulate_temperature_sensor(),
    }


def _formatted_preview(df):
    """
    Returns a copy of the DataFrame with the timestamp split into separate Date and Time columns, and the index labelled as 'Reading #' for a clearer printed preview.
    """
    preview = df.copy()
    preview.insert(0, "Date", preview["timestamp"].dt.date)
    preview.insert(1, "Time", preview["timestamp"].dt.time)
    preview = preview.drop(columns=["timestamp"])
    preview.index.name = "Reading #"
    return preview


if __name__ == "__main__":
    data = simulate_all_sensors()
    for name, df in data.items():
        print(f"\n{name}: {len(df)} readings")
        print(_formatted_preview(df).head())