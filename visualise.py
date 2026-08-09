"""
visualise.py

Plots each sensor's readings over time, with detected anomalies highlighted, and saves the results as PNG files.
"""

import os
import matplotlib.pyplot as plt

OUTPUT_DIR = "plots"

SENSOR_LABELS = {
    "fence_vibration": ("Fence Vibration Sensor", "Vibration (arbitrary units)"),
    "approach_motion": ("Approach Motion Sensor", "Motion count"),
    "cabinet_temperature": ("Cabinet Temperature Sensor", "Temperature (deg C)"),
}


def plot_sensor(df, sensor_name, save=True):
    """
    Plots one sensor's readings over time as a line, with flagged anomalies (from is_anomaly / is_dropout columns, if present) marked in red.
    """
    title, ylabel = SENSOR_LABELS.get(sensor_name, (sensor_name, "Value"))

    fig, ax = plt.subplots(figsize=(11, 4))

    ax.plot(df["timestamp"], df["value"], color="#2b6cb0", linewidth=1, label="Reading")

    flagged_mask = df.get("is_anomaly", False) | df.get("is_dropout", False)
    if flagged_mask.any():
        flagged = df[flagged_mask]
        ax.scatter(
            flagged["timestamp"], flagged["value"],
            color="#e53e3e", zorder=5, s=25, label="Flagged anomaly",
        )

    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper right")
    fig.autofmt_xdate()
    fig.tight_layout()

    if save:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        path = os.path.join(OUTPUT_DIR, f"{sensor_name}.png")
        fig.savefig(path, dpi=150)
        print(f"Saved: {path}")

    return fig


if __name__ == "__main__":
    from sensor_simulator import simulate_all_sensors
    from anomaly_detector import detect_anomalies, detect_dropout, SENSOR_SETTINGS

    data = simulate_all_sensors()

    for name, df in data.items():
        settings = SENSOR_SETTINGS.get(name, {})
        analysed = detect_anomalies(df, **settings)
        analysed = detect_dropout(analysed)
        plot_sensor(analysed, name)