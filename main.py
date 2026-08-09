"""
main.py

Runs the full sensor network anomaly detection pipeline end to end:

  1. Simulate sensor data (3 sensors, with known anomalies injected)
  2. Run anomaly detection on each sensor, with per-sensor tuned settings
  3. Print a summary of what was flagged and when
  4. Save a plot per sensor to plots/, with anomalies highlighted
  5. Evaluate detector performance against known ground truth, and
     compare against an ML (Isolation Forest) alternative

Run this file directly to reproduce the full demo:
    python main.py
"""

import pandas as pd

from sensor_simulator import simulate_all_sensors
from anomaly_detector import detect_anomalies, detect_dropout, summarise_anomalies, SENSOR_SETTINGS
from visualise import plot_sensor
from evaluate import evaluate_all_sensors, compare_methods


def run_pipeline():
    print("Simulating sensor network...")
    data = simulate_all_sensors()

    print("\nRunning anomaly detection...")
    analysed_data = {}
    for name, df in data.items():
        settings = SENSOR_SETTINGS.get(name, {})
        analysed = detect_anomalies(df, **settings)
        analysed = detect_dropout(analysed)
        analysed_data[name] = analysed

    print("\n--- Detection summary ---")
    for name, df in analysed_data.items():
        summarise_anomalies(df, name)

    print("\nSaving plots...")
    for name, df in analysed_data.items():
        plot_sensor(df, name)

    pd.set_option("display.float_format", "{:.2f}".format)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 160)

    print("\n--- Evaluation vs. known ground truth ---")
    results = evaluate_all_sensors("statistical")
    print(results)
    print(f"\nMean F1 (statistical): {results['f1'].mean():.2f}")

    print("\n--- Statistical vs. ML (Isolation Forest) ---")
    comparison = compare_methods()
    print(comparison)
    print(f"\nMean F1 (statistical): {comparison['statistical_f1'].mean():.2f}")
    print(f"Mean F1 (ML):          {comparison['ml_f1'].mean():.2f}")

    print("\nDone. See the plots/ folder for results.")


if __name__ == "__main__":
    run_pipeline()