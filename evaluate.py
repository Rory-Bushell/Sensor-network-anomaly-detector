"""
evaluate.py

Scores the anomaly detector's performance against known ground truth.

Because the anomalies in this project are deliberately injected (see
sensor_simulator.py), we know exactly which readings are genuinely
anomalous. That means detector output can be checked properly, rather
than just visually inspected on a plot:

- Precision: of everything the detector flagged, how much was correct?
  (low precision = too many false alarms)
- Recall: of everything genuinely anomalous, how much did the detector
  actually catch? (low recall = missed real events)
- F1 score: combined measure balancing both

This is the same kind of evaluation a real detection system would be
judged on before deployment.
"""

import pandas as pd

from sensor_simulator import simulate_all_sensors, get_ground_truth_mask
from anomaly_detector import detect_anomalies, detect_dropout, SENSOR_SETTINGS
from ml_detector import detect_anomalies_ml


def _confusion_counts(predicted, actual):
    """
    Returns (true_positives, false_positives, false_negatives, true_negatives) for two boolean arrays of equal length.
    """
    tp = (predicted & actual).sum()
    fp = (predicted & ~actual).sum()
    fn = (~predicted & actual).sum()
    tn = (~predicted & ~actual).sum()
    return tp, fp, fn, tn


def score_sensor(predicted, actual):
    """
    Returns a dictionary of precision, recall, f1, and the raw confusion counts, for one sensor's predicted vs. actual anomaly flags.
    """
    tp, fp, fn, tn = _confusion_counts(predicted, actual)

    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else float("nan")
    )

    return {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def evaluate_all_sensors(method="statistical"):
    """
    Runs the chosen detector ("statistical" or "ml") against every sensor and scores each one against its ground truth.
    Returns a DataFrame, one row per sensor.
    """
    data = simulate_all_sensors()

    rows = []
    for name, df in data.items():
        if method == "statistical":
            settings = SENSOR_SETTINGS.get(name, {})
            analysed = detect_anomalies(df, **settings)
            analysed = detect_dropout(analysed)
            predicted = (analysed["is_anomaly"] | analysed["is_dropout"]).to_numpy()
        elif method == "ml":
            analysed = detect_anomalies_ml(df)
            predicted = analysed["is_anomaly_ml"].to_numpy()
        else:
            raise ValueError(f"Unknown method: {method}")

        actual = get_ground_truth_mask(name, length=len(df))

        scores = score_sensor(predicted, actual)
        scores["sensor"] = name
        rows.append(scores)

    results = pd.DataFrame(rows).set_index("sensor")
    results = results[
        ["true_positives", "false_positives", "false_negatives",
         "true_negatives", "precision", "recall", "f1"]
    ]
    return results


def compare_methods():
    """
    Runs both detectors and returns a side-by-side comparison of their precision, recall, and F1 per sensor.
    """
    statistical = evaluate_all_sensors("statistical")
    ml = evaluate_all_sensors("ml")

    comparison = pd.DataFrame({
        "statistical_precision": statistical["precision"],
        "statistical_recall": statistical["recall"],
        "statistical_f1": statistical["f1"],
        "ml_precision": ml["precision"],
        "ml_recall": ml["recall"],
        "ml_f1": ml["f1"],
    })
    return comparison


if __name__ == "__main__":
    pd.set_option("display.float_format", "{:.2f}".format)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 160)

    results = evaluate_all_sensors("statistical")
    print("\nStatistical detector performance vs. known ground truth\n")
    print(results)
    print(f"\nMean F1 (statistical): {results['f1'].mean():.2f}")

    comparison = compare_methods()
    print("\n\nStatistical vs. ML (Isolation Forest) comparison\n")
    print(comparison)
    print(f"\nMean F1 (statistical): {comparison['statistical_f1'].mean():.2f}")
    print(f"Mean F1 (ML):          {comparison['ml_f1'].mean():.2f}")