"""
ml_detector.py

An alternative approach: using Isolation Forest (scikit-learn), for comparison against the statistical median/MAD-threshold detector in anomaly_detector.py.
Whether this actually performs better is checked properly in evaluate.py, using the same precision/recall/F1 framework as the statistical detector - not assumed just because it's a more advanced technique.
"""

import numpy as np
from sklearn.ensemble import IsolationForest


# Expected proportion of anomalous readings. Deliberately set a little higher than the true injected rate (roughly 1-8% depending on sensor) since Isolation Forest is sensitive to this parameter and a small overestimate is safer than under-flagging real events.
DEFAULT_CONTAMINATION = 0.08


def detect_anomalies_ml(df, contamination=DEFAULT_CONTAMINATION, window=10, random_state=42):
    """
    Takes a sensor DataFrame (columns: timestamp, value) and returns a
    copy with an added is_anomaly_ml column.
    The model is given a short rolling window of recent values as features (not just the single raw reading) so it has some sense of local context, similar to the rolling window used by the statistical detector.
    """
    result = df.copy()

    # Build simple rolling features: current value, rolling mean,
    # rolling std over a short trailing window
    rolling_mean = result["value"].rolling(window=window, min_periods=1).mean()
    rolling_std = result["value"].rolling(window=window, min_periods=1).std().fillna(0)

    features = np.column_stack([
        result["value"].to_numpy(),
        rolling_mean.to_numpy(),
        rolling_std.to_numpy(),
    ])

    model = IsolationForest(contamination=contamination, random_state=random_state)
    predictions = model.fit_predict(features)  # -1 = anomaly, 1 = normal

    result["is_anomaly_ml"] = predictions == -1

    return result


if __name__ == "__main__":
    from sensor_simulator import simulate_all_sensors

    data = simulate_all_sensors()

    for name, df in data.items():
        analysed = detect_anomalies_ml(df)
        n_flagged = analysed["is_anomaly_ml"].sum()
        print(f"{name}: {n_flagged} readings flagged (Isolation Forest)")
