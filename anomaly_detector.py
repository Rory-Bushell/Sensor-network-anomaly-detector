"""
anomaly_detector.py

Statistical anomaly detection for the simulated sensor network.

Approach: rolling mean + standard deviation threshold. For each reading, compare it to the recent local average (rolling window).
If it deviates by more than a set number of standard deviations, flag it as anomaly.
Only works for spikes, drifts and drop-outs.
"""

import pandas as pd


# How many standard deviations away from rolling mean to be an anomaly
DEFAULT_THRESHOLD = 3.0

# Rolling mean size, in number of readings, used to calculate the local mean/std that each point is compared against.
DEFAULT_WINDOW = 30


def detect_anomalies(df, threshold=DEFAULT_THRESHOLD, window=DEFAULT_WINDOW, min_scale=0.5):
    """
    Takes a sensor DataFrame (columns: timestamp, value) and returns a copy with extra columns.
    rolling_median, rolling_mad: the local statistics each point is compared against
    is_anomaly: True/False flag for each reading
    """
    result = df.copy()

    result["rolling_median"] = result["value"].rolling(
        window=window, min_periods=5
    ).median()

    def _mad(series):
        return (series - series.median()).abs().median()

    result["rolling_mad"] = result["value"].rolling(
        window=window, min_periods=5
    ).apply(_mad, raw=False)

    # Back fill the first few readings (no full window yet) with the first valid statistics, so early readings aren't left unscored
    result["rolling_median"] = result["rolling_median"].bfill()
    result["rolling_mad"] = result["rolling_mad"].bfill()

    # MAD scaled to be comparable to standard deviation for a normal distribution (standard constant, ~1.4826), with a floor applied
    scaled_mad = (result["rolling_mad"] * 1.4826).clip(lower=min_scale)

    deviation = (result["value"] - result["rolling_median"]).abs()
    result["is_anomaly"] = deviation > (threshold * scaled_mad)

    return result


def detect_dropout(df, flat_run_length=15):
    """
    Separate check for sensor dropouts: a sustained run of identical values that a pure std-deviation check can miss, since a flat line technically has zero deviation from itself.
    Flags any reading that is part of a run of `flat_run_length` or identical consecutive values.
    """
    result = df.copy()

    same_as_prev = result["value"].eq(result["value"].shift())
    # Identify consecutive runs of identical values
    run_id = (~same_as_prev).cumsum()
    run_lengths = result.groupby(run_id)["value"].transform("size")

    result["is_dropout"] = run_lengths >= flat_run_length

    return result


def summarise_anomalies(df, sensor_name):
    """
    Prints a short, readable summary of what was flagged and when.
    """
    flagged = df[df.get("is_anomaly", False) | df.get("is_dropout", False)]

    print(f"\n{sensor_name}: {len(flagged)} readings flagged")
    if not flagged.empty:
        # Report contiguous flagged blocks rather than every single row
        flagged_index = flagged.index.to_series()
        breaks = flagged_index.diff().fillna(1) != 1
        block_id = breaks.cumsum()
        for _, block in flagged.groupby(block_id):
            start = block["timestamp"].iloc[0]
            end = block["timestamp"].iloc[-1]
            reason = "dropout" if block.get("is_dropout", pd.Series([False])).any() else "threshold spike"
            print(f"  {start}  to  {end}   ({reason})")


# Different sensors operate on different scales, so a single global sensitivity floor doesn't work for all of them (a floor tuned for a ~20 degree C temperature sensor completely swamps a vibration sensor whose normal readings sit around 0.05).
# Real systems handle this with per-sensor tuning - this small dictionary is the equivalent for this project.
SENSOR_SETTINGS = {
    "fence_vibration": {"threshold": 4.0, "min_scale": 0.02},
    "approach_motion": {"threshold": 5.0, "min_scale": 1.0},
    "cabinet_temperature": {"threshold": 4.0, "min_scale": 0.3},
}


if __name__ == "__main__":
    from sensor_simulator import simulate_all_sensors

    data = simulate_all_sensors()

    for name, df in data.items():
        settings = SENSOR_SETTINGS.get(name, {})
        analysed = detect_anomalies(df, **settings)
        analysed = detect_dropout(analysed)
        summarise_anomalies(analysed, name)