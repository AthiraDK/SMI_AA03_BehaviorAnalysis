from typing import List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



def segment_experiment_epochs(
    df: pd.DataFrame, df_blocks: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    """Segments a fly's tracking DataFrame into 3 meaningful behavioral epochs

    based on actual reward encounter boundaries: Baseline, Active Reward, and
    Extinction.
    """
    if df_blocks is None or len(df_blocks) == 0:
        # If no rewards were ever triggered (e.g., NR fly that never hit zone),
        # return whole trial as Baseline
        return {"Baseline": df, "Active_Reward": None, "Extinction": None}

    first_onset_timetick = df_blocks.iloc[0]["onset_timetick"]
    last_offset_timetick = df_blocks.iloc[-1].get(
        "offset_timetick", df_blocks.iloc[-1]["onset_timetick"]
    )

    # Extract row indices from tracking dataframe matching timeticks
    first_onset_idx = df[df["timetick"] == first_onset_timetick].index[0]
    last_offset_idx = df[df["timetick"] == last_offset_timetick].index[-1]

    # Chunk into 3 dataframes
    baseline_df = df.iloc[:first_onset_idx].copy()
    active_df = df.iloc[first_onset_idx:last_offset_idx].copy()
    extinction_df = df.iloc[last_offset_idx:].copy()

    return {
        "Baseline": baseline_df,
        "Active_Reward": active_df,
        "Extinction": extinction_df,
    }


def compute_reward_triggered_kinematics(
    df_track: pd.DataFrame,
    df_blocks: pd.DataFrame,
    window_sec: Tuple[float, float] = (-2.0, 4.0),
    fps: float = 60.0,
    metric_col: str = "Vf_smooth",
) -> Tuple[np.ndarray, np.ndarray]:
    """Extracts peri-event kinematic traces centered on reward onset ticks.

    Parameters
    ----------
    df_track : pd.DataFrame
        Tracking dataframe containing kinematics columns.
    df_blocks : pd.DataFrame
        Blocks dataframe containing 'onset_timetick' entries.
    window_sec : tuple
        (pre_event_sec, post_event_sec) window around reward onset.
    fps : float
        Sampling frequency in Hz.
    metric_col : str
        Target column name (e.g. 'Vf_smooth', 'Vr_smooth', 'dist_to_zone_mm').

    Returns
    -------
    time_vec : np.ndarray
        Relative time vector in seconds centered at t=0.
    traces : np.ndarray
        2D array of shape (n_events, n_window_frames).
    """
    pre_frames = int(abs(window_sec[0]) * fps)
    post_frames = int(window_sec[1] * fps)
    total_frames = pre_frames + post_frames

    time_vec = np.linspace(window_sec[0], window_sec[1], total_frames)
    traces = []

    if df_blocks.empty or metric_col not in df_track.columns:
        return time_vec, np.array([])

    for _, block in df_blocks.iterrows():
        onset_tick = block["onset_timetick"]

        # Find matching row index in df_track
        match = df_track[df_track["LowResFrameNumber"] == onset_tick]
        if match.empty:
            continue

        center_idx = match.index[0]
        start_idx = center_idx - pre_frames
        end_idx = center_idx + post_frames

        # Ensure window stays within valid bounds
        if start_idx >= 0 and end_idx <= len(df_track):
            trace_vals = df_track[metric_col].iloc[start_idx:end_idx].values
            if len(trace_vals) == total_frames:
                traces.append(trace_vals)

    return time_vec, np.array(traces)