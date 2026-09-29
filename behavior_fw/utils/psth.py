# behavior_fw/kinematics/psth.py

from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def extract_experiment_psth(
    exp: Dict,
    metrics: List[str] = ["Vf_smooth", "Vr_smooth", "Vs_smooth"],
    window_sec: Tuple[float, float] = (-2.0, 4.0),
    fps: float = 60.0,
) -> Dict[str, np.ndarray]:
    """Extracts PSTH matrices for a single experiment across reward onset events.

    Returns a dict with 'time_vec' and 2D arrays for each metric (n_trials x
    n_frames).
    """
    df_track = exp["df_track"]
    fpath_stim = exp.get("folder_path", "")

    pre_frames = int(abs(window_sec[0]) * fps)
    post_frames = int(window_sec[1] * fps)
    total_frames = pre_frames + post_frames

    time_vec = np.linspace(window_sec[0], window_sec[1], total_frames)
    psth_data = {"time_vec": time_vec}

    for m in metrics:
        psth_data[m] = []

    # Get reward onset frames
    # Re-extract onset ticks from df_track where LEDstate switches from 0 to 1
    if "LEDstate" in df_track.columns:
        led_diff = np.diff(df_track["LEDstate"].values, prepend=0)
        onset_indices = np.where(led_diff == 1)[0]
    else:
        onset_indices = []

    if len(onset_indices) == 0:
        for m in metrics:
            psth_data[m] = np.empty((0, total_frames))
        return psth_data

    for center_idx in onset_indices:
        start_idx = center_idx - pre_frames
        end_idx = center_idx + post_frames

        if start_idx >= 0 and end_idx <= len(df_track):
            for m in metrics:
                if m in df_track.columns:
                    psth_data[m].append(
                        df_track[m].iloc[start_idx:end_idx].values
                    )

    # Convert list of arrays into 2D matrices (n_trials, n_frames)
    for m in metrics:
        if len(psth_data[m]) > 0:
            psth_data[m] = np.array(psth_data[m])
        else:
            psth_data[m] = np.empty((0, total_frames))

    return psth_data