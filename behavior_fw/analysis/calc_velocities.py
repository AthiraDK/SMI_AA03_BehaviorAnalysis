# behavior_fw/kinematics/velocities.py

from typing import Any, Dict, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from statsmodels.nonparametric.smoothers_lowess import lowess


def compute_velocities(
    X: np.ndarray,
    Y: np.ndarray,
    angle_deg_corr: np.ndarray,
    fps: float = 60.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Computes raw forward (Vf), sideslip (Vs), total (Vt), and rotational (Vr) velocities

    using the exact lab orientation projection equations.
    """
    Vx = np.diff(X, prepend=X[0])
    Vy = np.diff(Y, prepend=Y[0])

    angle_rad = np.deg2rad(angle_deg_corr)

    Vs_raw = fps * ((Vx * np.cos(angle_rad)) + (Vy * np.sin(angle_rad)))
    Vf_raw = fps * ((Vy * np.cos(angle_rad)) - (Vx * np.sin(angle_rad)))
    Vt_raw = np.sqrt(Vf_raw**2 + Vs_raw**2)

    Vr_raw = -fps * np.diff(angle_deg_corr, prepend=angle_deg_corr[0])

    return Vf_raw, Vs_raw, Vt_raw, Vr_raw


def clear_jumps(
    Vt: np.ndarray,
    Vf: np.ndarray,
    Vs: np.ndarray,
    Vr: np.ndarray,
    angle_deg: np.ndarray,
    flp: np.ndarray,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    int,
    np.ndarray,
]:
    """Detects and corrects 180-degree orientation flips and acceleration spikes in raw tracking."""
    acc_Vt = np.diff(Vt)
    ind_peaks_Vt = np.array([], dtype=int)

    if len(acc_Vt) > 0 and np.nanmax(acc_Vt) > 15:
        ind_peaks_Vt, _ = find_peaks(
            np.abs(acc_Vt), height=20, distance=6
        )
        ind_peaks_Vt = np.insert(
            ind_peaks_Vt, [0, len(ind_peaks_Vt)], [0, len(Vf) - 1]
        )

        for pk_i, pk_j in zip(ind_peaks_Vt[:-1], ind_peaks_Vt[1:]):
            mean_Vf_btw_peaks = np.nanmean(Vf[pk_i:pk_j])
            if mean_Vf_btw_peaks < 0:
                Vf[pk_i:pk_j] = -1 * Vf[pk_i:pk_j]
                Vs[pk_i:pk_j] = -1 * Vs[pk_i:pk_j]
                angle_deg[pk_i:pk_j] = angle_deg[pk_i:pk_j] - 180.0
                flp[pk_i:pk_j] = np.abs(flp[pk_i:pk_j] - 180.0)

            if (pk_i + 5) < len(Vf):
                if pk_i > 6:
                    Vf[pk_i - 5 : pk_i + 5] = 0
                    Vs[pk_i - 5 : pk_i + 5] = 0
                    Vr[pk_i - 5 : pk_i + 5] = 0
                elif pk_i > 2:
                    Vf[pk_i - 2 : pk_i + 5] = 0
                    Vs[pk_i - 2 : pk_i + 5] = 0
                    Vr[pk_i - 2 : pk_i + 5] = 0
                else:
                    Vf[pk_i : pk_i + 5] = 0
                    Vs[pk_i : pk_i + 5] = 0
                    Vr[pk_i : pk_i + 5] = 0
            else:
                Vf[pk_i] = 0
                Vs[pk_i] = 0
                Vr[pk_i] = 0

    Vt = np.sqrt(Vf**2 + Vs**2)
    n_jumps = 0
    return Vt, Vf, Vs, Vr, angle_deg, flp, n_jumps, ind_peaks_Vt


def smooth_lowess(
    signal_y: np.ndarray, signal_x: np.ndarray, span: float, delta: float
) -> np.ndarray:
    """Applies LOWESS smoothing to continuous kinematic velocity streams."""
    smoothed = lowess(
        signal_y,
        signal_x,
        frac=span,
        it=3,
        delta=delta,
        return_sorted=False,
    )
    return smoothed


def get_velocities(df_track: pd.DataFrame, fps: float = 60.0) -> pd.DataFrame:
    """Processes raw tracking dataframe to calculate and append cleaned and smoothed

    velocities (Vf_smooth, Vs_smooth, Vr_smooth, Vt_smooth).
    """
    df = df_track.copy()

    # Column mappings with safe fallbacks
    col_x = "X" if "X" in df.columns else "x_mm"
    col_y = "Y" if "Y" in df.columns else "y_mm"
    col_angle = (
        "ori_angle_cal"
        if "ori_angle_cal" in df.columns
        else ("Angle" if "Angle" in df.columns else "angle")
    )

    X = df[col_x].values
    Y = df[col_y].values

    # Fallback to zero-centered angles if orientation tracking is absent
    if col_angle in df.columns:
        angle_deg_corr = df[col_angle].values
    else:
        angle_deg_corr = np.zeros_like(X)

    # Calculate timestamps from frame numbers if Timestamp column is missing
    if "Timestamp" in df.columns:
        timestamps = pd.to_datetime(df["Timestamp"])
        timestamps_track = (
            timestamps - timestamps.iloc[0]
        ).dt.total_seconds().values
    elif "LowResFrameNumber" in df.columns:
        frame_numbers = df["LowResFrameNumber"].values
        timestamps_track = (frame_numbers - frame_numbers[0]) / fps
    else:
        timestamps_track = np.arange(len(X)) / fps

    # 1. Compute Raw Velocities
    Vf_raw, Vs_raw, Vt_raw, Vr_raw = compute_velocities(
        X, Y, angle_deg_corr, fps=fps
    )

    Vt, Vf, Vs, Vr = Vt_raw.copy(), Vf_raw.copy(), Vs_raw.copy(), Vr_raw.copy()

    # 2. Iterative Jump Cleaning
    n_jumps_b = 0
    n_jumps = 1
    flp = np.zeros_like(Vf_raw)

    while n_jumps != n_jumps_b:
        n_jumps_b = n_jumps
        Vt, Vf, Vs, Vr, angle_deg_corr, flp, n_jumps, _ = clear_jumps(
            Vt, Vf, Vs, Vr, angle_deg_corr, flp
        )

    # 3. LOWESS Smoothing
    time_range = np.max(timestamps_track) - np.min(timestamps_track)
    if time_range <= 0 or len(timestamps_track) < 10:
        # Avoid zero division or edge cases on small slices
        Vf_smooth, Vs_smooth, Vr_smooth = Vf, Vs, Vr
    else:
        span = min(6.0 / len(timestamps_track), 0.1)
        delta = 0.01 * 0.001 * time_range

        Vr_smooth = smooth_lowess(Vr, timestamps_track, span, delta)
        Vf_smooth = smooth_lowess(Vf, timestamps_track, span, delta)
        Vs_smooth = smooth_lowess(Vs, timestamps_track, span, delta)

    # Append metrics into DataFrame
    df["Vf_smooth"] = Vf_smooth
    df["Vs_smooth"] = Vs_smooth
    df["Vr_smooth"] = Vr_smooth
    df["Vt_smooth"] = np.sqrt(Vf_smooth**2 + Vs_smooth**2)

    df["Vr_abs"] = df["Vr_smooth"].abs()

    return df