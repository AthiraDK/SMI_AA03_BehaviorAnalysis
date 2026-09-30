from typing import Dict, List
import numpy as np
import pandas as pd
from scipy.stats import skew


def extract_iri_rotational_velocity(
    exp: Dict,
    min_iri_sec: float = 1.0,
    fps: float = 60.0
) -> List[Dict]:
    """Extracts inter-reward intervals (IRIs) from reward offset to next reward onset.

    Returns a list of dicts, each containing 'time_vec', 'vr_segment', 
    'cum_heading', and metadata for a single IRI.
    """
    df_track = exp["df_track"]

    if "LEDstate" not in df_track.columns or "Vr_smooth" not in df_track.columns:
        raise KeyError("Required columns ('LEDstate', 'Vr_smooth') not found in exp['df_track'].")

    led_states = df_track["LEDstate"].values
    led_diff = np.diff(led_states, prepend=0)

    # Onsets: 0 -> 1 | Offsets: 1 -> 0
    onsets = np.where(led_diff == 1)[0]
    offsets = np.where(led_diff == -1)[0]

    iri_data = []

    for off_idx in offsets:
        # Find the immediate next reward onset
        next_onsets = onsets[onsets > off_idx]
        if len(next_onsets) > 0:
            next_on_idx = next_onsets[0]
            duration_sec = (next_on_idx - off_idx) / fps

            if duration_sec >= min_iri_sec:
                # Slicing the DataFrame column for this interval gives vr_segment
                vr_segment = df_track["Vr_smooth"].iloc[off_idx:next_on_idx].values
                time_vec = np.arange(len(vr_segment)) / fps
                cum_heading = np.cumsum(vr_segment) / fps  # Integrated angle

                iri_data.append({
                    "start_idx": off_idx,
                    "end_idx": next_on_idx,
                    "duration_sec": duration_sec,
                    "time_vec": time_vec,
                    "vr_segment": vr_segment,
                    "cum_heading": cum_heading
                })

    return iri_data




import numpy as np
import pandas as pd
from scipy.stats import skew


def extract_enhanced_iri_metrics(
    exp_list: list,
    group_label: str,  # 'ATR' or 'NR'
    fps: float = 60.0,
    skip_start_sec: float = 1.0,  # Skips initial visual startle window
    min_iri_sec: float = 1.0,
    max_iri_sec: float = 15.0,
) -> pd.DataFrame:
    records = []
    skip_frames = int(skip_start_sec * fps)

    for exp_idx, exp in enumerate(exp_list):
        # 1. Safely extract Fly ID using 'uid' or 'folder_path'
        raw_id = exp.get("uid", exp.get("folder_path", f"fly_{exp_idx+1}"))
        if hasattr(raw_id, "name"):  # Handles PosixPath objects
            raw_id = raw_id.name
        fly_id = f"{group_label}_{raw_id}"

        df_track = exp.get("df_track", None)

        if df_track is None or "LEDstate" not in df_track.columns:
            continue

        # Check for velocity columns (or fallback if named differently)
        vr_col = "Vr_smooth" if "Vr_smooth" in df_track.columns else "Vr"
        vxy_col = "Vxy_smooth" if "Vxy_smooth" in df_track.columns else "Vxy"

        if vr_col not in df_track.columns:
            continue

        led_states = df_track["LEDstate"].values
        led_diff = np.diff(led_states, prepend=0)

        onsets = np.where(led_diff == 1)[0]
        offsets = np.where(led_diff == -1)[0]

        for iri_idx, off_idx in enumerate(offsets):
            next_onsets = onsets[onsets > off_idx]
            if len(next_onsets) == 0:
                continue

            next_on_idx = next_onsets[0]
            full_dur = (next_on_idx - off_idx) / fps

            if full_dur < min_iri_sec or full_dur > max_iri_sec:
                continue

            # 2. Apply Exclusion Window (Skip transient startle)
            analysis_start_idx = off_idx + skip_frames
            if analysis_start_idx >= next_on_idx:
                continue  # Skip short IRIs fully covered by exclusion window

            vr = df_track[vr_col].iloc[analysis_start_idx:next_on_idx].values
            analyzed_dur = len(vr) / fps

            if len(vr) == 0:
                continue

            # 3. Calculate Angular Power Asymmetry (V_r^2)
            pos_power = np.sum(vr[vr > 0] ** 2)
            neg_power = np.sum(vr[vr < 0] ** 2)
            total_power = pos_power + neg_power
            power_asymmetry = (
                (pos_power - neg_power) / total_power
                if total_power > 0
                else 0.0
            )

            # 4. Extract Forward Velocity (Vxy) if available
            mean_vxy = (
                np.mean(df_track[vxy_col].iloc[analysis_start_idx:next_on_idx])
                if vxy_col in df_track.columns
                else np.nan
            )

            records.append({
                "group": group_label,
                "fly_id": fly_id,
                "iri_index": iri_idx,
                "duration_sec": full_dur,
                "analyzed_dur_sec": analyzed_dur,
                "abs_power_asym": abs(power_asymmetry),
                "abs_skewness": abs(skew(vr)) if len(vr) > 2 else 0.0,
                "mean_vxy": mean_vxy,
            })

    return pd.DataFrame(records)