# behavior_fw/io/dataset_loader.py

from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
from behavior_fw.io.read_metaprotocol import get_protocol_params
from behavior_fw.io.read_rawtracking import get_data_tracking
from behavior_fw.io.read_stimulusLED import load_and_parse_led_stimulus
from behavior_fw.analysis.calc_velocities import get_velocities


def load_experiment_dataset(
    df_metadata: pd.DataFrame,
    fps: float = 60.0,
    force_recompute: bool = False,
) -> List[Dict]:
    """Loads a structured dataset from raw files or pre-processed disk caches.

    Parameters
    ----------
    df_metadata : pd.DataFrame
        Metadata dataframe containing 'folder_path', 'date', and 'time' columns.
    fps : float
        Frame rate in Hz (default 60.0).
    force_recompute : bool
        If True, ignores cached ProcessedTracking.csv and recomputes velocities.

    Returns
    -------
    dataset : List[Dict]
        List of experiment dictionaries ready for analysis.
    """
    dataset = []

    for i, (ind, row) in enumerate(df_metadata.iterrows()):
        uid = f"{row['date']}_{row['time']}"
        folder_path = Path(row["folder_path"])
        processed_track_path = folder_path / "ProcessedTracking.csv"

        try:
            # 1. Load or Compute Tracking & Kinematics
            if processed_track_path.exists() and not force_recompute:
                # FAST PATH: Read directly from cache
                df_merged = pd.read_csv(processed_track_path)
            else:
                # SLOW PATH: Compute raw tracking + LED state + velocities
                fpath_track = folder_path / "RawTracking.csv"
                df_track = get_data_tracking(fpath_track)

                fpath_stim = folder_path / "stimulusLED.csv"
                df_stim, _ = load_and_parse_led_stimulus(fpath_stim)

                # Merge LED state on frame index
                df_merged = pd.merge(
                    df_track,
                    df_stim[["LowResFrameNumber", "LEDstate"]],
                    on="LowResFrameNumber",
                    how="left",
                )
                df_merged["LEDstate"] = (
                    df_merged["LEDstate"].fillna(0).astype(int)
                )

                # Compute LOWESS smoothed kinematics
                df_merged = get_velocities(df_merged, fps=fps)

                # Cache to disk for instant loading next time
                df_merged.to_csv(processed_track_path, index=False)
                print(f"[{uid}] Calculated and cached velocities to disk.")

            # 2. Parse Protocol & Reward Metadata
            fpath_protocol = folder_path / "ProtocolMeta.csv"
            df_protocol = get_protocol_params(fpath_protocol)

            fpath_stim = folder_path / "stimulusLED.csv"
            _, df_blocks = load_and_parse_led_stimulus(fpath_stim)

            n_rewards = len(df_blocks)

            # Extract first and last reward frame indices safely
            if n_rewards > 0:
                first_block = df_blocks.iloc[0]
                first_row = df_merged[
                    df_merged["LowResFrameNumber"]
                    == first_block["onset_timetick"]
                ]
                first_reward_idx = (
                    first_row.index[0] if not first_row.empty else None
                )

                last_block = df_blocks.iloc[-1]
                last_offset_tick = last_block.get("offset_timetick", None)

                if last_offset_tick is not None:
                    last_row = df_merged[
                        df_merged["LowResFrameNumber"] == last_offset_tick
                    ]
                    last_reward_idx = (
                        last_row.index[0] if not last_row.empty else None
                    )
                else:
                    last_reward_idx = None
            else:
                first_reward_idx = None
                last_reward_idx = None

            # 3. Assemble Clean Standardized Record
            exp_dict = {
                "file_index": i,
                "uid": uid,
                "folder_path": folder_path,
                "df_track": df_merged,  # Tracking + LEDstate + Vf/Vs/Vr/Vt_smooth
                "zone_X": df_protocol["ZoneCenterX_mm"],
                "zone_Y": df_protocol["ZoneCenterY_mm"],
                "zone_r": df_protocol["ZoneRadius_mm"],
                "n_rewards": n_rewards,
                "first_reward_idx": first_reward_idx,
                "last_reward_idx": last_reward_idx,
            }

            dataset.append(exp_dict)

        except Exception as e:
            print(f"Error processing index {i} ({uid}): {e}")

    return dataset