from pathlib import Path
import numpy as np
import pandas as pd


def load_and_parse_led_stimulus(
    stim_csv_path: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reads the 3-column LED stimulus CSV and extracts discrete reward block events.

    Columns:
    0: Frame Index
    1: Low-res timetick
    2: LED State (0 or 1)
    """
    stim_csv_path = Path(stim_csv_path)

    # 1. Load CSV (no header)
    df_stim = pd.read_csv(
        stim_csv_path,
        header=0
    )

    if "LowResFrameNumber" not in df_stim.columns:
        if len(df_stim.columns) == 3:
                df_stim.columns = ["Index", "LowResFrameNumber", "LEDstate"]

    # Ensure led_state is integer (0 or 1)
    df_stim["LEDstate"] = df_stim["LEDstate"].astype(int)

    # 2. Detect transitions (diff = +1 for ON, -1 for OFF)
    led_diff = np.diff(df_stim["LEDstate"].values, prepend=0)

    # Onset indices where LED flips from 0 -> 1
    onset_mask = led_diff == 1
    offset_mask = led_diff == -1

    onset_df = df_stim[onset_mask].copy().reset_index(drop=True)
    offset_df = df_stim[offset_mask].copy().reset_index(drop=True)

    # Build summary of discrete reward blocks
    reward_blocks = []
    for i in range(len(onset_df)):
        block_info = {
            "encounter_id": i + 1,
            "onset_frame_stim": onset_df.loc[i, "Index"],
            "onset_timetick": onset_df.loc[i, "LowResFrameNumber"],
        }
        if i < len(offset_df):
            block_info["offset_frame_stim"] = offset_df.loc[i, "Index"]
            block_info["offset_timetick"] = offset_df.loc[i, "LowResFrameNumber"]

        reward_blocks.append(block_info)

    df_blocks = pd.DataFrame(reward_blocks)

    return df_stim, df_blocks