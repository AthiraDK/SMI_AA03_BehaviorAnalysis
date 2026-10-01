from typing import Dict, List, Optional, Tuple
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_single_fly_trajectory(
    ds: Dict,
    arena_r: float = 45.0,
    fps: float = 60.0,
    x_col: str = "X",
    y_col: str = "Y",
) -> Tuple[plt.Figure, plt.Axes]:
    """Plots baseline, active reward, and extinction trajectories for a single fly record."""
    df_track = ds["df_track"]
    df_blocks = ds.get("df_blocks", None)

    col_x = x_col if x_col in df_track.columns else ("x_mm" if "x_mm" in df_track.columns else "X")
    col_y = y_col if y_col in df_track.columns else ("y_mm" if "y_mm" in df_track.columns else "Y")

    x = df_track[col_x].values
    y = df_track[col_y].values

    f_idx = ds.get("first_reward_idx", None)
    l_idx = ds.get("last_reward_idx", None)

    # Use exact stimulus blocks timeticks if present
    if df_blocks is not None and len(df_blocks) > 0:
        first_onset_tick = df_blocks.iloc[0]["onset_timetick"]
        last_offset_tick = df_blocks.iloc[-1].get(
            "offset_timetick", df_blocks.iloc[-1]["onset_timetick"]
        )

        f_match = df_track[df_track["timetick"] == first_onset_tick] if "timetick" in df_track.columns else pd.DataFrame()
        l_match = df_track[df_track["timetick"] == last_offset_tick] if "timetick" in df_track.columns else pd.DataFrame()

        if not f_match.empty:
            f_idx = f_match.index[0]
        if not l_match.empty:
            l_idx = l_match.index[-1] + 1  # Include final frame offset

    fig, ax = plt.subplots(figsize=(7, 7), dpi=300, layout="constrained")

    # 1. Full Path (Background)
    ax.plot(
        x,
        y,
        color="#888888",
        alpha=0.6,
        linewidth=0.8,
        label="Path (Non-Active)",
    )

    # 2. Active Reward Path (1st onset to last offset)
    if f_idx is not None and l_idx is not None and f_idx < l_idx:
        ax.plot(
            x[f_idx:l_idx],
            y[f_idx:l_idx],
            color="#d62728",
            linewidth=1.2,
            alpha=0.85,
            label="Active Reward Epoch",
        )
        t0_min = f_idx / (fps * 60.0)
        active_dur_sec = (l_idx - f_idx) / fps
        title_suffix = (
            f"n_rewards = {ds.get('n_rewards', 0)} | 1st rew @ {t0_min:.2f} min | Active Dur: {active_dur_sec:.1f}s"
        )
    else:
        title_suffix = f"n_rewards = {ds.get('n_rewards', 0)} (No rewards triggered)"

    # 3. Reward Zone & Arena Circles
    zone_x = ds.get("zone_X", 0.0)
    zone_y = ds.get("zone_Y", 0.0)
    zone_r = ds.get("zone_r", 3.0)

    reward_zone = patches.Circle(
        (zone_x, zone_y),
        zone_r,
        ec="#272424",
        fc="none",
        fill=False,
        linewidth=2,
        zorder=10,
        label="Reward Zone",
    )
    arena_boundary = patches.Circle(
        (0, 0),
        arena_r,
        ec="#333333",
        fc="none",
        fill=False,
        linestyle="--",
        linewidth=1.2,
        alpha=0.7,
    )

    ax.add_patch(reward_zone)
    ax.add_patch(arena_boundary)

    # Axis formatting
    pad = 2.0
    ax.set_xlim([-arena_r - pad, arena_r + pad])
    ax.set_ylim([-arena_r - pad, arena_r + pad])
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_title(
        f"{ds.get('uid', 'Fly')}\n{title_suffix}", fontsize=11, pad=10, fontweight="bold"
    )
    ax.legend(loc="lower right", fontsize=8, frameon=True)

    return fig, ax


def plot_multi_experiment_split_trajectories(
    dataset: List[Dict],
    arena_r: float = 45.0,
    fps: float = 60.0,
    x_col: str = "X",
    y_col: str = "Y",
    color_map: Optional[Dict[str, str]] = None,
    fig_title: Optional[str] = None,
) -> Tuple[plt.Figure, np.ndarray]:
    """Plots split trajectory subplots for multiple experiments."""
    if color_map is None:
        color_map = {
            "Baseline": "#BA90EB",
            "Stimulus": "#F16237",
            "Washout": "#009E73",
        }

    n_experiments = len(dataset)
    if n_experiments == 0:
        raise ValueError("The provided dataset list is empty.")

    fig, axes = plt.subplots(
        nrows=n_experiments,
        ncols=3,
        figsize=(13, 3.5 * n_experiments),
        dpi=300,
        sharex=True,
        sharey=True,
        layout="constrained",
    )

    if n_experiments == 1:
        axes = np.expand_dims(axes, axis=0)

    column_titles = [
        "Pre-1st Reward",
        "Active Reward (1st to Last Offset)",
        "Washout",
    ]

    for row_idx, ds in enumerate(dataset):
        df_track = ds["df_track"]
        df_blocks = ds.get("df_blocks", None)

        col_x = x_col if x_col in df_track.columns else ("x_mm" if "x_mm" in df_track.columns else "X")
        col_y = y_col if y_col in df_track.columns else ("y_mm" if "y_mm" in df_track.columns else "Y")

        x = df_track[col_x].values
        y = df_track[col_y].values
        total_frames = len(x)

        uid = ds.get("uid", f"Fly_{row_idx}")
        n_rewards = ds.get("n_rewards", 0)

        f_idx = ds.get("first_reward_idx", None)
        l_idx = ds.get("last_reward_idx", None)

        # Precise slice bounds from stimulus block timeticks
        if df_blocks is not None and len(df_blocks) > 0:
            first_onset_tick = df_blocks.iloc[0]["onset_timetick"]
            last_offset_tick = df_blocks.iloc[-1].get(
                "offset_timetick", df_blocks.iloc[-1]["onset_timetick"]
            )

            f_match = df_track[df_track["timetick"] == first_onset_tick] if "timetick" in df_track.columns else pd.DataFrame()
            l_match = df_track[df_track["timetick"] == last_offset_tick] if "timetick" in df_track.columns else pd.DataFrame()

            if not f_match.empty:
                f_idx = f_match.index[0]
            if not l_match.empty:
                l_idx = l_match.index[-1] + 1  # Include the full last reward offset frame

        if f_idx is not None:
            t_first_min = f_idx / (fps * 60.0)
            reward_str = f"1st rew: {t_first_min:.2f} m"
        else:
            f_idx = total_frames
            reward_str = "No rewards"

        if l_idx is None or l_idx < f_idx:
            l_idx = f_idx

        # Slice boundaries
        slice_p1 = slice(0, f_idx)
        slice_p2 = slice(f_idx, l_idx)
        slice_p3 = slice(l_idx, total_frames) if l_idx < total_frames else None

        slices = [slice_p1, slice_p2, slice_p3]
        epoch_keys = ["Baseline", "Stimulus", "Washout"]

        for col_idx, (slc, epoch) in enumerate(zip(slices, epoch_keys)):
            ax = axes[row_idx, col_idx]

            if slc is not None and slc.start < slc.stop:
                ax.plot(
                    x[slc],
                    y[slc],
                    color=color_map[epoch],
                    linewidth=0.8,
                    alpha=0.85,
                )

            zone_x = ds.get("zone_X", 0.0)
            zone_y = ds.get("zone_Y", 0.0)
            zone_r = ds.get("zone_r", 3.0)

            reward_zone = patches.Circle(
                (zone_x, zone_y),
                zone_r,
                ec="#272424",
                fc="none",
                fill=False,
                linewidth=1.2,
                zorder=10,
            )
            arena_boundary = patches.Circle(
                (0, 0),
                arena_r,
                ec="#888888",
                fc="none",
                fill=False,
                linestyle="--",
                linewidth=1.0,
                alpha=0.6,
            )
            ax.add_patch(reward_zone)
            ax.add_patch(arena_boundary)

            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            
            if row_idx == 0:
                ax.set_title(column_titles[col_idx], fontsize=11, fontweight="bold", pad=10)

        active_dur_sec = (l_idx - f_idx) / fps if (f_idx is not None and l_idx is not None) else 0.0
        row_label = f"UID: {uid}\nRewards: {n_rewards}\n{reward_str}\nActive: {active_dur_sec:.1f}s"
        axes[row_idx, 0].set_ylabel(
            row_label, 
            fontsize=8, 
            fontweight="bold", 
            labelpad=12,
            rotation=0,
            ha="right",
            va="center"
        )

    pad = 2.0
    for ax in axes.ravel():
        ax.set_xlim([-arena_r - pad, arena_r + pad])
        ax.set_ylim([-arena_r - pad, arena_r + pad])

    if fig_title:
        fig.suptitle(fig_title, fontsize=13, fontweight="bold", y=1.02)

    return fig, axes