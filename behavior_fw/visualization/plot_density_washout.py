from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.patches as patches

from behavior_fw.analysis.spatial_density import (
    compute_2d_cartesian_histogram,
    compute_2d_polar_histogram,
)
import seaborn as sns

def _get_coords(
    df: pd.DataFrame, x_col: str = "X", y_col: str = "Y"
) -> Tuple[np.ndarray, np.ndarray]:
    """Extracts X and Y coordinates with fallback column names."""
    x_name = x_col if x_col in df.columns else ("X" if "X" in df.columns else "x_mm")
    y_name = y_col if y_col in df.columns else ("Y" if "Y" in df.columns else "y_mm")

    if x_name not in df.columns or y_name not in df.columns:
        raise KeyError(f"Could not find coordinate columns ({x_col}, {y_col}) in DataFrame.")

    return df[x_name].values, df[y_name].values


def _extract_washout_dataframe(ds: Dict) -> pd.DataFrame:
    """Extracts post-reward trajectory for a fly."""
    df_track = ds["df_track"]
    
    # 1. Direct index fallback
    if "last_reward_idx" in ds and ds["last_reward_idx"] is not None:
        l_idx = ds["last_reward_idx"]
        if 0 <= l_idx < len(df_track):
            return df_track.iloc[l_idx:].copy()

    # 2. Block offset timetick fallback
    df_blocks = ds.get("df_blocks", None)
    if df_blocks is not None and len(df_blocks) > 0:
        last_offset = df_blocks.iloc[-1].get(
            "offset_timetick", df_blocks.iloc[-1].get("onset_timetick", None)
        )
        if last_offset is not None and "timetick" in df_track.columns:
            l_match = df_track[np.isclose(df_track["timetick"], last_offset, atol=1e-3)]
            if not l_match.empty:
                l_idx = l_match.index[-1] + 1
                return df_track.iloc[l_idx:].copy()

    # 3. Whole track fallback
    return df_track.copy()


def _chunk_washout_dataframe(
    df_washout: pd.DataFrame, chunk_duration_sec: float = 120.0, fps: float = 60.0
) -> List[pd.DataFrame]:
    """Slices washout DataFrame into temporal windows."""
    if df_washout.empty:
        return []

    chunk_frames = int(chunk_duration_sec * fps)
    total_frames = len(df_washout)
    return [
        df_washout.iloc[i : i + chunk_frames]
        for i in range(0, total_frames, chunk_frames)
        if not df_washout.iloc[i : i + chunk_frames].empty
    ]


def plot_washout_chunks_cartesian(
    dataset: List[Dict],
    dataset_name: str = "Dataset",
    chunk_duration_sec: float = 120.0,
    max_chunks: Optional[int] = None,
    arena_r: float = 45.0,
    fps: float = 60.0,
    n_bins: int = 100,
    gamma: float = 0.3,
    x_col: str = "X",
    y_col: str = "Y",
) -> Tuple[plt.Figure, np.ndarray]:
    """Cartesian density across washout time chunks by summing counts across flies."""
    bin_edges = np.linspace(-arena_r, arena_r, n_bins + 1)
    X_mesh, Y_mesh = np.meshgrid(bin_edges, bin_edges)

    # Slice washout for each fly
    all_fly_chunks = [
        _chunk_washout_dataframe(_extract_washout_dataframe(ds), chunk_duration_sec, fps)
        for ds in dataset
    ]
    max_n_chunks = max((len(chunks) for chunks in all_fly_chunks), default=0)

    if max_chunks is not None:
        max_n_chunks = min(max_n_chunks, max_chunks)

    if max_n_chunks == 0:
        raise ValueError("No washout data found in the provided dataset.")

    chunk_histograms = []
    global_max = 0.0

    # Aggregate counts per chunk
    for chunk_idx in range(max_n_chunks):
        x_all, y_all = [], []

        for fly_chunks in all_fly_chunks:
            if chunk_idx < len(fly_chunks):
                df_c = fly_chunks[chunk_idx]
                if not df_c.empty:
                    x, y = _get_coords(df_c, x_col=x_col, y_col=y_col)
                    valid = ~np.isnan(x) & ~np.isnan(y)
                    x_all.append(x[valid])
                    y_all.append(y[valid])

        if x_all:
            x_cat = np.concatenate(x_all)
            y_cat = np.concatenate(y_all)

            # Compute histogram on concatenated positions
            counts, _, _ = compute_2d_cartesian_histogram(
                x_cat, y_cat, n_bins=n_bins, arena_r=arena_r
            )
            # Normalize: % of total group time spent in bin
            density_map = (counts / len(x_cat)) * 100.0
        else:
            density_map = np.full((n_bins, n_bins), np.nan)

        # Mask outside arena boundary
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
        X_grid, Y_grid = np.meshgrid(bin_centers, bin_centers)
        density_map[np.sqrt(X_grid**2 + Y_grid**2) > arena_r] = np.nan

        chunk_histograms.append(density_map)
        max_val = np.nanmax(density_map) if not np.all(np.isnan(density_map)) else 0.0
        if max_val > global_max:
            global_max = max_val

    if global_max == 0:
        global_max = 1.0

    # Plot subplots
    fig, axes = plt.subplots(
        1, max_n_chunks, figsize=(3.8 * max_n_chunks, 4.0), dpi=300, layout="constrained"
    )
    if max_n_chunks == 1:
        axes = np.array([axes])

    cmap = plt.get_cmap("rocket").copy()
    cmap.set_bad("none")
    norm = colors.PowerNorm(gamma=gamma, vmin=0, vmax=global_max)

    ref_ds = dataset[0] if dataset else {}
    zx, zy, zr = ref_ds.get("zone_X", 0.0), ref_ds.get("zone_Y", 0.0), ref_ds.get("zone_r", 3.0)

    for i in range(max_n_chunks):
        ax = axes[i]
        density_map = chunk_histograms[i]

        ax_zone = patches.Circle((0, 0), arena_r, transform=ax.transData, fill=False, ec="#EEE6E6")
        ax.add_patch(ax_zone)

        im = ax.pcolormesh(
            X_mesh, Y_mesh, density_map, cmap=cmap, norm=norm, shading="flat", rasterized=True
        )
        im.set_clip_path(ax_zone)

        ax.add_patch(patches.Circle((0, 0), arena_r, color="#444444", fill=False, linestyle="--"))
        ax.add_patch(
            patches.Circle(
                (zx, zy),
                zr,
                edgecolor="white",
                facecolor="none",
                linewidth=0.8,
                linestyle=":",
                alpha=0.8,
                zorder=10,
            )
        )

        t_start = (i * chunk_duration_sec) / 60.0
        t_end = ((i + 1) * chunk_duration_sec) / 60.0
        ax.set_title(f"Chunk {i+1}\n({t_start:.1f} - {t_end:.1f} min)", fontsize=10, fontweight="bold")
        ax.set_aspect("equal")
        ax.axis("off")

    cbar = fig.colorbar(im, ax=axes, shrink=0.75, pad=0.02)
    cbar.set_label("% Time in Bin", fontsize=10, fontweight="bold")

    fly_str = dataset[0].get("uid", "Fly") if len(dataset) == 1 else f"N = {len(dataset)} flies"
    fig.suptitle(f"{dataset_name} | Washout Extinction ({fly_str})", fontsize=12, fontweight="bold")

    return fig, axes