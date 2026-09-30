import numpy as np
import random

import matplotlib.pyplot as plt
from typing import Dict, List

def plot_iri_rotational_velocity(
    iri_data: List[Dict],
    max_plots: int = 6,
    saccade_threshold: float = 200.0
):
    """Plots Vr velocity spikes and cumulative heading drift for extracted IRIs."""
    if not iri_data:
        print("No IRI data provided to plot.")
        return

    n_plots = min(len(iri_data), max_plots)
    fig, axes = plt.subplots(n_plots, 2, figsize=(12, 2.5 * n_plots), sharex=False, sharey='col')

    if n_plots == 1:
        axes = np.array([axes])

    for i in range(n_plots):
        iri = iri_data[i]
        time_vec = iri["time_vec"]
        vr_segment = iri["vr_segment"]
        cum_heading = iri["cum_heading"]

        # Panel 1: Vr Velocity Spikes
        axes[i, 0].plot(time_vec, vr_segment, color="tab:blue", lw=1.2)
        axes[i, 0].axhline(0, color="gray", linestyle="--", alpha=0.6)
        axes[i, 0].axhline(saccade_threshold, color="red", linestyle=":", alpha=0.4)
        axes[i, 0].axhline(-saccade_threshold, color="red", linestyle=":", alpha=0.4)
        axes[i, 0].set_ylabel("Vr (deg/s)")
        axes[i, 0].set_title(f"IRI #{i+1} | Duration: {iri['duration_sec']:.2f}s")

        # Panel 2: Cumulative Angle Drift
        axes[i, 1].plot(time_vec, cum_heading, color="tab:purple", lw=1.5)
        axes[i, 1].axhline(0, color="gray", linestyle="--", alpha=0.6)
        axes[i, 1].set_ylabel("Cum. Angle (deg)")
        axes[i, 1].set_title("Integrated Heading Trend")

        if i == n_plots - 1:
            axes[i, 0].set_xlabel("Time from Reward Offset (s)")
            axes[i, 1].set_xlabel("Time from Reward Offset (s)")

    plt.tight_layout()
    plt.show()




def plot_random_iri_samples(
    iri_data: list,
    n_samples: int = 6,
    saccade_threshold: float = 300.0,
    seed: int = None
):
    """Plots a random sample of IRIs."""
    if not iri_data:
        print("No IRI data provided to plot.")
        return

    if seed is not None:
        random.seed(seed)  # Set seed for reproducible random samples

    # Pick min(n_samples, available) random IRIs
    sampled_iris = random.sample(iri_data, min(n_samples, len(iri_data)))

    n_plots = len(sampled_iris)
    fig, axes = plt.subplots(n_plots, 2, figsize=(12, 2.5 * n_plots), sharex=False)

    if n_plots == 1:
        axes = np.array([axes])

    for i, iri in enumerate(sampled_iris):
        time_vec = iri["time_vec"]
        vr_segment = iri["vr_segment"]
        cum_heading = iri["cum_heading"]

        # Panel 1: Vr Velocity Spikes
        axes[i, 0].plot(time_vec, vr_segment, color="tab:blue", lw=1.2)
        axes[i, 0].axhline(0, color="gray", linestyle="--", alpha=0.6)
        axes[i, 0].axhline(saccade_threshold, color="red", linestyle=":", alpha=0.4)
        axes[i, 0].axhline(-saccade_threshold, color="red", linestyle=":", alpha=0.4)
        axes[i, 0].set_ylabel("Vr (deg/s)")
        axes[i, 0].set_title(f"IRI Sample | Dur: {iri['duration_sec']:.2f}s | Frames: {iri['start_idx']}-{iri['end_idx']}")

        # Panel 2: Cumulative Angle Drift
        axes[i, 1].plot(time_vec, cum_heading, color="tab:purple", lw=1.5)
        axes[i, 1].axhline(0, color="gray", linestyle="--", alpha=0.6)
        axes[i, 1].set_ylabel("Cum. Angle (deg)")
        axes[i, 1].set_title("Integrated Heading Trend")

        if i == n_plots - 1:
            axes[i, 0].set_xlabel("Time from Reward Offset (s)")
            axes[i, 1].set_xlabel("Time from Reward Offset (s)")

    plt.tight_layout()
    plt.show()