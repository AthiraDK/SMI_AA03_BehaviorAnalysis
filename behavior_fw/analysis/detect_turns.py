import numpy as np
from scipy.signal import find_peaks

def detect_iri_saccades(
    iri_data: list,
    velocity_threshold: float = 300.0,  # deg/s cutoff for saccades
    min_distance_sec: float = 0.3,      # Minimum time between distinct saccades (100ms)
    fps: float = 60.0
) -> list:
    """Detects discrete saccade events in each IRI segment.

    Appends 'saccade_directions', 'saccade_times', and 'saccade_counts' 
    to each dictionary in iri_data.
    """
    min_distance_frames = max(1, int(min_distance_sec * fps))

    for iri in iri_data:
        vr = iri["vr_segment"]
        time_vec = iri["time_vec"]

        # 1. Detect Positive Peaks (Left Turns: Vr > threshold)
        left_peaks, _ = find_peaks(
            vr, height=velocity_threshold, distance=min_distance_frames
        )

        # 2. Detect Negative Peaks (Right Turns: Vr < -threshold)
        right_peaks, _ = find_peaks(
            -vr, height=velocity_threshold, distance=min_distance_frames
        )

        # 3. Combine and sort events chronologically
        all_peak_indices = np.concatenate([left_peaks, right_peaks])
        
        if len(all_peak_indices) > 0:
            # Sort indices chronologically
            sort_order = np.argsort(all_peak_indices)
            sorted_indices = all_peak_indices[sort_order]

            # Label direction: +1 for Left, -1 for Right
            saccade_dirs = np.array([1 if idx in left_peaks else -1 for idx in sorted_indices])
            saccade_times = time_vec[sorted_indices]
        else:
            sorted_indices = np.array([])
            saccade_dirs = np.array([])
            saccade_times = np.array([])

        # Store detected saccade information back into the dictionary
        iri["saccade_indices"] = sorted_indices
        iri["saccade_times"] = saccade_times
        iri["saccade_directions"] = saccade_dirs  # Array of +1s and -1s (e.g., [1, 1, 1, -1, 1])
        iri["n_left"] = np.sum(saccade_dirs == 1)
        iri["n_right"] = np.sum(saccade_dirs == -1)

    return iri_data



def detect_iri_saccade_magnitudes(
    iri_data: list,
    velocity_threshold: float = 300.0,
    min_distance_sec: float = 0.3,
    fps: float = 60.0
) -> list:
    """Detects saccades and integrates Vr to extract individual saccade magnitudes (degrees)."""
    min_dist_frames = max(1, int(min_distance_sec * fps))

    for iri in iri_data:
        vr = iri["vr_segment"]
        time_vec = iri["time_vec"]

        # Detect peak indices
        left_peaks, _ = find_peaks(vr, height=velocity_threshold, distance=min_dist_frames)
        right_peaks, _ = find_peaks(-vr, height=velocity_threshold, distance=min_dist_frames)

        left_amps = []
        right_amps = []

        # Integrate around Left peaks to get amplitude (deg)
        for peak in left_peaks:
            # Find bounds where Vr drops below 50 deg/s threshold (saccade start/end)
            start = peak
            while start > 0 and vr[start] > 50:
                start -= 1
            end = peak
            while end < len(vr) - 1 and vr[end] > 50:
                end += 1
            
            # Trapz integration over time (sec) converted to degrees
            amp = np.trapezoid(vr[start:end], time_vec[start:end])
            left_amps.append(amp)

        # Integrate around Right peaks (take absolute value for magnitude)
        for peak in right_peaks:
            start = peak
            while start > 0 and vr[start] < -50:
                start -= 1
            end = peak
            while end < len(vr) - 1 and vr[end] < -50:
                end += 1
            
            amp = np.abs(np.trapezoid(vr[start:end], time_vec[start:end]))
            right_amps.append(amp)

        # Save metrics
        iri["left_amplitudes"] = np.array(left_amps)      # Array of Left magnitudes (deg)
        iri["right_amplitudes"] = np.array(right_amps)    # Array of Right magnitudes (deg)
        
        mean_L = np.mean(left_amps) if len(left_amps) > 0 else 0
        mean_R = np.mean(right_amps) if len(right_amps) > 0 else 0
        
        # Magnitude Asymmetry Ratio
        if (mean_L + mean_R) > 0:
            iri["amp_asymmetry"] = (mean_L - mean_R) / (mean_L + mean_R)
        else:
            iri["amp_asymmetry"] = 0.0

    return iri_data