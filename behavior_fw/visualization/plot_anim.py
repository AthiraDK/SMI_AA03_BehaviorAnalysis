import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# =========================================================================
# CONFIGURATION & PARAMETERS
# =========================================================================

LOWRES_VIDEO_PATH = "/media/athira/Crucial_AA/Data_backup/LocalSearch_OptoCL/Gr5a_Gal4/20260927_214323_Gr5a_OptoCL_ATR/LowResVideo.avi"     # Path to 60 FPS low-res video
HIGHRES_VIDEO_PATH = "/media/athira/Crucial_AA/Data_backup/LocalSearch_OptoCL/Gr5a_Gal4/20260927_214323_Gr5a_OptoCL_ATR/HighResVideo.avi"   # Path to 120 FPS high-res video
OUTPUT_VIDEO_PATH = "annotated_4panel_3to5min.mp4"



# Arena & Zone dimensions (mm or px)
ARENA_R = 45.0
ZONE_X, ZONE_Y, ZONE_R = -5.121893862193295, 0.38520028219800495, 3.0149359929088706


START_TIME_SEC = 180.0  # 3 minutes
END_TIME_SEC = 240.0    # 4 minutes

# Set playback FPS to 30.0 for 2x slow-motion (Data is sampled at 60 Hz)
SAMPLING_FPS = 60.0
OUTPUT_FPS = 30.0       # 60 Hz / 30 FPS = 2x slowdown



def get_canvas_bgr_image(fig):
    """Extracts BGR frame from Matplotlib canvas safely across all Matplotlib versions."""
    fig.canvas.draw()
    
    # Matplotlib >= 3.8
    if hasattr(fig.canvas, "buffer_rgba"):
        buf = np.asarray(fig.canvas.buffer_rgba())
        return cv2.cvtColor(buf, cv2.COLOR_RGBA2BGR)
    
    # Matplotlib < 3.8 fallback
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return cv2.cvtColor(buf, cv2.COLOR_RGB2BGR)


def create_synchronized_animation():
    # -------------------------------------------------------------------------
    # 1. LOAD DATA & FILTER TIME WINDOW (180s - 240s)
    # -------------------------------------------------------------------------
    df_proc = pd.read_csv("/home/athira/CF/PhD/Data/LocalSearch_OptoCL/Gr5a_Gal4/20260927_214323_Gr5a_OptoCL_ATR/ProcessedTracking.csv")
    df_meta = pd.read_csv("/home/athira/CF/PhD/Data/LocalSearch_OptoCL/Gr5a_Gal4/20260927_214323_Gr5a_OptoCL_ATR/HighResMeta.csv")

    start_idx = int(START_TIME_SEC * SAMPLING_FPS)
    end_idx = int(END_TIME_SEC * SAMPLING_FPS)

    sub_proc = df_proc.iloc[start_idx:end_idx].copy().reset_index(drop=True)
    sub_proc["Time_window_s"] = sub_proc.index / SAMPLING_FPS

    # Video Frame Index Lookups relative to video start
    lowres_base_frame = df_proc["LowResFrameNumber"].iloc[0]
    highres_base_frame = df_meta["HighResFrameNumber"].iloc[0]

    sub_proc["LowRes_File_Frame"] = sub_proc["LowResFrameNumber"] - lowres_base_frame
    sub_proc["HighRes_File_Frame"] = sub_proc["HighResFrameNumber"] - highres_base_frame

    n_frames = len(sub_proc)
    print(f"Loaded {n_frames} frames for 3-4 min window (Slow motion output @ {OUTPUT_FPS} FPS).")

    # -------------------------------------------------------------------------
    # 2. OPEN VIDEO CAPTURES
    # -------------------------------------------------------------------------
    cap_low = cv2.VideoCapture(LOWRES_VIDEO_PATH)
    cap_high = cv2.VideoCapture(HIGHRES_VIDEO_PATH)

    # Setup Matplotlib Layout with GridSpec
    fig = plt.figure(figsize=(16, 9), dpi=100)
    gs = GridSpec(2, 2, figure=fig, wspace=0.22, hspace=0.25)

    ax_low = fig.add_subplot(gs[0, 0])
    ax_high = fig.add_subplot(gs[0, 1])
    ax_traj = fig.add_subplot(gs[1, 0])

    # Panel 4: Subsplit into 2 stacked subplots sharing time axis
    gs_kin = gs[1, 1].subgridspec(2, 1, hspace=0.20)
    ax_vf = fig.add_subplot(gs_kin[0, 0])
    ax_vr = fig.add_subplot(gs_kin[1, 0], sharex=ax_vf)

    # -------------------------------------------------------------------------
    # 3. SETUP VIDEO WRITER
    # -------------------------------------------------------------------------
    frame_bgr = get_canvas_bgr_image(fig)
    height, width, _ = frame_bgr.shape

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, OUTPUT_FPS, (width, height))

    print("Rendering slow-motion video frames...")

    time_vec = sub_proc["Time_window_s"] + START_TIME_SEC

    # -------------------------------------------------------------------------
    # 4. ANIMATION RENDER LOOP
    # -------------------------------------------------------------------------
    for i in range(n_frames):
        row = sub_proc.iloc[i]
        curr_time = row["Time_window_s"] + START_TIME_SEC

        # Read Low-Res Frame
        cap_low.set(cv2.CAP_PROP_POS_FRAMES, int(row["LowRes_File_Frame"]))
        ret_low, img_low = cap_low.read()
        if not ret_low:
            img_low = np.zeros((480, 640, 3), dtype=np.uint8)

        # Read High-Res Frame (120 Hz sync)
        cap_high.set(cv2.CAP_PROP_POS_FRAMES, int(row["HighRes_File_Frame"]))
        ret_high, img_high = cap_high.read()
        if not ret_high:
            img_high = np.zeros((480, 640, 3), dtype=np.uint8)

        # Clear Axes for current frame redraw
        ax_low.clear()
        ax_high.clear()
        ax_traj.clear()
        ax_vf.clear()
        ax_vr.clear()

        # --- Panel 1: Low-Res Video Frame ---
        ax_low.imshow(cv2.cvtColor(img_low, cv2.COLOR_BGR2RGB))
        ax_low.set_title(f"Low-Res Video (60Hz) | t = {curr_time:.2f}s", fontsize=10, fontweight="bold")
        ax_low.axis("off")

        # --- Panel 2: High-Res Video Frame ---
        ax_high.imshow(cv2.cvtColor(img_high, cv2.COLOR_BGR2RGB))
        ax_high.set_title(f"High-Res Video (120Hz) | Frame: {int(row['HighResFrameNumber'])}", fontsize=10, fontweight="bold")
        ax_high.axis("off")

        # Overlay LED Stimulus State if active
        if row["LEDstate"] == 1:
            ax_low.text(10, 35, "LED ON", color="red", fontsize=11, fontweight="bold",
                        bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.85))

        # --- Panel 3: Trajectory Plot ---
        traj_sub = sub_proc.iloc[: i + 1]
        ax_traj.plot(traj_sub["X"], traj_sub["Y"], color="#1f77b4", linewidth=1.2, alpha=0.7)
        ax_traj.plot(row["X"], row["Y"], "ro", markersize=6, label="Current Position")

        arena_circle = plt.Circle((0, 0), ARENA_R, fill=False, color="gray", linestyle="--")
        zone_circle = plt.Circle((ZONE_X, ZONE_Y), ZONE_R, fill=False, color="red", linestyle=":")
        ax_traj.add_patch(arena_circle)
        ax_traj.add_patch(zone_circle)

        ax_traj.set_xlim(-ARENA_R - 5, ARENA_R + 5)
        ax_traj.set_ylim(-ARENA_R - 5, ARENA_R + 5)
        ax_traj.set_aspect("equal")
        ax_traj.set_xlabel("X (mm)", fontsize=9)
        ax_traj.set_ylabel("Y (mm)", fontsize=9)
        ax_traj.set_title("Synchronized Trajectory", fontsize=10, fontweight="bold")

        # --- Panel 4a: Forward Velocity (Vf_smooth) ---
        ax_vf.plot(time_vec, sub_proc["Vf_smooth"], color="crimson", linewidth=1.0)
        ax_vf.axvline(curr_time, color="black", linestyle="--", linewidth=1.2)
        ax_vf.set_ylabel("V_fwd (mm/s)", fontsize=8, fontweight="bold")
        ax_vf.set_title("Kinematics Timeline", fontsize=10, fontweight="bold")
        ax_vf.tick_params(labelbottom=False)  # Hide top subplot x-labels

        # --- Panel 4b: Rotational Velocity (Vr_smooth) ---
        ax_vr.plot(time_vec, sub_proc["Vr_smooth"], color="teal", linewidth=1.0)
        ax_vr.axvline(curr_time, color="black", linestyle="--", linewidth=1.2)
        ax_vr.set_xlabel("Time (s)", fontsize=9)
        ax_vr.set_ylabel("V_rot (deg/s)", fontsize=8, fontweight="bold")
        ax_vr.set_xlim(START_TIME_SEC, END_TIME_SEC)

        # Render & Write Video Frame
        frame_bgr = get_canvas_bgr_image(fig)
        out_writer.write(frame_bgr)

        if (i + 1) % 300 == 0:
            print(f"Processed {i + 1} / {n_frames} frames ({(i+1)/n_frames*100:.1f}%)")

    # Cleanup
    cap_low.release()
    cap_high.release()
    out_writer.release()
    plt.close(fig)
    print(f"Successfully rendered slow-motion animation: {OUTPUT_VIDEO_PATH}")



if __name__ == "__main__":
    create_synchronized_animation()