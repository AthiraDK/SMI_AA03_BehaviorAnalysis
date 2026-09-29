import numpy as np
import pandas as pd

def get_data_tracking(fpath_track):

    df_track = pd.read_csv(fpath_track, header=0)
    
    df_track["ori_angle_cal"] = -1 * df_track["ori_angle_cal"].astype(np.float32) #TODO: check if bonsai already does it

    df_track["X_cal"] = df_track["X_cal"].astype(np.float32)
    df_track["Y_cal"] = df_track["Y_cal"].astype(np.float32)

    ### Calibration values from Corinna (December 2025)
    # X = -dt.X/6.63364;
    # Y = dt.Y/6.63364;

    df_track["X"] = df_track["X_cal"].apply(lambda x: (-1*x)/6.63364)
    df_track["Y"] = df_track["Y_cal"].apply(lambda y: y/6.63364)
    
    return df_track

def px_to_mm(df_track, calib_x = -6.63364, calib_y=6.63364):

    df_track["X"] = df_track["X_cal"].apply(lambda x: x/calib_x)
    df_track["Y"] = df_track["Y_cal"].apply(lambda y: y/calib_y)

    return df_track