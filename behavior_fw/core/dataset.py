import pandas as pd
from pathlib import Path



def create_experiments_metadata(path_dir_dataset ):

    experiment_folders = [
        f.parent for f in path_dir_dataset.rglob("RawTracking.csv") if f.is_file()
    ]

    records = []
    for exp_folder in sorted(experiment_folders):
        folder_name = exp_folder.name
        
        """
        Folder format: YYYYMMDD_HHMMSS_GENOTYPE_EXPTYPEPROTOCOL_MEDIUM
        Example: 20260816_150347_Gr43a_OptoCL_ATR
        """
        folder_name = folder_name.rstrip("_")
        parts = folder_name.split("_")
        
        if len(parts) == 5:
            date_str, time_str, genotype, protocol, medium = parts

            records.append(
                {
                    "folder_name": folder_name,
                    "folder_path": str(exp_folder.resolve()),
                    "date": date_str,
                    "time": time_str,
                    "datetime": pd.to_datetime(
                        f"{date_str}_{time_str}", format="%Y%m%d_%H%M%S"
                    ),
                    "genotype": genotype,
                    "protocol": protocol,
                    "medium": medium,
                }
            )

    df_metadata = pd.DataFrame(records)
    return df_metadata

def select_exp_condition(df_metadata, genotype=None, protocol=None, medium=None):

    df_meta = df_metadata.copy()
    if genotype:
        df_meta = df_meta[df_meta["genotype"]==genotype]
    if protocol:
        df_meta = df_meta[df_meta["protocol"]==protocol]
    if medium:
        df_meta = df_meta[df_meta["medium"]==medium]

    return df_meta

