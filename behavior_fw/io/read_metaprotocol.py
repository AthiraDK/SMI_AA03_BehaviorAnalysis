import pandas as pd
from behavior_fw.preprocessing.calibrations import zone_px_to_mm


def get_protocol_params(fpath_meta_protocol):

    dict_protocol = pd.read_csv(fpath_meta_protocol, header=0).to_dict(orient="records")[0]
    

    center_zone_mm_x, center_zone_mm_y, radius_zone_mm = zone_px_to_mm(dict_protocol["ZoneCenterX_px"], 
                                                                       dict_protocol["ZoneCenterY_px"], 
                                                                       dict_protocol["ZoneRadius_px"])
    

    dict_protocol["ZoneCenterX_mm"] = center_zone_mm_x
    dict_protocol["ZoneCenterY_mm"] = center_zone_mm_y
    dict_protocol["ZoneRadius_mm"] = radius_zone_mm

    return dict_protocol
    
    


