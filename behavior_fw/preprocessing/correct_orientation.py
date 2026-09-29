import numpy as np

def mod2pi(angle):
    while(angle <= -180.0):
        angle =  angle + (2 * 180.0)
    while(angle > 180.0):
        angle = angle - (2 * 180.0)
    return angle


def corr_orientation(or_curr, or_prev):

    try:
        # number of 180 degree differences
        npi = int(np.round((or_prev - or_curr) / 180.0))
        
        term1 = np.abs(mod2pi(or_curr - or_prev + 180.0) / 2.0)
        term2 = np.abs(mod2pi(or_curr - or_prev))
        
        if term1 > term2:
            corrected_or = or_curr + (npi - (npi % 2)) * 180.0
            # if npi !=0:
            #     print(or_curr, or_prev, corrected_or, mod2pi(or_curr - or_prev + 180.0), mod2pi(or_curr - or_prev), npi)
        else:
            corrected_or = or_curr + npi * 180.0
        return corrected_or
    except ValueError as ve: # Nan cases
        return np.nan