def transformXY_FlyVrena(X, Y, coeffs):
    X_transform = coeffs[0] + (coeffs[1] * X) + (coeffs[2] * Y) + (coeffs[3] * (X * Y)) + (coeffs[4] * (Y**2))
    return -0.78 * X_transform

def zone_px_to_mm(center_zone_px_x = 300, center_zone_px_y = 300, radius_zone_px = 20, aquisition="FlyVrena"):

    if aquisition == "FlyVrena":
        coeffs_Y =  [-493.2, 1.647, 0.2436, -0.000762, -0.0000964]
        coeffs_X = [460.8, -0.0375, -1.914, 0.000119, 0.000782]

        center_zone_mm_y = transformXY_FlyVrena(center_zone_px_x, center_zone_px_y, coeffs=coeffs_Y)
        center_zone_mm_x = transformXY_FlyVrena(center_zone_px_x, center_zone_px_y, coeffs=coeffs_X)

    
    center_zone_mm_x = (-1*center_zone_mm_x)/6.63364
    center_zone_mm_y = center_zone_mm_y/6.63364

    radius_zone_mm = radius_zone_px/6.63364 

    return center_zone_mm_x, center_zone_mm_y, radius_zone_mm