# Constants
c = 2.99792458e8  # Speed of light

# Reference positions for the sphere measurements
measured_sphere_pos = {
    1: (-0.10775, 0.04108, 0.33035),
    2: (-0.09275, 0.04108, 0.33035),
    3: (-0.03275, 0.04108, 0.33035),
    4: (0.027525, 0.04108, 0.33035),
}

measured_sphere_plane_pos = {
    1: (-0.1095, 0.0415, 0.345),
    2: (-0.0495, 0.0415, 0.345),
    3: (0.0005, 0.0415, 0.345),
    4: (0.0405, 0.0415, 0.345),
    5: (0.0705, 0.0415, 0.345),
    6: (0.0905, 0.0415, 0.345),
}

measured_torus_pos = {
    1: (-0.03975, 0.04108, 0.32035),  # 33035
    2: (-0.02475, 0.04108, 0.32035),
    3: (0.03525, 0.04108, 0.32035),
    4: (0.095525, 0.04108, 0.32035),
}

measured_plane_pos = {1: 0.345, 2: 0.336, 3: 0.325}

measured_sphere_rad = {1: 0.02, 2: 0.015, 3: 0.0125, 4: 0.01}

measured_sphere_height = {1: 0.330, 2: 0.327, 3: 0.320}

measured_wood_pos = {1: 0.313, 2: 0.323, 3: 0.333}

# MIMO SAR System
import mimosar as mimo

mimo_system = mimo.MIMOSAR()
