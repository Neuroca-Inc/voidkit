# Below is the exact Python implementation to generate the grid points and 
# calculate the crucial Yin-to-Yang and Yang-to-Yin coordinate transformations 
# required for ghost-zone boundary interpolation: https://www.aanda.org/articles/aa/full_html/2010/06/aa13435-09/aa13435-09.html

import numpy as np

def create_component_grid(N_theta, N_phi):
    """Generates the base 2D angular grid for a single Yin or Yang patch."""
    # Bounds for the Yin-Yang component grids
    theta_min, theta_max = np.pi / 4.0, 3.0 * np.pi / 4.0
    phi_min, phi_max = -3.0 * np.pi / 4.0, 3.0 * np.pi / 4.0
    
    theta = np.linspace(theta_min, theta_max, N_theta)
    phi = np.linspace(phi_min, phi_max, N_phi)
    
    return np.meshgrid(theta, phi, indexing='ij')

def transform_yin_to_yang(theta_n, phi_n):
    """
    Transforms spherical coordinates from the Yin patch (n) to the Yang patch (e).
    Based on the analytic rotation matrix relation: (xe, ye, ze) = (-xn, zn, yn)
    """
    # 1. Convert Yin angular coordinates to Yin Cartesian coordinates (on a unit sphere)
    x_n = np.sin(theta_n) * np.cos(phi_n)
    y_n = np.sin(theta_n) * np.sin(phi_n)
    z_n = np.cos(theta_n)
    
    # 2. Apply the Kageyama transformation matrix rotation
    x_e = -x_n
    y_e = z_n
    z_e = y_n
    
    # 3. Convert back to Yang spherical coordinates
    theta_e = np.arccos(z_e)
    phi_e = np.arctan2(y_e, x_e)
    
    return theta_e, phi_e

def transform_yang_to_yin(theta_e, phi_e):
    """
    Transforms spherical coordinates from the Yang patch (e) back to the Yin patch (n).
    The inverse operation shares the exact same rotation symmetry.
    """
    x_e = np.sin(theta_e) * np.cos(phi_e)
    y_e = np.sin(theta_e) * np.sin(phi_e)
    z_e = np.cos(theta_e)
    
    x_n = -x_e
    y_n = z_e
    z_n = y_e
    
    theta_n = np.arccos(z_n)
    phi_n = np.arctan2(y_n, x_n)
    
    return theta_n, phi_n

# Example Usage:
# Generate a 100x200 grid for a patch
N_theta, N_phi = 100, 200
theta_grid, phi_grid = create_component_grid(N_theta, N_phi)

# Transform the entire Yin grid boundary into Yang space to find interpolation weights
theta_mapped_to_yang, phi_mapped_to_yang = transform_yin_to_yang(theta_grid, phi_grid)
