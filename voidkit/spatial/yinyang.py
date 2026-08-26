"""Yin-Yang overset spherical-grid coordinates and patch transforms."""
from __future__ import annotations

import numpy as np


def create_component_grid(n_theta: int, n_phi: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate the base angular grid for one Yin-Yang component patch."""
    if n_theta < 2 or n_phi < 2:
        raise ValueError("n_theta and n_phi must each be at least 2")
    theta_min, theta_max = np.pi / 4.0, 3.0 * np.pi / 4.0
    phi_min, phi_max = -3.0 * np.pi / 4.0, 3.0 * np.pi / 4.0
    theta = np.linspace(theta_min, theta_max, n_theta)
    phi = np.linspace(phi_min, phi_max, n_phi)
    return np.meshgrid(theta, phi, indexing="ij")


def yin_to_yang(theta: np.ndarray, phi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Transform spherical coordinates from the Yin patch to the Yang patch."""
    theta = np.asarray(theta)
    phi = np.asarray(phi)
    x_n = np.sin(theta) * np.cos(phi)
    y_n = np.sin(theta) * np.sin(phi)
    z_n = np.cos(theta)
    x_e = -x_n
    y_e = z_n
    z_e = y_n
    return np.arccos(np.clip(z_e, -1.0, 1.0)), np.arctan2(y_e, x_e)


def yang_to_yin(theta: np.ndarray, phi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Transform spherical coordinates from the Yang patch to the Yin patch."""
    theta = np.asarray(theta)
    phi = np.asarray(phi)
    x_e = np.sin(theta) * np.cos(phi)
    y_e = np.sin(theta) * np.sin(phi)
    z_e = np.cos(theta)
    x_n = -x_e
    y_n = z_e
    z_n = y_e
    return np.arccos(np.clip(z_n, -1.0, 1.0)), np.arctan2(y_n, x_n)


# Historical source names.
transform_yin_to_yang = yin_to_yang
transform_yang_to_yin = yang_to_yin
