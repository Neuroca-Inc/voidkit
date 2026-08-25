from __future__ import annotations
from math import pi
from pathlib import Path
from typing import Dict, List
import numpy as np
from PIL import Image

TAU = 2.0 * pi
READOUTS = ['binary_sign','grayscale_tanh','completion_occupancy','recurrence_density']


def _angdiff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.abs((a - b + np.pi) % (2*np.pi) - np.pi)


def _center_half(s: Dict[str, int]) -> tuple[float, float]:
    cL = pi * (s['cL_num'] / s['c_den'])
    cR = pi * (s['cR_num'] / s['c_den'])
    return ((cL + cR) / 2.0) % TAU, max(abs(cR-cL)/2.0, 1e-8)


def _grid(size: int):
    xs = np.linspace(-1.0, 1.0, size)
    ys = np.linspace(1.0, -1.0, size)
    X, Y = np.meshgrid(xs, ys)
    R = np.sqrt(X*X + Y*Y)
    T = np.arctan2(Y, X) % TAU
    inside = R <= 1.0
    rho = np.zeros_like(R)
    rho[inside] = -np.log(np.clip(1.0 - R[inside], 1e-6, None))
    return X, Y, R, T, rho, inside


def _base_fields(trace: List[Dict[str, int]], size: int):
    X, Y, R, T, rho, inside = _grid(size)
    F = np.zeros_like(R)
    O = np.zeros_like(R)
    D = np.zeros_like(R)
    max_rden = max(s['r_den'] for s in trace)
    log_max = np.log1p(max_rden)
    for s in trace:
        center, half = _center_half(s)
        ang = np.exp(-(_angdiff(T, center)/(0.7*half + 1e-9))**2)
        rc = 0.08 + 0.84 * (np.log1p(s['r_den']) / log_max)
        shell = 0.02 + 0.05/(1.0+s['A'])
        radial = np.exp(-((R-rc)/shell)**2)
        packet = ang * radial
        phase = s['u']*T + s['v']*rho + (s['theta_ticks']*pi/2.0) + s['kappa']*half
        amp = (1.0 + 0.15*s['A'] + 0.45*s['window_ready'] + 0.7*s['carry_event']) / (1.0 + 0.05*np.sqrt(s['step']))
        F += amp * packet * np.cos(phase)
        O += (0.3 + 0.7*s['window_ready']) * packet
        D += packet / (1.0 + 0.08*s['step'])
    for _ in range(3):
        F = (4*F + np.roll(F,1,0)+np.roll(F,-1,0)+np.roll(F,1,1)+np.roll(F,-1,1))/8.0
        O = (4*O + np.roll(O,1,0)+np.roll(O,-1,0)+np.roll(O,1,1)+np.roll(O,-1,1))/8.0
        D = (4*D + np.roll(D,1,0)+np.roll(D,-1,0)+np.roll(D,1,1)+np.roll(D,-1,1))/8.0
    F[~inside] = 0.0; O[~inside] = 0.0; D[~inside] = 0.0
    return {'F':F,'O':O,'D':D,'inside':inside}


def render_readout(trace: List[Dict[str, int]], size: int, readout: str) -> np.ndarray:
    base = _base_fields(trace, size)
    F, O, D, inside = base['F'], base['O'], base['D'], base['inside']
    if readout == 'binary_sign':
        arr = np.zeros_like(F)
        arr[inside] = (F[inside] >= 0.0).astype(float)
        return arr
    if readout == 'grayscale_tanh':
        arr = np.zeros_like(F)
        arr[inside] = (np.tanh(1.4*F[inside]) + 1.0)/2.0
        return arr
    if readout == 'completion_occupancy':
        arr = np.zeros_like(O); m = O[inside].max() if np.any(inside) else 1.0
        arr[inside] = np.clip(O[inside]/(m+1e-12), 0.0, 1.0)
        return arr
    if readout == 'recurrence_density':
        arr = np.zeros_like(D); m = D[inside].max() if np.any(inside) else 1.0
        arr[inside] = np.sqrt(np.clip(D[inside]/(m+1e-12), 0.0, 1.0))
        return arr
    raise ValueError(readout)


def render_readout_images(trace: List[Dict[str, int]], output_dir: str | Path, size: int = 512) -> dict[str, np.ndarray]:
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    imgs = {}
    for name in READOUTS:
        arr = render_readout(trace, size=size, readout=name)
        Image.fromarray((255*np.clip(arr,0.0,1.0)).astype(np.uint8), mode='L').save(out/f'{name}.png')
        imgs[name] = arr
    return imgs
