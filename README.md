# VoidKit

**Unified Void Dynamics Model (VDM) Toolkit**

VoidKit is a comprehensive Python package for void dynamics modeling, scientific computing, and mathematical analysis. It provides tools for the Unified Void Dynamics Model (VDM) along with advanced mathematical functions, statistical analysis, and more.

## Features

- **Void Dynamics**: Core VDM functions including REVGSP (Resonance-Enhanced Valence-Gated Synaptic Plasticity) and GDSP (Goal-Directed Structural Plasticity)
- **Advanced Mathematics**: Descriptive statistics, symbolic differentiation, and mathematical utilities
- **Scientific Computing**: Graph theory, time series analysis, numerical methods, and more
- **CLI Tools**: Command-line interfaces for common operations

## Installation

```bash
pip install voidkit
```

For symbolic math functionality:
```bash
pip install voidkit[symbolic]
```

## Quick Start

### Descriptive Statistics

```python
from voidkit.advanced_math import descriptive_stats

data = [1, 2, 3, 4, 5]
stats = descriptive_stats(data, ddof=1)
print(f"Mean: {stats['mean']}")
print(f"Standard deviation: {stats['std']}")
```

### Symbolic Differentiation

```python
from voidkit.advanced_math import symbolic_diff

# Differentiate x^2 + 3x
result = symbolic_diff("x**2 + 3*x", var="x")
print(result)  # 2*x + 3
```

### Void Dynamics

```python
from voidkit.void_dynamics.void_equations import delta_re_vgsp, delta_gdsp

# VDM core functions
W = 0.5  # void state
t = 1.0  # time step

# Resonance-Enhanced dynamics
alpha_delta = delta_re_vgsp(W, t)
print(f"RE-VGSP delta: {alpha_delta}")

# Goal-Directed dynamics  
omega_delta = delta_gdsp(W, t)
print(f"GDSP delta: {omega_delta}")
```

## CLI Usage

### Descriptive Statistics
```bash
voidkit-stats 1 2 3 4 5 --json
```

### Symbolic Differentiation  
```bash
voidkit-diff "sin(x)**2 + x**3" --var x --order 1
```

## Package Structure

- `voidkit.advanced_math` - Mathematical utilities and statistics
- `voidkit.void_dynamics` - Core VDM functions and equations  
- `voidkit.graph` - Graph theory and network analysis
- `voidkit.time_series` - Time series analysis tools
- `voidkit.numerical` - Numerical methods and solvers
- And many more specialized modules...

## License

This research is protected under a dual-license to foster open academic research while ensuring commercial applications are aligned with the project's ethical principles. Commercial use requires written permission from Justin K. Lietz. See LICENSE file for full terms.

## Copyright

Copyright © 2025 Justin K. Lietz, Neuroca, Inc. All Rights Reserved.
