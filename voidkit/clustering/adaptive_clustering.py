"""
Copyright © 2025 Justin K. Lietz, Neuroca, Inc.
SPDX-License-Identifier: BSD-3-Clause

Licensed under the BSD 3-Clause License. See LICENSE in the repository root.
"""

import numpy as np
import networkx as nx
from scipy.stats import entropy

def calculate_adaptive_clustering_interval(
    graph: nx.Graph,
    base_interval: float = 100000.0,
    alpha: float = 0.05
) -> float:
    """
    Calculates the adaptive clustering interval based on graph entropy.

    t_cluster = base_interval * e^(-α * graph_entropy)

    Parameters
    ----------
    graph : nx.Graph
        The graph to analyze.
    base_interval : float, optional
        The base clustering interval, by default 100000.0.
    alpha : float, optional
        The scaling factor for the entropy, by default 0.05.

    Returns
    -------
    float
        The calculated adaptive clustering interval.
    """
    degrees = [d for n, d in graph.degree()]
    if not degrees:
        return base_interval
        
    degree_counts = np.bincount(degrees)
    degree_distribution = degree_counts / np.sum(degree_counts)
    
    graph_entropy = entropy(degree_distribution, base=2)
    
    return base_interval * np.exp(-alpha * graph_entropy)
