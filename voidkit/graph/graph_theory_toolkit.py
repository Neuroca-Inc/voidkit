"""Curated graph-analysis adapters."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import networkx as nx
import numpy as np


def calculate_graph_metrics(graph: nx.Graph) -> Dict[str, Any]:
    """Calculate a normalized set of common graph summary metrics."""
    if not isinstance(graph, nx.Graph):
        raise TypeError("Input must be a NetworkX graph.")

    degrees = [degree for _, degree in graph.degree()]
    connected = graph.number_of_nodes() > 0 and nx.is_connected(graph.to_undirected())
    return {
        "num_nodes": graph.number_of_nodes(),
        "num_edges": graph.number_of_edges(),
        "density": nx.density(graph),
        "avg_degree": float(np.mean(degrees)) if degrees else 0.0,
        "avg_clustering_coefficient": float(nx.average_clustering(graph)),
        "avg_shortest_path_length": (
            float(nx.average_shortest_path_length(graph)) if connected else float("inf")
        ),
    }


def detect_communities(
    graph: nx.Graph,
    method: str = "louvain",
    *,
    seed: Optional[int] = None,
) -> List[List[Any]]:
    """Detect graph communities with an explicitly named NetworkX algorithm."""
    if not isinstance(graph, nx.Graph):
        raise TypeError("Input must be a NetworkX graph.")
    if graph.number_of_nodes() == 0:
        return []

    if method == "louvain":
        communities = nx.community.louvain_communities(graph, seed=seed)
    elif method in {"greedy", "greedy_modularity"}:
        communities = nx.community.greedy_modularity_communities(graph)
    else:
        raise ValueError("method must be 'louvain', 'greedy', or 'greedy_modularity'.")
    return [list(community) for community in communities]
