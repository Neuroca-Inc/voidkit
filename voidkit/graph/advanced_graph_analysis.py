"""Curated advanced NetworkX adapters."""

from __future__ import annotations

from typing import Any, Dict, Optional

import networkx as nx


def calculate_graph_edit_distance(graph1: nx.Graph, graph2: nx.Graph) -> Optional[float]:
    """Return NetworkX graph edit distance, preserving ``None`` on timeout/failure."""
    return nx.graph_edit_distance(graph1, graph2)


def calculate_pagerank(graph: nx.Graph, alpha: float = 0.85) -> Dict[Any, float]:
    """Return PageRank scores through VoidKit's normalized graph namespace."""
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must satisfy 0 < alpha < 1.")
    return nx.pagerank(graph, alpha=alpha)
