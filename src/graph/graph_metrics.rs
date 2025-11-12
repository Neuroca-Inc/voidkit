// Copyright © 2025 Justin K. Lietz, Neuroca, Inc. All Rights Reserved.
//
// This research is protected under a dual-license to foster open academic
// research while ensuring commercial applications are aligned with the project's ethical principles.
// Commercial use requires written permission from Justin K. Lietz.
// See LICENSE file for full terms.

use petgraph::graph::UnGraph;
use petgraph::algo::connected_components;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashMap;

/// Calculates standard metrics for a graph from edge list.
///
/// # Arguments
///
/// * `edges` - List of edges as tuples (node1, node2)
/// * `num_nodes` - Total number of nodes in the graph
///
/// # Returns
///
/// A dictionary containing:
/// - `num_nodes`: Number of nodes
/// - `num_edges`: Number of edges  
/// - `density`: Graph density (actual edges / possible edges)
/// - `avg_degree`: Average node degree
/// - `avg_clustering_coefficient`: Average clustering coefficient
/// - `is_connected`: Whether the graph is connected
/// - `num_components`: Number of connected components
///
/// # Example
///
/// ```
/// use voidkit_rust::graph::calculate_graph_metrics;
///
/// let edges = vec![(0, 1), (1, 2), (2, 0), (2, 3)];
/// let metrics = calculate_graph_metrics(&edges, 4);
/// assert_eq!(metrics.get("num_nodes"), Some(&4.0));
/// assert_eq!(metrics.get("num_edges"), Some(&4.0));
/// ```
pub fn calculate_graph_metrics(
    edges: &[(usize, usize)],
    num_nodes: usize,
) -> HashMap<String, f64> {
    // Build undirected graph
    let mut graph = UnGraph::<(), ()>::with_capacity(num_nodes, edges.len());
    
    // Add nodes
    let mut node_indices = Vec::new();
    for _ in 0..num_nodes {
        node_indices.push(graph.add_node(()));
    }
    
    // Add edges
    for &(i, j) in edges {
        if i < num_nodes && j < num_nodes {
            graph.add_edge(node_indices[i], node_indices[j], ());
        }
    }
    
    let mut metrics = HashMap::new();
    
    // Basic metrics
    metrics.insert("num_nodes".to_string(), num_nodes as f64);
    metrics.insert("num_edges".to_string(), edges.len() as f64);
    
    // Density: actual_edges / possible_edges
    let possible_edges = if num_nodes > 1 {
        (num_nodes * (num_nodes - 1)) / 2
    } else {
        1
    };
    let density = if possible_edges > 0 {
        edges.len() as f64 / possible_edges as f64
    } else {
        0.0
    };
    metrics.insert("density".to_string(), density);
    
    // Average degree
    let total_degree: usize = edges.len() * 2; // Each edge contributes to 2 nodes
    let avg_degree = if num_nodes > 0 {
        total_degree as f64 / num_nodes as f64
    } else {
        0.0
    };
    metrics.insert("avg_degree".to_string(), avg_degree);
    
    // Clustering coefficient (simplified - local clustering)
    let avg_clustering = calculate_average_clustering(&graph);
    metrics.insert("avg_clustering_coefficient".to_string(), avg_clustering);
    
    // Connectivity
    let num_components = connected_components(&graph);
    let is_conn = num_components == 1 && num_nodes > 0;
    metrics.insert("is_connected".to_string(), if is_conn { 1.0 } else { 0.0 });
    metrics.insert("num_components".to_string(), num_components as f64);
    
    metrics
}

/// Calculates the average clustering coefficient of a graph
fn calculate_average_clustering(graph: &UnGraph<(), ()>) -> f64 {
    let node_count = graph.node_count();
    if node_count == 0 {
        return 0.0;
    }
    
    let mut total_clustering = 0.0;
    
    for node in graph.node_indices() {
        // Get neighbors
        let neighbors: Vec<_> = graph.neighbors(node).collect();
        let k = neighbors.len();
        
        if k < 2 {
            continue; // Clustering coefficient is 0 for nodes with < 2 neighbors
        }
        
        // Count triangles (edges between neighbors)
        let mut triangles = 0;
        for i in 0..neighbors.len() {
            for j in (i + 1)..neighbors.len() {
                if graph.contains_edge(neighbors[i], neighbors[j]) {
                    triangles += 1;
                }
            }
        }
        
        // Local clustering coefficient
        let possible_edges = (k * (k - 1)) / 2;
        let local_clustering = if possible_edges > 0 {
            triangles as f64 / possible_edges as f64
        } else {
            0.0
        };
        
        total_clustering += local_clustering;
    }
    
    total_clustering / node_count as f64
}

/// Python wrapper for calculate_graph_metrics
#[pyfunction]
#[pyo3(name = "calculate_graph_metrics")]
pub fn calculate_graph_metrics_py<'py>(
    py: Python<'py>,
    edges: Vec<(usize, usize)>,
    num_nodes: usize,
) -> PyResult<Py<PyDict>> {
    let metrics = calculate_graph_metrics(&edges, num_nodes);
    
    // Convert to Python dictionary
    let dict = PyDict::new_bound(py);
    for (key, value) in metrics {
        dict.set_item(key, value)?;
    }
    
    Ok(dict.unbind())
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_empty_graph() {
        let edges = vec![];
        let metrics = calculate_graph_metrics(&edges, 0);
        
        assert_eq!(metrics.get("num_nodes"), Some(&0.0));
        assert_eq!(metrics.get("num_edges"), Some(&0.0));
    }

    #[test]
    fn test_single_node() {
        let edges = vec![];
        let metrics = calculate_graph_metrics(&edges, 1);
        
        assert_eq!(metrics.get("num_nodes"), Some(&1.0));
        assert_eq!(metrics.get("num_edges"), Some(&0.0));
        assert_relative_eq!(metrics["density"], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_two_connected_nodes() {
        let edges = vec![(0, 1)];
        let metrics = calculate_graph_metrics(&edges, 2);
        
        assert_eq!(metrics.get("num_nodes"), Some(&2.0));
        assert_eq!(metrics.get("num_edges"), Some(&1.0));
        assert_relative_eq!(metrics["density"], 1.0, epsilon = 1e-10); // Complete graph
        assert_relative_eq!(metrics["avg_degree"], 1.0, epsilon = 1e-10);
        assert_relative_eq!(metrics["is_connected"], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_triangle() {
        let edges = vec![(0, 1), (1, 2), (2, 0)];
        let metrics = calculate_graph_metrics(&edges, 3);
        
        assert_eq!(metrics.get("num_nodes"), Some(&3.0));
        assert_eq!(metrics.get("num_edges"), Some(&3.0));
        assert_relative_eq!(metrics["density"], 1.0, epsilon = 1e-10); // Complete graph K3
        assert_relative_eq!(metrics["avg_degree"], 2.0, epsilon = 1e-10);
        assert_relative_eq!(metrics["avg_clustering_coefficient"], 1.0, epsilon = 1e-10); // Perfect triangle
    }

    #[test]
    fn test_disconnected_graph() {
        // Two separate edges
        let edges = vec![(0, 1), (2, 3)];
        let metrics = calculate_graph_metrics(&edges, 4);
        
        assert_eq!(metrics.get("num_nodes"), Some(&4.0));
        assert_eq!(metrics.get("num_edges"), Some(&2.0));
        assert_relative_eq!(metrics["is_connected"], 0.0, epsilon = 1e-10); // Not connected
        assert_eq!(metrics.get("num_components"), Some(&2.0)); // 2 components
    }

    #[test]
    fn test_star_graph() {
        // Central node connected to 3 others
        let edges = vec![(0, 1), (0, 2), (0, 3)];
        let metrics = calculate_graph_metrics(&edges, 4);
        
        assert_eq!(metrics.get("num_nodes"), Some(&4.0));
        assert_eq!(metrics.get("num_edges"), Some(&3.0));
        assert_relative_eq!(metrics["avg_degree"], 1.5, epsilon = 1e-10); // (3+1+1+1)/4
        // Star has 0 clustering (no triangles)
        assert_relative_eq!(metrics["avg_clustering_coefficient"], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_complete_graph_k4() {
        // Complete graph with 4 nodes
        let edges = vec![
            (0, 1), (0, 2), (0, 3),
            (1, 2), (1, 3),
            (2, 3),
        ];
        let metrics = calculate_graph_metrics(&edges, 4);
        
        assert_eq!(metrics.get("num_nodes"), Some(&4.0));
        assert_eq!(metrics.get("num_edges"), Some(&6.0));
        assert_relative_eq!(metrics["density"], 1.0, epsilon = 1e-10); // All possible edges
        assert_relative_eq!(metrics["avg_degree"], 3.0, epsilon = 1e-10); // Each node connected to 3 others
        assert_relative_eq!(metrics["avg_clustering_coefficient"], 1.0, epsilon = 1e-10); // Perfect clustering
    }

    #[test]
    fn test_path_graph() {
        // Linear path: 0-1-2-3
        let edges = vec![(0, 1), (1, 2), (2, 3)];
        let metrics = calculate_graph_metrics(&edges, 4);
        
        assert_eq!(metrics.get("num_nodes"), Some(&4.0));
        assert_eq!(metrics.get("num_edges"), Some(&3.0));
        assert_relative_eq!(metrics["is_connected"], 1.0, epsilon = 1e-10);
        assert_eq!(metrics.get("num_components"), Some(&1.0));
        // Path has 0 clustering
        assert_relative_eq!(metrics["avg_clustering_coefficient"], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_cycle_graph() {
        // Cycle: 0-1-2-3-0
        let edges = vec![(0, 1), (1, 2), (2, 3), (3, 0)];
        let metrics = calculate_graph_metrics(&edges, 4);
        
        assert_eq!(metrics.get("num_nodes"), Some(&4.0));
        assert_eq!(metrics.get("num_edges"), Some(&4.0));
        assert_relative_eq!(metrics["avg_degree"], 2.0, epsilon = 1e-10);
        assert_relative_eq!(metrics["is_connected"], 1.0, epsilon = 1e-10);
        // Cycle has 0 clustering (no triangles)
        assert_relative_eq!(metrics["avg_clustering_coefficient"], 0.0, epsilon = 1e-10);
    }
}
