// Copyright © 2025 Justin K. Lietz, Neuroca, Inc.
// SPDX-License-Identifier: BSD-3-Clause
//
// Licensed under the BSD 3-Clause License. See LICENSE in the repository root.

use petgraph::graph::DiGraph;
use pyo3::prelude::*;
use std::collections::HashMap;

/// Calculates PageRank scores for nodes in a directed graph.
///
/// PageRank is an algorithm that measures the importance of nodes in a graph.
/// The basic idea is that a node is important if it is linked to by other important nodes.
///
/// # Arguments
///
/// * `edges` - List of directed edges as tuples (from_node, to_node)
/// * `num_nodes` - Total number of nodes in the graph
/// * `alpha` - Damping factor (default: 0.85)
/// * `max_iterations` - Maximum number of iterations (default: 100)
/// * `tolerance` - Convergence tolerance (default: 1e-6)
///
/// # Returns
///
/// A HashMap mapping node indices to their PageRank scores
///
/// # Example
///
/// ```
/// use voidkit_rust::graph::calculate_pagerank;
///
/// // Simple directed graph: 0 -> 1 -> 2 -> 0 (cycle)
/// let edges = vec![(0, 1), (1, 2), (2, 0)];
/// let pagerank = calculate_pagerank(&edges, 3, 0.85, 100, 1e-6);
/// 
/// // All nodes should have equal PageRank in a symmetric cycle
/// assert!((pagerank[&0] - 0.333).abs() < 0.01);
/// ```
pub fn calculate_pagerank(
    edges: &[(usize, usize)],
    num_nodes: usize,
    alpha: f64,
    max_iterations: usize,
    tolerance: f64,
) -> HashMap<usize, f64> {
    if num_nodes == 0 {
        return HashMap::new();
    }
    
    // Build directed graph
    let mut graph = DiGraph::<(), ()>::with_capacity(num_nodes, edges.len());
    
    // Add nodes
    let mut node_indices = Vec::new();
    for _ in 0..num_nodes {
        node_indices.push(graph.add_node(()));
    }
    
    // Add edges
    for &(from, to) in edges {
        if from < num_nodes && to < num_nodes {
            graph.add_edge(node_indices[from], node_indices[to], ());
        }
    }
    
    // Initialize PageRank scores uniformly
    let mut pagerank = vec![1.0 / num_nodes as f64; num_nodes];
    let mut new_pagerank = vec![0.0; num_nodes];
    
    // Calculate out-degrees for each node
    let mut out_degrees = vec![0; num_nodes];
    for &(from, _) in edges {
        if from < num_nodes {
            out_degrees[from] += 1;
        }
    }
    
    // Power iteration
    for _ in 0..max_iterations {
        // Reset new scores
        new_pagerank.fill((1.0 - alpha) / num_nodes as f64);
        
        // Distribute PageRank along outgoing edges
        for &(from, to) in edges {
            if from < num_nodes && to < num_nodes && out_degrees[from] > 0 {
                new_pagerank[to] += alpha * pagerank[from] / out_degrees[from] as f64;
            }
        }
        
        // Handle dangling nodes (nodes with no outgoing edges)
        let dangling_sum: f64 = (0..num_nodes)
            .filter(|&i| out_degrees[i] == 0)
            .map(|i| pagerank[i])
            .sum();
        
        let dangling_contribution = alpha * dangling_sum / num_nodes as f64;
        for score in new_pagerank.iter_mut().take(num_nodes) {
            *score += dangling_contribution;
        }
        
        // Check convergence
        let diff: f64 = (0..num_nodes)
            .map(|i| (new_pagerank[i] - pagerank[i]).abs())
            .sum();
        
        pagerank.copy_from_slice(&new_pagerank);
        
        if diff < tolerance {
            break;
        }
    }
    
    // Convert to HashMap
    pagerank.into_iter().enumerate().collect()
}

/// Python wrapper for calculate_pagerank
#[pyfunction]
#[pyo3(name = "calculate_pagerank", signature = (edges, num_nodes, alpha=None, max_iterations=None, tolerance=None))]
pub fn calculate_pagerank_py(
    edges: Vec<(usize, usize)>,
    num_nodes: usize,
    alpha: Option<f64>,
    max_iterations: Option<usize>,
    tolerance: Option<f64>,
) -> PyResult<HashMap<usize, f64>> {
    let alpha = alpha.unwrap_or(0.85);
    let max_iterations = max_iterations.unwrap_or(100);
    let tolerance = tolerance.unwrap_or(1e-6);
    
    Ok(calculate_pagerank(&edges, num_nodes, alpha, max_iterations, tolerance))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_pagerank_single_node() {
        let edges = vec![];
        let pr = calculate_pagerank(&edges, 1, 0.85, 100, 1e-6);
        
        assert_eq!(pr.len(), 1);
        assert_relative_eq!(pr[&0], 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_pagerank_two_nodes_one_edge() {
        // 0 -> 1
        let edges = vec![(0, 1)];
        let pr = calculate_pagerank(&edges, 2, 0.85, 100, 1e-6);
        
        assert_eq!(pr.len(), 2);
        // Node 1 should have higher PageRank (receives link from 0)
        assert!(pr[&1] > pr[&0]);
    }

    #[test]
    fn test_pagerank_symmetric_cycle() {
        // 0 -> 1 -> 2 -> 0 (symmetric cycle)
        let edges = vec![(0, 1), (1, 2), (2, 0)];
        let pr = calculate_pagerank(&edges, 3, 0.85, 100, 1e-6);
        
        assert_eq!(pr.len(), 3);
        // All nodes should have equal PageRank
        assert_relative_eq!(pr[&0], pr[&1], epsilon = 1e-4);
        assert_relative_eq!(pr[&1], pr[&2], epsilon = 1e-4);
        assert_relative_eq!(pr[&0], 1.0 / 3.0, epsilon = 1e-2);
    }

    #[test]
    fn test_pagerank_star_graph() {
        // All nodes point to node 0
        let edges = vec![(1, 0), (2, 0), (3, 0)];
        let pr = calculate_pagerank(&edges, 4, 0.85, 100, 1e-6);
        
        assert_eq!(pr.len(), 4);
        // Node 0 should have the highest PageRank
        assert!(pr[&0] > pr[&1]);
        assert!(pr[&0] > pr[&2]);
        assert!(pr[&0] > pr[&3]);
    }

    #[test]
    fn test_pagerank_chain() {
        // 0 -> 1 -> 2 -> 3 (linear chain)
        let edges = vec![(0, 1), (1, 2), (2, 3)];
        let pr = calculate_pagerank(&edges, 4, 0.85, 100, 1e-6);
        
        assert_eq!(pr.len(), 4);
        // Later nodes should generally have higher PageRank
        assert!(pr[&3] > pr[&0]);
    }

    #[test]
    fn test_pagerank_complete_graph() {
        // Complete directed graph (all pairs connected)
        let edges = vec![
            (0, 1), (0, 2),
            (1, 0), (1, 2),
            (2, 0), (2, 1),
        ];
        let pr = calculate_pagerank(&edges, 3, 0.85, 100, 1e-6);
        
        assert_eq!(pr.len(), 3);
        // All nodes should have equal PageRank
        assert_relative_eq!(pr[&0], pr[&1], epsilon = 1e-6);
        assert_relative_eq!(pr[&1], pr[&2], epsilon = 1e-6);
        assert_relative_eq!(pr[&0], 1.0 / 3.0, epsilon = 1e-2);
    }

    #[test]
    fn test_pagerank_hub_and_spoke() {
        // Hub (0) connects to all, all connect back to hub
        let edges = vec![
            (0, 1), (0, 2), (0, 3),
            (1, 0), (2, 0), (3, 0),
        ];
        let pr = calculate_pagerank(&edges, 4, 0.85, 100, 1e-6);
        
        assert_eq!(pr.len(), 4);
        // Hub should have higher PageRank due to multiple incoming links
        assert!(pr[&0] > pr[&1]);
        assert!(pr[&0] > pr[&2]);
        assert!(pr[&0] > pr[&3]);
    }

    #[test]
    fn test_pagerank_dangling_node() {
        // Node 2 is dangling (no outgoing edges)
        // 0 -> 1, 1 -> 2
        let edges = vec![(0, 1), (1, 2)];
        let pr = calculate_pagerank(&edges, 3, 0.85, 100, 1e-6);
        
        assert_eq!(pr.len(), 3);
        // All scores should sum to approximately 1
        let total: f64 = pr.values().sum();
        assert_relative_eq!(total, 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_pagerank_empty_graph() {
        let edges = vec![];
        let pr = calculate_pagerank(&edges, 0, 0.85, 100, 1e-6);
        
        assert_eq!(pr.len(), 0);
    }

    #[test]
    fn test_pagerank_different_alpha() {
        // Test with different damping factors
        let edges = vec![(0, 1), (1, 2), (2, 0)];
        
        let pr_high = calculate_pagerank(&edges, 3, 0.95, 100, 1e-6);
        let pr_low = calculate_pagerank(&edges, 3, 0.50, 100, 1e-6);
        
        // Higher alpha should lead to more differentiation
        // (but in a symmetric cycle, they should still be equal)
        assert_relative_eq!(pr_high[&0], pr_high[&1], epsilon = 1e-4);
        assert_relative_eq!(pr_low[&0], pr_low[&1], epsilon = 1e-4);
    }

    #[test]
    fn test_pagerank_convergence() {
        // Test that it converges within reasonable iterations
        let edges = vec![(0, 1), (1, 2), (2, 0), (1, 0)];
        let pr = calculate_pagerank(&edges, 3, 0.85, 10, 1e-3); // Low iterations, higher tolerance
        
        assert_eq!(pr.len(), 3);
        // Should still produce reasonable results
        let total: f64 = pr.values().sum();
        assert_relative_eq!(total, 1.0, epsilon = 1e-2);
    }
}
