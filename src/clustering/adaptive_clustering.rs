// Copyright © 2025 Justin K. Lietz, Neuroca, Inc. All Rights Reserved.
//
// This research is protected under a dual-license to foster open academic
// research while ensuring commercial applications are aligned with the project's ethical principles.
// Commercial use requires written permission from Justin K. Lietz.
// See LICENSE file for full terms.

use ndarray::Array1;
use numpy::{PyArray1, PyArrayMethods};
use pyo3::prelude::*;

/// Calculates entropy from a probability distribution.
///
/// H(X) = -Σ p(x) × log₂(p(x))
fn calculate_entropy(probabilities: &[f64]) -> f64 {
    probabilities
        .iter()
        .filter(|&&p| p > 0.0)
        .map(|&p| -p * p.log2())
        .sum()
}

/// Calculates the adaptive clustering interval based on graph entropy from degree sequence.
///
/// This function computes: t_cluster = base_interval × e^(-α × graph_entropy)
/// where graph_entropy is calculated from the degree distribution of the graph.
///
/// # Arguments
///
/// * `degrees` - Array of node degrees in the graph
/// * `base_interval` - The base clustering interval (default: 100000.0)
/// * `alpha` - The scaling factor for the entropy (default: 0.05)
///
/// # Returns
///
/// The calculated adaptive clustering interval as a float
///
/// # Example
///
/// ```
/// use ndarray::array;
/// use voidkit_rust::clustering::calculate_adaptive_clustering_interval;
///
/// // A small graph with 5 nodes: degrees [2, 2, 3, 1, 2]
/// let degrees = array![2, 2, 3, 1, 2];
/// let interval = calculate_adaptive_clustering_interval(&degrees, 100000.0, 0.05);
/// assert!(interval > 0.0);
/// assert!(interval <= 100000.0);
/// ```
pub fn calculate_adaptive_clustering_interval(
    degrees: &Array1<usize>,
    base_interval: f64,
    alpha: f64,
) -> f64 {
    if degrees.is_empty() {
        return base_interval;
    }
    
    // Find the maximum degree to determine array size
    let max_degree = degrees.iter().max().copied().unwrap_or(0);
    
    // Count occurrences of each degree
    let mut degree_counts = vec![0usize; max_degree + 1];
    for &degree in degrees.iter() {
        degree_counts[degree] += 1;
    }
    
    // Convert to probability distribution
    let total_nodes = degrees.len() as f64;
    let degree_distribution: Vec<f64> = degree_counts
        .iter()
        .map(|&count| count as f64 / total_nodes)
        .collect();
    
    // Calculate entropy of degree distribution
    let graph_entropy = calculate_entropy(&degree_distribution);
    
    // Return adaptive interval
    base_interval * (-alpha * graph_entropy).exp()
}

/// Python wrapper for calculate_adaptive_clustering_interval
#[pyfunction]
#[pyo3(name = "calculate_adaptive_clustering_interval")]
pub fn calculate_adaptive_clustering_interval_py<'py>(
    _py: Python<'py>,
    degrees: &Bound<'py, PyArray1<usize>>,
    base_interval: f64,
    alpha: f64,
) -> PyResult<f64> {
    // Convert numpy array to ndarray
    let degrees_array = degrees.readonly().as_array().to_owned();
    
    // Call the Rust function
    Ok(calculate_adaptive_clustering_interval(&degrees_array, base_interval, alpha))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::array;

    #[test]
    fn test_calculate_entropy_uniform() {
        // Uniform distribution has maximum entropy
        let probs = vec![0.25, 0.25, 0.25, 0.25];
        let entropy = calculate_entropy(&probs);
        assert_relative_eq!(entropy, 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_calculate_entropy_deterministic() {
        // Deterministic distribution has zero entropy
        let probs = vec![1.0, 0.0, 0.0, 0.0];
        let entropy = calculate_entropy(&probs);
        assert_relative_eq!(entropy, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_calculate_entropy_binary() {
        // Binary distribution
        let probs = vec![0.5, 0.5];
        let entropy = calculate_entropy(&probs);
        assert_relative_eq!(entropy, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_adaptive_clustering_uniform_degrees() {
        // All nodes have the same degree -> low entropy -> high interval
        let degrees = array![2, 2, 2, 2, 2];
        let base = 100000.0;
        let alpha = 0.05;
        
        let interval = calculate_adaptive_clustering_interval(&degrees, base, alpha);
        
        // With zero entropy, interval should equal base_interval
        assert_relative_eq!(interval, base, epsilon = 1e-6);
    }

    #[test]
    fn test_adaptive_clustering_diverse_degrees() {
        // Diverse degrees -> high entropy -> low interval
        let degrees = array![1, 2, 3, 4, 5];
        let base = 100000.0;
        let alpha = 0.05;
        
        let interval = calculate_adaptive_clustering_interval(&degrees, base, alpha);
        
        // Should be less than base interval due to non-zero entropy
        assert!(interval < base);
        assert!(interval > 0.0);
    }

    #[test]
    fn test_adaptive_clustering_empty_graph() {
        // Empty graph should return base interval
        let degrees = array![];
        let base = 100000.0;
        let alpha = 0.05;
        
        let interval = calculate_adaptive_clustering_interval(&degrees, base, alpha);
        
        assert_relative_eq!(interval, base, epsilon = 1e-10);
    }

    #[test]
    fn test_adaptive_clustering_single_node() {
        // Single node with degree 0
        let degrees = array![0];
        let base = 100000.0;
        let alpha = 0.05;
        
        let interval = calculate_adaptive_clustering_interval(&degrees, base, alpha);
        
        // Single degree value -> zero entropy -> base interval
        assert_relative_eq!(interval, base, epsilon = 1e-6);
    }

    #[test]
    fn test_adaptive_clustering_star_graph() {
        // Star graph: one hub with high degree, others with degree 1
        // 1 node with degree 4, 4 nodes with degree 1
        let degrees = array![4, 1, 1, 1, 1];
        let base = 100000.0;
        let alpha = 0.05;
        
        let interval = calculate_adaptive_clustering_interval(&degrees, base, alpha);
        
        // Should be between 0 and base (moderate entropy)
        assert!(interval < base);
        assert!(interval > 0.0);
    }

    #[test]
    fn test_adaptive_clustering_complete_graph() {
        // Complete graph: all nodes have same degree (n-1)
        let degrees = array![4, 4, 4, 4, 4]; // 5 nodes, each connected to 4 others
        let base = 100000.0;
        let alpha = 0.05;
        
        let interval = calculate_adaptive_clustering_interval(&degrees, base, alpha);
        
        // Zero entropy -> base interval
        assert_relative_eq!(interval, base, epsilon = 1e-6);
    }

    #[test]
    fn test_adaptive_clustering_alpha_sensitivity() {
        // Test that alpha parameter affects output as expected
        let degrees = array![1, 2, 3, 2, 1];
        let base = 100000.0;
        
        let interval_small_alpha = calculate_adaptive_clustering_interval(&degrees, base, 0.01);
        let interval_large_alpha = calculate_adaptive_clustering_interval(&degrees, base, 0.1);
        
        // Larger alpha should result in smaller interval (exponential decay)
        assert!(interval_large_alpha < interval_small_alpha);
    }

    #[test]
    fn test_adaptive_clustering_base_interval_scaling() {
        // Test that base interval scales output proportionally
        let degrees = array![1, 2, 3, 2, 1];
        let alpha = 0.05;
        
        let interval1 = calculate_adaptive_clustering_interval(&degrees, 100000.0, alpha);
        let interval2 = calculate_adaptive_clustering_interval(&degrees, 200000.0, alpha);
        
        // Double base interval should double output
        assert_relative_eq!(interval2 / interval1, 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_adaptive_clustering_typical_network() {
        // Test with a realistic degree sequence
        let degrees = array![3, 2, 4, 2, 3, 5, 2, 3, 4, 2];
        let base = 100000.0;
        let alpha = 0.05;
        
        let interval = calculate_adaptive_clustering_interval(&degrees, base, alpha);
        
        // Should be a reasonable value
        assert!(interval > 0.0);
        assert!(interval <= base);
        
        // With moderate diversity, should be somewhat less than base
        assert!(interval < base * 0.99); // At least 1% reduction
    }
}
