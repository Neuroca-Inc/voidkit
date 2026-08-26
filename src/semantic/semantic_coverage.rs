// Copyright © 2025 Justin K. Lietz, Neuroca, Inc.
// SPDX-License-Identifier: BSD-3-Clause
//
// Licensed under the BSD 3-Clause License. See LICENSE in the repository root.

use ndarray::{Array1, Array2};
use numpy::{PyArray2, PyArrayMethods};
use pyo3::prelude::*;

/// Calculates the cosine similarity between two vectors.
///
/// cosine_similarity(u, v) = (u · v) / (||u|| × ||v||)
fn cosine_similarity(u: &Array1<f64>, v: &Array1<f64>) -> f64 {
    let dot_product = u.dot(v);
    let norm_u = u.dot(u).sqrt();
    let norm_v = v.dot(v).sqrt();
    
    if norm_u == 0.0 || norm_v == 0.0 {
        0.0
    } else {
        dot_product / (norm_u * norm_v)
    }
}

/// Calculates the semantic coverage of a set of primitives with respect to input embeddings.
///
/// Semantic coverage measures how well a set of primitive embeddings covers the semantic
/// space of input embeddings. For each input embedding, it finds the maximum cosine
/// similarity to any primitive, then returns the mean of these maximum similarities.
///
/// C_sem = mean(max_j(cosine_similarity(input_i, primitive_j)))
///
/// # Arguments
///
/// * `input_embeddings` - 2D array of input embeddings, shape (n_inputs, embedding_dim)
/// * `primitive_set` - 2D array of primitive embeddings, shape (n_primitives, embedding_dim)
///
/// # Returns
///
/// The semantic coverage as a float in the range [0, 1], where 1 indicates perfect coverage
///
/// # Example
///
/// ```
/// use ndarray::array;
/// use voidkit_rust::semantic::calculate_semantic_coverage;
///
/// let inputs = array![[1.0, 0.0], [0.0, 1.0]];
/// let primitives = array![[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]];
///
/// let coverage = calculate_semantic_coverage(&inputs, &primitives);
/// assert!((coverage - 1.0).abs() < 1e-6); // Perfect coverage
/// ```
pub fn calculate_semantic_coverage(
    input_embeddings: &Array2<f64>,
    primitive_set: &Array2<f64>,
) -> f64 {
    let n_inputs = input_embeddings.nrows();
    let embedding_dim = input_embeddings.ncols();
    
    // Validate dimensions
    assert_eq!(
        primitive_set.ncols(),
        embedding_dim,
        "Primitive set must have same embedding dimension as inputs"
    );
    
    if n_inputs == 0 {
        return 0.0;
    }
    
    let mut total_similarity = 0.0;
    
    // For each input embedding
    for i in 0..n_inputs {
        let input_vec = input_embeddings.row(i).to_owned();
        
        // Find maximum similarity to any primitive
        let mut max_similarity = f64::NEG_INFINITY;
        for j in 0..primitive_set.nrows() {
            let primitive_vec = primitive_set.row(j).to_owned();
            let similarity = cosine_similarity(&input_vec, &primitive_vec);
            max_similarity = max_similarity.max(similarity);
        }
        
        total_similarity += max_similarity;
    }
    
    total_similarity / n_inputs as f64
}

/// Python wrapper for calculate_semantic_coverage
#[pyfunction]
#[pyo3(name = "calculate_semantic_coverage")]
pub fn calculate_semantic_coverage_py<'py>(
    _py: Python<'py>,
    input_embeddings: &Bound<'py, PyArray2<f64>>,
    primitive_set: &Bound<'py, PyArray2<f64>>,
) -> PyResult<f64> {
    // Convert numpy arrays to ndarray
    let input_array = input_embeddings.readonly().as_array().to_owned();
    let primitive_array = primitive_set.readonly().as_array().to_owned();
    
    // Call the Rust function
    Ok(calculate_semantic_coverage(&input_array, &primitive_array))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::array;

    #[test]
    fn test_cosine_similarity_identical() {
        let u = array![1.0, 0.0, 0.0];
        let v = array![1.0, 0.0, 0.0];
        let sim = cosine_similarity(&u, &v);
        assert_relative_eq!(sim, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let u = array![1.0, 0.0, 0.0];
        let v = array![0.0, 1.0, 0.0];
        let sim = cosine_similarity(&u, &v);
        assert_relative_eq!(sim, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_cosine_similarity_opposite() {
        let u = array![1.0, 0.0];
        let v = array![-1.0, 0.0];
        let sim = cosine_similarity(&u, &v);
        assert_relative_eq!(sim, -1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_cosine_similarity_angle() {
        // 45 degree angle in 2D
        let u = array![1.0, 0.0];
        let v = array![1.0, 1.0];
        let sim = cosine_similarity(&u, &v);
        // cos(45°) = 1/√2 ≈ 0.7071
        assert_relative_eq!(sim, 0.7071067811865475, epsilon = 1e-10);
    }

    #[test]
    fn test_calculate_semantic_coverage_perfect() {
        // Perfect coverage: primitives exactly match inputs
        let inputs = array![[1.0, 0.0], [0.0, 1.0]];
        let primitives = array![[1.0, 0.0], [0.0, 1.0]];
        
        let coverage = calculate_semantic_coverage(&inputs, &primitives);
        assert_relative_eq!(coverage, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_calculate_semantic_coverage_partial() {
        // Partial coverage: one primitive matches, one doesn't
        let inputs = array![[1.0, 0.0], [0.0, 1.0]];
        let primitives = array![[1.0, 0.0]]; // Only matches first input
        
        let coverage = calculate_semantic_coverage(&inputs, &primitives);
        // (1.0 + 0.0) / 2 = 0.5
        assert_relative_eq!(coverage, 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_calculate_semantic_coverage_extra_primitives() {
        // More primitives than inputs
        let inputs = array![[1.0, 0.0], [0.0, 1.0]];
        let primitives = array![
            [1.0, 0.0],
            [0.0, 1.0],
            [0.7, 0.7]  // Extra primitive
        ];
        
        let coverage = calculate_semantic_coverage(&inputs, &primitives);
        // Each input finds perfect match
        assert_relative_eq!(coverage, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_calculate_semantic_coverage_approximate() {
        // Primitives don't exactly match but are close
        let inputs = array![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let primitives = array![[0.9, 0.1, 0.0], [0.1, 0.9, 0.0]];
        
        let coverage = calculate_semantic_coverage(&inputs, &primitives);
        // Should be high but not perfect
        assert!(coverage > 0.8);
        assert!(coverage < 1.0);
    }

    #[test]
    fn test_calculate_semantic_coverage_no_overlap() {
        // Primitives orthogonal to inputs
        let inputs = array![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let primitives = array![[0.0, 0.0, 1.0]];
        
        let coverage = calculate_semantic_coverage(&inputs, &primitives);
        // All similarities should be 0
        assert_relative_eq!(coverage, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_calculate_semantic_coverage_empty_inputs() {
        // Empty input set
        let inputs = Array2::<f64>::zeros((0, 3));
        let primitives = array![[1.0, 0.0, 0.0]];
        
        let coverage = calculate_semantic_coverage(&inputs, &primitives);
        assert_relative_eq!(coverage, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_calculate_semantic_coverage_normalized() {
        // Test with unnormalized vectors
        let inputs = array![[2.0, 0.0], [0.0, 3.0]];
        let primitives = array![[1.0, 0.0], [0.0, 1.0]];
        
        let coverage = calculate_semantic_coverage(&inputs, &primitives);
        // Cosine similarity normalizes, so should be perfect
        assert_relative_eq!(coverage, 1.0, epsilon = 1e-10);
    }

    #[test]
    #[should_panic(expected = "Primitive set must have same embedding dimension as inputs")]
    fn test_calculate_semantic_coverage_dimension_mismatch() {
        let inputs = array![[1.0, 0.0], [0.0, 1.0]];
        let primitives = array![[1.0, 0.0, 0.0]]; // Wrong dimension
        
        calculate_semantic_coverage(&inputs, &primitives);
    }
}
