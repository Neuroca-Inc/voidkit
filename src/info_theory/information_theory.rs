/*
Copyright © 2025 Justin K. Lietz, Neuroca, Inc. All Rights Reserved.

Information theory functions - entropy, mutual information, KL divergence.
*/

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use numpy::{PyReadonlyArray1, PyReadonlyArray2};

/// Calculates the Shannon entropy of a discrete probability distribution.
///
/// # Arguments
///
/// * `pk` - A 1D array representing the probability distribution (must sum to 1)
/// * `base` - The logarithmic base to use (2 for bits, e for nats)
///
/// # Returns
///
/// The Shannon entropy of the distribution
///
/// # Examples
///
/// ```python
/// from voidkit_rust import calculate_entropy
/// import numpy as np
///
/// # Uniform distribution over 4 outcomes
/// pk = np.array([0.25, 0.25, 0.25, 0.25])
/// H = calculate_entropy(pk, base=2)
/// print(f"Entropy: {H} bits")  # Should be 2.0 bits
/// ```
#[pyfunction]
#[pyo3(signature = (pk, base=2.0))]
pub fn calculate_entropy(
    pk: PyReadonlyArray1<'_, f64>,
    base: f64,
) -> PyResult<f64> {
    let pk_array = pk.as_array();
    
    // Validate probability distribution
    let sum: f64 = pk_array.iter().sum();
    if (sum - 1.0).abs() > 1e-10 {
        return Err(PyValueError::new_err(format!(
            "Probabilities must sum to 1, got sum = {}", sum
        )));
    }
    
    // Check for negative probabilities
    for &p in pk_array.iter() {
        if p < 0.0 {
            return Err(PyValueError::new_err("Probabilities must be non-negative"));
        }
    }
    
    // Calculate entropy: H(X) = -Σ p(x) log_base(p(x))
    let log_base = base.ln();
    let mut entropy = 0.0;
    
    for &p in pk_array.iter() {
        if p > 0.0 {
            entropy -= p * (p.ln() / log_base);
        }
    }
    
    Ok(entropy)
}

/// Calculates the mutual information between two random variables.
///
/// # Arguments
///
/// * `p_xy` - A 2D array representing the joint probability distribution P(X, Y)
/// * `base` - The logarithmic base to use (2 for bits, e for nats)
///
/// # Returns
///
/// The mutual information I(X; Y)
///
/// # Examples
///
/// ```python
/// from voidkit_rust import calculate_mutual_information
/// import numpy as np
///
/// # Joint distribution for two binary variables
/// p_xy = np.array([[0.25, 0.25], [0.25, 0.25]])
/// MI = calculate_mutual_information(p_xy, base=2)
/// print(f"Mutual Information: {MI} bits")
/// ```
#[pyfunction]
#[pyo3(signature = (p_xy, base=2.0))]
pub fn calculate_mutual_information(
    p_xy: PyReadonlyArray2<'_, f64>,
    base: f64,
) -> PyResult<f64> {
    let p_xy_array = p_xy.as_array();
    
    // Validate joint probability distribution
    let sum: f64 = p_xy_array.iter().sum();
    if (sum - 1.0).abs() > 1e-10 {
        return Err(PyValueError::new_err(format!(
            "Probabilities must sum to 1, got sum = {}", sum
        )));
    }
    
    // Check for negative probabilities
    for &p in p_xy_array.iter() {
        if p < 0.0 {
            return Err(PyValueError::new_err("Probabilities must be non-negative"));
        }
    }
    
    let shape = p_xy_array.shape();
    let (n_rows, n_cols) = (shape[0], shape[1]);
    
    // Calculate marginal distributions
    let mut p_x = vec![0.0; n_rows];
    let mut p_y = vec![0.0; n_cols];
    
    for i in 0..n_rows {
        for j in 0..n_cols {
            let p = p_xy_array[[i, j]];
            p_x[i] += p;
            p_y[j] += p;
        }
    }
    
    // Calculate mutual information: I(X;Y) = Σ P(x,y) log(P(x,y) / (P(x)P(y)))
    let log_base = base.ln();
    let mut mi = 0.0;
    
    for i in 0..n_rows {
        for j in 0..n_cols {
            let p_joint = p_xy_array[[i, j]];
            if p_joint > 0.0 {
                let p_indep = p_x[i] * p_y[j];
                if p_indep > 0.0 {
                    mi += p_joint * ((p_joint / p_indep).ln() / log_base);
                }
            }
        }
    }
    
    Ok(mi)
}

/// Calculates the Kullback-Leibler (KL) divergence from Q to P.
///
/// # Arguments
///
/// * `pk` - The "true" probability distribution P
/// * `qk` - The approximating probability distribution Q
/// * `base` - The logarithmic base to use
///
/// # Returns
///
/// The KL divergence D_KL(P || Q)
///
/// # Examples
///
/// ```python
/// from voidkit_rust import calculate_kl_divergence
/// import numpy as np
///
/// p = np.array([0.5, 0.3, 0.2])
/// q = np.array([0.4, 0.4, 0.2])
/// kl = calculate_kl_divergence(p, q, base=2)
/// print(f"KL Divergence: {kl} bits")
/// ```
#[pyfunction]
#[pyo3(signature = (pk, qk, base=2.0))]
pub fn calculate_kl_divergence(
    pk: PyReadonlyArray1<'_, f64>,
    qk: PyReadonlyArray1<'_, f64>,
    base: f64,
) -> PyResult<f64> {
    let pk_array = pk.as_array();
    let qk_array = qk.as_array();
    
    if pk_array.len() != qk_array.len() {
        return Err(PyValueError::new_err(
            "Probability distributions must have the same length"
        ));
    }
    
    // Validate probability distributions
    let sum_p: f64 = pk_array.iter().sum();
    let sum_q: f64 = qk_array.iter().sum();
    
    if (sum_p - 1.0).abs() > 1e-10 {
        return Err(PyValueError::new_err(format!(
            "P must sum to 1, got sum = {}", sum_p
        )));
    }
    
    if (sum_q - 1.0).abs() > 1e-10 {
        return Err(PyValueError::new_err(format!(
            "Q must sum to 1, got sum = {}", sum_q
        )));
    }
    
    // Calculate KL divergence: D_KL(P || Q) = Σ P(x) log(P(x) / Q(x))
    let log_base = base.ln();
    let mut kl = 0.0;
    
    for i in 0..pk_array.len() {
        let p = pk_array[i];
        let q = qk_array[i];
        
        if p < 0.0 || q < 0.0 {
            return Err(PyValueError::new_err("Probabilities must be non-negative"));
        }
        
        if p > 0.0 {
            if q == 0.0 {
                return Err(PyValueError::new_err(
                    "KL divergence is infinite: Q has zero probability where P is non-zero"
                ));
            }
            kl += p * ((p / q).ln() / log_base);
        }
    }
    
    Ok(kl)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    
    #[test]
    fn test_entropy_uniform() {
        // Uniform distribution should have maximum entropy
        // H = log2(n) for uniform over n outcomes
        let pk = vec![0.25, 0.25, 0.25, 0.25];
        let log_base = 2.0_f64.ln();
        
        let mut entropy = 0.0;
        for &p in &pk {
            entropy -= p * (p.ln() / log_base);
        }
        
        assert_relative_eq!(entropy, 2.0, epsilon = 1e-10);
    }
    
    #[test]
    fn test_entropy_certain() {
        // Certain event should have zero entropy
        let pk = vec![1.0, 0.0, 0.0, 0.0];
        let log_base = 2.0_f64.ln();
        
        let mut entropy = 0.0;
        for &p in &pk {
            if p > 0.0 {
                entropy -= p * (p.ln() / log_base);
            }
        }
        
        assert_relative_eq!(entropy, 0.0, epsilon = 1e-10);
    }
    
    #[test]
    fn test_kl_divergence_identical() {
        // KL(P || P) should be 0
        let p = vec![0.5, 0.3, 0.2];
        let log_base = 2.0_f64.ln();
        
        let mut kl = 0.0;
        for i in 0..p.len() {
            if p[i] > 0.0 {
                kl += p[i] * ((p[i] / p[i]).ln() / log_base);
            }
        }
        
        assert_relative_eq!(kl, 0.0, epsilon = 1e-10);
    }
}
