/*
Copyright © 2025 Justin K. Lietz, Neuroca, Inc. All Rights Reserved.

This research is protected under a dual-license to foster open academic
research while ensuring commercial applications are aligned with the project's
ethical principles. Commercial use requires written permission from Justin K. Lietz.
See LICENSE file for full terms.
*/

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use pyo3::Bound;
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;

/// Performs random search optimization (a simpler alternative to Bayesian optimization).
///
/// This function performs random search over the parameter space to find the
/// minimum of an objective function. While not as sophisticated as Bayesian
/// optimization, it provides a baseline optimization approach.
///
/// # Arguments
/// * `objective_func` - A Python callable that takes parameters and returns a float
/// * `param_space` - List of parameter space definitions as dictionaries
/// * `n_calls` - Number of evaluations to perform (default: 50)
/// * `random_state` - Random seed for reproducibility (default: 0)
///
/// # Returns
/// A dictionary containing:
/// - 'best_params': Dictionary of the best parameters found
/// - 'best_value': The minimum objective function value found
/// - 'all_params': List of all parameter sets evaluated
/// - 'all_values': List of all objective function values
///
/// # Note
/// This is a simplified optimization approach. For full Bayesian optimization,
/// consider using dedicated libraries like scikit-optimize from Python.
#[pyfunction]
#[pyo3(signature = (objective_func, param_space, n_calls=50, random_state=0))]
pub fn random_search_optimization<'py>(
    py: Python<'py>,
    objective_func: PyObject,
    param_space: &Bound<'py, PyList>,
    n_calls: usize,
    random_state: u64,
) -> PyResult<Bound<'py, PyDict>> {
    let mut rng = StdRng::seed_from_u64(random_state);
    
    // Parse parameter space
    let mut param_specs = Vec::new();
    let mut param_names = Vec::new();
    
    for item in param_space.iter() {
        let dict = item.downcast::<PyDict>()?;
        let param_type: String = dict.get_item("type")?.unwrap().extract()?;
        let name: String = dict.get_item("name")?.unwrap().extract()?;
        
        param_names.push(name.clone());
        
        match param_type.as_str() {
            "real" => {
                let range: Vec<f64> = dict.get_item("range")?.unwrap().extract()?;
                param_specs.push(ParamSpec::Real { 
                    min: range[0], 
                    max: range[1] 
                });
            }
            "integer" => {
                let range: Vec<i64> = dict.get_item("range")?.unwrap().extract()?;
                param_specs.push(ParamSpec::Integer { 
                    min: range[0], 
                    max: range[1] 
                });
            }
            "categorical" => {
                let categories: Vec<String> = dict.get_item("range")?.unwrap().extract()?;
                param_specs.push(ParamSpec::Categorical { 
                    values: categories 
                });
            }
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    format!("Unsupported parameter type: {}", param_type)
                ));
            }
        }
    }
    
    let mut best_value = f64::INFINITY;
    let mut best_params = Vec::new();
    let all_params_list = PyList::new_bound(py, &[] as &[i32]);
    let all_values_list = PyList::new_bound(py, &[] as &[f64]);
    
    // Random search iterations
    for _ in 0..n_calls {
        let mut current_params = Vec::new();
        
        for spec in &param_specs {
            match spec {
                ParamSpec::Real { min, max } => {
                    let value = rng.gen_range(*min..*max);
                    current_params.push(ParamValue::Real(value));
                }
                ParamSpec::Integer { min, max } => {
                    let value = rng.gen_range(*min..=*max);
                    current_params.push(ParamValue::Integer(value));
                }
                ParamSpec::Categorical { values } => {
                    let idx = rng.gen_range(0..values.len());
                    current_params.push(ParamValue::Categorical(values[idx].clone()));
                }
            }
        }
        
        // Convert params to Python list for objective function
        let params_py = PyList::new_bound(py, &[] as &[i32]);
        for param in &current_params {
            match param {
                ParamValue::Real(v) => params_py.append(*v)?,
                ParamValue::Integer(v) => params_py.append(*v)?,
                ParamValue::Categorical(v) => params_py.append(v.as_str())?,
            }
        }
        
        // Evaluate objective function
        let value: f64 = objective_func.call1(py, (&params_py,))?.extract(py)?;
        
        // Store all evaluations
        all_params_list.append(&params_py)?;
        all_values_list.append(value)?;
        
        // Update best
        if value < best_value {
            best_value = value;
            best_params = current_params.clone();
        }
    }
    
    // Create result dictionary
    let result = PyDict::new_bound(py);
    
    // Convert best params to dictionary
    let best_params_dict = PyDict::new_bound(py);
    for (i, param) in best_params.iter().enumerate() {
        match param {
            ParamValue::Real(v) => best_params_dict.set_item(&param_names[i], *v)?,
            ParamValue::Integer(v) => best_params_dict.set_item(&param_names[i], *v)?,
            ParamValue::Categorical(v) => best_params_dict.set_item(&param_names[i], v.as_str())?,
        }
    }
    
    result.set_item("best_params", best_params_dict)?;
    result.set_item("best_value", best_value)?;
    result.set_item("all_params", all_params_list)?;
    result.set_item("all_values", all_values_list)?;
    
    Ok(result)
}

#[derive(Clone)]
enum ParamSpec {
    Real { min: f64, max: f64 },
    Integer { min: i64, max: i64 },
    Categorical { values: Vec<String> },
}

#[derive(Clone)]
enum ParamValue {
    Real(f64),
    Integer(i64),
    Categorical(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    // Note: Tests requiring Python/numpy integration are disabled pending
    // test infrastructure updates for PyO3 0.22 API
    
    /*
    #[test]
    fn test_random_search_optimization_basic() {
        pyo3::prepare_freethreaded_python();
        
        Python::with_gil(|py| {
            // Simple quadratic objective: (x - 2)^2
            let objective = PyModule::from_code(
                py,
                "def objective(params): return (params[0] - 2.0) ** 2",
                "",
                "",
            ).unwrap();
            let func = objective.getattr("objective").unwrap();
            
            // Parameter space: single real parameter from 0 to 5
            let param_space = PyList::empty(py);
            let param_dict = PyDict::new(py);
            param_dict.set_item("type", "real").unwrap();
            param_dict.set_item("name", "x").unwrap();
            param_dict.set_item("range", vec![0.0, 5.0]).unwrap();
            param_space.append(param_dict).unwrap();
            
            let result = random_search_optimization(
                py,
                func.into(),
                param_space,
                100,
                42,
            ).unwrap();
            
            let best_value: f64 = result.get_item("best_value").unwrap().unwrap().extract().unwrap();
            
            // Should find a value close to 0 (minimum at x=2)
            assert!(best_value < 0.5); // Reasonable tolerance for random search
        });
    }
    */
}
