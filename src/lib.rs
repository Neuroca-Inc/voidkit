/*
Copyright © 2025 Justin K. Lietz, Neuroca, Inc. All Rights Reserved.

This research is protected under a dual-license to foster open academic
research while ensuring commercial applications are aligned with the project's
ethical principles. Commercial use requires written permission from Justin K. Lietz.
See LICENSE file for full terms.
*/

use pyo3::prelude::*;

pub mod numerical;
pub mod advanced_math;
pub mod void_dynamics;

/// VoidKit Rust - High-performance math/physics library
#[pymodule]
fn voidkit_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Numerical methods
    m.add_function(wrap_pyfunction!(numerical::linear_solver::linear_system_solver, m)?)?;
    m.add_function(wrap_pyfunction!(numerical::numerical_integration::numerical_integrate, m)?)?;
    m.add_function(wrap_pyfunction!(numerical::ode_solver::numerical_ode_solver, m)?)?;
    
    // Void dynamics equations
    m.add_function(wrap_pyfunction!(void_dynamics::void_equations::delta_re_vgsp, m)?)?;
    m.add_function(wrap_pyfunction!(void_dynamics::void_equations::delta_gdsp, m)?)?;
    m.add_function(wrap_pyfunction!(void_dynamics::void_equations::vdm_step, m)?)?;
    
    // Advanced math - statistics
    m.add_function(wrap_pyfunction!(advanced_math::descriptive_stats::descriptive_stats, m)?)?;
    
    Ok(())
}
