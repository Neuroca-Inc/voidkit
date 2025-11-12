/*
Copyright © 2025 Justin K. Lietz, Neuroca, Inc. All Rights Reserved.

This research is protected under a dual-license to foster open academic
research while ensuring commercial applications are aligned with the project's
ethical principles. Commercial use requires written permission from Justin K. Lietz.
See LICENSE file for full terms.
*/

use pyo3::prelude::*;

pub mod linear_solver;
pub mod numerical_integration;
pub mod ode_solver;
pub mod void_equations;

/// VoidKit Rust - High-performance math/physics library
#[pymodule]
fn voidkit_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Linear algebra
    m.add_function(wrap_pyfunction!(linear_solver::linear_system_solver, m)?)?;
    
    // Numerical integration
    m.add_function(wrap_pyfunction!(numerical_integration::numerical_integrate, m)?)?;
    
    // ODE solver
    m.add_function(wrap_pyfunction!(ode_solver::numerical_ode_solver, m)?)?;
    
    // Void dynamics equations
    m.add_function(wrap_pyfunction!(void_equations::delta_re_vgsp, m)?)?;
    m.add_function(wrap_pyfunction!(void_equations::delta_gdsp, m)?)?;
    m.add_function(wrap_pyfunction!(void_equations::vdm_step, m)?)?;
    
    Ok(())
}
