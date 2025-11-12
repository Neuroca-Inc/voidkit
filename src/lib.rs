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
pub mod info_theory;
pub mod thermodynamics;
pub mod fractional_calculus;
pub mod dynamical_systems;
pub mod soc_analysis;
pub mod ot;
pub mod fractal_analysis;
pub mod stochastic;

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
    
    // Information theory
    m.add_function(wrap_pyfunction!(info_theory::information_theory::calculate_entropy, m)?)?;
    m.add_function(wrap_pyfunction!(info_theory::information_theory::calculate_mutual_information, m)?)?;
    m.add_function(wrap_pyfunction!(info_theory::information_theory::calculate_kl_divergence, m)?)?;
    
    // Thermodynamics
    m.add_function(wrap_pyfunction!(thermodynamics::free_energy::calculate_free_energy, m)?)?;
    m.add_function(wrap_pyfunction!(thermodynamics::free_energy::minimize_free_energy_step, m)?)?;
    
    // Fractional calculus
    m.add_function(wrap_pyfunction!(fractional_calculus::caputo_derivative::caputo_derivative, m)?)?;
    
    // Dynamical systems
    m.add_function(wrap_pyfunction!(dynamical_systems::calculate_jacobian::calculate_jacobian, m)?)?;
    
    // SOC analysis
    m.add_function(wrap_pyfunction!(soc_analysis::fit_power_law::fit_power_law, m)?)?;
    
    // Optimal transport
    m.add_function(wrap_pyfunction!(ot::wasserstein_distance::calculate_wasserstein_distance, m)?)?;
    
    // Fractal analysis
    m.add_function(wrap_pyfunction!(fractal_analysis::calculate_fractal_dimension::calculate_fractal_dimension, m)?)?;
    
    // Stochastic simulation
    m.add_function(wrap_pyfunction!(stochastic::gillespie_simulation::gillespie_simulation, m)?)?;
    
    Ok(())
}
