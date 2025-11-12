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
pub mod time_series;
pub mod sde;
pub mod evolutionary;
pub mod spatial;
pub mod neuro;

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
    
    // SIE formulas
    m.add_function(wrap_pyfunction!(void_dynamics::sie_formulas::calculate_td_error, m)?)?;
    m.add_function(wrap_pyfunction!(void_dynamics::sie_formulas::calculate_novelty_score, m)?)?;
    m.add_function(wrap_pyfunction!(void_dynamics::sie_formulas::calculate_habituation_score, m)?)?;
    m.add_function(wrap_pyfunction!(void_dynamics::sie_formulas::calculate_hsi, m)?)?;
    m.add_function(wrap_pyfunction!(void_dynamics::sie_formulas::calculate_total_reward, m)?)?;
    
    // RE-VGSP formulas
    m.add_function(wrap_pyfunction!(void_dynamics::revgsp_formulas::calculate_modulated_learning_rate, m)?)?;
    m.add_function(wrap_pyfunction!(void_dynamics::revgsp_formulas::calculate_modulated_trace_decay, m)?)?;
    m.add_function(wrap_pyfunction!(void_dynamics::revgsp_formulas::calculate_plasticity_impulse, m)?)?;
    m.add_function(wrap_pyfunction!(void_dynamics::revgsp_formulas::update_eligibility_trace, m)?)?;
    m.add_function(wrap_pyfunction!(void_dynamics::revgsp_formulas::calculate_weight_change, m)?)?;
    
    // Diagnostics formulas
    m.add_function(wrap_pyfunction!(void_dynamics::diagnostics_formulas::calculate_pathology_score, m)?)?;
    m.add_function(wrap_pyfunction!(void_dynamics::diagnostics_formulas::calculate_graph_entropy, m)?)?;
    m.add_function(wrap_pyfunction!(void_dynamics::diagnostics_formulas::calculate_cartography_time, m)?)?;
    
    // Advanced math - statistics
    m.add_function(wrap_pyfunction!(advanced_math::descriptive_stats::descriptive_stats, m)?)?;
    
    // Information theory
    m.add_function(wrap_pyfunction!(info_theory::information_theory::calculate_entropy, m)?)?;
    m.add_function(wrap_pyfunction!(info_theory::information_theory::calculate_mutual_information, m)?)?;
    m.add_function(wrap_pyfunction!(info_theory::information_theory::calculate_kl_divergence, m)?)?;
    m.add_function(wrap_pyfunction!(info_theory::information_theory::information_bottleneck, m)?)?;
    
    // Neural plasticity
    m.add_function(wrap_pyfunction!(neuro::advanced_sie::calculate_stabilized_reward, m)?)?;
    m.add_function(wrap_pyfunction!(neuro::advanced_sie::apply_quadratic_stdp_modulation, m)?)?;
    
    // Thermodynamics
    m.add_function(wrap_pyfunction!(thermodynamics::free_energy::calculate_free_energy, m)?)?;
    m.add_function(wrap_pyfunction!(thermodynamics::free_energy::minimize_free_energy_step, m)?)?;
    
    // Fractional calculus
    m.add_function(wrap_pyfunction!(fractional_calculus::caputo_derivative::caputo_derivative, m)?)?;
    
    // Dynamical systems
    m.add_function(wrap_pyfunction!(dynamical_systems::calculate_jacobian::calculate_jacobian, m)?)?;
    m.add_function(wrap_pyfunction!(dynamical_systems::analyze_stability::analyze_stability, m)?)?;
    m.add_function(wrap_pyfunction!(dynamical_systems::find_fixed_points::find_fixed_points, m)?)?;
    
    // SOC analysis
    m.add_function(wrap_pyfunction!(soc_analysis::fit_power_law::fit_power_law, m)?)?;
    m.add_function(wrap_pyfunction!(soc_analysis::detect_neuronal_avalanches::detect_neuronal_avalanches, m)?)?;
    
    // Optimal transport
    m.add_function(wrap_pyfunction!(ot::wasserstein_distance::calculate_wasserstein_distance, m)?)?;
    
    // Fractal analysis
    m.add_function(wrap_pyfunction!(fractal_analysis::calculate_fractal_dimension::calculate_fractal_dimension, m)?)?;
    m.add_function(wrap_pyfunction!(fractal_analysis::fractal_spike_train::generate_fractal_spike_train, m)?)?;
    
    // Stochastic simulation
    m.add_function(wrap_pyfunction!(stochastic::gillespie_simulation::gillespie_simulation, m)?)?;
    
    // Time series
    m.add_function(wrap_pyfunction!(time_series::time_series_analysis::calculate_autocorrelation, m)?)?;
    m.add_function(wrap_pyfunction!(time_series::time_series_analysis::calculate_cross_correlation, m)?)?;
    
    // SDE solver
    m.add_function(wrap_pyfunction!(sde::sde_solver::sde_solver, m)?)?;
    
    // Evolutionary algorithms
    m.add_function(wrap_pyfunction!(evolutionary::apply_mutation::apply_mutation, m)?)?;
    m.add_function(wrap_pyfunction!(evolutionary::apply_recombination::apply_recombination, m)?)?;
    
    // Spatial data structures
    m.add_class::<spatial::spatial_hash_grid::SpatialHashGrid>()?;
    
    Ok(())
}
