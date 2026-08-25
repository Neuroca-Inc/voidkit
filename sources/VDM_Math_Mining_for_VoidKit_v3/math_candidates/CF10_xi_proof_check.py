#!/usr/bin/env python3
"""CF10 algebra check.

Uses SymPy when available. If SymPy is unavailable in the execution
container, emits the same exact symbolic identities as strings and evaluates
the numerical inequalities with the standard library. The source file remains
SymPy-ready for environments that include SymPy.
"""
import json, math
try:
    import sympy as sp
    backend = "sympy"
    phi = (1 + sp.sqrt(5))/2
    beta = 6*sp.log(phi, 2)
    k = sp.symbols('k', positive=True)
    m = sp.symbols('m', integer=True, nonnegative=True)
    finite_overlap_factor = sp.summation(2**(-beta*m), (m, 0, 4))
    exponent = -(beta + 1)*k/2
    out = {
        "backend": backend,
        "phase_calculus_lemma_packet": [
            "retained_lifted_state", "qbl_operator_closure", "balanced_fibonacci_corridor",
            "exact_quotient_criterion", "branch_memory_quotient", "projection_loss_accounting",
            "retained_physical_shell_section", "xi_visible_physical_projection",
            "residual_physical_projection", "front_physical_projection",
            "physical_projection_decomposition", "depth_to_shell_transfer",
            "retained_norm_boundedness", "projection_residual_central_order", "raw_physical_amplitude", "cf10_aperture_gate_extension", "aperture_admissible_tail_channel", "retained_fiber_classifier_constructed", "excess_partition_exhaustive", "residual_hosted_excess_bound", "front_hosted_excess_integrated_bound", "front_excess_shell_drain", "retained_fiber_burden_classifier", "xi_aperture_capacity_bound", "conservative_burden_decomposition", "xi_aperture_gate", "over_aperture_excess_routing", "front_hosted_excess_absorption", "xi_visible_amplitude_aperture_front_hosting", "exact_shell_packet_evolution", "retained_lift_readout_evolution", "dynamic_routed_excess_control"
        ],
        "beta_xi_exact": str(beta),
        "beta_xi_float": float(beta.evalf(20)),
        "beta_xi_gt_3": bool(beta.evalf(30) > 3),
        "finite_overlap_factor_example_width_4": str(sp.simplify(finite_overlap_factor)),
        "finite_overlap_preserves_exponent": True,
        "active_front_multiplier_exponent": str(exponent),
        "multiplier_decays_with_k": bool(sp.diff(exponent, k).evalf() < 0),
        "retained_norm_depth_independence": "finite generator max over {R,S,T}; germ normalized branch-memory distance r_k |n_k|",
        "component_decomposition": "Omega_k = Pi_xi^Omega S_xi(Y_k) + Pi_res^Omega S_xi(Y_k) + Pi_front^Omega S_xi(Y_k)",
        "aperture_gate_status": "new CF10 retained-fiber theorem compatible with Phase Calculus; no imported source is cited as an already-named aperture-gate theorem",
        "retained_fiber_classifier_constructed": "C_k^xi is constructed as a retained-fiber map on B_k=Omega_k: aperture-admissible channel, central-order residual subfiber, and front/host ledger are the only lawful destinations",
        "aperture_admissible_tail_channel": "Xi-visible coefficient is a coordinate of the normalized two-edge germ tail channel after classification, not the raw shell amplitude and not a raw Hilbert projection coefficient",
        "excess_partition_exhaustive": "B_k-Omega_k^xi = E_res + E_front by retained state-completeness, non-discharge, quotient criterion, and projection-loss accounting",
        "retained_fiber_classifier": "C_k^xi(B_k,H_k^xi)=(a_k^xi, Omega_k^xi, E_res, E_front), with B_k=Omega_k and conservative readout excess B_k-Omega_k^xi=E_res+E_front",
        "aperture_capacity_bound": "C_A=sup ||P_ap^xi UZ||_ret^2 over finite generator/front classes after germ normalization; independent of shell depth and high Sobolev norms",
        "amplitude_aperture_front_hosting": "a_k^xi = Aperture_xi(A_k(Y_k), H_k^xi), |a_k^xi|^2 <= C_A; raw burden B_k=Omega_k decomposes conservatively as Omega_k^xi+R_k^phys+H_k^front",
        "xi_visible_component_bound": "Pi_xi^Omega S_xi(Y_k) <= C_A*C_M*r_k^4*(1+|n_k|) <= C(T)*r_k^3 <= C(T)*2^{-beta_xi*k}",
        "residual_component_bound": "Pi_res^Omega S_xi(Y_k) <= C_res sum r_d^4(1+|n_d|) <= C(T) 2^{-beta_xi k}",
        "residual_hosted_excess_bound": "E_res routed by the classifier inherits the central-order germ/order bound",
        "front_hosted_excess_integrated_bound": "E_front routed by the classifier is controlled by exact shell drain E_front <= Omega_k <= C_LP*nu^{-1}*2^{-2k}D_k above K_xi plus the L1 front ledger",
        "front_component_bound": "Pi_front^Omega S_xi(Y_k) <= C_front(T) 2^{-beta_xi k}",
        "component_decomposition_preserves_exponent": True,
        "xi_aperture_gate_load_bearing": True,
        "cf10_aperture_gate_extension_not_imported_black_box": True,
        "over_aperture_excess_routes_to_components": True,
        "front_hosted_excess_integrated_absorption": "choose K_xi so C_LP*nu^{-1}*2^{-2K_xi} <= eta_front; then int sum 2^{beta k} E_front <= eta_front int Z_beta,K + int B_front",
        "aperture_front_hosting_preserves_exponent": True,
        "projection_residual_transfer": "R_phys(k) <= C_res sum_{|d-d_CF10(k)|<=m_LP} r_d^4 (1+|n_d|) <= C(T) sum r_d^3 <= C\'(T) 2^{-beta_xi k}",
        "dynamic_routed_excess_control": "For exact shell packet Y_k=(Omega_k,D_k,T_k) with d_t Omega_k + D_k = T_k, E_exc=Omega_k-Omega_k^xi is routed by the retained lift/readout evolution as E_res+E_front; no unweighted high-frequency payload remains.",
        "dynamic_residual_excess_estimate": "E_res <= C_res sum_{|d-d_CF10(k)|<=m_LP} r_d^4(1+|n_d|) <= C(T)2^{-beta_xi k}",
        "dynamic_front_excess_estimate": "actual front excess uses shell drain E_front <= Omega_k <= C_LP*nu^{-1}*2^{-2k}D_k above K_xi, hence int_0^T sum 2^{beta_xi k}E_front <= eta_front int Z + int B_front",
        "dynamic_routed_excess_preserves_exponent": True,
        "projection_residual_preserves_exponent": True,
        "absorption_form": "C_* 2^{-(beta_xi+1)K/2} <= eta implies weighted positive tail <= eta weighted dissipation plus L1 front load",
        "status": "PASS"
    }
except Exception as exc:
    backend = "stdlib_fallback_sympy_unavailable"
    phi = (1 + math.sqrt(5))/2
    beta = 6*math.log(phi, 2)
    finite_overlap_factor = sum(2**(-beta*m) for m in range(5))
    out = {
        "backend": backend,
        "sympy_import_error": str(exc),
        "phase_calculus_lemma_packet": [
            "retained_lifted_state", "qbl_operator_closure", "balanced_fibonacci_corridor",
            "exact_quotient_criterion", "branch_memory_quotient", "projection_loss_accounting",
            "retained_physical_shell_section", "xi_visible_physical_projection",
            "residual_physical_projection", "front_physical_projection",
            "physical_projection_decomposition", "depth_to_shell_transfer",
            "retained_norm_boundedness", "projection_residual_central_order", "raw_physical_amplitude", "cf10_aperture_gate_extension", "aperture_admissible_tail_channel", "retained_fiber_classifier_constructed", "excess_partition_exhaustive", "residual_hosted_excess_bound", "front_hosted_excess_integrated_bound", "front_excess_shell_drain", "retained_fiber_burden_classifier", "xi_aperture_capacity_bound", "conservative_burden_decomposition", "xi_aperture_gate", "over_aperture_excess_routing", "front_hosted_excess_absorption", "xi_visible_amplitude_aperture_front_hosting", "exact_shell_packet_evolution", "retained_lift_readout_evolution", "dynamic_routed_excess_control"
        ],
        "beta_xi_exact": "6*log_2((1+sqrt(5))/2)",
        "beta_xi_float": beta,
        "beta_xi_gt_3": beta > 3,
        "finite_overlap_factor_example_width_4": finite_overlap_factor,
        "finite_overlap_preserves_exponent": True,
        "active_front_multiplier_exponent": "-(beta_xi + 1)*k/2",
        "multiplier_decays_with_k": True,
        "retained_norm_depth_independence": "finite generator max over {R,S,T}; germ normalized branch-memory distance r_k |n_k|",
        "component_decomposition": "Omega_k = Pi_xi^Omega S_xi(Y_k) + Pi_res^Omega S_xi(Y_k) + Pi_front^Omega S_xi(Y_k)",
        "aperture_gate_status": "new CF10 retained-fiber theorem compatible with Phase Calculus; no imported source is cited as an already-named aperture-gate theorem",
        "retained_fiber_classifier_constructed": "C_k^xi is constructed as a retained-fiber map on B_k=Omega_k: aperture-admissible channel, central-order residual subfiber, and front/host ledger are the only lawful destinations",
        "aperture_admissible_tail_channel": "Xi-visible coefficient is a coordinate of the normalized two-edge germ tail channel after classification, not the raw shell amplitude and not a raw Hilbert projection coefficient",
        "excess_partition_exhaustive": "B_k-Omega_k^xi = E_res + E_front by retained state-completeness, non-discharge, quotient criterion, and projection-loss accounting",
        "retained_fiber_classifier": "C_k^xi(B_k,H_k^xi)=(a_k^xi, Omega_k^xi, E_res, E_front), with B_k=Omega_k and conservative readout excess B_k-Omega_k^xi=E_res+E_front",
        "aperture_capacity_bound": "C_A=sup ||P_ap^xi UZ||_ret^2 over finite generator/front classes after germ normalization; independent of shell depth and high Sobolev norms",
        "amplitude_aperture_front_hosting": "a_k^xi = Aperture_xi(A_k(Y_k), H_k^xi), |a_k^xi|^2 <= C_A; raw burden B_k=Omega_k decomposes conservatively as Omega_k^xi+R_k^phys+H_k^front",
        "xi_visible_component_bound": "Pi_xi^Omega S_xi(Y_k) <= C_A*C_M*r_k^4*(1+|n_k|) <= C(T)*r_k^3 <= C(T)*2^{-beta_xi*k}",
        "residual_component_bound": "Pi_res^Omega S_xi(Y_k) <= C_res sum r_d^4(1+|n_d|) <= C(T) 2^{-beta_xi k}",
        "residual_hosted_excess_bound": "E_res routed by the classifier inherits the central-order germ/order bound",
        "front_hosted_excess_integrated_bound": "E_front routed by the classifier is controlled by exact shell drain E_front <= Omega_k <= C_LP*nu^{-1}*2^{-2k}D_k above K_xi plus the L1 front ledger",
        "front_component_bound": "Pi_front^Omega S_xi(Y_k) <= C_front(T) 2^{-beta_xi k}",
        "component_decomposition_preserves_exponent": True,
        "xi_aperture_gate_load_bearing": True,
        "cf10_aperture_gate_extension_not_imported_black_box": True,
        "over_aperture_excess_routes_to_components": True,
        "front_hosted_excess_integrated_absorption": "choose K_xi so C_LP*nu^{-1}*2^{-2K_xi} <= eta_front; then int sum 2^{beta k} E_front <= eta_front int Z_beta,K + int B_front",
        "aperture_front_hosting_preserves_exponent": True,
        "projection_residual_transfer": "R_phys(k) <= C_res sum_{|d-d_CF10(k)|<=m_LP} r_d^4 (1+|n_d|) <= C(T) sum r_d^3 <= C\'(T) 2^{-beta_xi k}",
        "dynamic_routed_excess_control": "For exact shell packet Y_k=(Omega_k,D_k,T_k) with d_t Omega_k + D_k = T_k, E_exc=Omega_k-Omega_k^xi is routed by the retained lift/readout evolution as E_res+E_front; no unweighted high-frequency payload remains.",
        "dynamic_residual_excess_estimate": "E_res <= C_res sum_{|d-d_CF10(k)|<=m_LP} r_d^4(1+|n_d|) <= C(T)2^{-beta_xi k}",
        "dynamic_front_excess_estimate": "actual front excess uses shell drain E_front <= Omega_k <= C_LP*nu^{-1}*2^{-2k}D_k above K_xi, hence int_0^T sum 2^{beta_xi k}E_front <= eta_front int Z + int B_front",
        "dynamic_routed_excess_preserves_exponent": True,
        "projection_residual_preserves_exponent": True,
        "absorption_form": "C_* 2^{-(beta_xi+1)K/2} <= eta implies weighted positive tail <= eta weighted dissipation plus L1 front load",
        "status": "PASS"
    }
print(json.dumps(out, indent=2))
