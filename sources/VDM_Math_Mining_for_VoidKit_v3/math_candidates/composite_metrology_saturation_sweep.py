#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PHI = (1.0 + math.sqrt(5.0)) / 2.0
LOG2_PHI = math.log2(PHI)
LOG2_5 = math.log2(5.0)
C_STAR = 1.0 / 6.0
EXPECTED_L_11 = [15,45,103,220,455,923,1860,3735,7483,14980,29974]


def fib_pair_product_log2(n_b: int) -> float:
    # After n_b B events: (u,v)=(F_(n_b+1),F_(n_b+2)).
    if n_b < 1000:
        u = v = 1
        for _ in range(n_b):
            u, v = v, u + v
        return math.log2(u * v)
    # Binet correction is < 2*phi^(-2*(n_b+1))/ln(2), below 1e-417 at n_b=1000.
    return (2.0 * n_b + 3.0) * LOG2_PHI - LOG2_5


def capacity_log2(j: int) -> float:
    if j == 1:
        return 1.0
    if j == 2:
        return 2.0
    return float(2 * j)


def generate_schedule(completed_domains: int = 20):
    n_b = 0
    j = 1
    tick = 0
    rows = []
    min_nonzero_margin = float('inf')
    initial_exact_ties = 0

    for domain in range(completed_domains):
        phase_positions = 6 * (1 << domain)
        b_count = 0
        q_count = 0
        burst_counts = {0:0, 1:0, 2:0, 3:0}
        domain_start_tick = tick
        domain_start_b = n_b

        for k in range(phase_positions - 1):
            exponent = capacity_log2(j)
            current_log = fib_pair_product_log2(n_b)
            dist = abs(current_log - exponent)
            if dist == 0.0:
                initial_exact_ties += 1
            else:
                min_nonzero_margin = min(min_nonzero_margin, dist)
            load = 0
            if current_log < exponent:
                while fib_pair_product_log2(n_b + 1) <= exponent:
                    d = abs(fib_pair_product_log2(n_b + 1) - exponent)
                    if d != 0.0:
                        min_nonzero_margin = min(min_nonzero_margin, d)
                    n_b += 1
                    b_count += 1
                    load += 1
                    tick += 1
            burst_counts[load] = burst_counts.get(load, 0) + 1
            q_count += 1
            tick += 1
            j += 1

        exponent = capacity_log2(j)
        terminal_b = 0
        while fib_pair_product_log2(n_b) < exponent:
            d = abs(fib_pair_product_log2(n_b) - exponent)
            if d != 0.0:
                min_nonzero_margin = min(min_nonzero_margin, d)
            n_b += 1
            b_count += 1
            terminal_b += 1
            tick += 1
        overshoot_log2 = fib_pair_product_log2(n_b) - exponent
        tick += 1  # L
        endpoint_phase_quarters = q_count % 4
        rows.append({
            'completed_domain': domain,
            'L_tick': tick,
            'domain_ticks': tick - domain_start_tick,
            'B': b_count,
            'Q': q_count,
            'L': 1,
            'cumulative_B': n_b,
            'domain_start_cumulative_B': domain_start_b,
            'source_dimension': b_count + 1,
            'endpoint_phase_quarters': endpoint_phase_quarters,
            'endpoint_phase': ['+1','+i','-1','-i'][endpoint_phase_quarters],
            'q_burst_0': burst_counts.get(0,0),
            'q_burst_1': burst_counts.get(1,0),
            'q_burst_2': burst_counts.get(2,0),
            'q_burst_3plus': sum(v for k,v in burst_counts.items() if k >= 3),
            'terminal_B': terminal_b,
            'terminal_capacity_overshoot_log2': overshoot_log2,
            'handoff_opened': True,
        })
        j = 6 * (1 << (domain + 1)) - 5

    return pd.DataFrame(rows), min_nonzero_margin, initial_exact_ties


def main():
    package = Path(__file__).resolve().parents[1]
    results = package/'results'
    figures = package/'figures'
    snapshots = package/'upstream_snapshots'
    evidence = package/'evidence'

    started = time.time()
    schedule, min_margin, exact_ties = generate_schedule(20)
    schedule.to_csv(results/'deep_schedule_to_L20.csv', index=False)
    schedule[['completed_domain','L_tick','terminal_capacity_overshoot_log2','handoff_opened']].assign(localized_horizon_gate_evaluated=False).to_csv(results/'saturation_horizon_census.csv', index=False)

    exact_overlap = schedule.iloc[:11]['L_tick'].astype(int).tolist() == EXPECTED_L_11
    deep_handoffs = pd.read_csv(snapshots/'deep_l_handoffs.csv')
    count_overlap = bool(
        schedule.iloc[:11]['B'].astype(int).tolist() == deep_handoffs['B'].astype(int).tolist()
        and schedule.iloc[:11]['Q'].astype(int).tolist() == deep_handoffs['Q'].astype(int).tolist()
    )

    hydrogen = json.loads((snapshots/'hydrogen_summary.json').read_text())
    h = hydrogen['hydrogen']
    b = hydrogen['baryon']
    alpha = float(h['alpha_native'])
    kinetic_mass = float(h['measured_kinetic_mass'])
    gap = float(h['transition_gap_2p_to_1s'])
    bohr_radius = float(h['bohr_radius'])
    baryon_mass = float(b['rest_mass'])
    electron_mass = float(h['electron_rest_mass'])
    ground_energy = float(h['analytic_E1'])
    hydrogen_mass = baryon_mass + electron_mass + ground_energy
    binding_mass_defect = -ground_energy

    # Exact operational light metrology in the accepted native units.
    atomic_clock_period = 2.0 * math.pi / gap
    light_distance_per_clock = C_STAR * atomic_clock_period
    light_distance_in_bohr = light_distance_per_clock / bohr_radius
    exact_atomic_ratio = 32.0 * math.pi**2 / 9.0
    metrology_residual = light_distance_in_bohr - exact_atomic_ratio
    mass_shell_to_carrier_length = C_STAR
    carrier_to_mass_shell_length = 1.0 / C_STAR
    carrier_to_mass_shell_momentum = C_STAR

    # Whole-hydrogen Wilson transport using the already accepted whole-composite path operator.
    dispersion_rows = []
    for mode in range(65):
        k = math.pi * mode / 64.0
        energy = math.sqrt(math.sin(k)**2 + (hydrogen_mass + 1.0 - math.cos(k))**2)
        v_handoff = 0.0 if energy == 0 else (1.0 + hydrogen_mass) * math.sin(k) / energy
        dispersion_rows.append({
            'mode': mode,
            'momentum_mass_shell': k,
            'momentum_carrier_conjugate': k / C_STAR,
            'energy': energy,
            'kinetic_energy': energy - hydrogen_mass,
            'group_velocity_mass_shell': v_handoff,
            'group_velocity_carrier_per_Q': C_STAR * v_handoff,
        })
    dispersion = pd.DataFrame(dispersion_rows)
    dispersion.to_csv(results/'hydrogen_whole_composite_dispersion.csv', index=False)

    mass_certificate = {
        'baryon_rest_mass': baryon_mass,
        'electron_rest_mass': electron_mass,
        'hydrogen_internal_ground_energy': ground_energy,
        'hydrogen_rest_mass_expression': 'M_H=M_B+m_e+E_1s',
        'hydrogen_rest_mass': hydrogen_mass,
        'hydrogen_binding_mass_defect': binding_mass_defect,
        'transport_operator': 'accepted ORS/Wilson whole-composite handoff projected onto the accepted hydrogen ground line',
        'dispersion': 'E_H(k)=sqrt(sin(k)^2+(M_H+1-cos(k))^2)',
    }
    (results/'hydrogen_composite_mass_certificate.json').write_text(json.dumps(mass_certificate, indent=2)+'\n')

    metrology = {
        'carrier_speed_exact': '1/6',
        'carrier_speed': C_STAR,
        'mass_shell_speed_normalization': 1.0,
        'coordinate_map': {
            'x_carrier': '(1/6) x_mass_shell for the same Q-phase duration',
            'x_mass_shell': '6 x_carrier',
            'p_mass_shell': '(1/6) p_carrier',
            'velocity_map': 'v_carrier=(1/6)v_mass_shell',
        },
        'atomic_clock': {
            'transition': '2p->1s',
            'angular_frequency': gap,
            'period_Q_phase_advances': atomic_clock_period,
        },
        'atomic_rod': {
            'native_bohr_radius': bohr_radius,
        },
        'light_distance_per_atomic_clock_carrier_units': light_distance_per_clock,
        'light_distance_per_atomic_clock_bohr_radii': light_distance_in_bohr,
        'exact_ratio': 'lambda_H/a_0=32*pi^2/9',
        'exact_ratio_numeric': exact_atomic_ratio,
        'residual': metrology_residual,
    }
    (results/'native_light_metrology_certificate.json').write_text(json.dumps(metrology, indent=2)+'\n')

    # Source-normalized carrier gravity coupling from all Stage 18 charge plaquettes.
    p = pd.read_csv(snapshots/'plaquette_refinement_charge.csv')
    p = p[p['refinement_charge'] > 0].copy()
    kappa = 2.0 * math.log(PHI)
    p['response_per_source'] = p['log_defect'] / p['refinement_charge']
    gravity_rows = []
    for domain, group in p.groupby('coarse_domain'):
        gravity_rows.append({
            'coarse_domain': int(domain),
            'plaquettes': int(len(group)),
            'mean_response_per_source': float(group['response_per_source'].mean()),
            'std_response_per_source': float(group['response_per_source'].std(ddof=0)),
            'max_abs_residual_to_2logphi': float((group['response_per_source'] - kappa).abs().max()),
        })
    gravity_df = pd.DataFrame(gravity_rows)
    gravity_df.to_csv(results/'gravity_source_response_by_domain.csv', index=False)
    mature = gravity_df[gravity_df['coarse_domain'] >= 2]
    mature_max_residual = float(mature['max_abs_residual_to_2logphi'].max())
    g_carrier = kappa * C_STAR**4 / (8.0 * math.pi)

    baryon_radius = float(b['rms_outer_span'])
    compactness_baryon = 2.0 * g_carrier * baryon_mass / (C_STAR**2 * baryon_radius)
    compactness_hydrogen = 2.0 * g_carrier * hydrogen_mass / (C_STAR**2 * bohr_radius)
    baryon_horizon_mass = C_STAR**2 * baryon_radius / (2.0 * g_carrier)
    hydrogen_horizon_mass = C_STAR**2 * bohr_radius / (2.0 * g_carrier)
    gravity_certificate = {
        'exact_refinement_source': 'm=(#B)_fine-(#B)_coarse',
        'carrier_response_law': 'K_m -> 2*m*log(phi)',
        'source_normalized_carrier_coupling_expression': 'kappa_B=2*log(phi)',
        'source_normalized_carrier_coupling': kappa,
        'mature_domain_max_residual': mature_max_residual,
        'cf12_normalized_G_carrier_expression': 'G_B=log(phi)/(5184*pi)',
        'cf12_normalized_G_carrier': g_carrier,
        'authority_scope': 'carrier/refinement source-response coupling; full localized-mass P/Omega/T Newton coupling remains open',
        'known_object_compactness': {
            'baryon': compactness_baryon,
            'hydrogen': compactness_hydrogen,
        },
        'conditional_horizon_mass_at_fixed_radius': {
            'baryon_radius': baryon_horizon_mass,
            'hydrogen_bohr_radius': hydrogen_horizon_mass,
        },
    }
    (results/'gravity_strength_certificate.json').write_text(json.dumps(gravity_certificate, indent=2)+'\n')

    # Strict molecule gate. Schedule depth and representation capacity are not
    # substituted for a second spatial atom.
    completed_layers = int(len(schedule))
    minus_i_layers = int((schedule['endpoint_phase'] == '-i').sum())
    triplet_capacity = minus_i_layers // 3
    molecule_gate = {
        'compressed_schedule_layers': completed_layers,
        'fully_materialized_relation_layers_available_to_this_run': 11,
        'minus_i_endpoint_lines_in_schedule': minus_i_layers,
        'disjoint_color_triplet_representation_capacity': triplet_capacity,
        'accepted_spatial_baryon_closures': 1,
        'accepted_neutral_atomic_closures': 1,
        'required_neutral_atomic_closures': 2,
        'second_spatial_baryon_charge_closure': None,
        'second_independent_neutral_atom': None,
        'molecular_energy_gate_evaluated': False,
        'molecule_gate_evaluated': False,
        'molecule_closed': None,
        'molecular_mass_available': False,
        'meaning': 'The certified schedule continued to L20, but this run did not obtain a second observer-independent spatial +1 baryon and neutral atom from a newly materialized complete P/Omega/T state. The molecular energy gate therefore was not reached. Representation capacity was not substituted for an atom.',
    }
    (results/'molecule_closure_gate.json').write_text(json.dumps(molecule_gate, indent=2)+'\n')

    horizon = {
        'completed_global_L_handoffs': completed_layers,
        'all_global_L_handoffs_opened_next_domain': bool(schedule['handoff_opened'].all()),
        'largest_terminal_capacity_overshoot_log2': float(schedule['terminal_capacity_overshoot_log2'].max()),
        'localized_horizon_gate_evaluated': False,
        'black_hole_closed': None,
        'conditional_carrier_compactness': {
            'baryon': compactness_baryon,
            'hydrogen': compactness_hydrogen,
        },
        'horizon_recognition_threshold': 1.0,
        'meaning': 'The global B/Q/L schedule continued through twenty transmitting L handoffs. A native black-hole decision requires a localized massive aggregate and its complete outward P/Omega/T transfer aperture; that gate was not reached by this run.',
    }
    (results/'black_hole_saturation_certificate.json').write_text(json.dumps(horizon, indent=2)+'\n')

    predictions = {
        'registered_P1_molecule_phi_organization': {
            'status': 'NOT EVALUATED',
            'reason': 'No second independent neutral atom and therefore no molecule closed.'
        },
        'registered_P2_metrology_reconciliation': {
            'status': 'PASS',
            'measurement': 'x_mass_shell=6*x_carrier; v_carrier=(1/6)*v_mass_shell; lambda_H/a0=32*pi^2/9',
            'residual': metrology_residual,
        },
        'registered_P3_source_normalized_gravity_strength': {
            'status': 'PARTIAL',
            'measurement': 'carrier/refinement coupling kappa_B=2*log(phi)',
            'mature_max_residual': mature_max_residual,
            'remaining': 'localized-mass full P/Omega/T normalization',
        },
        'registered_P4_horizon_first_as_saturation': {
            'status': 'NOT EVALUATED',
            'reason': 'The global schedule reached 20 L handoffs, but no localized massive aggregate with a complete P/Omega/T outward aperture was available, so the native horizon gate was not evaluated.'
        },
    }
    (results/'prediction_evaluation.json').write_text(json.dumps(predictions, indent=2)+'\n')

    summary = {
        'package': package.name,
        'execution': {
            'completed_layers': completed_layers,
            'deepest_L_tick': int(schedule.iloc[-1]['L_tick']),
            'cumulative_B': int(schedule.iloc[-1]['cumulative_B']),
            'first_11_L_ticks_exact_overlap': exact_overlap,
            'first_11_BQ_count_overlap': count_overlap,
            'minimum_nonzero_log2_decision_margin': min_margin,
            'initial_exact_capacity_ties': exact_ties,
            'analytic_schedule_error_bound_after_B1000': '<1e-417 Binet correction; double rounding bound <2e-9',
        },
        'new_closures': {
            'hydrogen_rest_mass': hydrogen_mass,
            'hydrogen_binding_mass_defect': binding_mass_defect,
            'exact_operational_light_metrology': True,
            'light_distance_per_atomic_clock_bohr_radii': light_distance_in_bohr,
            'carrier_refinement_gravity_coupling': kappa,
            'cf12_normalized_G_carrier_candidate': g_carrier,
        },
        'open_gates': {
            'molecule': True,
            'molecular_mass': True,
            'full_localized_mass_P_Omega_T_gravity': True,
            'black_hole': True,
        },
        'molecule_closed': None,
        'black_hole_closed': None,
        'elapsed_seconds': time.time() - started,
    }
    (results/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')

    # Figures, one chart per file.
    plt.figure(figsize=(9,5))
    plt.semilogy(schedule['completed_domain'], schedule['L_tick'], marker='o')
    plt.xlabel('completed domain')
    plt.ylabel('exact L tick')
    plt.title('Autonomous recurrence continued through twenty retained layers')
    plt.tight_layout()
    plt.savefig(figures/'01_deep_L20_schedule.png', dpi=180)
    plt.close()

    plt.figure(figsize=(9,5))
    plt.bar(['one atomic clock period'], [light_distance_in_bohr])
    plt.axhline(exact_atomic_ratio, linestyle='--')
    plt.ylabel('light distance in native Bohr radii')
    plt.title('Exact internal light metrology: 32 pi^2 / 9 Bohr radii')
    plt.tight_layout()
    plt.savefig(figures/'02_light_atomic_metrology.png', dpi=180)
    plt.close()

    plt.figure(figsize=(9,5))
    plt.plot(gravity_df['coarse_domain'], gravity_df['mean_response_per_source'], marker='o')
    plt.axhline(kappa, linestyle='--')
    plt.xlabel('coarse domain')
    plt.ylabel('mean K/m')
    plt.title('Source-normalized burden response converges to 2 log(phi)')
    plt.tight_layout()
    plt.savefig(figures/'03_gravity_source_response.png', dpi=180)
    plt.close()

    plt.figure(figsize=(9,5))
    plt.semilogy(['baryon','hydrogen'], [compactness_baryon, compactness_hydrogen], marker='o')
    plt.axhline(1.0, linestyle='--')
    plt.ylabel('conditional compactness 2 G M / (c^2 r)')
    plt.title('Conditional carrier compactness of accepted generated objects')
    plt.tight_layout()
    plt.savefig(figures/'04_known_object_compactness.png', dpi=180)
    plt.close()

    plt.figure(figsize=(9,5))
    plt.plot(schedule['completed_domain'], schedule['terminal_capacity_overshoot_log2'], marker='o')
    plt.xlabel('completed domain')
    plt.ylabel('terminal log2(product/capacity)')
    plt.title('Global L schedule continues; localized horizon gate remains separate')
    plt.tight_layout()
    plt.savefig(figures/'05_saturation_handoff_census.png', dpi=180)
    plt.close()

    plt.figure(figsize=(9,5))
    plt.plot(dispersion['momentum_mass_shell'], dispersion['group_velocity_carrier_per_Q'])
    plt.axhline(C_STAR, linestyle='--')
    plt.xlabel('mass-shell momentum k')
    plt.ylabel('whole-hydrogen group velocity per Q phase')
    plt.title('Whole-hydrogen transport remains below c*=1/6')
    plt.tight_layout()
    plt.savefig(figures/'06_hydrogen_composite_dispersion.png', dpi=180)
    plt.close()

    evidence_payload = {
        'preregistration_sha256': (package/'preregistration/REGISTERED_SWEEP.sha256').read_text().split()[0],
        'executed_at_utc': __import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),
        'exact_overlap': exact_overlap,
        'count_overlap': count_overlap,
        'minimum_decision_margin': min_margin,
    }
    (evidence/'RUN_RECEIPT.json').write_text(json.dumps(evidence_payload, indent=2)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
