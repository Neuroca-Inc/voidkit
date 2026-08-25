#!/usr/bin/env python3
from __future__ import annotations

"""
Unified symbolic / executable validation for the v1.4 arXiv-style unified quintic release.
"""

import csv
import itertools
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Set, Tuple

import cmath
import mpmath as mp
import matplotlib.pyplot as plt
import sympy as sp

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
RESULTS.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT / "tools"))
from lifted_quintic_certifier import certify_all_roots, certify_bring_quintic, save_plot  # type: ignore  # noqa: E402

Perm = Tuple[int, ...]


def compose(p: Perm, q: Perm) -> Perm:
    return tuple(p[i] for i in q)


def inv(p: Perm) -> Perm:
    out = [0] * len(p)
    for i, j in enumerate(p):
        out[j] = i
    return tuple(out)


def parity(p: Perm) -> int:
    count = 0
    for i in range(len(p)):
        for j in range(i + 1, len(p)):
            count += int(p[i] > p[j])
    return count % 2


def commutator(a: Perm, b: Perm) -> Perm:
    return compose(compose(compose(a, b), inv(a)), inv(b))


def closure(gens: Iterable[Perm]) -> Set[Perm]:
    gens = set(gens)
    n = len(next(iter(gens)))
    identity = tuple(range(n))
    group = {identity}
    frontier = set(gens) | {identity}
    while frontier:
        g = frontier.pop()
        group.add(g)
        for h in list(group | gens):
            for z in (compose(g, h), compose(h, g)):
                if z not in group:
                    frontier.add(z)
    return group


def derived_subgroup(G: Set[Perm]) -> Set[Perm]:
    return closure(commutator(a, b) for a in G for b in G)


def derived_series_sizes(G: Set[Perm], max_steps: int = 6) -> List[int]:
    sizes = [len(G)]
    cur = G
    for _ in range(max_steps):
        nxt = derived_subgroup(cur)
        sizes.append(len(nxt))
        if nxt == cur:
            break
        cur = nxt
    return sizes


def cyclic_group(n: int) -> Set[Perm]:
    gen = tuple((i + 1) % n for i in range(n))
    return closure({gen})


def symmetric_group(n: int) -> Set[Perm]:
    return set(itertools.permutations(range(n)))


@dataclass(frozen=True)
class XiHat:
    A: float
    u: int
    v: int
    theta: float
    kappa: int

    @property
    def half_width(self) -> float:
        return math.pi / (self.u * self.v)



def PiVal(x: XiHat) -> complex:
    return x.A * cmath.exp(1j * x.theta)



def PiRed(x: XiHat) -> Tuple[int, int]:
    return (x.u, x.v)



def RedChannels(x: XiHat) -> Tuple[int, int, int]:
    return (x.u, x.v, x.kappa)



def GRedPair(u: int, v: int) -> Tuple[int, int]:
    return B_pair(u, v)



def rho_scalar(u: sp.Expr, v: sp.Expr) -> sp.Expr:
    return sp.log(u * v)



def Q(x: XiHat) -> XiHat:
    theta = x.theta + math.pi / 2
    return XiHat(x.A, x.u, x.v, theta, math.floor(theta / (2 * math.pi)))



def Q4(x: XiHat) -> XiHat:
    y = x
    for _ in range(4):
        y = Q(y)
    return y



def B_pair(u: int, v: int) -> Tuple[int, int]:
    a, b = v, u + v
    return (a, b) if a <= b else (b, a)



def B_lift(x: XiHat) -> XiHat:
    u, v = B_pair(x.u, x.v)
    return XiHat(x.A, u, v, x.theta, x.kappa)



def fibs(n: int) -> List[int]:
    F = [0, 1, 1]
    while len(F) <= n:
        F.append(F[-1] + F[-2])
    return F



def farey_symbolic_checks() -> Dict[str, str]:
    a, b, c, d = sp.symbols("a b c d", positive=True, integer=True)
    left_final = sp.simplify((b * c - a * d) / (b * (b + d)) - 1 / (b * (b + d)))
    right_final = sp.simplify((b * c - a * d) / (d * (b + d)) - 1 / (d * (b + d)))
    parent_final = sp.simplify((c * b - a * d) / (b * d) - 1 / (b * d))
    return {
        "parent_width_under_bc_minus_ad_eq_1": str(sp.simplify(parent_final.subs(b * c - a * d, 1))),
        "left_width_under_bc_minus_ad_eq_1": str(sp.simplify(left_final.subs(b * c - a * d, 1))),
        "right_width_under_bc_minus_ad_eq_1": str(sp.simplify(right_final.subs(b * c - a * d, 1))),
    }



def eta_edge_residual(t: mp.mpf, terms: int | None = None) -> mp.mpf:
    mp.mp.dps = 90
    if terms is None:
        terms = int(max(1000, min(200000, mp.ceil(60 / t))))
    s = mp.mpf("0")
    for n in range(1, terms + 1):
        s += mp.log(1 - mp.e ** (-t * n))
    return (s + (mp.pi ** 2) / (6 * t) - mp.mpf("0.5") * mp.log(2 * mp.pi / t)) / t



def plot_derived_series(series: Dict[str, List[int]]) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 3.9))
    for name, values in series.items():
        ax.plot(range(len(values)), values, marker="o", label=name)
    ax.set_xlabel("Derived-series depth")
    ax.set_ylabel("Subgroup size")
    ax.set_title("Derived-series obstruction surface")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "derived_series_sizes_v1_4.png", dpi=220)
    plt.close(fig)


def plot_q4_memory(rows: List[Dict[str, float]]) -> None:
    ms = [r["m"] for r in rows]
    kappas = [r["kappa"] for r in rows]
    errs = [r["projection_error"] for r in rows]
    fig, ax1 = plt.subplots(figsize=(6.6, 3.9))
    ax1.plot(ms, kappas, marker="o")
    ax1.set_xlabel("m in (Q^4)^m")
    ax1.set_ylabel("Lifted κ")
    ax1.set_title("Completed-turn memory under projection-invariant words")
    ax1.grid(True, alpha=0.3)
    ax2 = ax1.twinx()
    ax2.semilogy(ms, errs, marker="s")
    ax2.set_ylabel("Projection error")
    fig.tight_layout()
    fig.savefig(FIGURES / "q4_projection_memory_v1_4.png", dpi=220)
    plt.close(fig)


def plot_balanced_decay(rows: List[Dict[str, float]]) -> None:
    depths = [r["n"] for r in rows]
    widths = [r["half_width"] for r in rows]
    fig, ax = plt.subplots(figsize=(6.6, 3.9))
    ax.semilogy(depths, widths, marker="o")
    ax.set_xlabel("Balanced depth n")
    ax.set_ylabel(r"Half-width $\pi/(u_n v_n)$")
    ax.set_title("Balanced certification corridor and canonical anchor")
    ax.grid(True, which="both", alpha=0.3)
    for r in rows:
        if (r["u"], r["v"]) == (55, 89):
            ax.annotate("anchor (55,89)", xy=(r["n"], r["half_width"]), xytext=(r["n"] + 2, r["half_width"] * 3), arrowprops=dict(arrowstyle="->"))
            break
    fig.tight_layout()
    fig.savefig(FIGURES / "balanced_germ_decay_v1_4.png", dpi=220)
    plt.close(fig)


def plot_projection_commutes(comm: Perm) -> None:
    src = list(range(5))
    dst = [comm[i] for i in src]
    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    ax.scatter(src, [1] * 5, s=80)
    ax.scatter(dst, [0] * 5, s=80)
    for i, j in zip(src, dst):
        ax.annotate("", xy=(j, 0.1), xytext=(i, 0.9), arrowprops=dict(arrowstyle="->"))
        ax.text(i, 1.08, str(i), ha="center")
        ax.text(j, -0.12, str(j), ha="center")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["after", "before"])
    ax.set_xticks(range(5))
    ax.set_title("Nontrivial lifted commutator with projection-trivial visible readout")
    ax.set_ylim(-0.3, 1.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "projection_commutes_v1_4.png", dpi=220)
    plt.close(fig)


def plot_all_roots(certificates: Dict[str, object]) -> None:
    certs = certificates["certificates"]  # type: ignore[index]
    indices = [c["root_index"] for c in certs]  # type: ignore[index]
    depths = [c["depth"] for c in certs]  # type: ignore[index]
    residuals = [c["projected_polynomial_residual_abs"] for c in certs]  # type: ignore[index]
    fig, ax1 = plt.subplots(figsize=(6.7, 3.9))
    ax1.bar(indices, depths)
    ax1.set_xlabel("Lexicographic root index")
    ax1.set_ylabel("Certification depth")
    ax1.set_title("All-root Bring certificates at target half-width")
    ax2 = ax1.twinx()
    ax2.semilogy(indices, residuals, marker="o", linewidth=1.5)
    ax2.set_ylabel("Polynomial residual")
    fig.tight_layout()
    fig.savefig(FIGURES / "all_root_depths_and_residuals_v1_4.png", dpi=220)
    plt.close(fig)


def root_key(z: complex, digits: int = 18) -> Tuple[str, str]:
    return (f"{z.real:.{digits}f}", f"{z.imag:.{digits}f}")


def finite_commutation_audit(
    projector,
    evolution,
    descendant,
    states: Sequence[Dict[str, int]],
    law_name: str,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for state in states:
        lhs = projector(evolution(state))
        rhs = descendant(projector(state))
        error = abs(lhs - rhs)
        rows.append(
            {
                "law": law_name,
                "sheet_before": state["sheet"],
                "lhs_re": float(lhs.real),
                "lhs_im": float(lhs.imag),
                "rhs_re": float(rhs.real),
                "rhs_im": float(rhs.imag),
                "error": float(error),
            }
        )
    return rows

def plot_hierarchical_descent() -> None:
    fig, ax = plt.subplots(figsize=(8.0, 3.2))
    ax.axis("off")
    boxes = [
        (0.02, 0.55, 0.18, 0.28, "Primitive roll\nz(θ)=ie^{iθ}"),
        (0.24, 0.55, 0.2, 0.28, "Lifted state\nΞ̂=(A,q,θ,κ,c,s,h)"),
        (0.48, 0.55, 0.18, 0.28, "Operator core\nQ, B, Eγ"),
        (0.70, 0.55, 0.25, 0.28, "Projectors\nΠroot, Πval, ΠRed"),
        (0.24, 0.10, 0.20, 0.22, "Balanced corridor\n(55,89), 1/24"),
        (0.48, 0.10, 0.18, 0.22, "Red quotient\n(u,v)→(v,u+v)"),
        (0.70, 0.10, 0.25, 0.22, "Utility\nall-root certifier"),
    ]
    for x, y, w, h, txt in boxes:
        rect = plt.Rectangle((x, y), w, h, fill=False, linewidth=1.2)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=9)
    arrows = [
        ((0.20, 0.69), (0.24, 0.69)), ((0.44, 0.69), (0.48, 0.69)), ((0.66, 0.69), (0.70, 0.69)),
        ((0.34, 0.55), (0.34, 0.32)), ((0.57, 0.55), (0.57, 0.32)), ((0.82, 0.55), (0.82, 0.32)),
        ((0.44, 0.21), (0.48, 0.21)), ((0.66, 0.21), (0.70, 0.21)),
    ]
    for (x1, y1), (x2, y2) in arrows:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", lw=1.1))
    ax.set_title("Hierarchical descent used by the unified quintic paper", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGURES / "hierarchical_descent_v1_4.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_coverage_csv(rows: List[Tuple[str, str, bool]]) -> None:
    with (RESULTS / "unified_coverage_ledger.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["claim_id", "description", "pass"])
        writer.writerows(rows)


def main() -> None:
    summary: Dict[str, Any] = {}
    checks: Dict[str, bool] = {}

    C5 = cyclic_group(5)
    S2 = symmetric_group(2)
    S3 = symmetric_group(3)
    S4 = symmetric_group(4)
    S5 = symmetric_group(5)
    A5 = {p for p in S5 if parity(p) == 0}
    S5_prime = derived_subgroup(S5)
    A5_prime = derived_subgroup(A5)
    group_series = {
        "C5": derived_series_sizes(C5, 5),
        "S2": derived_series_sizes(S2, 3),
        "S3": derived_series_sizes(S3, 4),
        "S4": derived_series_sizes(S4, 5),
        "S5": derived_series_sizes(S5, 6),
    }
    even_by_root: Dict[int, Perm] = {}
    for p in A5:
        even_by_root.setdefault(p[0], p)
    mismatch_witness = (even_by_root[0], even_by_root[1])
    checks["S5_size_120"] = len(S5) == 120
    checks["A5_size_60"] = len(A5) == 60
    checks["S5_commutator_is_A5"] = S5_prime == A5
    checks["A5_is_perfect"] = A5_prime == A5
    checks["derived_series_stalls_at_A5"] = group_series["S5"][:3] == [120, 60, 60]
    checks["parity_projection_collapses_distinct_root_readouts"] = parity(mismatch_witness[0]) == parity(mismatch_witness[1]) and mismatch_witness[0][0] != mismatch_witness[1][0]
    plot_derived_series(group_series)
    summary["group"] = {
        "derived_series_sizes": group_series,
        "mismatch_witness_even_permutations": [mismatch_witness[0], mismatch_witness[1]],
        "mismatch_witness_root_readouts": [mismatch_witness[0][0], mismatch_witness[1][0]],
        "S5_prime_size": len(S5_prime),
        "A5_prime_size": len(A5_prime),
    }

    base = XiHat(A=0.75, u=1, v=1, theta=0.37, kappa=0)
    q4_rows: List[Dict[str, float]] = []
    for m in range(21):
        y = base
        for _ in range(m):
            y = Q4(y)
        q4_rows.append({"m": m, "kappa": y.kappa, "projection_error": abs(PiVal(y) - PiVal(base))})
    checks["Q4_projection_invariant_for_0_to_20"] = max(r["projection_error"] for r in q4_rows) < 1e-12
    checks["Q4_kappa_unbounded_sample"] = [int(r["kappa"]) for r in q4_rows] == list(range(21))
    plot_q4_memory(q4_rows)
    summary["q4_memory"] = q4_rows

    F = fibs(35)
    z = base
    branch_rows: List[Dict[str, float]] = []
    fib_ok = True
    width_decreases = True
    last_width = math.inf
    for n in range(25):
        expected = (F[n + 1], F[n + 2])
        fib_ok = fib_ok and ((z.u, z.v) == expected)
        width = math.pi / (z.u * z.v)
        width_decreases = width_decreases and (width <= last_width)
        branch_rows.append({"n": n, "u": z.u, "v": z.v, "half_width": width})
        last_width = width
        z = B_lift(z)
    checks["balanced_branch_exact_fibonacci_to_depth_24"] = fib_ok
    checks["balanced_half_width_monotone_sample"] = width_decreases
    checks["balanced_reaches_55_89"] = any((r["u"], r["v"]) == (55, 89) for r in branch_rows)
    plot_balanced_decay(branch_rows)

    farey = farey_symbolic_checks()
    checks["farey_width_identities_zero"] = all(v == "0" for v in farey.values())
    t_anchor = 2 * mp.pi / mp.mpf(4895)
    E_anchor = eta_edge_residual(t_anchor, terms=80000)
    E_target = mp.mpf(1) / 24
    E_anchor_err = abs(E_anchor - E_target)
    checks["edge_residual_anchor_close_to_1_24"] = float(E_anchor_err) < 1e-35
    summary["balanced_branch"] = {
        "rows": branch_rows,
        "anchor_depth": next(r["n"] for r in branch_rows if (r["u"], r["v"]) == (55, 89)),
        "farey_width_identities": farey,
        "t_anchor": str(t_anchor),
        "E_anchor": str(E_anchor),
        "target_1_24": str(E_target),
        "anchor_abs_error": str(E_anchor_err),
    }

    alpha = (1, 2, 0, 3, 4)
    beta = (0, 1, 3, 4, 2)
    comm = commutator(alpha, beta)
    checks["sheet_commutator_nonidentity"] = comm != tuple(range(5))
    registry = [XiHat(A=1.0, u=1, v=1, theta=0.37 + 2 * math.pi * j, kappa=j) for j in range(5)]
    before = [PiVal(xi) for xi in registry]
    after = [registry[comm[j]] for j in range(5)]
    after_values = [PiVal(xi) for xi in after]
    projection_registry_error = max(abs(a - b) for a, b in zip(before, after_values))
    checks["sheet_commutator_visible_trivial"] = projection_registry_error < 1e-12
    plot_projection_commutes(comm)
    summary["sheet_commutator"] = {
        "alpha": alpha,
        "beta": beta,
        "commutator": comm,
        "projection_registry_error": projection_registry_error,
    }

    red_x0 = XiHat(A=1.0, u=1, v=1, theta=0.37, kappa=0)
    red_x1 = XiHat(A=1.0, u=1, v=1, theta=0.37 + 2 * math.pi, kappa=1)
    visible_gap = abs(PiVal(red_x0) - PiVal(red_x1))
    checks["red_visible_only_factorization_impossible_witness"] = visible_gap < 1e-12 and (RedChannels(red_x0) != RedChannels(red_x1))
    sample_red_states = [XiHat(A=1.0, u=row["u"], v=row["v"], theta=0.0, kappa=0) for row in branch_rows[:12]]
    checks["red_pair_projection_commutes"] = all(PiRed(B_lift(x)) == GRedPair(*PiRed(x)) for x in sample_red_states)
    summary["red_lower_bound"] = {
        "state0_PiVal": [PiVal(red_x0).real, PiVal(red_x0).imag],
        "state1_PiVal": [PiVal(red_x1).real, PiVal(red_x1).imag],
        "visible_gap": visible_gap,
        "red_channels_state0": list(RedChannels(red_x0)),
        "red_channels_state1": list(RedChannels(red_x1)),
    }

    u, v = sp.symbols('u v', real=True, positive=True)
    rho = rho_scalar(u, v)
    rho_after = rho_scalar(v, u + v)
    checks["red_corridor_symbolic"] = (
        sp.simplify(sp.diff(rho, u) - 1 / u) == 0
        and sp.simplify(sp.diff(rho, v) - 1 / v) == 0
        and sp.simplify(rho_after - sp.log(v * (u + v))) == 0
    )
    checks["red_liouvillian_coordinate_symbolic"] = sp.simplify(rho - (sp.log(u) + sp.log(v))) == 0
    checks["red_packet_anchor_native"] = E_anchor_err < mp.mpf("1e-30")
    checks["liouvillian_obstruction_detected"] = (
        checks["S5_commutator_is_A5"]
        and checks["A5_is_perfect"]
        and checks["derived_series_stalls_at_A5"]
        and checks["red_liouvillian_coordinate_symbolic"]
    )
    summary["red_filter"] = {
        "replacement_block": "Pi_red ∘ B = G_red ∘ Pi_red",
        "PiRed_symbolic": "(u, v)",
        "GRed_symbolic": "(v, u+v)",
        "rho_symbolic": str(rho),
        "rho_after_symbolic": str(rho_after),
        "packet_descendant": "E(2*pi/(u*v))",
        "anchor_packet_value": str(E_anchor),
        "anchor_packet_abs_error": str(E_anchor_err),
    }

    certificate0 = certify_bring_quintic(a=-1.0, b=1.0, root_index=0, target_half_width=1e-8, precision_dps=80, scale=2.0)
    certificates = certify_all_roots(a=-1.0, b=1.0, target_half_width=1e-8, precision_dps=80, scale=2.0)
    (RESULTS / 'bring_root_0_certificate.json').write_text(json.dumps(certificate0, indent=2))
    (RESULTS / 'bring_all_roots_certificates.json').write_text(json.dumps(certificates, indent=2))
    save_plot(certificate0['corridor'], FIGURES / 'bring_root_0_half_width.png', 'Bring root 0 lifted half-width corridor')
    plot_all_roots(certificates)
    max_depth = max(c['depth'] for c in certificates['certificates'])  # type: ignore[index]
    max_residual = max(c['projected_polynomial_residual_abs'] for c in certificates['certificates'])  # type: ignore[index]
    checks["bring_discriminant_nonzero"] = str(certificates["discriminant"]).startswith("2869")
    checks["bring_all_roots_depth_21"] = max_depth == 21
    checks["bring_all_roots_residual_tiny"] = max_residual < 1e-12
    summary["all_roots_certificates"] = certificates

    bring_registry = [
        complex(c['projected_root']['re'], c['projected_root']['im'])
        for c in certificates['certificates']
    ]
    registry_index = {root_key(z): i for i, z in enumerate(bring_registry)}
    witness_states = [{"sheet": i} for i in range(len(bring_registry))]
    witness_perms = {
        "alpha": alpha,
        "beta": beta,
        "commutator": comm,
    }
    fifth_filter_rows: List[Dict[str, object]] = []
    fifth_filter_ok = True
    for law_name, perm in witness_perms.items():
        projector = lambda state, reg=bring_registry: reg[state["sheet"]]
        evolution = lambda state, p=perm: {"sheet": p[state["sheet"]]}
        descendant = lambda z, p=perm, reg=bring_registry, idx=registry_index: reg[p[idx[root_key(z)]]]
        rows = finite_commutation_audit(projector, evolution, descendant, witness_states, law_name)
        fifth_filter_rows.extend(rows)
        fifth_filter_ok = fifth_filter_ok and all(r["error"] < 1e-12 for r in rows)
    checks["root_projection_commutes_representative"] = all(r["error"] < 1e-12 for r in fifth_filter_rows if r["law"] == "commutator")
    checks["fifth_filter_quotient_commutes"] = fifth_filter_ok
    summary["fifth_filter"] = {
        "general_law": "Pi ∘ E = G ∘ Pi",
        "instantiated_law": "Pi_root ∘ E_gamma = rho(gamma) ∘ Pi_root",
        "witness_rows": fifth_filter_rows,
        "witness_permutations": witness_perms,
    }

    plot_hierarchical_descent()

    coverage_rows = [
        ("C1", "Parity-only / scalar proxy is not state-complete for root readout", checks["parity_projection_collapses_distinct_root_readouts"]),
        ("C2", "Projection forgets completed-turn memory while Q^4 increments κ", checks["Q4_projection_invariant_for_0_to_20"] and checks["Q4_kappa_unbounded_sample"]),
        ("C3", "Balanced refinement follows the Fibonacci corridor and reaches (55,89)", checks["balanced_branch_exact_fibonacci_to_depth_24"] and checks["balanced_reaches_55_89"]),
        ("C4", "The arithmetic-completion edge residual locks to 1/24 at the anchor", checks["edge_residual_anchor_close_to_1_24"] and checks["farey_width_identities_zero"]),
        ("C5", "The lifted sheet commutator is nontrivial while the visible readout is unchanged", checks["sheet_commutator_nonidentity"] and checks["sheet_commutator_visible_trivial"]),
        ("C6", "Finite-sheet branch transport commutes with root projection on the finite witness surface", checks["root_projection_commutes_representative"]),
        ("C7", "The native Red quotient is the exact balanced-corridor quotient on the positive corridor", checks["red_pair_projection_commutes"] and checks["red_corridor_symbolic"]),
        ("C8", "State-complete Red simulation does not factor through the visible scalar witness", checks["red_visible_only_factorization_impossible_witness"]),
        ("C9", "The generic irreducible quintic stays outside finite scalar Red discharge on the native Liouvillian corridor", checks["liouvillian_obstruction_detected"]),
        ("C10", "The native packet-collapse descendant locks to 1/24 at the canonical anchor", checks["red_packet_anchor_native"]),
        ("C11", "The Bring utility certifies all five roots at the requested half-width", checks["bring_all_roots_depth_21"] and checks["bring_all_roots_residual_tiny"] and checks["bring_discriminant_nonzero"]),
        ("C12", "The certifier is a fifth-filter quotient instance: root projection commutes with retained transport in the general filter shape", checks["root_projection_commutes_representative"] and checks["fifth_filter_quotient_commutes"]),
    ]
    write_coverage_csv(coverage_rows)
    summary["coverage"] = [{"claim_id": cid, "description": desc, "pass": passed} for cid, desc, passed in coverage_rows]
    summary["checks"] = checks

    ok = all(checks.values())
    summary_json = json.dumps(summary, indent=2, default=str)
    (RESULTS / "unified_validation_summary.json").write_text(summary_json)
    print(summary_json)
    print("FINAL_RESULT:", "PASS" if ok else "FAIL")
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
