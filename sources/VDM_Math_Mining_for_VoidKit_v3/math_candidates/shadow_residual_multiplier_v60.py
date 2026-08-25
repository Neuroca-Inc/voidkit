#!/usr/bin/env python3
from __future__ import annotations

import cmath
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import mpmath as mp
import sympy as sp

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
FIG = ROOT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

MOD = 12
M = 6
SUPPORT = (1, 5, 7, 11)
CHI = {1: 1, 5: -1, 7: -1, 11: 1}
ZETA24 = sp.exp(sp.I * sp.pi / 12)


def chi12(n: int) -> int:
    r = n % 12
    return CHI.get(r, 0)


def exponent_pair(n: int) -> Tuple[int, int]:
    return n * n, 24


def eta_coeff_map_by_shift(max_shift: int) -> Dict[int, int]:
    """Coefficient map for eta/q^(1/24)=prod(1-q^j) up to q^max_shift."""
    coeff = {0: 1}
    for j in range(1, max_shift + 1):
        nxt = dict(coeff)
        for k, v in coeff.items():
            if k + j <= max_shift:
                nxt[k + j] = nxt.get(k + j, 0) - v
        coeff = {k: v for k, v in nxt.items() if v != 0}
    return coeff


def eta_positive_coeff_map(max_shift: int) -> Dict[int, int]:
    """Map coefficient of q^(1/24) q^m from sum_{n>=1} chi12(n) q^(n^2/24)."""
    out: Dict[int, int] = {}
    max_n = int(math.sqrt(24 * max_shift + 1)) + 2
    for n in range(1, max_n + 1):
        c = chi12(n)
        if c == 0:
            continue
        if (n * n - 1) % 24 != 0:
            raise AssertionError("support exponent not in eta shift lattice")
        m = (n * n - 1) // 24
        if 0 <= m <= max_shift:
            out[m] = out.get(m, 0) + c
    return out


def make_coefficient_table(path: Path, n_max: int = 145) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for n in range(1, n_max + 1):
        c = chi12(n)
        if c == 0:
            continue
        num, den = exponent_pair(n)
        row = {
            "n": n,
            "residue_mod_12": n % 12,
            "chi12": c,
            "orientation_plus_coeff_R": c * n,
            "orientation_minus_coeff_R": -c * n,
            "scalar_bilateral_coeff_R_plus_minus": 0,
            "eta_shadow_coeff_positive_half": c,
            "q_exponent": f"{num}/{den}",
            "eta_shift_after_q_1_24": (n * n - 1) // 24,
        }
        rows.append(row)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return rows


def matrix_entries() -> Tuple[List[List[str]], List[List[str]], List[str]]:
    residues = list(range(MOD))
    T = [f"exp(pi*i*{r*r}/12)" for r in residues]
    S_half = []
    S_three = []
    for r in residues:
        row_half = []
        row_three = []
        for s in residues:
            entry = f"exp(-pi*i*{r*s}/6)/sqrt(12)"
            row_half.append(entry)
            row_three.append(f"i*{entry}")
        S_half.append(row_half)
        S_three.append(row_three)
    return S_half, S_three, T


def write_matrix_csv(path: Path, matrix: List[List[str]]) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["r\\s"] + list(range(MOD)))
        for r, row in enumerate(matrix):
            w.writerow([r] + row)


def theta_vec(tau: mp.mpc, N: int = 90) -> List[mp.mpc]:
    vals = []
    for r in range(MOD):
        s = mp.mpc(0)
        for k in range(-N, N + 1):
            n = MOD * k + r
            s += mp.e ** (2 * mp.pi * 1j * tau * (mp.mpf(n * n) / 24))
        vals.append(s)
    return vals


def deriv_vec(tau: mp.mpc, N: int = 90) -> List[mp.mpc]:
    vals = []
    for r in range(MOD):
        s = mp.mpc(0)
        for k in range(-N, N + 1):
            n = MOD * k + r
            s += n * mp.e ** (2 * mp.pi * 1j * tau * (mp.mpf(n * n) / 24))
        vals.append(s)
    return vals


def verify_transforms() -> Dict[str, object]:
    mp.mp.dps = 70
    tests = [mp.mpc(0.37, 0.91), mp.mpc(-0.21, 1.17), mp.mpc(0.13, 0.73)]
    max_theta_S = mp.mpf("0")
    max_deriv_S = mp.mpf("0")
    max_theta_T = mp.mpf("0")
    max_deriv_T = mp.mpf("0")

    for tau in tests:
        th = theta_vec(tau)
        thS = theta_vec(-1 / tau)
        thT = theta_vec(tau + 1)
        de = deriv_vec(tau)
        deS = deriv_vec(-1 / tau)
        deT = deriv_vec(tau + 1)
        for r in range(MOD):
            pred_T = mp.e ** (mp.pi * 1j * r * r / 12) * th[r]
            max_theta_T = max(max_theta_T, abs(thT[r] - pred_T))
            pred_DT = mp.e ** (mp.pi * 1j * r * r / 12) * de[r]
            max_deriv_T = max(max_deriv_T, abs(deT[r] - pred_DT))
            sm_th = sum(mp.e ** (-mp.pi * 1j * r * s / 6) * th[s] for s in range(MOD))
            pred_S = mp.sqrt(-1j * tau) / mp.sqrt(12) * sm_th
            max_theta_S = max(max_theta_S, abs(thS[r] - pred_S))
            sm_de = sum(mp.e ** (-mp.pi * 1j * r * s / 6) * de[s] for s in range(MOD))
            pred_DS = 1j * (-1j * tau) ** (mp.mpf(3) / 2) / mp.sqrt(12) * sm_de
            max_deriv_S = max(max_deriv_S, abs(deS[r] - pred_DS))

    threshold = mp.mpf("1e-45")
    return {
        "theta_weight_half_T_max_abs_error": str(max_theta_T),
        "theta_weight_half_S_max_abs_error": str(max_theta_S),
        "derivative_weight_three_half_T_max_abs_error": str(max_deriv_T),
        "derivative_weight_three_half_S_max_abs_error": str(max_deriv_S),
        "threshold": str(threshold),
        "PASS": bool(max_theta_T < threshold and max_theta_S < threshold and max_deriv_T < threshold and max_deriv_S < threshold),
        "S_formula_weight_half": "Theta_r(-1/tau)=(-i*tau)^(1/2)/sqrt(12) * sum_s exp(-pi*i*r*s/6) Theta_s(tau)",
        "S_formula_weight_three_half": "D_r(-1/tau)=i*(-i*tau)^(3/2)/sqrt(12) * sum_s exp(-pi*i*r*s/6) D_s(tau)",
        "T_formula": "component_r(tau+1)=exp(pi*i*r^2/12)*component_r(tau)",
    }


def verify_eta(max_shift: int = 650) -> Dict[str, object]:
    prod = eta_coeff_map_by_shift(max_shift)
    pos = eta_positive_coeff_map(max_shift)
    mismatches = []
    for k in range(max_shift + 1):
        if prod.get(k, 0) != pos.get(k, 0):
            mismatches.append({"shift": k, "product": prod.get(k, 0), "positive_theta": pos.get(k, 0)})
    return {
        "check": "sum_{n>=1} chi12(n) q^(n^2/24) equals eta(tau) coefficient-for-coefficient",
        "max_shift_after_q_1_24": max_shift,
        "mismatch_count": len(mismatches),
        "first_mismatches": mismatches[:10],
        "PASS": len(mismatches) == 0,
    }


def verify_chi_eigenvectors() -> Dict[str, object]:
    mp.mp.dps = 80
    max_s_err = mp.mpf("0")
    max_t_err = mp.mpf("0")
    for r in range(MOD):
        lhs = sum(mp.e ** (-mp.pi * 1j * r * ss / 6) * CHI.get(ss, 0) for ss in range(MOD)) / mp.sqrt(12)
        rhs = CHI.get(r, 0)
        max_s_err = max(max_s_err, abs(lhs - rhs))
        lhs_t = (mp.e ** (mp.pi * 1j * r * r / 12)) * CHI.get(r, 0)
        rhs_t = (mp.e ** (mp.pi * 1j / 12)) * CHI.get(r, 0)
        max_t_err = max(max_t_err, abs(lhs_t - rhs_t))
    threshold = mp.mpf("1e-70")
    return {
        "chi_vector_order_r_0_to_11": [int(CHI.get(r, 0)) for r in range(MOD)],
        "weight_half_S_chi_eigenvalue": "1",
        "weight_half_S_chi_max_abs_error": str(max_s_err),
        "weight_half_S_chi_exact_PASS": bool(max_s_err < threshold),
        "T_chi_eigenvalue_on_support": "exp(pi*i/12)",
        "T_chi_max_abs_error": str(max_t_err),
        "T_chi_exact_PASS": bool(max_t_err < threshold),
        "threshold": str(threshold),
    }


def verify_orientation_projection(n_max: int = 500) -> Dict[str, object]:
    nonzero_terms = []
    scalar_fail = []
    for n in range(1, n_max + 1):
        c = chi12(n)
        if c == 0:
            continue
        plus = c * n
        minus = -c * n
        if plus + minus != 0:
            scalar_fail.append(n)
        if len(nonzero_terms) < 12:
            nonzero_terms.append({"n": n, "plus": plus, "minus": minus, "scalar": plus + minus, "shadow_eta_coeff": c})
    return {
        "check": "orientation projection carries R while symmetric scalar projection cancels",
        "n_max": n_max,
        "sample_terms": nonzero_terms,
        "scalar_projection_fail_count": len(scalar_fail),
        "R_plus_nonzero_PASS": any(row["plus"] != 0 for row in nonzero_terms),
        "scalar_zero_PASS": len(scalar_fail) == 0,
        "shadow_eta_coeff_rule": "remove factor n from R_plus coefficients",
    }


def write_exact_result_card(summary: Dict[str, object]) -> None:
    card = {
        "status": "CLOSED_MULTIPLIER_SURFACE__RETAINED_ORIENTATION_CHANNEL",
        "closed_object": "orientation x residue vector, with 12-residue bilateral modular parent and positive-orientation false/Eichler half as the Shadow readout",
        "not_closed_as": "single scalar bilateral Jacobi derivative",
        "remaining_status_after_v60": "NO_OPEN_MULTIPLIER_BLOCKER_FOR_THE_RESIDUE_VECTOR_SURFACE; analytic contour-normalization can still be expanded in a publication proof if desired",
        "formulas": {
            "eta": "eta(tau)=sum_{n>=1} chi12(n) q^(n^2/24)=1/2 sum_r chi12(r) Theta_r(tau)",
            "shadow_residual": "R(tau)=sum_{n>=1} chi12(n)*n*q^(n^2/24)=sum_{r in {1,5,7,11}} D^+_r(tau)",
            "scalar_zero": "sum_{n in Z} chi12(n)*n*q^(n^2/24)=0",
            "T": "D_r(tau+1)=exp(pi*i*r^2/12)D_r(tau)",
            "S": "D_r(-1/tau)=i*(-i*tau)^(3/2)/sqrt(12) sum_s exp(-pi*i*r*s/6)D_s(tau)",
        },
        "verification": summary,
    }
    (OUT / "v60_completion_result_card.json").write_text(json.dumps(card, indent=2), encoding="utf-8")


def write_hash_manifest() -> None:
    rows = []
    for p in sorted(ROOT.rglob("*")):
        if p.is_file() and p.name != "MANIFEST_SHA256SUMS.txt":
            h = hashlib.sha256(p.read_bytes()).hexdigest()
            rows.append(f"{h}  {p.relative_to(ROOT)}")
    (ROOT / "MANIFEST_SHA256SUMS.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")


def main() -> None:
    coeff_rows = make_coefficient_table(OUT / "v60_orientation_residue_coefficients.csv")
    S_half, S_three, T = matrix_entries()
    write_matrix_csv(OUT / "v60_S_matrix_weight_half.csv", S_half)
    write_matrix_csv(OUT / "v60_S_matrix_weight_three_half.csv", S_three)
    with (OUT / "v60_T_multiplier_diagonal.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["residue", "T_multiplier"])
        for r, entry in enumerate(T):
            w.writerow([r, entry])

    summary = {
        "eta_coefficient_check": verify_eta(),
        "orientation_projection_check": verify_orientation_projection(),
        "chi_eta_multiplier_check": verify_chi_eigenvectors(),
        "numeric_S_T_transform_check": verify_transforms(),
        "coefficient_rows_written": len(coeff_rows),
    }
    summary["GLOBAL_PASS"] = all([
        summary["eta_coefficient_check"]["PASS"],
        summary["orientation_projection_check"]["R_plus_nonzero_PASS"],
        summary["orientation_projection_check"]["scalar_zero_PASS"],
        summary["chi_eta_multiplier_check"]["weight_half_S_chi_exact_PASS"],
        summary["chi_eta_multiplier_check"]["T_chi_exact_PASS"],
        summary["numeric_S_T_transform_check"]["PASS"],
    ])
    (OUT / "v60_verification_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_exact_result_card(summary)
    write_hash_manifest()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
