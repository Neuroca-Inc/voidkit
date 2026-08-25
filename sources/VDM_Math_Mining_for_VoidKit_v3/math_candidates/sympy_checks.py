from __future__ import annotations

import json
from pathlib import Path

import sympy as sp


def main() -> None:
    pkg_dir = Path(__file__).resolve().parent
    results_dir = pkg_dir / 'results'
    results_dir.mkdir(exist_ok=True)

    phi, delta, lam = sp.symbols('phi delta lam', real=True)
    phase = phi + lam
    psi = sp.Matrix([
        sp.cos(delta),
        sp.exp(sp.I * phase) * sp.sin(delta),
    ])
    psi_dag = psi.conjugate().T
    Pperp = sp.eye(2) - psi * psi_dag

    dphi = sp.diff(psi, phi)
    ddelta = sp.diff(psi, delta)

    def q(a: sp.Matrix, b: sp.Matrix) -> sp.Expr:
        return sp.simplify((a.conjugate().T * Pperp * b)[0])

    Qpp = sp.simplify(sp.trigsimp(q(dphi, dphi)))
    Qpd = sp.simplify(sp.trigsimp(q(dphi, ddelta)))
    Qdp = sp.simplify(sp.trigsimp(q(ddelta, dphi)))
    Qdd = sp.simplify(sp.trigsimp(q(ddelta, ddelta)))

    Q_expected = {
        'Q_phiphi': sp.sin(delta) ** 2 * sp.cos(delta) ** 2,
        'Q_phidelta': -sp.I * sp.sin(2 * delta) / 2,
        'Q_deltaphi': sp.I * sp.sin(2 * delta) / 2,
        'Q_deltadelta': sp.Integer(1),
    }

    residuals = {
        'Q_phiphi': sp.simplify(sp.trigsimp(Qpp - Q_expected['Q_phiphi'])),
        'Q_phidelta': sp.simplify(sp.trigsimp(Qpd - Q_expected['Q_phidelta'])),
        'Q_deltaphi': sp.simplify(sp.trigsimp(Qdp - Q_expected['Q_deltaphi'])),
        'Q_deltadelta': sp.simplify(sp.trigsimp(Qdd - Q_expected['Q_deltadelta'])),
    }

    g11 = sp.simplify(sp.re(Qpp))
    g12 = sp.simplify(sp.re(Qpd))
    g21 = sp.simplify(sp.re(Qdp))
    g22 = sp.simplify(sp.re(Qdd))

    Om12 = sp.simplify(-2 * sp.im(Qpd))
    Om21 = sp.simplify(-2 * sp.im(Qdp))

    g_residuals = {
        'g11': sp.simplify(sp.trigsimp(g11 - sp.sin(delta) ** 2 * sp.cos(delta) ** 2)),
        'g12': sp.simplify(g12),
        'g21': sp.simplify(g21),
        'g22': sp.simplify(g22 - 1),
    }
    Om_residuals = {
        'Omega12': sp.simplify(sp.trigsimp(Om12 - sp.sin(2 * delta))),
        'Omega21': sp.simplify(sp.trigsimp(Om21 + sp.sin(2 * delta))),
    }

    # Check invariance under phi -> phi + pi/2.
    subs_shift = {phi: phi + sp.pi / 2}
    invariance_residuals = {
        'Q_phiphi_shift': sp.simplify(sp.trigsimp(Qpp.subs(subs_shift) - Qpp)),
        'Q_deltadelta_shift': sp.simplify(sp.trigsimp(Qdd.subs(subs_shift) - Qdd)),
        'g11_shift': sp.simplify(sp.trigsimp(g11.subs(subs_shift) - g11)),
        'Omega12_shift': sp.simplify(sp.trigsimp(Om12.subs(subs_shift) - Om12)),
    }

    # Biological counting checks.
    base_counts = {'u': 56, 'c': 72, 'a': 56, 'g': 56}
    family_map = {
        'A0_mk': {'positive': ('c', 'a'), 'negative': ('u', 'g')},
        'A1_yr': {'positive': ('c', 'u'), 'negative': ('a', 'g')},
        'A2_sw': {'positive': ('c', 'g'), 'negative': ('u', 'a')},
    }

    bio_checks = {}
    for name, fam in family_map.items():
        pos = tuple(base_counts[b] for b in fam['positive'])
        neg = tuple(base_counts[b] for b in fam['negative'])
        bio_checks[name] = {
            'positive_pair': pos,
            'negative_pair': neg,
            'positive_total': sum(pos),
            'negative_total': sum(neg),
            'total': sum(pos) + sum(neg),
        }

    payload = {
        'qgt_symbolic': {
            'Q_phiphi': str(Qpp),
            'Q_phidelta': str(Qpd),
            'Q_deltaphi': str(Qdp),
            'Q_deltadelta': str(Qdd),
            'g11': str(g11),
            'g12': str(g12),
            'g21': str(g21),
            'g22': str(g22),
            'Omega12': str(Om12),
            'Omega21': str(Om21),
        },
        'qgt_residuals': {k: str(v) for k, v in residuals.items()},
        'metric_residuals': {k: str(v) for k, v in g_residuals.items()},
        'curvature_residuals': {k: str(v) for k, v in Om_residuals.items()},
        'phase_shift_invariance_residuals': {k: str(v) for k, v in invariance_residuals.items()},
        'biological_checks': bio_checks,
    }

    out = results_dir / 'sympy_exactness.json'
    out.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print(out)


if __name__ == '__main__':
    main()
