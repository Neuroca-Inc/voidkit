#!/usr/bin/env python3
"""SymPy check for the CF11 entropy-compatible exchange scalar."""
from __future__ import annotations

import json
import os
import sympy as sp


def run_check() -> dict[str, object]:
    Pi, tau, Theta = sp.symbols("Pi tau Theta", positive=True, finite=True)
    sigma = sp.simplify(Pi**2 / tau)
    Q_sigma = sp.simplify(Theta * sigma)

    Pi0 = sp.Rational(3, 5)
    tau0 = sp.Rational(7, 3)
    Theta0 = sp.Rational(1, 1)
    sigma0 = sp.simplify(sigma.subs({Pi: Pi0, tau: tau0}))
    Q0 = sp.simplify(Q_sigma.subs({Pi: Pi0, tau: tau0, Theta: Theta0}))

    return {
        "check_id": "CF11-SYMPY-6",
        "description": "entropy-compatible irreversible exchange scalar Q_sigma = Theta_eff Pi^2/tau",
        "sigma_formula": str(sigma),
        "Q_sigma_formula": str(Q_sigma),
        "representative_Pi": str(Pi0),
        "representative_tau": str(tau0),
        "representative_sigma_exact": str(sigma0),
        "representative_sigma_float": float(sigma0),
        "representative_Q_sigma_exact": str(Q0),
        "representative_Q_sigma_float": float(Q0),
        "representative_nonnegative": bool(Q0 >= 0),
    }


if __name__ == "__main__":
    payload = json.dumps(run_check(), indent=2, sort_keys=True) + "\n"
    os.write(1, payload.encode("utf-8"))
    os._exit(0)
