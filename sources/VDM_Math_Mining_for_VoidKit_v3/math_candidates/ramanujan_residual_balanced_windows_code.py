from pathlib import Path
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Balanced Fibonacci corridor anchors
F = [1, 1]
while F[-1] < 100000:
    F.append(F[-1] + F[-2])

rows = []
for i in range(1, 15):
    u, v = F[i], F[i+1]
    uv = u * v
    t = 2 * math.pi / uv              # completion germ width
    q = math.exp(-t)

    # Numerical log(q;q)_∞ = sum_{m>=1} log(1 - q^m)
    # Truncate when q^M ~ e^-40
    M = int(max(2000, math.ceil(40 / t)))
    m = np.arange(1, M + 1, dtype=np.float64)
    log_qpoch = float(np.log1p(-np.exp(-t * m)).sum())

    # Strip off the blown-up bulk and log bulk
    # Residual coefficient along the balanced windows
    edge_coeff = (log_qpoch + math.pi**2 / (6 * t) - 0.5 * math.log(2 * math.pi / t)) / t
    full_germ_coeff = 2 * edge_coeff

    rows.append({
        "corridor_index": i,
        "u": u,
        "v": v,
        "uv": uv,
        "t_width": t,
        "q": q,
        "M_trunc": M,
        "log_qpoch": log_qpoch,
        "edge_coeff": edge_coeff,
        "full_germ_coeff": full_germ_coeff,
        "edge_target_1_over_24": 1/24,
        "full_target_1_over_12": 1/12,
        "edge_error": edge_coeff - 1/24,
        "full_error": full_germ_coeff - 1/12,
        "vacuum_edge_coeff_neg": -edge_coeff,
        "vacuum_full_coeff_neg": -full_germ_coeff,
    })

df = pd.DataFrame(rows)

outdir = Path("/mnt/data")
csv_path = outdir / "ramanujan_residual_balanced_windows.csv"
txt_path = outdir / "ramanujan_residual_balanced_windows.txt"
plot1 = outdir / "ramanujan_edge_coeff_convergence.png"
plot2 = outdir / "ramanujan_full_germ_coeff_convergence.png"
plot3 = outdir / "ramanujan_residual_errors.png"

df.to_csv(csv_path, index=False)
txt_path.write_text(df.to_string(index=False))

plt.figure(figsize=(8,5))
plt.plot(df["corridor_index"], df["edge_coeff"], marker="o", label="numerical edge coefficient")
plt.axhline(1/24, linestyle="--", label="1/24")
plt.xlabel("balanced corridor index")
plt.ylabel("edge coefficient")
plt.title("Edge residual coefficient along balanced windows")
plt.legend()
plt.savefig(plot1, bbox_inches="tight")
plt.close()

plt.figure(figsize=(8,5))
plt.plot(df["corridor_index"], df["full_germ_coeff"], marker="o", label="numerical full-germ coefficient")
plt.axhline(1/12, linestyle="--", label="1/12")
plt.xlabel("balanced corridor index")
plt.ylabel("full-germ coefficient")
plt.title("Two-sided germ residual coefficient along balanced windows")
plt.legend()
plt.savefig(plot2, bbox_inches="tight")
plt.close()

plt.figure(figsize=(8,5))
plt.plot(df["corridor_index"], np.abs(df["edge_error"]), marker="o", label="|edge_coeff - 1/24|")
plt.plot(df["corridor_index"], np.abs(df["full_error"]), marker="s", label="|full_germ_coeff - 1/12|")
plt.yscale("log")
plt.xlabel("balanced corridor index")
plt.ylabel("absolute error")
plt.title("Residual coefficient errors")
plt.legend()
plt.savefig(plot3, bbox_inches="tight")
plt.close()

print(df.to_string(index=False))
print("\\nSaved:")
print(csv_path)
print(txt_path)
print(plot1)
print(plot2)
print(plot3)
