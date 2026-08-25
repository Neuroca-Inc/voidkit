import math
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def balanced_step(u, v):
    a, b = sorted((int(u), int(v)))
    return b, a + b

def trace_to_anchor(u0, v0, floor_den=4096):
    rows = []
    u, v = int(u0), int(v0)
    step = 0
    while True:
        uv = u * v
        r = 1 / uv
        rows.append({
            "step": step,
            "u": u,
            "v": v,
            "uv": uv,
            "r": r,
            "germ_halfwidth_over_pi": r,
            "germ_width_over_pi": 2 * r,
            "germ_halfwidth_rad": math.pi * r,
            "germ_width_rad": 2 * math.pi * r,
            "hits_floor": uv >= floor_den,
        })
        if uv >= floor_den:
            break
        u, v = balanced_step(u, v)
        step += 1
    return pd.DataFrame(rows)

floor_den = 4096
trace_11 = trace_to_anchor(1, 1, floor_den)
trace_12 = trace_to_anchor(1, 2, floor_den)

summary = pd.DataFrame([
    {
        "lift": "(1,1)",
        "steps_to_floor": int(trace_11["step"].iloc[-1]),
        "anchor_u": int(trace_11["u"].iloc[-1]),
        "anchor_v": int(trace_11["v"].iloc[-1]),
        "anchor_uv": int(trace_11["uv"].iloc[-1]),
        "anchor_ratio": float(trace_11["v"].iloc[-1] / trace_11["u"].iloc[-1]),
        "anchor_r": float(trace_11["r"].iloc[-1]),
        "anchor_germ_width_over_pi": float(trace_11["germ_width_over_pi"].iloc[-1]),
        "anchor_germ_width_rad": float(trace_11["germ_width_rad"].iloc[-1]),
    },
    {
        "lift": "(1,2)",
        "steps_to_floor": int(trace_12["step"].iloc[-1]),
        "anchor_u": int(trace_12["u"].iloc[-1]),
        "anchor_v": int(trace_12["v"].iloc[-1]),
        "anchor_uv": int(trace_12["uv"].iloc[-1]),
        "anchor_ratio": float(trace_12["v"].iloc[-1] / trace_12["u"].iloc[-1]),
        "anchor_r": float(trace_12["r"].iloc[-1]),
        "anchor_germ_width_over_pi": float(trace_12["germ_width_over_pi"].iloc[-1]),
        "anchor_germ_width_rad": float(trace_12["germ_width_rad"].iloc[-1]),
    }
])

anchor_u = int(trace_11["u"].iloc[-1])
anchor_v = int(trace_11["v"].iloc[-1])
anchor_uv = int(trace_11["uv"].iloc[-1])

symbolic_text = f"""Overflow survivor -> balanced Farey anchor

Primitive survivor:
  1

Smallest lifts that were checked:
  (1,1) and (1,2)

Exact balanced route from (1,1):
""" + "\n".join(
    f"  step {int(r.step)}: ({int(r.u)},{int(r.v)})   r = 1/{int(r.uv)}"
    for _, r in trace_11.iterrows()
) + f"""

Exact balanced route from (1,2):
""" + "\n".join(
    f"  step {int(r.step)}: ({int(r.u)},{int(r.v)})   r = 1/{int(r.uv)}"
    for _, r in trace_12.iterrows()
) + f"""

First floor hit above uv >= {floor_den}:
  ({anchor_u},{anchor_v})
  uv = {anchor_uv}
  r = 1/{anchor_uv}

Completion germ at the anchor:
  c_n = [theta_n - pi/{anchor_uv}, theta_n + pi/{anchor_uv}]
  width(c_n) = 2*pi/{anchor_uv}
  width(c_n)/pi = 2/{anchor_uv}
"""

outdir = Path("/mnt/data")
summary.to_csv(outdir / "overflow_to_55_89_summary.csv", index=False)
trace_11.to_csv(outdir / "overflow_to_55_89_trace_11.csv", index=False)
trace_12.to_csv(outdir / "overflow_to_55_89_trace_12.csv", index=False)
(outdir / "overflow_to_55_89_symbolic_trace.txt").write_text(symbolic_text)

plt.figure(figsize=(8,5))
plt.plot(trace_11["step"], trace_11["germ_width_over_pi"], marker="o", label="lift (1,1)")
plt.plot(trace_12["step"], trace_12["germ_width_over_pi"], marker="o", label="lift (1,2)")
plt.yscale("log")
plt.xlabel("balanced refinement step")
plt.ylabel("germ width / pi")
plt.title("Completion germ width contraction to the 55/89 anchor")
plt.legend()
plt.savefig(outdir / "overflow_to_55_89_germ_width.png", bbox_inches="tight")
plt.close()

plt.figure(figsize=(8,5))
plt.plot(trace_11["step"], trace_11["v"] / trace_11["u"], marker="o", label="lift (1,1)")
plt.plot(trace_12["step"], trace_12["v"] / trace_12["u"], marker="o", label="lift (1,2)")
phi = (1 + 5**0.5) / 2
plt.axhline(phi, linestyle="--", label="phi")
plt.xlabel("balanced refinement step")
plt.ylabel("v/u")
plt.title("Ratio trajectory into the 55/89 corridor anchor")
plt.legend()
plt.savefig(outdir / "overflow_to_55_89_ratio_trajectory.png", bbox_inches="tight")
plt.close()

plt.figure(figsize=(7,6))
plt.plot(trace_11["u"], trace_11["v"], marker="o", label="lift (1,1)")
plt.plot(trace_12["u"], trace_12["v"], marker="s", label="lift (1,2)")
for _, r in trace_11.iterrows():
    plt.annotate(str(int(r.step)), (r.u, r.v), fontsize=8)
for _, r in trace_12.iterrows():
    plt.annotate(str(int(r.step)), (r.u, r.v), fontsize=8)
plt.xlabel("u")
plt.ylabel("v")
plt.title("Exact balanced path from the survivor lift to the anchor")
plt.legend()
plt.savefig(outdir / "overflow_to_55_89_pair_path.png", bbox_inches="tight")
plt.close()

print(symbolic_text)
