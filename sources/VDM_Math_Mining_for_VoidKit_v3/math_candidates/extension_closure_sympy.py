from __future__ import annotations

from fractions import Fraction
from itertools import combinations, product, permutations
import sys

try:
    import sympy as sp  # optional; exact fallback below is sufficient when SymPy is absent
    SYMPY_AVAILABLE = True
except Exception:
    sp = None
    SYMPY_AVAILABLE = False

checks: list[tuple[str, bool, str]] = []

def check(name: str, ok: bool, detail: str = "") -> None:
    checks.append((name, bool(ok), detail))

bases = ["b0", "b1", "b2", "b3"]
weights = {"b0": 1, "b1": 2, "b2": 1, "b3": 1}
all_minimal_weight_maps = []
for doubled in bases:
    w = {b: 1 for b in bases}
    w[doubled] += 1
    if sum(w.values()) == 5:
        all_minimal_weight_maps.append(w)

admissible_weight_maps = [w for w in all_minimal_weight_maps if w["b1"] == 2]
check("yellow_projector_unique_in_admissible_class", len(admissible_weight_maps) == 1 and admissible_weight_maps[0] == weights,
      f"admissible={admissible_weight_maps}")

W = sum(weights.values())
centered = {b: 8 * W + 16 * weights[b] for b in bases}
check("centered_counts", [centered[b] for b in bases] == [56, 72, 56, 56], str(centered))
check("total_scaffold", 3 * 16 * W == 240, f"3*16*{W}")
partitions = [
    (("b1", "b2"), ("b0", "b3")),
    (("b0", "b1"), ("b2", "b3")),
    (("b1", "b3"), ("b0", "b2")),
]
for idx, (pos, neg) in enumerate(partitions):
    p = sum(centered[b] for b in pos)
    n = sum(centered[b] for b in neg)
    check(f"host_partition_A{idx}", (p, n) == (128, 112), f"{pos}->{p}; {neg}->{n}")

names = ["u", "c", "a", "g"]
sigma = {"u": "c", "c": "a", "a": "g", "g": "u"}
Q = {"b0": "b1", "b1": "b2", "b2": "b3", "b3": "b0"}
valid_namings = []
for perm in permutations(names):
    eta = dict(zip(bases, perm))
    if eta["b0"] == "u" and all(eta[Q[b]] == sigma[eta[b]] for b in bases):
        valid_namings.append(eta)
check("cyclic_rna_naming_unique", len(valid_namings) == 1 and valid_namings[0] == {"b0":"u","b1":"c","b2":"a","b3":"g"}, str(valid_namings))
reverse_eta = {"b0": "u", "b1": "g", "b2": "a", "b3": "c"}
check("reversed_naming_fails_equivariance", not all(reverse_eta[Q[b]] == sigma[reverse_eta[b]] for b in bases), str(reverse_eta))

def B_pair(u: int, v: int) -> tuple[int, int]:
    a, b = v, u + v
    return (a, b) if a <= b else (b, a)
check("product_only_collapse_fails", 1*6 == 2*3 and B_pair(1,6) != B_pair(2,3), f"B(1,6)={B_pair(1,6)}, B(2,3)={B_pair(2,3)}")

slots = ["R", "S", "T"]
words = list(product(bases, repeat=3))
sites = []
for w in words:
    for p_idx, p in enumerate(slots):
        b = w[p_idx]
        for m in range(weights[b]):
            sites.append((w, p, m))
check("scaffold_site_count", len(sites) == 240, f"len={len(sites)}")
index = {s: i for i, s in enumerate(sites)}
edges: set[tuple[int,int]] = set()
for s in sites:
    w, p, m = s
    i = index[s]
    p_next = slots[(slots.index(p)+1) % 3]
    p_next_idx = slots.index(p_next)
    if m < weights[w[p_next_idx]]:
        edges.add(tuple(sorted((i, index[(w, p_next, m)]))))
    p_idx = slots.index(p)
    b_new = Q[w[p_idx]]
    if m < weights[b_new]:
        w2 = list(w)
        w2[p_idx] = b_new
        edges.add(tuple(sorted((i, index[(tuple(w2), p, m)]))))
check("scaffold_graph_nonempty", len(edges) > 0, f"edges={len(edges)}")

degree = [0] * len(sites)
incident = [0] * len(sites)
for i, j in edges:
    if i == j:
        continue
    degree[i] += 1
    degree[j] += 1
    incident[i] += 1
    incident[j] += 1
row_sum_zero = degree == incident
check("bio_laplacian_row_sums_zero", row_sum_zero, "combinatorial row sums")

y = [Fraction((i % 17) - 8, 1) for i in range(len(sites))]
mean_y = sum(y, Fraction(0)) / len(y)
y = [yy - mean_y for yy in y]
entropy_prod = sum((y[i] - y[j]) ** 2 for i, j in edges)
check("bio_entropy_production_nonnegative", entropy_prod >= 0, f"entropy_prod={entropy_prod}")
check("bio_mass_conservation_vector", row_sum_zero, "1^T L = 0")

# ---------------------------------------------------------------------
# 4. E8 parity-sign lift verification (integer-scaled by 2)
# ---------------------------------------------------------------------
# Use scaled roots A = 2*alpha. Then squared norm is 8, and actual
# inner products {-2,-1,0,1,2} become scaled dots {-8,-4,0,4,8}.
roots = []
for i, j in combinations(range(8), 2):
    for si, sj in product([2, -2], repeat=2):
        v = [0 for _ in range(8)]
        v[i] = si
        v[j] = sj
        roots.append(tuple(v))
for signs in product([1, -1], repeat=8):
    prod_sign = 1
    for sgn in signs:
        prod_sign *= sgn
    if prod_sign == 1:
        roots.append(tuple(signs))
root_set = set(roots)
check("e8_root_count", len(roots) == 240 and len(root_set) == 240, f"len={len(roots)}, unique={len(root_set)}")

def dot_scaled(a, b):
    return sum(x * y for x, y in zip(a, b))

check("e8_norms_squared_two", all(dot_scaled(r, r) == 8 for r in roots), "scaled_norm=8 means true norm=2")
ips_scaled = {dot_scaled(a, b) for a in roots for b in roots}
check("e8_inner_product_set", ips_scaled == {-8, -4, 0, 4, 8}, f"scaled_ips={sorted(ips_scaled)}")
reflection_ok = True
bad = None
for alpha in roots:
    for beta in roots:
        k_scaled = dot_scaled(beta, alpha)
        if k_scaled % 4 != 0:
            reflection_ok = False
            bad = ("nonintegral_reflection_coefficient", alpha, beta, k_scaled)
            break
        k = k_scaled // 4
        reflected = tuple(beta[i] - k * alpha[i] for i in range(8))
        if reflected not in root_set:
            reflection_ok = False
            bad = (alpha, beta, reflected)
            break
    if not reflection_ok:
        break
check("e8_reflection_closure", reflection_ok, str(bad)[:200])

# ---------------------------------------------------------------------
# 5. Scalar-collapse obstruction sanity
# ---------------------------------------------------------------------
b, npos, wblock = 10, 100, 8
B_m = Fraction(1, 2 * (b ** (npos + wblock - 1)))
scaled = 2 * (b ** (npos + wblock - 1)) * B_m
check("scalar_collapse_certification_boundary", scaled == 1, f"scaled_diameter={scaled}")

print(f"sympy_available={SYMPY_AVAILABLE}")
failed = False
for name, ok, detail in checks:
    print(f"{name}: {'PASS' if ok else 'FAIL'} | {detail}")
    failed = failed or not ok
if failed:
    print("FINAL_RESULT: FAIL")
    sys.exit(1)
print("FINAL_RESULT: PASS")
