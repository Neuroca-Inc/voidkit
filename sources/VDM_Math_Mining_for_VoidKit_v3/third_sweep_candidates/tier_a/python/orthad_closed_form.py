"""Orthad closed form.

One custody core, evaluated rather than executed. Supersedes the separate
carry, interface, and read-port modules, which each recomputed the same scaled
constants.

The accepted selector admits B while B < floor(lambda j + beta), with
lambda = log_phi 2 and beta = log_phi(sqrt 5) - 3/2 the Binet correction from
F_{n+1} F_{n+2} = (phi^(2n+3) - (-1)^(n+1) sqrt 5 + psi^(2n+3)) / 5. Every
retained quantity follows from that one floor, so none of them requires a tick.

    domain span        j in [6*2^A - 5, 12*2^A - 6]
    custody total      N(A) = floor(lambda j_A + beta) + 1 at the close
    point count        n_A  = N(A) - N(A-1) + 2
    orientation        theta_A = i^(6*2^A - 1); for A >= 1 always -i
    determination      |p_k| = F_{m+1} F_{m+2} / (F_{m+k+1} F_{m+k+2}),  m = N(A-1)
    crossing           exactly one per domain, admitted by the terminal rule
    interface          T_A = phi^(2 (1 - g_A)),  g_A = frac(lambda j_A + beta)
    carry              carry_A = floor(2 g_{A-1} + c) - 1,  c = 6 lambda - beta

The carry and the interface amplitude are the integer and fractional parts of
one affine-doubling orbit: 2 g_{A-1} + c = (carry_A + 1) + g_A.

Validated against direct execution of the accepted selector and against the
accepted billion-tick release record: B, Q, L, A, k, j, active points,
completed points, within edges, directed cross placements, and total relation
entries all reproduce exactly.

Offline analysis only. No engine dependency, no runtime role, no recurrence
authority. Nothing here may be fed back into execution.
"""

from __future__ import annotations

import argparse
import functools
from dataclasses import dataclass
from fractions import Fraction
from typing import Iterator

import mpmath as mp

_PHASE = ("1", "i", "-1", "-i")


# --------------------------------------------------------------------------
# arithmetic helpers


@functools.lru_cache(maxsize=None)
def _fib_pair(n: int) -> tuple[int, int]:
    """(F_n, F_{n+1}) by fast doubling."""
    if n == 0:
        return 0, 1
    a, b = _fib_pair(n >> 1)
    c = a * ((b << 1) - a)
    d = a * a + b * b
    return (d, c + d) if n & 1 else (c, d)


def fib(n: int) -> int:
    return _fib_pair(n)[0]


# --------------------------------------------------------------------------
# custody core


class Custody:
    """Scaled-integer custody totals. Every floor decision is exact."""

    def __init__(self, depth: int, extra_bits: int = 256) -> None:
        if depth < 0:
            raise ValueError("depth must be non-negative")
        self.depth = depth
        self.bits = depth + extra_bits
        mp.mp.prec = self.bits + 512
        phi = (1 + mp.sqrt(5)) / 2
        lam = mp.log(2) / mp.log(phi)
        bet = mp.log(5) / (2 * mp.log(phi)) - mp.mpf(3) / 2
        scale = mp.mpf(2) ** self.bits
        self._lam = int(mp.floor(lam * scale))
        self._bet = int(mp.floor(bet * scale))
        self._shift = int(mp.floor((6 * lam - bet) * scale))
        self._scale = 1 << self.bits
        self.alpha = float(mp.log(phi, 2))
        self.phi = float(phi)

    # -- domain geometry --------------------------------------------------

    @staticmethod
    def j_start(A: int) -> int:
        return (6 << A) - 5

    @staticmethod
    def j_terminal(A: int) -> int:
        return 12 * (1 << A) - 6

    @staticmethod
    def domain_of_j(j: int) -> int:
        """Domain index containing global position j."""
        if j < 1:
            raise ValueError("j must be positive")
        A = max(0, ((j + 5) // 6).bit_length() - 1)
        while (6 << A) - 5 > j:
            A -= 1
        while Custody.j_terminal(A) < j:
            A += 1
        return A

    # -- custody totals ---------------------------------------------------

    def b_at_j(self, j: int) -> int:
        """Retained B total once every B admitted at global position j has fired.

        The accepted capacity ladder sets Delta(1) = 2 and Delta(2) = 4 rather
        than 2^(2j), so the Beatty form only governs j >= 3. At j <= 2 the
        accepted threshold is 1, matching `threshold()` in the deep runner.
        """
        if j <= 2:
            return 1
        base = (self._lam * j + self._bet) // self._scale
        return base + 1 if self.is_terminal_j(j) else base

    @staticmethod
    def is_terminal_j(j: int) -> bool:
        """True when j is the terminal position of some domain, j = 12*2^A - 6.

        The accepted selector switches from the look-ahead test to the
        look-at-now test there, admitting exactly one refinement past the
        floor. That is the +1 in the threshold, and it is the crossing.
        """
        if (j + 6) % 12:
            return False
        q = (j + 6) // 12
        return q >= 1 and q & (q - 1) == 0

    def b_total(self, A: int) -> int:
        """Retained B total at the close of domain A. N(-1) = 0."""
        if A < 0:
            return 0
        return self.b_at_j(self.j_terminal(A))

    def orbit(self, A: int) -> Fraction:
        """g_A = frac(lambda j_A + beta)."""
        return Fraction((self._lam * self.j_terminal(A) + self._bet) % self._scale, self._scale)

    def refinements(self, A: int) -> int:
        return self.b_total(A) - self.b_total(A - 1)

    def point_count(self, A: int) -> int:
        return self.refinements(A) + 2

    @staticmethod
    def phase_index(A: int) -> int:
        return ((6 << A) - 1) % 4

    # -- tick inversion ---------------------------------------------------

    def state_at_tick(self, t: int) -> "CustodyState":
        """Aggregate custody state at tick t. No iteration over ticks.

        Uses t = B + (j - 1), which holds because j advances on exactly the
        non-B primitives, and B is monotone in j.
        """
        if t < 1:
            raise ValueError("tick must be positive")
        lo, hi = 1, t + 2
        while lo < hi:
            mid = (lo + hi) // 2
            if self.b_at_j(mid) + mid - 1 < t:
                lo = mid + 1
            else:
                hi = mid
        j = lo
        b = t - j + 1
        A = self.domain_of_j(j)
        k = j - self.j_start(A)
        completed = self.b_total(A - 1)
        return CustodyState(
            tick=t,
            j=j,
            b_total=b,
            domain=A,
            k=k,
            active_refinements=b - completed,
            active_points=b - completed + 2,
            completed_layers=A,
            completed_points=completed + 2 * A,
        )


@dataclass(frozen=True)
class CustodyState:
    tick: int
    j: int
    b_total: int
    domain: int
    k: int
    active_refinements: int
    active_points: int
    completed_layers: int
    completed_points: int

    @property
    def q_plus_l(self) -> int:
        return self.j - 1


# --------------------------------------------------------------------------
# interface records


@dataclass(frozen=True)
class Interface:
    domain: int
    refinements: int
    interior: int
    crossings: int
    carry: int | None
    orbit_g: float
    log2_transgression: float
    transgression: float


@dataclass(frozen=True)
class ExactPoint:
    layer: int
    ordinal: int
    refinement: int
    phase_index: int
    magnitude: Fraction | None
    log2_magnitude: float

    @property
    def exact(self) -> bool:
        return self.magnitude is not None

    def gaussian_parts(self) -> tuple[Fraction, Fraction]:
        if self.magnitude is None:
            raise ValueError("magnitude not materialized; use log2_magnitude")
        m = self.magnitude
        return ((m, Fraction(0)), (Fraction(0), m), (-m, Fraction(0)), (Fraction(0), -m))[
            self.phase_index
        ]

    def render(self) -> str:
        if self.magnitude is None:
            return f"{_PHASE[self.phase_index]} * 2^({self.log2_magnitude:.6f})"
        m = self.magnitude
        if m == 0:
            return "0"
        num = "" if m.numerator == 1 else str(m.numerator)
        core = {
            0: num or "1",
            1: f"{num}i",
            2: f"-{num or '1'}",
            3: f"-{num}i",
        }[self.phase_index]
        return core if m.denominator == 1 else f"{core}/{m.denominator}"


@dataclass(frozen=True)
class LayerSummary:
    layer: int
    point_count: int
    refinements: int
    phase_index: int
    orientation: str
    b_total_at_close: int
    j_terminal: int
    complete: bool


# --------------------------------------------------------------------------
# read port


class OrthadClosedForm:
    """Execution-free exact read surface over the lifted object.

    Answers the `cortex-port` query set for completed layers at any depth, and
    for the active layer when constructed from a tick.
    """

    EXACT_DEPTH_LIMIT = 30

    def __init__(self, depth: int, tick: int | None = None) -> None:
        self.custody = Custody(depth)
        self.depth = depth
        self.state = self.custody.state_at_tick(tick) if tick is not None else None
        if self.state is not None and self.state.domain > depth:
            raise ValueError(
                f"tick {tick} reaches domain {self.state.domain}; construct with depth >= that"
            )

    @classmethod
    def at_tick(cls, tick: int) -> "OrthadClosedForm":
        probe = Custody(1).state_at_tick(tick) if tick < 64 else None
        depth = probe.domain if probe else max(1, (tick.bit_length()))
        port = cls(depth, tick=tick)
        return port

    # -- structure --------------------------------------------------------

    def is_active(self, A: int) -> bool:
        return self.state is not None and A == self.state.domain

    def point_count(self, A: int) -> int:
        self._check(A)
        if self.is_active(A):
            return self.state.active_points
        return self.custody.point_count(A)

    def layer_summary(self, A: int) -> LayerSummary:
        self._check(A)
        active = self.is_active(A)
        if active:
            phase = self.state.k % 4
            refine = self.state.active_refinements
            close = self.state.b_total
        else:
            phase = self.custody.phase_index(A)
            refine = self.custody.refinements(A)
            close = self.custody.b_total(A)
        return LayerSummary(
            layer=A,
            point_count=self.point_count(A),
            refinements=refine,
            phase_index=phase,
            orientation=_PHASE[phase],
            b_total_at_close=close,
            j_terminal=self.custody.j_terminal(A),
            complete=not active,
        )

    def layers(self) -> Iterator[LayerSummary]:
        last = self.state.domain if self.state else self.depth
        for A in range(last + 1):
            yield self.layer_summary(A)

    # -- points -----------------------------------------------------------

    def primary_point(self, A: int, ordinal: int, exact: bool | None = None) -> ExactPoint:
        n = self.point_count(A)
        if not 0 <= ordinal < n:
            raise IndexError(f"ordinal {ordinal} outside layer {A} of size {n}")
        return self._point(A, ordinal, n - 1 - ordinal, exact)

    def point_by_refinement(self, A: int, k: int, exact: bool | None = None) -> ExactPoint:
        """Determination at refinement depth k. k = 0 is the retained seed.

        The useful index at large A, where the ordinal of a point near the seed
        is itself astronomically large.
        """
        n = self.point_count(A)
        if not 0 <= k <= n - 1:
            raise IndexError(f"refinement {k} outside layer {A} (max {n - 1})")
        return self._point(A, n - 1 - k, k, exact)

    def _point(self, A: int, ordinal: int, k: int, exact: bool | None) -> ExactPoint:
        phase = self.layer_summary(A).phase_index
        if ordinal == 0:
            return ExactPoint(A, 0, k, phase, Fraction(0), float("-inf"))
        m = self.custody.b_total(A - 1)
        want = (A <= self.EXACT_DEPTH_LIMIT) if exact is None else exact
        log2 = -2.0 * k * self.custody.alpha
        magnitude = None
        if want:
            num = fib(m + 1) * fib(m + 2)
            den = fib(m + k + 1) * fib(m + k + 2)
            magnitude = Fraction(num, den)
            log2 = float(mp.log(mp.mpf(num) / mp.mpf(den), 2))
        return ExactPoint(A, ordinal, k, phase, magnitude, log2)

    # -- charts and transfers ---------------------------------------------

    def plus_chart_image(self, A: int, ordinal: int, exact: bool | None = None) -> ExactPoint:
        return self.primary_point(A, ordinal, exact)

    def minus_chart_image(self, A: int, ordinal: int, exact: bool | None = None) -> ExactPoint:
        """Omega- reads reversed. The reference is preserved; the value flips."""
        n = self.point_count(A)
        if not 0 <= ordinal < n:
            raise IndexError(f"ordinal {ordinal} outside layer {A} of size {n}")
        mirror = self.primary_point(A, n - 1 - ordinal, exact)
        return ExactPoint(A, ordinal, mirror.refinement, mirror.phase_index,
                          mirror.magnitude, mirror.log2_magnitude)

    def plus_to_minus_image(self, A: int, ordinal: int, exact: bool | None = None) -> ExactPoint:
        return self.minus_chart_image(A, ordinal, exact)

    def minus_to_plus_image(self, A: int, ordinal: int, exact: bool | None = None) -> ExactPoint:
        return self.primary_point(A, ordinal, exact)

    # -- relation ---------------------------------------------------------

    def same_layer_membership(self, A: int, left: int, right: int) -> bool:
        n = self.point_count(A)
        for o in (left, right):
            if not 0 <= o < n:
                raise IndexError(f"ordinal {o} outside layer {A} of size {n}")
        return left < right

    def cross_layer_membership(self, la: int, oa: int, lb: int, ob: int) -> bool:
        if la == lb:
            raise ValueError("use same_layer_membership within one layer")
        for A, o in ((la, oa), (lb, ob)):
            n = self.point_count(A)
            if not 0 <= o < n:
                raise IndexError(f"ordinal {o} outside layer {A} of size {n}")
        return True

    def relation_cardinalities(self) -> dict[str, int]:
        """Exact cardinalities over every retained layer, active one included."""
        sizes = [s.point_count for s in self.layers()]
        total = sum(sizes)
        sumsq = sum(n * n for n in sizes)
        within = (sumsq - total) // 2
        cross = total * total - sumsq
        return {
            "layers": len(sizes),
            "total_points": total,
            "within_edges": within,
            "directed_cross_placements": cross,
            "total_relation_entries": within + cross,
            "chart_point_entries": total,
            "transfer_entries_each_direction": total,
        }

    # -- interface --------------------------------------------------------

    def interface(self, A: int) -> Interface:
        """The single boundary crossing that closes domain A."""
        self._check(A)
        if self.is_active(A):
            raise ValueError(f"domain {A} is still active and has not crossed")
        refine = self.custody.refinements(A)
        g = self.custody.orbit(A)
        carry = None
        if A >= 1:
            gp = self.custody.orbit(A - 1)
            carry = int(2 * gp + Fraction(self.custody._shift, self.custody._scale)) - 1
        log2_t = 2.0 * self.custody.alpha * (1.0 - float(g))
        return Interface(
            domain=A,
            refinements=refine,
            interior=refine - 1,
            crossings=1,
            carry=carry,
            orbit_g=float(g),
            log2_transgression=log2_t,
            transgression=float(2.0**log2_t),
        )

    def _check(self, A: int) -> None:
        last = self.state.domain if self.state else self.depth
        if not 0 <= A <= last:
            raise IndexError(f"layer {A} outside retained range 0..{last}")


# --------------------------------------------------------------------------
# CLI


def _fmt(v: int) -> str:
    return str(v) if v < 10**14 else f"~10^{int(v.bit_length() * 0.30103)}"


def main() -> None:
    p = argparse.ArgumentParser(
        prog="orthad_closed_form",
        description="Execution-free exact read surface over the Orthad.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    t = sub.add_parser("tick", help="aggregate custody state at a tick")
    t.add_argument("tick", type=int)

    ls = sub.add_parser("layers", help="layer summaries at a tick")
    ls.add_argument("tick", type=int)

    d = sub.add_parser("layer", help="dump a completed layer")
    d.add_argument("layer", type=int)
    d.add_argument("--max-points", type=int, default=16)

    i = sub.add_parser("interface", help="crossing records over a domain range")
    i.add_argument("first", type=int)
    i.add_argument("last", type=int, nargs="?")

    r = sub.add_parser("relation", help="relation cardinalities at a tick")
    r.add_argument("tick", type=int)

    args = p.parse_args()

    if args.command in ("tick", "layers", "relation"):
        port = OrthadClosedForm.at_tick(args.tick)
        if args.command == "tick":
            s = port.state
            for k, v in (
                ("tick", s.tick), ("B", s.b_total), ("Q+L", s.q_plus_l),
                ("A", s.domain), ("k", s.k), ("j", s.j),
                ("active_points", s.active_points),
                ("completed_points", s.completed_points),
            ):
                print(f"  {k:<18} {_fmt(v)}")
            return
        if args.command == "layers":
            head = f"{'A':>6}{'points':>16}{'orientation':>13}{'state':>10}"
            print(head); print("-" * len(head))
            for s in port.layers():
                print(f"{s.layer:>6}{_fmt(s.point_count):>16}{s.orientation:>13}"
                      f"{('complete' if s.complete else 'ACTIVE'):>10}")
            return
        for k, v in port.relation_cardinalities().items():
            print(f"  {k:<34} {v}")
        return

    if args.command == "layer":
        port = OrthadClosedForm(args.layer)
        s = port.layer_summary(args.layer)
        n = s.point_count
        print(f"D{args.layer}: {n} points, closing orientation {s.orientation}")
        shown = min(args.max_points, n)
        tail = "" if shown == n else f", ... ({n - shown} more)"
        fwd = [port.primary_point(args.layer, r).render() for r in range(shown)]
        rev = [port.minus_chart_image(args.layer, r).render() for r in range(shown)]
        print(f"  Omega+ : ({', '.join(fwd)}{tail})")
        print(f"  Omega- : ({', '.join(rev)}{tail})")
        return

    if args.command == "interface":
        last = args.last if args.last is not None else args.first
        port = OrthadClosedForm(last)
        head = f"{'A':>7}{'refine':>16}{'interior':>16}{'cross':>7}{'carry':>7}{'g':>12}{'T':>10}"
        print(head); print("-" * len(head))
        for A in range(args.first, last + 1):
            r = port.interface(A)
            carry = "-" if r.carry is None else str(r.carry)
            print(f"{A:>7}{_fmt(r.refinements):>16}{_fmt(r.interior):>16}"
                  f"{r.crossings:>7}{carry:>7}{r.orbit_g:>12.8f}{r.transgression:>10.6f}")
        return


if __name__ == "__main__":
    main()
