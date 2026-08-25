from __future__ import annotations
import argparse, csv, json, math
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Sequence
import numpy as np

class Macro(str, Enum):
    R = "R"; S = "S"; T = "T"

@dataclass(frozen=True)
class BalancedPair:
    u: int = 1
    v: int = 1
    def __post_init__(self) -> None:
        if self.u < 1 or self.v < 1: raise ValueError("balanced pair coordinates must be positive")
        if self.u > self.v:
            old_u, old_v = self.u, self.v
            object.__setattr__(self, "u", old_v); object.__setattr__(self, "v", old_u)
    @property
    def product(self) -> int: return self.u * self.v
    def refine(self) -> "BalancedPair": return BalancedPair(self.v, self.u + self.v)

@dataclass(frozen=True)
class CompletionGerm:
    theta_tick: int
    denominator: int
    @property
    def left_pi_units(self) -> str: return f"{self.theta_tick}/2 - 1/{self.denominator}"
    @property
    def right_pi_units(self) -> str: return f"{self.theta_tick}/2 + 1/{self.denominator}"
    @property
    def half_width(self) -> float: return math.pi / float(self.denominator)

@dataclass(frozen=True)
class PhaseCoordinates:
    A: int = 0
    q: BalancedPair = field(default_factory=BalancedPair)
    theta_tick: int = 0
    kappa: int = 0
    c: CompletionGerm = field(default_factory=lambda: CompletionGerm(0, 1))
    @classmethod
    def initial(cls) -> "PhaseCoordinates": return cls().recomputed()
    def recomputed(self) -> "PhaseCoordinates":
        return PhaseCoordinates(self.A, self.q, self.theta_tick, self.theta_tick // 4, CompletionGerm(self.theta_tick, self.q.product))
    def with_values(self, *, A: int | None = None, q: BalancedPair | None = None, theta_tick: int | None = None) -> "PhaseCoordinates":
        return PhaseCoordinates(self.A if A is None else A, self.q if q is None else q, self.theta_tick if theta_tick is None else theta_tick).recomputed()

def Q(p: PhaseCoordinates) -> PhaseCoordinates: return p.with_values(theta_tick=p.theta_tick + 1)
def B(p: PhaseCoordinates) -> PhaseCoordinates: return p.with_values(q=p.q.refine())
def L(p: PhaseCoordinates) -> PhaseCoordinates:
    a = p.A + 1; return p.with_values(A=a, q=BalancedPair(1, max(1, a)), theta_tick=p.theta_tick + 1)
def R(p: PhaseCoordinates) -> PhaseCoordinates: return p.with_values(q=p.q.refine(), theta_tick=p.theta_tick + 1)
def S(p: PhaseCoordinates) -> PhaseCoordinates: return Q(p)
def T(p: PhaseCoordinates) -> PhaseCoordinates: return L(p)

def selector(phase: PhaseCoordinates, width: int, floor_den: int) -> Macro:
    if width < 1 or floor_den < 1: raise ValueError("width and floor denominator must be positive")
    if phase.theta_tick % width == width - 1: return Macro.T
    return Macro.R if phase.q.product < floor_den else Macro.S

def apply_macro_phase(phase: PhaseCoordinates, op: Macro) -> PhaseCoordinates:
    if op == Macro.R: return R(phase)
    if op == Macro.S: return S(phase)
    if op == Macro.T: return T(phase)
    raise ValueError(f"unknown macro {op}")

@dataclass
class ExtendedLiftedState:
    phase: PhaseCoordinates
    phi: np.ndarray
    psi: np.ndarray
    debt: np.ndarray
    kT: float
    walkers: int
    macro_step: int = 0
    projection_opened: bool = False
    @classmethod
    def zero(cls, dimension: int) -> "ExtendedLiftedState":
        z = np.zeros(dimension, dtype=float); return cls(PhaseCoordinates.initial(), z.copy(), z.copy(), z.copy(), 0.0, dimension)
    def copy_with(self, **kwargs: Any) -> "ExtendedLiftedState":
        data = dict(phase=self.phase, phi=self.phi.copy(), psi=self.psi.copy(), debt=self.debt.copy(), kT=float(self.kT), walkers=int(self.walkers), macro_step=int(self.macro_step), projection_opened=bool(self.projection_opened))
        data.update(kwargs); return ExtendedLiftedState(**data)

@dataclass(frozen=True)
class TelegraphConfig:
    dt: float = 0.20; damping: float = 1.50; ridge: float = 1e-10; stiffness: float = 0.0
    debt_rate: float = 0.015; debt_decay: float = 0.985; thermal_decay: float = 0.97; thermal_rate: float = 0.03
    walker_threshold: float = 1e-5; inner_steps: int = 1

@dataclass(frozen=True)
class TerminationConfig:
    low_field_variance_tol: float = 1e-9
    stationary_energy_tol: float = 1e-9
    stationarity_window: int = 14
    min_macro_steps: int = 64

@dataclass(frozen=True)
class RegressorConfig:
    width: int = 64; floor_den: int = 4096; max_macro_steps: int = 384
    projection_threshold: float = 1e-7; condensation_quantile: float = 0.80
    telegraph: TelegraphConfig = field(default_factory=TelegraphConfig)
    termination: TerminationConfig = field(default_factory=TerminationConfig)

@dataclass(frozen=True)
class TimeSeriesDataset:
    name: str
    values: np.ndarray
    @staticmethod
    def from_array(name: str, values: Sequence[float] | np.ndarray) -> "TimeSeriesDataset":
        arr = np.asarray(values, dtype=float)
        if arr.ndim == 1: arr = arr.reshape(-1, 1)
        if arr.ndim != 2 or arr.shape[0] < 8: raise ValueError("time series must have at least eight rows")
        arr = arr[np.isfinite(arr).all(axis=1)]
        if arr.shape[0] < 8: raise ValueError("time series has fewer than eight finite rows")
        return TimeSeriesDataset(name, arr)

def load_time_series(path: str | Path, name: str | None = None) -> TimeSeriesDataset:
    p = Path(path); rows: list[list[float]] = []
    with p.open("r", newline="") as f:
        sample = f.read(4096); f.seek(0)
        try: dialect = csv.Sniffer().sniff(sample, delimiters=",;\t ") if sample.strip() else csv.excel
        except csv.Error: dialect = csv.excel
        for row in csv.reader(f, dialect):
            vals = []
            for cell in row:
                try: vals.append(float(cell))
                except ValueError: pass
            if vals: rows.append(vals)
    if not rows: raise ValueError(f"no numeric data in {p}")
    width = max(len(r) for r in rows); arr = np.full((len(rows), width), np.nan, dtype=float)
    for i, row in enumerate(rows): arr[i, : len(row)] = row
    arr = arr[:, np.isfinite(arr).any(axis=0)]; means = np.nanmean(arr, axis=0); ii = np.where(~np.isfinite(arr)); arr[ii] = np.take(means, ii[1])
    return TimeSeriesDataset.from_array(name or p.stem, arr)

def normalize_columns(values: np.ndarray) -> np.ndarray:
    center = np.median(values, axis=0); spread = np.percentile(values, 90, axis=0) - np.percentile(values, 10, axis=0)
    spread[spread < 1e-12] = 1.0
    return np.tanh((values - center) / spread)

@dataclass(frozen=True)
class RetainedDesignFrame:
    names: tuple[str, ...]
    raw_design: np.ndarray
    orthogonal_design: np.ndarray
    upper: np.ndarray
    target: np.ndarray
    @classmethod
    def from_dataset(cls, dataset: TimeSeriesDataset) -> "RetainedDesignFrame":
        z = normalize_columns(dataset.values); rows = z.shape[0] - 2
        cols, names = [np.ones(rows), np.linspace(-1.0, 1.0, z.shape[0], dtype=float)[2:]], ["1", "time"]
        for c in range(min(z.shape[1], 6)):
            y1, y0 = z[1:-1, c], z[:-2, c]
            cols.extend([y1, y0, y1 - y0]); names.extend([f"c{c}_lag1", f"c{c}_lag2", f"c{c}_delta"])
        raw = np.column_stack(cols).astype(float); keep = [0] + [j for j in range(1, raw.shape[1]) if float(np.std(raw[:, j])) > 1e-12]
        raw, names = raw[:, keep], [names[j] for j in keep]
        q, r = np.linalg.qr(raw, mode="reduced"); scale = float(np.sqrt(raw.shape[0]))
        return cls(tuple(names), raw, q * scale, r / scale, z[2:, 0].copy())
    def raw_coefficients(self, phi: np.ndarray) -> np.ndarray: return np.linalg.lstsq(self.upper, phi, rcond=None)[0]
    def expression(self, coefficients: np.ndarray, threshold: float) -> str:
        terms = [f"({float(c):.10g})*{n}" for c, n in zip(coefficients, self.names) if abs(float(c)) > threshold]
        return " + ".join(terms) if terms else "0"

def ring_laplacian(size: int) -> np.ndarray:
    lap = np.zeros((size, size), dtype=float)
    if size <= 1: return lap
    for i in range(size): lap[i, i], lap[i, (i - 1) % size], lap[i, (i + 1) % size] = 2.0, -1.0, -1.0
    return lap

def residual(design: np.ndarray, y: np.ndarray, phi: np.ndarray) -> np.ndarray: return design @ phi - y

def gradient(design: np.ndarray, y: np.ndarray, phi: np.ndarray, laplacian: np.ndarray, cfg: TelegraphConfig) -> np.ndarray:
    res = residual(design, y, phi)
    return design.T @ res / float(design.shape[0]) + cfg.ridge * phi + cfg.stiffness * (laplacian @ phi)

def energy(design: np.ndarray, y: np.ndarray, phi: np.ndarray, psi: np.ndarray, laplacian: np.ndarray, cfg: TelegraphConfig) -> float:
    res = residual(design, y, phi)
    return 0.5 * float(np.mean(res * res)) + 0.5 * float(np.dot(psi, psi)) + 0.5 * cfg.ridge * float(np.dot(phi, phi)) + 0.5 * cfg.stiffness * float(phi @ laplacian @ phi)

def telegraph_step(state: ExtendedLiftedState, *, design: np.ndarray, y: np.ndarray, laplacian: np.ndarray, cfg: TelegraphConfig) -> ExtendedLiftedState:
    phi, psi, debt, kT = state.phi.copy(), state.psi.copy(), state.debt.copy(), float(state.kT)
    for _ in range(cfg.inner_steps):
        grad = gradient(design, y, phi, laplacian, cfg)
        debt = cfg.debt_decay * debt + cfg.debt_rate * np.abs(grad)
        psi = (1.0 - (cfg.damping + 0.05 * debt) * cfg.dt) * psi - cfg.dt * grad
        phi = phi + cfg.dt * psi
        kT = cfg.thermal_decay * kT + cfg.thermal_rate * float(np.mean(residual(design, y, phi) ** 2))
    return state.copy_with(phi=phi, psi=psi, debt=debt, kT=kT, walkers=int(np.count_nonzero(np.abs(psi) > cfg.walker_threshold)))

@dataclass(frozen=True)
class TerminationReport:
    lowFieldVariance: bool
    zeroWalkers: bool
    stationaryEnergy: bool
    lowFieldVarianceValue: float
    walkerCount: int
    stationaryEnergySpan: float
    reason: str
    @property
    def terminated(self) -> bool: return self.lowFieldVariance and self.zeroWalkers and self.stationaryEnergy

@dataclass(frozen=True)
class SolveResult:
    dataset: str; macro_steps: int; projection_open_count: int; phase: dict[str, Any]; projection: dict[str, Any]
    termination_report: TerminationReport; discovered_law: dict[str, Any]; history: list[dict[str, Any]]
    def to_jsonable(self) -> dict[str, Any]:
        return dict(dataset=self.dataset, macro_steps=self.macro_steps, projection_open_count=self.projection_open_count, phase=self.phase, projection=self.projection, termination_report=asdict(self.termination_report), discovered_law=self.discovered_law, history_tail=self.history[-8:])

class ProjectionGate:
    def __init__(self) -> None: self.open_count = 0
    def open(self) -> None:
        if self.open_count != 0: raise RuntimeError("projection already opened")
        self.open_count = 1

class InternalMetriplecticRuntime:
    def __init__(self, frame: RetainedDesignFrame, cfg: RegressorConfig) -> None:
        self.frame, self.cfg = frame, cfg
        self.laplacian = ring_laplacian(frame.orthogonal_design.shape[1])
        self.state = ExtendedLiftedState.zero(frame.orthogonal_design.shape[1])
        self.stimulus_count, self.stimulus_norm = 0, 0.0
    def stimulate(self, sample: np.ndarray) -> None:
        self.stimulus_count += 1; self.stimulus_norm += float(np.linalg.norm(sample))
    def step(self) -> tuple[ExtendedLiftedState, Macro, float]:
        op = selector(self.state.phase, self.cfg.width, self.cfg.floor_den)
        state = self.state.copy_with(phase=apply_macro_phase(self.state.phase, op), macro_step=self.state.macro_step + 1)
        state = telegraph_step(state, design=self.frame.orthogonal_design, y=self.frame.target, laplacian=self.laplacian, cfg=self.cfg.telegraph)
        e = energy(self.frame.orthogonal_design, self.frame.target, state.phi, state.psi, self.laplacian, self.cfg.telegraph)
        self.state = state; return state, op, e

def _rle(ops: Sequence[str]) -> str:
    out, last, count = [], None, 0
    for op in ops:
        if op == last: count += 1
        else:
            if last is not None: out.append(last if count == 1 else f"{last}^{count}")
            last, count = op, 1
    if last is not None: out.append(last if count == 1 else f"{last}^{count}")
    return " ".join(out) if out else "∅"

def _termination(state: ExtendedLiftedState, energies: list[float], cfg: RegressorConfig) -> TerminationReport:
    vel = float(np.mean(state.psi * state.psi)); w = cfg.termination.stationarity_window
    span = float(max(energies[-w:]) - min(energies[-w:])) if len(energies) >= w else float("inf")
    low, zero, stationary = vel <= cfg.termination.low_field_variance_tol, state.walkers == 0, span <= cfg.termination.stationary_energy_tol
    reason = "lowFieldVariance_zeroWalkers_stationaryEnergy" if low and zero and stationary else "active"
    return TerminationReport(low, zero, stationary, vel, state.walkers, span, reason)

def _condensed_bonds(phi: np.ndarray, debt: np.ndarray, quantile: float) -> tuple[list[dict[str, float | int]], dict[str, Any]]:
    n = phi.size
    if n <= 1: return [], {"total_edges": 0, "threshold": 0.0, "mean_strength": 0.0}
    edges = [(i, (i + 1) % n, 1.0 / (1.0 + abs(float(phi[i] - phi[(i + 1) % n])) + 0.05 * float(debt[i] + debt[(i + 1) % n]))) for i in range(n)]
    scores = np.asarray([e[2] for e in edges]); threshold = float(np.quantile(scores, quantile))
    selected = sorted([e for e in edges if e[2] >= threshold], key=lambda x: (-x[2], x[0], x[1]))
    return [{"u": int(u), "v": int(v), "strength": float(s)} for u, v, s in selected], {"total_edges": n, "threshold": threshold, "mean_strength": float(np.mean(scores))}

class VDMFullDynamicsRegressor:
    def __init__(self, config: RegressorConfig | None = None) -> None: self.config = config or RegressorConfig()
    def fit(self, dataset: TimeSeriesDataset) -> SolveResult:
        frame, zdata = RetainedDesignFrame.from_dataset(dataset), normalize_columns(dataset.values)
        runtime, ops, logs, energies = InternalMetriplecticRuntime(frame, self.config), [], [], []
        term = _termination(runtime.state, energies, self.config)
        for _ in range(self.config.max_macro_steps):
            runtime.stimulate(zdata[min(runtime.state.macro_step, zdata.shape[0] - 1)])
            state, op, e = runtime.step(); energies.append(e); ops.append(op.value)
            vel = float(np.mean(state.psi * state.psi))
            logs.append(dict(macro_step=state.macro_step, operator=op.value, A=state.phase.A, u=state.phase.q.u, v=state.phase.q.v, theta_tick=state.phase.theta_tick, kappa=state.phase.kappa, energy=e, lowFieldVarianceValue=vel, walkers=state.walkers, kT=state.kT))
            term = _termination(state, energies, self.config)
            if state.macro_step >= self.config.termination.min_macro_steps and term.terminated: break
        gate = ProjectionGate(); gate.open(); state = runtime.state.copy_with(projection_opened=True)
        coeffs = frame.raw_coefficients(state.phi); preds = frame.raw_design @ coeffs
        projection = dict(coefficients=[float(x) for x in coeffs], expression=frame.expression(coeffs, self.config.projection_threshold), rmse=float(np.sqrt(np.mean((preds - frame.target) ** 2))))
        bonds, graph = _condensed_bonds(state.phi, state.debt, self.config.condensation_quantile)
        macro, phase = _rle(ops), state.phase
        law = dict(expression=f"LAW[{macro}] :: {projection['expression']}", effective_macro_operator=macro, condensed_bond_pattern=bonds, stabilized_subgraph=graph, gauge_traffic=dict(stimulus_count=runtime.stimulus_count, stimulus_norm=runtime.stimulus_norm, walker_total=int(sum(x["walkers"] for x in logs)), final_kT=float(state.kT)), field_signature=dict(phi_mean=float(np.mean(state.phi)), phi_variance=float(np.var(state.phi)), debt_mean=float(np.mean(state.debt)), top_field_indices=[int(i) for i in np.argsort(-np.abs(state.phi))[: min(8, state.phi.size)]]))
        return SolveResult(dataset.name, state.macro_step, gate.open_count, dict(A=phase.A, u=phase.q.u, v=phase.q.v, theta_tick=phase.theta_tick, kappa=phase.kappa, germ_left_pi_units=phase.c.left_pi_units, germ_right_pi_units=phase.c.right_pi_units, germ_half_width=phase.c.half_width), projection, term, law, logs)

def fit_csv(path: str | Path, config: RegressorConfig | None = None) -> SolveResult:
    return VDMFullDynamicsRegressor(config).fit(load_time_series(path))

def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Self-contained PCVDM full-dynamics regressor")
    parser.add_argument("dataset"); parser.add_argument("--output", default=""); parser.add_argument("--max-steps", type=int, default=384)
    args = parser.parse_args(argv)
    result = fit_csv(args.dataset, RegressorConfig(max_macro_steps=args.max_steps)); payload = result.to_jsonable()
    text = json.dumps(payload, indent=2, sort_keys=True); print(text)
    ok = result.projection_open_count == 1 and result.termination_report.terminated
    print("FINAL_RESULT: PASS" if ok else "FINAL_RESULT: FAIL")
    if args.output: Path(args.output).write_text(text + "\n", encoding="utf-8")
    return 0 if ok else 1
if __name__ == "__main__":
    raise SystemExit(main())
