from __future__ import annotations

from dataclasses import replace

from orthad_canon.domain.models import AxisState, DualChartLens, ExactEntry, ExactMatrix, LiftState


CARRIER_SIZE = 12
OVERLAP_SIZE = 6


def _entry(support: bool, phase: int) -> ExactEntry:
    return ExactEntry(1 if support else 0, phase % 24)


def _pair_phase(row: int, col: int, shift: int, phase_mod4: int) -> int:
    return (-2 * ((row + shift) % CARRIER_SIZE) * ((col + shift) % CARRIER_SIZE) + 6 * phase_mod4) % 24


def _gauss_phase(residue: int, shift: int) -> int:
    x = (residue + shift) % CARRIER_SIZE
    return (x * x) % 24


def _coverage_plus(residue: int) -> bool:
    return residue != 6


def _coverage_minus(residue: int) -> bool:
    return residue != 0


def _pairing(axis: AxisState, event_count: int, sever_transfer: bool, lifted: bool) -> DualChartLens:
    shift = (axis.u + axis.v) % OVERLAP_SIZE
    plus_rows: list[tuple[ExactEntry, ...]] = []
    minus_rows: list[tuple[ExactEntry, ...]] = []
    pairing_rows: list[tuple[ExactEntry, ...]] = []
    p2m_rows: list[tuple[ExactEntry, ...]] = []
    m2p_rows: list[tuple[ExactEntry, ...]] = []
    for row in range(CARRIER_SIZE):
        p_row: list[ExactEntry] = []
        m_row: list[ExactEntry] = []
        pair_row: list[ExactEntry] = []
        p2m_row: list[ExactEntry] = []
        m2p_row: list[ExactEntry] = []
        for col in range(CARRIER_SIZE):
            base = _pair_phase(row, col, shift, axis.phase_mod4)
            twist = _gauss_phase(row, shift) if lifted else 0
            pair_row.append(_entry(True, base + twist))
            p_row.append(_entry(_coverage_plus(row) and _coverage_plus(col), base + twist))
            m_row.append(_entry(_coverage_minus(row) and _coverage_minus(col), -base - twist))
            mapped_col = (-row) % CARRIER_SIZE
            transfer_here = _coverage_plus(row) and _coverage_minus(mapped_col) and col == mapped_col and not sever_transfer
            seat = (row + shift) % OVERLAP_SIZE
            transfer_phase = (4 * seat + 6 * axis.phase_mod4) % 24
            p2m_row.append(_entry(transfer_here, transfer_phase + (twist if lifted else 0)))
            m2p_row.append(_entry(transfer_here, -transfer_phase - (twist if lifted else 0)))
        pairing_rows.append(tuple(pair_row))
        plus_rows.append(tuple(p_row))
        minus_rows.append(tuple(m_row))
        p2m_rows.append(tuple(p2m_row))
        m2p_rows.append(tuple(m2p_row))
    return DualChartLens(
        pairing=tuple(pairing_rows),
        omega_plus=tuple(plus_rows),
        omega_minus=tuple(minus_rows),
        transfer_plus_to_minus=tuple(p2m_rows),
        transfer_minus_to_plus=tuple(m2p_rows),
        carrier_size=CARRIER_SIZE,
        event_count=event_count,
    )


def refine_pair(axis: AxisState) -> AxisState:
    u, v = axis.v, axis.u + axis.v
    if u > v:
        u, v = v, u
    return AxisState(u=u, v=v, phase_mod4=axis.phase_mod4, clock=axis.clock + 1, frozen=False)


def apply_q(state: LiftState, sever_transfer: bool = False) -> None:
    axis = state.active
    state.axes[state.active_axis] = replace(axis, phase_mod4=(axis.phase_mod4 + 1) % 4, clock=axis.clock + 1)
    state.word += "Q"
    if state.lens is not None:
        state.lens = _pairing(state.active, state.lens.event_count + 1, sever_transfer, lifted=False)
    state.event_log.append({"event": "Q", "active_axis": state.active_axis, "phase_mod4": state.active.phase_mod4})


def apply_b(state: LiftState, sever_transfer: bool = False) -> None:
    refined = refine_pair(state.active)
    state.axes[state.active_axis] = refined
    state.word += "B"
    state.floor_bit = (refined.u * refined.v) % 2
    state.lens = _pairing(refined, 1 if state.lens is None else state.lens.event_count + 1, sever_transfer, lifted=False)
    state.event_log.append({
        "event": "B",
        "active_axis": state.active_axis,
        "u": refined.u,
        "v": refined.v,
        "uv": refined.uv,
        "shift_mod6": (refined.u + refined.v) % 6,
        "floor_bit": state.floor_bit,
    })


def apply_l(state: LiftState, sever_transfer: bool = False, corrupt_axis: bool = False) -> None:
    old = state.active
    frozen = replace(old, frozen=True, clock=old.clock + 1)
    state.axes[state.active_axis] = frozen
    matrix_axis = replace(frozen, v=frozen.v + 1) if corrupt_axis else frozen
    frozen_shift = (matrix_axis.u + matrix_axis.v) % 6
    state.frozen_shift_mod6 = frozen_shift
    state.axes.append(AxisState(1, 1))
    state.active_axis += 1
    state.word += "L"
    state.lens = _pairing(matrix_axis, 1 if state.lens is None else state.lens.event_count + 1, sever_transfer, lifted=True)
    state.event_log.append({
        "event": "L",
        "frozen_axis": state.active_axis - 1,
        "frozen_axis_value": frozen.lens_axis,
        "frozen_shift_mod6": frozen_shift,
        "new_active_axis": state.active_axis,
        "lap_relation": "lap2=-lap1",
    })
