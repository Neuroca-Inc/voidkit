#![no_std]

use qbl_abi::{
    QblAdmissionReason, QblInspectStateU64, QblInspectionU64, QblPairU64,
    QblPrimitive, QblStatus, QBL_MAX_U64_DOMAIN,
};

pub fn refine(pair: &mut QblPairU64) -> QblStatus {
    if pair.u == 0 || pair.u > pair.v {
        return QblStatus::INVALID_PAIR;
    }

    let Some(next_v) = pair.u.checked_add(pair.v) else {
        return QblStatus::OVERFLOW;
    };

    pair.u = pair.v;
    pair.v = next_v;
    QblStatus::OK
}

fn product_le_capacity(product: u128, exponent: u64) -> bool {
    exponent >= 128 || product <= (1_u128 << exponent)
}

fn product_lt_capacity(product: u128, exponent: u64) -> bool {
    exponent >= 128 || product < (1_u128 << exponent)
}

pub fn inspect(
    state: &QblInspectStateU64,
    output: &mut QblInspectionU64,
) -> QblStatus {
    if state.u == 0 || state.u > state.v {
        return QblStatus::INVALID_PAIR;
    }
    if state.domain > QBL_MAX_U64_DOMAIN {
        return QblStatus::DOMAIN_RANGE;
    }

    let phase_positions = 6_u64 << state.domain;
    if state.local_position >= phase_positions {
        return QblStatus::INVALID_STATE;
    }

    let global_position = phase_positions - 5 + state.local_position;
    let capacity_exponent = match global_position {
        1 => 1,
        2 => 2,
        position => 2 * position,
    };
    let terminal_position = state.local_position == phase_positions - 1;
    let current_product = u128::from(state.u) * u128::from(state.v);

    let mut candidate = QblInspectionU64 {
        terminal_position: u32::from(terminal_position),
        phase_positions,
        global_position,
        capacity_exponent,
        current_product_lo: current_product as u64,
        current_product_hi: (current_product >> 64) as u64,
        ..QblInspectionU64::default()
    };

    if terminal_position {
        if !product_lt_capacity(current_product, capacity_exponent) {
            candidate.primitive = QblPrimitive::L.0;
            candidate.reason = QblAdmissionReason::L_DOMAIN_SATURATED.0;
            *output = candidate;
            return QblStatus::OK;
        }

        let Some(next_v) = state.u.checked_add(state.v) else {
            return QblStatus::NEEDS_WIDE_PAIR;
        };
        candidate.next_u = state.v;
        candidate.next_v = next_v;
        let candidate_product = u128::from(candidate.next_u) * u128::from(next_v);
        candidate.candidate_product_lo = candidate_product as u64;
        candidate.candidate_product_hi = (candidate_product >> 64) as u64;
        candidate.can_b = 1;
        candidate.primitive = QblPrimitive::B.0;
        candidate.reason = QblAdmissionReason::B_FINAL_CROSSING.0;
        *output = candidate;
        return QblStatus::OK;
    }

    candidate.can_q = 1;
    if !product_lt_capacity(current_product, capacity_exponent) {
        candidate.primitive = QblPrimitive::Q.0;
        candidate.reason = QblAdmissionReason::Q_B_BLOCKED_POSITION_AVAILABLE.0;
        *output = candidate;
        return QblStatus::OK;
    }

    let Some(next_v) = state.u.checked_add(state.v) else {
        return QblStatus::NEEDS_WIDE_PAIR;
    };
    candidate.next_u = state.v;
    candidate.next_v = next_v;
    let candidate_product = u128::from(candidate.next_u) * u128::from(next_v);
    candidate.candidate_product_lo = candidate_product as u64;
    candidate.candidate_product_hi = (candidate_product >> 64) as u64;

    if product_le_capacity(candidate_product, capacity_exponent) {
        candidate.can_b = 1;
        candidate.primitive = QblPrimitive::B.0;
        candidate.reason = QblAdmissionReason::B_NEXT_WITHIN_CAPACITY.0;
    } else {
        candidate.primitive = QblPrimitive::Q.0;
        candidate.reason = QblAdmissionReason::Q_B_BLOCKED_POSITION_AVAILABLE.0;
    }

    *output = candidate;
    QblStatus::OK
}

fn capacity_exponent_for_position(global_position: u64) -> u64 {
    match global_position {
        1 => 1,
        2 => 2,
        position => 2 * position,
    }
}

pub fn step(
    state: &mut qbl_abi::QblCustodyStateU64,
    output: &mut qbl_abi::QblTransitionU64,
) -> QblStatus {
    use qbl_abi::{QblAdmissionReason, QblPrimitive, QblTransitionU64};

    if state.u == 0 || state.u > state.v {
        return QblStatus::INVALID_PAIR;
    }
    if state.domain > QBL_MAX_U64_DOMAIN {
        return QblStatus::DOMAIN_RANGE;
    }

    let before_phase_positions = 6_u64 << state.domain;
    if state.local_position >= before_phase_positions {
        return QblStatus::INVALID_STATE;
    }

    let before_global_position =
        before_phase_positions - 5 + state.local_position;
    let before_capacity_exponent =
        capacity_exponent_for_position(before_global_position);
    let before_product = u128::from(state.u) * u128::from(state.v);
    let terminal = state.local_position == before_phase_positions - 1;

    let mut next = *state;
    let mut receipt = QblTransitionU64 {
        before_phase_positions,
        before_global_position,
        before_capacity_exponent,
        before_product_lo: before_product as u64,
        before_product_hi: (before_product >> 64) as u64,
        ..QblTransitionU64::default()
    };

    if terminal {
        if product_lt_capacity(before_product, before_capacity_exponent) {
            let Some(next_v) = state.u.checked_add(state.v) else {
                return QblStatus::NEEDS_WIDE_PAIR;
            };
            next.u = state.v;
            next.v = next_v;
            receipt.primitive = QblPrimitive::B.0;
            receipt.reason = QblAdmissionReason::B_FINAL_CROSSING.0;
        } else {
            if state.domain == QBL_MAX_U64_DOMAIN {
                return QblStatus::DOMAIN_RANGE;
            }
            next.domain += 1;
            next.local_position = 0;
            receipt.primitive = QblPrimitive::L.0;
            receipt.reason = QblAdmissionReason::L_DOMAIN_SATURATED.0;
        }
    } else if product_lt_capacity(before_product, before_capacity_exponent) {
        let Some(next_v) = state.u.checked_add(state.v) else {
            return QblStatus::NEEDS_WIDE_PAIR;
        };
        let candidate_u = state.v;
        let candidate_product = u128::from(candidate_u) * u128::from(next_v);
        if product_le_capacity(candidate_product, before_capacity_exponent) {
            next.u = candidate_u;
            next.v = next_v;
            receipt.primitive = QblPrimitive::B.0;
            receipt.reason = QblAdmissionReason::B_NEXT_WITHIN_CAPACITY.0;
        } else {
            let Some(quarter_turns) = state.quarter_turns.checked_add(1) else {
                return QblStatus::OVERFLOW;
            };
            next.local_position += 1;
            next.quarter_turns = quarter_turns;
            receipt.primitive = QblPrimitive::Q.0;
            receipt.reason =
                QblAdmissionReason::Q_B_BLOCKED_POSITION_AVAILABLE.0;
        }
    } else {
        let Some(quarter_turns) = state.quarter_turns.checked_add(1) else {
            return QblStatus::OVERFLOW;
        };
        next.local_position += 1;
        next.quarter_turns = quarter_turns;
        receipt.primitive = QblPrimitive::Q.0;
        receipt.reason = QblAdmissionReason::Q_B_BLOCKED_POSITION_AVAILABLE.0;
    }

    let after_phase_positions = 6_u64 << next.domain;
    let after_global_position = after_phase_positions - 5 + next.local_position;
    let after_capacity_exponent =
        capacity_exponent_for_position(after_global_position);
    let after_product = u128::from(next.u) * u128::from(next.v);
    receipt.state_changed = 1;
    receipt.after_phase_positions = after_phase_positions;
    receipt.after_global_position = after_global_position;
    receipt.after_capacity_exponent = after_capacity_exponent;
    receipt.after_product_lo = after_product as u64;
    receipt.after_product_hi = (after_product >> 64) as u64;

    *output = receipt;
    *state = next;
    QblStatus::OK
}

pub fn word_get(bytes: &[u8], length: u64, index: u64) -> Result<QblPrimitive, QblStatus> {
    if index >= length {
        return Err(QblStatus::INVALID_WORD);
    }
    let byte_index = (index >> 2) as usize;
    if byte_index >= bytes.len() {
        return Err(QblStatus::INVALID_WORD);
    }
    let shift = ((index & 3) << 1) as u32;
    let code = u32::from((bytes[byte_index] >> shift) & 3);
    let primitive = QblPrimitive(code);
    if primitive == QblPrimitive::B || primitive == QblPrimitive::Q || primitive == QblPrimitive::L {
        Ok(primitive)
    } else {
        Err(QblStatus::INVALID_WORD)
    }
}

pub fn step_record(
    state: &mut qbl_abi::QblRetainedStateU64,
    output: &mut qbl_abi::QblTransitionU64,
) -> QblStatus {
    use qbl_abi::{QblCustodyStateU64, QblTransitionU64};

    if state.word_bytes.is_null() {
        return QblStatus::INVALID_WORD;
    }
    if state.word_length > state.word_capacity {
        return QblStatus::INVALID_WORD;
    }
    if state.word_length == state.word_capacity {
        return QblStatus::WORD_FULL;
    }

    let byte_index = (state.word_length >> 2) as usize;
    let shift = ((state.word_length & 3) << 1) as u32;
    let mask = 3_u8 << shift;
    let old_byte = unsafe { *state.word_bytes.add(byte_index) };
    if old_byte & mask != 0 {
        return QblStatus::INVALID_WORD;
    }

    let mut next = QblCustodyStateU64 {
        u: state.u,
        v: state.v,
        domain: state.domain,
        local_position: state.local_position,
        quarter_turns: state.quarter_turns,
    };
    let mut receipt = QblTransitionU64::default();
    let status = step(&mut next, &mut receipt);
    if !status.is_ok() {
        return status;
    }
    if receipt.primitive < QblPrimitive::B.0 || receipt.primitive > QblPrimitive::L.0 {
        return QblStatus::INVALID_STATE;
    }

    let new_byte = old_byte | ((receipt.primitive as u8) << shift);
    unsafe {
        *state.word_bytes.add(byte_index) = new_byte;
    }
    state.u = next.u;
    state.v = next.v;
    state.domain = next.domain;
    state.local_position = next.local_position;
    state.quarter_turns = next.quarter_turns;
    state.word_length += 1;
    *output = receipt;
    QblStatus::OK
}

pub fn step_orthad_local(
    state: &mut qbl_abi::QblOrthadLocalStateU64,
    output: &mut qbl_abi::QblTransitionU64,
) -> QblStatus {
    use qbl_abi::{
        QblCustodyStateU64, QblOrthadAxisU128, QblPrimitive,
        QBL_ORTHAD_AXIS_ACTIVE, QBL_ORTHAD_AXIS_LATCHED,
    };

    if state.axes.is_null() || state.axis_count == 0 ||
        state.axis_count > state.axis_capacity ||
        state.domain == u64::MAX || state.axis_count != state.domain + 1
    {
        return QblStatus::INVALID_ORTHAD;
    }
    let Ok(capacity) = usize::try_from(state.axis_capacity) else {
        return QblStatus::INVALID_ORTHAD;
    };
    let Ok(count) = usize::try_from(state.axis_count) else {
        return QblStatus::INVALID_ORTHAD;
    };
    let axes = unsafe { core::slice::from_raw_parts_mut(state.axes, capacity) };
    let active = axes[count - 1];
    if active.flags != QBL_ORTHAD_AXIS_ACTIVE ||
        active.phase_quadrant > 3 ||
        (active.origin_product_lo == 0 && active.origin_product_hi == 0) ||
        (active.current_product_lo == 0 && active.current_product_hi == 0) ||
        active.phase_quadrant != (state.local_position & 3) as u32
    {
        return QblStatus::INVALID_ORTHAD;
    }

    let mut next = QblCustodyStateU64::new(
        state.domain,
        state.local_position,
        state.quarter_turns,
        state.u,
        state.v,
    );
    let mut receipt = qbl_abi::QblTransitionU64::default();
    let status = step(&mut next, &mut receipt);
    if !status.is_ok() {
        return status;
    }
    if active.current_product_lo != receipt.before_product_lo ||
        active.current_product_hi != receipt.before_product_hi
    {
        return QblStatus::INVALID_ORTHAD;
    }

    let mut next_active = active;
    let mut new_axis = QblOrthadAxisU128::default();
    let append = if receipt.primitive == QblPrimitive::B.0 {
        next_active.current_product_lo = receipt.after_product_lo;
        next_active.current_product_hi = receipt.after_product_hi;
        false
    } else if receipt.primitive == QblPrimitive::Q.0 {
        next_active.phase_quadrant = (next_active.phase_quadrant + 1) & 3;
        false
    } else if receipt.primitive == QblPrimitive::L.0 {
        if count == capacity {
            return QblStatus::ORTHAD_FULL;
        }
        if axes[count] != QblOrthadAxisU128::default() {
            return QblStatus::INVALID_ORTHAD;
        }
        next_active.flags = QBL_ORTHAD_AXIS_LATCHED;
        new_axis = QblOrthadAxisU128::identity(
            receipt.after_product_lo,
            receipt.after_product_hi,
        );
        true
    } else {
        return QblStatus::INVALID_STATE;
    };

    axes[count - 1] = next_active;
    if append {
        axes[count] = new_axis;
        state.axis_count += 1;
    }
    state.u = next.u;
    state.v = next.v;
    state.domain = next.domain;
    state.local_position = next.local_position;
    state.quarter_turns = next.quarter_turns;
    *output = receipt;
    QblStatus::OK
}
