#![no_std]

use qbl_abi::{
    QblOrthadAxisU128, QblOrthadLocalStateU64, QblPrimitive, QblStatus,
    QblTransitionU64,
};
use qbl_kernel::step_orthad_local;
use world_core::{
    AdmittedCauseV0, MutationIntentV0, TransitionDisposition, TransitionResult,
    TransitionScratch, WorldCore, WorldFault, WorldTransitionError, ZoneCoord,
};

const ZERO_AXIS: QblOrthadAxisU128 = QblOrthadAxisU128 {
    origin_product_lo: 0,
    origin_product_hi: 0,
    current_product_lo: 0,
    current_product_hi: 0,
    phase_quadrant: 0,
    flags: 0,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PhaseWorldDisposition {
    NoWork,
    Committed,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PhaseWorldError {
    InvalidAxisCapacity,
    Phase(QblStatus),
    World(WorldTransitionError),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PhaseWorldResult {
    pub disposition: PhaseWorldDisposition,
    pub local_commit_id: u64,
    pub primitive: QblPrimitive,
    pub accepted_intents: u32,
    pub rejected_requests: u32,
    pub phase_fingerprint: u64,
    pub world_fingerprint: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PhaseLocalState<const AXES: usize> {
    u: u64,
    v: u64,
    domain: u64,
    local_position: u64,
    quarter_turns: u64,
    axis_count: usize,
    axes: [QblOrthadAxisU128; AXES],
}

impl<const AXES: usize> PhaseLocalState<AXES> {
    pub fn identity() -> Result<Self, PhaseWorldError> {
        if AXES == 0 {
            return Err(PhaseWorldError::InvalidAxisCapacity);
        }
        let mut axes = [ZERO_AXIS; AXES];
        axes[0] = QblOrthadAxisU128::identity(1, 0);
        Ok(Self {
            u: 1,
            v: 1,
            domain: 0,
            local_position: 0,
            quarter_turns: 0,
            axis_count: 1,
            axes,
        })
    }

    pub fn u(&self) -> u64 {
        self.u
    }

    pub fn v(&self) -> u64 {
        self.v
    }

    pub fn domain(&self) -> u64 {
        self.domain
    }

    pub fn local_position(&self) -> u64 {
        self.local_position
    }

    pub fn quarter_turns(&self) -> u64 {
        self.quarter_turns
    }

    pub fn axis_count(&self) -> usize {
        self.axis_count
    }

    pub fn axis_capacity(&self) -> usize {
        AXES
    }

    pub fn axes(&self) -> &[QblOrthadAxisU128] {
        &self.axes[..self.axis_count]
    }

    fn with_ffi_mut<R>(
        &mut self,
        operation: impl FnOnce(&mut QblOrthadLocalStateU64) -> R,
    ) -> R {
        let mut ffi = QblOrthadLocalStateU64::new(
            self.domain,
            self.local_position,
            self.quarter_turns,
            self.u,
            self.v,
            self.axes.as_mut_ptr(),
            AXES as u64,
            self.axis_count as u64,
        );
        let result = operation(&mut ffi);
        self.u = ffi.u;
        self.v = ffi.v;
        self.domain = ffi.domain;
        self.local_position = ffi.local_position;
        self.quarter_turns = ffi.quarter_turns;
        self.axis_count = ffi.axis_count as usize;
        result
    }

    pub fn diagnostic_fingerprint64(&self) -> u64 {
        let mut hash = 0xcbf29ce484222325_u64;
        fn feed(hash: &mut u64, bytes: &[u8]) {
            for byte in bytes {
                *hash ^= *byte as u64;
                *hash = hash.wrapping_mul(0x100000001b3);
            }
        }
        feed(&mut hash, &self.u.to_le_bytes());
        feed(&mut hash, &self.v.to_le_bytes());
        feed(&mut hash, &self.domain.to_le_bytes());
        feed(&mut hash, &self.local_position.to_le_bytes());
        feed(&mut hash, &self.quarter_turns.to_le_bytes());
        feed(&mut hash, &(self.axis_count as u64).to_le_bytes());
        for axis in self.axes() {
            feed(&mut hash, &axis.origin_product_lo.to_le_bytes());
            feed(&mut hash, &axis.origin_product_hi.to_le_bytes());
            feed(&mut hash, &axis.current_product_lo.to_le_bytes());
            feed(&mut hash, &axis.current_product_hi.to_le_bytes());
            feed(&mut hash, &axis.phase_quadrant.to_le_bytes());
            feed(&mut hash, &axis.flags.to_le_bytes());
        }
        hash
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PhaseWorldCell<
    const AXES: usize,
    const OBJECTS: usize,
    const BUCKETS: usize,
> {
    phase: PhaseLocalState<AXES>,
    world: WorldCore<OBJECTS, BUCKETS>,
}

impl<const AXES: usize, const OBJECTS: usize, const BUCKETS: usize>
    PhaseWorldCell<AXES, OBJECTS, BUCKETS>
{
    pub fn new(coord: ZoneCoord) -> Result<Self, PhaseWorldError> {
        let phase = PhaseLocalState::identity()?;
        let world = WorldCore::new(coord)
            .map_err(|fault| PhaseWorldError::World(WorldTransitionError::Fault(fault)))?;
        Ok(Self { phase, world })
    }

    pub fn phase(&self) -> &PhaseLocalState<AXES> {
        &self.phase
    }

    pub fn world(&self) -> &WorldCore<OBJECTS, BUCKETS> {
        &self.world
    }

    pub fn local_commit_id(&self) -> u64 {
        self.world.accepted_transition_id()
    }

    pub fn transact<
        const CAUSES: usize,
        const INTENTS: usize,
    >(
        &mut self,
        causes: &[AdmittedCauseV0],
        intents: &[MutationIntentV0],
        scratch: &mut TransitionScratch<CAUSES, INTENTS, OBJECTS>,
    ) -> Result<PhaseWorldResult, PhaseWorldError> {
        if causes.is_empty() && intents.is_empty() {
            return Ok(PhaseWorldResult {
                disposition: PhaseWorldDisposition::NoWork,
                local_commit_id: self.local_commit_id(),
                primitive: QblPrimitive::NONE,
                accepted_intents: 0,
                rejected_requests: 0,
                phase_fingerprint: self.phase.diagnostic_fingerprint64(),
                world_fingerprint: self.world.diagnostic_fingerprint64(),
            });
        }
        if causes.is_empty() {
            return Err(PhaseWorldError::World(WorldTransitionError::Fault(
                WorldFault::CauseMissing,
            )));
        }

        let mut staged_phase = self.phase.clone();
        let mut qbl_receipt = QblTransitionU64::default();
        let status = staged_phase.with_ffi_mut(|ffi| {
            step_orthad_local(ffi, &mut qbl_receipt)
        });
        if !status.is_ok() {
            return Err(PhaseWorldError::Phase(status));
        }

        let world_result = self
            .world
            .transact(causes, intents, scratch)
            .map_err(PhaseWorldError::World)?;
        if world_result.disposition != TransitionDisposition::Committed {
            return Err(PhaseWorldError::World(WorldTransitionError::Fault(
                WorldFault::CauseMissing,
            )));
        }

        self.phase = staged_phase;
        Ok(Self::committed_result(qbl_receipt, world_result, &self.phase))
    }

    fn committed_result(
        qbl_receipt: QblTransitionU64,
        world_result: TransitionResult,
        phase: &PhaseLocalState<AXES>,
    ) -> PhaseWorldResult {
        PhaseWorldResult {
            disposition: PhaseWorldDisposition::Committed,
            local_commit_id: world_result.transition_id,
            primitive: QblPrimitive(qbl_receipt.primitive),
            accepted_intents: world_result.accepted_intents,
            rejected_requests: world_result.rejected_requests,
            phase_fingerprint: phase.diagnostic_fingerprint64(),
            world_fingerprint: world_result.diagnostic_fingerprint,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use qbl_abi::QBL_ORTHAD_AXIS_ACTIVE;
    use world_core::{ObjectKey, TransitionScratch};

    type Cell = PhaseWorldCell<8, 8, 16>;
    type SmallAxisCell = PhaseWorldCell<1, 8, 16>;
    type Scratch = TransitionScratch<8, 32, 8>;

    fn coord(x: i32) -> ZoneCoord {
        ZoneCoord::new(x, 0, 0).unwrap()
    }

    fn cause(sequence: u32) -> [AdmittedCauseV0; 1] {
        [AdmittedCauseV0::external_input(sequence, 0, 0)]
    }

    fn cause_only(cell: &mut Cell, scratch: &mut Scratch, sequence: u32) {
        let result = cell.transact(&cause(sequence), &[], scratch).unwrap();
        assert_eq!(result.disposition, PhaseWorldDisposition::Committed);
    }

    #[test]
    fn no_work_changes_neither_phase_nor_world() {
        let mut cell = Cell::new(coord(0)).unwrap();
        let mut scratch = Scratch::new();
        let before = cell.clone();
        for _ in 0..1000 {
            let result = cell.transact(&[], &[], &mut scratch).unwrap();
            assert_eq!(result.disposition, PhaseWorldDisposition::NoWork);
            assert_eq!(result.primitive, QblPrimitive::NONE);
        }
        assert_eq!(cell, before);
    }

    #[test]
    fn one_local_cause_commits_phase_and_world_once() {
        let mut cell = Cell::new(coord(0)).unwrap();
        let mut scratch = Scratch::new();
        let intents = [MutationIntentV0::spawn_actor(0, 100, 0)];
        let result = cell.transact(&cause(0), &intents, &mut scratch).unwrap();
        assert_eq!(result.primitive, QblPrimitive::B);
        assert_eq!(result.local_commit_id, 1);
        assert_eq!((cell.phase().u(), cell.phase().v()), (1, 2));
        assert_eq!(cell.world().accepted_transition_id(), 1);
        assert_eq!(cell.world().objects().health(ObjectKey(1)), Some(100));
    }

    #[test]
    fn world_fault_rolls_back_the_staged_phase() {
        let mut cell = Cell::new(coord(1)).unwrap();
        let mut scratch = Scratch::new();
        let before = cell.clone();
        let invalid = [MutationIntentV0::replace_kinematics(
            0,
            ObjectKey(999),
            1,
            2,
            3,
            4,
            5,
            6,
        )];
        let result = cell.transact(&cause(0), &invalid, &mut scratch);
        assert_eq!(
            result,
            Err(PhaseWorldError::World(WorldTransitionError::Fault(
                WorldFault::ObjectNotFound
            )))
        );
        assert_eq!(cell, before);
    }

    #[test]
    fn phase_capacity_fault_prevents_world_commit() {
        let mut cell = SmallAxisCell::new(coord(2)).unwrap();
        let mut scratch = Scratch::new();
        for sequence in 0..14 {
            let result = cell.transact(&cause(sequence), &[], &mut scratch).unwrap();
            assert_eq!(result.disposition, PhaseWorldDisposition::Committed);
        }
        let before = cell.clone();
        let result = cell.transact(&cause(14), &[], &mut scratch);
        assert_eq!(result, Err(PhaseWorldError::Phase(QblStatus::ORTHAD_FULL)));
        assert_eq!(cell, before);
        assert_eq!(cell.local_commit_id(), 14);
    }

    #[test]
    fn local_cells_accumulate_independent_progression_depths() {
        let mut fast = Cell::new(coord(3)).unwrap();
        let mut slow = Cell::new(coord(4)).unwrap();
        let mut fast_scratch = Scratch::new();
        let mut slow_scratch = Scratch::new();
        for sequence in 0..6 {
            cause_only(&mut fast, &mut fast_scratch, sequence);
        }
        for sequence in 0..2 {
            cause_only(&mut slow, &mut slow_scratch, sequence);
        }
        let slow_before = slow.clone();
        cause_only(&mut fast, &mut fast_scratch, 6);
        assert_eq!(slow, slow_before);
        assert_eq!(fast.local_commit_id(), 7);
        assert_eq!(slow.local_commit_id(), 2);
        assert_ne!(fast.phase(), slow.phase());
    }

    #[test]
    fn primitive_receipt_does_not_change_world_intent_meaning() {
        let mut at_b = Cell::new(coord(5)).unwrap();
        let mut at_q = Cell::new(coord(6)).unwrap();
        let mut b_scratch = Scratch::new();
        let mut q_scratch = Scratch::new();
        cause_only(&mut at_q, &mut q_scratch, 0);
        let spawn = [MutationIntentV0::spawn_actor(0, 77, 0)];
        let b = at_b.transact(&cause(1), &spawn, &mut b_scratch).unwrap();
        let q = at_q.transact(&cause(1), &spawn, &mut q_scratch).unwrap();
        assert_eq!(b.primitive, QblPrimitive::B);
        assert_eq!(q.primitive, QblPrimitive::Q);
        assert_eq!(at_b.world().objects().health(ObjectKey(1)), Some(77));
        assert_eq!(at_q.world().objects().health(ObjectKey(1)), Some(77));
    }

    #[test]
    fn active_axis_is_nonzero_when_l_is_committed() {
        let mut cell = Cell::new(coord(7)).unwrap();
        let mut scratch = Scratch::new();
        let mut last = QblPrimitive::NONE;
        for sequence in 0..15 {
            last = cell
                .transact(&cause(sequence), &[], &mut scratch)
                .unwrap()
                .primitive;
        }
        assert_eq!(last, QblPrimitive::L);
        assert_eq!(cell.phase().axis_count(), 2);
        let active = cell.phase().axes()[1];
        assert_eq!(active.flags, QBL_ORTHAD_AXIS_ACTIVE);
        assert_eq!(active.origin_product_lo, 4895);
        assert_eq!(active.current_product_lo, 4895);
    }
}
