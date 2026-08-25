#![no_std]

use phase_world::{
    PhaseWorldCell, PhaseWorldDisposition, PhaseWorldError, PhaseWorldResult,
};
use qbl_abi::QblPrimitive;
use world_core::{
    AdmittedCauseV0, MutationIntentV0, TransitionScratch, ZoneKey,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PairDisposition {
    NoWork,
    Committed,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ParticipantPosition {
    First,
    Second,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PairError {
    SameCell,
    SameZone,
    ParticipantWorkMissing,
    StaleVersion {
        zone: ZoneKey,
        expected: u64,
        actual: u64,
    },
    CoordinationIdExhausted,
    Participant {
        position: ParticipantPosition,
        zone: ZoneKey,
        error: PhaseWorldError,
    },
}

#[derive(Clone, Copy, Debug)]
pub struct LocalWork<'a> {
    pub expected_local_commit_id: u64,
    pub causes: &'a [AdmittedCauseV0],
    pub intents: &'a [MutationIntentV0],
}

impl<'a> LocalWork<'a> {
    pub const fn new(
        expected_local_commit_id: u64,
        causes: &'a [AdmittedCauseV0],
        intents: &'a [MutationIntentV0],
    ) -> Self {
        Self {
            expected_local_commit_id,
            causes,
            intents,
        }
    }

    pub fn is_empty(self) -> bool {
        self.causes.is_empty() && self.intents.is_empty()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ParticipantReceipt {
    pub zone: ZoneKey,
    pub before_local_commit_id: u64,
    pub after_local_commit_id: u64,
    pub primitive: QblPrimitive,
    pub accepted_intents: u32,
    pub rejected_requests: u32,
    pub phase_fingerprint: u64,
    pub world_fingerprint: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PairReceipt {
    pub disposition: PairDisposition,
    pub coordination_commit_id: u64,
    pub first: ParticipantReceipt,
    pub second: ParticipantReceipt,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RegionCausalCoordinator {
    coordination_commit_id: u64,
}

impl RegionCausalCoordinator {
    pub const fn new() -> Self {
        Self {
            coordination_commit_id: 0,
        }
    }

    pub const fn coordination_commit_id(&self) -> u64 {
        self.coordination_commit_id
    }

    pub fn transact_pair<
        const AXES: usize,
        const OBJECTS: usize,
        const BUCKETS: usize,
        const CAUSES: usize,
        const INTENTS: usize,
    >(
        &mut self,
        left: &mut PhaseWorldCell<AXES, OBJECTS, BUCKETS>,
        left_work: LocalWork<'_>,
        left_scratch: &mut TransitionScratch<CAUSES, INTENTS, OBJECTS>,
        right: &mut PhaseWorldCell<AXES, OBJECTS, BUCKETS>,
        right_work: LocalWork<'_>,
        right_scratch: &mut TransitionScratch<CAUSES, INTENTS, OBJECTS>,
    ) -> Result<PairReceipt, PairError> {
        if core::ptr::eq(&*left, &*right) {
            return Err(PairError::SameCell);
        }

        let left_zone = left.world().zone().key;
        let right_zone = right.world().zone().key;
        if left_zone == right_zone {
            return Err(PairError::SameZone);
        }

        if left_work.is_empty() && right_work.is_empty() {
            return Ok(PairReceipt {
                disposition: PairDisposition::NoWork,
                coordination_commit_id: self.coordination_commit_id,
                first: snapshot_receipt(left),
                second: snapshot_receipt(right),
            }
            .canonicalized());
        }
        if left_work.is_empty() || right_work.is_empty() {
            return Err(PairError::ParticipantWorkMissing);
        }

        if left_zone < right_zone {
            self.transact_ordered(
                left,
                left_work,
                left_scratch,
                right,
                right_work,
                right_scratch,
            )
        } else {
            self.transact_ordered(
                right,
                right_work,
                right_scratch,
                left,
                left_work,
                left_scratch,
            )
        }
    }

    fn transact_ordered<
        const AXES: usize,
        const OBJECTS: usize,
        const BUCKETS: usize,
        const CAUSES: usize,
        const INTENTS: usize,
    >(
        &mut self,
        first: &mut PhaseWorldCell<AXES, OBJECTS, BUCKETS>,
        first_work: LocalWork<'_>,
        first_scratch: &mut TransitionScratch<CAUSES, INTENTS, OBJECTS>,
        second: &mut PhaseWorldCell<AXES, OBJECTS, BUCKETS>,
        second_work: LocalWork<'_>,
        second_scratch: &mut TransitionScratch<CAUSES, INTENTS, OBJECTS>,
    ) -> Result<PairReceipt, PairError> {
        let first_zone = first.world().zone().key;
        let second_zone = second.world().zone().key;

        verify_version(first_zone, first_work.expected_local_commit_id, first.local_commit_id())?;
        verify_version(
            second_zone,
            second_work.expected_local_commit_id,
            second.local_commit_id(),
        )?;

        let next_coordination_id = self
            .coordination_commit_id
            .checked_add(1)
            .ok_or(PairError::CoordinationIdExhausted)?;

        let first_before = first.local_commit_id();
        let second_before = second.local_commit_id();
        let mut staged_first = first.clone();
        let mut staged_second = second.clone();

        let first_result = staged_first
            .transact(first_work.causes, first_work.intents, first_scratch)
            .map_err(|error| PairError::Participant {
                position: ParticipantPosition::First,
                zone: first_zone,
                error,
            })?;
        require_committed(first_zone, ParticipantPosition::First, first_result)?;

        let second_result = staged_second
            .transact(second_work.causes, second_work.intents, second_scratch)
            .map_err(|error| PairError::Participant {
                position: ParticipantPosition::Second,
                zone: second_zone,
                error,
            })?;
        require_committed(second_zone, ParticipantPosition::Second, second_result)?;

        *first = staged_first;
        *second = staged_second;
        self.coordination_commit_id = next_coordination_id;

        Ok(PairReceipt {
            disposition: PairDisposition::Committed,
            coordination_commit_id: next_coordination_id,
            first: participant_receipt(first_zone, first_before, first_result),
            second: participant_receipt(second_zone, second_before, second_result),
        })
    }
}

impl Default for RegionCausalCoordinator {
    fn default() -> Self {
        Self::new()
    }
}

impl PairReceipt {
    fn canonicalized(mut self) -> Self {
        if self.first.zone > self.second.zone {
            core::mem::swap(&mut self.first, &mut self.second);
        }
        self
    }
}

fn verify_version(zone: ZoneKey, expected: u64, actual: u64) -> Result<(), PairError> {
    if expected != actual {
        return Err(PairError::StaleVersion {
            zone,
            expected,
            actual,
        });
    }
    Ok(())
}

fn require_committed(
    zone: ZoneKey,
    position: ParticipantPosition,
    result: PhaseWorldResult,
) -> Result<(), PairError> {
    if result.disposition != PhaseWorldDisposition::Committed {
        return Err(PairError::Participant {
            position,
            zone,
            error: PhaseWorldError::World(world_core::WorldTransitionError::Fault(
                world_core::WorldFault::CauseMissing,
            )),
        });
    }
    Ok(())
}

fn participant_receipt(
    zone: ZoneKey,
    before_local_commit_id: u64,
    result: PhaseWorldResult,
) -> ParticipantReceipt {
    ParticipantReceipt {
        zone,
        before_local_commit_id,
        after_local_commit_id: result.local_commit_id,
        primitive: result.primitive,
        accepted_intents: result.accepted_intents,
        rejected_requests: result.rejected_requests,
        phase_fingerprint: result.phase_fingerprint,
        world_fingerprint: result.world_fingerprint,
    }
}

fn snapshot_receipt<const AXES: usize, const OBJECTS: usize, const BUCKETS: usize>(
    cell: &PhaseWorldCell<AXES, OBJECTS, BUCKETS>,
) -> ParticipantReceipt {
    ParticipantReceipt {
        zone: cell.world().zone().key,
        before_local_commit_id: cell.local_commit_id(),
        after_local_commit_id: cell.local_commit_id(),
        primitive: QblPrimitive::NONE,
        accepted_intents: 0,
        rejected_requests: 0,
        phase_fingerprint: cell.phase().diagnostic_fingerprint64(),
        world_fingerprint: cell.world().diagnostic_fingerprint64(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use world_core::{ObjectKey, TransitionScratch, ZoneCoord};

    type Cell = PhaseWorldCell<8, 8, 16>;
    type Scratch = TransitionScratch<8, 32, 8>;

    fn cell(x: i32) -> Cell {
        Cell::new(ZoneCoord::new(x, 0, 0).unwrap()).unwrap()
    }

    fn cause(sequence: u32) -> [AdmittedCauseV0; 1] {
        [AdmittedCauseV0::external_input(sequence, 0, 0)]
    }

    fn advance(cell: &mut Cell, scratch: &mut Scratch, sequence: u32) {
        cell.transact(&cause(sequence), &[], scratch).unwrap();
    }

    #[test]
    fn no_work_preserves_cells_and_coordinator() {
        let mut coordinator = RegionCausalCoordinator::new();
        let mut a = cell(0);
        let mut b = cell(1);
        let before_a = a.clone();
        let before_b = b.clone();
        let mut sa = Scratch::new();
        let mut sb = Scratch::new();
        let receipt = coordinator
            .transact_pair(
                &mut a,
                LocalWork::new(0, &[], &[]),
                &mut sa,
                &mut b,
                LocalWork::new(0, &[], &[]),
                &mut sb,
            )
            .unwrap();
        assert_eq!(receipt.disposition, PairDisposition::NoWork);
        assert_eq!(coordinator.coordination_commit_id(), 0);
        assert_eq!(a, before_a);
        assert_eq!(b, before_b);
    }

    #[test]
    fn pair_commit_preserves_local_qbl_autonomy() {
        let mut coordinator = RegionCausalCoordinator::new();
        let mut a = cell(2);
        let mut b = cell(3);
        let mut sa = Scratch::new();
        let mut sb = Scratch::new();
        advance(&mut b, &mut sb, 0);

        let a_causes = cause(0);
        let b_causes = cause(1);
        let a_intents = [MutationIntentV0::spawn_actor(0, 91, 0)];
        let b_intents = [MutationIntentV0::spawn_actor(1, 73, 0)];
        let receipt = coordinator
            .transact_pair(
                &mut a,
                LocalWork::new(0, &a_causes, &a_intents),
                &mut sa,
                &mut b,
                LocalWork::new(1, &b_causes, &b_intents),
                &mut sb,
            )
            .unwrap();

        assert_eq!(receipt.disposition, PairDisposition::Committed);
        assert_eq!(receipt.coordination_commit_id, 1);
        assert_eq!(receipt.first.primitive, QblPrimitive::B);
        assert_eq!(receipt.second.primitive, QblPrimitive::Q);
        assert_eq!(a.world().objects().health(ObjectKey(1)), Some(91));
        assert_eq!(b.world().objects().health(ObjectKey(1)), Some(73));
    }

    #[test]
    fn second_participant_fault_rolls_back_both() {
        let mut coordinator = RegionCausalCoordinator::new();
        let mut a = cell(4);
        let mut b = cell(5);
        let before_a = a.clone();
        let before_b = b.clone();
        let mut sa = Scratch::new();
        let mut sb = Scratch::new();
        let a_causes = cause(0);
        let b_causes = cause(0);
        let a_intents = [MutationIntentV0::spawn_actor(0, 100, 0)];
        let b_intents = [MutationIntentV0::apply_health_delta(
            0,
            ObjectKey(999),
            -1,
        )];
        let result = coordinator.transact_pair(
            &mut a,
            LocalWork::new(0, &a_causes, &a_intents),
            &mut sa,
            &mut b,
            LocalWork::new(0, &b_causes, &b_intents),
            &mut sb,
        );
        assert!(matches!(
            result,
            Err(PairError::Participant {
                position: ParticipantPosition::Second,
                ..
            })
        ));
        assert_eq!(a, before_a);
        assert_eq!(b, before_b);
        assert_eq!(coordinator.coordination_commit_id(), 0);
    }

    #[test]
    fn stale_version_rejects_before_any_local_step() {
        let mut coordinator = RegionCausalCoordinator::new();
        let mut a = cell(6);
        let mut b = cell(7);
        let before_a = a.clone();
        let before_b = b.clone();
        let mut sa = Scratch::new();
        let mut sb = Scratch::new();
        let a_causes = cause(0);
        let b_causes = cause(0);
        let result = coordinator.transact_pair(
            &mut a,
            LocalWork::new(1, &a_causes, &[]),
            &mut sa,
            &mut b,
            LocalWork::new(0, &b_causes, &[]),
            &mut sb,
        );
        assert!(matches!(result, Err(PairError::StaleVersion { .. })));
        assert_eq!(a, before_a);
        assert_eq!(b, before_b);
    }

    #[test]
    fn participant_argument_order_does_not_change_result() {
        let mut c1 = RegionCausalCoordinator::new();
        let mut a1 = cell(8);
        let mut b1 = cell(9);
        let mut a2 = a1.clone();
        let mut b2 = b1.clone();
        let mut s1a = Scratch::new();
        let mut s1b = Scratch::new();
        let mut s2a = Scratch::new();
        let mut s2b = Scratch::new();
        let causes_a = cause(0);
        let causes_b = cause(0);
        let intents_a = [MutationIntentV0::spawn_actor(0, 51, 0)];
        let intents_b = [MutationIntentV0::spawn_actor(0, 52, 0)];
        let r1 = c1
            .transact_pair(
                &mut a1,
                LocalWork::new(0, &causes_a, &intents_a),
                &mut s1a,
                &mut b1,
                LocalWork::new(0, &causes_b, &intents_b),
                &mut s1b,
            )
            .unwrap();

        let mut c2 = RegionCausalCoordinator::new();
        let r2 = c2
            .transact_pair(
                &mut b2,
                LocalWork::new(0, &causes_b, &intents_b),
                &mut s2b,
                &mut a2,
                LocalWork::new(0, &causes_a, &intents_a),
                &mut s2a,
            )
            .unwrap();

        assert_eq!(r1, r2);
        assert_eq!(a1, a2);
        assert_eq!(b1, b2);
    }

    #[test]
    fn unrelated_cell_is_not_touched() {
        let mut coordinator = RegionCausalCoordinator::new();
        let mut a = cell(10);
        let mut b = cell(11);
        let c = cell(12);
        let before_c = c.clone();
        let mut sa = Scratch::new();
        let mut sb = Scratch::new();
        let ca = cause(0);
        let cb = cause(0);
        coordinator
            .transact_pair(
                &mut a,
                LocalWork::new(0, &ca, &[]),
                &mut sa,
                &mut b,
                LocalWork::new(0, &cb, &[]),
                &mut sb,
            )
            .unwrap();
        assert_eq!(c, before_c);
    }

    #[test]
    fn coordination_ids_are_scoped_not_global() {
        let mut first = RegionCausalCoordinator::new();
        let mut second = RegionCausalCoordinator::new();
        let mut a = cell(13);
        let mut b = cell(14);
        let mut c = cell(15);
        let mut d = cell(16);
        let mut sa = Scratch::new();
        let mut sb = Scratch::new();
        let mut sc = Scratch::new();
        let mut sd = Scratch::new();
        let ca = cause(0);
        let cb = cause(0);
        let cc = cause(0);
        let cd = cause(0);
        first
            .transact_pair(
                &mut a,
                LocalWork::new(0, &ca, &[]),
                &mut sa,
                &mut b,
                LocalWork::new(0, &cb, &[]),
                &mut sb,
            )
            .unwrap();
        second
            .transact_pair(
                &mut c,
                LocalWork::new(0, &cc, &[]),
                &mut sc,
                &mut d,
                LocalWork::new(0, &cd, &[]),
                &mut sd,
            )
            .unwrap();
        assert_eq!(first.coordination_commit_id(), 1);
        assert_eq!(second.coordination_commit_id(), 1);
    }
}
