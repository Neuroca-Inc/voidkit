#![no_std]

use phase_world::{
    PhaseWorldCell, PhaseWorldDisposition, PhaseWorldError, PhaseWorldResult,
};
use qbl_abi::{QblPrimitive, QblStatus};
use world_core::{
    AdmittedCauseV0, MutationIntentV0, TransitionScratch, ZoneKey,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FrontierDisposition {
    NoWork,
    Committed,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FrontierError {
    EmptyRegion,
    DuplicateZone {
        first_slot: usize,
        second_slot: usize,
        zone: ZoneKey,
    },
    TooManyParticipants,
    InvalidSlot {
        slot: usize,
    },
    DuplicateParticipant {
        slot: usize,
        zone: ZoneKey,
    },
    ParticipantWorkMissing {
        slot: usize,
        zone: ZoneKey,
    },
    StaleVersion {
        slot: usize,
        zone: ZoneKey,
        expected: u64,
        actual: u64,
    },
    CoordinationIdExhausted,
    InternalInvariant,
    Participant {
        slot: usize,
        zone: ZoneKey,
        error: PhaseWorldError,
    },
}

#[derive(Clone, Copy, Debug)]
pub struct FrontierWork<'a> {
    pub slot: usize,
    pub expected_local_commit_id: u64,
    pub causes: &'a [AdmittedCauseV0],
    pub intents: &'a [MutationIntentV0],
}

impl<'a> FrontierWork<'a> {
    pub const fn new(
        slot: usize,
        expected_local_commit_id: u64,
        causes: &'a [AdmittedCauseV0],
        intents: &'a [MutationIntentV0],
    ) -> Self {
        Self {
            slot,
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
pub struct FrontierParticipantReceipt {
    pub slot: usize,
    pub zone: ZoneKey,
    pub before_local_commit_id: u64,
    pub after_local_commit_id: u64,
    pub primitive: QblPrimitive,
    pub accepted_intents: u32,
    pub rejected_requests: u32,
    pub phase_fingerprint: u64,
    pub world_fingerprint: u64,
}

const EMPTY_PARTICIPANT_RECEIPT: FrontierParticipantReceipt =
    FrontierParticipantReceipt {
        slot: 0,
        zone: ZoneKey(0),
        before_local_commit_id: 0,
        after_local_commit_id: 0,
        primitive: QblPrimitive::NONE,
        accepted_intents: 0,
        rejected_requests: 0,
        phase_fingerprint: 0,
        world_fingerprint: 0,
    };

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FrontierReceipt<const CELLS: usize> {
    pub disposition: FrontierDisposition,
    pub coordination_commit_id: u64,
    pub participant_count: usize,
    pub participants: [FrontierParticipantReceipt; CELLS],
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PhaseRegion<
    const CELLS: usize,
    const AXES: usize,
    const OBJECTS: usize,
    const BUCKETS: usize,
> {
    coordination_commit_id: u64,
    cells: [PhaseWorldCell<AXES, OBJECTS, BUCKETS>; CELLS],
}

impl<
        const CELLS: usize,
        const AXES: usize,
        const OBJECTS: usize,
        const BUCKETS: usize,
    > PhaseRegion<CELLS, AXES, OBJECTS, BUCKETS>
{
    pub fn new(
        cells: [PhaseWorldCell<AXES, OBJECTS, BUCKETS>; CELLS],
    ) -> Result<Self, FrontierError> {
        if CELLS == 0 {
            return Err(FrontierError::EmptyRegion);
        }
        for first in 0..CELLS {
            let first_zone = cells[first].world().zone().key;
            for second in (first + 1)..CELLS {
                let second_zone = cells[second].world().zone().key;
                if first_zone == second_zone {
                    return Err(FrontierError::DuplicateZone {
                        first_slot: first,
                        second_slot: second,
                        zone: first_zone,
                    });
                }
            }
        }
        Ok(Self {
            coordination_commit_id: 0,
            cells,
        })
    }

    pub const fn coordination_commit_id(&self) -> u64 {
        self.coordination_commit_id
    }

    pub fn cells(&self) -> &[PhaseWorldCell<AXES, OBJECTS, BUCKETS>; CELLS] {
        &self.cells
    }

    pub fn cell(
        &self,
        slot: usize,
    ) -> Option<&PhaseWorldCell<AXES, OBJECTS, BUCKETS>> {
        self.cells.get(slot)
    }

    pub fn cell_mut(
        &mut self,
        slot: usize,
    ) -> Option<&mut PhaseWorldCell<AXES, OBJECTS, BUCKETS>> {
        self.cells.get_mut(slot)
    }

    pub fn transact_frontier<
        const CAUSES: usize,
        const INTENTS: usize,
    >(
        &mut self,
        work: &[FrontierWork<'_>],
        scratch: &mut [TransitionScratch<CAUSES, INTENTS, OBJECTS>; CELLS],
    ) -> Result<FrontierReceipt<CELLS>, FrontierError> {
        if work.len() > CELLS {
            return Err(FrontierError::TooManyParticipants);
        }

        let mut seen_slots = [false; CELLS];
        let mut any_empty = false;
        let mut any_nonempty = false;
        for item in work {
            if item.slot >= CELLS {
                return Err(FrontierError::InvalidSlot { slot: item.slot });
            }
            let zone = self.cells[item.slot].world().zone().key;
            if seen_slots[item.slot] {
                return Err(FrontierError::DuplicateParticipant {
                    slot: item.slot,
                    zone,
                });
            }
            seen_slots[item.slot] = true;
            if item.is_empty() {
                any_empty = true;
            } else {
                any_nonempty = true;
            }
        }

        if !any_nonempty {
            return self.snapshot_receipt(work);
        }
        if any_empty {
            for item in work {
                if item.is_empty() {
                    return Err(FrontierError::ParticipantWorkMissing {
                        slot: item.slot,
                        zone: self.cells[item.slot].world().zone().key,
                    });
                }
            }
        }

        for item in work {
            let actual = self.cells[item.slot].local_commit_id();
            if item.expected_local_commit_id != actual {
                return Err(FrontierError::StaleVersion {
                    slot: item.slot,
                    zone: self.cells[item.slot].world().zone().key,
                    expected: item.expected_local_commit_id,
                    actual,
                });
            }
        }

        let next_coordination_id = self
            .coordination_commit_id
            .checked_add(1)
            .ok_or(FrontierError::CoordinationIdExhausted)?;

        let mut staged_cells = self.cells.clone();
        let mut participants = [EMPTY_PARTICIPANT_RECEIPT; CELLS];
        let mut consumed = [false; CELLS];

        for rank in 0..work.len() {
            let selected = select_next_by_zone(&staged_cells, work, &consumed)
                .ok_or(FrontierError::InternalInvariant)?;
            consumed[selected] = true;
            let item = work[selected];
            let slot = item.slot;
            let zone = staged_cells[slot].world().zone().key;
            let before = staged_cells[slot].local_commit_id();
            let result = staged_cells[slot]
                .transact(item.causes, item.intents, &mut scratch[slot])
                .map_err(|error| FrontierError::Participant { slot, zone, error })?;
            if result.disposition != PhaseWorldDisposition::Committed {
                return Err(FrontierError::Participant {
                    slot,
                    zone,
                    error: PhaseWorldError::Phase(QblStatus::INVALID_STATE),
                });
            }
            participants[rank] = participant_receipt(slot, zone, before, result);
        }

        self.cells = staged_cells;
        self.coordination_commit_id = next_coordination_id;

        Ok(FrontierReceipt {
            disposition: FrontierDisposition::Committed,
            coordination_commit_id: next_coordination_id,
            participant_count: work.len(),
            participants,
        })
    }

    fn snapshot_receipt(
        &self,
        work: &[FrontierWork<'_>],
    ) -> Result<FrontierReceipt<CELLS>, FrontierError> {
        let mut participants = [EMPTY_PARTICIPANT_RECEIPT; CELLS];
        let mut consumed = [false; CELLS];
        for rank in 0..work.len() {
            let selected = select_next_by_zone(&self.cells, work, &consumed)
                .ok_or(FrontierError::InternalInvariant)?;
            consumed[selected] = true;
            let slot = work[selected].slot;
            let cell = &self.cells[slot];
            participants[rank] = FrontierParticipantReceipt {
                slot,
                zone: cell.world().zone().key,
                before_local_commit_id: cell.local_commit_id(),
                after_local_commit_id: cell.local_commit_id(),
                primitive: QblPrimitive::NONE,
                accepted_intents: 0,
                rejected_requests: 0,
                phase_fingerprint: cell.phase().diagnostic_fingerprint64(),
                world_fingerprint: cell.world().diagnostic_fingerprint64(),
            };
        }
        Ok(FrontierReceipt {
            disposition: FrontierDisposition::NoWork,
            coordination_commit_id: self.coordination_commit_id,
            participant_count: work.len(),
            participants,
        })
    }
}

fn select_next_by_zone<
    const CELLS: usize,
    const AXES: usize,
    const OBJECTS: usize,
    const BUCKETS: usize,
>(
    cells: &[PhaseWorldCell<AXES, OBJECTS, BUCKETS>; CELLS],
    work: &[FrontierWork<'_>],
    consumed: &[bool; CELLS],
) -> Option<usize> {
    let mut selected: Option<usize> = None;
    for (index, item) in work.iter().enumerate() {
        if consumed[index] {
            continue;
        }
        let zone = cells[item.slot].world().zone().key;
        match selected {
            None => selected = Some(index),
            Some(current) => {
                let current_zone = cells[work[current].slot].world().zone().key;
                if zone < current_zone {
                    selected = Some(index);
                }
            }
        }
    }
    selected
}

fn participant_receipt(
    slot: usize,
    zone: ZoneKey,
    before_local_commit_id: u64,
    result: PhaseWorldResult,
) -> FrontierParticipantReceipt {
    FrontierParticipantReceipt {
        slot,
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

#[cfg(test)]
mod tests {
    use super::*;
    use world_core::{ObjectKey, TransitionScratch, ZoneCoord};

    type Cell = PhaseWorldCell<8, 8, 16>;
    type Region = PhaseRegion<5, 8, 8, 16>;
    type Scratch = TransitionScratch<8, 32, 8>;

    fn cell(x: i32) -> Cell {
        Cell::new(ZoneCoord::new(x, 0, 0).unwrap()).unwrap()
    }

    fn region() -> Region {
        Region::new([cell(0), cell(1), cell(2), cell(3), cell(4)]).unwrap()
    }

    fn scratches() -> [Scratch; 5] {
        core::array::from_fn(|_| Scratch::new())
    }

    fn cause(sequence: u32) -> [AdmittedCauseV0; 1] {
        [AdmittedCauseV0::external_input(sequence, 0, 0)]
    }

    fn advance(region: &mut Region, scratch: &mut [Scratch; 5], slot: usize, count: u32) {
        for sequence in 0..count {
            region
                .cell_mut(slot)
                .unwrap()
                .transact(&cause(sequence), &[], &mut scratch[slot])
                .unwrap();
        }
    }

    #[test]
    fn no_work_preserves_region_and_coordination_id() {
        let mut region = region();
        let before = region.clone();
        let mut scratch = scratches();
        let empty: [AdmittedCauseV0; 0] = [];
        let work = [
            FrontierWork::new(3, 0, &empty, &[]),
            FrontierWork::new(1, 0, &empty, &[]),
        ];
        let receipt = region.transact_frontier(&work, &mut scratch).unwrap();
        assert_eq!(receipt.disposition, FrontierDisposition::NoWork);
        assert_eq!(receipt.coordination_commit_id, 0);
        assert_eq!(region, before);
    }

    #[test]
    fn four_cell_frontier_preserves_each_local_qbl_choice() {
        let mut region = region();
        let mut scratch = scratches();
        advance(&mut region, &mut scratch, 1, 1);
        advance(&mut region, &mut scratch, 2, 3);
        advance(&mut region, &mut scratch, 3, 14);

        let c0 = cause(100);
        let c1 = cause(101);
        let c2 = cause(102);
        let c3 = cause(103);
        let i0 = [MutationIntentV0::spawn_actor(100, 61, 0)];
        let i1 = [MutationIntentV0::spawn_actor(101, 62, 0)];
        let i2 = [MutationIntentV0::spawn_actor(102, 63, 0)];
        let i3 = [MutationIntentV0::spawn_actor(103, 64, 0)];
        let work = [
            FrontierWork::new(3, 14, &c3, &i3),
            FrontierWork::new(0, 0, &c0, &i0),
            FrontierWork::new(2, 3, &c2, &i2),
            FrontierWork::new(1, 1, &c1, &i1),
        ];

        let receipt = region.transact_frontier(&work, &mut scratch).unwrap();
        assert_eq!(receipt.disposition, FrontierDisposition::Committed);
        assert_eq!(receipt.coordination_commit_id, 1);
        assert_eq!(receipt.participant_count, 4);
        assert_eq!(receipt.participants[0].primitive, QblPrimitive::B);
        assert_eq!(receipt.participants[1].primitive, QblPrimitive::Q);
        assert_eq!(receipt.participants[2].primitive, QblPrimitive::B);
        assert_eq!(receipt.participants[3].primitive, QblPrimitive::L);
        assert_eq!(region.cell(0).unwrap().world().objects().health(ObjectKey(1)), Some(61));
        assert_eq!(region.cell(1).unwrap().world().objects().health(ObjectKey(1)), Some(62));
        assert_eq!(region.cell(2).unwrap().world().objects().health(ObjectKey(1)), Some(63));
        assert_eq!(region.cell(3).unwrap().world().objects().health(ObjectKey(1)), Some(64));
    }

    #[test]
    fn participant_argument_order_is_canonical() {
        let base = region();
        let mut first = base.clone();
        let mut second = base;
        let mut first_scratch = scratches();
        let mut second_scratch = scratches();
        let c0 = cause(0);
        let c1 = cause(1);
        let c2 = cause(2);
        let c3 = cause(3);
        let a = [
            FrontierWork::new(3, 0, &c3, &[]),
            FrontierWork::new(1, 0, &c1, &[]),
            FrontierWork::new(0, 0, &c0, &[]),
            FrontierWork::new(2, 0, &c2, &[]),
        ];
        let b = [
            FrontierWork::new(0, 0, &c0, &[]),
            FrontierWork::new(1, 0, &c1, &[]),
            FrontierWork::new(2, 0, &c2, &[]),
            FrontierWork::new(3, 0, &c3, &[]),
        ];
        let first_receipt = first.transact_frontier(&a, &mut first_scratch).unwrap();
        let second_receipt = second.transact_frontier(&b, &mut second_scratch).unwrap();
        assert_eq!(first_receipt, second_receipt);
        assert_eq!(first, second);
    }

    #[test]
    fn participant_fault_rolls_back_entire_frontier() {
        let mut region = region();
        let before = region.clone();
        let mut scratch = scratches();
        let c0 = cause(0);
        let c1 = cause(1);
        let c2 = cause(2);
        let valid0 = [MutationIntentV0::spawn_actor(0, 100, 0)];
        let valid1 = [MutationIntentV0::spawn_actor(1, 100, 0)];
        let invalid = [MutationIntentV0::apply_health_delta(2, ObjectKey(999), -1)];
        let work = [
            FrontierWork::new(0, 0, &c0, &valid0),
            FrontierWork::new(1, 0, &c1, &valid1),
            FrontierWork::new(2, 0, &c2, &invalid),
        ];
        let result = region.transact_frontier(&work, &mut scratch);
        assert!(matches!(result, Err(FrontierError::Participant { slot: 2, .. })));
        assert_eq!(region, before);
        assert_eq!(region.coordination_commit_id(), 0);
    }

    #[test]
    fn stale_version_rejects_before_any_participant_steps() {
        let mut region = region();
        let before = region.clone();
        let mut scratch = scratches();
        let c0 = cause(0);
        let c1 = cause(1);
        let work = [
            FrontierWork::new(0, 1, &c0, &[]),
            FrontierWork::new(1, 0, &c1, &[]),
        ];
        let result = region.transact_frontier(&work, &mut scratch);
        assert!(matches!(result, Err(FrontierError::StaleVersion { slot: 0, .. })));
        assert_eq!(region, before);
    }

    #[test]
    fn disjoint_frontiers_commute_and_leave_inactive_cell_untouched() {
        let base = region();
        let inactive_before = base.cell(4).unwrap().clone();
        let mut first = base.clone();
        let mut second = base;
        let mut first_scratch = scratches();
        let mut second_scratch = scratches();
        let c0 = cause(0);
        let c1 = cause(1);
        let c2 = cause(2);
        let c3 = cause(3);
        let left = [
            FrontierWork::new(0, 0, &c0, &[]),
            FrontierWork::new(1, 0, &c1, &[]),
        ];
        let right = [
            FrontierWork::new(2, 0, &c2, &[]),
            FrontierWork::new(3, 0, &c3, &[]),
        ];

        first.transact_frontier(&left, &mut first_scratch).unwrap();
        first.transact_frontier(&right, &mut first_scratch).unwrap();
        second.transact_frontier(&right, &mut second_scratch).unwrap();
        second.transact_frontier(&left, &mut second_scratch).unwrap();

        assert_eq!(first, second);
        assert_eq!(first.coordination_commit_id(), 2);
        assert_eq!(first.cell(4).unwrap(), &inactive_before);
    }

    #[test]
    fn duplicate_participant_is_rejected() {
        let mut region = region();
        let before = region.clone();
        let mut scratch = scratches();
        let c0 = cause(0);
        let c1 = cause(1);
        let work = [
            FrontierWork::new(0, 0, &c0, &[]),
            FrontierWork::new(0, 0, &c1, &[]),
        ];
        assert!(matches!(
            region.transact_frontier(&work, &mut scratch),
            Err(FrontierError::DuplicateParticipant { slot: 0, .. })
        ));
        assert_eq!(region, before);
    }
}
