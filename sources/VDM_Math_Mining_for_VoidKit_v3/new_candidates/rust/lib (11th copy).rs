#![no_std]

use qbl_abi::{
    QblPrimitive, QblStatus, QblWideCustodyV0, QblWideLimbDemandV0,
    QblWideTransitionV0, QBL_WIDE_MAX_PAIR_LIMBS, QBL_WIDE_MAX_PRODUCT_LIMBS,
};
use qbl_kernel::step_wide;
use world_core::{
    AdmittedCauseV0, MutationIntentV0, TransitionDisposition, TransitionScratch,
    WorldCore, WorldFault, WorldTransitionError, ZoneCoord,
};

pub const WIDE_AXIS_ACTIVE: u32 = 1;
pub const WIDE_AXIS_LATCHED: u32 = 2;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct WideAxis {
    pub origin_product: [u64; QBL_WIDE_MAX_PRODUCT_LIMBS],
    pub current_product: [u64; QBL_WIDE_MAX_PRODUCT_LIMBS],
    pub phase_quadrant: u32,
    pub flags: u32,
}

const ZERO_AXIS: WideAxis = WideAxis {
    origin_product: [0; QBL_WIDE_MAX_PRODUCT_LIMBS],
    current_product: [0; QBL_WIDE_MAX_PRODUCT_LIMBS],
    phase_quadrant: 0,
    flags: 0,
};

fn wide_is_zero<const N: usize>(value: &[u64; N]) -> bool {
    value.iter().all(|limb| *limb == 0)
}

fn wide_cmp<const N: usize>(left: &[u64; N], right: &[u64; N]) -> core::cmp::Ordering {
    for index in (0..N).rev() {
        match left[index].cmp(&right[index]) {
            core::cmp::Ordering::Equal => {}
            ordering => return ordering,
        }
    }
    core::cmp::Ordering::Equal
}

fn multiply_pair(
    left: &[u64; QBL_WIDE_MAX_PAIR_LIMBS],
    right: &[u64; QBL_WIDE_MAX_PAIR_LIMBS],
) -> [u64; QBL_WIDE_MAX_PRODUCT_LIMBS] {
    let mut output = [0_u64; QBL_WIDE_MAX_PRODUCT_LIMBS];
    for i in 0..QBL_WIDE_MAX_PAIR_LIMBS {
        let mut carry = 0_u128;
        for j in 0..QBL_WIDE_MAX_PAIR_LIMBS {
            let index = i + j;
            let total = output[index] as u128
                + left[i] as u128 * right[j] as u128
                + carry;
            output[index] = total as u64;
            carry = total >> 64;
        }
        let mut index = i + QBL_WIDE_MAX_PAIR_LIMBS;
        while carry != 0 && index < QBL_WIDE_MAX_PRODUCT_LIMBS {
            let total = output[index] as u128 + carry;
            output[index] = total as u64;
            carry = total >> 64;
            index += 1;
        }
    }
    output
}

impl WideAxis {
    const fn identity() -> Self {
        let mut product = [0; QBL_WIDE_MAX_PRODUCT_LIMBS];
        product[0] = 1;
        Self {
            origin_product: product,
            current_product: product,
            phase_quadrant: 0,
            flags: WIDE_AXIS_ACTIVE,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WideDisposition {
    NoWork,
    Committed,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WideError {
    InvalidCapacity,
    ProvisioningRequired { required_pair_limbs: u32 },
    Phase(QblStatus),
    OrthadFull,
    InvalidOrthad,
    World(WorldTransitionError),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct WideResult {
    pub disposition: WideDisposition,
    pub local_commit_id: u64,
    pub primitive: QblPrimitive,
    pub required_pair_limbs: u32,
    pub accepted_intents: u32,
    pub rejected_requests: u32,
    pub phase_fingerprint: u64,
    pub world_fingerprint: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WidePhaseState<const AXES: usize> {
    custody: QblWideCustodyV0,
    pair_limb_limit: u32,
    axis_count: usize,
    axes: [WideAxis; AXES],
}

impl<const AXES: usize> WidePhaseState<AXES> {
    pub fn identity(pair_limb_limit: u32) -> Result<Self, WideError> {
        if AXES == 0 || pair_limb_limit == 0 || pair_limb_limit as usize > QBL_WIDE_MAX_PAIR_LIMBS {
            return Err(WideError::InvalidCapacity);
        }
        let mut axes = [ZERO_AXIS; AXES];
        axes[0] = WideAxis::identity();
        Ok(Self {
            custody: QblWideCustodyV0::identity(),
            pair_limb_limit,
            axis_count: 1,
            axes,
        })
    }

    pub const fn custody(&self) -> &QblWideCustodyV0 { &self.custody }
    pub const fn pair_limb_limit(&self) -> u32 { self.pair_limb_limit }
    pub const fn axis_count(&self) -> usize { self.axis_count }
    pub fn axes(&self) -> &[WideAxis] { &self.axes[..self.axis_count] }

    pub fn from_snapshot_parts(
        custody: QblWideCustodyV0,
        source_axes: &[WideAxis],
    ) -> Result<Self, WideError> {
        if AXES == 0
            || source_axes.is_empty()
            || source_axes.len() > AXES
            || custody.domain > 59
            || source_axes.len() != custody.domain as usize + 1
            || wide_is_zero(&custody.u)
            || wide_is_zero(&custody.v)
            || wide_cmp(&custody.u, &custody.v) == core::cmp::Ordering::Greater
        {
            return Err(WideError::InvalidOrthad);
        }
        let phase_positions = 6_u64
            .checked_shl(custody.domain as u32)
            .ok_or(WideError::InvalidOrthad)?;
        if custody.local_position >= phase_positions {
            return Err(WideError::InvalidOrthad);
        }
        let pair_product = multiply_pair(&custody.u, &custody.v);
        for (index, axis) in source_axes.iter().enumerate() {
            let expected = if index + 1 == source_axes.len() {
                WIDE_AXIS_ACTIVE
            } else {
                WIDE_AXIS_LATCHED
            };
            if axis.flags != expected
                || axis.phase_quadrant > 3
                || wide_is_zero(&axis.origin_product)
                || wide_is_zero(&axis.current_product)
            {
                return Err(WideError::InvalidOrthad);
            }
            if let Some(next) = source_axes.get(index + 1) {
                if axis.current_product != next.origin_product {
                    return Err(WideError::InvalidOrthad);
                }
            }
        }
        let active = source_axes.last().ok_or(WideError::InvalidOrthad)?;
        if active.current_product != pair_product
            || active.phase_quadrant != (custody.local_position & 3) as u32
        {
            return Err(WideError::InvalidOrthad);
        }
        let mut axes = [ZERO_AXIS; AXES];
        axes[..source_axes.len()].copy_from_slice(source_axes);
        Ok(Self {
            custody,
            pair_limb_limit: QBL_WIDE_MAX_PAIR_LIMBS as u32,
            axis_count: source_axes.len(),
            axes,
        })
    }

    pub fn provision_pair_limbs(&mut self, new_limit: u32) -> Result<(), WideError> {
        if new_limit < self.pair_limb_limit || new_limit == 0 ||
            new_limit as usize > QBL_WIDE_MAX_PAIR_LIMBS {
            return Err(WideError::InvalidCapacity);
        }
        self.pair_limb_limit = new_limit;
        Ok(())
    }

    fn apply_one(&mut self) -> Result<QblPrimitive, WideError> {
        let mut next = self.custody;
        let mut transition = QblWideTransitionV0::default();
        let mut demand = QblWideLimbDemandV0::default();
        let status = step_wide(&mut next, self.pair_limb_limit, &mut transition, &mut demand);
        if status == QblStatus::NEEDS_MORE_LIMBS {
            return Err(WideError::ProvisioningRequired {
                required_pair_limbs: demand.required_pair_limbs,
            });
        }
        if !status.is_ok() {
            return Err(WideError::Phase(status));
        }
        let active = self.axis_count.checked_sub(1).ok_or(WideError::InvalidOrthad)?;
        if self.axes[active].flags != WIDE_AXIS_ACTIVE ||
            self.axes[active].current_product != transition.before_product {
            return Err(WideError::InvalidOrthad);
        }
        match transition.primitive {
            1 => self.axes[active].current_product = transition.after_product,
            2 => self.axes[active].phase_quadrant =
                (self.axes[active].phase_quadrant + 1) & 3,
            3 => {
                if self.axis_count == AXES { return Err(WideError::OrthadFull); }
                if self.axes[self.axis_count] != ZERO_AXIS { return Err(WideError::InvalidOrthad); }
                self.axes[active].flags = WIDE_AXIS_LATCHED;
                self.axes[self.axis_count] = WideAxis {
                    origin_product: transition.after_product,
                    current_product: transition.after_product,
                    phase_quadrant: 0,
                    flags: WIDE_AXIS_ACTIVE,
                };
                self.axis_count += 1;
            }
            _ => return Err(WideError::InvalidOrthad),
        }
        self.custody = next;
        Ok(QblPrimitive(transition.primitive))
    }

    pub fn diagnostic_fingerprint64(&self) -> u64 {
        let mut hash = 0xcbf29ce484222325_u64;
        fn feed(hash: &mut u64, bytes: &[u8]) {
            for byte in bytes {
                *hash ^= *byte as u64;
                *hash = hash.wrapping_mul(0x100000001b3);
            }
        }
        for limb in self.custody.u { feed(&mut hash, &limb.to_le_bytes()); }
        for limb in self.custody.v { feed(&mut hash, &limb.to_le_bytes()); }
        feed(&mut hash, &self.custody.domain.to_le_bytes());
        feed(&mut hash, &self.custody.local_position.to_le_bytes());
        feed(&mut hash, &self.custody.quarter_turns.to_le_bytes());
        feed(&mut hash, &(self.axis_count as u64).to_le_bytes());
        for axis in self.axes() {
            for limb in axis.origin_product { feed(&mut hash, &limb.to_le_bytes()); }
            for limb in axis.current_product { feed(&mut hash, &limb.to_le_bytes()); }
            feed(&mut hash, &axis.phase_quadrant.to_le_bytes());
            feed(&mut hash, &axis.flags.to_le_bytes());
        }
        hash
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WidePhaseWorldCell<const AXES: usize, const OBJECTS: usize, const BUCKETS: usize> {
    phase: WidePhaseState<AXES>,
    world: WorldCore<OBJECTS, BUCKETS>,
}

impl<const AXES: usize, const OBJECTS: usize, const BUCKETS: usize>
    WidePhaseWorldCell<AXES, OBJECTS, BUCKETS>
{
    pub fn new(coord: ZoneCoord, pair_limb_limit: u32) -> Result<Self, WideError> {
        Ok(Self {
            phase: WidePhaseState::identity(pair_limb_limit)?,
            world: WorldCore::new(coord)
                .map_err(|fault| WideError::World(WorldTransitionError::Fault(fault)))?,
        })
    }
    pub const fn phase(&self) -> &WidePhaseState<AXES> { &self.phase }
    pub const fn world(&self) -> &WorldCore<OBJECTS, BUCKETS> { &self.world }
    pub fn local_commit_id(&self) -> u64 { self.world.accepted_transition_id() }

    pub fn from_snapshot_parts(
        phase: WidePhaseState<AXES>,
        world: WorldCore<OBJECTS, BUCKETS>,
    ) -> Result<Self, WideError> {
        if phase.custody().domain as usize + 1 != phase.axis_count() {
            return Err(WideError::InvalidOrthad);
        }
        Ok(Self { phase, world })
    }
    pub fn provision_pair_limbs(&mut self, new_limit: u32) -> Result<(), WideError> {
        self.phase.provision_pair_limbs(new_limit)
    }

    pub fn transact<const CAUSES: usize, const INTENTS: usize>(
        &mut self,
        causes: &[AdmittedCauseV0],
        intents: &[MutationIntentV0],
        scratch: &mut TransitionScratch<CAUSES, INTENTS, OBJECTS>,
    ) -> Result<WideResult, WideError> {
        if causes.is_empty() && intents.is_empty() {
            return Ok(WideResult {
                disposition: WideDisposition::NoWork,
                local_commit_id: self.local_commit_id(),
                primitive: QblPrimitive::NONE,
                required_pair_limbs: 0,
                accepted_intents: 0,
                rejected_requests: 0,
                phase_fingerprint: self.phase.diagnostic_fingerprint64(),
                world_fingerprint: self.world.diagnostic_fingerprint64(),
            });
        }
        if causes.is_empty() {
            return Err(WideError::World(WorldTransitionError::Fault(WorldFault::CauseMissing)));
        }
        let mut staged = self.clone();
        let primitive = staged.phase.apply_one()?;
        let world_result = staged.world.transact(causes, intents, scratch)
            .map_err(WideError::World)?;
        if world_result.disposition != TransitionDisposition::Committed {
            return Err(WideError::World(WorldTransitionError::Fault(WorldFault::CauseMissing)));
        }
        let result = WideResult {
            disposition: WideDisposition::Committed,
            local_commit_id: world_result.transition_id,
            primitive,
            required_pair_limbs: 0,
            accepted_intents: world_result.accepted_intents,
            rejected_requests: world_result.rejected_requests,
            phase_fingerprint: staged.phase.diagnostic_fingerprint64(),
            world_fingerprint: world_result.diagnostic_fingerprint,
        };
        *self = staged;
        Ok(result)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct WideFrontierWork<'a> {
    pub slot: usize,
    pub expected_local_commit_id: u64,
    pub causes: &'a [AdmittedCauseV0],
    pub intents: &'a [MutationIntentV0],
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WideFrontierError {
    InvalidSlot,
    DuplicateParticipant,
    StaleVersion,
    CoordinationExhausted,
    Participant { slot: usize, error: WideError },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WideRegion<const CELLS: usize, const AXES: usize, const OBJECTS: usize, const BUCKETS: usize> {
    coordination_commit_id: u64,
    cells: [WidePhaseWorldCell<AXES, OBJECTS, BUCKETS>; CELLS],
}

impl<const CELLS: usize, const AXES: usize, const OBJECTS: usize, const BUCKETS: usize>
    WideRegion<CELLS, AXES, OBJECTS, BUCKETS>
{
    pub fn new(cells: [WidePhaseWorldCell<AXES, OBJECTS, BUCKETS>; CELLS]) -> Self {
        Self { coordination_commit_id: 0, cells }
    }
    pub const fn coordination_commit_id(&self) -> u64 { self.coordination_commit_id }
    pub fn cells(&self) -> &[WidePhaseWorldCell<AXES, OBJECTS, BUCKETS>; CELLS] { &self.cells }
    pub fn cell_mut(&mut self, slot: usize) -> Option<&mut WidePhaseWorldCell<AXES, OBJECTS, BUCKETS>> {
        self.cells.get_mut(slot)
    }

    pub fn transact_frontier<const CAUSES: usize, const INTENTS: usize>(
        &mut self,
        work: &[WideFrontierWork<'_>],
        scratch: &mut [TransitionScratch<CAUSES, INTENTS, OBJECTS>; CELLS],
    ) -> Result<[QblPrimitive; CELLS], WideFrontierError> {
        let mut seen = [false; CELLS];
        let mut any = false;
        for item in work {
            if item.slot >= CELLS { return Err(WideFrontierError::InvalidSlot); }
            if seen[item.slot] { return Err(WideFrontierError::DuplicateParticipant); }
            seen[item.slot] = true;
            if item.expected_local_commit_id != self.cells[item.slot].local_commit_id() {
                return Err(WideFrontierError::StaleVersion);
            }
            any |= !item.causes.is_empty() || !item.intents.is_empty();
        }
        let mut primitives = [QblPrimitive::NONE; CELLS];
        if !any { return Ok(primitives); }
        let next_id = self.coordination_commit_id.checked_add(1)
            .ok_or(WideFrontierError::CoordinationExhausted)?;
        let mut staged = self.cells.clone();
        let mut consumed = [false; CELLS];
        for rank in 0..work.len() {
            let mut selected: Option<usize> = None;
            for (index, item) in work.iter().enumerate() {
                if consumed[index] { continue; }
                selected = match selected {
                    None => Some(index),
                    Some(current) => {
                        let zone = staged[item.slot].world().zone().key;
                        let current_zone = staged[work[current].slot].world().zone().key;
                        if zone < current_zone { Some(index) } else { Some(current) }
                    }
                };
            }
            let index = selected.ok_or(WideFrontierError::InvalidSlot)?;
            consumed[index] = true;
            let item = work[index];
            let result = staged[item.slot].transact(item.causes, item.intents, &mut scratch[item.slot])
                .map_err(|error| WideFrontierError::Participant { slot: item.slot, error })?;
            primitives[rank] = result.primitive;
        }
        self.cells = staged;
        self.coordination_commit_id = next_id;
        Ok(primitives)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type Cell = WidePhaseWorldCell<16, 8, 16>;
    type Scratch = TransitionScratch<8, 16, 8>;

    fn coord(x: i32) -> ZoneCoord { ZoneCoord::new(x, 0, 0).unwrap() }
    fn cause(sequence: u32) -> [AdmittedCauseV0; 1] {
        [AdmittedCauseV0::external_input(sequence, 0, 0)]
    }

    #[test]
    fn provisioning_retry_is_exactly_once() {
        let mut cell = Cell::new(coord(0), 1).unwrap();
        let mut scratch = Scratch::new();
        for sequence in 0..154 {
            cell.transact(&cause(sequence), &[], &mut scratch).unwrap();
        }
        let before = cell.clone();
        let error = cell.transact(&cause(154), &[], &mut scratch).unwrap_err();
        assert_eq!(error, WideError::ProvisioningRequired { required_pair_limbs: 2 });
        assert_eq!(cell, before);
        cell.provision_pair_limbs(2).unwrap();
        let result = cell.transact(&cause(154), &[], &mut scratch).unwrap();
        assert_eq!(result.local_commit_id, 155);
        assert_ne!(cell.phase().custody().v[1], 0);
    }

    #[test]
    fn regional_provisioning_fault_rolls_back_every_participant() {
        let mut first = Cell::new(coord(1), 1).unwrap();
        let second = Cell::new(coord(2), 1).unwrap();
        let mut single_scratch = Scratch::new();
        for sequence in 0..154 {
            first.transact(&cause(sequence), &[], &mut single_scratch).unwrap();
        }
        let mut region = WideRegion::new([first, second]);
        let before = region.clone();
        let causes = [cause(154), cause(0)];
        let work = [
            WideFrontierWork { slot: 0, expected_local_commit_id: 154, causes: &causes[0], intents: &[] },
            WideFrontierWork { slot: 1, expected_local_commit_id: 0, causes: &causes[1], intents: &[] },
        ];
        let mut scratch = [Scratch::new(), Scratch::new()];
        assert_eq!(
            region.transact_frontier(&work, &mut scratch),
            Err(WideFrontierError::Participant {
                slot: 0,
                error: WideError::ProvisioningRequired { required_pair_limbs: 2 },
            })
        );
        assert_eq!(region, before);
        region.cell_mut(0).unwrap().provision_pair_limbs(2).unwrap();
        region.transact_frontier(&work, &mut scratch).unwrap();
        assert_eq!(region.coordination_commit_id(), 1);
        assert_eq!(region.cells()[0].local_commit_id(), 155);
        assert_eq!(region.cells()[1].local_commit_id(), 1);
    }
}
