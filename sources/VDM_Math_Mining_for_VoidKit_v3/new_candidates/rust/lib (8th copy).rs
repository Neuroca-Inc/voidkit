use phase_cell_persistence::{decode, encode, CellSnapshotError, CellSnapshotReceipt, PersistedCell};
use phase_wide::{WideDisposition, WideError, WideResult};
use qbl_abi::QblPrimitive;
use world_core::{
    AdmittedCauseV0, MutationIntentV0, ObjectKey, TransitionScratch, ZoneCoord,
    NON_SPATIAL_SITE,
};

pub const ROUTE_MAX_SLOTS: usize = 4;

#[derive(Debug)]
pub enum RouteError {
    InvalidIdentity,
    InvalidLength,
    DuplicateZone,
    AlreadyBootstrapped,
    NotBootstrapped,
    SlotRange,
    SlotNotResident,
    SlotAlreadyResident,
    SlotNotEmpty,
    ActiveSlot,
    NotAdjacent,
    ActorMissing,
    StaleVersion,
    RouteCommitExhausted,
    Snapshot(CellSnapshotError),
    SnapshotMismatch,
    SourcePhase(WideError),
    DestinationPhase(WideError),
    DestinationSpawnRejected,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RouteDisposition {
    Committed,
    DuplicateIgnored,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RouteReceipt {
    pub disposition: RouteDisposition,
    pub route_commit_id: u64,
    pub source_index: usize,
    pub destination_index: usize,
    pub active_index: usize,
    pub source_primitive: QblPrimitive,
    pub destination_primitive: QblPrimitive,
    pub source_local_commit_id: u64,
    pub destination_local_commit_id: u64,
    pub traveler_key: u64,
    pub actor_key: ObjectKey,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct RouteSlot {
    zone_key: u64,
    cell: Option<PersistedCell>,
    snapshot: Option<Vec<u8>>,
    snapshot_receipt: Option<CellSnapshotReceipt>,
}

impl RouteSlot {
    fn resident(&self) -> bool { self.cell.is_some() }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RetainedRoute {
    route_key: u64,
    traveler_key: u64,
    slots: Vec<RouteSlot>,
    active_index: usize,
    actor_key: Option<ObjectKey>,
    route_commit_id: u64,
    consumed_handoff_cause_id: u64,
    bootstrapped: bool,
}

impl RetainedRoute {
    pub fn new(
        route_key: u64,
        traveler_key: u64,
        zones: &[ZoneCoord],
        pair_limb_limit: u32,
    ) -> Result<Self, RouteError> {
        if route_key == 0 || traveler_key == 0 || pair_limb_limit == 0 {
            return Err(RouteError::InvalidIdentity);
        }
        if zones.len() < 2 || zones.len() > ROUTE_MAX_SLOTS {
            return Err(RouteError::InvalidLength);
        }
        for left in 0..zones.len() {
            for right in left + 1..zones.len() {
                if zones[left].key() == zones[right].key() {
                    return Err(RouteError::DuplicateZone);
                }
            }
        }
        let mut slots = Vec::with_capacity(zones.len());
        for zone in zones {
            let cell = PersistedCell::new(*zone, pair_limb_limit)
                .map_err(RouteError::SourcePhase)?;
            slots.push(RouteSlot {
                zone_key: zone.key().0,
                cell: Some(cell),
                snapshot: None,
                snapshot_receipt: None,
            });
        }
        Ok(Self {
            route_key,
            traveler_key,
            slots,
            active_index: 0,
            actor_key: None,
            route_commit_id: 0,
            consumed_handoff_cause_id: 0,
            bootstrapped: false,
        })
    }

    pub const fn route_key(&self) -> u64 { self.route_key }
    pub const fn traveler_key(&self) -> u64 { self.traveler_key }
    pub fn route_length(&self) -> usize { self.slots.len() }
    pub const fn active_index(&self) -> usize { self.active_index }
    pub const fn route_commit_id(&self) -> u64 { self.route_commit_id }
    pub const fn actor_key(&self) -> Option<ObjectKey> { self.actor_key }
    pub const fn bootstrapped(&self) -> bool { self.bootstrapped }

    pub fn resident(&self, index: usize) -> Result<bool, RouteError> {
        self.slots.get(index).map(RouteSlot::resident).ok_or(RouteError::SlotRange)
    }

    pub fn cell(&self, index: usize) -> Option<&PersistedCell> {
        self.slots.get(index).and_then(|slot| slot.cell.as_ref())
    }

    pub fn snapshot_bytes(&self, index: usize) -> Option<&[u8]> {
        self.slots.get(index).and_then(|slot| slot.snapshot.as_deref())
    }

    pub fn install_snapshot(&mut self, index: usize, bytes: Vec<u8>) -> Result<(), RouteError> {
        let slot = self.slots.get_mut(index).ok_or(RouteError::SlotRange)?;
        if slot.cell.is_some() || slot.snapshot.is_none() {
            return Err(RouteError::SlotAlreadyResident);
        }
        slot.snapshot = Some(bytes);
        Ok(())
    }

    pub fn bootstrap<const CAUSES: usize, const INTENTS: usize>(
        &mut self,
        bootstrap_cause_id: u64,
        source_sequence: u32,
        actor_health: u16,
        scratch: &mut TransitionScratch<CAUSES, INTENTS, 8>,
    ) -> Result<WideResult, RouteError> {
        if bootstrap_cause_id == 0 || source_sequence == 0 || actor_health == 0 {
            return Err(RouteError::InvalidIdentity);
        }
        if self.bootstrapped {
            return Err(RouteError::AlreadyBootstrapped);
        }
        let mut staged = self.clone();
        let cell = staged.slots[0].cell.as_mut().ok_or(RouteError::SlotNotResident)?;
        let expected_key = ObjectKey(cell.world().next_object_key());
        let causes = [AdmittedCauseV0::external_input(
            source_sequence,
            staged.traveler_key,
            bootstrap_cause_id,
        )];
        let intents = [MutationIntentV0::spawn_actor(
            source_sequence,
            actor_health,
            NON_SPATIAL_SITE,
        )];
        let result = cell
            .transact(&causes, &intents, scratch)
            .map_err(RouteError::SourcePhase)?;
        if result.disposition != WideDisposition::Committed
            || result.rejected_requests != 0
            || cell.world().objects().health(expected_key) != Some(actor_health)
        {
            return Err(RouteError::DestinationSpawnRejected);
        }
        staged.actor_key = Some(expected_key);
        staged.bootstrapped = true;
        *self = staged;
        Ok(result)
    }

    pub fn advance_active<const CAUSES: usize, const INTENTS: usize>(
        &mut self,
        source_sequence: u32,
        payload0: u64,
        payload1: u64,
        scratch: &mut TransitionScratch<CAUSES, INTENTS, 8>,
    ) -> Result<WideResult, RouteError> {
        if !self.bootstrapped {
            return Err(RouteError::NotBootstrapped);
        }
        let mut staged = self.clone();
        let causes = [AdmittedCauseV0::external_input(source_sequence, payload0, payload1)];
        let result = staged.slots[staged.active_index]
            .cell
            .as_mut()
            .ok_or(RouteError::SlotNotResident)?
            .transact(&causes, &[], scratch)
            .map_err(RouteError::SourcePhase)?;
        *self = staged;
        Ok(result)
    }

    pub fn evict_slot(&mut self, index: usize) -> Result<CellSnapshotReceipt, RouteError> {
        if index >= self.slots.len() {
            return Err(RouteError::SlotRange);
        }
        if index == self.active_index {
            return Err(RouteError::ActiveSlot);
        }
        let cell = self.slots[index].cell.as_ref().ok_or(RouteError::SlotNotResident)?;
        if cell.world().objects().alive_count() != 0 {
            return Err(RouteError::SlotNotEmpty);
        }
        let (bytes, receipt) = encode(cell).map_err(RouteError::Snapshot)?;
        let mut staged = self.clone();
        staged.slots[index].cell = None;
        staged.slots[index].snapshot = Some(bytes);
        staged.slots[index].snapshot_receipt = Some(receipt);
        *self = staged;
        Ok(receipt)
    }

    pub fn restore_slot(&mut self, index: usize) -> Result<CellSnapshotReceipt, RouteError> {
        let slot = self.slots.get(index).ok_or(RouteError::SlotRange)?;
        if slot.cell.is_some() {
            return Err(RouteError::SlotAlreadyResident);
        }
        let expected = slot.snapshot_receipt.ok_or(RouteError::SnapshotMismatch)?;
        let bytes = slot.snapshot.as_ref().ok_or(RouteError::SnapshotMismatch)?;
        let (cell, receipt) = decode(bytes).map_err(RouteError::Snapshot)?;
        if receipt != expected || receipt.source_zone_key != slot.zone_key {
            return Err(RouteError::SnapshotMismatch);
        }
        let mut staged = self.clone();
        staged.slots[index].cell = Some(cell);
        staged.slots[index].snapshot = None;
        staged.slots[index].snapshot_receipt = None;
        *self = staged;
        Ok(receipt)
    }

    pub fn provision_slot_pair_limbs(
        &mut self,
        index: usize,
        new_limit: u32,
    ) -> Result<(), RouteError> {
        self.slots
            .get_mut(index)
            .ok_or(RouteError::SlotRange)?
            .cell
            .as_mut()
            .ok_or(RouteError::SlotNotResident)?
            .provision_pair_limbs(new_limit)
            .map_err(RouteError::SourcePhase)
    }

    pub fn handoff<
        const SOURCE_CAUSES: usize,
        const SOURCE_INTENTS: usize,
        const DEST_CAUSES: usize,
        const DEST_INTENTS: usize,
    >(
        &mut self,
        handoff_cause_id: u64,
        destination_index: usize,
        expected_source_local_commit_id: u64,
        expected_destination_local_commit_id: u64,
        source_sequence: u32,
        destination_sequence: u32,
        source_scratch: &mut TransitionScratch<SOURCE_CAUSES, SOURCE_INTENTS, 8>,
        destination_scratch: &mut TransitionScratch<DEST_CAUSES, DEST_INTENTS, 8>,
    ) -> Result<RouteReceipt, RouteError> {
        if !self.bootstrapped {
            return Err(RouteError::NotBootstrapped);
        }
        if handoff_cause_id == 0 || source_sequence == 0 || destination_sequence == 0 {
            return Err(RouteError::InvalidIdentity);
        }
        if self.consumed_handoff_cause_id == handoff_cause_id {
            let actor_key = self.actor_key.ok_or(RouteError::ActorMissing)?;
            return Ok(RouteReceipt {
                disposition: RouteDisposition::DuplicateIgnored,
                route_commit_id: self.route_commit_id,
                source_index: self.active_index,
                destination_index: self.active_index,
                active_index: self.active_index,
                source_primitive: QblPrimitive::NONE,
                destination_primitive: QblPrimitive::NONE,
                source_local_commit_id: self.slots[self.active_index]
                    .cell
                    .as_ref()
                    .ok_or(RouteError::SlotNotResident)?
                    .local_commit_id(),
                destination_local_commit_id: self.slots[self.active_index]
                    .cell
                    .as_ref()
                    .ok_or(RouteError::SlotNotResident)?
                    .local_commit_id(),
                traveler_key: self.traveler_key,
                actor_key,
            });
        }
        if destination_index >= self.slots.len() {
            return Err(RouteError::SlotRange);
        }
        let source_index = self.active_index;
        if destination_index == source_index || destination_index.abs_diff(source_index) != 1 {
            return Err(RouteError::NotAdjacent);
        }
        let source = self.slots[source_index]
            .cell
            .as_ref()
            .ok_or(RouteError::SlotNotResident)?;
        let destination = self.slots[destination_index]
            .cell
            .as_ref()
            .ok_or(RouteError::SlotNotResident)?;
        if source.local_commit_id() != expected_source_local_commit_id
            || destination.local_commit_id() != expected_destination_local_commit_id
        {
            return Err(RouteError::StaleVersion);
        }
        let actor_key = self.actor_key.ok_or(RouteError::ActorMissing)?;
        let health = source
            .world()
            .objects()
            .health(actor_key)
            .ok_or(RouteError::ActorMissing)?;
        let next_commit_id = self
            .route_commit_id
            .checked_add(1)
            .ok_or(RouteError::RouteCommitExhausted)?;
        let mut staged = self.clone();

        let source_causes = [AdmittedCauseV0::external_input(
            source_sequence,
            staged.traveler_key,
            staged.slots[destination_index].zone_key,
        )];
        let source_intents = [MutationIntentV0::despawn(source_sequence, actor_key)];
        let source_result = staged.slots[source_index]
            .cell
            .as_mut()
            .ok_or(RouteError::SlotNotResident)?
            .transact(&source_causes, &source_intents, source_scratch)
            .map_err(RouteError::SourcePhase)?;
        if source_result.disposition != WideDisposition::Committed
            || source_result.rejected_requests != 0
        {
            return Err(RouteError::ActorMissing);
        }

        let expected_destination_key = ObjectKey(
            staged.slots[destination_index]
                .cell
                .as_ref()
                .ok_or(RouteError::SlotNotResident)?
                .world()
                .next_object_key(),
        );
        let destination_causes = [AdmittedCauseV0::external_input(
            destination_sequence,
            staged.traveler_key,
            handoff_cause_id,
        )];
        let destination_intents = [MutationIntentV0::spawn_actor(
            destination_sequence,
            health,
            NON_SPATIAL_SITE,
        )];
        let destination_result = staged.slots[destination_index]
            .cell
            .as_mut()
            .ok_or(RouteError::SlotNotResident)?
            .transact(&destination_causes, &destination_intents, destination_scratch)
            .map_err(RouteError::DestinationPhase)?;
        let destination_cell = staged.slots[destination_index]
            .cell
            .as_ref()
            .ok_or(RouteError::SlotNotResident)?;
        if destination_result.disposition != WideDisposition::Committed
            || destination_result.rejected_requests != 0
            || destination_cell.world().objects().health(expected_destination_key) != Some(health)
        {
            return Err(RouteError::DestinationSpawnRejected);
        }

        staged.active_index = destination_index;
        staged.actor_key = Some(expected_destination_key);
        staged.consumed_handoff_cause_id = handoff_cause_id;
        staged.route_commit_id = next_commit_id;
        let receipt = RouteReceipt {
            disposition: RouteDisposition::Committed,
            route_commit_id: next_commit_id,
            source_index,
            destination_index,
            active_index: destination_index,
            source_primitive: source_result.primitive,
            destination_primitive: destination_result.primitive,
            source_local_commit_id: source_result.local_commit_id,
            destination_local_commit_id: destination_result.local_commit_id,
            traveler_key: staged.traveler_key,
            actor_key: expected_destination_key,
        };
        *self = staged;
        Ok(receipt)
    }

    pub fn diagnostic_fingerprint64(&self) -> u64 {
        let mut hash = 0xcbf29ce484222325_u64;
        fn feed(hash: &mut u64, bytes: &[u8]) {
            for byte in bytes {
                *hash ^= *byte as u64;
                *hash = hash.wrapping_mul(0x100000001b3);
            }
        }
        feed(&mut hash, &self.route_key.to_le_bytes());
        feed(&mut hash, &self.traveler_key.to_le_bytes());
        feed(&mut hash, &self.route_commit_id.to_le_bytes());
        feed(&mut hash, &self.consumed_handoff_cause_id.to_le_bytes());
        feed(&mut hash, &(self.active_index as u64).to_le_bytes());
        feed(&mut hash, &self.actor_key.unwrap_or(ObjectKey(0)).0.to_le_bytes());
        feed(&mut hash, &[self.bootstrapped as u8]);
        for slot in &self.slots {
            feed(&mut hash, &slot.zone_key.to_le_bytes());
            feed(&mut hash, &[slot.resident() as u8]);
            if let Some(cell) = &slot.cell {
                feed(&mut hash, &cell.phase().diagnostic_fingerprint64().to_le_bytes());
                feed(&mut hash, &cell.world().diagnostic_fingerprint64().to_le_bytes());
            } else if let Some(bytes) = &slot.snapshot {
                feed(&mut hash, bytes);
            }
        }
        hash
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use phase_wide::WideError;

    type Scratch = TransitionScratch<8, 16, 8>;

    fn route(limit: u32, health: u16) -> RetainedRoute {
        let zones = [
            ZoneCoord::new(0, 0, 0).unwrap(),
            ZoneCoord::new(1, 0, 0).unwrap(),
            ZoneCoord::new(2, 0, 0).unwrap(),
        ];
        let mut route = RetainedRoute::new(71, 9001, &zones, limit).unwrap();
        let mut scratch = Scratch::new();
        route.bootstrap(100, 1, health, &mut scratch).unwrap();
        route
    }

    fn active_health(route: &RetainedRoute) -> u16 {
        let key = route.actor_key().unwrap();
        route.cell(route.active_index()).unwrap().world().objects().health(key).unwrap()
    }

    #[test]
    fn bounded_route_advances_with_nonactive_members_nonresident() {
        let mut route = route(4, 91);
        let slot2_phase = route.cell(2).unwrap().phase().diagnostic_fingerprint64();
        let slot2_world = route.cell(2).unwrap().world().diagnostic_fingerprint64();
        route.evict_slot(2).unwrap();
        let mut source = Scratch::new();
        let mut destination = Scratch::new();
        route.handoff(
            201,
            1,
            route.cell(0).unwrap().local_commit_id(),
            route.cell(1).unwrap().local_commit_id(),
            2,
            1,
            &mut source,
            &mut destination,
        ).unwrap();
        route.advance_active(2, 11, 22, &mut source).unwrap();
        route.restore_slot(2).unwrap();
        assert_eq!(route.cell(2).unwrap().phase().diagnostic_fingerprint64(), slot2_phase);
        assert_eq!(route.cell(2).unwrap().world().diagnostic_fingerprint64(), slot2_world);
        route.handoff(
            202,
            2,
            route.cell(1).unwrap().local_commit_id(),
            route.cell(2).unwrap().local_commit_id(),
            3,
            1,
            &mut source,
            &mut destination,
        ).unwrap();
        assert_eq!(route.route_commit_id(), 2);
        assert_eq!(active_health(&route), 91);
        route.evict_slot(0).unwrap();
        route.evict_slot(1).unwrap();
        route.advance_active(2, 33, 44, &mut source).unwrap();
        assert!(!route.resident(0).unwrap());
        assert!(!route.resident(1).unwrap());
        assert!(route.resident(2).unwrap());
    }

    #[test]
    fn stale_guard_precedes_pair_mutation_and_nonparticipant_is_untouched() {
        let mut route = route(4, 77);
        let before = route.clone();
        let slot2 = route.cell(2).unwrap().clone();
        let mut source = Scratch::new();
        let mut destination = Scratch::new();
        assert!(matches!(
            route.handoff(
                301,
                1,
                route.cell(0).unwrap().local_commit_id() + 1,
                route.cell(1).unwrap().local_commit_id(),
                2,
                1,
                &mut source,
                &mut destination,
            ),
            Err(RouteError::StaleVersion)
        ));
        assert_eq!(route, before);
        route.handoff(
            301,
            1,
            route.cell(0).unwrap().local_commit_id(),
            route.cell(1).unwrap().local_commit_id(),
            2,
            1,
            &mut source,
            &mut destination,
        ).unwrap();
        assert_eq!(route.cell(2).unwrap(), &slot2);
    }

    #[test]
    fn corrupt_retained_slot_snapshot_rejection_is_atomic() {
        let mut route = route(4, 88);
        route.evict_slot(2).unwrap();
        let mut bytes = route.snapshot_bytes(2).unwrap().to_vec();
        bytes[9] ^= 0x40;
        route.install_snapshot(2, bytes).unwrap();
        let before = route.clone();
        assert!(matches!(route.restore_slot(2), Err(RouteError::Snapshot(_))));
        assert_eq!(route, before);
    }

    #[test]
    fn provisioning_fault_rolls_back_route_and_retries_same_cause() {
        let mut route = route(1, 66);
        let mut source = Scratch::new();
        let mut destination = Scratch::new();
        for sequence in 2..=154 {
            route.advance_active(sequence, sequence as u64, 0, &mut source).unwrap();
        }
        let before = route.clone();
        assert!(matches!(
            route.handoff(
                401,
                1,
                route.cell(0).unwrap().local_commit_id(),
                route.cell(1).unwrap().local_commit_id(),
                155,
                1,
                &mut source,
                &mut destination,
            ),
            Err(RouteError::SourcePhase(WideError::ProvisioningRequired {
                required_pair_limbs: 2
            }))
        ));
        assert_eq!(route, before);
        route.provision_slot_pair_limbs(0, 2).unwrap();
        let receipt = route.handoff(
            401,
            1,
            route.cell(0).unwrap().local_commit_id(),
            route.cell(1).unwrap().local_commit_id(),
            155,
            1,
            &mut source,
            &mut destination,
        ).unwrap();
        assert_eq!(receipt.route_commit_id, 1);
        assert_eq!(active_health(&route), 66);
    }

    #[test]
    fn duplicate_handoff_is_idempotent() {
        let mut route = route(4, 91);
        let mut source = Scratch::new();
        let mut destination = Scratch::new();
        route.handoff(
            501,
            1,
            route.cell(0).unwrap().local_commit_id(),
            route.cell(1).unwrap().local_commit_id(),
            2,
            1,
            &mut source,
            &mut destination,
        ).unwrap();
        let before = route.clone();
        let duplicate = route.handoff(
            501,
            1,
            route.cell(1).unwrap().local_commit_id(),
            route.cell(1).unwrap().local_commit_id(),
            2,
            1,
            &mut source,
            &mut destination,
        ).unwrap();
        assert_eq!(duplicate.disposition, RouteDisposition::DuplicateIgnored);
        assert_eq!(route, before);
    }

    #[test]
    fn deterministic_replay_reproduces_route_custody() {
        let mut left = route(4, 91);
        let mut right = route(4, 91);
        let mut ls = Scratch::new();
        let mut ld = Scratch::new();
        let mut rs = Scratch::new();
        let mut rd = Scratch::new();
        left.evict_slot(2).unwrap();
        right.evict_slot(2).unwrap();
        left.handoff(601, 1, left.cell(0).unwrap().local_commit_id(), left.cell(1).unwrap().local_commit_id(), 2, 1, &mut ls, &mut ld).unwrap();
        right.handoff(601, 1, right.cell(0).unwrap().local_commit_id(), right.cell(1).unwrap().local_commit_id(), 2, 1, &mut rs, &mut rd).unwrap();
        left.advance_active(2, 7, 8, &mut ls).unwrap();
        right.advance_active(2, 7, 8, &mut rs).unwrap();
        left.restore_slot(2).unwrap();
        right.restore_slot(2).unwrap();
        left.handoff(602, 2, left.cell(1).unwrap().local_commit_id(), left.cell(2).unwrap().local_commit_id(), 3, 1, &mut ls, &mut ld).unwrap();
        right.handoff(602, 2, right.cell(1).unwrap().local_commit_id(), right.cell(2).unwrap().local_commit_id(), 3, 1, &mut rs, &mut rd).unwrap();
        assert_eq!(left, right);
        assert_eq!(left.diagnostic_fingerprint64(), right.diagnostic_fingerprint64());
    }
}
