use phase_cell_persistence::{
    decode, encode, CellSnapshotError, CellSnapshotReceipt, PersistedCell,
};
use phase_traversal::StreamedZoneTraversal;
use phase_wide::{WideDisposition, WideError, WideResult};
use qbl_abi::QblPrimitive;
use world_core::{
    AdmittedCauseV0, MutationIntentV0, ObjectKey, TransitionScratch, NON_SPATIAL_SITE,
};

pub type CompletedTraversal = StreamedZoneTraversal<16, 8, 16, 8, 8, 8>;

#[derive(Debug)]
pub enum ReturnError {
    InvalidCompletedTraversal,
    OriginNotEmpty,
    AlreadyEvicted,
    NotEvicted,
    Snapshot(CellSnapshotError),
    SnapshotMismatch,
    OriginNotResident,
    AlreadyReturned,
    ActiveActorMissing,
    ReturnCommitExhausted,
    ActivePhase(WideError),
    OriginPhase(WideError),
    OriginSpawnRejected,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReturnDisposition {
    Committed,
    DuplicateIgnored,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EvictionReceipt {
    pub snapshot_bytes: usize,
    pub origin_zone_key: u64,
    pub origin_local_commit_id: u64,
    pub phase_fingerprint: u64,
    pub world_fingerprint: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RestoreReceipt {
    pub origin_zone_key: u64,
    pub origin_local_commit_id: u64,
    pub phase_fingerprint: u64,
    pub world_fingerprint: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ReturnReceipt {
    pub disposition: ReturnDisposition,
    pub return_commit_id: u64,
    pub active_primitive: QblPrimitive,
    pub origin_primitive: QblPrimitive,
    pub active_local_commit_id: u64,
    pub origin_local_commit_id: u64,
    pub return_cause_id: u64,
    pub traveler_key: u64,
    pub active_actor_key: ObjectKey,
    pub return_actor_key: ObjectKey,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReturnJourney {
    region_key: u64,
    traveler_key: u64,
    origin_zone_key: u64,
    active_zone_key: u64,
    outbound_traversal_commit_id: u64,
    origin: Option<PersistedCell>,
    active: PersistedCell,
    origin_snapshot: Option<Vec<u8>>,
    origin_snapshot_receipt: Option<CellSnapshotReceipt>,
    active_actor_key: ObjectKey,
    return_actor_key: Option<ObjectKey>,
    consumed_return_cause_id: u64,
    return_commit_id: u64,
    returned: bool,
}

impl ReturnJourney {
    pub fn from_completed(completed: &CompletedTraversal) -> Result<Self, ReturnError> {
        if !completed.traversed()
            || completed.traversal_commit_id() == 0
            || completed.source().world().objects().alive_count() != 0
            || completed.destination().world().objects().alive_count() != 1
        {
            return Err(ReturnError::InvalidCompletedTraversal);
        }
        let active_actor_key = completed
            .destination_actor_key()
            .ok_or(ReturnError::ActiveActorMissing)?;
        if completed
            .destination()
            .world()
            .objects()
            .health(active_actor_key)
            .is_none()
        {
            return Err(ReturnError::ActiveActorMissing);
        }
        Ok(Self {
            region_key: completed.region_key(),
            traveler_key: completed.traveler_key(),
            origin_zone_key: completed.source().world().zone().key.0,
            active_zone_key: completed.destination().world().zone().key.0,
            outbound_traversal_commit_id: completed.traversal_commit_id(),
            origin: Some(completed.source().clone()),
            active: completed.destination().clone(),
            origin_snapshot: None,
            origin_snapshot_receipt: None,
            active_actor_key,
            return_actor_key: None,
            consumed_return_cause_id: 0,
            return_commit_id: 0,
            returned: false,
        })
    }

    pub const fn region_key(&self) -> u64 { self.region_key }
    pub const fn traveler_key(&self) -> u64 { self.traveler_key }
    pub const fn origin_zone_key(&self) -> u64 { self.origin_zone_key }
    pub const fn active_zone_key(&self) -> u64 { self.active_zone_key }
    pub const fn outbound_traversal_commit_id(&self) -> u64 {
        self.outbound_traversal_commit_id
    }
    pub const fn origin(&self) -> Option<&PersistedCell> { self.origin.as_ref() }
    pub const fn active(&self) -> &PersistedCell { &self.active }
    pub const fn origin_resident(&self) -> bool { self.origin.is_some() }
    pub const fn origin_evicted(&self) -> bool { self.origin.is_none() && self.origin_snapshot.is_some() }
    pub const fn returned(&self) -> bool { self.returned }
    pub const fn return_commit_id(&self) -> u64 { self.return_commit_id }
    pub const fn active_actor_key(&self) -> ObjectKey { self.active_actor_key }
    pub const fn return_actor_key(&self) -> Option<ObjectKey> { self.return_actor_key }
    pub fn origin_snapshot_bytes(&self) -> Option<&[u8]> { self.origin_snapshot.as_deref() }

    pub fn evict_origin(&mut self) -> Result<EvictionReceipt, ReturnError> {
        if self.returned {
            return Err(ReturnError::AlreadyReturned);
        }
        let origin = self.origin.as_ref().ok_or(ReturnError::AlreadyEvicted)?;
        if origin.world().objects().alive_count() != 0 {
            return Err(ReturnError::OriginNotEmpty);
        }
        let (bytes, receipt) = encode(origin).map_err(ReturnError::Snapshot)?;
        let output = EvictionReceipt {
            snapshot_bytes: receipt.snapshot_bytes,
            origin_zone_key: receipt.source_zone_key,
            origin_local_commit_id: receipt.source_local_commit_id,
            phase_fingerprint: receipt.phase_fingerprint,
            world_fingerprint: receipt.world_fingerprint,
        };
        let mut staged = self.clone();
        staged.origin = None;
        staged.origin_snapshot = Some(bytes);
        staged.origin_snapshot_receipt = Some(receipt);
        *self = staged;
        Ok(output)
    }

    pub fn install_origin_snapshot(&mut self, bytes: Vec<u8>) -> Result<(), ReturnError> {
        if self.origin.is_some() || self.origin_snapshot.is_none() {
            return Err(ReturnError::NotEvicted);
        }
        self.origin_snapshot = Some(bytes);
        Ok(())
    }

    pub fn restore_origin(&mut self) -> Result<RestoreReceipt, ReturnError> {
        if self.origin.is_some() || self.origin_snapshot.is_none() {
            return Err(ReturnError::NotEvicted);
        }
        let expected = self
            .origin_snapshot_receipt
            .ok_or(ReturnError::SnapshotMismatch)?;
        let bytes = self
            .origin_snapshot
            .as_ref()
            .ok_or(ReturnError::NotEvicted)?;
        let (cell, receipt) = decode(bytes).map_err(ReturnError::Snapshot)?;
        if receipt != expected || receipt.source_zone_key != self.origin_zone_key {
            return Err(ReturnError::SnapshotMismatch);
        }
        let output = RestoreReceipt {
            origin_zone_key: receipt.source_zone_key,
            origin_local_commit_id: receipt.source_local_commit_id,
            phase_fingerprint: receipt.phase_fingerprint,
            world_fingerprint: receipt.world_fingerprint,
        };
        let mut staged = self.clone();
        staged.origin = Some(cell);
        staged.origin_snapshot = None;
        staged.origin_snapshot_receipt = None;
        *self = staged;
        Ok(output)
    }

    pub fn advance_active<const CAUSES: usize, const INTENTS: usize>(
        &mut self,
        source_sequence: u32,
        payload0: u64,
        payload1: u64,
        scratch: &mut TransitionScratch<CAUSES, INTENTS, 8>,
    ) -> Result<WideResult, ReturnError> {
        if self.returned {
            return Err(ReturnError::AlreadyReturned);
        }
        let mut staged = self.clone();
        let causes = [AdmittedCauseV0::external_input(source_sequence, payload0, payload1)];
        let result = staged
            .active
            .transact(&causes, &[], scratch)
            .map_err(ReturnError::ActivePhase)?;
        *self = staged;
        Ok(result)
    }

    pub fn provision_active_pair_limbs(&mut self, new_limit: u32) -> Result<(), ReturnError> {
        self.active
            .provision_pair_limbs(new_limit)
            .map_err(ReturnError::ActivePhase)
    }

    pub fn return_to_origin<
        const ACTIVE_CAUSES: usize,
        const ACTIVE_INTENTS: usize,
        const ORIGIN_CAUSES: usize,
        const ORIGIN_INTENTS: usize,
    >(
        &mut self,
        return_cause_id: u64,
        active_sequence: u32,
        origin_sequence: u32,
        active_scratch: &mut TransitionScratch<ACTIVE_CAUSES, ACTIVE_INTENTS, 8>,
        origin_scratch: &mut TransitionScratch<ORIGIN_CAUSES, ORIGIN_INTENTS, 8>,
    ) -> Result<ReturnReceipt, ReturnError> {
        if self.returned {
            if self.consumed_return_cause_id == return_cause_id {
                return Ok(ReturnReceipt {
                    disposition: ReturnDisposition::DuplicateIgnored,
                    return_commit_id: self.return_commit_id,
                    active_primitive: QblPrimitive::NONE,
                    origin_primitive: QblPrimitive::NONE,
                    active_local_commit_id: self.active.local_commit_id(),
                    origin_local_commit_id: self
                        .origin
                        .as_ref()
                        .ok_or(ReturnError::OriginNotResident)?
                        .local_commit_id(),
                    return_cause_id,
                    traveler_key: self.traveler_key,
                    active_actor_key: self.active_actor_key,
                    return_actor_key: self.return_actor_key.ok_or(ReturnError::ActiveActorMissing)?,
                });
            }
            return Err(ReturnError::AlreadyReturned);
        }
        let origin = self.origin.as_ref().ok_or(ReturnError::OriginNotResident)?;
        let health = self
            .active
            .world()
            .objects()
            .health(self.active_actor_key)
            .ok_or(ReturnError::ActiveActorMissing)?;
        let next_commit_id = self
            .return_commit_id
            .checked_add(1)
            .ok_or(ReturnError::ReturnCommitExhausted)?;
        let expected_origin_key = ObjectKey(origin.world().next_object_key());
        let mut staged = self.clone();

        let active_causes = [AdmittedCauseV0::external_input(
            active_sequence,
            staged.traveler_key,
            staged.origin_zone_key,
        )];
        let active_intents = [MutationIntentV0::despawn(
            active_sequence,
            staged.active_actor_key,
        )];
        let active_result = staged
            .active
            .transact(&active_causes, &active_intents, active_scratch)
            .map_err(ReturnError::ActivePhase)?;
        if active_result.disposition != WideDisposition::Committed
            || active_result.rejected_requests != 0
        {
            return Err(ReturnError::ActiveActorMissing);
        }

        let origin_causes = [AdmittedCauseV0::external_input(
            origin_sequence,
            staged.traveler_key,
            return_cause_id,
        )];
        let origin_intents = [MutationIntentV0::spawn_actor(
            origin_sequence,
            health,
            NON_SPATIAL_SITE,
        )];
        let origin_result = staged
            .origin
            .as_mut()
            .ok_or(ReturnError::OriginNotResident)?
            .transact(&origin_causes, &origin_intents, origin_scratch)
            .map_err(ReturnError::OriginPhase)?;
        let staged_origin = staged.origin.as_ref().ok_or(ReturnError::OriginNotResident)?;
        if origin_result.disposition != WideDisposition::Committed
            || origin_result.rejected_requests != 0
            || staged_origin.world().objects().health(expected_origin_key) != Some(health)
        {
            return Err(ReturnError::OriginSpawnRejected);
        }

        staged.return_actor_key = Some(expected_origin_key);
        staged.consumed_return_cause_id = return_cause_id;
        staged.return_commit_id = next_commit_id;
        staged.returned = true;
        let receipt = ReturnReceipt {
            disposition: ReturnDisposition::Committed,
            return_commit_id: next_commit_id,
            active_primitive: active_result.primitive,
            origin_primitive: origin_result.primitive,
            active_local_commit_id: active_result.local_commit_id,
            origin_local_commit_id: origin_result.local_commit_id,
            return_cause_id,
            traveler_key: staged.traveler_key,
            active_actor_key: staged.active_actor_key,
            return_actor_key: expected_origin_key,
        };
        *self = staged;
        Ok(receipt)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use phase_completions::{CompletionInput, CompletionResult};
    use phase_traversal::TraversalDisposition;
    use world_core::ZoneCoord;

    type Scratch = TransitionScratch<8, 16, 8>;

    fn completed(destination_limit: u32, health: u16, hash: u64) -> CompletedTraversal {
        let mut traversal = CompletedTraversal::new(
            7,
            9001,
            ZoneCoord::new(0, 0, 0).unwrap(),
            ZoneCoord::new(1, 0, 0).unwrap(),
            200,
            4,
            0,
            4,
            destination_limit,
        )
        .unwrap();
        let mut source_scratch = Scratch::new();
        let mut destination_scratch = Scratch::new();
        traversal
            .bootstrap_source(1, health, &mut source_scratch)
            .unwrap();
        traversal
            .request_destination(1, &mut destination_scratch)
            .unwrap();
        let admission = traversal
            .admit_completion(CompletionInput {
                effect_id: traversal.stream_effect_id(),
                asset_key: 200,
                content_version: 4,
                variant: 0,
                result: CompletionResult::Ready,
                content_hash: hash,
                byte_length: 8192,
            })
            .unwrap();
        let receipt = traversal
            .traverse(
                admission.record.admission_id,
                2,
                2,
                &mut source_scratch,
                &mut destination_scratch,
            )
            .unwrap();
        assert_eq!(receipt.disposition, TraversalDisposition::Committed);
        traversal
    }

    #[test]
    fn origin_can_be_evicted_while_active_zone_continues() {
        let traversal = completed(4, 91, 0xaa);
        let mut journey = ReturnJourney::from_completed(&traversal).unwrap();
        let origin_phase = journey.origin().unwrap().phase().diagnostic_fingerprint64();
        let origin_world = journey.origin().unwrap().world().diagnostic_fingerprint64();
        let active_before = journey.active().clone();
        let eviction = journey.evict_origin().unwrap();
        assert!(journey.origin_evicted());
        assert_eq!(journey.active(), &active_before);
        let mut scratch = Scratch::new();
        journey.advance_active(3, 11, 22, &mut scratch).unwrap();
        assert_eq!(journey.active().local_commit_id(), active_before.local_commit_id() + 1);
        let restored = journey.restore_origin().unwrap();
        assert_eq!(restored.phase_fingerprint, origin_phase);
        assert_eq!(restored.world_fingerprint, origin_world);
        assert_eq!(eviction.origin_zone_key, restored.origin_zone_key);
    }

    #[test]
    fn corrupt_snapshot_rejection_is_atomic() {
        let traversal = completed(4, 91, 0xbb);
        let mut journey = ReturnJourney::from_completed(&traversal).unwrap();
        journey.evict_origin().unwrap();
        let mut bytes = journey.origin_snapshot_bytes().unwrap().to_vec();
        bytes[9] ^= 0x40;
        journey.install_origin_snapshot(bytes).unwrap();
        let before = journey.clone();
        assert!(matches!(journey.restore_origin(), Err(ReturnError::Snapshot(_))));
        assert_eq!(journey, before);
        assert!(!journey.origin_resident());
    }

    #[test]
    fn explicit_return_preserves_health_and_traveler_identity() {
        let traversal = completed(4, 91, 0xcc);
        let mut journey = ReturnJourney::from_completed(&traversal).unwrap();
        journey.evict_origin().unwrap();
        journey.restore_origin().unwrap();
        let mut active_scratch = Scratch::new();
        let mut origin_scratch = Scratch::new();
        let receipt = journey
            .return_to_origin(77, 3, 3, &mut active_scratch, &mut origin_scratch)
            .unwrap();
        assert_eq!(receipt.disposition, ReturnDisposition::Committed);
        assert_ne!(receipt.active_primitive, QblPrimitive::NONE);
        assert_ne!(receipt.origin_primitive, QblPrimitive::NONE);
        assert_eq!(receipt.traveler_key, 9001);
        assert_eq!(journey.active().world().objects().alive_count(), 0);
        assert_eq!(
            journey
                .origin()
                .unwrap()
                .world()
                .objects()
                .health(receipt.return_actor_key),
            Some(91)
        );
    }

    #[test]
    fn provisioning_fault_rolls_back_and_retries_same_return_cause() {
        let traversal = completed(1, 77, 0xdd);
        let mut journey = ReturnJourney::from_completed(&traversal).unwrap();
        journey.evict_origin().unwrap();
        let mut active_scratch = Scratch::new();
        for sequence in 3..=154 {
            journey
                .advance_active(sequence, sequence as u64, 0, &mut active_scratch)
                .unwrap();
        }
        journey.restore_origin().unwrap();
        let before = journey.clone();
        let mut origin_scratch = Scratch::new();
        assert!(matches!(
            journey.return_to_origin(88, 155, 3, &mut active_scratch, &mut origin_scratch),
            Err(ReturnError::ActivePhase(WideError::ProvisioningRequired {
                required_pair_limbs: 2
            }))
        ));
        assert_eq!(journey, before);
        let origin_phase = journey.origin().unwrap().phase().diagnostic_fingerprint64();
        let origin_world = journey.origin().unwrap().world().diagnostic_fingerprint64();
        let active_phase = journey.active().phase().diagnostic_fingerprint64();
        let active_world = journey.active().world().diagnostic_fingerprint64();
        journey.provision_active_pair_limbs(2).unwrap();
        assert_eq!(journey.origin().unwrap().phase().diagnostic_fingerprint64(), origin_phase);
        assert_eq!(journey.origin().unwrap().world().diagnostic_fingerprint64(), origin_world);
        assert_eq!(journey.active().phase().diagnostic_fingerprint64(), active_phase);
        assert_eq!(journey.active().world().diagnostic_fingerprint64(), active_world);
        let receipt = journey
            .return_to_origin(88, 155, 3, &mut active_scratch, &mut origin_scratch)
            .unwrap();
        assert_eq!(receipt.return_commit_id, 1);
    }

    #[test]
    fn duplicate_return_is_idempotent() {
        let traversal = completed(4, 91, 0xee);
        let mut journey = ReturnJourney::from_completed(&traversal).unwrap();
        journey.evict_origin().unwrap();
        journey.restore_origin().unwrap();
        let mut active_scratch = Scratch::new();
        let mut origin_scratch = Scratch::new();
        journey
            .return_to_origin(99, 3, 3, &mut active_scratch, &mut origin_scratch)
            .unwrap();
        let before = journey.clone();
        let duplicate = journey
            .return_to_origin(99, 3, 3, &mut active_scratch, &mut origin_scratch)
            .unwrap();
        assert_eq!(duplicate.disposition, ReturnDisposition::DuplicateIgnored);
        assert_eq!(journey, before);
    }

    #[test]
    fn deterministic_replay_reproduces_return_journey() {
        let left_traversal = completed(4, 91, 0xff);
        let right_traversal = completed(4, 91, 0xff);
        let mut left = ReturnJourney::from_completed(&left_traversal).unwrap();
        let mut right = ReturnJourney::from_completed(&right_traversal).unwrap();
        let mut la = Scratch::new();
        let mut lo = Scratch::new();
        let mut ra = Scratch::new();
        let mut ro = Scratch::new();
        left.evict_origin().unwrap();
        right.evict_origin().unwrap();
        left.advance_active(3, 1, 2, &mut la).unwrap();
        right.advance_active(3, 1, 2, &mut ra).unwrap();
        left.restore_origin().unwrap();
        right.restore_origin().unwrap();
        let lr = left.return_to_origin(111, 4, 3, &mut la, &mut lo).unwrap();
        let rr = right.return_to_origin(111, 4, 3, &mut ra, &mut ro).unwrap();
        assert_eq!(lr, rr);
        assert_eq!(left, right);
    }
}
