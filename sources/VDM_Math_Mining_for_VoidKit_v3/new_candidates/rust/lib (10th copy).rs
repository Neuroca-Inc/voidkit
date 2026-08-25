#![no_std]

use phase_completions::{
    AdmissionReceipt, CompletionError, CompletionInput, CompletionRecord, CompletionResult,
    CompletionStream,
};
use phase_effects::{
    ConsumeDisposition, EffectError, EffectKind, EffectPipeline, EffectRequest, SourceVersion,
};
use phase_wide::{WideDisposition, WideError, WidePhaseWorldCell, WideResult};
use qbl_abi::QblPrimitive;
use world_core::{
    AdmittedCauseV0, MutationIntentV0, ObjectKey, TransitionScratch, ZoneCoord,
    NON_SPATIAL_SITE,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TraversalError {
    InvalidIdentity,
    AlreadyBootstrapped,
    NotBootstrapped,
    AlreadyRequested,
    NotRequested,
    Completion(CompletionError),
    Effect(EffectError),
    CompletionUnknown,
    CompletionNotReady,
    CompletionStale,
    AlreadyTraversed,
    SourceActorMissing,
    DestinationSpawnRejected,
    TraversalCommitExhausted,
    SourcePhase(WideError),
    DestinationPhase(WideError),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TraversalDisposition {
    Committed,
    DuplicateIgnored,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SourceBootstrapReceipt {
    pub primitive: QblPrimitive,
    pub source_local_commit_id: u64,
    pub traveler_key: u64,
    pub source_actor_key: ObjectKey,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DestinationRequestReceipt {
    pub primitive: QblPrimitive,
    pub destination_local_commit_id: u64,
    pub stream_effect_id: u64,
    pub destination_source: SourceVersion,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TraversalReceipt {
    pub disposition: TraversalDisposition,
    pub traversal_commit_id: u64,
    pub source_primitive: QblPrimitive,
    pub destination_primitive: QblPrimitive,
    pub source_local_commit_id: u64,
    pub destination_local_commit_id: u64,
    pub completion_admission_id: u64,
    pub traveler_key: u64,
    pub source_actor_key: ObjectKey,
    pub destination_actor_key: ObjectKey,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StreamedZoneTraversal<
    const AXES: usize,
    const OBJECTS: usize,
    const BUCKETS: usize,
    const EFFECTS: usize,
    const PENDING: usize,
    const ADMITTED: usize,
> {
    region_key: u64,
    traveler_key: u64,
    source: WidePhaseWorldCell<AXES, OBJECTS, BUCKETS>,
    destination: WidePhaseWorldCell<AXES, OBJECTS, BUCKETS>,
    effects: EffectPipeline<EFFECTS>,
    completions: CompletionStream<PENDING, ADMITTED>,
    destination_asset_key: u64,
    content_version: u64,
    variant: u64,
    stream_effect_id: u64,
    consumed_admission_id: u64,
    traversal_commit_id: u64,
    source_actor_key: Option<ObjectKey>,
    destination_actor_key: Option<ObjectKey>,
    destination_requested: bool,
    traversed: bool,
}

impl<
        const AXES: usize,
        const OBJECTS: usize,
        const BUCKETS: usize,
        const EFFECTS: usize,
        const PENDING: usize,
        const ADMITTED: usize,
    > StreamedZoneTraversal<AXES, OBJECTS, BUCKETS, EFFECTS, PENDING, ADMITTED>
{
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        region_key: u64,
        traveler_key: u64,
        source_zone: ZoneCoord,
        destination_zone: ZoneCoord,
        destination_asset_key: u64,
        content_version: u64,
        variant: u64,
        source_pair_limb_limit: u32,
        destination_pair_limb_limit: u32,
    ) -> Result<Self, TraversalError> {
        if region_key == 0 || traveler_key == 0 || destination_asset_key == 0 {
            return Err(TraversalError::InvalidIdentity);
        }
        if source_zone.key() == destination_zone.key() {
            return Err(TraversalError::InvalidIdentity);
        }
        Ok(Self {
            region_key,
            traveler_key,
            source: WidePhaseWorldCell::new(source_zone, source_pair_limb_limit)
                .map_err(TraversalError::SourcePhase)?,
            destination: WidePhaseWorldCell::new(destination_zone, destination_pair_limb_limit)
                .map_err(TraversalError::DestinationPhase)?,
            effects: EffectPipeline::new(),
            completions: CompletionStream::new(),
            destination_asset_key,
            content_version,
            variant,
            stream_effect_id: 0,
            consumed_admission_id: 0,
            traversal_commit_id: 0,
            source_actor_key: None,
            destination_actor_key: None,
            destination_requested: false,
            traversed: false,
        })
    }

    pub const fn region_key(&self) -> u64 { self.region_key }
    pub const fn traveler_key(&self) -> u64 { self.traveler_key }
    pub const fn source(&self) -> &WidePhaseWorldCell<AXES, OBJECTS, BUCKETS> { &self.source }
    pub const fn destination(&self) -> &WidePhaseWorldCell<AXES, OBJECTS, BUCKETS> { &self.destination }
    pub const fn effects(&self) -> &EffectPipeline<EFFECTS> { &self.effects }
    pub const fn completions(&self) -> &CompletionStream<PENDING, ADMITTED> { &self.completions }
    pub const fn source_actor_key(&self) -> Option<ObjectKey> { self.source_actor_key }
    pub const fn destination_actor_key(&self) -> Option<ObjectKey> { self.destination_actor_key }
    pub const fn stream_effect_id(&self) -> u64 { self.stream_effect_id }
    pub const fn consumed_admission_id(&self) -> u64 { self.consumed_admission_id }
    pub const fn traversal_commit_id(&self) -> u64 { self.traversal_commit_id }
    pub const fn destination_requested(&self) -> bool { self.destination_requested }
    pub const fn traversed(&self) -> bool { self.traversed }

    pub fn source_version(&self) -> SourceVersion {
        source_version(self.region_key, &self.source)
    }

    pub fn destination_version(&self) -> SourceVersion {
        source_version(self.region_key, &self.destination)
    }

    pub fn bootstrap_source<const CAUSES: usize, const INTENTS: usize>(
        &mut self,
        source_sequence: u32,
        health: u16,
        scratch: &mut TransitionScratch<CAUSES, INTENTS, OBJECTS>,
    ) -> Result<SourceBootstrapReceipt, TraversalError> {
        if source_sequence == 0 || health == 0 {
            return Err(TraversalError::InvalidIdentity);
        }
        if self.source_actor_key.is_some() {
            return Err(TraversalError::AlreadyBootstrapped);
        }
        let mut staged = self.clone();
        let causes = [AdmittedCauseV0::external_input(
            source_sequence,
            staged.traveler_key,
            staged.destination_asset_key,
        )];
        let intents = [MutationIntentV0::spawn_actor(
            source_sequence,
            health,
            NON_SPATIAL_SITE,
        )];
        let result = staged
            .source
            .transact(&causes, &intents, scratch)
            .map_err(TraversalError::SourcePhase)?;
        if result.disposition != WideDisposition::Committed || result.rejected_requests != 0 {
            return Err(TraversalError::SourceActorMissing);
        }
        let actor_key = staged
            .source
            .world()
            .objects()
            .object_key_at(0)
            .ok_or(TraversalError::SourceActorMissing)?;
        staged.source_actor_key = Some(actor_key);
        let receipt = SourceBootstrapReceipt {
            primitive: result.primitive,
            source_local_commit_id: result.local_commit_id,
            traveler_key: staged.traveler_key,
            source_actor_key: actor_key,
        };
        *self = staged;
        Ok(receipt)
    }

    pub fn advance_source_local_cause<const CAUSES: usize, const INTENTS: usize>(
        &mut self,
        source_sequence: u32,
        payload0: u64,
        payload1: u64,
        scratch: &mut TransitionScratch<CAUSES, INTENTS, OBJECTS>,
    ) -> Result<WideResult, TraversalError> {
        if source_sequence == 0 {
            return Err(TraversalError::InvalidIdentity);
        }
        if self.source_actor_key.is_none() || self.traversed {
            return Err(TraversalError::NotBootstrapped);
        }
        let mut staged = self.clone();
        let causes = [AdmittedCauseV0::external_input(source_sequence, payload0, payload1)];
        let result = staged
            .source
            .transact(&causes, &[], scratch)
            .map_err(TraversalError::SourcePhase)?;
        *self = staged;
        Ok(result)
    }

    pub fn advance_destination_local_cause<const CAUSES: usize, const INTENTS: usize>(
        &mut self,
        source_sequence: u32,
        payload0: u64,
        payload1: u64,
        scratch: &mut TransitionScratch<CAUSES, INTENTS, OBJECTS>,
    ) -> Result<WideResult, TraversalError> {
        if source_sequence == 0 {
            return Err(TraversalError::InvalidIdentity);
        }
        if self.traversed {
            return Err(TraversalError::AlreadyTraversed);
        }
        let mut staged = self.clone();
        let causes = [AdmittedCauseV0::external_input(source_sequence, payload0, payload1)];
        let result = staged
            .destination
            .transact(&causes, &[], scratch)
            .map_err(TraversalError::DestinationPhase)?;
        *self = staged;
        Ok(result)
    }

    pub fn request_destination<const CAUSES: usize, const INTENTS: usize>(
        &mut self,
        source_sequence: u32,
        scratch: &mut TransitionScratch<CAUSES, INTENTS, OBJECTS>,
    ) -> Result<DestinationRequestReceipt, TraversalError> {
        if source_sequence == 0 {
            return Err(TraversalError::InvalidIdentity);
        }
        if self.source_actor_key.is_none() {
            return Err(TraversalError::NotBootstrapped);
        }
        if self.destination_requested {
            return Err(TraversalError::AlreadyRequested);
        }
        let mut staged = self.clone();
        let causes = [AdmittedCauseV0::external_input(
            source_sequence,
            staged.traveler_key,
            staged.destination_asset_key,
        )];
        let result = staged
            .destination
            .transact(&causes, &[], scratch)
            .map_err(TraversalError::DestinationPhase)?;
        if result.disposition != WideDisposition::Committed {
            return Err(TraversalError::DestinationPhase(WideError::InvalidOrthad));
        }
        let source = staged.destination_version();
        staged
            .effects
            .emit_batch(
                source,
                &[EffectRequest {
                    kind: EffectKind::StreamRequest,
                    channel: 0,
                    flags: 1,
                    subject_key: staged.destination_asset_key,
                    payload0: staged.content_version,
                    payload1: staged.variant,
                }],
            )
            .map_err(TraversalError::Effect)?;
        let consumed = staged.effects.consume_next(&[source]);
        if consumed.disposition != ConsumeDisposition::Ready {
            return Err(TraversalError::Effect(EffectError::InvalidSource));
        }
        let effect = consumed.effect.ok_or(TraversalError::Effect(EffectError::InvalidSource))?;
        staged
            .completions
            .register(effect)
            .map_err(TraversalError::Completion)?;
        staged.stream_effect_id = effect.effect_id;
        staged.destination_requested = true;
        let receipt = DestinationRequestReceipt {
            primitive: result.primitive,
            destination_local_commit_id: result.local_commit_id,
            stream_effect_id: effect.effect_id,
            destination_source: source,
        };
        *self = staged;
        Ok(receipt)
    }

    pub fn admit_completion(
        &mut self,
        input: CompletionInput,
    ) -> Result<AdmissionReceipt, TraversalError> {
        self.completions.admit(input).map_err(TraversalError::Completion)
    }

    pub fn provision_destination_pair_limbs(&mut self, new_limit: u32) -> Result<(), TraversalError> {
        self.destination
            .provision_pair_limbs(new_limit)
            .map_err(TraversalError::DestinationPhase)
    }

    pub fn traverse<
        const SOURCE_CAUSES: usize,
        const SOURCE_INTENTS: usize,
        const DEST_CAUSES: usize,
        const DEST_INTENTS: usize,
    >(
        &mut self,
        completion_admission_id: u64,
        source_sequence: u32,
        destination_sequence: u32,
        source_scratch: &mut TransitionScratch<SOURCE_CAUSES, SOURCE_INTENTS, OBJECTS>,
        destination_scratch: &mut TransitionScratch<DEST_CAUSES, DEST_INTENTS, OBJECTS>,
    ) -> Result<TraversalReceipt, TraversalError> {
        if completion_admission_id == 0 || source_sequence == 0 || destination_sequence == 0 {
            return Err(TraversalError::InvalidIdentity);
        }
        let source_actor_key = self.source_actor_key.ok_or(TraversalError::NotBootstrapped)?;
        if !self.destination_requested {
            return Err(TraversalError::NotRequested);
        }
        if self.traversed {
            if self.consumed_admission_id == completion_admission_id {
                return Ok(TraversalReceipt {
                    disposition: TraversalDisposition::DuplicateIgnored,
                    traversal_commit_id: self.traversal_commit_id,
                    source_primitive: QblPrimitive::NONE,
                    destination_primitive: QblPrimitive::NONE,
                    source_local_commit_id: self.source.local_commit_id(),
                    destination_local_commit_id: self.destination.local_commit_id(),
                    completion_admission_id,
                    traveler_key: self.traveler_key,
                    source_actor_key,
                    destination_actor_key: self.destination_actor_key.ok_or(TraversalError::SourceActorMissing)?,
                });
            }
            return Err(TraversalError::AlreadyTraversed);
        }

        let record = self
            .find_completion(completion_admission_id)
            .ok_or(TraversalError::CompletionUnknown)?;
        if record.effect_id != self.stream_effect_id
            || record.asset_key != self.destination_asset_key
            || record.content_version != self.content_version
            || record.variant != self.variant
            || record.result != CompletionResult::Ready
        {
            return Err(TraversalError::CompletionNotReady);
        }
        if record.source != self.destination_version() {
            return Err(TraversalError::CompletionStale);
        }
        let health = self
            .source
            .world()
            .objects()
            .health(source_actor_key)
            .ok_or(TraversalError::SourceActorMissing)?;
        let next_traversal_commit_id = self
            .traversal_commit_id
            .checked_add(1)
            .ok_or(TraversalError::TraversalCommitExhausted)?;

        let mut staged = self.clone();
        let source_causes = [AdmittedCauseV0::external_input(
            source_sequence,
            staged.traveler_key,
            staged.destination.world().zone().key.0,
        )];
        let source_intents = [MutationIntentV0::despawn(source_sequence, source_actor_key)];
        let source_result = staged
            .source
            .transact(&source_causes, &source_intents, source_scratch)
            .map_err(TraversalError::SourcePhase)?;
        if source_result.disposition != WideDisposition::Committed || source_result.rejected_requests != 0 {
            return Err(TraversalError::SourceActorMissing);
        }

        let expected_destination_key = ObjectKey(staged.destination.world().next_object_key());
        let destination_causes = [AdmittedCauseV0::external_input(
            destination_sequence,
            staged.traveler_key,
            record.admission_id,
        )];
        let destination_intents = [MutationIntentV0::spawn_actor(
            destination_sequence,
            health,
            NON_SPATIAL_SITE,
        )];
        let destination_result = staged
            .destination
            .transact(&destination_causes, &destination_intents, destination_scratch)
            .map_err(TraversalError::DestinationPhase)?;
        if destination_result.disposition != WideDisposition::Committed
            || destination_result.rejected_requests != 0
            || staged.destination.world().objects().health(expected_destination_key) != Some(health)
        {
            return Err(TraversalError::DestinationSpawnRejected);
        }

        staged.destination_actor_key = Some(expected_destination_key);
        staged.consumed_admission_id = completion_admission_id;
        staged.traversal_commit_id = next_traversal_commit_id;
        staged.traversed = true;
        let receipt = TraversalReceipt {
            disposition: TraversalDisposition::Committed,
            traversal_commit_id: next_traversal_commit_id,
            source_primitive: source_result.primitive,
            destination_primitive: destination_result.primitive,
            source_local_commit_id: source_result.local_commit_id,
            destination_local_commit_id: destination_result.local_commit_id,
            completion_admission_id,
            traveler_key: staged.traveler_key,
            source_actor_key,
            destination_actor_key: expected_destination_key,
        };
        *self = staged;
        Ok(receipt)
    }

    fn find_completion(&self, admission_id: u64) -> Option<CompletionRecord> {
        self.completions
            .admitted()
            .iter()
            .copied()
            .find(|record| record.admission_id == admission_id)
    }
}

fn source_version<const AXES: usize, const OBJECTS: usize, const BUCKETS: usize>(
    region_key: u64,
    cell: &WidePhaseWorldCell<AXES, OBJECTS, BUCKETS>,
) -> SourceVersion {
    SourceVersion {
        region_key,
        region_coordination_id: 0,
        zone_key: cell.world().zone().key.0,
        local_commit_id: cell.local_commit_id(),
        phase_fingerprint: cell.phase().diagnostic_fingerprint64(),
        world_fingerprint: cell.world().diagnostic_fingerprint64(),
    }
}

#[cfg(test)]
mod tests {
    extern crate std;

    use super::*;

    type Traversal = StreamedZoneTraversal<16, 8, 16, 8, 8, 8>;
    type Scratch = TransitionScratch<8, 16, 8>;

    fn traversal(source_limit: u32, destination_limit: u32) -> Traversal {
        Traversal::new(
            7,
            9001,
            ZoneCoord::new(0, 0, 0).unwrap(),
            ZoneCoord::new(1, 0, 0).unwrap(),
            200,
            4,
            0,
            source_limit,
            destination_limit,
        )
        .unwrap()
    }

    fn ready_input(traversal: &Traversal, hash: u64) -> CompletionInput {
        CompletionInput {
            effect_id: traversal.stream_effect_id(),
            asset_key: 200,
            content_version: 4,
            variant: 0,
            result: CompletionResult::Ready,
            content_hash: hash,
            byte_length: 8192,
        }
    }

    #[test]
    fn source_can_advance_while_destination_loads_and_traversal_remains_valid() {
        let mut traversal = traversal(4, 4);
        let mut source_scratch = Scratch::new();
        let mut destination_scratch = Scratch::new();
        let bootstrap = traversal.bootstrap_source(1, 91, &mut source_scratch).unwrap();
        traversal.request_destination(1, &mut destination_scratch).unwrap();
        let before_completion_source = traversal.source().clone();
        let before_completion_destination = traversal.destination().clone();
        let admission = traversal.admit_completion(ready_input(&traversal, 0xAA)).unwrap();
        assert_eq!(traversal.source(), &before_completion_source);
        assert_eq!(traversal.destination(), &before_completion_destination);
        traversal.advance_source_local_cause(2, 1, 0, &mut source_scratch).unwrap();
        traversal.advance_source_local_cause(3, 2, 0, &mut source_scratch).unwrap();
        let receipt = traversal
            .traverse(admission.record.admission_id, 4, 2, &mut source_scratch, &mut destination_scratch)
            .unwrap();
        assert_eq!(receipt.source_primitive.0, 1);
        assert_eq!(receipt.destination_primitive.0, 2);
        assert_eq!(traversal.source().world().objects().alive_count(), 0);
        assert_eq!(traversal.destination().world().objects().health(receipt.destination_actor_key), Some(91));
        assert_eq!(bootstrap.traveler_key, receipt.traveler_key);
    }

    #[test]
    fn stale_destination_completion_rejects_before_either_cell_mutates() {
        let mut traversal = traversal(4, 4);
        let mut source_scratch = Scratch::new();
        let mut destination_scratch = Scratch::new();
        traversal.bootstrap_source(1, 91, &mut source_scratch).unwrap();
        traversal.request_destination(1, &mut destination_scratch).unwrap();
        let admission = traversal.admit_completion(ready_input(&traversal, 0xBB)).unwrap();
        traversal.advance_destination_local_cause(2, 0, 0, &mut destination_scratch).unwrap();
        let before = traversal.clone();
        assert_eq!(
            traversal.traverse(admission.record.admission_id, 2, 3, &mut source_scratch, &mut destination_scratch),
            Err(TraversalError::CompletionStale)
        );
        assert_eq!(traversal, before);
    }

    #[test]
    fn destination_provisioning_fault_rolls_back_source_and_retries_same_completion() {
        let mut traversal = traversal(4, 1);
        let mut source_scratch = Scratch::new();
        let mut destination_scratch = Scratch::new();
        traversal.bootstrap_source(1, 77, &mut source_scratch).unwrap();
        for sequence in 1..=153 {
            traversal
                .advance_destination_local_cause(sequence, sequence as u64, 0, &mut destination_scratch)
                .unwrap();
        }
        let request = traversal.request_destination(154, &mut destination_scratch).unwrap();
        assert_eq!(request.destination_local_commit_id, 154);
        let admission = traversal.admit_completion(ready_input(&traversal, 0xCC)).unwrap();
        let before = traversal.clone();
        assert_eq!(
            traversal.traverse(admission.record.admission_id, 2, 155, &mut source_scratch, &mut destination_scratch),
            Err(TraversalError::DestinationPhase(WideError::ProvisioningRequired { required_pair_limbs: 2 }))
        );
        assert_eq!(traversal, before);
        let source_fingerprint = traversal.source_version();
        let destination_fingerprint = traversal.destination_version();
        traversal.provision_destination_pair_limbs(2).unwrap();
        assert_eq!(traversal.source_version(), source_fingerprint);
        assert_eq!(traversal.destination_version(), destination_fingerprint);
        let receipt = traversal
            .traverse(admission.record.admission_id, 2, 155, &mut source_scratch, &mut destination_scratch)
            .unwrap();
        assert_eq!(receipt.traversal_commit_id, 1);
    }

    #[test]
    fn duplicate_traversal_is_idempotent() {
        let mut traversal = traversal(4, 4);
        let mut source_scratch = Scratch::new();
        let mut destination_scratch = Scratch::new();
        traversal.bootstrap_source(1, 91, &mut source_scratch).unwrap();
        traversal.request_destination(1, &mut destination_scratch).unwrap();
        let admission = traversal.admit_completion(ready_input(&traversal, 0xDD)).unwrap();
        traversal.traverse(admission.record.admission_id, 2, 2, &mut source_scratch, &mut destination_scratch).unwrap();
        let before = traversal.clone();
        let duplicate = traversal.traverse(admission.record.admission_id, 2, 2, &mut source_scratch, &mut destination_scratch).unwrap();
        assert_eq!(duplicate.disposition, TraversalDisposition::DuplicateIgnored);
        assert_eq!(traversal, before);
    }

    #[test]
    fn deterministic_replay_reproduces_both_cells_and_transfer_receipt() {
        let mut left = traversal(4, 4);
        let mut right = traversal(4, 4);
        let mut ls = Scratch::new();
        let mut ld = Scratch::new();
        let mut rs = Scratch::new();
        let mut rd = Scratch::new();
        left.bootstrap_source(1, 91, &mut ls).unwrap();
        right.bootstrap_source(1, 91, &mut rs).unwrap();
        left.request_destination(1, &mut ld).unwrap();
        right.request_destination(1, &mut rd).unwrap();
        let input = ready_input(&left, 0xEE);
        let la = left.admit_completion(input).unwrap();
        let ra = right.admit_completion(input).unwrap();
        let lr = left.traverse(la.record.admission_id, 2, 2, &mut ls, &mut ld).unwrap();
        let rr = right.traverse(ra.record.admission_id, 2, 2, &mut rs, &mut rd).unwrap();
        assert_eq!(lr, rr);
        assert_eq!(left, right);
    }
}
