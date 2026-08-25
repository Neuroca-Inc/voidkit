#![no_std]

use phase_completions::{
    AdmissionReceipt, CompletionError, CompletionInput, CompletionRecord, CompletionResult,
    CompletionStream,
};
use phase_effects::{
    ConsumeDisposition, EffectError, EffectPipeline, EffectRequest, SourceVersion,
};
use phase_wide::{
    WideDisposition, WideError, WidePhaseWorldCell, WideResult,
};
use qbl_abi::QblPrimitive;
use world_core::{
    AdmittedCauseV0, MutationIntentV0, ObjectKey, TransitionScratch, ZoneCoord,
    NON_SPATIAL_SITE,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SliceError {
    InvalidIdentity,
    AlreadyBootstrapped,
    NotBootstrapped,
    Effect(EffectError),
    Completion(CompletionError),
    CompletionUnknown,
    CompletionNotReady,
    CompletionStale,
    AlreadyActive,
    WorldActorMissing,
    Phase(WideError),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ActivationDisposition {
    Committed,
    DuplicateIgnored,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BootstrapReceipt {
    pub primitive: QblPrimitive,
    pub local_commit_id: u64,
    pub stream_effect_id: u64,
    pub actor_key: ObjectKey,
    pub source: SourceVersion,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ActivationReceipt {
    pub disposition: ActivationDisposition,
    pub primitive: QblPrimitive,
    pub local_commit_id: u64,
    pub completion_admission_id: u64,
    pub source_before: SourceVersion,
    pub source_after: SourceVersion,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StreamedZoneSlice<
    const AXES: usize,
    const OBJECTS: usize,
    const BUCKETS: usize,
    const EFFECTS: usize,
    const PENDING: usize,
    const ADMITTED: usize,
> {
    region_key: u64,
    cell: WidePhaseWorldCell<AXES, OBJECTS, BUCKETS>,
    effects: EffectPipeline<EFFECTS>,
    completions: CompletionStream<PENDING, ADMITTED>,
    actor_key: Option<ObjectKey>,
    asset_key: u64,
    content_version: u64,
    variant: u64,
    stream_effect_id: u64,
    consumed_admission_id: u64,
    zone_active: bool,
}

impl<
        const AXES: usize,
        const OBJECTS: usize,
        const BUCKETS: usize,
        const EFFECTS: usize,
        const PENDING: usize,
        const ADMITTED: usize,
    > StreamedZoneSlice<AXES, OBJECTS, BUCKETS, EFFECTS, PENDING, ADMITTED>
{
    pub fn new(
        region_key: u64,
        zone: ZoneCoord,
        asset_key: u64,
        content_version: u64,
        variant: u64,
        pair_limb_limit: u32,
    ) -> Result<Self, SliceError> {
        if region_key == 0 || asset_key == 0 {
            return Err(SliceError::InvalidIdentity);
        }
        Ok(Self {
            region_key,
            cell: WidePhaseWorldCell::new(zone, pair_limb_limit)
                .map_err(SliceError::Phase)?,
            effects: EffectPipeline::new(),
            completions: CompletionStream::new(),
            actor_key: None,
            asset_key,
            content_version,
            variant,
            stream_effect_id: 0,
            consumed_admission_id: 0,
            zone_active: false,
        })
    }

    pub const fn region_key(&self) -> u64 {
        self.region_key
    }

    pub const fn cell(&self) -> &WidePhaseWorldCell<AXES, OBJECTS, BUCKETS> {
        &self.cell
    }

    pub const fn effects(&self) -> &EffectPipeline<EFFECTS> {
        &self.effects
    }

    pub const fn completions(&self) -> &CompletionStream<PENDING, ADMITTED> {
        &self.completions
    }

    pub const fn actor_key(&self) -> Option<ObjectKey> {
        self.actor_key
    }

    pub const fn stream_effect_id(&self) -> u64 {
        self.stream_effect_id
    }

    pub const fn consumed_admission_id(&self) -> u64 {
        self.consumed_admission_id
    }

    pub const fn zone_active(&self) -> bool {
        self.zone_active
    }

    pub fn source_version(&self) -> SourceVersion {
        SourceVersion {
            region_key: self.region_key,
            region_coordination_id: 0,
            zone_key: self.cell.world().zone().key.0,
            local_commit_id: self.cell.local_commit_id(),
            phase_fingerprint: self.cell.phase().diagnostic_fingerprint64(),
            world_fingerprint: self.cell.world().diagnostic_fingerprint64(),
        }
    }

    pub fn bootstrap_actor<const CAUSES: usize, const INTENTS: usize>(
        &mut self,
        source_sequence: u32,
        health: u16,
        scratch: &mut TransitionScratch<CAUSES, INTENTS, OBJECTS>,
    ) -> Result<BootstrapReceipt, SliceError> {
        if source_sequence == 0 || health == 0 {
            return Err(SliceError::InvalidIdentity);
        }
        if self.actor_key.is_some() {
            return Err(SliceError::AlreadyBootstrapped);
        }

        let mut staged = self.clone();
        let causes = [AdmittedCauseV0::external_input(
            source_sequence,
            staged.asset_key,
            staged.content_version,
        )];
        let intents = [MutationIntentV0::spawn_actor(
            source_sequence,
            health,
            NON_SPATIAL_SITE,
        )];
        let result = staged
            .cell
            .transact(&causes, &intents, scratch)
            .map_err(SliceError::Phase)?;
        if result.disposition != WideDisposition::Committed {
            return Err(SliceError::Phase(WideError::InvalidOrthad));
        }

        let actor_key = staged
            .cell
            .world()
            .objects()
            .object_key_at(0)
            .ok_or(SliceError::WorldActorMissing)?;
        staged.actor_key = Some(actor_key);
        let source = staged.source_version();
        staged
            .effects
            .emit_batch(
                source,
                &[EffectRequest {
                    kind: phase_effects::EffectKind::StreamRequest,
                    channel: 0,
                    flags: 1,
                    subject_key: staged.asset_key,
                    payload0: staged.content_version,
                    payload1: staged.variant,
                }],
            )
            .map_err(SliceError::Effect)?;
        let consumed = staged.effects.consume_next(&[source]);
        if consumed.disposition != ConsumeDisposition::Ready {
            return Err(SliceError::Effect(EffectError::InvalidSource));
        }
        let effect = consumed
            .effect
            .ok_or(SliceError::Effect(EffectError::InvalidSource))?;
        staged
            .completions
            .register(effect)
            .map_err(SliceError::Completion)?;
        staged.stream_effect_id = effect.effect_id;

        let receipt = BootstrapReceipt {
            primitive: result.primitive,
            local_commit_id: result.local_commit_id,
            stream_effect_id: effect.effect_id,
            actor_key,
            source,
        };
        *self = staged;
        Ok(receipt)
    }

    pub fn admit_completion(
        &mut self,
        input: CompletionInput,
    ) -> Result<AdmissionReceipt, SliceError> {
        self.completions
            .admit(input)
            .map_err(SliceError::Completion)
    }

    pub fn advance_local_cause<const CAUSES: usize, const INTENTS: usize>(
        &mut self,
        source_sequence: u32,
        payload0: u64,
        payload1: u64,
        scratch: &mut TransitionScratch<CAUSES, INTENTS, OBJECTS>,
    ) -> Result<WideResult, SliceError> {
        if source_sequence == 0 {
            return Err(SliceError::InvalidIdentity);
        }
        if self.actor_key.is_none() {
            return Err(SliceError::NotBootstrapped);
        }
        let mut staged = self.clone();
        let causes = [AdmittedCauseV0::external_input(
            source_sequence,
            payload0,
            payload1,
        )];
        let result = staged
            .cell
            .transact(&causes, &[], scratch)
            .map_err(SliceError::Phase)?;
        *self = staged;
        Ok(result)
    }

    pub fn activate_zone<const CAUSES: usize, const INTENTS: usize>(
        &mut self,
        completion_admission_id: u64,
        source_sequence: u32,
        target: (i32, i32, i32),
        scratch: &mut TransitionScratch<CAUSES, INTENTS, OBJECTS>,
    ) -> Result<ActivationReceipt, SliceError> {
        if completion_admission_id == 0 || source_sequence == 0 {
            return Err(SliceError::InvalidIdentity);
        }
        let actor_key = self.actor_key.ok_or(SliceError::NotBootstrapped)?;
        if self.zone_active {
            if self.consumed_admission_id == completion_admission_id {
                let source = self.source_version();
                return Ok(ActivationReceipt {
                    disposition: ActivationDisposition::DuplicateIgnored,
                    primitive: QblPrimitive::NONE,
                    local_commit_id: self.cell.local_commit_id(),
                    completion_admission_id,
                    source_before: source,
                    source_after: source,
                });
            }
            return Err(SliceError::AlreadyActive);
        }

        let record = self
            .find_completion(completion_admission_id)
            .ok_or(SliceError::CompletionUnknown)?;
        if record.effect_id != self.stream_effect_id
            || record.asset_key != self.asset_key
            || record.content_version != self.content_version
            || record.variant != self.variant
            || record.result != CompletionResult::Ready
        {
            return Err(SliceError::CompletionNotReady);
        }
        let source_before = self.source_version();
        if record.source != source_before {
            return Err(SliceError::CompletionStale);
        }

        let mut staged = self.clone();
        let causes = [AdmittedCauseV0::external_input(
            source_sequence,
            record.asset_key,
            record.admission_id,
        )];
        let intents = [MutationIntentV0::replace_kinematics(
            source_sequence,
            actor_key,
            target.0,
            target.1,
            target.2,
            0,
            0,
            0,
        )];
        let WideResult {
            disposition,
            local_commit_id,
            primitive,
            ..
        } = staged
            .cell
            .transact(&causes, &intents, scratch)
            .map_err(SliceError::Phase)?;
        if disposition != WideDisposition::Committed {
            return Err(SliceError::Phase(WideError::InvalidOrthad));
        }
        staged.zone_active = true;
        staged.consumed_admission_id = completion_admission_id;
        let source_after = staged.source_version();
        *self = staged;
        Ok(ActivationReceipt {
            disposition: ActivationDisposition::Committed,
            primitive,
            local_commit_id,
            completion_admission_id,
            source_before,
            source_after,
        })
    }

    fn find_completion(&self, admission_id: u64) -> Option<CompletionRecord> {
        self.completions
            .admitted()
            .iter()
            .copied()
            .find(|record| record.admission_id == admission_id)
    }
}

#[cfg(test)]
mod tests {
    extern crate std;

    use super::*;
    use phase_completions::CompletionResult;
    use world_core::{LocalPosQ16_16, WorldFault};

    type Slice = StreamedZoneSlice<16, 8, 16, 8, 8, 8>;
    type Scratch = TransitionScratch<8, 16, 8>;

    fn slice() -> Slice {
        Slice::new(7, ZoneCoord::new(0, 0, 0).unwrap(), 100, 3, 0, 4).unwrap()
    }

    fn ready_input(slice: &Slice, hash: u64) -> CompletionInput {
        CompletionInput {
            effect_id: slice.stream_effect_id(),
            asset_key: 100,
            content_version: 3,
            variant: 0,
            result: CompletionResult::Ready,
            content_hash: hash,
            byte_length: 4096,
        }
    }

    #[test]
    fn complete_streamed_zone_round_trip_is_playable() {
        let mut slice = slice();
        let mut scratch = Scratch::new();
        let bootstrap = slice.bootstrap_actor(1, 100, &mut scratch).unwrap();
        assert_eq!(bootstrap.local_commit_id, 1);
        assert_eq!(bootstrap.primitive.0, 1);
        let before_completion = slice.clone();
        let admitted = slice.admit_completion(ready_input(&slice, 0xAA)).unwrap();
        assert_eq!(slice.cell(), before_completion.cell());
        let activation = slice
            .activate_zone(admitted.record.admission_id, 2, (65_536, 0, 0), &mut scratch)
            .unwrap();
        assert_eq!(activation.local_commit_id, 2);
        assert_eq!(activation.primitive.0, 2);
        assert!(slice.zone_active());
        assert_eq!(
            slice.cell().world().objects().kinematics(bootstrap.actor_key),
            Some((65_536, 0, 0, 0, 0, 0))
        );
    }

    #[test]
    fn completion_alone_has_no_simulation_authority() {
        let mut slice = slice();
        let mut scratch = Scratch::new();
        slice.bootstrap_actor(1, 100, &mut scratch).unwrap();
        let phase_before = slice.cell().phase().clone();
        let world_before = slice.cell().world().clone();
        slice.admit_completion(ready_input(&slice, 0xAA)).unwrap();
        assert_eq!(slice.cell().phase(), &phase_before);
        assert_eq!(slice.cell().world(), &world_before);
        assert!(!slice.zone_active());
    }

    #[test]
    fn stale_completion_cannot_activate_advanced_cell() {
        let mut slice = slice();
        let mut scratch = Scratch::new();
        slice.bootstrap_actor(1, 100, &mut scratch).unwrap();
        let admitted = slice.admit_completion(ready_input(&slice, 0xAA)).unwrap();
        let causes = [AdmittedCauseV0::external_input(2, 0, 0)];
        slice.cell.transact(&causes, &[], &mut scratch).unwrap();
        let before = slice.clone();
        assert_eq!(
            slice.activate_zone(admitted.record.admission_id, 3, (65_536, 0, 0), &mut scratch),
            Err(SliceError::CompletionStale)
        );
        assert_eq!(slice, before);
    }

    #[test]
    fn world_fault_rolls_back_phase_and_preserves_completion_for_retry() {
        let mut slice = slice();
        let mut scratch = Scratch::new();
        slice.bootstrap_actor(1, 100, &mut scratch).unwrap();
        let admitted = slice.admit_completion(ready_input(&slice, 0xAA)).unwrap();
        let before = slice.clone();
        let invalid = LocalPosQ16_16::new(32 << 16);
        assert_eq!(invalid, Err(WorldFault::LocalPositionRange));
        let result = slice.activate_zone(
            admitted.record.admission_id,
            2,
            (32 << 16, 0, 0),
            &mut scratch,
        );
        assert!(matches!(result, Err(SliceError::Phase(WideError::World(_)))));
        assert_eq!(slice, before);
        slice.activate_zone(admitted.record.admission_id, 2, (65_536, 0, 0), &mut scratch).unwrap();
        assert!(slice.zone_active());
    }

    #[test]
    fn duplicate_activation_is_idempotent() {
        let mut slice = slice();
        let mut scratch = Scratch::new();
        slice.bootstrap_actor(1, 100, &mut scratch).unwrap();
        let admitted = slice.admit_completion(ready_input(&slice, 0xAA)).unwrap();
        slice.activate_zone(admitted.record.admission_id, 2, (65_536, 0, 0), &mut scratch).unwrap();
        let before = slice.clone();
        let duplicate = slice.activate_zone(admitted.record.admission_id, 2, (65_536, 0, 0), &mut scratch).unwrap();
        assert_eq!(duplicate.disposition, ActivationDisposition::DuplicateIgnored);
        assert_eq!(slice, before);
    }

    #[test]
    fn deterministic_replay_reproduces_complete_slice() {
        let mut left = slice();
        let mut right = slice();
        let mut left_scratch = Scratch::new();
        let mut right_scratch = Scratch::new();
        left.bootstrap_actor(1, 100, &mut left_scratch).unwrap();
        right.bootstrap_actor(1, 100, &mut right_scratch).unwrap();
        let input = ready_input(&left, 0xAA);
        let left_record = left.admit_completion(input).unwrap().record;
        let right_record = right.admit_completion(input).unwrap().record;
        assert_eq!(left_record, right_record);
        left.activate_zone(left_record.admission_id, 2, (65_536, 0, 0), &mut left_scratch).unwrap();
        right.activate_zone(right_record.admission_id, 2, (65_536, 0, 0), &mut right_scratch).unwrap();
        assert_eq!(left, right);
    }
}
