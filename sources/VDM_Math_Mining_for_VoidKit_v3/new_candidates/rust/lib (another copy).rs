#![no_std]

use phase_effects::{EffectKind, EffectRecord, SourceVersion};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CompletionResult {
    Ready,
    NotFound,
    Failed,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PendingRequest {
    pub effect_id: u64,
    pub source: SourceVersion,
    pub asset_key: u64,
    pub content_version: u64,
    pub variant: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CompletionInput {
    pub effect_id: u64,
    pub asset_key: u64,
    pub content_version: u64,
    pub variant: u64,
    pub result: CompletionResult,
    pub content_hash: u64,
    pub byte_length: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CompletionRecord {
    pub admission_id: u64,
    pub effect_id: u64,
    pub source: SourceVersion,
    pub asset_key: u64,
    pub content_version: u64,
    pub variant: u64,
    pub result: CompletionResult,
    pub content_hash: u64,
    pub byte_length: u64,
}

const EMPTY_SOURCE: SourceVersion = SourceVersion {
    region_key: 0,
    region_coordination_id: 0,
    zone_key: 0,
    local_commit_id: 0,
    phase_fingerprint: 0,
    world_fingerprint: 0,
};

const EMPTY_PENDING: PendingRequest = PendingRequest {
    effect_id: 0,
    source: EMPTY_SOURCE,
    asset_key: 0,
    content_version: 0,
    variant: 0,
};

const EMPTY_RECORD: CompletionRecord = CompletionRecord {
    admission_id: 0,
    effect_id: 0,
    source: EMPTY_SOURCE,
    asset_key: 0,
    content_version: 0,
    variant: 0,
    result: CompletionResult::Failed,
    content_hash: 0,
    byte_length: 0,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CompletionError {
    InvalidRequest,
    InvalidCompletion,
    PendingFull,
    JournalFull,
    AdmissionIdExhausted,
    UnknownRequest,
    Mismatch,
    ReplaySequence,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RegistrationDisposition {
    Registered,
    DuplicateIgnored,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AdmissionDisposition {
    Accepted,
    DuplicateIgnored,
    Replayed,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AdmissionReceipt {
    pub disposition: AdmissionDisposition,
    pub pending_count: usize,
    pub admitted_count: usize,
    pub record: CompletionRecord,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CompletionStream<const PENDING: usize, const ADMITTED: usize> {
    next_admission_id: u64,
    pending_count: usize,
    admitted_count: usize,
    pending: [PendingRequest; PENDING],
    admitted: [CompletionRecord; ADMITTED],
}

impl<const PENDING: usize, const ADMITTED: usize> CompletionStream<PENDING, ADMITTED> {
    pub const fn new() -> Self {
        Self {
            next_admission_id: 1,
            pending_count: 0,
            admitted_count: 0,
            pending: [EMPTY_PENDING; PENDING],
            admitted: [EMPTY_RECORD; ADMITTED],
        }
    }

    pub const fn next_admission_id(&self) -> u64 {
        self.next_admission_id
    }

    pub const fn pending_count(&self) -> usize {
        self.pending_count
    }

    pub const fn admitted_count(&self) -> usize {
        self.admitted_count
    }

    pub fn pending(&self) -> &[PendingRequest] {
        &self.pending[..self.pending_count]
    }

    pub fn admitted(&self) -> &[CompletionRecord] {
        &self.admitted[..self.admitted_count]
    }

    pub fn register(
        &mut self,
        effect: EffectRecord,
    ) -> Result<RegistrationDisposition, CompletionError> {
        if effect.effect_id == 0
            || effect.request.kind != EffectKind::StreamRequest
            || !effect.source.is_valid()
            || effect.request.subject_key == 0
        {
            return Err(CompletionError::InvalidRequest);
        }

        let candidate = PendingRequest {
            effect_id: effect.effect_id,
            source: effect.source,
            asset_key: effect.request.subject_key,
            content_version: effect.request.payload0,
            variant: effect.request.payload1,
        };

        if let Some(existing) = self
            .pending()
            .iter()
            .copied()
            .find(|pending| pending.effect_id == effect.effect_id)
        {
            return if existing == candidate {
                Ok(RegistrationDisposition::DuplicateIgnored)
            } else {
                Err(CompletionError::Mismatch)
            };
        }

        if let Some(existing) = self
            .admitted()
            .iter()
            .copied()
            .find(|record| record.effect_id == effect.effect_id)
        {
            let same = existing.source == candidate.source
                && existing.asset_key == candidate.asset_key
                && existing.content_version == candidate.content_version
                && existing.variant == candidate.variant;
            return if same {
                Ok(RegistrationDisposition::DuplicateIgnored)
            } else {
                Err(CompletionError::Mismatch)
            };
        }

        if self.pending_count == PENDING {
            return Err(CompletionError::PendingFull);
        }
        self.pending[self.pending_count] = candidate;
        self.pending_count += 1;
        Ok(RegistrationDisposition::Registered)
    }

    pub fn admit(
        &mut self,
        input: CompletionInput,
    ) -> Result<AdmissionReceipt, CompletionError> {
        if input.effect_id == 0 || input.asset_key == 0 {
            return Err(CompletionError::InvalidCompletion);
        }

        if let Some(existing) = self
            .admitted()
            .iter()
            .copied()
            .find(|record| record.effect_id == input.effect_id)
        {
            let same = input_matches_record(input, existing);
            return if same {
                Ok(AdmissionReceipt {
                    disposition: AdmissionDisposition::DuplicateIgnored,
                    pending_count: self.pending_count,
                    admitted_count: self.admitted_count,
                    record: existing,
                })
            } else {
                Err(CompletionError::Mismatch)
            };
        }

        let index = self
            .pending()
            .iter()
            .position(|pending| pending.effect_id == input.effect_id)
            .ok_or(CompletionError::UnknownRequest)?;
        let pending = self.pending[index];
        if !input_matches_pending(input, pending) {
            return Err(CompletionError::Mismatch);
        }
        if self.admitted_count == ADMITTED {
            return Err(CompletionError::JournalFull);
        }
        if self.next_admission_id == 0 || self.next_admission_id == u64::MAX {
            return Err(CompletionError::AdmissionIdExhausted);
        }

        let record = CompletionRecord {
            admission_id: self.next_admission_id,
            effect_id: input.effect_id,
            source: pending.source,
            asset_key: input.asset_key,
            content_version: input.content_version,
            variant: input.variant,
            result: input.result,
            content_hash: input.content_hash,
            byte_length: input.byte_length,
        };
        self.admitted[self.admitted_count] = record;
        self.admitted_count += 1;
        self.next_admission_id += 1;
        self.remove_pending(index);
        Ok(AdmissionReceipt {
            disposition: AdmissionDisposition::Accepted,
            pending_count: self.pending_count,
            admitted_count: self.admitted_count,
            record,
        })
    }

    pub fn replay(
        &mut self,
        record: CompletionRecord,
    ) -> Result<AdmissionReceipt, CompletionError> {
        if record.admission_id == 0 || record.effect_id == 0 || record.asset_key == 0 {
            return Err(CompletionError::InvalidCompletion);
        }
        if record.admission_id != self.next_admission_id {
            return Err(CompletionError::ReplaySequence);
        }
        if self
            .admitted()
            .iter()
            .any(|existing| existing.effect_id == record.effect_id)
        {
            return Err(CompletionError::Mismatch);
        }
        let index = self
            .pending()
            .iter()
            .position(|pending| pending.effect_id == record.effect_id)
            .ok_or(CompletionError::UnknownRequest)?;
        let pending = self.pending[index];
        if record.source != pending.source
            || record.asset_key != pending.asset_key
            || record.content_version != pending.content_version
            || record.variant != pending.variant
        {
            return Err(CompletionError::Mismatch);
        }
        if self.admitted_count == ADMITTED {
            return Err(CompletionError::JournalFull);
        }
        if self.next_admission_id == u64::MAX {
            return Err(CompletionError::AdmissionIdExhausted);
        }

        self.admitted[self.admitted_count] = record;
        self.admitted_count += 1;
        self.next_admission_id += 1;
        self.remove_pending(index);
        Ok(AdmissionReceipt {
            disposition: AdmissionDisposition::Replayed,
            pending_count: self.pending_count,
            admitted_count: self.admitted_count,
            record,
        })
    }

    fn remove_pending(&mut self, index: usize) {
        let mut cursor = index + 1;
        while cursor < self.pending_count {
            self.pending[cursor - 1] = self.pending[cursor];
            cursor += 1;
        }
        self.pending_count -= 1;
        self.pending[self.pending_count] = EMPTY_PENDING;
    }
}

impl<const PENDING: usize, const ADMITTED: usize> Default
    for CompletionStream<PENDING, ADMITTED>
{
    fn default() -> Self {
        Self::new()
    }
}

fn input_matches_pending(input: CompletionInput, pending: PendingRequest) -> bool {
    input.effect_id == pending.effect_id
        && input.asset_key == pending.asset_key
        && input.content_version == pending.content_version
        && input.variant == pending.variant
}

fn input_matches_record(input: CompletionInput, record: CompletionRecord) -> bool {
    input.effect_id == record.effect_id
        && input.asset_key == record.asset_key
        && input.content_version == record.content_version
        && input.variant == record.variant
        && input.result == record.result
        && input.content_hash == record.content_hash
        && input.byte_length == record.byte_length
}

#[cfg(test)]
mod tests {
    extern crate std;

    use super::*;
    use phase_effects::{EffectPipeline, EffectRequest};

    fn source(region: u64, region_version: u64, zone: u64, local: u64) -> SourceVersion {
        SourceVersion {
            region_key: region,
            region_coordination_id: region_version,
            zone_key: zone,
            local_commit_id: local,
            phase_fingerprint: 0x1111_0000_0000_0000 ^ zone ^ local,
            world_fingerprint: 0x2222_0000_0000_0000 ^ zone ^ local,
        }
    }

    fn stream_effects() -> [EffectRecord; 2] {
        let source = source(7, 3, 100, 10);
        let mut pipeline = EffectPipeline::<4>::new();
        pipeline
            .emit_batch(
                source,
                &[
                    EffectRequest::stream(10, 3),
                    EffectRequest {
                        kind: EffectKind::StreamRequest,
                        channel: 0,
                        flags: 0,
                        subject_key: 20,
                        payload0: 4,
                        payload1: 2,
                    },
                ],
            )
            .unwrap();
        [pipeline.pending()[0], pipeline.pending()[1]]
    }

    fn completion(effect: EffectRecord, hash: u64) -> CompletionInput {
        CompletionInput {
            effect_id: effect.effect_id,
            asset_key: effect.request.subject_key,
            content_version: effect.request.payload0,
            variant: effect.request.payload1,
            result: CompletionResult::Ready,
            content_hash: hash,
            byte_length: 4096,
        }
    }

    #[test]
    fn only_stream_requests_are_registered() {
        let source = source(1, 1, 2, 3);
        let mut pipeline = EffectPipeline::<4>::new();
        pipeline
            .emit_batch(source, &[EffectRequest::render(9, 99)])
            .unwrap();
        let mut stream = CompletionStream::<4, 4>::new();
        let before = stream.clone();
        assert_eq!(
            stream.register(pipeline.pending()[0]),
            Err(CompletionError::InvalidRequest)
        );
        assert_eq!(stream, before);
    }

    #[test]
    fn admitted_completion_inherits_exact_source_version() {
        let effects = stream_effects();
        let mut stream = CompletionStream::<4, 4>::new();
        stream.register(effects[0]).unwrap();
        let receipt = stream.admit(completion(effects[0], 0xAA)).unwrap();
        assert_eq!(receipt.disposition, AdmissionDisposition::Accepted);
        assert_eq!(receipt.record.source, effects[0].source);
        assert_eq!(receipt.record.admission_id, 1);
    }

    #[test]
    fn duplicate_completion_is_idempotent() {
        let effects = stream_effects();
        let mut stream = CompletionStream::<4, 4>::new();
        stream.register(effects[0]).unwrap();
        let input = completion(effects[0], 0xBB);
        stream.admit(input).unwrap();
        let before = stream.clone();
        let receipt = stream.admit(input).unwrap();
        assert_eq!(receipt.disposition, AdmissionDisposition::DuplicateIgnored);
        assert_eq!(stream, before);
    }

    #[test]
    fn mismatched_completion_is_atomic() {
        let effects = stream_effects();
        let mut stream = CompletionStream::<4, 4>::new();
        stream.register(effects[0]).unwrap();
        let before = stream.clone();
        let mut input = completion(effects[0], 0xCC);
        input.content_version += 1;
        assert_eq!(stream.admit(input), Err(CompletionError::Mismatch));
        assert_eq!(stream, before);
    }

    #[test]
    fn arrival_order_is_replayed_exactly() {
        let effects = stream_effects();
        let mut original = CompletionStream::<4, 4>::new();
        original.register(effects[0]).unwrap();
        original.register(effects[1]).unwrap();
        original.admit(completion(effects[1], 0x22)).unwrap();
        original.admit(completion(effects[0], 0x11)).unwrap();
        assert_eq!(original.admitted()[0].effect_id, effects[1].effect_id);
        assert_eq!(original.admitted()[1].effect_id, effects[0].effect_id);

        let mut replay = CompletionStream::<4, 4>::new();
        replay.register(effects[0]).unwrap();
        replay.register(effects[1]).unwrap();
        for record in original.admitted() {
            replay.replay(*record).unwrap();
        }
        assert_eq!(replay, original);
    }

    #[test]
    fn admission_ids_are_stream_scoped() {
        let effects = stream_effects();
        let mut first = CompletionStream::<2, 2>::new();
        let mut second = CompletionStream::<2, 2>::new();
        first.register(effects[0]).unwrap();
        second.register(effects[0]).unwrap();
        assert_eq!(first.admit(completion(effects[0], 1)).unwrap().record.admission_id, 1);
        assert_eq!(second.admit(completion(effects[0], 1)).unwrap().record.admission_id, 1);
    }

    #[test]
    fn journal_capacity_failure_is_atomic() {
        let effects = stream_effects();
        let mut stream = CompletionStream::<2, 1>::new();
        stream.register(effects[0]).unwrap();
        stream.register(effects[1]).unwrap();
        stream.admit(completion(effects[0], 1)).unwrap();
        let before = stream.clone();
        assert_eq!(
            stream.admit(completion(effects[1], 2)),
            Err(CompletionError::JournalFull)
        );
        assert_eq!(stream, before);
    }

    #[test]
    fn replay_sequence_mismatch_is_atomic() {
        let effects = stream_effects();
        let mut original = CompletionStream::<2, 2>::new();
        original.register(effects[0]).unwrap();
        let mut record = original.admit(completion(effects[0], 7)).unwrap().record;

        let mut replay = CompletionStream::<2, 2>::new();
        replay.register(effects[0]).unwrap();
        let before = replay.clone();
        record.admission_id = 2;
        assert_eq!(replay.replay(record), Err(CompletionError::ReplaySequence));
        assert_eq!(replay, before);
    }
}
