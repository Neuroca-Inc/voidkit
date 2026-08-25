#![no_std]

pub const MAX_EMIT_BATCH: usize = 8;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EffectKind {
    RenderCommands,
    AudioCommands,
    StreamRequest,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SourceVersion {
    pub region_key: u64,
    pub region_coordination_id: u64,
    pub zone_key: u64,
    pub local_commit_id: u64,
    pub phase_fingerprint: u64,
    pub world_fingerprint: u64,
}

impl SourceVersion {
    pub const fn is_valid(self) -> bool {
        self.region_key != 0 && self.zone_key != 0
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EffectRequest {
    pub kind: EffectKind,
    pub channel: u16,
    pub flags: u32,
    pub subject_key: u64,
    pub payload0: u64,
    pub payload1: u64,
}

impl EffectRequest {
    pub const fn render(subject_key: u64, payload0: u64) -> Self {
        Self {
            kind: EffectKind::RenderCommands,
            channel: 0,
            flags: 0,
            subject_key,
            payload0,
            payload1: 0,
        }
    }

    pub const fn audio(subject_key: u64, payload0: u64) -> Self {
        Self {
            kind: EffectKind::AudioCommands,
            channel: 0,
            flags: 0,
            subject_key,
            payload0,
            payload1: 0,
        }
    }

    pub const fn stream(subject_key: u64, payload0: u64) -> Self {
        Self {
            kind: EffectKind::StreamRequest,
            channel: 0,
            flags: 0,
            subject_key,
            payload0,
            payload1: 0,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EffectRecord {
    pub effect_id: u64,
    pub source: SourceVersion,
    pub request: EffectRequest,
}

const EMPTY_SOURCE: SourceVersion = SourceVersion {
    region_key: 0,
    region_coordination_id: 0,
    zone_key: 0,
    local_commit_id: 0,
    phase_fingerprint: 0,
    world_fingerprint: 0,
};

const EMPTY_REQUEST: EffectRequest = EffectRequest {
    kind: EffectKind::RenderCommands,
    channel: 0,
    flags: 0,
    subject_key: 0,
    payload0: 0,
    payload1: 0,
};

const EMPTY_RECORD: EffectRecord = EffectRecord {
    effect_id: 0,
    source: EMPTY_SOURCE,
    request: EMPTY_REQUEST,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EffectError {
    InvalidSource,
    EmptyBatch,
    BatchTooLarge,
    PipelineFull,
    EffectIdExhausted,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EmitReceipt {
    pub published_count: usize,
    pub first_effect_id: u64,
    pub last_effect_id: u64,
    pub source: SourceVersion,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ConsumeDisposition {
    Ready,
    StaleDiscarded,
    NotReady,
    SourceNotFound,
    NoWork,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ConsumeReceipt {
    pub disposition: ConsumeDisposition,
    pub remaining_count: usize,
    pub effect: Option<EffectRecord>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EffectPipeline<const CAPACITY: usize> {
    next_effect_id: u64,
    count: usize,
    records: [EffectRecord; CAPACITY],
}

impl<const CAPACITY: usize> EffectPipeline<CAPACITY> {
    pub const fn new() -> Self {
        Self {
            next_effect_id: 1,
            count: 0,
            records: [EMPTY_RECORD; CAPACITY],
        }
    }

    pub const fn pending_count(&self) -> usize {
        self.count
    }

    pub const fn next_effect_id(&self) -> u64 {
        self.next_effect_id
    }

    pub fn pending(&self) -> &[EffectRecord] {
        &self.records[..self.count]
    }

    pub fn emit_batch(
        &mut self,
        source: SourceVersion,
        requests: &[EffectRequest],
    ) -> Result<EmitReceipt, EffectError> {
        if !source.is_valid() {
            return Err(EffectError::InvalidSource);
        }
        if requests.is_empty() {
            return Err(EffectError::EmptyBatch);
        }
        if requests.len() > MAX_EMIT_BATCH {
            return Err(EffectError::BatchTooLarge);
        }
        if self.count + requests.len() > CAPACITY {
            return Err(EffectError::PipelineFull);
        }
        if self.next_effect_id == 0
            || self.next_effect_id.checked_add(requests.len() as u64).is_none()
        {
            return Err(EffectError::EffectIdExhausted);
        }

        let mut sorted = [EMPTY_REQUEST; MAX_EMIT_BATCH];
        let mut index = 0;
        while index < requests.len() {
            sorted[index] = requests[index];
            index += 1;
        }
        insertion_sort(&mut sorted[..requests.len()]);

        let first = self.next_effect_id;
        index = 0;
        while index < requests.len() {
            self.records[self.count] = EffectRecord {
                effect_id: self.next_effect_id,
                source,
                request: sorted[index],
            };
            self.next_effect_id += 1;
            self.count += 1;
            index += 1;
        }

        Ok(EmitReceipt {
            published_count: requests.len(),
            first_effect_id: first,
            last_effect_id: self.next_effect_id - 1,
            source,
        })
    }

    pub fn consume_next(&mut self, current: &[SourceVersion]) -> ConsumeReceipt {
        if self.count == 0 {
            return ConsumeReceipt {
                disposition: ConsumeDisposition::NoWork,
                remaining_count: 0,
                effect: None,
            };
        }

        let effect = self.records[0];
        let version = current.iter().copied().find(|candidate| {
            candidate.region_key == effect.source.region_key
                && candidate.zone_key == effect.source.zone_key
        });
        let disposition = match version {
            None => ConsumeDisposition::SourceNotFound,
            Some(current) if current.region_coordination_id < effect.source.region_coordination_id
                || current.local_commit_id < effect.source.local_commit_id =>
            {
                ConsumeDisposition::NotReady
            }
            Some(current)
                if current.region_coordination_id == effect.source.region_coordination_id
                    && current.local_commit_id == effect.source.local_commit_id
                    && current.phase_fingerprint == effect.source.phase_fingerprint
                    && current.world_fingerprint == effect.source.world_fingerprint =>
            {
                ConsumeDisposition::Ready
            }
            Some(_) => ConsumeDisposition::StaleDiscarded,
        };

        if matches!(
            disposition,
            ConsumeDisposition::Ready | ConsumeDisposition::StaleDiscarded
        ) {
            self.remove_first();
        }

        ConsumeReceipt {
            disposition,
            remaining_count: self.count,
            effect: Some(effect),
        }
    }

    fn remove_first(&mut self) {
        let mut index = 1;
        while index < self.count {
            self.records[index - 1] = self.records[index];
            index += 1;
        }
        self.count -= 1;
        self.records[self.count] = EMPTY_RECORD;
    }
}

impl<const CAPACITY: usize> Default for EffectPipeline<CAPACITY> {
    fn default() -> Self {
        Self::new()
    }
}

fn kind_rank(kind: EffectKind) -> u8 {
    match kind {
        EffectKind::RenderCommands => 1,
        EffectKind::AudioCommands => 2,
        EffectKind::StreamRequest => 3,
    }
}

fn request_less(left: EffectRequest, right: EffectRequest) -> bool {
    (
        kind_rank(left.kind),
        left.channel,
        left.subject_key,
        left.flags,
        left.payload0,
        left.payload1,
    ) < (
        kind_rank(right.kind),
        right.channel,
        right.subject_key,
        right.flags,
        right.payload0,
        right.payload1,
    )
}

fn insertion_sort(values: &mut [EffectRequest]) {
    let mut index = 1;
    while index < values.len() {
        let value = values[index];
        let mut cursor = index;
        while cursor > 0 && request_less(value, values[cursor - 1]) {
            values[cursor] = values[cursor - 1];
            cursor -= 1;
        }
        values[cursor] = value;
        index += 1;
    }
}

#[cfg(test)]
mod tests {
    extern crate std;

    use super::*;

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

    #[test]
    fn canonical_order_is_input_permutation_invariant() {
        let tag = source(7, 3, 100, 10);
        let ordered = [
            EffectRequest::render(4, 40),
            EffectRequest::audio(3, 30),
            EffectRequest::stream(2, 20),
        ];
        let permuted = [ordered[2], ordered[0], ordered[1]];
        let mut first = EffectPipeline::<8>::new();
        let mut second = EffectPipeline::<8>::new();
        first.emit_batch(tag, &ordered).unwrap();
        second.emit_batch(tag, &permuted).unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn fresh_effect_is_ready() {
        let tag = source(8, 5, 101, 12);
        let mut pipeline = EffectPipeline::<4>::new();
        pipeline.emit_batch(tag, &[EffectRequest::render(9, 99)]).unwrap();
        let receipt = pipeline.consume_next(&[tag]);
        assert_eq!(receipt.disposition, ConsumeDisposition::Ready);
        assert_eq!(pipeline.pending_count(), 0);
    }

    #[test]
    fn stale_effect_is_discarded_without_simulation_authority() {
        let emitted = source(9, 2, 102, 7);
        let current = source(9, 3, 102, 8);
        let simulation = [0xA5u8; 256];
        let before = simulation;
        let mut pipeline = EffectPipeline::<4>::new();
        pipeline.emit_batch(emitted, &[EffectRequest::audio(5, 55)]).unwrap();
        let receipt = pipeline.consume_next(&[current]);
        assert_eq!(receipt.disposition, ConsumeDisposition::StaleDiscarded);
        assert_eq!(simulation, before);
    }

    #[test]
    fn future_effect_remains_pending() {
        let emitted = source(10, 4, 103, 9);
        let current = source(10, 3, 103, 8);
        let mut pipeline = EffectPipeline::<4>::new();
        pipeline.emit_batch(emitted, &[EffectRequest::stream(44, 77)]).unwrap();
        let before = pipeline.clone();
        let receipt = pipeline.consume_next(&[current]);
        assert_eq!(receipt.disposition, ConsumeDisposition::NotReady);
        assert_eq!(pipeline, before);
    }

    #[test]
    fn capacity_failure_is_atomic() {
        let tag = source(11, 1, 104, 1);
        let mut pipeline = EffectPipeline::<1>::new();
        let before = pipeline.clone();
        assert_eq!(
            pipeline.emit_batch(
                tag,
                &[EffectRequest::render(1, 1), EffectRequest::audio(2, 2)]
            ),
            Err(EffectError::PipelineFull)
        );
        assert_eq!(pipeline, before);
    }

    #[test]
    fn source_versions_are_scoped_versions() {
        let first = source(12, 4, 105, 9);
        let second = source(13, 7, 106, 2);
        let mut pipeline = EffectPipeline::<4>::new();
        pipeline.emit_batch(first, &[EffectRequest::render(1, 1)]).unwrap();
        pipeline.emit_batch(second, &[EffectRequest::audio(2, 2)]).unwrap();
        assert_eq!(pipeline.consume_next(&[first, second]).disposition, ConsumeDisposition::Ready);
        assert_eq!(pipeline.consume_next(&[first, second]).disposition, ConsumeDisposition::Ready);
    }

    #[test]
    fn pipeline_has_no_qbl_or_simulation_mutation_surface() {
        let tag = source(14, 1, 107, 1);
        let mut pipeline = EffectPipeline::<4>::new();
        pipeline.emit_batch(tag, &[EffectRequest::render(1, 1)]).unwrap();
        assert_eq!(pipeline.pending()[0].source, tag);
    }
}
