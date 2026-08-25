#![no_std]

use core::cmp::Ordering;
use core::mem::{align_of, size_of};

pub const WORLD_SCHEMA_VERSION_V0: u32 = 0;
pub const MUTATION_PROTOCOL_VERSION_V0: u32 = 0;
pub const ZONE_COORD_MIN: i32 = -1_048_576;
pub const ZONE_COORD_MAX: i32 = 1_048_575;
pub const ZONE_COORD_BIAS: i64 = 1_048_576;
pub const ZONE_AXIS_BITS: u32 = 21;
pub const ZONE_EDGE_CELLS: i32 = 32;
pub const LOCAL_POSITION_LIMIT_RAW: i32 = ZONE_EDGE_CELLS << 16;
pub const EMPTY_SLOT: u32 = u32::MAX;
pub const NON_SPATIAL_SITE: u16 = u16::MAX;
pub const INVALID_OBJECT_KEY: u64 = 0;
pub const INVALID_GENERATION: u32 = 0;
pub const LOOKUP_EMPTY: u8 = 0;
pub const LOOKUP_OCCUPIED: u8 = 1;
pub const LOOKUP_TOMBSTONE: u8 = 2;
pub const PLAN_NONE: u32 = u32::MAX;
pub const PLAN_REJECTED: u32 = u32::MAX - 1;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ZoneCoord {
    pub x: i32,
    pub y: i32,
    pub z: i32,
}

impl ZoneCoord {
    pub const fn new(x: i32, y: i32, z: i32) -> Result<Self, WorldFault> {
        if x < ZONE_COORD_MIN
            || x > ZONE_COORD_MAX
            || y < ZONE_COORD_MIN
            || y > ZONE_COORD_MAX
            || z < ZONE_COORD_MIN
            || z > ZONE_COORD_MAX
        {
            return Err(WorldFault::ZoneCoordinateRange);
        }
        Ok(Self { x, y, z })
    }

    pub const fn key(self) -> ZoneKey {
        let x = (self.x as i64 + ZONE_COORD_BIAS) as u64;
        let y = (self.y as i64 + ZONE_COORD_BIAS) as u64;
        let z = (self.z as i64 + ZONE_COORD_BIAS) as u64;
        ZoneKey(x | (y << ZONE_AXIS_BITS) | (z << (ZONE_AXIS_BITS * 2)))
    }
}

#[repr(transparent)]
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct ZoneKey(pub u64);

#[repr(transparent)]
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct ObjectKey(pub u64);

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ObjectHandle {
    pub slot: u32,
    pub generation: u32,
}

#[repr(transparent)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LocalPosQ16_16(pub i32);

impl LocalPosQ16_16 {
    pub const fn new(raw: i32) -> Result<Self, WorldFault> {
        if raw < 0 || raw >= LOCAL_POSITION_LIMIT_RAW {
            return Err(WorldFault::LocalPositionRange);
        }
        Ok(Self(raw))
    }
}

#[repr(transparent)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct VelocityQ16_16(pub i32);

#[repr(transparent)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ObjectKind(pub u16);

impl ObjectKind {
    pub const ACTOR: Self = Self(1);
    pub const ITEM: Self = Self(2);

    pub const fn is_valid(self) -> bool {
        self.0 == Self::ACTOR.0 || self.0 == Self::ITEM.0
    }
}

#[repr(transparent)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ReducerClass(pub u8);

impl ReducerClass {
    pub const LIFECYCLE: Self = Self(0);
    pub const EXCLUSIVE: Self = Self(1);
    pub const ADDITIVE: Self = Self(2);
}

#[repr(transparent)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct IntentKind(pub u16);

impl IntentKind {
    pub const DESPAWN_OBJECT: Self = Self(0);
    pub const SPAWN_OBJECT: Self = Self(2);
    pub const REPLACE_KINEMATICS: Self = Self(3);
    pub const APPLY_HEALTH_DELTA: Self = Self(6);
}

#[repr(transparent)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CauseKind(pub u16);

impl CauseKind {
    pub const EXTERNAL_INPUT: Self = Self(1);
    pub const LOCAL_INTEGRATION: Self = Self(2);
    pub const BOUNDARY_MESSAGE: Self = Self(3);
    pub const RETAINED_WORK: Self = Self(4);
    pub const SUBSYSTEM_EVENT: Self = Self(5);

    pub const fn is_valid(self) -> bool {
        self.0 >= Self::EXTERNAL_INPUT.0 && self.0 <= Self::SUBSYSTEM_EVENT.0
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct AdmittedCauseV0 {
    pub lane_rank: u8,
    pub subsystem_rank: u8,
    pub kind: u16,
    pub source_sequence: u32,
    pub source_zone: u64,
    pub source_object: u64,
    pub payload0: u64,
    pub payload1: u64,
}

impl AdmittedCauseV0 {
    pub const ZERO: Self = Self {
        lane_rank: 0,
        subsystem_rank: 0,
        kind: 0,
        source_sequence: 0,
        source_zone: 0,
        source_object: 0,
        payload0: 0,
        payload1: 0,
    };

    pub const fn external_input(source_sequence: u32, payload0: u64, payload1: u64) -> Self {
        Self {
            lane_rank: 0,
            subsystem_rank: 0,
            kind: CauseKind::EXTERNAL_INPUT.0,
            source_sequence,
            source_zone: 0,
            source_object: 0,
            payload0,
            payload1,
        }
    }

    fn canonical_cmp(&self, other: &Self) -> Ordering {
        (
            self.lane_rank,
            self.subsystem_rank,
            self.source_zone,
            self.source_object,
            self.source_sequence,
            self.kind,
        )
            .cmp(&(
                other.lane_rank,
                other.subsystem_rank,
                other.source_zone,
                other.source_object,
                other.source_sequence,
                other.kind,
            ))
    }

    fn same_source_sequence(&self, other: &Self) -> bool {
        self.lane_rank == other.lane_rank
            && self.subsystem_rank == other.subsystem_rank
            && self.source_zone == other.source_zone
            && self.source_object == other.source_object
            && self.source_sequence == other.source_sequence
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct MutationIntentV0 {
    pub lane_rank: u8,
    pub subsystem_rank: u8,
    pub kind: u16,
    pub flags: u32,
    pub source_zone: u64,
    pub source_object: u64,
    pub source_sequence: u32,
    pub target_field: u16,
    pub reducer: u8,
    pub reserved: u8,
    pub target: u64,
    pub aux: u64,
    pub value0: u64,
    pub value1: u64,
}

impl MutationIntentV0 {
    pub const ZERO: Self = Self {
        lane_rank: 0,
        subsystem_rank: 0,
        kind: 0,
        flags: 0,
        source_zone: 0,
        source_object: 0,
        source_sequence: 0,
        target_field: 0,
        reducer: 0,
        reserved: 0,
        target: 0,
        aux: 0,
        value0: 0,
        value1: 0,
    };

    pub const fn spawn_actor(
        source_sequence: u32,
        health: u16,
        surface_site: u16,
    ) -> Self {
        Self {
            lane_rank: 0,
            subsystem_rank: 0,
            kind: IntentKind::SPAWN_OBJECT.0,
            flags: 0,
            source_zone: 0,
            source_object: 0,
            source_sequence,
            target_field: 0,
            reducer: ReducerClass::LIFECYCLE.0,
            reserved: 0,
            target: 0,
            aux: (ObjectKind::ACTOR.0 as u64) | ((surface_site as u64) << 16),
            value0: health as u64,
            value1: 0,
        }
    }

    pub const fn spawn_item(
        source_sequence: u32,
        item_type: u16,
        item_stack: u16,
        surface_site: u16,
    ) -> Self {
        Self {
            lane_rank: 0,
            subsystem_rank: 0,
            kind: IntentKind::SPAWN_OBJECT.0,
            flags: 0,
            source_zone: 0,
            source_object: 0,
            source_sequence,
            target_field: 0,
            reducer: ReducerClass::LIFECYCLE.0,
            reserved: 0,
            target: 0,
            aux: (ObjectKind::ITEM.0 as u64) | ((surface_site as u64) << 16),
            value0: (item_type as u64) | ((item_stack as u64) << 16),
            value1: 0,
        }
    }

    pub const fn despawn(source_sequence: u32, target: ObjectKey) -> Self {
        Self {
            lane_rank: 0,
            subsystem_rank: 0,
            kind: IntentKind::DESPAWN_OBJECT.0,
            flags: 0,
            source_zone: 0,
            source_object: 0,
            source_sequence,
            target_field: 0,
            reducer: ReducerClass::LIFECYCLE.0,
            reserved: 0,
            target: target.0,
            aux: 0,
            value0: 0,
            value1: 0,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub const fn replace_kinematics(
        source_sequence: u32,
        target: ObjectKey,
        pos_x: i32,
        pos_y: i32,
        pos_z: i32,
        vel_x: i32,
        vel_y: i32,
        vel_z: i32,
    ) -> Self {
        Self {
            lane_rank: 0,
            subsystem_rank: 0,
            kind: IntentKind::REPLACE_KINEMATICS.0,
            flags: 0,
            source_zone: 0,
            source_object: 0,
            source_sequence,
            target_field: 0,
            reducer: ReducerClass::EXCLUSIVE.0,
            reserved: 0,
            target: target.0,
            aux: pack_i32_pair(vel_y, vel_z),
            value0: pack_i32_pair(pos_x, pos_y),
            value1: pack_i32_pair(pos_z, vel_x),
        }
    }

    pub const fn apply_health_delta(
        source_sequence: u32,
        target: ObjectKey,
        delta: i64,
    ) -> Self {
        Self {
            lane_rank: 0,
            subsystem_rank: 0,
            kind: IntentKind::APPLY_HEALTH_DELTA.0,
            flags: 0,
            source_zone: 0,
            source_object: 0,
            source_sequence,
            target_field: 0,
            reducer: ReducerClass::ADDITIVE.0,
            reserved: 0,
            target: target.0,
            aux: 0,
            value0: delta as u64,
            value1: 0,
        }
    }

    fn canonical_cmp(&self, other: &Self) -> Ordering {
        (
            self.lane_rank,
            self.subsystem_rank,
            self.source_zone,
            self.source_object,
            self.source_sequence,
            self.kind,
        )
            .cmp(&(
                other.lane_rank,
                other.subsystem_rank,
                other.source_zone,
                other.source_object,
                other.source_sequence,
                other.kind,
            ))
    }

    fn same_source_sequence(&self, other: &Self) -> bool {
        self.lane_rank == other.lane_rank
            && self.subsystem_rank == other.subsystem_rank
            && self.source_zone == other.source_zone
            && self.source_object == other.source_object
            && self.source_sequence == other.source_sequence
    }
}

const fn pack_i32_pair(low: i32, high: i32) -> u64 {
    (low as u32 as u64) | ((high as u32 as u64) << 32)
}

const fn unpack_i32_low(value: u64) -> i32 {
    value as u32 as i32
}

const fn unpack_i32_high(value: u64) -> i32 {
    (value >> 32) as u32 as i32
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WorldFault {
    ZoneCoordinateRange,
    LocalPositionRange,
    InvalidProfile,
    CauseCapacity,
    IntentCapacity,
    CauseMissing,
    InvalidCause,
    DuplicateCauseSequence,
    InvalidIntent,
    DuplicateIntentSequence,
    ObjectNotFound,
    DuplicateLifecycle,
    LifecycleWriteConflict,
    ExclusiveConflict,
    WideAccumulatorOverflow,
    ObjectKeyExhausted,
    LookupInvariant,
    TransitionIdExhausted,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RequestRejection {
    ObjectCapacity,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TransitionDisposition {
    NoWork,
    Committed,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TransitionResult {
    pub disposition: TransitionDisposition,
    pub transition_id: u64,
    pub accepted_intents: u32,
    pub rejected_requests: u32,
    pub diagnostic_fingerprint: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WorldTransitionError {
    Fault(WorldFault),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ZoneHeaderV0 {
    pub key: ZoneKey,
    pub schema_version: u32,
    pub generator_version: u32,
    pub content_rule_version: u32,
    pub next_generation_receipt: u32,
    pub object_count: u32,
    pub relation_count: u32,
    pub overlay_count: u32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ObjectStore<const OBJECTS: usize, const BUCKETS: usize> {
    alive: [u8; OBJECTS],
    generation: [u32; OBJECTS],
    object_key: [u64; OBJECTS],
    kind: [u16; OBJECTS],
    flags: [u16; OBJECTS],
    surface_site: [u16; OBJECTS],
    next_in_site: [u32; OBJECTS],
    pos_x: [i32; OBJECTS],
    pos_y: [i32; OBJECTS],
    pos_z: [i32; OBJECTS],
    vel_x: [i32; OBJECTS],
    vel_y: [i32; OBJECTS],
    vel_z: [i32; OBJECTS],
    orientation: [u16; OBJECTS],
    actor_health: [u16; OBJECTS],
    item_type: [u16; OBJECTS],
    item_stack: [u16; OBJECTS],
    bucket_key: [u64; BUCKETS],
    bucket_slot: [u32; BUCKETS],
    bucket_state: [u8; BUCKETS],
}

impl<const OBJECTS: usize, const BUCKETS: usize> ObjectStore<OBJECTS, BUCKETS> {
    pub const fn new() -> Self {
        Self {
            alive: [0; OBJECTS],
            generation: [1; OBJECTS],
            object_key: [0; OBJECTS],
            kind: [0; OBJECTS],
            flags: [0; OBJECTS],
            surface_site: [NON_SPATIAL_SITE; OBJECTS],
            next_in_site: [EMPTY_SLOT; OBJECTS],
            pos_x: [0; OBJECTS],
            pos_y: [0; OBJECTS],
            pos_z: [0; OBJECTS],
            vel_x: [0; OBJECTS],
            vel_y: [0; OBJECTS],
            vel_z: [0; OBJECTS],
            orientation: [0; OBJECTS],
            actor_health: [0; OBJECTS],
            item_type: [0; OBJECTS],
            item_stack: [0; OBJECTS],
            bucket_key: [0; BUCKETS],
            bucket_slot: [EMPTY_SLOT; BUCKETS],
            bucket_state: [LOOKUP_EMPTY; BUCKETS],
        }
    }

    pub fn handle_for_key(&self, key: ObjectKey) -> Option<ObjectHandle> {
        let slot = self.resolve_key(key)?;
        Some(ObjectHandle {
            slot: slot as u32,
            generation: self.generation[slot],
        })
    }

    pub fn validate_handle(&self, handle: ObjectHandle) -> Option<ObjectKey> {
        let slot = handle.slot as usize;
        if slot >= OBJECTS
            || self.alive[slot] == 0
            || self.generation[slot] != handle.generation
        {
            return None;
        }
        Some(ObjectKey(self.object_key[slot]))
    }

    pub fn object_key_at(&self, slot: usize) -> Option<ObjectKey> {
        if slot >= OBJECTS || self.alive[slot] == 0 {
            None
        } else {
            Some(ObjectKey(self.object_key[slot]))
        }
    }

    pub fn health(&self, key: ObjectKey) -> Option<u16> {
        self.resolve_key(key).map(|slot| self.actor_health[slot])
    }

    pub fn kinematics(
        &self,
        key: ObjectKey,
    ) -> Option<(i32, i32, i32, i32, i32, i32)> {
        self.resolve_key(key).map(|slot| {
            (
                self.pos_x[slot],
                self.pos_y[slot],
                self.pos_z[slot],
                self.vel_x[slot],
                self.vel_y[slot],
                self.vel_z[slot],
            )
        })
    }

    pub fn alive_count(&self) -> usize {
        self.alive.iter().filter(|&&value| value != 0).count()
    }

    fn resolve_key(&self, key: ObjectKey) -> Option<usize> {
        if key.0 == INVALID_OBJECT_KEY || BUCKETS == 0 {
            return None;
        }
        let start = (mix64(key.0) as usize) % BUCKETS;
        let mut probe = 0;
        while probe < BUCKETS {
            let index = (start + probe) % BUCKETS;
            match self.bucket_state[index] {
                LOOKUP_EMPTY => return None,
                LOOKUP_OCCUPIED if self.bucket_key[index] == key.0 => {
                    let slot = self.bucket_slot[index] as usize;
                    if slot < OBJECTS
                        && self.alive[slot] != 0
                        && self.object_key[slot] == key.0
                    {
                        return Some(slot);
                    }
                    return None;
                }
                _ => {}
            }
            probe += 1;
        }
        None
    }

    fn lookup_insert(&mut self, key: ObjectKey, slot: usize) -> Result<(), WorldFault> {
        if key.0 == INVALID_OBJECT_KEY || BUCKETS == 0 {
            return Err(WorldFault::LookupInvariant);
        }
        let start = (mix64(key.0) as usize) % BUCKETS;
        let mut first_tombstone = None;
        let mut probe = 0;
        while probe < BUCKETS {
            let index = (start + probe) % BUCKETS;
            match self.bucket_state[index] {
                LOOKUP_EMPTY => {
                    let target = first_tombstone.unwrap_or(index);
                    self.bucket_key[target] = key.0;
                    self.bucket_slot[target] = slot as u32;
                    self.bucket_state[target] = LOOKUP_OCCUPIED;
                    return Ok(());
                }
                LOOKUP_TOMBSTONE => {
                    if first_tombstone.is_none() {
                        first_tombstone = Some(index);
                    }
                }
                LOOKUP_OCCUPIED if self.bucket_key[index] == key.0 => {
                    return Err(WorldFault::LookupInvariant);
                }
                _ => {}
            }
            probe += 1;
        }
        if let Some(target) = first_tombstone {
            self.bucket_key[target] = key.0;
            self.bucket_slot[target] = slot as u32;
            self.bucket_state[target] = LOOKUP_OCCUPIED;
            return Ok(());
        }
        Err(WorldFault::LookupInvariant)
    }

    fn lookup_remove(&mut self, key: ObjectKey) -> Result<(), WorldFault> {
        if key.0 == INVALID_OBJECT_KEY || BUCKETS == 0 {
            return Err(WorldFault::LookupInvariant);
        }
        let start = (mix64(key.0) as usize) % BUCKETS;
        let mut probe = 0;
        while probe < BUCKETS {
            let index = (start + probe) % BUCKETS;
            match self.bucket_state[index] {
                LOOKUP_EMPTY => return Err(WorldFault::LookupInvariant),
                LOOKUP_OCCUPIED if self.bucket_key[index] == key.0 => {
                    self.bucket_key[index] = 0;
                    self.bucket_slot[index] = EMPTY_SLOT;
                    self.bucket_state[index] = LOOKUP_TOMBSTONE;
                    return Ok(());
                }
                _ => {}
            }
            probe += 1;
        }
        Err(WorldFault::LookupInvariant)
    }

    fn clear_slot_payload(&mut self, slot: usize) {
        self.object_key[slot] = 0;
        self.kind[slot] = 0;
        self.flags[slot] = 0;
        self.surface_site[slot] = NON_SPATIAL_SITE;
        self.next_in_site[slot] = EMPTY_SLOT;
        self.pos_x[slot] = 0;
        self.pos_y[slot] = 0;
        self.pos_z[slot] = 0;
        self.vel_x[slot] = 0;
        self.vel_y[slot] = 0;
        self.vel_z[slot] = 0;
        self.orientation[slot] = 0;
        self.actor_health[slot] = 0;
        self.item_type[slot] = 0;
        self.item_stack[slot] = 0;
    }

    fn despawn_slot(&mut self, slot: usize) -> Result<(), WorldFault> {
        let key = ObjectKey(self.object_key[slot]);
        self.lookup_remove(key)?;
        self.alive[slot] = 0;
        self.clear_slot_payload(slot);
        let mut next = self.generation[slot].wrapping_add(1);
        if next == INVALID_GENERATION {
            next = 1;
        }
        self.generation[slot] = next;
        Ok(())
    }

    fn spawn_slot(
        &mut self,
        slot: usize,
        key: ObjectKey,
        intent: &MutationIntentV0,
    ) -> Result<(), WorldFault> {
        let kind = ObjectKind((intent.aux & 0xffff) as u16);
        let site = ((intent.aux >> 16) & 0xffff) as u16;
        self.lookup_insert(key, slot)?;
        self.alive[slot] = 1;
        self.object_key[slot] = key.0;
        self.kind[slot] = kind.0;
        self.flags[slot] = intent.flags as u16;
        self.surface_site[slot] = site;
        self.next_in_site[slot] = EMPTY_SLOT;
        if kind == ObjectKind::ACTOR {
            self.actor_health[slot] = (intent.value0 & 0xffff) as u16;
        } else {
            self.item_type[slot] = (intent.value0 & 0xffff) as u16;
            self.item_stack[slot] = ((intent.value0 >> 16) & 0xffff) as u16;
        }
        Ok(())
    }
}

impl<const OBJECTS: usize, const BUCKETS: usize> Default for ObjectStore<OBJECTS, BUCKETS> {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct WorldSlotImageV0 {
    pub alive: u8,
    pub generation: u32,
    pub object_key: u64,
    pub kind: u16,
    pub flags: u16,
    pub surface_site: u16,
    pub next_in_site: u32,
    pub pos_x: i32,
    pub pos_y: i32,
    pub pos_z: i32,
    pub vel_x: i32,
    pub vel_y: i32,
    pub vel_z: i32,
    pub orientation: u16,
    pub actor_health: u16,
    pub item_type: u16,
    pub item_stack: u16,
}

impl WorldSlotImageV0 {
    pub const EMPTY: Self = Self {
        alive: 0,
        generation: 1,
        object_key: 0,
        kind: 0,
        flags: 0,
        surface_site: NON_SPATIAL_SITE,
        next_in_site: EMPTY_SLOT,
        pos_x: 0,
        pos_y: 0,
        pos_z: 0,
        vel_x: 0,
        vel_y: 0,
        vel_z: 0,
        orientation: 0,
        actor_health: 0,
        item_type: 0,
        item_stack: 0,
    };
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WorldImageV0<const OBJECTS: usize> {
    pub accepted_transition_id: u64,
    pub next_object_key: u64,
    pub zone: ZoneHeaderV0,
    pub slots: [WorldSlotImageV0; OBJECTS],
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WorldCore<const OBJECTS: usize, const BUCKETS: usize> {
    accepted_transition_id: u64,
    next_object_key: u64,
    zone: ZoneHeaderV0,
    objects: ObjectStore<OBJECTS, BUCKETS>,
}

impl<const OBJECTS: usize, const BUCKETS: usize> WorldCore<OBJECTS, BUCKETS> {
    pub fn accepted_transition_id(&self) -> u64 {
        self.accepted_transition_id
    }

    pub fn next_object_key(&self) -> u64 {
        self.next_object_key
    }

    pub fn zone(&self) -> &ZoneHeaderV0 {
        &self.zone
    }

    pub fn objects(&self) -> &ObjectStore<OBJECTS, BUCKETS> {
        &self.objects
    }

    pub fn export_image(&self) -> WorldImageV0<OBJECTS> {
        let mut slots = [WorldSlotImageV0::EMPTY; OBJECTS];
        for (slot, image) in slots.iter_mut().enumerate() {
            *image = WorldSlotImageV0 {
                alive: self.objects.alive[slot],
                generation: self.objects.generation[slot],
                object_key: self.objects.object_key[slot],
                kind: self.objects.kind[slot],
                flags: self.objects.flags[slot],
                surface_site: self.objects.surface_site[slot],
                next_in_site: self.objects.next_in_site[slot],
                pos_x: self.objects.pos_x[slot],
                pos_y: self.objects.pos_y[slot],
                pos_z: self.objects.pos_z[slot],
                vel_x: self.objects.vel_x[slot],
                vel_y: self.objects.vel_y[slot],
                vel_z: self.objects.vel_z[slot],
                orientation: self.objects.orientation[slot],
                actor_health: self.objects.actor_health[slot],
                item_type: self.objects.item_type[slot],
                item_stack: self.objects.item_stack[slot],
            };
        }
        WorldImageV0 {
            accepted_transition_id: self.accepted_transition_id,
            next_object_key: self.next_object_key,
            zone: self.zone,
            slots,
        }
    }

    pub fn from_image(image: &WorldImageV0<OBJECTS>) -> Result<Self, WorldFault> {
        if OBJECTS == 0 || BUCKETS < OBJECTS.saturating_mul(2) {
            return Err(WorldFault::InvalidProfile);
        }
        if image.zone.schema_version != WORLD_SCHEMA_VERSION_V0
            || (image.zone.key.0 >> 63) != 0
            || image.zone.relation_count != 0
            || image.zone.overlay_count != 0
            || image.next_object_key == INVALID_OBJECT_KEY
        {
            return Err(WorldFault::InvalidProfile);
        }

        let mut alive_count = 0_u32;
        let mut max_key = 0_u64;
        for (index, slot) in image.slots.iter().enumerate() {
            if slot.generation == INVALID_GENERATION {
                return Err(WorldFault::LookupInvariant);
            }
            if slot.alive == 0 {
                if slot.object_key != 0
                    || slot.kind != 0
                    || slot.flags != 0
                    || slot.surface_site != NON_SPATIAL_SITE
                    || slot.next_in_site != EMPTY_SLOT
                    || slot.pos_x != 0
                    || slot.pos_y != 0
                    || slot.pos_z != 0
                    || slot.vel_x != 0
                    || slot.vel_y != 0
                    || slot.vel_z != 0
                    || slot.orientation != 0
                    || slot.actor_health != 0
                    || slot.item_type != 0
                    || slot.item_stack != 0
                {
                    return Err(WorldFault::LookupInvariant);
                }
                continue;
            }
            if slot.alive != 1
                || slot.object_key == INVALID_OBJECT_KEY
                || slot.object_key >= image.next_object_key
                || !ObjectKind(slot.kind).is_valid()
                || slot.next_in_site != EMPTY_SLOT
                || (slot.surface_site != NON_SPATIAL_SITE && slot.surface_site >= 1024)
                || LocalPosQ16_16::new(slot.pos_x).is_err()
                || LocalPosQ16_16::new(slot.pos_y).is_err()
                || LocalPosQ16_16::new(slot.pos_z).is_err()
            {
                return Err(WorldFault::LookupInvariant);
            }
            if ObjectKind(slot.kind) == ObjectKind::ACTOR {
                if slot.item_type != 0 || slot.item_stack != 0 {
                    return Err(WorldFault::LookupInvariant);
                }
            } else if slot.actor_health != 0 {
                return Err(WorldFault::LookupInvariant);
            }
            for prior in &image.slots[..index] {
                if prior.alive != 0 && prior.object_key == slot.object_key {
                    return Err(WorldFault::LookupInvariant);
                }
            }
            alive_count = alive_count.saturating_add(1);
            max_key = max_key.max(slot.object_key);
        }
        if alive_count != image.zone.object_count || max_key >= image.next_object_key {
            return Err(WorldFault::LookupInvariant);
        }

        let mut objects = ObjectStore::<OBJECTS, BUCKETS>::new();
        for (index, slot) in image.slots.iter().enumerate() {
            objects.alive[index] = slot.alive;
            objects.generation[index] = slot.generation;
            objects.object_key[index] = slot.object_key;
            objects.kind[index] = slot.kind;
            objects.flags[index] = slot.flags;
            objects.surface_site[index] = slot.surface_site;
            objects.next_in_site[index] = slot.next_in_site;
            objects.pos_x[index] = slot.pos_x;
            objects.pos_y[index] = slot.pos_y;
            objects.pos_z[index] = slot.pos_z;
            objects.vel_x[index] = slot.vel_x;
            objects.vel_y[index] = slot.vel_y;
            objects.vel_z[index] = slot.vel_z;
            objects.orientation[index] = slot.orientation;
            objects.actor_health[index] = slot.actor_health;
            objects.item_type[index] = slot.item_type;
            objects.item_stack[index] = slot.item_stack;
            if slot.alive != 0 {
                objects.lookup_insert(ObjectKey(slot.object_key), index)?;
            }
        }

        Ok(Self {
            accepted_transition_id: image.accepted_transition_id,
            next_object_key: image.next_object_key,
            zone: image.zone,
            objects,
        })
    }

    pub fn new(coord: ZoneCoord) -> Result<Self, WorldFault> {
        if OBJECTS == 0 || BUCKETS < OBJECTS.saturating_mul(2) {
            return Err(WorldFault::InvalidProfile);
        }
        Ok(Self {
            accepted_transition_id: 0,
            next_object_key: 1,
            zone: ZoneHeaderV0 {
                key: coord.key(),
                schema_version: WORLD_SCHEMA_VERSION_V0,
                generator_version: 0,
                content_rule_version: 0,
                next_generation_receipt: 1,
                object_count: 0,
                relation_count: 0,
                overlay_count: 0,
            },
            objects: ObjectStore::new(),
        })
    }

    pub fn transact<const CAUSES: usize, const INTENTS: usize>(
        &mut self,
        causes: &[AdmittedCauseV0],
        intents: &[MutationIntentV0],
        scratch: &mut TransitionScratch<CAUSES, INTENTS, OBJECTS>,
    ) -> Result<TransitionResult, WorldTransitionError> {
        if causes.is_empty() {
            if intents.is_empty() {
                return Ok(TransitionResult {
                    disposition: TransitionDisposition::NoWork,
                    transition_id: self.accepted_transition_id,
                    accepted_intents: 0,
                    rejected_requests: 0,
                    diagnostic_fingerprint: self.diagnostic_fingerprint64(),
                });
            }
            return Err(WorldTransitionError::Fault(WorldFault::CauseMissing));
        }
        if causes.len() > CAUSES {
            return Err(WorldTransitionError::Fault(WorldFault::CauseCapacity));
        }
        if intents.len() > INTENTS {
            return Err(WorldTransitionError::Fault(WorldFault::IntentCapacity));
        }
        if self.accepted_transition_id == u64::MAX {
            return Err(WorldTransitionError::Fault(
                WorldFault::TransitionIdExhausted,
            ));
        }

        scratch.reset();
        scratch.causes[..causes.len()].copy_from_slice(causes);
        sort_causes(&mut scratch.causes[..causes.len()]);
        for cause in &scratch.causes[..causes.len()] {
            if !CauseKind(cause.kind).is_valid() {
                return Err(WorldTransitionError::Fault(WorldFault::InvalidCause));
            }
        }
        for pair in scratch.causes[..causes.len()].windows(2) {
            if pair[0].same_source_sequence(&pair[1]) {
                return Err(WorldTransitionError::Fault(
                    WorldFault::DuplicateCauseSequence,
                ));
            }
        }

        scratch.intents[..intents.len()].copy_from_slice(intents);
        sort_intents(&mut scratch.intents[..intents.len()]);
        for pair in scratch.intents[..intents.len()].windows(2) {
            if pair[0].same_source_sequence(&pair[1]) {
                return Err(WorldTransitionError::Fault(
                    WorldFault::DuplicateIntentSequence,
                ));
            }
        }

        scratch.simulated_alive.copy_from_slice(&self.objects.alive);
        let mut accepted_spawns = 0_u64;
        let mut rejected = 0_u32;

        for index in 0..intents.len() {
            let intent = scratch.intents[index];
            validate_intent_encoding(&intent)
                .map_err(|fault| WorldTransitionError::Fault(fault))?;
            match IntentKind(intent.kind) {
                IntentKind::DESPAWN_OBJECT => {
                    let slot = self
                        .objects
                        .resolve_key(ObjectKey(intent.target))
                        .ok_or(WorldTransitionError::Fault(WorldFault::ObjectNotFound))?;
                    if scratch.lifecycle_seen[slot] != 0 {
                        return Err(WorldTransitionError::Fault(
                            WorldFault::DuplicateLifecycle,
                        ));
                    }
                    scratch.lifecycle_seen[slot] = 1;
                    scratch.plan_slots[index] = slot as u32;
                    scratch.simulated_alive[slot] = 0;
                }
                IntentKind::SPAWN_OBJECT => {
                    let mut slot = None;
                    for candidate in 0..OBJECTS {
                        if scratch.simulated_alive[candidate] == 0 {
                            slot = Some(candidate);
                            break;
                        }
                    }
                    if let Some(slot) = slot {
                        let key = self
                            .next_object_key
                            .checked_add(accepted_spawns)
                            .ok_or(WorldTransitionError::Fault(
                                WorldFault::ObjectKeyExhausted,
                            ))?;
                        if key == INVALID_OBJECT_KEY {
                            return Err(WorldTransitionError::Fault(
                                WorldFault::ObjectKeyExhausted,
                            ));
                        }
                        scratch.plan_slots[index] = slot as u32;
                        scratch.plan_keys[index] = key;
                        scratch.simulated_alive[slot] = 1;
                        accepted_spawns += 1;
                    } else {
                        scratch.plan_slots[index] = PLAN_REJECTED;
                        rejected = rejected.saturating_add(1);
                    }
                }
                IntentKind::REPLACE_KINEMATICS => {
                    let slot = self
                        .objects
                        .resolve_key(ObjectKey(intent.target))
                        .ok_or(WorldTransitionError::Fault(WorldFault::ObjectNotFound))?;
                    if scratch.exclusive_seen[slot] != 0 {
                        return Err(WorldTransitionError::Fault(
                            WorldFault::ExclusiveConflict,
                        ));
                    }
                    scratch.exclusive_seen[slot] = 1;
                    scratch.plan_slots[index] = slot as u32;
                    let px = unpack_i32_low(intent.value0);
                    let py = unpack_i32_high(intent.value0);
                    let pz = unpack_i32_low(intent.value1);
                    if LocalPosQ16_16::new(px).is_err()
                        || LocalPosQ16_16::new(py).is_err()
                        || LocalPosQ16_16::new(pz).is_err()
                    {
                        return Err(WorldTransitionError::Fault(
                            WorldFault::LocalPositionRange,
                        ));
                    }
                }
                IntentKind::APPLY_HEALTH_DELTA => {
                    let slot = self
                        .objects
                        .resolve_key(ObjectKey(intent.target))
                        .ok_or(WorldTransitionError::Fault(WorldFault::ObjectNotFound))?;
                    if self.objects.kind[slot] != ObjectKind::ACTOR.0 {
                        return Err(WorldTransitionError::Fault(WorldFault::InvalidIntent));
                    }
                    scratch.plan_slots[index] = slot as u32;
                    let delta = intent.value0 as i64 as i128;
                    let next = scratch.health_accum[slot] + delta;
                    if next < i64::MIN as i128 || next > i64::MAX as i128 {
                        return Err(WorldTransitionError::Fault(
                            WorldFault::WideAccumulatorOverflow,
                        ));
                    }
                    scratch.health_accum[slot] = next;
                    scratch.health_touched[slot] = 1;
                }
                _ => {
                    return Err(WorldTransitionError::Fault(WorldFault::InvalidIntent));
                }
            }
        }

        for slot in 0..OBJECTS {
            if scratch.lifecycle_seen[slot] != 0
                && (scratch.exclusive_seen[slot] != 0 || scratch.health_touched[slot] != 0)
            {
                return Err(WorldTransitionError::Fault(
                    WorldFault::LifecycleWriteConflict,
                ));
            }
        }

        let next_key = self
            .next_object_key
            .checked_add(accepted_spawns)
            .ok_or(WorldTransitionError::Fault(
                WorldFault::ObjectKeyExhausted,
            ))?;

        for index in 0..intents.len() {
            let intent = scratch.intents[index];
            if IntentKind(intent.kind) == IntentKind::DESPAWN_OBJECT {
                let slot = scratch.plan_slots[index] as usize;
                self.objects
                    .despawn_slot(slot)
                    .map_err(|fault| WorldTransitionError::Fault(fault))?;
                self.zone.object_count -= 1;
            }
        }

        for index in 0..intents.len() {
            let intent = scratch.intents[index];
            if IntentKind(intent.kind) == IntentKind::SPAWN_OBJECT
                && scratch.plan_slots[index] != PLAN_REJECTED
            {
                let slot = scratch.plan_slots[index] as usize;
                let key = ObjectKey(scratch.plan_keys[index]);
                self.objects
                    .spawn_slot(slot, key, &intent)
                    .map_err(|fault| WorldTransitionError::Fault(fault))?;
                self.zone.object_count += 1;
            }
        }

        for index in 0..intents.len() {
            let intent = scratch.intents[index];
            if IntentKind(intent.kind) == IntentKind::REPLACE_KINEMATICS {
                let slot = scratch.plan_slots[index] as usize;
                self.objects.pos_x[slot] = unpack_i32_low(intent.value0);
                self.objects.pos_y[slot] = unpack_i32_high(intent.value0);
                self.objects.pos_z[slot] = unpack_i32_low(intent.value1);
                self.objects.vel_x[slot] = unpack_i32_high(intent.value1);
                self.objects.vel_y[slot] = unpack_i32_low(intent.aux);
                self.objects.vel_z[slot] = unpack_i32_high(intent.aux);
            }
        }

        for slot in 0..OBJECTS {
            if scratch.health_touched[slot] != 0 {
                let base = self.objects.actor_health[slot] as i128;
                let value = base + scratch.health_accum[slot];
                self.objects.actor_health[slot] = if value <= 0 {
                    0
                } else if value >= u16::MAX as i128 {
                    u16::MAX
                } else {
                    value as u16
                };
            }
        }

        self.next_object_key = next_key;
        self.accepted_transition_id += 1;

        Ok(TransitionResult {
            disposition: TransitionDisposition::Committed,
            transition_id: self.accepted_transition_id,
            accepted_intents: intents.len() as u32 - rejected,
            rejected_requests: rejected,
            diagnostic_fingerprint: self.diagnostic_fingerprint64(),
        })
    }

    pub fn diagnostic_fingerprint64(&self) -> u64 {
        let mut hash = 0xcbf29ce484222325_u64;
        fn feed(hash: &mut u64, bytes: &[u8]) {
            for byte in bytes {
                *hash ^= *byte as u64;
                *hash = hash.wrapping_mul(0x100000001b3);
            }
        }
        feed(&mut hash, &self.accepted_transition_id.to_le_bytes());
        feed(&mut hash, &self.next_object_key.to_le_bytes());
        feed(&mut hash, &self.zone.key.0.to_le_bytes());
        feed(&mut hash, &self.zone.object_count.to_le_bytes());
        for slot in 0..OBJECTS {
            feed(&mut hash, &[self.objects.alive[slot]]);
            feed(&mut hash, &self.objects.generation[slot].to_le_bytes());
            feed(&mut hash, &self.objects.object_key[slot].to_le_bytes());
            feed(&mut hash, &self.objects.kind[slot].to_le_bytes());
            feed(&mut hash, &self.objects.flags[slot].to_le_bytes());
            feed(&mut hash, &self.objects.surface_site[slot].to_le_bytes());
            feed(&mut hash, &self.objects.pos_x[slot].to_le_bytes());
            feed(&mut hash, &self.objects.pos_y[slot].to_le_bytes());
            feed(&mut hash, &self.objects.pos_z[slot].to_le_bytes());
            feed(&mut hash, &self.objects.vel_x[slot].to_le_bytes());
            feed(&mut hash, &self.objects.vel_y[slot].to_le_bytes());
            feed(&mut hash, &self.objects.vel_z[slot].to_le_bytes());
            feed(&mut hash, &self.objects.actor_health[slot].to_le_bytes());
            feed(&mut hash, &self.objects.item_type[slot].to_le_bytes());
            feed(&mut hash, &self.objects.item_stack[slot].to_le_bytes());
        }
        hash
    }
}

pub type WorldCoreV0 = WorldCore<2048, 4096>;

#[derive(Clone, Debug)]
pub struct TransitionScratch<const CAUSES: usize, const INTENTS: usize, const OBJECTS: usize> {
    causes: [AdmittedCauseV0; CAUSES],
    intents: [MutationIntentV0; INTENTS],
    simulated_alive: [u8; OBJECTS],
    lifecycle_seen: [u8; OBJECTS],
    exclusive_seen: [u8; OBJECTS],
    health_touched: [u8; OBJECTS],
    health_accum: [i128; OBJECTS],
    plan_slots: [u32; INTENTS],
    plan_keys: [u64; INTENTS],
}

impl<const CAUSES: usize, const INTENTS: usize, const OBJECTS: usize>
    TransitionScratch<CAUSES, INTENTS, OBJECTS>
{
    pub const fn new() -> Self {
        Self {
            causes: [AdmittedCauseV0::ZERO; CAUSES],
            intents: [MutationIntentV0::ZERO; INTENTS],
            simulated_alive: [0; OBJECTS],
            lifecycle_seen: [0; OBJECTS],
            exclusive_seen: [0; OBJECTS],
            health_touched: [0; OBJECTS],
            health_accum: [0; OBJECTS],
            plan_slots: [PLAN_NONE; INTENTS],
            plan_keys: [0; INTENTS],
        }
    }

    fn reset(&mut self) {
        self.lifecycle_seen.fill(0);
        self.exclusive_seen.fill(0);
        self.health_touched.fill(0);
        self.health_accum.fill(0);
        self.plan_slots.fill(PLAN_NONE);
        self.plan_keys.fill(0);
    }
}

impl<const CAUSES: usize, const INTENTS: usize, const OBJECTS: usize> Default
    for TransitionScratch<CAUSES, INTENTS, OBJECTS>
{
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum XiPortStatus {
    NotDerived,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct InertXiPort;

impl InertXiPort {
    pub const fn evaluate(self) -> XiPortStatus {
        XiPortStatus::NotDerived
    }
}

fn sort_causes(values: &mut [AdmittedCauseV0]) {
    let mut index = 1;
    while index < values.len() {
        let value = values[index];
        let mut cursor = index;
        while cursor > 0 && value.canonical_cmp(&values[cursor - 1]) == Ordering::Less {
            values[cursor] = values[cursor - 1];
            cursor -= 1;
        }
        values[cursor] = value;
        index += 1;
    }
}

fn sort_intents(values: &mut [MutationIntentV0]) {
    let mut index = 1;
    while index < values.len() {
        let value = values[index];
        let mut cursor = index;
        while cursor > 0 && value.canonical_cmp(&values[cursor - 1]) == Ordering::Less {
            values[cursor] = values[cursor - 1];
            cursor -= 1;
        }
        values[cursor] = value;
        index += 1;
    }
}

fn validate_intent_encoding(intent: &MutationIntentV0) -> Result<(), WorldFault> {
    if intent.reserved != 0 {
        return Err(WorldFault::InvalidIntent);
    }
    match IntentKind(intent.kind) {
        IntentKind::DESPAWN_OBJECT => {
            if intent.reducer != ReducerClass::LIFECYCLE.0
                || intent.target == INVALID_OBJECT_KEY
            {
                return Err(WorldFault::InvalidIntent);
            }
        }
        IntentKind::SPAWN_OBJECT => {
            let kind = ObjectKind((intent.aux & 0xffff) as u16);
            let site = ((intent.aux >> 16) & 0xffff) as u16;
            if intent.reducer != ReducerClass::LIFECYCLE.0
                || intent.target != 0
                || intent.flags > u16::MAX as u32
                || !kind.is_valid()
                || (site != NON_SPATIAL_SITE && site >= 1024)
            {
                return Err(WorldFault::InvalidIntent);
            }
        }
        IntentKind::REPLACE_KINEMATICS => {
            if intent.reducer != ReducerClass::EXCLUSIVE.0
                || intent.target == INVALID_OBJECT_KEY
            {
                return Err(WorldFault::InvalidIntent);
            }
        }
        IntentKind::APPLY_HEALTH_DELTA => {
            if intent.reducer != ReducerClass::ADDITIVE.0
                || intent.target == INVALID_OBJECT_KEY
            {
                return Err(WorldFault::InvalidIntent);
            }
        }
        _ => return Err(WorldFault::InvalidIntent),
    }
    Ok(())
}

pub const fn mix64(mut value: u64) -> u64 {
    value ^= value >> 30;
    value = value.wrapping_mul(0xbf58476d1ce4e5b9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94d049bb133111eb);
    value ^ (value >> 31)
}

const _: [(); 64] = [(); size_of::<MutationIntentV0>()];
const _: [(); 8] = [(); align_of::<MutationIntentV0>()];

#[cfg(test)]
mod tests {
    use super::*;

    type TestWorld = WorldCore<4, 8>;
    type Scratch = TransitionScratch<8, 16, 4>;

    fn world() -> TestWorld {
        TestWorld::new(ZoneCoord::new(0, 0, 0).unwrap()).unwrap()
    }

    fn cause() -> [AdmittedCauseV0; 1] {
        [AdmittedCauseV0::external_input(0, 0, 0)]
    }

    fn spawn_one(world: &mut TestWorld, scratch: &mut Scratch, health: u16) -> ObjectKey {
        world
            .transact(&cause(), &[MutationIntentV0::spawn_actor(0, health, 0)], scratch)
            .unwrap();
        world.objects.object_key_at(0).unwrap()
    }

    #[test]
    fn zone_key_is_collision_free_at_v0_bounds() {
        let min = ZoneCoord::new(ZONE_COORD_MIN, ZONE_COORD_MIN, ZONE_COORD_MIN)
            .unwrap()
            .key();
        let max = ZoneCoord::new(ZONE_COORD_MAX, ZONE_COORD_MAX, ZONE_COORD_MAX)
            .unwrap()
            .key();
        assert_eq!(min.0, 0);
        assert_eq!(max.0, (1_u64 << 63) - 1);
        assert!(ZoneCoord::new(ZONE_COORD_MAX + 1, 0, 0).is_err());
    }

    #[test]
    fn no_work_does_not_advance_state() {
        let mut world = world();
        let before = world.clone();
        let mut scratch = Scratch::new();
        let result = world.transact(&[], &[], &mut scratch).unwrap();
        assert_eq!(result.disposition, TransitionDisposition::NoWork);
        assert_eq!(world, before);
    }

    #[test]
    fn canonical_input_permutations_produce_identical_worlds() {
        let mut left = world();
        let mut right = world();
        let mut scratch_left = Scratch::new();
        let mut scratch_right = Scratch::new();
        let intents = [
            MutationIntentV0::spawn_actor(0, 100, 0),
            MutationIntentV0::spawn_item(1, 7, 2, 1),
        ];
        let reversed = [intents[1], intents[0]];
        left.transact(&cause(), &intents, &mut scratch_left).unwrap();
        right
            .transact(&cause(), &reversed, &mut scratch_right)
            .unwrap();
        assert_eq!(left, right);
        assert_eq!(left.objects.object_key_at(0), Some(ObjectKey(1)));
        assert_eq!(left.objects.object_key_at(1), Some(ObjectKey(2)));
    }

    #[test]
    fn stale_handle_is_invalid_after_slot_reuse() {
        let mut world = world();
        let mut scratch = Scratch::new();
        let key = spawn_one(&mut world, &mut scratch, 100);
        let old_handle = world.objects.handle_for_key(key).unwrap();
        world
            .transact(&cause(), &[MutationIntentV0::despawn(0, key)], &mut scratch)
            .unwrap();
        world
            .transact(&cause(), &[MutationIntentV0::spawn_actor(0, 50, 0)], &mut scratch)
            .unwrap();
        assert!(world.objects.validate_handle(old_handle).is_none());
        let new_key = world.objects.object_key_at(0).unwrap();
        let new_handle = world.objects.handle_for_key(new_key).unwrap();
        assert_eq!(new_handle.slot, old_handle.slot);
        assert_ne!(new_handle.generation, old_handle.generation);
    }

    #[test]
    fn exclusive_conflict_faults_without_partial_mutation() {
        let mut world = world();
        let mut scratch = Scratch::new();
        let key = spawn_one(&mut world, &mut scratch, 100);
        let before = world.clone();
        let intents = [
            MutationIntentV0::replace_kinematics(0, key, 1, 2, 3, 4, 5, 6),
            MutationIntentV0::replace_kinematics(1, key, 7, 8, 9, 10, 11, 12),
        ];
        assert_eq!(
            world.transact(&cause(), &intents, &mut scratch),
            Err(WorldTransitionError::Fault(WorldFault::ExclusiveConflict))
        );
        assert_eq!(world, before);
    }

    #[test]
    fn additive_health_uses_one_wide_fold_and_is_permutation_invariant() {
        let mut left = world();
        let mut right = world();
        let mut scratch_left = Scratch::new();
        let mut scratch_right = Scratch::new();
        let key_left = spawn_one(&mut left, &mut scratch_left, 10);
        let key_right = spawn_one(&mut right, &mut scratch_right, 10);
        let a = MutationIntentV0::apply_health_delta(0, key_left, 70_000);
        let b = MutationIntentV0::apply_health_delta(1, key_left, -70_000);
        left.transact(&cause(), &[a, b], &mut scratch_left).unwrap();
        let a = MutationIntentV0::apply_health_delta(0, key_right, 70_000);
        let b = MutationIntentV0::apply_health_delta(1, key_right, -70_000);
        right
            .transact(&cause(), &[b, a], &mut scratch_right)
            .unwrap();
        assert_eq!(left.objects.health(key_left), Some(10));
        assert_eq!(right.objects.health(key_right), Some(10));
        assert_eq!(left, right);
    }

    #[test]
    fn full_zone_spawn_is_explicit_rejection_without_key_consumption() {
        let mut world = world();
        let mut scratch = Scratch::new();
        for sequence in 0..4 {
            world
                .transact(
                    &[AdmittedCauseV0::external_input(sequence, 0, 0)],
                    &[MutationIntentV0::spawn_actor(0, 1, 0)],
                    &mut scratch,
                )
                .unwrap();
        }
        let next_key = world.next_object_key;
        let result = world
            .transact(&cause(), &[MutationIntentV0::spawn_actor(0, 1, 0)], &mut scratch)
            .unwrap();
        assert_eq!(result.rejected_requests, 1);
        assert_eq!(world.next_object_key, next_key);
        assert_eq!(world.zone.object_count, 4);
    }

    #[test]
    fn intent_without_cause_is_a_fault_and_state_is_unchanged() {
        let mut world = world();
        let before = world.clone();
        let mut scratch = Scratch::new();
        assert_eq!(
            world.transact(
                &[],
                &[MutationIntentV0::spawn_actor(0, 1, 0)],
                &mut scratch
            ),
            Err(WorldTransitionError::Fault(WorldFault::CauseMissing))
        );
        assert_eq!(world, before);
    }

    #[test]
    fn xi_port_is_inert_and_not_derived() {
        let world = world();
        let before = world.clone();
        assert_eq!(InertXiPort.evaluate(), XiPortStatus::NotDerived);
        assert_eq!(world, before);
    }

    #[test]
    fn world_image_roundtrip_rebuilds_lookup_without_changing_semantics() {
        let mut original = world();
        let mut scratch = Scratch::new();
        let actor = spawn_one(&mut original, &mut scratch, 77);
        original
            .transact(
                &cause(),
                &[MutationIntentV0::replace_kinematics(
                    0, actor, 1, 2, 3, 4, 5, 6,
                )],
                &mut scratch,
            )
            .unwrap();
        let image = original.export_image();
        let restored = TestWorld::from_image(&image).unwrap();
        assert_eq!(restored, original);
        assert_eq!(restored.objects().health(actor), Some(77));
        assert_eq!(restored.objects().kinematics(actor), Some((1, 2, 3, 4, 5, 6)));
    }

}
