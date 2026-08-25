use phase_cell_persistence::{decode, encode, CellSnapshotError, CellSnapshotReceipt, PersistedCell};
use phase_wide::{WideDisposition, WideError};
use qbl_abi::QblPrimitive;
use world_core::{
    AdmittedCauseV0, MutationIntentV0, ObjectKey, TransitionScratch, ZoneCoord,
    NON_SPATIAL_SITE,
};

pub const MULTI_GRAPH_MAX_NODES: usize = 5;
pub const MULTI_GRAPH_MAX_EDGES: usize = 8;
pub const MULTI_GRAPH_MAX_TRAVELERS: usize = 4;
pub const MULTI_GRAPH_MAX_REQUESTS: usize = 8;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ClaimStatus {
    Granted,
    DuplicateIgnored,
    Invalid,
    TravelerNotFound,
    NodeRange,
    NodeNotResident,
    EdgeNotFound,
    StaleVersion,
    EdgeConflictLost,
    NodeConflictLost,
}

#[derive(Debug)]
pub enum MultiGraphError {
    InvalidIdentity,
    InvalidLength,
    DuplicateZone,
    DuplicateEdge,
    GraphDisconnected,
    DuplicateRequest,
    TravelerExists,
    TravelerNotFound,
    TravelerLimit,
    NodeRange,
    NodeNotResident,
    NodeAlreadyResident,
    NodeNotEmpty,
    ClaimsPending,
    NoClaim,
    StaleVersion,
    ActorMissing,
    GraphCommitExhausted,
    Snapshot(CellSnapshotError),
    SnapshotMismatch,
    SourcePhase(WideError),
    DestinationPhase(WideError),
    DestinationSpawnRejected,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ClaimRequest {
    pub request_key: u64,
    pub traveler_key: u64,
    pub handoff_cause_id: u64,
    pub destination_node: usize,
    pub expected_source_local_commit_id: u64,
    pub expected_destination_local_commit_id: u64,
    pub source_sequence: u32,
    pub destination_sequence: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ClaimResult {
    pub status: ClaimStatus,
    pub claim_set_id: u64,
    pub request_key: u64,
    pub traveler_key: u64,
    pub source_node: usize,
    pub destination_node: usize,
    pub edge_index: usize,
    pub edge_key: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HandoffReceipt {
    pub graph_commit_id: u64,
    pub claim_set_id: u64,
    pub request_key: u64,
    pub traveler_key: u64,
    pub actor_key: ObjectKey,
    pub source_node: usize,
    pub destination_node: usize,
    pub edge_index: usize,
    pub edge_key: u64,
    pub source_primitive: QblPrimitive,
    pub destination_primitive: QblPrimitive,
    pub source_local_commit_id: u64,
    pub destination_local_commit_id: u64,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct MultiGraphEdge {
    pub node_a: usize,
    pub node_b: usize,
    pub edge_key: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct MultiGraphNode {
    zone_key: u64,
    cell: Option<PersistedCell>,
    snapshot: Option<Vec<u8>>,
    snapshot_receipt: Option<CellSnapshotReceipt>,
    occupant_mask: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct PendingClaim {
    claim_set_id: u64,
    request_key: u64,
    handoff_cause_id: u64,
    edge_index: usize,
    destination_node: usize,
    expected_source_local_commit_id: u64,
    expected_destination_local_commit_id: u64,
    source_sequence: u32,
    destination_sequence: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Traveler {
    traveler_key: u64,
    actor_key: ObjectKey,
    current_node: usize,
    consumed_handoff_cause_id: u64,
    claim: Option<PendingClaim>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MultiTravelerGraph {
    graph_key: u64,
    nodes: Vec<MultiGraphNode>,
    edges: Vec<MultiGraphEdge>,
    travelers: Vec<Traveler>,
    node_claim_owner: Vec<Option<u64>>,
    edge_claim_owner: Vec<Option<u64>>,
    graph_commit_id: u64,
    active_claim_set_id: Option<u64>,
    claims_pending: usize,
}

#[derive(Clone, Copy)]
struct Candidate {
    request: ClaimRequest,
    source_node: usize,
    edge_index: Option<usize>,
    edge_key: u64,
}

impl MultiTravelerGraph {
    pub fn new(
        graph_key: u64,
        zones: &[ZoneCoord],
        edge_pairs: &[(usize, usize)],
        pair_limb_limit: u32,
    ) -> Result<Self, MultiGraphError> {
        if graph_key == 0 || pair_limb_limit == 0 {
            return Err(MultiGraphError::InvalidIdentity);
        }
        if zones.len() < 3
            || zones.len() > MULTI_GRAPH_MAX_NODES
            || edge_pairs.len() < zones.len() - 1
            || edge_pairs.len() > MULTI_GRAPH_MAX_EDGES
        {
            return Err(MultiGraphError::InvalidLength);
        }
        for left in 0..zones.len() {
            for right in left + 1..zones.len() {
                if zones[left].key() == zones[right].key() {
                    return Err(MultiGraphError::DuplicateZone);
                }
            }
        }
        let mut edges = Vec::with_capacity(edge_pairs.len());
        for &(left, right) in edge_pairs {
            if left >= zones.len() || right >= zones.len() || left == right {
                return Err(MultiGraphError::InvalidLength);
            }
            let (node_a, node_b) = if left < right { (left, right) } else { (right, left) };
            edges.push(MultiGraphEdge { node_a, node_b, edge_key: 0 });
        }
        edges.sort();
        for index in 0..edges.len() {
            if index > 0
                && edges[index - 1].node_a == edges[index].node_a
                && edges[index - 1].node_b == edges[index].node_b
            {
                return Err(MultiGraphError::DuplicateEdge);
            }
            edges[index].edge_key = index as u64 + 1;
        }
        let mut visited = vec![false; zones.len()];
        visited[0] = true;
        loop {
            let mut changed = false;
            for edge in &edges {
                if visited[edge.node_a] && !visited[edge.node_b] {
                    visited[edge.node_b] = true;
                    changed = true;
                }
                if visited[edge.node_b] && !visited[edge.node_a] {
                    visited[edge.node_a] = true;
                    changed = true;
                }
            }
            if !changed { break; }
        }
        if visited.iter().any(|value| !*value) {
            return Err(MultiGraphError::GraphDisconnected);
        }
        let mut nodes = Vec::with_capacity(zones.len());
        for zone in zones {
            nodes.push(MultiGraphNode {
                zone_key: zone.key().0,
                cell: Some(PersistedCell::new(*zone, pair_limb_limit)
                    .map_err(MultiGraphError::SourcePhase)?),
                snapshot: None,
                snapshot_receipt: None,
                occupant_mask: 0,
            });
        }
        Ok(Self {
            graph_key,
            node_claim_owner: vec![None; nodes.len()],
            edge_claim_owner: vec![None; edges.len()],
            nodes,
            edges,
            travelers: Vec::with_capacity(MULTI_GRAPH_MAX_TRAVELERS),
            graph_commit_id: 0,
            active_claim_set_id: None,
            claims_pending: 0,
        })
    }

    pub const fn graph_key(&self) -> u64 { self.graph_key }
    pub const fn graph_commit_id(&self) -> u64 { self.graph_commit_id }
    pub const fn claims_pending(&self) -> usize { self.claims_pending }
    pub fn node_count(&self) -> usize { self.nodes.len() }
    pub fn edge_count(&self) -> usize { self.edges.len() }
    pub fn traveler_count(&self) -> usize { self.travelers.len() }
    pub fn edges(&self) -> &[MultiGraphEdge] { &self.edges }
    pub fn node_claim_owner(&self, index: usize) -> Option<u64> {
        self.node_claim_owner.get(index).copied().flatten()
    }
    pub fn edge_claim_owner(&self, index: usize) -> Option<u64> {
        self.edge_claim_owner.get(index).copied().flatten()
    }
    pub fn occupant_mask(&self, index: usize) -> Result<u32, MultiGraphError> {
        self.nodes.get(index).map(|node| node.occupant_mask).ok_or(MultiGraphError::NodeRange)
    }
    pub fn resident(&self, index: usize) -> Result<bool, MultiGraphError> {
        self.nodes.get(index).map(|node| node.cell.is_some()).ok_or(MultiGraphError::NodeRange)
    }
    pub fn cell(&self, index: usize) -> Option<&PersistedCell> {
        self.nodes.get(index)?.cell.as_ref()
    }
    pub fn traveler_node(&self, traveler_key: u64) -> Option<usize> {
        self.travelers.iter().find(|t| t.traveler_key == traveler_key).map(|t| t.current_node)
    }
    pub fn traveler_actor_key(&self, traveler_key: u64) -> Option<ObjectKey> {
        self.travelers.iter().find(|t| t.traveler_key == traveler_key).map(|t| t.actor_key)
    }
    pub fn edge_for(&self, left: usize, right: usize) -> Option<usize> {
        let (a, b) = if left < right { (left, right) } else { (right, left) };
        self.edges.iter().position(|edge| edge.node_a == a && edge.node_b == b)
    }
    fn traveler_index(&self, traveler_key: u64) -> Option<usize> {
        self.travelers.iter().position(|traveler| traveler.traveler_key == traveler_key)
    }

    pub fn bootstrap_traveler<const CAUSES: usize, const INTENTS: usize>(
        &mut self,
        traveler_key: u64,
        node_index: usize,
        bootstrap_cause_id: u64,
        source_sequence: u32,
        health: u16,
        scratch: &mut TransitionScratch<CAUSES, INTENTS, 8>,
    ) -> Result<QblPrimitive, MultiGraphError> {
        if traveler_key == 0 || bootstrap_cause_id == 0 || source_sequence == 0 || health == 0 {
            return Err(MultiGraphError::InvalidIdentity);
        }
        if self.traveler_index(traveler_key).is_some() {
            return Err(MultiGraphError::TravelerExists);
        }
        if self.travelers.len() >= MULTI_GRAPH_MAX_TRAVELERS {
            return Err(MultiGraphError::TravelerLimit);
        }
        if node_index >= self.nodes.len() { return Err(MultiGraphError::NodeRange); }
        if self.node_claim_owner[node_index].is_some() { return Err(MultiGraphError::ClaimsPending); }
        let node = self.nodes[node_index].cell.as_ref().ok_or(MultiGraphError::NodeNotResident)?;
        let expected_key = ObjectKey(node.world().next_object_key());
        let slot = self.travelers.len();
        let mut staged = self.clone();
        let causes = [AdmittedCauseV0::external_input(source_sequence, traveler_key, bootstrap_cause_id)];
        let intents = [MutationIntentV0::spawn_actor(source_sequence, health, NON_SPATIAL_SITE)];
        let result = staged.nodes[node_index].cell.as_mut().unwrap()
            .transact(&causes, &intents, scratch)
            .map_err(MultiGraphError::SourcePhase)?;
        if result.disposition != WideDisposition::Committed
            || result.rejected_requests != 0
            || staged.nodes[node_index].cell.as_ref().unwrap().world().objects().health(expected_key) != Some(health)
        {
            return Err(MultiGraphError::DestinationSpawnRejected);
        }
        staged.travelers.push(Traveler {
            traveler_key,
            actor_key: expected_key,
            current_node: node_index,
            consumed_handoff_cause_id: 0,
            claim: None,
        });
        staged.nodes[node_index].occupant_mask |= 1_u32 << slot;
        *self = staged;
        Ok(result.primitive)
    }

    pub fn advance_node<const CAUSES: usize, const INTENTS: usize>(
        &mut self,
        node_index: usize,
        source_sequence: u32,
        payload0: u64,
        payload1: u64,
        scratch: &mut TransitionScratch<CAUSES, INTENTS, 8>,
    ) -> Result<QblPrimitive, MultiGraphError> {
        if source_sequence == 0 { return Err(MultiGraphError::InvalidIdentity); }
        if node_index >= self.nodes.len() { return Err(MultiGraphError::NodeRange); }
        if self.node_claim_owner[node_index].is_some() { return Err(MultiGraphError::ClaimsPending); }
        let mut staged = self.clone();
        let causes = [AdmittedCauseV0::external_input(source_sequence, payload0, payload1)];
        let result = staged.nodes[node_index].cell.as_mut().ok_or(MultiGraphError::NodeNotResident)?
            .transact(&causes, &[], scratch)
            .map_err(MultiGraphError::SourcePhase)?;
        *self = staged;
        Ok(result.primitive)
    }

    pub fn admit_claims(
        &mut self,
        claim_set_id: u64,
        requests: &[ClaimRequest],
    ) -> Result<Vec<ClaimResult>, MultiGraphError> {
        if claim_set_id == 0 || requests.is_empty() || requests.len() > MULTI_GRAPH_MAX_REQUESTS {
            return Err(MultiGraphError::InvalidIdentity);
        }
        if self.claims_pending != 0 { return Err(MultiGraphError::ClaimsPending); }
        for left in 0..requests.len() {
            for right in left + 1..requests.len() {
                if requests[left].request_key == requests[right].request_key
                    || requests[left].traveler_key == requests[right].traveler_key
                {
                    return Err(MultiGraphError::DuplicateRequest);
                }
            }
        }
        let mut candidates = Vec::with_capacity(requests.len());
        for &request in requests {
            let (source_node, edge_index, edge_key) = match self.traveler_index(request.traveler_key) {
                Some(index) => {
                    let source = self.travelers[index].current_node;
                    match self.edge_for(source, request.destination_node) {
                        Some(edge) => (source, Some(edge), self.edges[edge].edge_key),
                        None => (source, None, u64::MAX),
                    }
                }
                None => (usize::MAX, None, u64::MAX),
            };
            candidates.push(Candidate { request, source_node, edge_index, edge_key });
        }
        candidates.sort_by_key(|candidate| {
            (candidate.edge_key, candidate.request.traveler_key, candidate.request.request_key)
        });
        let mut staged = self.clone();
        let mut results = Vec::with_capacity(candidates.len());
        let mut grants = 0;
        for candidate in candidates {
            let request = candidate.request;
            let mut result = ClaimResult {
                status: ClaimStatus::Invalid,
                claim_set_id,
                request_key: request.request_key,
                traveler_key: request.traveler_key,
                source_node: candidate.source_node,
                destination_node: request.destination_node,
                edge_index: candidate.edge_index.unwrap_or(usize::MAX),
                edge_key: if candidate.edge_key == u64::MAX { 0 } else { candidate.edge_key },
            };
            if request.request_key == 0 || request.traveler_key == 0
                || request.handoff_cause_id == 0 || request.source_sequence == 0
                || request.destination_sequence == 0
            {
                results.push(result);
                continue;
            }
            let Some(traveler_index) = staged.traveler_index(request.traveler_key) else {
                result.status = ClaimStatus::TravelerNotFound;
                results.push(result);
                continue;
            };
            if staged.travelers[traveler_index].consumed_handoff_cause_id == request.handoff_cause_id {
                result.status = ClaimStatus::DuplicateIgnored;
                results.push(result);
                continue;
            }
            if request.destination_node >= staged.nodes.len() {
                result.status = ClaimStatus::NodeRange;
                results.push(result);
                continue;
            }
            let Some(edge_index) = candidate.edge_index else {
                result.status = ClaimStatus::EdgeNotFound;
                results.push(result);
                continue;
            };
            if staged.nodes[candidate.source_node].cell.is_none()
                || staged.nodes[request.destination_node].cell.is_none()
            {
                result.status = ClaimStatus::NodeNotResident;
                results.push(result);
                continue;
            }
            if staged.nodes[candidate.source_node].cell.as_ref().unwrap().local_commit_id()
                    != request.expected_source_local_commit_id
                || staged.nodes[request.destination_node].cell.as_ref().unwrap().local_commit_id()
                    != request.expected_destination_local_commit_id
            {
                result.status = ClaimStatus::StaleVersion;
                results.push(result);
                continue;
            }
            if staged.edge_claim_owner[edge_index].is_some() {
                result.status = ClaimStatus::EdgeConflictLost;
                results.push(result);
                continue;
            }
            if staged.node_claim_owner[candidate.source_node].is_some()
                || staged.node_claim_owner[request.destination_node].is_some()
            {
                result.status = ClaimStatus::NodeConflictLost;
                results.push(result);
                continue;
            }
            staged.edge_claim_owner[edge_index] = Some(request.traveler_key);
            staged.node_claim_owner[candidate.source_node] = Some(request.traveler_key);
            staged.node_claim_owner[request.destination_node] = Some(request.traveler_key);
            staged.travelers[traveler_index].claim = Some(PendingClaim {
                claim_set_id,
                request_key: request.request_key,
                handoff_cause_id: request.handoff_cause_id,
                edge_index,
                destination_node: request.destination_node,
                expected_source_local_commit_id: request.expected_source_local_commit_id,
                expected_destination_local_commit_id: request.expected_destination_local_commit_id,
                source_sequence: request.source_sequence,
                destination_sequence: request.destination_sequence,
            });
            result.status = ClaimStatus::Granted;
            grants += 1;
            results.push(result);
        }
        if grants != 0 {
            staged.active_claim_set_id = Some(claim_set_id);
            staged.claims_pending = grants;
            *self = staged;
        }
        Ok(results)
    }

    pub fn publish_claim<
        const SOURCE_CAUSES: usize,
        const SOURCE_INTENTS: usize,
        const DEST_CAUSES: usize,
        const DEST_INTENTS: usize,
    >(
        &mut self,
        traveler_key: u64,
        source_scratch: &mut TransitionScratch<SOURCE_CAUSES, SOURCE_INTENTS, 8>,
        destination_scratch: &mut TransitionScratch<DEST_CAUSES, DEST_INTENTS, 8>,
    ) -> Result<HandoffReceipt, MultiGraphError> {
        let traveler_index = self.traveler_index(traveler_key).ok_or(MultiGraphError::TravelerNotFound)?;
        let traveler = self.travelers[traveler_index];
        let claim = traveler.claim.ok_or(MultiGraphError::NoClaim)?;
        let source_node = traveler.current_node;
        let destination_node = claim.destination_node;
        if self.node_claim_owner[source_node] != Some(traveler_key)
            || self.node_claim_owner[destination_node] != Some(traveler_key)
            || self.edge_claim_owner[claim.edge_index] != Some(traveler_key)
        {
            return Err(MultiGraphError::NoClaim);
        }
        let source = self.nodes[source_node].cell.as_ref().ok_or(MultiGraphError::NodeNotResident)?;
        let destination = self.nodes[destination_node].cell.as_ref().ok_or(MultiGraphError::NodeNotResident)?;
        if source.local_commit_id() != claim.expected_source_local_commit_id
            || destination.local_commit_id() != claim.expected_destination_local_commit_id
        {
            return Err(MultiGraphError::StaleVersion);
        }
        let health = source.world().objects().health(traveler.actor_key).ok_or(MultiGraphError::ActorMissing)?;
        let next_commit_id = self.graph_commit_id.checked_add(1).ok_or(MultiGraphError::GraphCommitExhausted)?;
        let edge_key = self.edges[claim.edge_index].edge_key;
        let mut staged = self.clone();
        let source_causes = [AdmittedCauseV0::external_input(claim.source_sequence, traveler_key, edge_key)];
        let source_intents = [MutationIntentV0::despawn(claim.source_sequence, traveler.actor_key)];
        let source_result = staged.nodes[source_node].cell.as_mut().unwrap()
            .transact(&source_causes, &source_intents, source_scratch)
            .map_err(MultiGraphError::SourcePhase)?;
        if source_result.disposition != WideDisposition::Committed || source_result.rejected_requests != 0 {
            return Err(MultiGraphError::ActorMissing);
        }
        let expected_destination_key = ObjectKey(staged.nodes[destination_node].cell.as_ref().unwrap().world().next_object_key());
        let destination_causes = [AdmittedCauseV0::external_input(
            claim.destination_sequence,
            traveler_key,
            claim.handoff_cause_id,
        )];
        let destination_intents = [MutationIntentV0::spawn_actor(
            claim.destination_sequence,
            health,
            NON_SPATIAL_SITE,
        )];
        let destination_result = staged.nodes[destination_node].cell.as_mut().unwrap()
            .transact(&destination_causes, &destination_intents, destination_scratch)
            .map_err(MultiGraphError::DestinationPhase)?;
        if destination_result.disposition != WideDisposition::Committed
            || destination_result.rejected_requests != 0
            || staged.nodes[destination_node].cell.as_ref().unwrap().world().objects().health(expected_destination_key) != Some(health)
        {
            return Err(MultiGraphError::DestinationSpawnRejected);
        }
        staged.nodes[source_node].occupant_mask &= !(1_u32 << traveler_index);
        staged.nodes[destination_node].occupant_mask |= 1_u32 << traveler_index;
        staged.node_claim_owner[source_node] = None;
        staged.node_claim_owner[destination_node] = None;
        staged.edge_claim_owner[claim.edge_index] = None;
        staged.travelers[traveler_index].current_node = destination_node;
        staged.travelers[traveler_index].actor_key = expected_destination_key;
        staged.travelers[traveler_index].consumed_handoff_cause_id = claim.handoff_cause_id;
        staged.travelers[traveler_index].claim = None;
        staged.graph_commit_id = next_commit_id;
        staged.claims_pending -= 1;
        if staged.claims_pending == 0 { staged.active_claim_set_id = None; }
        let receipt = HandoffReceipt {
            graph_commit_id: next_commit_id,
            claim_set_id: claim.claim_set_id,
            request_key: claim.request_key,
            traveler_key,
            actor_key: expected_destination_key,
            source_node,
            destination_node,
            edge_index: claim.edge_index,
            edge_key,
            source_primitive: source_result.primitive,
            destination_primitive: destination_result.primitive,
            source_local_commit_id: source_result.local_commit_id,
            destination_local_commit_id: destination_result.local_commit_id,
        };
        *self = staged;
        Ok(receipt)
    }

    pub fn evict_node(&mut self, index: usize) -> Result<CellSnapshotReceipt, MultiGraphError> {
        let node = self.nodes.get(index).ok_or(MultiGraphError::NodeRange)?;
        if node.occupant_mask != 0 { return Err(MultiGraphError::NodeNotEmpty); }
        if self.node_claim_owner[index].is_some() { return Err(MultiGraphError::ClaimsPending); }
        let cell = node.cell.as_ref().ok_or(MultiGraphError::NodeNotResident)?;
        let (bytes, receipt) = encode(cell).map_err(MultiGraphError::Snapshot)?;
        let mut staged = self.clone();
        staged.nodes[index].cell = None;
        staged.nodes[index].snapshot = Some(bytes);
        staged.nodes[index].snapshot_receipt = Some(receipt);
        *self = staged;
        Ok(receipt)
    }

    pub fn restore_node(&mut self, index: usize) -> Result<CellSnapshotReceipt, MultiGraphError> {
        let node = self.nodes.get(index).ok_or(MultiGraphError::NodeRange)?;
        if node.cell.is_some() { return Err(MultiGraphError::NodeAlreadyResident); }
        let expected = node.snapshot_receipt.ok_or(MultiGraphError::SnapshotMismatch)?;
        let bytes = node.snapshot.as_ref().ok_or(MultiGraphError::SnapshotMismatch)?;
        let (cell, receipt) = decode(bytes).map_err(MultiGraphError::Snapshot)?;
        if receipt != expected || receipt.source_zone_key != node.zone_key {
            return Err(MultiGraphError::SnapshotMismatch);
        }
        let mut staged = self.clone();
        staged.nodes[index].cell = Some(cell);
        staged.nodes[index].snapshot = None;
        staged.nodes[index].snapshot_receipt = None;
        *self = staged;
        Ok(receipt)
    }

    pub fn snapshot_bytes(&self, index: usize) -> Option<&[u8]> {
        self.nodes.get(index)?.snapshot.as_deref()
    }

    pub fn install_snapshot(&mut self, index: usize, bytes: Vec<u8>) -> Result<(), MultiGraphError> {
        let node = self.nodes.get_mut(index).ok_or(MultiGraphError::NodeRange)?;
        if node.cell.is_some() { return Err(MultiGraphError::NodeAlreadyResident); }
        node.snapshot = Some(bytes);
        Ok(())
    }

    pub fn provision_node_pair_limbs(&mut self, index: usize, new_limit: u32) -> Result<(), MultiGraphError> {
        self.nodes.get_mut(index).ok_or(MultiGraphError::NodeRange)?
            .cell.as_mut().ok_or(MultiGraphError::NodeNotResident)?
            .provision_pair_limbs(new_limit)
            .map_err(MultiGraphError::SourcePhase)
    }

    pub fn diagnostic_fingerprint64(&self) -> u64 {
        let mut hash = 0xcbf29ce484222325_u64;
        fn feed(hash: &mut u64, bytes: &[u8]) {
            for byte in bytes {
                *hash ^= *byte as u64;
                *hash = hash.wrapping_mul(0x100000001b3);
            }
        }
        feed(&mut hash, &self.graph_key.to_le_bytes());
        feed(&mut hash, &self.graph_commit_id.to_le_bytes());
        feed(&mut hash, &self.active_claim_set_id.unwrap_or(0).to_le_bytes());
        feed(&mut hash, &(self.claims_pending as u64).to_le_bytes());
        for edge in &self.edges {
            feed(&mut hash, &(edge.node_a as u64).to_le_bytes());
            feed(&mut hash, &(edge.node_b as u64).to_le_bytes());
            feed(&mut hash, &edge.edge_key.to_le_bytes());
        }
        for owner in &self.edge_claim_owner { feed(&mut hash, &owner.unwrap_or(0).to_le_bytes()); }
        for (index, node) in self.nodes.iter().enumerate() {
            feed(&mut hash, &node.zone_key.to_le_bytes());
            feed(&mut hash, &node.occupant_mask.to_le_bytes());
            feed(&mut hash, &self.node_claim_owner[index].unwrap_or(0).to_le_bytes());
            feed(&mut hash, &[node.cell.is_some() as u8]);
            if let Some(cell) = &node.cell {
                feed(&mut hash, &cell.phase().diagnostic_fingerprint64().to_le_bytes());
                feed(&mut hash, &cell.world().diagnostic_fingerprint64().to_le_bytes());
            } else if let Some(bytes) = &node.snapshot {
                feed(&mut hash, bytes);
            }
        }
        for traveler in &self.travelers {
            feed(&mut hash, &traveler.traveler_key.to_le_bytes());
            feed(&mut hash, &traveler.actor_key.0.to_le_bytes());
            feed(&mut hash, &(traveler.current_node as u64).to_le_bytes());
            feed(&mut hash, &traveler.consumed_handoff_cause_id.to_le_bytes());
            if let Some(claim) = traveler.claim {
                feed(&mut hash, &claim.claim_set_id.to_le_bytes());
                feed(&mut hash, &claim.request_key.to_le_bytes());
                feed(&mut hash, &claim.handoff_cause_id.to_le_bytes());
                feed(&mut hash, &(claim.edge_index as u64).to_le_bytes());
                feed(&mut hash, &(claim.destination_node as u64).to_le_bytes());
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

    fn zones() -> [ZoneCoord; 5] {
        [
            ZoneCoord::new(0, 0, 0).unwrap(),
            ZoneCoord::new(1, 0, 0).unwrap(),
            ZoneCoord::new(2, 0, 0).unwrap(),
            ZoneCoord::new(3, 0, 0).unwrap(),
            ZoneCoord::new(4, 0, 0).unwrap(),
        ]
    }

    fn graph(limit: u32) -> MultiTravelerGraph {
        MultiTravelerGraph::new(91, &zones(), &[(2, 4), (0, 2), (1, 3), (0, 1)], limit).unwrap()
    }

    fn request(
        graph: &MultiTravelerGraph,
        request_key: u64,
        traveler_key: u64,
        cause: u64,
        destination: usize,
        source_sequence: u32,
        destination_sequence: u32,
    ) -> ClaimRequest {
        let source = graph.traveler_node(traveler_key).unwrap();
        ClaimRequest {
            request_key,
            traveler_key,
            handoff_cause_id: cause,
            destination_node: destination,
            expected_source_local_commit_id: graph.cell(source).unwrap().local_commit_id(),
            expected_destination_local_commit_id: graph.cell(destination).unwrap().local_commit_id(),
            source_sequence,
            destination_sequence,
        }
    }

    fn health(graph: &MultiTravelerGraph, traveler: u64) -> u16 {
        let node = graph.traveler_node(traveler).unwrap();
        let key = graph.traveler_actor_key(traveler).unwrap();
        graph.cell(node).unwrap().world().objects().health(key).unwrap()
    }

    #[test]
    fn same_edge_conflict_uses_stable_traveler_key() {
        let mut graph = graph(4);
        let mut scratch = Scratch::new();
        let mut destination = Scratch::new();
        graph.bootstrap_traveler(9002, 0, 100, 1, 80, &mut scratch).unwrap();
        graph.bootstrap_traveler(9001, 0, 101, 2, 91, &mut scratch).unwrap();
        let requests = [
            request(&graph, 22, 9002, 202, 1, 3, 1),
            request(&graph, 11, 9001, 201, 1, 3, 1),
        ];
        let results = graph.admit_claims(501, &requests).unwrap();
        assert_eq!(results[0].traveler_key, 9001);
        assert_eq!(results[0].status, ClaimStatus::Granted);
        assert_eq!(results[1].status, ClaimStatus::EdgeConflictLost);
        let receipt = graph.publish_claim(9001, &mut scratch, &mut destination).unwrap();
        assert_eq!(receipt.edge_key, 1);
        assert_eq!(graph.traveler_node(9001), Some(1));
        assert_eq!(graph.traveler_node(9002), Some(0));
        assert_eq!(health(&graph, 9001), 91);
        assert_eq!(health(&graph, 9002), 80);
    }

    #[test]
    fn disjoint_claims_commute() {
        let mut left = graph(4);
        let mut right = graph(4);
        let mut a = Scratch::new();
        let mut b = Scratch::new();
        let mut c = Scratch::new();
        let mut d = Scratch::new();
        for graph in [&mut left, &mut right] {
            graph.bootstrap_traveler(9001, 0, 100, 1, 91, &mut a).unwrap();
            graph.bootstrap_traveler(9003, 3, 101, 1, 77, &mut a).unwrap();
            let requests = [
                request(graph, 31, 9003, 301, 1, 2, 1),
                request(graph, 21, 9001, 201, 2, 2, 1),
            ];
            let results = graph.admit_claims(601, &requests).unwrap();
            assert!(results.iter().all(|result| result.status == ClaimStatus::Granted));
        }
        left.publish_claim(9003, &mut a, &mut b).unwrap();
        left.publish_claim(9001, &mut a, &mut b).unwrap();
        right.publish_claim(9001, &mut c, &mut d).unwrap();
        right.publish_claim(9003, &mut c, &mut d).unwrap();
        assert_eq!(left, right);
    }

    #[test]
    fn node_claim_blocks_overlapping_work() {
        let mut graph = graph(4);
        let mut scratch = Scratch::new();
        graph.bootstrap_traveler(9001, 0, 100, 1, 91, &mut scratch).unwrap();
        graph.bootstrap_traveler(9002, 2, 101, 1, 80, &mut scratch).unwrap();
        let requests = [
            request(&graph, 11, 9001, 201, 2, 2, 1),
            request(&graph, 22, 9002, 202, 4, 2, 1),
        ];
        let results = graph.admit_claims(701, &requests).unwrap();
        assert_eq!(results[0].status, ClaimStatus::Granted);
        assert_eq!(results[1].status, ClaimStatus::NodeConflictLost);
        assert_eq!(graph.node_claim_owner(2), Some(9001));
        assert!(matches!(graph.advance_node(0, 3, 1, 2, &mut scratch), Err(MultiGraphError::ClaimsPending)));
    }

    #[test]
    fn provisioning_failure_preserves_claim_for_retry() {
        let mut graph = graph(1);
        let mut source = Scratch::new();
        let mut destination = Scratch::new();
        graph.bootstrap_traveler(9001, 0, 100, 1, 66, &mut source).unwrap();
        for sequence in 2..=154 {
            graph.advance_node(0, sequence, sequence as u64, 0, &mut source).unwrap();
        }
        let request = request(&graph, 41, 9001, 401, 2, 155, 1);
        assert_eq!(graph.admit_claims(801, &[request]).unwrap()[0].status, ClaimStatus::Granted);
        let before = graph.clone();
        assert!(matches!(
            graph.publish_claim(9001, &mut source, &mut destination),
            Err(MultiGraphError::SourcePhase(WideError::ProvisioningRequired { required_pair_limbs: 2 }))
        ));
        assert_eq!(graph, before);
        graph.provision_node_pair_limbs(0, 2).unwrap();
        graph.publish_claim(9001, &mut source, &mut destination).unwrap();
        assert_eq!(health(&graph, 9001), 66);
    }

    #[test]
    fn duplicate_and_corrupt_snapshot_are_atomic() {
        let mut graph = graph(4);
        let mut source = Scratch::new();
        let mut destination = Scratch::new();
        graph.bootstrap_traveler(9001, 0, 100, 1, 91, &mut source).unwrap();
        graph.evict_node(4).unwrap();
        let mut bytes = graph.snapshot_bytes(4).unwrap().to_vec();
        bytes[9] ^= 0x40;
        graph.install_snapshot(4, bytes).unwrap();
        let before = graph.clone();
        assert!(matches!(graph.restore_node(4), Err(MultiGraphError::Snapshot(_))));
        assert_eq!(graph, before);
        let mut fixed = graph.snapshot_bytes(4).unwrap().to_vec();
        fixed[9] ^= 0x40;
        graph.install_snapshot(4, fixed).unwrap();
        graph.restore_node(4).unwrap();
        let first = request(&graph, 51, 9001, 501, 2, 2, 1);
        graph.admit_claims(901, &[first]).unwrap();
        graph.publish_claim(9001, &mut source, &mut destination).unwrap();
        let before_duplicate = graph.clone();
        let duplicate = request(&graph, 52, 9001, 501, 4, 2, 1);
        assert_eq!(graph.admit_claims(902, &[duplicate]).unwrap()[0].status, ClaimStatus::DuplicateIgnored);
        assert_eq!(graph, before_duplicate);
    }
}
