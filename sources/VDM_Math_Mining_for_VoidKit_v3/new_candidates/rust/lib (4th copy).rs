use phase_cell_persistence::{decode, encode, CellSnapshotError, CellSnapshotReceipt, PersistedCell};
use phase_wide::{WideDisposition, WideError, WideResult};
use qbl_abi::QblPrimitive;
use world_core::{
    AdmittedCauseV0, MutationIntentV0, ObjectKey, TransitionScratch, ZoneCoord,
    NON_SPATIAL_SITE,
};

pub const GRAPH_MAX_NODES: usize = 5;
pub const GRAPH_MAX_EDGES: usize = 8;

#[derive(Debug)]
pub enum GraphError {
    InvalidIdentity,
    InvalidLength,
    DuplicateZone,
    DuplicateEdge,
    GraphDisconnected,
    AlreadyBootstrapped,
    NotBootstrapped,
    NodeRange,
    NodeNotResident,
    NodeAlreadyResident,
    NodeNotEmpty,
    ActiveNode,
    EdgeNotFound,
    ActorMissing,
    StaleVersion,
    GraphCommitExhausted,
    Snapshot(CellSnapshotError),
    SnapshotMismatch,
    SourcePhase(WideError),
    DestinationPhase(WideError),
    DestinationSpawnRejected,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GraphDisposition {
    Committed,
    DuplicateIgnored,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GraphReceipt {
    pub disposition: GraphDisposition,
    pub graph_commit_id: u64,
    pub source_index: usize,
    pub destination_index: usize,
    pub active_index: usize,
    pub edge_index: usize,
    pub edge_key: u64,
    pub source_primitive: QblPrimitive,
    pub destination_primitive: QblPrimitive,
    pub source_local_commit_id: u64,
    pub destination_local_commit_id: u64,
    pub traveler_key: u64,
    pub actor_key: ObjectKey,
}


#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct GraphEdge {
    pub node_a: usize,
    pub node_b: usize,
    pub edge_key: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct GraphNode {
    zone_key: u64,
    cell: Option<PersistedCell>,
    snapshot: Option<Vec<u8>>,
    snapshot_receipt: Option<CellSnapshotReceipt>,
}

impl GraphNode {
    fn resident(&self) -> bool { self.cell.is_some() }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RetainedGraph {
    graph_key: u64,
    traveler_key: u64,
    nodes: Vec<GraphNode>,
    edges: Vec<GraphEdge>,
    active_index: usize,
    actor_key: Option<ObjectKey>,
    graph_commit_id: u64,
    consumed_handoff_cause_id: u64,
    bootstrapped: bool,
}

impl RetainedGraph {
    pub fn new(
        graph_key: u64,
        traveler_key: u64,
        zones: &[ZoneCoord],
        edge_pairs: &[(usize, usize)],
        pair_limb_limit: u32,
    ) -> Result<Self, GraphError> {
        if graph_key == 0 || traveler_key == 0 || pair_limb_limit == 0 {
            return Err(GraphError::InvalidIdentity);
        }
        if zones.len() < 3 || zones.len() > GRAPH_MAX_NODES
            || edge_pairs.len() < zones.len() - 1
            || edge_pairs.len() > GRAPH_MAX_EDGES
        {
            return Err(GraphError::InvalidLength);
        }
        for left in 0..zones.len() {
            for right in left + 1..zones.len() {
                if zones[left].key() == zones[right].key() {
                    return Err(GraphError::DuplicateZone);
                }
            }
        }
        let mut edges = Vec::with_capacity(edge_pairs.len());
        for &(left, right) in edge_pairs {
            if left >= zones.len() || right >= zones.len() || left == right {
                return Err(GraphError::InvalidLength);
            }
            let (node_a, node_b) = if left < right { (left, right) } else { (right, left) };
            edges.push(GraphEdge { node_a, node_b, edge_key: 0 });
        }
        edges.sort();
        for index in 0..edges.len() {
            if index > 0
                && edges[index - 1].node_a == edges[index].node_a
                && edges[index - 1].node_b == edges[index].node_b
            {
                return Err(GraphError::DuplicateEdge);
            }
            edges[index].edge_key = index as u64 + 1;
        }
        let mut visited = vec![false; zones.len()];
        visited[0] = true;
        loop {
            let mut changed = false;
            for edge in &edges {
                let a_visited = visited[edge.node_a];
                let b_visited = visited[edge.node_b];
                if a_visited && !b_visited {
                    visited[edge.node_b] = true;
                    changed = true;
                }
                if b_visited && !a_visited {
                    visited[edge.node_a] = true;
                    changed = true;
                }
            }
            if !changed { break; }
        }
        if visited.iter().any(|value| !*value) {
            return Err(GraphError::GraphDisconnected);
        }
        let mut nodes = Vec::with_capacity(zones.len());
        for zone in zones {
            let cell = PersistedCell::new(*zone, pair_limb_limit)
                .map_err(GraphError::SourcePhase)?;
            nodes.push(GraphNode {
                zone_key: zone.key().0,
                cell: Some(cell),
                snapshot: None,
                snapshot_receipt: None,
            });
        }
        Ok(Self {
            graph_key,
            traveler_key,
            nodes,
            edges,
            active_index: 0,
            actor_key: None,
            graph_commit_id: 0,
            consumed_handoff_cause_id: 0,
            bootstrapped: false,
        })
    }

    pub const fn graph_key(&self) -> u64 { self.graph_key }
    pub const fn traveler_key(&self) -> u64 { self.traveler_key }
    pub fn node_count(&self) -> usize { self.nodes.len() }
    pub fn edge_count(&self) -> usize { self.edges.len() }
    pub fn edge(&self, index: usize) -> Option<GraphEdge> { self.edges.get(index).copied() }
    pub fn edge_for(&self, left: usize, right: usize) -> Option<usize> {
        if left == right { return None; }
        let (node_a, node_b) = if left < right { (left, right) } else { (right, left) };
        self.edges.iter().position(|edge| edge.node_a == node_a && edge.node_b == node_b)
    }
    pub const fn active_index(&self) -> usize { self.active_index }
    pub const fn graph_commit_id(&self) -> u64 { self.graph_commit_id }
    pub const fn actor_key(&self) -> Option<ObjectKey> { self.actor_key }
    pub const fn bootstrapped(&self) -> bool { self.bootstrapped }

    pub fn resident(&self, index: usize) -> Result<bool, GraphError> {
        self.nodes.get(index).map(GraphNode::resident).ok_or(GraphError::NodeRange)
    }

    pub fn cell(&self, index: usize) -> Option<&PersistedCell> {
        self.nodes.get(index).and_then(|node| node.cell.as_ref())
    }

    pub fn snapshot_bytes(&self, index: usize) -> Option<&[u8]> {
        self.nodes.get(index).and_then(|node| node.snapshot.as_deref())
    }

    pub fn install_snapshot(&mut self, index: usize, bytes: Vec<u8>) -> Result<(), GraphError> {
        let node = self.nodes.get_mut(index).ok_or(GraphError::NodeRange)?;
        if node.cell.is_some() || node.snapshot.is_none() {
            return Err(GraphError::NodeAlreadyResident);
        }
        node.snapshot = Some(bytes);
        Ok(())
    }

    pub fn bootstrap<const CAUSES: usize, const INTENTS: usize>(
        &mut self,
        bootstrap_cause_id: u64,
        source_sequence: u32,
        actor_health: u16,
        scratch: &mut TransitionScratch<CAUSES, INTENTS, 8>,
    ) -> Result<WideResult, GraphError> {
        if bootstrap_cause_id == 0 || source_sequence == 0 || actor_health == 0 {
            return Err(GraphError::InvalidIdentity);
        }
        if self.bootstrapped {
            return Err(GraphError::AlreadyBootstrapped);
        }
        let mut staged = self.clone();
        let cell = staged.nodes[0].cell.as_mut().ok_or(GraphError::NodeNotResident)?;
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
            .map_err(GraphError::SourcePhase)?;
        if result.disposition != WideDisposition::Committed
            || result.rejected_requests != 0
            || cell.world().objects().health(expected_key) != Some(actor_health)
        {
            return Err(GraphError::DestinationSpawnRejected);
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
    ) -> Result<WideResult, GraphError> {
        if !self.bootstrapped {
            return Err(GraphError::NotBootstrapped);
        }
        let mut staged = self.clone();
        let causes = [AdmittedCauseV0::external_input(source_sequence, payload0, payload1)];
        let result = staged.nodes[staged.active_index]
            .cell
            .as_mut()
            .ok_or(GraphError::NodeNotResident)?
            .transact(&causes, &[], scratch)
            .map_err(GraphError::SourcePhase)?;
        *self = staged;
        Ok(result)
    }

    pub fn evict_node(&mut self, index: usize) -> Result<CellSnapshotReceipt, GraphError> {
        if index >= self.nodes.len() {
            return Err(GraphError::NodeRange);
        }
        if index == self.active_index {
            return Err(GraphError::ActiveNode);
        }
        let cell = self.nodes[index].cell.as_ref().ok_or(GraphError::NodeNotResident)?;
        if cell.world().objects().alive_count() != 0 {
            return Err(GraphError::NodeNotEmpty);
        }
        let (bytes, receipt) = encode(cell).map_err(GraphError::Snapshot)?;
        let mut staged = self.clone();
        staged.nodes[index].cell = None;
        staged.nodes[index].snapshot = Some(bytes);
        staged.nodes[index].snapshot_receipt = Some(receipt);
        *self = staged;
        Ok(receipt)
    }

    pub fn restore_node(&mut self, index: usize) -> Result<CellSnapshotReceipt, GraphError> {
        let node = self.nodes.get(index).ok_or(GraphError::NodeRange)?;
        if node.cell.is_some() {
            return Err(GraphError::NodeAlreadyResident);
        }
        let expected = node.snapshot_receipt.ok_or(GraphError::SnapshotMismatch)?;
        let bytes = node.snapshot.as_ref().ok_or(GraphError::SnapshotMismatch)?;
        let (cell, receipt) = decode(bytes).map_err(GraphError::Snapshot)?;
        if receipt != expected || receipt.source_zone_key != node.zone_key {
            return Err(GraphError::SnapshotMismatch);
        }
        let mut staged = self.clone();
        staged.nodes[index].cell = Some(cell);
        staged.nodes[index].snapshot = None;
        staged.nodes[index].snapshot_receipt = None;
        *self = staged;
        Ok(receipt)
    }

    pub fn provision_node_pair_limbs(
        &mut self,
        index: usize,
        new_limit: u32,
    ) -> Result<(), GraphError> {
        self.nodes
            .get_mut(index)
            .ok_or(GraphError::NodeRange)?
            .cell
            .as_mut()
            .ok_or(GraphError::NodeNotResident)?
            .provision_pair_limbs(new_limit)
            .map_err(GraphError::SourcePhase)
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
    ) -> Result<GraphReceipt, GraphError> {
        if !self.bootstrapped {
            return Err(GraphError::NotBootstrapped);
        }
        if handoff_cause_id == 0 || source_sequence == 0 || destination_sequence == 0 {
            return Err(GraphError::InvalidIdentity);
        }
        if self.consumed_handoff_cause_id == handoff_cause_id {
            let actor_key = self.actor_key.ok_or(GraphError::ActorMissing)?;
            return Ok(GraphReceipt {
                disposition: GraphDisposition::DuplicateIgnored,
                graph_commit_id: self.graph_commit_id,
                source_index: self.active_index,
                destination_index: self.active_index,
                active_index: self.active_index,
                edge_index: usize::MAX,
                edge_key: 0,
                source_primitive: QblPrimitive::NONE,
                destination_primitive: QblPrimitive::NONE,
                source_local_commit_id: self.nodes[self.active_index]
                    .cell
                    .as_ref()
                    .ok_or(GraphError::NodeNotResident)?
                    .local_commit_id(),
                destination_local_commit_id: self.nodes[self.active_index]
                    .cell
                    .as_ref()
                    .ok_or(GraphError::NodeNotResident)?
                    .local_commit_id(),
                traveler_key: self.traveler_key,
                actor_key,
            });
        }
        if destination_index >= self.nodes.len() {
            return Err(GraphError::NodeRange);
        }
        let source_index = self.active_index;
        let edge_index = self
            .edge_for(source_index, destination_index)
            .ok_or(GraphError::EdgeNotFound)?;
        let edge_key = self.edges[edge_index].edge_key;
        let source = self.nodes[source_index]
            .cell
            .as_ref()
            .ok_or(GraphError::NodeNotResident)?;
        let destination = self.nodes[destination_index]
            .cell
            .as_ref()
            .ok_or(GraphError::NodeNotResident)?;
        if source.local_commit_id() != expected_source_local_commit_id
            || destination.local_commit_id() != expected_destination_local_commit_id
        {
            return Err(GraphError::StaleVersion);
        }
        let actor_key = self.actor_key.ok_or(GraphError::ActorMissing)?;
        let health = source
            .world()
            .objects()
            .health(actor_key)
            .ok_or(GraphError::ActorMissing)?;
        let next_commit_id = self
            .graph_commit_id
            .checked_add(1)
            .ok_or(GraphError::GraphCommitExhausted)?;
        let mut staged = self.clone();

        let source_causes = [AdmittedCauseV0::external_input(
            source_sequence,
            staged.traveler_key,
            edge_key,
        )];
        let source_intents = [MutationIntentV0::despawn(source_sequence, actor_key)];
        let source_result = staged.nodes[source_index]
            .cell
            .as_mut()
            .ok_or(GraphError::NodeNotResident)?
            .transact(&source_causes, &source_intents, source_scratch)
            .map_err(GraphError::SourcePhase)?;
        if source_result.disposition != WideDisposition::Committed
            || source_result.rejected_requests != 0
        {
            return Err(GraphError::ActorMissing);
        }

        let expected_destination_key = ObjectKey(
            staged.nodes[destination_index]
                .cell
                .as_ref()
                .ok_or(GraphError::NodeNotResident)?
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
        let destination_result = staged.nodes[destination_index]
            .cell
            .as_mut()
            .ok_or(GraphError::NodeNotResident)?
            .transact(&destination_causes, &destination_intents, destination_scratch)
            .map_err(GraphError::DestinationPhase)?;
        let destination_cell = staged.nodes[destination_index]
            .cell
            .as_ref()
            .ok_or(GraphError::NodeNotResident)?;
        if destination_result.disposition != WideDisposition::Committed
            || destination_result.rejected_requests != 0
            || destination_cell.world().objects().health(expected_destination_key) != Some(health)
        {
            return Err(GraphError::DestinationSpawnRejected);
        }

        staged.active_index = destination_index;
        staged.actor_key = Some(expected_destination_key);
        staged.consumed_handoff_cause_id = handoff_cause_id;
        staged.graph_commit_id = next_commit_id;
        let receipt = GraphReceipt {
            disposition: GraphDisposition::Committed,
            graph_commit_id: next_commit_id,
            source_index,
            destination_index,
            active_index: destination_index,
            edge_index,
            edge_key,
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
        feed(&mut hash, &self.graph_key.to_le_bytes());
        feed(&mut hash, &self.traveler_key.to_le_bytes());
        feed(&mut hash, &self.graph_commit_id.to_le_bytes());
        feed(&mut hash, &self.consumed_handoff_cause_id.to_le_bytes());
        feed(&mut hash, &(self.active_index as u64).to_le_bytes());
        for edge in &self.edges {
            feed(&mut hash, &(edge.node_a as u64).to_le_bytes());
            feed(&mut hash, &(edge.node_b as u64).to_le_bytes());
            feed(&mut hash, &edge.edge_key.to_le_bytes());
        }
        feed(&mut hash, &self.actor_key.unwrap_or(ObjectKey(0)).0.to_le_bytes());
        feed(&mut hash, &[self.bootstrapped as u8]);
        for node in &self.nodes {
            feed(&mut hash, &node.zone_key.to_le_bytes());
            feed(&mut hash, &[node.resident() as u8]);
            if let Some(cell) = &node.cell {
                feed(&mut hash, &cell.phase().diagnostic_fingerprint64().to_le_bytes());
                feed(&mut hash, &cell.world().diagnostic_fingerprint64().to_le_bytes());
            } else if let Some(bytes) = &node.snapshot {
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

    fn graph(limit: u32, health: u16) -> RetainedGraph {
        let zones = [
            ZoneCoord::new(0, 0, 0).unwrap(),
            ZoneCoord::new(1, 0, 0).unwrap(),
            ZoneCoord::new(2, 0, 0).unwrap(),
            ZoneCoord::new(3, 0, 0).unwrap(),
            ZoneCoord::new(4, 0, 0).unwrap(),
        ];
        let edges = [(2, 4), (0, 2), (1, 3), (0, 1)];
        let mut graph = RetainedGraph::new(81, 9001, &zones, &edges, limit).unwrap();
        let mut scratch = Scratch::new();
        graph.bootstrap(100, 1, health, &mut scratch).unwrap();
        graph
    }

    fn active_health(graph: &RetainedGraph) -> u16 {
        let key = graph.actor_key().unwrap();
        graph.cell(graph.active_index()).unwrap().world().objects().health(key).unwrap()
    }

    #[test]
    fn branching_graph_advances_with_inactive_branches_nonresident() {
        let mut graph = graph(4, 91);
        assert_eq!(graph.edge(0), Some(GraphEdge { node_a: 0, node_b: 1, edge_key: 1 }));
        assert_eq!(graph.edge(1), Some(GraphEdge { node_a: 0, node_b: 2, edge_key: 2 }));
        assert_eq!(graph.edge(2), Some(GraphEdge { node_a: 1, node_b: 3, edge_key: 3 }));
        assert_eq!(graph.edge(3), Some(GraphEdge { node_a: 2, node_b: 4, edge_key: 4 }));
        graph.evict_node(1).unwrap();
        graph.evict_node(3).unwrap();
        let mut source = Scratch::new();
        let mut destination = Scratch::new();
        let first = graph.handoff(201, 2, graph.cell(0).unwrap().local_commit_id(), graph.cell(2).unwrap().local_commit_id(), 2, 1, &mut source, &mut destination).unwrap();
        assert_eq!(first.edge_index, 1);
        assert_eq!(first.edge_key, 2);
        graph.advance_active(2, 11, 22, &mut source).unwrap();
        let second = graph.handoff(202, 4, graph.cell(2).unwrap().local_commit_id(), graph.cell(4).unwrap().local_commit_id(), 3, 1, &mut source, &mut destination).unwrap();
        assert_eq!(second.edge_index, 3);
        assert_eq!(graph.active_index(), 4);
        assert_eq!(active_health(&graph), 91);
        graph.evict_node(0).unwrap();
        graph.evict_node(2).unwrap();
        assert_eq!((0..5).filter(|index| graph.resident(*index).unwrap()).count(), 1);
    }

    #[test]
    fn edge_input_order_canonicalizes() {
        let zones = [
            ZoneCoord::new(0, 0, 0).unwrap(),
            ZoneCoord::new(1, 0, 0).unwrap(),
            ZoneCoord::new(2, 0, 0).unwrap(),
            ZoneCoord::new(3, 0, 0).unwrap(),
            ZoneCoord::new(4, 0, 0).unwrap(),
        ];
        let left_edges = [(2, 4), (0, 2), (1, 3), (0, 1)];
        let right_edges = [(3, 1), (1, 0), (4, 2), (2, 0)];
        let left = RetainedGraph::new(81, 9001, &zones, &left_edges, 4).unwrap();
        let right = RetainedGraph::new(81, 9001, &zones, &right_edges, 4).unwrap();
        assert_eq!(left, right);
        assert_eq!(left.edge_for(2, 0), Some(1));
        assert_eq!(left.edge_for(4, 2), Some(3));
    }

    #[test]
    fn nonedge_and_stale_guards_precede_mutation() {
        let mut graph = graph(4, 77);
        let before = graph.clone();
        let node1 = graph.cell(1).unwrap().clone();
        let node3 = graph.cell(3).unwrap().clone();
        let mut source = Scratch::new();
        let mut destination = Scratch::new();
        assert!(matches!(graph.handoff(301, 4, graph.cell(0).unwrap().local_commit_id(), graph.cell(4).unwrap().local_commit_id(), 2, 1, &mut source, &mut destination), Err(GraphError::EdgeNotFound)));
        assert_eq!(graph, before);
        assert!(matches!(graph.handoff(301, 2, graph.cell(0).unwrap().local_commit_id() + 1, graph.cell(2).unwrap().local_commit_id(), 2, 1, &mut source, &mut destination), Err(GraphError::StaleVersion)));
        assert_eq!(graph, before);
        graph.handoff(301, 2, graph.cell(0).unwrap().local_commit_id(), graph.cell(2).unwrap().local_commit_id(), 2, 1, &mut source, &mut destination).unwrap();
        assert_eq!(graph.cell(1).unwrap(), &node1);
        assert_eq!(graph.cell(3).unwrap(), &node3);
    }

    #[test]
    fn corrupt_snapshot_rejection_is_atomic() {
        let mut graph = graph(4, 88);
        graph.evict_node(3).unwrap();
        let mut bytes = graph.snapshot_bytes(3).unwrap().to_vec();
        bytes[9] ^= 0x40;
        graph.install_snapshot(3, bytes).unwrap();
        let before = graph.clone();
        assert!(matches!(graph.restore_node(3), Err(GraphError::Snapshot(_))));
        assert_eq!(graph, before);
    }

    #[test]
    fn provisioning_fault_rolls_back_and_retries_same_edge_cause() {
        let mut graph = graph(1, 66);
        let mut source = Scratch::new();
        let mut destination = Scratch::new();
        for sequence in 2..=154 {
            graph.advance_active(sequence, sequence as u64, 0, &mut source).unwrap();
        }
        let before = graph.clone();
        assert!(matches!(graph.handoff(401, 2, graph.cell(0).unwrap().local_commit_id(), graph.cell(2).unwrap().local_commit_id(), 155, 1, &mut source, &mut destination), Err(GraphError::SourcePhase(WideError::ProvisioningRequired { required_pair_limbs: 2 }))));
        assert_eq!(graph, before);
        graph.provision_node_pair_limbs(0, 2).unwrap();
        let receipt = graph.handoff(401, 2, graph.cell(0).unwrap().local_commit_id(), graph.cell(2).unwrap().local_commit_id(), 155, 1, &mut source, &mut destination).unwrap();
        assert_eq!(receipt.graph_commit_id, 1);
        assert_eq!(active_health(&graph), 66);
    }

    #[test]
    fn duplicate_and_replay_are_deterministic() {
        let mut left = graph(4, 91);
        let mut right = graph(4, 91);
        let mut ls = Scratch::new();
        let mut ld = Scratch::new();
        let mut rs = Scratch::new();
        let mut rd = Scratch::new();
        left.handoff(501, 2, left.cell(0).unwrap().local_commit_id(), left.cell(2).unwrap().local_commit_id(), 2, 1, &mut ls, &mut ld).unwrap();
        right.handoff(501, 2, right.cell(0).unwrap().local_commit_id(), right.cell(2).unwrap().local_commit_id(), 2, 1, &mut rs, &mut rd).unwrap();
        let before = left.clone();
        let duplicate = left.handoff(501, 4, left.cell(2).unwrap().local_commit_id(), left.cell(4).unwrap().local_commit_id(), 2, 1, &mut ls, &mut ld).unwrap();
        assert_eq!(duplicate.disposition, GraphDisposition::DuplicateIgnored);
        assert_eq!(left, before);
        left.advance_active(2, 7, 8, &mut ls).unwrap();
        right.advance_active(2, 7, 8, &mut rs).unwrap();
        left.handoff(502, 4, left.cell(2).unwrap().local_commit_id(), left.cell(4).unwrap().local_commit_id(), 3, 1, &mut ls, &mut ld).unwrap();
        right.handoff(502, 4, right.cell(2).unwrap().local_commit_id(), right.cell(4).unwrap().local_commit_id(), 3, 1, &mut rs, &mut rd).unwrap();
        assert_eq!(left, right);
        assert_eq!(left.diagnostic_fingerprint64(), right.diagnostic_fingerprint64());
    }
}
