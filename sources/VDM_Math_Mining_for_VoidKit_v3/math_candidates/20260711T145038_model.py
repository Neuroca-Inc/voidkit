from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import csv
import hashlib
import json

STAMP = "20260711T145038"
STEP = "p5_v8x"
EXPECTED_WORD = "BQQBBBQBQBBQBBL"

STATUS = {
    "PAIRING_FIRST_REALIGNMENT": "PASS",
    "SOURCE_FORCED_PAIRING_INTERFACE": "DERIVED",
    "DUALITY_MORPHISM_MODEL": "ADMISSIBLE_CANDIDATE",
    "PAIRING_REPRESENTABILITY": "NOT_YET_DERIVED",
    "SCALAR_VARIANCE_STATUS": "DOWNSTREAM",
    "FIRST_L_RIGHT_MIXED_BLOCK": "NOT_YET_DERIVED",
    "FIRST_L_LEFT_MIXED_BLOCK": "NOT_YET_DERIVED",
    "FIRST_L_PAIRING_RANK_LAW": "NOT_YET_TYPED",
    "EXACT_PRIMARY_PAIRING_TYPE": "NOT_YET_DERIVED",
    "EXACT_PRIMARY_PAIRING_SEED": "NOT_YET_DERIVED",
    "Xi_hat_t VALUES": "NOT_INSTANTIATED",
    "REAL_CORRUPTION_CONTROLS": "PASS",
    "TERMINAL_PROJECTION": "NOT_RUN",
}

SOURCE_ROWS = [
    {
        "source_id": "S01_LIFTED_STATE",
        "exact_source_path": "inputs/20260711T145038_QBL_PRIMITIVE_CUSTODY_AND_ORTHAD_LAW_v2.md",
        "section_or_lines": "§11, lines 438-457",
        "literal_formula": "Xhat_t=(X_t,P_t,Omega_t^+,Omega_t^-,T_t^{+->-},T_t^{-->+}); P_t is primary; charts are restrictions; T are directed transfers",
        "authority": "PRIMARY",
        "what_it_forces": "one primary pairing object and four descendants in the retained lifted schema",
        "what_it_leaves_open": "pairing codomain, scalar structure, representability, symmetry, basis",
        "contamination_risk": "LOW",
    },
    {
        "source_id": "S02_PAIRING_FIRST",
        "exact_source_path": "inputs/20260711T145038_QBL_PRIMITIVE_CUSTODY_AND_ORTHAD_LAW_v2.md",
        "section_or_lines": "§13, lines 509-523",
        "literal_formula": "primary pairing is generative; diagonal chart matrices are restrictions; off-diagonal blocks are transfers",
        "authority": "PRIMARY",
        "what_it_forces": "pairing-first dependency and one common source for all four blocks",
        "what_it_leaves_open": "exact pairing type and restriction mechanism",
        "contamination_risk": "LOW",
    },
    {
        "source_id": "S03_FOUR_PULLBACKS",
        "exact_source_path": "inputs/20260711T145038_QBL_PRIMITIVE_CUSTODY_AND_ORTHAD_LAW_v2.md",
        "section_or_lines": "§13, lines 525-541",
        "literal_formula": "Omega_a=iota_a^* P iota_a; T_{+->-}=iota_-^* P iota_+; T_{-->+}=iota_+^* P iota_-",
        "authority": "PRIMARY",
        "what_it_forces": "restriction/pullback in both argument positions for one two-slot pairing object",
        "what_it_leaves_open": "whether star means adjoint, dual pullback, conjugation, transpose, or only notation",
        "contamination_risk": "LOW",
    },
    {
        "source_id": "S04_B_MUTATION",
        "exact_source_path": "inputs/20260711T145038_QBL_PRIMITIVE_CUSTODY_AND_ORTHAD_LAW_v2.md",
        "section_or_lines": "§14.1, lines 547-570",
        "literal_formula": "B updates active pairing data from the new pair; rank does not change; prior latched axes preserved",
        "authority": "PRIMARY",
        "what_it_forces": "same-argument-object mutation schema before L",
        "what_it_leaves_open": "actual recurrence and type of active pairing data",
        "contamination_risk": "LOW",
    },
    {
        "source_id": "S05_Q_QUARTER_TURN",
        "exact_source_path": "inputs/20260711T145038_QBL_PRIMITIVE_CUSTODY_AND_ORTHAD_LAW_v2.md",
        "section_or_lines": "§14.2, lines 572-593",
        "literal_formula": "Q rotates active pairing data by the quarter-turn witness i; pair and rank unchanged",
        "authority": "PRIMARY",
        "what_it_forces": "existence of a Q-indexed orientation mutation on active pairing data",
        "what_it_leaves_open": "coefficient object, scalar action, whether i is scalar multiplication on P, and full recurrence",
        "contamination_risk": "LOW",
    },
    {
        "source_id": "S06_L_EXTENSION",
        "exact_source_path": "inputs/20260711T145038_QBL_PRIMITIVE_CUSTODY_AND_ORTHAD_LAW_v2.md",
        "section_or_lines": "§14.3, lines 595-613",
        "literal_formula": "preserve old pairing block; append one new orthogonal active axis; pairing rank rises by one",
        "authority": "PRIMARY",
        "what_it_forces": "architectural axis-block extension and old-block preservation obligation",
        "what_it_leaves_open": "left/right/two-sided orthogonality, rank notion, mixed blocks, newborn self-pairing",
        "contamination_risk": "LOW",
    },
    {
        "source_id": "S07_LOCAL_I",
        "exact_source_path": "inputs/20260711T145038_QBL_PRIMITIVE_CUSTODY_AND_ORTHAD_LAW_v2.md",
        "section_or_lines": "§15, lines 625-664",
        "literal_formula": "local active-axis shorthand reaches i/4895; historical diagonal shorthand is not the complete Orthad",
        "authority": "PRIMARY",
        "what_it_forces": "local trace witness only",
        "what_it_leaves_open": "relationship to a pairing value or chart entry",
        "contamination_risk": "LOW",
    },
    {
        "source_id": "S08_GAUGE_SHAPE",
        "exact_source_path": "inputs/20260711T145038_QBL_PRIMITIVE_CUSTODY_AND_ORTHAD_LAW_v2.md",
        "section_or_lines": "§17.1, lines 690-715",
        "literal_formula": "a lawful change of representation may act schematically by P -> U^* P U",
        "authority": "PRIMARY",
        "what_it_forces": "two-sided re-expression under some lawful representation changes",
        "what_it_leaves_open": "gauge subgroup, star semantics, whether every automorphism is lawful",
        "contamination_risk": "LOW",
    },
    {
        "source_id": "S09_FQM_POLARIZATION",
        "exact_source_path": "inputs/20260711T145038_QBL_PRIMITIVE_CUSTODY_AND_ORTHAD_LAW_v2.md",
        "section_or_lines": "§18, lines 730-746",
        "literal_formula": "q:A->Q/Z; b(x,y)=q(x+y)-q(x)-q(y) is bilinear; Orthad later produces an FQM presentation",
        "authority": "PRIMARY_DOWNSTREAM",
        "what_it_forces": "later scalar-valued bilinear polarization at the FQM layer",
        "what_it_leaves_open": "whether the primary pairing itself is scalar-valued or polarized quadratic",
        "contamination_risk": "MEDIUM_IF_PROMOTED_UPSTREAM",
    },
    {
        "source_id": "S10_DIAGRAM_PAIRING",
        "exact_source_path": "inputs/20260711T145038_orthad-diagram-v5.png",
        "section_or_lines": "Panels 3, 4a, 4b, 7, 8",
        "literal_formula": "P_t primary; both charts and transfers derived from P_t; L preserves old block and appends axis",
        "authority": "SCHEMATIC_AID",
        "what_it_forces": "architectural ordering consistent with the primary text",
        "what_it_leaves_open": "all exact recurrences and mathematical types",
        "contamination_risk": "MEDIUM",
    },
    {
        "source_id": "S11_AUDIT_RETYPE",
        "exact_source_path": "inputs/20260711T145038_p5_v8w_AUDIT_AND_p5_v8x_TASK.zip",
        "section_or_lines": "p5_v8w audit F1-F7",
        "literal_formula": "P:H->D(H), scalar variance, zero mixed blocks, rank+1, and Pair/Aut are candidate or untyped",
        "authority": "RATIFIED_AUDIT_BOUNDARY",
        "what_it_forces": "claim-boundary corrections for p5_v8x",
        "what_it_leaves_open": "positive pairing construction",
        "contamination_risk": "LOW",
    },
]

INFERENCE_RULES = [
    {
        "rule_id": "R01_TWO_SLOT_PULLBACK",
        "premises": ["S02_PAIRING_FIRST", "S03_FOUR_PULLBACKS"],
        "conclusion": "A primary two-slot pairing object with contravariant restriction in each slot is sufficient and source-forced at the interface level.",
        "evidence_class": "SOURCE_DERIVED",
    },
    {
        "rule_id": "R02_NO_REPRESENTABILITY",
        "premises": ["S03_FOUR_PULLBACKS", "S11_AUDIT_RETYPE"],
        "conclusion": "No source row states a representing dual object or natural currying isomorphism; P:H->D(H) is not derived.",
        "evidence_class": "SOURCE_DERIVED",
    },
    {
        "rule_id": "R03_SCALAR_DOWNSTREAM",
        "premises": ["S05_Q_QUARTER_TURN", "S09_FQM_POLARIZATION", "S11_AUDIT_RETYPE"],
        "conclusion": "Scalar variance is downstream of a coefficient object, scalar action, and involution.",
        "evidence_class": "SOURCE_DERIVED",
    },
    {
        "rule_id": "R04_L_MIXED_OPEN",
        "premises": ["S06_L_EXTENSION", "S11_AUDIT_RETYPE"],
        "conclusion": "The word orthogonal does not select left, right, or two-sided vanishing; neither mixed block is derived zero.",
        "evidence_class": "SOURCE_DERIVED",
    },
    {
        "rule_id": "R05_RANK_UNTYPED",
        "premises": ["S06_L_EXTENSION", "S11_AUDIT_RETYPE"],
        "conclusion": "Architectural axis count rises; pairing morphism rank is not yet typed.",
        "evidence_class": "SOURCE_DERIVED",
    },
]

CLAIM_MODEL: dict[str, Any] = {
    "statuses": STATUS,
    "source_forced_interface": {
        "name": "two_slot_pullback_pairing_system",
        "primary": "P_t in Pair(H_t,H_t)",
        "pullback": "Pair(A,B) --(f,g)^*--> Pair(A',B') for f:A'->A, g:B'->B",
        "laws": ["identity", "composition in both slots", "one common primary P_t"],
        "star_semantics": "first-slot pullback notation only at this layer",
        "scalar_object_required": False,
        "duality_object_required": False,
    },
    "duality_morphism_model": {
        "status": "ADMISSIBLE_CANDIDATE",
        "formula": "P_t:H_t->D(H_t)",
        "missing_axiom": "For every H, a representing dual object D(H) and a natural isomorphism Pair(A,H) ~= Hom(A,D(H)) compatible with both-slot pullback.",
    },
    "scalar_variance": {
        "status": "DOWNSTREAM",
        "dependencies": ["coefficient object K", "scalar action on H_t", "involution/star on K", "pairing compatibility"],
    },
    "first_L": {
        "general_block": "[[P_t,C_right],[C_left,p_new]]",
        "right_mixed": "NOT_YET_DERIVED",
        "left_mixed": "NOT_YET_DERIVED",
        "orthogonality_missing_axiom": "Specify left, right, or two-sided orthogonality, or add symmetry/Hermitianity linking the slots.",
        "architectural_axis_count": "DERIVED_1_TO_2",
        "pairing_rank_law": "NOT_YET_TYPED",
    },
    "rank_semantics": {
        "architectural_axis_count": "DERIVED",
        "argument_object_rank_or_dimension": "NOT_YET_DERIVED",
        "block_presentation_size": "CONDITIONAL_ON_REPRESENTATION",
        "pairing_morphism_rank": "NOT_YET_TYPED",
        "nondegenerate_pairing_rank": "NOT_YET_TYPED",
    },
    "gauge_boundary": {
        "licensed": "equivalence generated by source-authorized lawful representation changes G_law",
        "full_Aut_quotient": "ADMISSIBLE_MODEL_NOT_DERIVED",
    },
    "lifted_state_schema": {
        "name": "lifted_state_schema",
        "pairing": None,
        "omega_plus": None,
        "omega_minus": None,
        "transfer_plus_to_minus": None,
        "transfer_minus_to_plus": None,
        "Xi_hat_t_emitted": False,
    },
    "downstream": {
        "chart_values": "NOT_EMITTED",
        "transfer_values": "NOT_EMITTED",
        "projection": "NOT_RUN",
        "gauge_quotient": "NOT_CONSTRUCTED",
        "FQM": "NOT_RUN",
        "Weil": "NOT_RUN",
        "affine": "NOT_RUN",
        "MHD": "NOT_RUN",
    },
    "verifier_evidence_mode": "SOURCE_ROWS_PLUS_EXPLICIT_INFERENCE_RULES",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))
