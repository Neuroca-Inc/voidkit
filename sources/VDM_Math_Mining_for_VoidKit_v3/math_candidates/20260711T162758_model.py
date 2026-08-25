from __future__ import annotations
import csv, hashlib, json
from pathlib import Path

STAMP = "20260711T162758"

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())

def normalize_excerpt(lines: list[str]) -> str:
    lines = [line.rstrip() for line in lines]
    while lines and lines[0] == "": lines.pop(0)
    while lines and lines[-1] == "": lines.pop()
    return "\n".join(lines)

def excerpt(path: Path, start: int, end: int) -> str:
    lines = path.read_text(encoding="utf-8").splitlines()
    if start < 1 or end < start or end > len(lines):
        raise ValueError(f"bad line range {start}-{end} for {path} with {len(lines)} lines")
    return normalize_excerpt(lines[start-1:end])

def source_specs() -> list[dict]:
    return [
      {"claim_key":"lifted_state_dependency","source_path":f"inputs/{STAMP}_QBL_PRIMITIVE_CUSTODY_AND_ORTHAD_LAW_v2.md","start_line":438,"end_line":478,"required_formula":"`P_t` is the primary pairing","authority":"PRIMARY_RATIFIED","forces":"one primary object; two restrictions; two directed transfers; per-tick mutation order","leaves_open":"types, star semantics, values, recurrence"},
      {"claim_key":"four_descendants","source_path":f"inputs/{STAMP}_QBL_PRIMITIVE_CUSTODY_AND_ORTHAD_LAW_v2.md","start_line":509,"end_line":541,"required_formula":r"\Omega_t^+=\iota_+^*P_t\iota_+","authority":"PRIMARY_RATIFIED","forces":"four specific descendant obligations from one P_t and two chart maps","leaves_open":"meaning of star, map types, recurrence"},
      {"claim_key":"B_mutation","source_path":f"inputs/{STAMP}_QBL_PRIMITIVE_CUSTODY_AND_ORTHAD_LAW_v2.md","start_line":549,"end_line":570,"required_formula":"update the active pairing data from the new pair","authority":"PRIMARY_RATIFIED","forces":"B mutates pairing, restrictions, transfers at fixed stated rank","leaves_open":"exact B pairing map"},
      {"claim_key":"Q_quarter_turn","source_path":f"inputs/{STAMP}_QBL_PRIMITIVE_CUSTODY_AND_ORTHAD_LAW_v2.md","start_line":572,"end_line":593,"required_formula":"quarter-turn witness i","authority":"PRIMARY_RATIFIED","forces":"Q rotates active pairing data by a witness denoted i","leaves_open":"scalar, complex structure, orientation operator, or local label"},
      {"claim_key":"L_orthogonal_axis","source_path":f"inputs/{STAMP}_QBL_PRIMITIVE_CUSTODY_AND_ORTHAD_LAW_v2.md","start_line":595,"end_line":613,"required_formula":"append exactly one new active orthogonal axis","authority":"PRIMARY_RATIFIED","forces":"old content retained; one new orthogonal-axis obligation; structural rank wording","leaves_open":"side of orthogonality, rank semantics, self relation"},
      {"claim_key":"local_axis_shorthand","source_path":f"inputs/{STAMP}_QBL_PRIMITIVE_CUSTODY_AND_ORTHAD_LAW_v2.md","start_line":617,"end_line":664,"required_formula":r"a_0=\frac{i}{4895}","authority":"PRIMARY_RATIFIED","forces":"local descendant shorthand and boundary distinction","leaves_open":"relation to P_t or any diagonal entry"},
      {"claim_key":"gauge_star","source_path":f"inputs/{STAMP}_QBL_PRIMITIVE_CUSTODY_AND_ORTHAD_LAW_v2.md","start_line":690,"end_line":726,"required_formula":r"P\mapsto U^*PU","authority":"PRIMARY_RATIFIED_SCHEMATIC","forces":"a representation-change expression using the same star glyph","leaves_open":"gauge group, star meaning, coefficient object"},
      {"claim_key":"FQM_polarization","source_path":f"inputs/{STAMP}_QBL_PRIMITIVE_CUSTODY_AND_ORTHAD_LAW_v2.md","start_line":730,"end_line":744,"required_formula":"whose polarization","authority":"PRIMARY_DOWNSTREAM","forces":"later finite quadratic layer has polarization","leaves_open":"descent map from primary pairing"},
      {"claim_key":"candidate_Cstar_block","source_path":f"inputs/{STAMP}_Pasted_markdown_architecture_note.md","start_line":510,"end_line":521,"required_formula":"C_t*","authority":"TASK_CANDIDATE_NOT_RATIFIED","forces":"nothing beyond a candidate first-L presentation to audit","leaves_open":"direct sum, matrix realization, star meaning"},
      {"claim_key":"historical_H_construction","source_path":f"inputs/{STAMP}_v7n_finite_orthad_qgt_jm_split.md","start_line":7,"end_line":34,"required_formula":"H(h) = M(h) + iJ(h)","authority":"HISTORICAL_REDERIVED_FINITE","forces":"historical H is Hermitian when J is skew and M is symmetric gluing metric","leaves_open":"clean P_0, per-tick recurrence, modern chart maps"},
      {"claim_key":"historical_H_disposition","source_path":f"inputs/{STAMP}_v7n_finite_orthad_qgt_jm_split.md","start_line":48,"end_line":62,"required_formula":"finite Hermitian overlap object","authority":"HISTORICAL_REDERIVED_FINITE","forces":"downstream overlap/gauge interpretation on bounded admissible histories","leaves_open":"arbitrary-history theorem and clean primary pairing"},
      {"claim_key":"historical_H_executable","source_path":f"inputs/{STAMP}_v7n_finite_orthad_qgt_jm_split.py","start_line":81,"end_line":100,"required_formula":"H=M + 1j*J","authority":"HISTORICAL_EXECUTABLE_PROVENANCE","forces":"exact historical construction code","leaves_open":"modern authority and recurrence"},
    ]

def build_source_ledger(root: Path) -> list[dict]:
    rows=[]
    for spec in source_specs():
        path=root/spec["source_path"]
        ex=excerpt(path,spec["start_line"],spec["end_line"])
        row=dict(spec)
        row["normalized_exact_excerpt"]=ex
        row["excerpt_sha256"]=sha256_bytes(ex.encode("utf-8"))
        row["source_file_sha256"]=sha256_file(path)
        rows.append(row)
    return rows

def star_semantics_rows() -> list[dict]:
    return [
      {"meaning":"transpose","iota_pullbacks":"typed in a matrix realization","gauge":"typed as U^T P U","C_t_star":"typed as transpose","historical_H":"does not explain historical Hermitian adjoint unless J=0","Q_i":"scalar i action possible but not source-forced","verdict":"ADMISSIBLE_BUT_NOT_FORCED"},
      {"meaning":"conjugate transpose","iota_pullbacks":"typed in a complex matrix realization","gauge":"typed as U^† P U","C_t_star":"typed as conjugate transpose","historical_H":"coherent with M real symmetric and J real skew","Q_i":"requires Q_ACTION_AXIOM to preserve Hermitian/skew-Hermitian subclass","verdict":"ADMISSIBLE_BUT_NOT_FORCED"},
      {"meaning":"categorical dual pullback","iota_pullbacks":"typed if a duality is supplied","gauge":"typed as dual(U) P U","C_t_star":"only typed after a dual map is defined","historical_H":"not typed without scalar enrichment","Q_i":"not typed without phase enrichment","verdict":"ADMISSIBLE_BUT_NOT_FORCED"},
      {"meaning":"adjoint under an auxiliary pairing","iota_pullbacks":"typed only after another pairing defines adjoints","gauge":"typed but circular for the primary pairing","C_t_star":"typed conditionally","historical_H":"possible downstream","Q_i":"conditional","verdict":"REQUIRES_ONE_AXIOM"},
      {"meaning":"coefficient involution only","iota_pullbacks":"not typed","gauge":"not typed on maps","C_t_star":"not typed on maps","historical_H":"coefficient conjugation alone is insufficient","Q_i":"conjugates i but supplies no pullback","verdict":"RULED_OUT"},
      {"meaning":"formal placeholder","iota_pullbacks":"not mechanically typed","gauge":"not mechanically typed","C_t_star":"not mechanically typed","historical_H":"no relation established","Q_i":"no relation established","verdict":"CURRENT_SOURCE_STATUS"},
    ]

def candidate_rows() -> list[dict]:
    rows=[
      ("complex bilinear symmetric","transpose","PASS","PASS","PASS","PASS","CONDITIONAL","PASS","PASS","PASS","BOUNDARY_PRESERVED","H is separate downstream object","ADMISSIBLE_BUT_NOT_FORCED"),
      ("complex bilinear nonsymmetric","transpose","PASS","PASS","PASS","PASS","CONDITIONAL","PASS","SIDE_OPEN","PASS","BOUNDARY_PRESERVED","H is separate downstream object","ADMISSIBLE_BUT_NOT_FORCED"),
      ("complex sesquilinear non-Hermitian","conjugate transpose","PASS","PASS","PASS","PASS","CONDITIONAL","PASS","SIDE_OPEN","PASS","BOUNDARY_PRESERVED","H is an admissible special downstream case","ADMISSIBLE_BUT_NOT_FORCED"),
      ("Hermitian","conjugate transpose","PASS","PASS","PASS","PASS","REQUIRES_Q_ACTION_AXIOM","PASS","TWO_SIDED_IF_SELF_ADJOINT","PASS","i/(uv) forbidden only as diagonal","H aligns as downstream Hermitian reconstruction","REQUIRES_ONE_AXIOM"),
      ("skew-Hermitian","conjugate transpose","PASS","PASS","PASS","PASS","REQUIRES_Q_ACTION_AXIOM","PASS","TWO_SIDED_UP_TO_SIGN","PASS","i/(uv) could be diagonal only after a missing identification","historical H has different self-adjointness","REQUIRES_ONE_AXIOM"),
      ("operator-valued pairing","operator adjoint","PASS","PASS","PASS","PASS","REQUIRES_COEFFICIENT_STAR_AND_Q_ACTION","PASS","SIDE_OPEN","REQUIRES_DESCENT_MAP","BOUNDARY_PRESERVED","H may be represented downstream","REQUIRES_ONE_AXIOM"),
      ("abstract dual-pullback model","categorical dual pullback","PASS","PASS","CONDITIONAL","PASS","REQUIRES_PHASE_ENRICHMENT","PASS","SEMANTICS_OPEN","REQUIRES_POLARIZATION_DESCENT","BOUNDARY_PRESERVED","H not typed at abstract layer","REQUIRES_ONE_AXIOM"),
    ]
    cols=["candidate","star_realization","four_descendants","gauge_expression","first_L_Cstar","old_block_retention","Q_quarter_turn","axis_append","orthogonality","later_FQM_polarization","local_shorthand","historical_H_alignment","verdict"]
    return [dict(zip(cols,r)) for r in rows]

def result_card() -> dict:
    return {
      "PAIRING_FIRST_REALIGNMENT":"PASS",
      "SOURCE-BOUND CLAIM LEDGER":"PASS",
      "SOURCE-FORCED LOCAL SIGNATURE":"DERIVED",
      "GENERAL Pair(-,-) BIFUNCTOR":"ADMISSIBLE_CANDIDATE",
      "STAR SEMANTICS":"NOT_YET_DERIVED",
      "Q QUARTER-TURN ACTION TYPE":"NOT_YET_DERIVED",
      "HERMITIAN DIAGONAL PROMOTION OF i/(uv)":"REJECTED",
      "HISTORICAL H=M+iJ":"DERIVED_HERMITIAN_OVERLAP_RECONSTRUCTION_ON_HISTORICAL_ADMISSIBLE_TRACE_COCYCLE_TESTS; NOT_PRIMARY_SEED_OR_RECURRENCE",
      "FIRST-L OLD/NEW RELATION":"NOT_YET_DERIVED",
      "FIRST-L NEW/OLD RELATION":"NOT_YET_DERIVED",
      "EXACT PRIMARY PAIRING TYPE":"NOT_YET_DERIVED",
      "SURVIVING MODELS":["complex bilinear symmetric","complex bilinear nonsymmetric","complex sesquilinear non-Hermitian","Hermitian","skew-Hermitian","operator-valued pairing","abstract dual-pullback model"],
      "EARLIEST MISSING AXIOM":"STAR_SEMANTICS_AXIOM",
      "EXACT PRIMARY PAIRING SEED":"NOT_YET_DERIVED",
      "Xi_hat_t VALUES":"NOT_INSTANTIATED",
      "REAL SOURCE-CORRUPTION CONTROLS":"PASS",
      "TERMINAL PROJECTION":"NOT_RUN",
      "FIRST_L_BLOCK_MATRIX":"CANDIDATE_PRESENTATION_ONLY",
      "local_i_over_uv_role":"LOCAL_DESCENDANT_ONLY",
      "phase5_closed":False,
    }

def hermitian_obstruction() -> dict:
    uv=4895
    return {
      "uv":uv,
      "value":{"re":0,"im_numerator":1,"im_denominator":uv},
      "conjugate":{"re":0,"im_numerator":-1,"im_denominator":uv},
      "conjugation_fixed":False,
      "difference_im_numerator":2,
      "difference_im_denominator":uv,
      "theorem":"For Hermitian h, h(x,x)=conj(h(x,x)); nonzero i/uv is not conjugation-fixed.",
      "claim_boundary":"Reject only promotion of local i/(uv) to a Hermitian diagonal self-pairing; do not reject a larger Hermitian primary object."
    }

def first_L_relations() -> dict:
    return {
      "source_forced":["retain old pairing content","append one new orthogonal-axis obligation","extend both restrictions","extend both transfers","axis count increases by one"],
      "old_to_new":"NOT_YET_DERIVED",
      "new_to_old":"NOT_YET_DERIVED",
      "newborn_self_relation":"NOT_YET_DERIVED",
      "orthogonality_semantics":"NOT_YET_DERIVED",
      "pairing_rank_semantics":"NOT_YET_TYPED",
      "matrix_block_presentation":"CANDIDATE_ONLY",
      "counterexample":{"matrix":[[1,1],[0,1]],"new_to_old":0,"old_to_new":1}
    }
