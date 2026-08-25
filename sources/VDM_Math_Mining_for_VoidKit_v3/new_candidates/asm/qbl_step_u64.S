.intel_syntax noprefix

.section .text

# CP3 ABI probes for qbl_transition_u64.
.global qbl_kernel_transition_size
.type qbl_kernel_transition_size, @function
qbl_kernel_transition_size:
    mov eax, 96
    ret
.size qbl_kernel_transition_size, .-qbl_kernel_transition_size

.global qbl_kernel_transition_align
.type qbl_kernel_transition_align, @function
qbl_kernel_transition_align:
    mov eax, 8
    ret
.size qbl_kernel_transition_align, .-qbl_kernel_transition_align

.macro OFFSET_PROBE name, value
.global \name
.type \name, @function
\name:
    mov eax, \value
    ret
.size \name, .-\name
.endm

OFFSET_PROBE qbl_kernel_transition_primitive_offset, 0
OFFSET_PROBE qbl_kernel_transition_reason_offset, 4
OFFSET_PROBE qbl_kernel_transition_state_changed_offset, 8
OFFSET_PROBE qbl_kernel_transition_before_phase_positions_offset, 16
OFFSET_PROBE qbl_kernel_transition_before_global_position_offset, 24
OFFSET_PROBE qbl_kernel_transition_before_capacity_exponent_offset, 32
OFFSET_PROBE qbl_kernel_transition_after_phase_positions_offset, 40
OFFSET_PROBE qbl_kernel_transition_after_global_position_offset, 48
OFFSET_PROBE qbl_kernel_transition_after_capacity_exponent_offset, 56
OFFSET_PROBE qbl_kernel_transition_before_product_lo_offset, 64
OFFSET_PROBE qbl_kernel_transition_before_product_hi_offset, 72
OFFSET_PROBE qbl_kernel_transition_after_product_lo_offset, 80
OFFSET_PROBE qbl_kernel_transition_after_product_hi_offset, 88

# uint32_t qbl_step_u64(qbl_inspect_state_u64 *state,
#                       qbl_transition_u64 *transition)
#
# The function derives exactly one primitive from the current state under
# B > Q > L and commits both the new custody state and transition receipt only
# after the complete step has succeeded. There is no step counter, cadence,
# modulo-64 rule, external word, or projection.
#
# Successful state mutation:
#   B: (u,v) <- (v,u+v); A,k,quarter_turns unchanged
#   Q: k <- k+1; quarter_turns <- quarter_turns+1; pair,A unchanged
#   L: A <- A+1; k <- 0; pair,quarter_turns unchanged
#
# Any nonzero return leaves both caller-owned buffers unchanged.
.global qbl_step_u64
.type qbl_step_u64, @function
qbl_step_u64:
    test rdi, rdi
    jz .null
    test rsi, rsi
    jz .null
    cmp rdi, rsi
    je .invalid_state_direct

    push rbx
    push r12
    push r13
    push r14
    push r15
    sub rsp, 144

    mov r12, rdi                    # caller state
    mov r13, rsi                    # caller transition
    mov r14, qword ptr [r12 + 0]    # u
    mov r15, qword ptr [r12 + 8]    # v
    mov rbx, qword ptr [r12 + 16]   # domain A

    # Zero temporary transition and temporary state.
    lea rdi, [rsp]
    xor eax, eax
    mov ecx, 18
    cld
    rep stosq

    # Copy the complete old state into the uncommitted state image.
    mov qword ptr [rsp + 96], r14
    mov qword ptr [rsp + 104], r15
    mov qword ptr [rsp + 112], rbx
    mov rax, qword ptr [r12 + 24]
    mov qword ptr [rsp + 120], rax
    mov rax, qword ptr [r12 + 32]
    mov qword ptr [rsp + 128], rax

    test r14, r14
    jz .invalid_pair
    cmp r14, r15
    ja .invalid_pair
    cmp rbx, 59
    ja .domain_range

    # N(A) = 6 * 2^A.
    mov rcx, rbx
    mov r9, 6
    shl r9, cl
    mov qword ptr [rsp + 16], r9

    mov r11, qword ptr [rsp + 120]  # local position k
    cmp r11, r9
    jae .invalid_state

    # j = (N(A)-5)+k.
    mov r10, r9
    sub r10, 5
    add r10, r11
    mov qword ptr [rsp + 24], r10

    # capacity exponent e: Delta(j)=2^e.
    cmp r10, 1
    je .before_exp_one
    cmp r10, 2
    je .before_exp_two
    shl r10, 1
    jmp .before_exp_ready
.before_exp_one:
    mov r10, 1
    jmp .before_exp_ready
.before_exp_two:
    mov r10, 2
.before_exp_ready:
    mov qword ptr [rsp + 32], r10

    # Exact current product in u128.
    mov rax, r14
    mul r15
    mov qword ptr [rsp + 64], rax
    mov qword ptr [rsp + 72], rdx

    # Final position iff k == N(A)-1.
    lea rax, [r9 - 1]
    cmp r11, rax
    je .terminal_position

.nonterminal_position:
    # If current product already meets/exceeds capacity, B is blocked and Q is
    # selected without constructing a candidate pair.
    mov r10, qword ptr [rsp + 32]
    cmp r10, 128
    jae .nonterminal_current_below
    cmp r10, 64
    jb .nonterminal_current_low
    je .nonterminal_current_64

    mov rcx, r10
    sub rcx, 64
    mov r8, 1
    shl r8, cl
    cmp qword ptr [rsp + 72], r8
    jb .nonterminal_current_below
    jmp .select_q

.nonterminal_current_64:
    cmp qword ptr [rsp + 72], 0
    je .nonterminal_current_below
    jmp .select_q

.nonterminal_current_low:
    cmp qword ptr [rsp + 72], 0
    jne .select_q
    mov rcx, r10
    mov r8, 1
    shl r8, cl
    cmp qword ptr [rsp + 64], r8
    jb .nonterminal_current_below
    jmp .select_q

.nonterminal_current_below:
    # Construct B candidate (v,u+v). A representation overflow is not a lawful
    # reason to choose Q.
    mov r11, r14
    add r11, r15
    jc .needs_wide_pair
    mov rax, r15
    mul r11
    mov r8, rax                     # candidate product low
    mov r9, rdx                     # candidate product high

    # Candidate is admitted iff candidate_product <= capacity.
    mov r10, qword ptr [rsp + 32]
    cmp r10, 128
    jae .select_b_nonterminal
    cmp r10, 64
    jb .candidate_low
    je .candidate_64

    mov rcx, r10
    sub rcx, 64
    mov rax, 1
    shl rax, cl
    cmp r9, rax
    jb .select_b_nonterminal
    ja .select_q
    cmp r8, 0
    je .select_b_nonterminal
    jmp .select_q

.candidate_64:
    cmp r9, 1
    jb .select_b_nonterminal
    ja .select_q
    cmp r8, 0
    je .select_b_nonterminal
    jmp .select_q

.candidate_low:
    test r9, r9
    jne .select_q
    mov rcx, r10
    mov rax, 1
    shl rax, cl
    cmp r8, rax
    jbe .select_b_nonterminal
    jmp .select_q

.terminal_position:
    # Final-position B uses the current-anchor rule: current_product < capacity.
    mov r10, qword ptr [rsp + 32]
    cmp r10, 128
    jae .terminal_b_admitted
    cmp r10, 64
    jb .terminal_current_low
    je .terminal_current_64

    mov rcx, r10
    sub rcx, 64
    mov r8, 1
    shl r8, cl
    cmp qword ptr [rsp + 72], r8
    jb .terminal_b_admitted
    jmp .select_l

.terminal_current_64:
    cmp qword ptr [rsp + 72], 0
    je .terminal_b_admitted
    jmp .select_l

.terminal_current_low:
    cmp qword ptr [rsp + 72], 0
    jne .select_l
    mov rcx, r10
    mov r8, 1
    shl r8, cl
    cmp qword ptr [rsp + 64], r8
    jb .terminal_b_admitted
    jmp .select_l

.terminal_b_admitted:
    mov r11, r14
    add r11, r15
    jc .needs_wide_pair
    mov dword ptr [rsp + 0], 1      # B
    mov dword ptr [rsp + 4], 2      # final crossing
    mov qword ptr [rsp + 96], r15
    mov qword ptr [rsp + 104], r11
    jmp .finalize

.select_b_nonterminal:
    mov dword ptr [rsp + 0], 1      # B
    mov dword ptr [rsp + 4], 1      # next pair within capacity
    mov qword ptr [rsp + 96], r15
    mov qword ptr [rsp + 104], r11
    jmp .finalize

.select_q:
    mov rax, qword ptr [rsp + 128]
    cmp rax, -1
    je .overflow
    add qword ptr [rsp + 120], 1
    add qword ptr [rsp + 128], 1
    mov dword ptr [rsp + 0], 2      # Q
    mov dword ptr [rsp + 4], 3      # B blocked, position available
    jmp .finalize

.select_l:
    # CP3's u64 profile cannot represent a successor state outside its admitted
    # domain range. This is a machine profile limit, not mathematical halt.
    cmp rbx, 59
    je .domain_range
    add qword ptr [rsp + 112], 1
    mov qword ptr [rsp + 120], 0
    mov dword ptr [rsp + 0], 3      # L
    mov dword ptr [rsp + 4], 4      # domain saturated

.finalize:
    mov dword ptr [rsp + 8], 1      # state changed

    # Derive after-state N, j, and capacity from the uncommitted new state.
    mov rax, qword ptr [rsp + 112]
    mov rcx, rax
    mov r9, 6
    shl r9, cl
    mov qword ptr [rsp + 40], r9

    mov r10, r9
    sub r10, 5
    add r10, qword ptr [rsp + 120]
    mov qword ptr [rsp + 48], r10

    cmp r10, 1
    je .after_exp_one
    cmp r10, 2
    je .after_exp_two
    shl r10, 1
    jmp .after_exp_ready
.after_exp_one:
    mov r10, 1
    jmp .after_exp_ready
.after_exp_two:
    mov r10, 2
.after_exp_ready:
    mov qword ptr [rsp + 56], r10

    mov rax, qword ptr [rsp + 96]
    mul qword ptr [rsp + 104]
    mov qword ptr [rsp + 80], rax
    mov qword ptr [rsp + 88], rdx

    # Commit the transition receipt, then the complete state image. All failure
    # paths return before this point.
    lea rsi, [rsp]
    mov rdi, r13
    mov ecx, 12
    rep movsq

    lea rsi, [rsp + 96]
    mov rdi, r12
    mov ecx, 5
    rep movsq

    xor eax, eax
    jmp .restore

.invalid_pair:
    mov eax, 2
    jmp .restore
.invalid_state:
    mov eax, 4
    jmp .restore
.domain_range:
    mov eax, 5
    jmp .restore
.needs_wide_pair:
    mov eax, 6
    jmp .restore
.overflow:
    mov eax, 3

.restore:
    add rsp, 144
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    ret

.invalid_state_direct:
    mov eax, 4
    ret
.null:
    mov eax, 1
    ret
.size qbl_step_u64, .-qbl_step_u64


# CP4 ABI probes for qbl_retained_state_u64.
.global qbl_kernel_retained_state_size
.type qbl_kernel_retained_state_size, @function
qbl_kernel_retained_state_size:
    mov eax, 64
    ret
.size qbl_kernel_retained_state_size, .-qbl_kernel_retained_state_size

.global qbl_kernel_retained_state_align
.type qbl_kernel_retained_state_align, @function
qbl_kernel_retained_state_align:
    mov eax, 8
    ret
.size qbl_kernel_retained_state_align, .-qbl_kernel_retained_state_align

OFFSET_PROBE qbl_kernel_retained_state_u_offset, 0
OFFSET_PROBE qbl_kernel_retained_state_v_offset, 8
OFFSET_PROBE qbl_kernel_retained_state_domain_offset, 16
OFFSET_PROBE qbl_kernel_retained_state_local_position_offset, 24
OFFSET_PROBE qbl_kernel_retained_state_quarter_turns_offset, 32
OFFSET_PROBE qbl_kernel_retained_state_word_bytes_offset, 40
OFFSET_PROBE qbl_kernel_retained_state_word_capacity_offset, 48
OFFSET_PROBE qbl_kernel_retained_state_word_length_offset, 56

# uint32_t qbl_step_record_u64(qbl_retained_state_u64 *state,
#                              qbl_transition_u64 *transition)
#
# Extends qbl_step_u64 with exact two-bit word custody. The target history slot
# is checked before the primitive step is evaluated. The primitive transition
# runs against stack-local state and receipt images. Caller-owned state, word,
# length, and receipt are changed only after every check has succeeded.
#
# Encoding: 00 unused, 01 B, 10 Q, 11 L. Four primitives per byte, least
# significant two-bit slot first.
.global qbl_step_record_u64
.type qbl_step_record_u64, @function
qbl_step_record_u64:
    test rdi, rdi
    jz .record_null
    test rsi, rsi
    jz .record_null
    cmp rdi, rsi
    je .record_invalid_state_direct

    # Reject any overlap between the 64-byte retained state and 96-byte receipt.
    lea rax, [rdi + 64]
    cmp rsi, rax
    jae .record_state_receipt_disjoint
    lea rdx, [rsi + 96]
    cmp rdi, rdx
    jb .record_invalid_state_direct
.record_state_receipt_disjoint:

    push rbx
    push r12
    push r13
    push r14
    push r15
    sub rsp, 160

    mov r12, rdi                    # retained state
    mov r13, rsi                    # caller receipt
    mov r14, qword ptr [r12 + 40]   # word bytes
    test r14, r14
    jz .record_invalid_word

    mov rax, qword ptr [r12 + 56]   # word length
    mov rcx, qword ptr [r12 + 48]   # word capacity
    cmp rax, rcx
    ja .record_invalid_word
    je .record_word_full

    mov rdx, rax
    shr rdx, 2                      # byte index
    add r14, rdx                    # target byte address
    jc .record_invalid_word

    # The live target byte may not alias the retained descriptor or receipt.
    cmp r14, r12
    jb .record_target_not_in_state
    lea rdx, [r12 + 64]
    cmp r14, rdx
    jb .record_invalid_word
.record_target_not_in_state:
    cmp r14, r13
    jb .record_target_not_in_receipt
    lea rdx, [r13 + 96]
    cmp r14, rdx
    jb .record_invalid_word
.record_target_not_in_receipt:

    mov rax, qword ptr [r12 + 56]
    and eax, 3
    shl eax, 1                      # target shift
    mov ebx, eax                    # preserve shift across call

    movzx edx, byte ptr [r14]
    mov ecx, ebx
    mov r8d, 3
    shl r8d, cl
    test edx, r8d
    jnz .record_invalid_word
    mov r15d, edx                   # preserve original byte

    # Stack-local custody state at rsp+0.
    mov rax, qword ptr [r12 + 0]
    mov qword ptr [rsp + 0], rax
    mov rax, qword ptr [r12 + 8]
    mov qword ptr [rsp + 8], rax
    mov rax, qword ptr [r12 + 16]
    mov qword ptr [rsp + 16], rax
    mov rax, qword ptr [r12 + 24]
    mov qword ptr [rsp + 24], rax
    mov rax, qword ptr [r12 + 32]
    mov qword ptr [rsp + 32], rax

    lea rdi, [rsp + 0]
    lea rsi, [rsp + 48]
    call qbl_step_u64
    test eax, eax
    jnz .record_restore

    mov eax, dword ptr [rsp + 48]   # primitive code
    cmp eax, 1
    jb .record_internal_invalid
    cmp eax, 3
    ja .record_internal_invalid
    mov ecx, ebx
    shl eax, cl
    or eax, r15d

    # Commit exact history slot, new custody coordinates, word length, and
    # receipt. No failure path is reachable after the first store.
    mov byte ptr [r14], al

    mov rax, qword ptr [rsp + 0]
    mov qword ptr [r12 + 0], rax
    mov rax, qword ptr [rsp + 8]
    mov qword ptr [r12 + 8], rax
    mov rax, qword ptr [rsp + 16]
    mov qword ptr [r12 + 16], rax
    mov rax, qword ptr [rsp + 24]
    mov qword ptr [r12 + 24], rax
    mov rax, qword ptr [rsp + 32]
    mov qword ptr [r12 + 32], rax
    add qword ptr [r12 + 56], 1

    lea rsi, [rsp + 48]
    mov rdi, r13
    mov ecx, 12
    cld
    rep movsq

    xor eax, eax
    jmp .record_restore

.record_internal_invalid:
    mov eax, 4
    jmp .record_restore
.record_invalid_word:
    mov eax, 8
    jmp .record_restore
.record_word_full:
    mov eax, 7

.record_restore:
    add rsp, 160
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    ret

.record_invalid_state_direct:
    mov eax, 4
    ret
.record_null:
    mov eax, 1
    ret
.size qbl_step_record_u64, .-qbl_step_record_u64

.section .note.GNU-stack,"",@progbits

.section .text

# CP5 ABI probes for the recovered local Orthad active-axis state.
.global qbl_kernel_orthad_axis_size
.type qbl_kernel_orthad_axis_size, @function
qbl_kernel_orthad_axis_size:
    mov eax, 40
    ret
.size qbl_kernel_orthad_axis_size, .-qbl_kernel_orthad_axis_size

.global qbl_kernel_orthad_axis_align
.type qbl_kernel_orthad_axis_align, @function
qbl_kernel_orthad_axis_align:
    mov eax, 8
    ret
.size qbl_kernel_orthad_axis_align, .-qbl_kernel_orthad_axis_align

.global qbl_kernel_orthad_local_state_size
.type qbl_kernel_orthad_local_state_size, @function
qbl_kernel_orthad_local_state_size:
    mov eax, 64
    ret
.size qbl_kernel_orthad_local_state_size, .-qbl_kernel_orthad_local_state_size

.global qbl_kernel_orthad_local_state_align
.type qbl_kernel_orthad_local_state_align, @function
qbl_kernel_orthad_local_state_align:
    mov eax, 8
    ret
.size qbl_kernel_orthad_local_state_align, .-qbl_kernel_orthad_local_state_align

OFFSET_PROBE qbl_kernel_orthad_axis_origin_product_lo_offset, 0
OFFSET_PROBE qbl_kernel_orthad_axis_origin_product_hi_offset, 8
OFFSET_PROBE qbl_kernel_orthad_axis_current_product_lo_offset, 16
OFFSET_PROBE qbl_kernel_orthad_axis_current_product_hi_offset, 24
OFFSET_PROBE qbl_kernel_orthad_axis_phase_quadrant_offset, 32
OFFSET_PROBE qbl_kernel_orthad_axis_flags_offset, 36
OFFSET_PROBE qbl_kernel_orthad_local_state_u_offset, 0
OFFSET_PROBE qbl_kernel_orthad_local_state_v_offset, 8
OFFSET_PROBE qbl_kernel_orthad_local_state_domain_offset, 16
OFFSET_PROBE qbl_kernel_orthad_local_state_local_position_offset, 24
OFFSET_PROBE qbl_kernel_orthad_local_state_quarter_turns_offset, 32
OFFSET_PROBE qbl_kernel_orthad_local_state_axes_offset, 40
OFFSET_PROBE qbl_kernel_orthad_local_state_axis_capacity_offset, 48
OFFSET_PROBE qbl_kernel_orthad_local_state_axis_count_offset, 56

# int qbl_ranges_overlap_internal(start_a, size_a, start_b, size_b)
# Returns eax=1 when the half-open ranges overlap, otherwise eax=0.
.type qbl_ranges_overlap_internal, @function
qbl_ranges_overlap_internal:
    cmp rdi, rdx
    ja .ranges_a_after_b
    mov rax, rdx
    sub rax, rdi
    cmp rax, rsi
    setb al
    movzx eax, al
    ret
.ranges_a_after_b:
    mov rax, rdi
    sub rax, rdx
    cmp rax, rcx
    setb al
    movzx eax, al
    ret
.size qbl_ranges_overlap_internal, .-qbl_ranges_overlap_internal

# uint32_t qbl_step_orthad_local_u64(qbl_orthad_local_state_u64 *state,
#                                    qbl_transition_u64 *transition)
#
# The primitive exists only as a transient receipt value. The same atomic tick
# mutates the custody coordinates and the recovered local Orthad active-axis
# descendant. No flat B/Q/L word is retained or consulted.
#
# Exact local-axis representation:
#   a = i^phase * origin_product / current_product
#
# B updates current_product to the new pair product.
# Q advances phase modulo four.
# L latches the completed axis and appends a new identity axis whose origin and
# current products both equal the carried pair product.
#
# This is the recovered active-axis kernel only. It does not invent the missing
# primary-pairing, counter-oriented dual-chart, or directed-transfer recurrence.
.global qbl_step_orthad_local_u64
.type qbl_step_orthad_local_u64, @function
qbl_step_orthad_local_u64:
    test rdi, rdi
    jz .orthad_null_direct
    test rsi, rsi
    jz .orthad_null_direct

    # Caller state and receipt ranges may not overlap.
    push rbx
    push r12
    push r13
    push r14
    push r15
    sub rsp, 256

    mov r12, rdi                    # caller state
    mov r13, rsi                    # caller receipt

    mov rdi, r12
    mov rsi, 64
    mov rdx, r13
    mov rcx, 96
    call qbl_ranges_overlap_internal
    test eax, eax
    jnz .orthad_invalid_state

    mov r14, qword ptr [r12 + 40]   # axes pointer
    test r14, r14
    jz .orthad_invalid
    mov rbx, qword ptr [r12 + 56]   # axis_count
    test rbx, rbx
    jz .orthad_invalid
    cmp rbx, qword ptr [r12 + 48]   # axis_capacity
    ja .orthad_invalid

    # Exactly one active axis per admitted domain: axis_count == A + 1.
    mov rax, qword ptr [r12 + 16]
    cmp rax, -1
    je .orthad_invalid
    add rax, 1
    cmp rbx, rax
    jne .orthad_invalid

    # Validate the complete used axis range against overflow and aliasing.
    mov rax, rbx
    imul rax, rax, 40
    jo .orthad_invalid
    mov r15, rax                    # used axis bytes
    mov rdx, r14
    add rdx, r15
    jc .orthad_invalid

    mov rdi, r14
    mov rsi, r15
    mov rdx, r12
    mov rcx, 64
    call qbl_ranges_overlap_internal
    test eax, eax
    jnz .orthad_invalid
    mov rdi, r14
    mov rsi, r15
    mov rdx, r13
    mov rcx, 96
    call qbl_ranges_overlap_internal
    test eax, eax
    jnz .orthad_invalid

    # Active axis is the final used entry.
    mov rax, rbx
    sub rax, 1
    imul rax, rax, 40
    jo .orthad_invalid
    mov r15, r14
    add r15, rax
    jc .orthad_invalid

    cmp dword ptr [r15 + 36], 1     # ACTIVE
    jne .orthad_invalid
    cmp dword ptr [r15 + 32], 3
    ja .orthad_invalid
    mov rax, qword ptr [r15 + 0]
    or rax, qword ptr [r15 + 8]
    jz .orthad_invalid
    mov rax, qword ptr [r15 + 16]
    or rax, qword ptr [r15 + 24]
    jz .orthad_invalid

    # Active local phase is the domain-local Q position modulo four.
    mov eax, dword ptr [r12 + 24]
    and eax, 3
    cmp eax, dword ptr [r15 + 32]
    jne .orthad_invalid

    # Clear receipt, custody image, active-axis image, and new-axis image.
    lea rdi, [rsp]
    xor eax, eax
    mov ecx, 27                     # 216 bytes
    cld
    rep stosq

    # Stack-local custody image at rsp+96.
    mov rax, qword ptr [r12 + 0]
    mov qword ptr [rsp + 96], rax
    mov rax, qword ptr [r12 + 8]
    mov qword ptr [rsp + 104], rax
    mov rax, qword ptr [r12 + 16]
    mov qword ptr [rsp + 112], rax
    mov rax, qword ptr [r12 + 24]
    mov qword ptr [rsp + 120], rax
    mov rax, qword ptr [r12 + 32]
    mov qword ptr [rsp + 128], rax

    # Stack-local active-axis image at rsp+136.
    mov rax, qword ptr [r15 + 0]
    mov qword ptr [rsp + 136], rax
    mov rax, qword ptr [r15 + 8]
    mov qword ptr [rsp + 144], rax
    mov rax, qword ptr [r15 + 16]
    mov qword ptr [rsp + 152], rax
    mov rax, qword ptr [r15 + 24]
    mov qword ptr [rsp + 160], rax
    mov rax, qword ptr [r15 + 32]
    mov qword ptr [rsp + 168], rax

    # Derive and mutate one primitive against local custody only.
    lea rdi, [rsp + 96]
    lea rsi, [rsp]
    call qbl_step_u64
    test eax, eax
    jnz .orthad_restore

    # The active lens value must be coherent with the before-pair product.
    mov rax, qword ptr [rsp + 64]
    cmp qword ptr [rsp + 152], rax
    jne .orthad_invalid
    mov rax, qword ptr [rsp + 72]
    cmp qword ptr [rsp + 160], rax
    jne .orthad_invalid

    mov eax, dword ptr [rsp + 0]    # transient primitive
    cmp eax, 1
    je .orthad_apply_b
    cmp eax, 2
    je .orthad_apply_q
    cmp eax, 3
    je .orthad_apply_l
    jmp .orthad_invalid_state

.orthad_apply_b:
    mov rax, qword ptr [rsp + 80]
    mov qword ptr [rsp + 152], rax
    mov rax, qword ptr [rsp + 88]
    mov qword ptr [rsp + 160], rax
    xor r11d, r11d                  # no appended axis
    jmp .orthad_commit

.orthad_apply_q:
    mov eax, dword ptr [rsp + 168]
    add eax, 1
    and eax, 3
    mov dword ptr [rsp + 168], eax
    xor r11d, r11d
    jmp .orthad_commit

.orthad_apply_l:
    cmp rbx, qword ptr [r12 + 48]
    je .orthad_full

    # Compute and validate the clean target slot at axes[axis_count].
    mov rax, rbx
    imul rax, rax, 40
    jo .orthad_invalid
    mov r10, r14
    add r10, rax
    jc .orthad_invalid

    mov rdi, r10
    mov rsi, 40
    mov rdx, r12
    mov rcx, 64
    call qbl_ranges_overlap_internal
    test eax, eax
    jnz .orthad_invalid
    mov rdi, r10
    mov rsi, 40
    mov rdx, r13
    mov rcx, 96
    call qbl_ranges_overlap_internal
    test eax, eax
    jnz .orthad_invalid

    mov rdi, r10
    mov ecx, 5
    xor eax, eax
.orthad_check_target_zero:
    cmp qword ptr [rdi], 0
    jne .orthad_invalid
    add rdi, 8
    loop .orthad_check_target_zero

    # Latch old active axis.
    mov dword ptr [rsp + 172], 2    # LATCHED

    # New identity axis at rsp+176: origin=current=carried pair product.
    mov rax, qword ptr [rsp + 80]
    mov qword ptr [rsp + 176], rax
    mov qword ptr [rsp + 192], rax
    mov rax, qword ptr [rsp + 88]
    mov qword ptr [rsp + 184], rax
    mov qword ptr [rsp + 200], rax
    mov dword ptr [rsp + 208], 0
    mov dword ptr [rsp + 212], 1    # ACTIVE
    mov r11d, 1                    # append axis

.orthad_commit:
    # No failure path exists beyond this point.
    # Commit active-axis mutation.
    lea rsi, [rsp + 136]
    mov rdi, r15
    mov ecx, 5
    rep movsq

    # L appends one active identity axis.
    test r11d, r11d
    jz .orthad_commit_state
    mov rax, rbx
    imul rax, rax, 40
    mov rdi, r14
    add rdi, rax
    lea rsi, [rsp + 176]
    mov ecx, 5
    rep movsq
    add rbx, 1
    mov qword ptr [r12 + 56], rbx

.orthad_commit_state:
    lea rsi, [rsp + 96]
    mov rdi, r12
    mov ecx, 5
    rep movsq

    lea rsi, [rsp]
    mov rdi, r13
    mov ecx, 12
    rep movsq

    xor eax, eax
    jmp .orthad_restore

.orthad_full:
    mov eax, 9
    jmp .orthad_restore
.orthad_invalid:
    mov eax, 10
    jmp .orthad_restore
.orthad_invalid_state:
    mov eax, 4

.orthad_restore:
    add rsp, 256
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    ret

.orthad_null_direct:
    mov eax, 1
    ret
.size qbl_step_orthad_local_u64, .-qbl_step_orthad_local_u64
