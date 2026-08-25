.intel_syntax noprefix

.section .text

# CP2 additive ABI probes for qbl_inspect_state_u64.
.global qbl_kernel_inspect_state_size
.type qbl_kernel_inspect_state_size, @function
qbl_kernel_inspect_state_size:
    mov eax, 40
    ret
.size qbl_kernel_inspect_state_size, .-qbl_kernel_inspect_state_size

.global qbl_kernel_inspect_state_align
.type qbl_kernel_inspect_state_align, @function
qbl_kernel_inspect_state_align:
    mov eax, 8
    ret
.size qbl_kernel_inspect_state_align, .-qbl_kernel_inspect_state_align

.global qbl_kernel_inspect_state_u_offset
.type qbl_kernel_inspect_state_u_offset, @function
qbl_kernel_inspect_state_u_offset:
    xor eax, eax
    ret
.size qbl_kernel_inspect_state_u_offset, .-qbl_kernel_inspect_state_u_offset

.global qbl_kernel_inspect_state_v_offset
.type qbl_kernel_inspect_state_v_offset, @function
qbl_kernel_inspect_state_v_offset:
    mov eax, 8
    ret
.size qbl_kernel_inspect_state_v_offset, .-qbl_kernel_inspect_state_v_offset

.global qbl_kernel_inspect_state_domain_offset
.type qbl_kernel_inspect_state_domain_offset, @function
qbl_kernel_inspect_state_domain_offset:
    mov eax, 16
    ret
.size qbl_kernel_inspect_state_domain_offset, .-qbl_kernel_inspect_state_domain_offset

.global qbl_kernel_inspect_state_local_position_offset
.type qbl_kernel_inspect_state_local_position_offset, @function
qbl_kernel_inspect_state_local_position_offset:
    mov eax, 24
    ret
.size qbl_kernel_inspect_state_local_position_offset, .-qbl_kernel_inspect_state_local_position_offset

.global qbl_kernel_inspect_state_quarter_turns_offset
.type qbl_kernel_inspect_state_quarter_turns_offset, @function
qbl_kernel_inspect_state_quarter_turns_offset:
    mov eax, 32
    ret
.size qbl_kernel_inspect_state_quarter_turns_offset, .-qbl_kernel_inspect_state_quarter_turns_offset

# CP2 additive ABI probes for qbl_inspection_u64.
.global qbl_kernel_inspection_size
.type qbl_kernel_inspection_size, @function
qbl_kernel_inspection_size:
    mov eax, 96
    ret
.size qbl_kernel_inspection_size, .-qbl_kernel_inspection_size

.global qbl_kernel_inspection_align
.type qbl_kernel_inspection_align, @function
qbl_kernel_inspection_align:
    mov eax, 8
    ret
.size qbl_kernel_inspection_align, .-qbl_kernel_inspection_align

.macro OFFSET_PROBE name, value
.global \name
.type \name, @function
\name:
    mov eax, \value
    ret
.size \name, .-\name
.endm

OFFSET_PROBE qbl_kernel_inspection_primitive_offset, 0
OFFSET_PROBE qbl_kernel_inspection_reason_offset, 4
OFFSET_PROBE qbl_kernel_inspection_can_b_offset, 8
OFFSET_PROBE qbl_kernel_inspection_can_q_offset, 12
OFFSET_PROBE qbl_kernel_inspection_terminal_position_offset, 16
OFFSET_PROBE qbl_kernel_inspection_phase_positions_offset, 24
OFFSET_PROBE qbl_kernel_inspection_global_position_offset, 32
OFFSET_PROBE qbl_kernel_inspection_capacity_exponent_offset, 40
OFFSET_PROBE qbl_kernel_inspection_next_u_offset, 48
OFFSET_PROBE qbl_kernel_inspection_next_v_offset, 56
OFFSET_PROBE qbl_kernel_inspection_current_product_lo_offset, 64
OFFSET_PROBE qbl_kernel_inspection_current_product_hi_offset, 72
OFFSET_PROBE qbl_kernel_inspection_candidate_product_lo_offset, 80
OFFSET_PROBE qbl_kernel_inspection_candidate_product_hi_offset, 88

.global qbl_inspect_u64
.type qbl_inspect_u64, @function
# System V AMD64 ABI
#
# Input:
#   rdi -> const qbl_inspect_state_u64
#   rsi -> qbl_inspection_u64
#
# Success:
#   eax = QBL_STATUS_OK
#   output contains the unique next primitive and exact admission reason
#
# Failure:
#   state and output are unchanged
#   eax is QBL_STATUS_NULL, QBL_STATUS_INVALID_PAIR,
#   QBL_STATUS_INVALID_STATE, QBL_STATUS_DOMAIN_RANGE, or
#   QBL_STATUS_NEEDS_WIDE_PAIR
#
# This function never mutates custody state and never schedules by time.
qbl_inspect_u64:
    test rdi, rdi
    jz .null
    test rsi, rsi
    jz .null

    push rbx
    push r12
    push r13
    push r14
    push r15
    sub rsp, 96

    mov r12, rdi                    # state
    mov r13, rsi                    # destination
    mov r14, qword ptr [r12 + 0]    # u
    mov r15, qword ptr [r12 + 8]    # v

    # Clear the temporary output. The caller's output is committed only on success.
    xor eax, eax
    mov qword ptr [rsp + 0], rax
    mov qword ptr [rsp + 8], rax
    mov qword ptr [rsp + 16], rax
    mov qword ptr [rsp + 24], rax
    mov qword ptr [rsp + 32], rax
    mov qword ptr [rsp + 40], rax
    mov qword ptr [rsp + 48], rax
    mov qword ptr [rsp + 56], rax
    mov qword ptr [rsp + 64], rax
    mov qword ptr [rsp + 72], rax
    mov qword ptr [rsp + 80], rax
    mov qword ptr [rsp + 88], rax

    test r14, r14
    jz .invalid_pair
    cmp r14, r15
    ja .invalid_pair

    mov rax, qword ptr [r12 + 16]   # domain A
    cmp rax, 59
    ja .domain_range

    # N(A) = 6 * 2^A. Domain <= 59 keeps N, j, and 2j in u64.
    mov rcx, rax
    mov rbx, 6
    shl rbx, cl
    mov qword ptr [rsp + 24], rbx

    mov r11, qword ptr [r12 + 24]   # local position k
    cmp r11, rbx
    jae .invalid_state

    # j_start(A) = N(A) - 5; j = j_start(A) + k.
    mov r10, rbx
    sub r10, 5
    add r10, r11
    mov qword ptr [rsp + 32], r10

    # Delta(j) = 2^e, with e=1 at j=1, e=2 at j=2, e=2j for j>=3.
    cmp r10, 1
    je .capacity_exp_one
    cmp r10, 2
    je .capacity_exp_two
    shl r10, 1
    jmp .capacity_exp_ready
.capacity_exp_one:
    mov r10, 1
    jmp .capacity_exp_ready
.capacity_exp_two:
    mov r10, 2
.capacity_exp_ready:
    mov qword ptr [rsp + 40], r10

    # terminal_position = (k == N(A)-1)
    lea rax, [rbx - 1]
    cmp r11, rax
    sete al
    movzx eax, al
    mov dword ptr [rsp + 16], eax

    # current product u*v, exact in 128 bits.
    mov rax, r14
    mul r15
    mov qword ptr [rsp + 64], rax
    mov qword ptr [rsp + 72], rdx

    cmp dword ptr [rsp + 16], 0
    jne .terminal_position

.nonterminal_position:
    # Q remains available at every nonterminal position.
    mov dword ptr [rsp + 12], 1

    # Candidate product is strictly greater than current product. If the current
    # product already meets or exceeds capacity, B is blocked without requiring
    # a representable candidate pair.
    mov r10, qword ptr [rsp + 40]
    cmp r10, 128
    jae .nonterminal_current_below_capacity
    cmp r10, 64
    jb .nonterminal_current_capacity_low
    je .nonterminal_current_capacity_64

    mov rcx, r10
    sub rcx, 64
    mov r8, 1
    shl r8, cl
    cmp qword ptr [rsp + 72], r8
    jb .nonterminal_current_below_capacity
    jmp .admit_q

.nonterminal_current_capacity_64:
    cmp qword ptr [rsp + 72], 0
    je .nonterminal_current_below_capacity
    jmp .admit_q

.nonterminal_current_capacity_low:
    cmp qword ptr [rsp + 72], 0
    jne .admit_q
    mov rcx, r10
    mov r8, 1
    shl r8, cl
    cmp qword ptr [rsp + 64], r8
    jb .nonterminal_current_below_capacity
    jmp .admit_q

.nonterminal_current_below_capacity:
    # Candidate pair (v, u+v). Representation overflow is not a lawful B block.
    mov r11, r14
    add r11, r15
    jc .needs_wide_pair
    mov qword ptr [rsp + 48], r15
    mov qword ptr [rsp + 56], r11

    mov rax, r15
    mul r11
    mov qword ptr [rsp + 80], rax
    mov qword ptr [rsp + 88], rdx

    # Test candidate_product <= 2^capacity_exponent.
    mov r10, qword ptr [rsp + 40]
    cmp r10, 128
    jae .admit_b_nonterminal
    cmp r10, 64
    jb .candidate_capacity_low
    je .candidate_capacity_64

    mov rcx, r10
    sub rcx, 64
    mov r8, 1
    shl r8, cl
    mov r9, qword ptr [rsp + 88]
    cmp r9, r8
    jb .admit_b_nonterminal
    ja .admit_q
    cmp qword ptr [rsp + 80], 0
    je .admit_b_nonterminal
    jmp .admit_q

.candidate_capacity_64:
    mov r9, qword ptr [rsp + 88]
    cmp r9, 1
    jb .admit_b_nonterminal
    ja .admit_q
    cmp qword ptr [rsp + 80], 0
    je .admit_b_nonterminal
    jmp .admit_q

.candidate_capacity_low:
    cmp qword ptr [rsp + 88], 0
    jne .admit_q
    mov rcx, r10
    mov r8, 1
    shl r8, cl
    cmp qword ptr [rsp + 80], r8
    jbe .admit_b_nonterminal
    jmp .admit_q

.terminal_position:
    # Q is blocked at the final phase position. B uses the current-anchor rule:
    # admit iff current_product < 2^capacity_exponent.
    mov r10, qword ptr [rsp + 40]
    cmp r10, 128
    jae .terminal_b_mathematically_admitted
    cmp r10, 64
    jb .current_capacity_low
    je .current_capacity_64

    mov rcx, r10
    sub rcx, 64
    mov r8, 1
    shl r8, cl
    cmp qword ptr [rsp + 72], r8
    jb .terminal_b_mathematically_admitted
    jmp .admit_l

.current_capacity_64:
    cmp qword ptr [rsp + 72], 0
    je .terminal_b_mathematically_admitted
    jmp .admit_l

.current_capacity_low:
    cmp qword ptr [rsp + 72], 0
    jne .admit_l
    mov rcx, r10
    mov r8, 1
    shl r8, cl
    cmp qword ptr [rsp + 64], r8
    jb .terminal_b_mathematically_admitted
    jmp .admit_l

.terminal_b_mathematically_admitted:
    # The crossing B is lawful. Refuse to misclassify it if the u64 pair cannot carry it.
    mov r11, r14
    add r11, r15
    jc .needs_wide_pair
    mov qword ptr [rsp + 48], r15
    mov qword ptr [rsp + 56], r11
    mov rax, r15
    mul r11
    mov qword ptr [rsp + 80], rax
    mov qword ptr [rsp + 88], rdx

    mov dword ptr [rsp + 0], 1      # B
    mov dword ptr [rsp + 4], 2      # final crossing
    mov dword ptr [rsp + 8], 1
    jmp .commit

.admit_b_nonterminal:
    mov dword ptr [rsp + 0], 1      # B
    mov dword ptr [rsp + 4], 1      # next pair within capacity
    mov dword ptr [rsp + 8], 1
    jmp .commit

.admit_q:
    mov dword ptr [rsp + 0], 2      # Q
    mov dword ptr [rsp + 4], 3      # B blocked, position available
    jmp .commit

.admit_l:
    mov dword ptr [rsp + 0], 3      # L
    mov dword ptr [rsp + 4], 4      # B blocked, domain saturated
    jmp .commit

.commit:
    mov rax, qword ptr [rsp + 0]
    mov qword ptr [r13 + 0], rax
    mov rax, qword ptr [rsp + 8]
    mov qword ptr [r13 + 8], rax
    mov rax, qword ptr [rsp + 16]
    mov qword ptr [r13 + 16], rax
    mov rax, qword ptr [rsp + 24]
    mov qword ptr [r13 + 24], rax
    mov rax, qword ptr [rsp + 32]
    mov qword ptr [r13 + 32], rax
    mov rax, qword ptr [rsp + 40]
    mov qword ptr [r13 + 40], rax
    mov rax, qword ptr [rsp + 48]
    mov qword ptr [r13 + 48], rax
    mov rax, qword ptr [rsp + 56]
    mov qword ptr [r13 + 56], rax
    mov rax, qword ptr [rsp + 64]
    mov qword ptr [r13 + 64], rax
    mov rax, qword ptr [rsp + 72]
    mov qword ptr [r13 + 72], rax
    mov rax, qword ptr [rsp + 80]
    mov qword ptr [r13 + 80], rax
    mov rax, qword ptr [rsp + 88]
    mov qword ptr [r13 + 88], rax
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

.restore:
    add rsp, 96
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    ret

.null:
    mov eax, 1
    ret
.size qbl_inspect_u64, .-qbl_inspect_u64

.section .note.GNU-stack,"",@progbits
