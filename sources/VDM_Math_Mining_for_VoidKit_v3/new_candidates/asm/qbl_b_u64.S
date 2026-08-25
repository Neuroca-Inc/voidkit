.intel_syntax noprefix

.section .text

.global qbl_kernel_abi_version
.type qbl_kernel_abi_version, @function
qbl_kernel_abi_version:
    mov eax, 0x00010004
    ret
.size qbl_kernel_abi_version, .-qbl_kernel_abi_version

.global qbl_kernel_pair_size
.type qbl_kernel_pair_size, @function
qbl_kernel_pair_size:
    mov eax, 16
    ret
.size qbl_kernel_pair_size, .-qbl_kernel_pair_size

.global qbl_kernel_pair_align
.type qbl_kernel_pair_align, @function
qbl_kernel_pair_align:
    mov eax, 8
    ret
.size qbl_kernel_pair_align, .-qbl_kernel_pair_align

.global qbl_kernel_pair_u_offset
.type qbl_kernel_pair_u_offset, @function
qbl_kernel_pair_u_offset:
    xor eax, eax
    ret
.size qbl_kernel_pair_u_offset, .-qbl_kernel_pair_u_offset

.global qbl_kernel_pair_v_offset
.type qbl_kernel_pair_v_offset, @function
qbl_kernel_pair_v_offset:
    mov eax, 8
    ret
.size qbl_kernel_pair_v_offset, .-qbl_kernel_pair_v_offset

.global qbl_b_u64
.type qbl_b_u64, @function
# System V AMD64 ABI
#
# Input:
#   rdi -> qbl_pair_u64 { u, v }
#
# Success:
#   (u, v) becomes (v, u + v)
#   eax = QBL_STATUS_OK
#
# Failure:
#   state is unchanged
#   eax is one of QBL_STATUS_NULL, QBL_STATUS_INVALID_PAIR,
#   QBL_STATUS_OVERFLOW
qbl_b_u64:
    test rdi, rdi
    jz .null

    mov rax, qword ptr [rdi + 0]
    mov rdx, qword ptr [rdi + 8]

    test rax, rax
    jz .invalid
    cmp rax, rdx
    ja .invalid

    mov rcx, rax
    add rcx, rdx
    jc .overflow

    # Commit only after every precondition and capacity check succeeds.
    mov qword ptr [rdi + 0], rdx
    mov qword ptr [rdi + 8], rcx

    xor eax, eax
    ret

.null:
    mov eax, 1
    ret

.invalid:
    mov eax, 2
    ret

.overflow:
    mov eax, 3
    ret
.size qbl_b_u64, .-qbl_b_u64

.section .note.GNU-stack,"",@progbits
