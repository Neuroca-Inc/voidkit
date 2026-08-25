.intel_syntax noprefix
.text
.global balanced_trace_to_floor
.type balanced_trace_to_floor, @function
# int balanced_trace_to_floor(uint64_t u0, uint64_t v0, uint64_t floor_den, uint64_t *out_rows, uint64_t max_rows)
# row layout (5 qwords / 40 bytes): step,u,v,uv,hit_floor
# returns RAX = rows_written
balanced_trace_to_floor:
    push rbx
    push r12
    push r13
    push r14
    push r15

    mov r10, rdi          # u
    mov r11, rsi          # v
    mov r12, rdx          # floor_den
    mov r13, rcx          # out_rows
    mov r14, r8           # max_rows
    xor r15, r15          # step
    xor rax, rax          # rows_written

.loop:
    cmp rax, r14
    jae .done

    mov rbx, r10
    imul rbx, r11         # uv = u*v

    xor r9d, r9d
    cmp rbx, r12
    setae r9b             # hit = (uv >= floor_den)

    lea rdx, [rax + rax*4]
    shl rdx, 3            # byte offset = rows_written * 40
    lea rcx, [r13 + rdx]
    mov [rcx + 0], r15
    mov [rcx + 8], r10
    mov [rcx + 16], r11
    mov [rcx + 24], rbx
    mov [rcx + 32], r9

    inc rax
    test r9, r9
    jnz .done

    # balanced_step(u,v) = (b, a+b), with a<=b
    mov rcx, r10
    mov rdx, r11
    cmp rcx, rdx
    jbe .ordered
    xchg rcx, rdx
.ordered:
    lea r8, [rcx + rdx]
    mov r10, rdx
    mov r11, r8
    inc r15
    jmp .loop

.done:
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    ret
