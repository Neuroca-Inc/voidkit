	.file	"qbl_step_wide_v0_kernel.c"
	.intel_syntax noprefix
	.text
	.p2align 4
	.type	mul_pair, @function
mul_pair:
	pxor	xmm0, xmm0
	push	rbp
	mov	r11, rdi
	lea	r10, 32[rdx]
	push	rbx
	mov	ebp, 4
	mov	rbx, rsi
	movups	XMMWORD PTR [rdx], xmm0
	movups	XMMWORD PTR 16[rdx], xmm0
	movups	XMMWORD PTR 32[rdx], xmm0
	movups	XMMWORD PTR 48[rdx], xmm0
.L2:
	mov	rax, QWORD PTR [r11]
	mul	QWORD PTR [rbx]
	mov	rcx, r10
	add	rax, QWORD PTR -32[r10]
	mov	QWORD PTR -32[r10], rax
	adc	rdx, 0
	mov	rax, rdx
	xor	edx, edx
	mov	rsi, rax
	mov	rdi, rdx
	mov	rax, QWORD PTR [r11]
	mul	QWORD PTR 8[rbx]
	add	rax, QWORD PTR -24[r10]
	adc	rdx, 0
	add	rsi, rax
	mov	QWORD PTR -24[r10], rsi
	adc	rdi, rdx
	mov	rax, QWORD PTR [r11]
	mul	QWORD PTR 16[rbx]
	mov	rsi, rdi
	xor	edi, edi
	add	rax, QWORD PTR -16[r10]
	adc	rdx, 0
	add	rsi, rax
	adc	rdi, rdx
	mov	QWORD PTR -16[r10], rsi
	mov	rax, QWORD PTR [r11]
	mul	QWORD PTR 24[rbx]
	mov	rsi, rdi
	xor	edi, edi
	add	rax, QWORD PTR -8[r10]
	adc	rdx, 0
	add	rax, rsi
	mov	esi, ebp
	adc	rdx, rdi
	mov	QWORD PTR -8[r10], rax
	mov	rax, rdx
	xor	edx, edx
	mov	rdi, rax
	or	rdi, rdx
	je	.L6
	.p2align 6
	.p2align 4
	.p2align 3
.L3:
	add	rax, QWORD PTR [rcx]
	adc	rdx, 0
	mov	QWORD PTR [rcx], rax
	add	esi, 1
	add	rcx, 8
	mov	r9, rdx
	xor	edx, edx
	mov	eax, 1
	mov	rdi, r9
	and	edi, 1
	je	.L6
	cmp	esi, 8
	jne	.L3
.L6:
	add	ebp, 1
	add	r11, 8
	add	r10, 8
	cmp	ebp, 8
	jne	.L2
	pop	rbx
	pop	rbp
	ret
	.size	mul_pair, .-mul_pair
	.p2align 4
	.globl	qbl_step_wide_v0
	.type	qbl_step_wide_v0, @function
qbl_step_wide_v0:
	push	r15
	push	r14
	push	r13
	push	r12
	push	rbp
	push	rbx
	mov	rbx, rdx
	sub	rsp, 536
	test	rdx, rdx
	sete	al
	test	rcx, rcx
	sete	dl
	or	al, dl
	jne	.L98
	mov	r8, rdi
	test	rdi, rdi
	je	.L98
	lea	edx, -1[rsi]
	mov	r12d, esi
	mov	eax, 4
	cmp	edx, 3
	ja	.L15
	mov	rdx, QWORD PTR 56[rdi]
	mov	rbp, rcx
	test	rdx, rdx
	jne	.L17
	cmp	QWORD PTR 48[rdi], 0
	jne	.L100
	cmp	QWORD PTR 40[rdi], 0
	jne	.L101
	cmp	QWORD PTR 32[rdi], 0
	jne	.L102
	mov	rax, QWORD PTR 24[rdi]
	lea	r15, 24[rdi]
	test	rax, rax
	jne	.L20
	cmp	QWORD PTR 16[rdi], 0
	jne	.L24
	cmp	QWORD PTR 8[rdi], 0
	jne	.L24
	xor	r13d, r13d
	cmp	QWORD PTR [rdi], 0
	jne	.L24
.L27:
	lea	eax, -1[r13]
	cmp	eax, r12d
	jnb	.L173
.L246:
	test	rdx, rdx
	jne	.L34
.L32:
	mov	rax, QWORD PTR 16[r8]
	mov	rdx, QWORD PTR 48[r8]
	cmp	rax, rdx
	jb	.L34
	cmp	rdx, rax
	jb	.L117
	mov	rax, QWORD PTR 8[r8]
	mov	rdx, QWORD PTR 40[r8]
	cmp	rax, rdx
	jb	.L34
	cmp	rdx, rax
	jb	.L117
	mov	rax, QWORD PTR [r8]
	mov	rdx, QWORD PTR 32[r8]
	cmp	rax, rdx
	jb	.L34
	cmp	rdx, rax
	jb	.L117
.L34:
	mov	rcx, QWORD PTR 64[r8]
	cmp	rcx, 59
	ja	.L73
	mov	eax, 6
	mov	rdi, QWORD PTR 72[r8]
	sal	rax, cl
	mov	r14, rax
	mov	QWORD PTR 24[rsp], rdi
	mov	eax, 4
	cmp	rdi, r14
	jnb	.L15
	lea	rax, 32[r8]
	mov	QWORD PTR 16[rsp], rax
	lea	rax, -5[rdi+r14]
	mov	QWORD PTR 8[rsp], rax
	cmp	rax, 1
	je	.L247
	lea	rax, -1[r14]
	cmp	QWORD PTR 8[rsp], 2
	mov	QWORD PTR 32[rsp], rax
	je	.L248
	mov	rax, QWORD PTR 8[rsp]
	mov	rsi, QWORD PTR 16[rsp]
	lea	rdx, 96[rsp]
	mov	rdi, r8
	lea	r11, [rax+rax]
	mov	QWORD PTR 48[rsp], r11
	call	mul_pair
	mov	r11, QWORD PTR 48[rsp]
	mov	DWORD PTR 44[rsp], -1
	cmp	r11, 511
	ja	.L40
	mov	rax, r11
	mov	ecx, r11d
	shr	rax, 6
	and	ecx, 63
	mov	esi, eax
	lea	edx, 1[rax]
	cmp	eax, 7
	je	.L122
	cmp	QWORD PTR 152[rsp], 0
	jne	.L37
	cmp	edx, 7
	je	.L123
	cmp	QWORD PTR 144[rsp], 0
	jne	.L37
	cmp	edx, 6
	je	.L41
	cmp	QWORD PTR 136[rsp], 0
	jne	.L37
	cmp	edx, 5
	je	.L41
.L92:
	cmp	QWORD PTR 128[rsp], 0
	jne	.L37
	cmp	edx, 4
	je	.L41
	cmp	QWORD PTR 120[rsp], 0
	jne	.L37
	cmp	edx, 3
	je	.L41
	cmp	QWORD PTR 112[rsp], 0
	jne	.L37
	cmp	edx, 1
	jne	.L41
	cmp	QWORD PTR 104[rsp], 0
	jne	.L37
.L41:
	mov	edx, 1
	mov	rax, QWORD PTR 96[rsp+rax*8]
	mov	DWORD PTR 44[rsp], -1
	sal	rdx, cl
	cmp	rax, rdx
	jb	.L40
	cmp	rdx, rax
	jb	.L37
	test	esi, esi
	je	.L131
	cmp	QWORD PTR 96[rsp], 0
	jne	.L37
	cmp	esi, 1
	je	.L131
	cmp	QWORD PTR 104[rsp], 0
	jne	.L37
	cmp	esi, 2
	je	.L131
	cmp	QWORD PTR 112[rsp], 0
	jne	.L37
	cmp	esi, 3
	je	.L131
	cmp	QWORD PTR 120[rsp], 0
	jne	.L37
	cmp	esi, 4
	je	.L131
	cmp	QWORD PTR 128[rsp], 0
	jne	.L37
	cmp	esi, 5
	je	.L131
	cmp	QWORD PTR 136[rsp], 0
	jne	.L37
	cmp	esi, 7
	jne	.L131
	cmp	QWORD PTR 144[rsp], 0
	jne	.L37
.L131:
	mov	DWORD PTR 44[rsp], 0
	.p2align 4
	.p2align 3
.L40:
	mov	rcx, QWORD PTR [r8]
	mov	rsi, QWORD PTR 32[r8]
	xor	edx, edx
	mov	rax, QWORD PTR 40[r8]
	add	rax, QWORD PTR 8[r8]
	adc	rdx, 0
	lea	r9, [rcx+rsi]
	xor	edi, edi
	add	rsi, rcx
	adc	rdi, 0
	mov	QWORD PTR 64[rsp], r9
	mov	rsi, rdi
	xor	edi, edi
	add	rsi, rax
	mov	rax, QWORD PTR 48[r8]
	adc	rdi, rdx
	mov	QWORD PTR 72[rsp], rsi
	xor	edx, edx
	mov	r10, rsi
	mov	rsi, rdi
	xor	edi, edi
	add	rax, QWORD PTR 16[r8]
	adc	rdx, 0
	add	rax, rsi
	mov	rsi, QWORD PTR 56[r8]
	adc	rdx, rdi
	mov	QWORD PTR 80[rsp], rax
	xor	edi, edi
	mov	rcx, rax
	mov	rax, rdx
	xor	edx, edx
	add	rsi, QWORD PTR [r15]
	adc	rdi, 0
	add	rsi, rax
	adc	rdi, rdx
	mov	QWORD PTR 88[rsp], rsi
	mov	rdx, rsi
	mov	eax, edi
	test	edi, edi
	jne	.L42
	test	rdx, rdx
	jne	.L132
	test	rcx, rcx
	jne	.L133
	test	r10, r10
	jne	.L134
	test	r9, r9
	mov	edx, 1
	cmovne	eax, edx
.L44:
	mov	rdi, QWORD PTR 16[rsp]
	lea	rsi, 64[rsp]
	mov	QWORD PTR 56[rsp], r11
	lea	r15, 160[rsp]
	mov	rdx, r15
	mov	DWORD PTR 48[rsp], eax
	call	mul_pair
	mov	r11, QWORD PTR 56[rsp]
	mov	eax, DWORD PTR 48[rsp]
	cmp	r11, 511
	ja	.L240
	mov	rsi, r11
	shr	rsi, 6
	lea	edx, 1[rsi]
	cmp	esi, 7
	je	.L49
	cmp	QWORD PTR 216[rsp], 0
	jne	.L58
	cmp	edx, 7
	je	.L49
	cmp	QWORD PTR 208[rsp], 0
	jne	.L58
	cmp	edx, 6
	je	.L49
	cmp	QWORD PTR 200[rsp], 0
	jne	.L58
	cmp	edx, 5
	je	.L49
	cmp	QWORD PTR 192[rsp], 0
	jne	.L58
	cmp	edx, 4
	je	.L49
	cmp	QWORD PTR 184[rsp], 0
	jne	.L58
	cmp	edx, 3
	je	.L49
	cmp	QWORD PTR 176[rsp], 0
	jne	.L58
	cmp	edx, 1
	jne	.L49
	cmp	QWORD PTR 168[rsp], 0
	jne	.L58
.L49:
	mov	ecx, r11d
	mov	edx, 1
	sal	rdx, cl
	mov	rcx, QWORD PTR 160[rsp+rsi*8]
	cmp	rcx, rdx
	jb	.L240
	cmp	rdx, rcx
	jb	.L58
	test	rsi, rsi
	je	.L220
	lea	edx, -1[rsi]
	lea	rdx, 168[rsp+rdx*8]
	jmp	.L64
	.p2align 4
	.p2align 4,,10
	.p2align 3
.L249:
	add	r15, 8
	cmp	r15, rdx
	je	.L220
.L64:
	cmp	QWORD PTR [r15], 0
	je	.L249
	mov	rdi, QWORD PTR 32[rsp]
	cmp	QWORD PTR 24[rsp], rdi
	jne	.L67
.L66:
	cmp	DWORD PTR 44[rsp], -1
	je	.L250
.L46:
	mov	ecx, 4
	mov	edx, 3
	jmp	.L62
	.p2align 4,,10
	.p2align 3
.L98:
	mov	eax, 1
.L15:
	add	rsp, 536
	pop	rbx
	pop	rbp
	pop	r12
	pop	r13
	pop	r14
	pop	r15
	ret
.L17:
	mov	rax, QWORD PTR 24[rdi]
	lea	r15, 24[rdi]
	test	rax, rax
	jne	.L20
	cmp	QWORD PTR 16[r8], 0
	mov	r13d, 4
	jne	.L27
.L21:
	cmp	QWORD PTR 8[r8], 0
	je	.L27
	cmp	r13d, 1
	jne	.L27
.L24:
	cmp	QWORD PTR 16[r8], 0
	jne	.L251
	cmp	QWORD PTR 8[r8], 0
	jne	.L29
	cmp	QWORD PTR [r8], 0
	mov	eax, 4
	je	.L15
	mov	r13d, 1
	jmp	.L246
	.p2align 4,,10
	.p2align 3
.L42:
	mov	rcx, QWORD PTR 32[rsp]
	mov	eax, 17
	cmp	QWORD PTR 24[rsp], rcx
	je	.L252
	cmp	r11, 256
	ja	.L15
.L67:
	mov	ecx, 3
	mov	edx, 2
.L62:
	movdqu	xmm0, XMMWORD PTR [r8]
	mov	rax, QWORD PTR 80[r8]
	cmp	QWORD PTR 152[rsp], 0
	movaps	XMMWORD PTR 224[rsp], xmm0
	movdqu	xmm0, XMMWORD PTR 16[r8]
	mov	QWORD PTR 304[rsp], rax
	movaps	XMMWORD PTR 240[rsp], xmm0
	movdqu	xmm0, XMMWORD PTR 32[r8]
	movaps	XMMWORD PTR 256[rsp], xmm0
	movdqu	xmm0, XMMWORD PTR 48[r8]
	movaps	XMMWORD PTR 272[rsp], xmm0
	movdqu	xmm0, XMMWORD PTR 64[r8]
	movaps	XMMWORD PTR 288[rsp], xmm0
	pxor	xmm0, xmm0
	movaps	XMMWORD PTR 320[rsp], xmm0
	movaps	XMMWORD PTR 336[rsp], xmm0
	mov	DWORD PTR 320[rsp], edx
	mov	DWORD PTR 324[rsp], ecx
	mov	DWORD PTR 328[rsp], 1
	mov	DWORD PTR 336[rsp], r13d
	movaps	XMMWORD PTR 368[rsp], xmm0
	movaps	XMMWORD PTR 384[rsp], xmm0
	movaps	XMMWORD PTR 464[rsp], xmm0
	movaps	XMMWORD PTR 480[rsp], xmm0
	movaps	XMMWORD PTR 496[rsp], xmm0
	movaps	XMMWORD PTR 512[rsp], xmm0
	jne	.L144
	cmp	QWORD PTR 144[rsp], 0
	jne	.L145
	cmp	QWORD PTR 136[rsp], 0
	jne	.L146
	cmp	QWORD PTR 128[rsp], 0
	jne	.L147
	cmp	QWORD PTR 120[rsp], 0
	jne	.L148
	cmp	QWORD PTR 112[rsp], 0
	jne	.L149
	cmp	QWORD PTR 104[rsp], 0
	jne	.L150
	xor	eax, eax
	cmp	QWORD PTR 96[rsp], 0
	setne	al
.L69:
	movdqa	xmm0, XMMWORD PTR 96[rsp]
	mov	DWORD PTR 344[rsp], eax
	mov	rax, QWORD PTR 8[rsp]
	mov	QWORD PTR 352[rsp], r14
	movaps	XMMWORD PTR 400[rsp], xmm0
	movdqa	xmm0, XMMWORD PTR 112[rsp]
	mov	QWORD PTR 360[rsp], rax
	movaps	XMMWORD PTR 416[rsp], xmm0
	movdqa	xmm0, XMMWORD PTR 128[rsp]
	mov	QWORD PTR 368[rsp], r11
	movaps	XMMWORD PTR 432[rsp], xmm0
	movdqa	xmm0, XMMWORD PTR 144[rsp]
	movaps	XMMWORD PTR 448[rsp], xmm0
	cmp	edx, 1
	je	.L253
	cmp	edx, 2
	je	.L254
	mov	rax, QWORD PTR 288[rsp]
	cmp	rax, 59
	je	.L73
	lea	rcx, 1[rax]
	mov	r14d, 6
	mov	QWORD PTR 296[rsp], 0
	sal	r14, cl
	mov	QWORD PTR 288[rsp], rcx
	lea	r12, -5[r14]
	cmp	r14, 6
	je	.L255
.L75:
	lea	r13, [r12+r12]
.L74:
	lea	rdx, 464[rsp]
	lea	rdi, 224[rsp]
	lea	rsi, 256[rsp]
	call	mul_pair
	cmp	QWORD PTR 280[rsp], 0
	jne	.L256
	cmp	QWORD PTR 272[rsp], 0
	jne	.L257
	cmp	QWORD PTR 264[rsp], 0
	jne	.L258
	cmp	QWORD PTR 256[rsp], 0
	jne	.L259
	cmp	QWORD PTR 248[rsp], 0
	mov	DWORD PTR 340[rsp], 0
	jne	.L80
	cmp	QWORD PTR 240[rsp], 0
	jne	.L82
	xor	eax, eax
	cmp	QWORD PTR 232[rsp], 0
	jne	.L85
.L84:
	cmp	QWORD PTR 224[rsp], 0
	mov	edx, 1
	jne	.L86
.L77:
	cmp	QWORD PTR 520[rsp], 0
	jne	.L164
	cmp	QWORD PTR 512[rsp], 0
	jne	.L165
	cmp	QWORD PTR 504[rsp], 0
	jne	.L166
	cmp	QWORD PTR 496[rsp], 0
	jne	.L167
	cmp	QWORD PTR 488[rsp], 0
	jne	.L168
	cmp	QWORD PTR 480[rsp], 0
	jne	.L169
	cmp	QWORD PTR 472[rsp], 0
	jne	.L170
	xor	eax, eax
	cmp	QWORD PTR 464[rsp], 0
	setne	al
.L90:
	movdqa	xmm0, XMMWORD PTR 224[rsp]
	mov	DWORD PTR 348[rsp], eax
	mov	rax, QWORD PTR 304[rsp]
	mov	QWORD PTR 376[rsp], r14
	movups	XMMWORD PTR [r8], xmm0
	movdqa	xmm0, XMMWORD PTR 240[rsp]
	mov	QWORD PTR 80[r8], rax
	xor	eax, eax
	movups	XMMWORD PTR 16[r8], xmm0
	movdqa	xmm0, XMMWORD PTR 256[rsp]
	mov	QWORD PTR 384[rsp], r12
	movups	XMMWORD PTR 32[r8], xmm0
	movdqa	xmm0, XMMWORD PTR 272[rsp]
	mov	QWORD PTR 392[rsp], r13
	movups	XMMWORD PTR 48[r8], xmm0
	movdqa	xmm0, XMMWORD PTR 288[rsp]
	movups	XMMWORD PTR 64[r8], xmm0
	movdqa	xmm0, XMMWORD PTR 320[rsp]
	movups	XMMWORD PTR [rbx], xmm0
	movdqa	xmm0, XMMWORD PTR 336[rsp]
	movups	XMMWORD PTR 16[rbx], xmm0
	movdqa	xmm0, XMMWORD PTR 352[rsp]
	movups	XMMWORD PTR 32[rbx], xmm0
	movdqa	xmm0, XMMWORD PTR 368[rsp]
	movups	XMMWORD PTR 48[rbx], xmm0
	movdqa	xmm0, XMMWORD PTR 384[rsp]
	movups	XMMWORD PTR 64[rbx], xmm0
	movdqa	xmm0, XMMWORD PTR 400[rsp]
	movups	XMMWORD PTR 80[rbx], xmm0
	movdqa	xmm0, XMMWORD PTR 416[rsp]
	movups	XMMWORD PTR 96[rbx], xmm0
	movdqa	xmm0, XMMWORD PTR 432[rsp]
	movups	XMMWORD PTR 112[rbx], xmm0
	movdqa	xmm0, XMMWORD PTR 448[rsp]
	movups	XMMWORD PTR 128[rbx], xmm0
	movdqa	xmm0, XMMWORD PTR 464[rsp]
	movups	XMMWORD PTR 144[rbx], xmm0
	movdqa	xmm0, XMMWORD PTR 480[rsp]
	movups	XMMWORD PTR 160[rbx], xmm0
	movdqa	xmm0, XMMWORD PTR 496[rsp]
	movups	XMMWORD PTR 176[rbx], xmm0
	movdqa	xmm0, XMMWORD PTR 512[rsp]
	movups	XMMWORD PTR 192[rbx], xmm0
	pxor	xmm0, xmm0
	movups	XMMWORD PTR 0[rbp], xmm0
	jmp	.L15
	.p2align 4,,10
	.p2align 3
.L73:
	mov	eax, 5
	jmp	.L15
.L123:
	mov	esi, 6
	mov	eax, 6
	jmp	.L41
.L256:
	cmp	QWORD PTR 248[rsp], 0
	mov	DWORD PTR 340[rsp], 4
	jne	.L77
	cmp	QWORD PTR 240[rsp], 0
	mov	eax, 4
	jne	.L77
.L78:
	cmp	QWORD PTR 232[rsp], 0
	je	.L84
	mov	edx, 2
.L86:
	cmp	eax, edx
	jnb	.L77
.L85:
	cmp	QWORD PTR 232[rsp], 0
	jne	.L162
	xor	eax, eax
	cmp	QWORD PTR 224[rsp], 0
	setne	al
.L88:
	mov	DWORD PTR 340[rsp], eax
	jmp	.L77
	.p2align 4,,10
	.p2align 3
.L253:
	movdqu	xmm0, XMMWORD PTR 32[r8]
	mov	rax, QWORD PTR 296[rsp]
	mov	rcx, QWORD PTR 288[rsp]
	movaps	XMMWORD PTR 224[rsp], xmm0
	movdqu	xmm0, XMMWORD PTR 48[r8]
	lea	r12, -5[rax]
	movaps	XMMWORD PTR 240[rsp], xmm0
	movdqa	xmm0, XMMWORD PTR 64[rsp]
	movaps	XMMWORD PTR 256[rsp], xmm0
	movdqa	xmm0, XMMWORD PTR 80[rsp]
	movaps	XMMWORD PTR 272[rsp], xmm0
.L71:
	mov	r14d, 6
	mov	r13d, 1
	sal	r14, cl
	add	r12, r14
	cmp	r12, 1
	je	.L74
	mov	r13d, 2
	cmp	r12, 2
	jne	.L75
	jmp	.L74
	.p2align 4,,10
	.p2align 3
.L240:
	mov	rsi, QWORD PTR 32[rsp]
	mov	ecx, 1
	cmp	QWORD PTR 24[rsp], rsi
	je	.L66
.L48:
	mov	edx, 1
	cmp	r12d, eax
	jnb	.L62
	mov	DWORD PTR 0[rbp], eax
	mov	eax, 16
	mov	DWORD PTR 4[rbp], r12d
	mov	DWORD PTR 8[rbp], 1
	mov	DWORD PTR 12[rbp], ecx
	jmp	.L15
.L247:
	mov	rsi, QWORD PTR 16[rsp]
	lea	rdx, 96[rsp]
	mov	rdi, r8
	call	mul_pair
	mov	r11d, 1
	cmp	QWORD PTR 152[rsp], 0
	jne	.L119
	mov	QWORD PTR 32[rsp], 5
	mov	ecx, 1
.L38:
	cmp	QWORD PTR 144[rsp], 0
	jne	.L37
	mov	rax, QWORD PTR 136[rsp]
	test	rax, rax
	jne	.L37
	mov	edx, 1
	xor	esi, esi
	jmp	.L92
.L117:
	mov	eax, 2
	jmp	.L15
.L254:
	mov	rax, QWORD PTR 304[rsp]
	cmp	rax, -1
	je	.L152
	mov	r12, QWORD PTR 296[rsp]
	add	rax, 1
	mov	rcx, QWORD PTR 288[rsp]
	mov	QWORD PTR 304[rsp], rax
	lea	rdx, 1[r12]
	sub	r12, 4
	mov	QWORD PTR 296[rsp], rdx
	jmp	.L71
.L252:
	cmp	DWORD PTR 44[rsp], -1
	jne	.L46
	jmp	.L15
.L119:
	mov	QWORD PTR 32[rsp], 5
	.p2align 4
	.p2align 3
.L37:
	mov	DWORD PTR 44[rsp], 1
	jmp	.L40
.L100:
	mov	r13d, 3
.L18:
	cmp	QWORD PTR 24[r8], 0
	lea	r15, 24[r8]
	jne	.L260
	cmp	QWORD PTR 16[r8], 0
	je	.L21
	cmp	r13d, 3
	je	.L27
	cmp	r12d, 2
	jbe	.L173
	mov	r13d, 3
	jmp	.L32
.L58:
	mov	rdi, QWORD PTR 32[rsp]
	cmp	QWORD PTR 24[rsp], rdi
	jne	.L67
	jmp	.L66
.L101:
	mov	r13d, 2
	jmp	.L18
.L102:
	mov	r13d, 1
	jmp	.L18
.L173:
	mov	eax, 4
	jmp	.L15
.L248:
	mov	rsi, QWORD PTR 16[rsp]
	lea	rdx, 96[rsp]
	mov	rdi, r8
	call	mul_pair
	mov	r11d, 2
	cmp	QWORD PTR 152[rsp], 0
	jne	.L37
	mov	ecx, 2
	jmp	.L38
.L255:
	mov	r13d, 1
	mov	r14d, 6
	jmp	.L74
.L29:
	cmp	r12d, 1
	je	.L173
	mov	r13d, 2
	jmp	.L246
.L250:
	mov	ecx, 2
	jmp	.L48
.L251:
	cmp	r12d, 2
	jbe	.L173
	mov	r13d, 3
	jmp	.L246
.L144:
	mov	eax, 8
	jmp	.L69
.L145:
	mov	eax, 7
	jmp	.L69
.L220:
	mov	rdi, QWORD PTR 32[rsp]
	cmp	QWORD PTR 24[rsp], rdi
	je	.L66
	mov	ecx, 1
	jmp	.L48
.L164:
	mov	eax, 8
	jmp	.L90
.L146:
	mov	eax, 6
	jmp	.L69
.L165:
	mov	eax, 7
	jmp	.L90
.L147:
	mov	eax, 5
	jmp	.L69
.L257:
	cmp	QWORD PTR 248[rsp], 0
	mov	DWORD PTR 340[rsp], 3
	jne	.L80
	cmp	QWORD PTR 240[rsp], 0
	mov	eax, 3
	je	.L78
	jmp	.L77
	.p2align 4,,10
	.p2align 3
.L20:
	cmp	r12d, 4
	jne	.L173
	mov	r13d, 4
	cmp	rax, rdx
	jb	.L34
	cmp	rdx, rax
	jb	.L117
	mov	r13d, 4
	jmp	.L32
.L166:
	mov	eax, 6
	jmp	.L90
.L148:
	mov	eax, 4
	jmp	.L69
.L258:
	cmp	QWORD PTR 248[rsp], 0
	mov	DWORD PTR 340[rsp], 2
	jne	.L80
	cmp	QWORD PTR 240[rsp], 0
	mov	eax, 2
	je	.L78
.L82:
	mov	eax, 3
	jmp	.L88
.L259:
	cmp	QWORD PTR 248[rsp], 0
	mov	DWORD PTR 340[rsp], 1
	jne	.L80
	cmp	QWORD PTR 240[rsp], 0
	mov	eax, 1
	je	.L78
	jmp	.L82
	.p2align 4,,10
	.p2align 3
.L168:
	mov	eax, 4
	jmp	.L90
.L167:
	mov	eax, 5
	jmp	.L90
.L149:
	mov	eax, 3
	jmp	.L69
.L169:
	mov	eax, 3
	jmp	.L90
.L150:
	mov	eax, 2
	jmp	.L69
.L133:
	mov	eax, 3
	jmp	.L44
.L132:
	mov	eax, 4
	jmp	.L44
.L170:
	mov	eax, 2
	jmp	.L90
.L134:
	mov	eax, 2
	jmp	.L44
.L152:
	mov	eax, 3
	jmp	.L15
.L260:
	xor	eax, eax
	cmp	r12d, 4
	setne	al
	lea	eax, 2[rax+rax]
	jmp	.L15
.L122:
	mov	esi, 7
	mov	eax, 7
	jmp	.L41
.L162:
	mov	eax, 2
	jmp	.L88
.L80:
	mov	eax, 4
	jmp	.L88
	.size	qbl_step_wide_v0, .-qbl_step_wide_v0
	.section	.note.GNU-stack,"",@progbits
