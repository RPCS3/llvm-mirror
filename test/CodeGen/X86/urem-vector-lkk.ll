; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py
; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mattr=+sse4.1 | FileCheck %s --check-prefix=CHECK --check-prefix=SSE
; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mattr=+avx  | FileCheck %s --check-prefix=CHECK --check-prefix=AVX --check-prefix=AVX1
; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mattr=+avx2 | FileCheck %s --check-prefix=CHECK --check-prefix=AVX --check-prefix=AVX2

define <4 x i16> @fold_urem_vec_1(<4 x i16> %x) {
; SSE-LABEL: fold_urem_vec_1:
; SSE:       # %bb.0:
; SSE-NEXT:    pextrw $1, %xmm0, %eax
; SSE-NEXT:    movl %eax, %ecx
; SSE-NEXT:    shrl $2, %ecx
; SSE-NEXT:    imull $16913, %ecx, %ecx # imm = 0x4211
; SSE-NEXT:    shrl $19, %ecx
; SSE-NEXT:    imull $124, %ecx, %ecx
; SSE-NEXT:    subl %ecx, %eax
; SSE-NEXT:    movd %xmm0, %ecx
; SSE-NEXT:    movzwl %cx, %edx
; SSE-NEXT:    imull $44151, %edx, %edx # imm = 0xAC77
; SSE-NEXT:    shrl $22, %edx
; SSE-NEXT:    imull $95, %edx, %edx
; SSE-NEXT:    subl %edx, %ecx
; SSE-NEXT:    movd %ecx, %xmm1
; SSE-NEXT:    pinsrw $1, %eax, %xmm1
; SSE-NEXT:    pextrw $2, %xmm0, %eax
; SSE-NEXT:    movl %eax, %ecx
; SSE-NEXT:    shrl %ecx
; SSE-NEXT:    imull $2675, %ecx, %ecx # imm = 0xA73
; SSE-NEXT:    shrl $17, %ecx
; SSE-NEXT:    imull $98, %ecx, %ecx
; SSE-NEXT:    subl %ecx, %eax
; SSE-NEXT:    pinsrw $2, %eax, %xmm1
; SSE-NEXT:    pextrw $3, %xmm0, %eax
; SSE-NEXT:    imull $1373, %eax, %ecx # imm = 0x55D
; SSE-NEXT:    shrl $16, %ecx
; SSE-NEXT:    movl %eax, %edx
; SSE-NEXT:    subl %ecx, %edx
; SSE-NEXT:    movzwl %dx, %edx
; SSE-NEXT:    shrl %edx
; SSE-NEXT:    addl %ecx, %edx
; SSE-NEXT:    shrl $9, %edx
; SSE-NEXT:    imull $1003, %edx, %ecx # imm = 0x3EB
; SSE-NEXT:    subl %ecx, %eax
; SSE-NEXT:    pinsrw $3, %eax, %xmm1
; SSE-NEXT:    movdqa %xmm1, %xmm0
; SSE-NEXT:    retq
;
; AVX-LABEL: fold_urem_vec_1:
; AVX:       # %bb.0:
; AVX-NEXT:    vpextrw $1, %xmm0, %eax
; AVX-NEXT:    movl %eax, %ecx
; AVX-NEXT:    shrl $2, %ecx
; AVX-NEXT:    imull $16913, %ecx, %ecx # imm = 0x4211
; AVX-NEXT:    shrl $19, %ecx
; AVX-NEXT:    imull $124, %ecx, %ecx
; AVX-NEXT:    subl %ecx, %eax
; AVX-NEXT:    vmovd %xmm0, %ecx
; AVX-NEXT:    movzwl %cx, %edx
; AVX-NEXT:    imull $44151, %edx, %edx # imm = 0xAC77
; AVX-NEXT:    shrl $22, %edx
; AVX-NEXT:    imull $95, %edx, %edx
; AVX-NEXT:    subl %edx, %ecx
; AVX-NEXT:    vmovd %ecx, %xmm1
; AVX-NEXT:    vpinsrw $1, %eax, %xmm1, %xmm1
; AVX-NEXT:    vpextrw $2, %xmm0, %eax
; AVX-NEXT:    movl %eax, %ecx
; AVX-NEXT:    shrl %ecx
; AVX-NEXT:    imull $2675, %ecx, %ecx # imm = 0xA73
; AVX-NEXT:    shrl $17, %ecx
; AVX-NEXT:    imull $98, %ecx, %ecx
; AVX-NEXT:    subl %ecx, %eax
; AVX-NEXT:    vpinsrw $2, %eax, %xmm1, %xmm1
; AVX-NEXT:    vpextrw $3, %xmm0, %eax
; AVX-NEXT:    imull $1373, %eax, %ecx # imm = 0x55D
; AVX-NEXT:    shrl $16, %ecx
; AVX-NEXT:    movl %eax, %edx
; AVX-NEXT:    subl %ecx, %edx
; AVX-NEXT:    movzwl %dx, %edx
; AVX-NEXT:    shrl %edx
; AVX-NEXT:    addl %ecx, %edx
; AVX-NEXT:    shrl $9, %edx
; AVX-NEXT:    imull $1003, %edx, %ecx # imm = 0x3EB
; AVX-NEXT:    subl %ecx, %eax
; AVX-NEXT:    vpinsrw $3, %eax, %xmm1, %xmm0
; AVX-NEXT:    retq
  %1 = urem <4 x i16> %x, <i16 95, i16 124, i16 98, i16 1003>
  ret <4 x i16> %1
}

define <4 x i16> @fold_urem_vec_2(<4 x i16> %x) {
; SSE-LABEL: fold_urem_vec_2:
; SSE:       # %bb.0:
; SSE-NEXT:    movdqa {{.*#+}} xmm1 = [44151,44151,44151,44151,44151,44151,44151,44151]
; SSE-NEXT:    pmulhuw %xmm0, %xmm1
; SSE-NEXT:    psrlw $6, %xmm1
; SSE-NEXT:    pmullw {{\.?LCPI[0-9]+_[0-9]+}}(%rip), %xmm1
; SSE-NEXT:    psubw %xmm1, %xmm0
; SSE-NEXT:    retq
;
; AVX-LABEL: fold_urem_vec_2:
; AVX:       # %bb.0:
; AVX-NEXT:    vpmulhuw {{\.?LCPI[0-9]+_[0-9]+}}(%rip), %xmm0, %xmm1
; AVX-NEXT:    vpsrlw $6, %xmm1, %xmm1
; AVX-NEXT:    vpmullw {{\.?LCPI[0-9]+_[0-9]+}}(%rip), %xmm1, %xmm1
; AVX-NEXT:    vpsubw %xmm1, %xmm0, %xmm0
; AVX-NEXT:    retq
  %1 = urem <4 x i16> %x, <i16 95, i16 95, i16 95, i16 95>
  ret <4 x i16> %1
}


; Don't fold if we can combine urem with udiv.
define <4 x i16> @combine_urem_udiv(<4 x i16> %x) {
; SSE-LABEL: combine_urem_udiv:
; SSE:       # %bb.0:
; SSE-NEXT:    movdqa {{.*#+}} xmm1 = [44151,44151,44151,44151,44151,44151,44151,44151]
; SSE-NEXT:    pmulhuw %xmm0, %xmm1
; SSE-NEXT:    psrlw $6, %xmm1
; SSE-NEXT:    movdqa {{.*#+}} xmm2 = [95,95,95,95,95,95,95,95]
; SSE-NEXT:    pmullw %xmm1, %xmm2
; SSE-NEXT:    psubw %xmm2, %xmm0
; SSE-NEXT:    paddw %xmm1, %xmm0
; SSE-NEXT:    retq
;
; AVX-LABEL: combine_urem_udiv:
; AVX:       # %bb.0:
; AVX-NEXT:    vpmulhuw {{\.?LCPI[0-9]+_[0-9]+}}(%rip), %xmm0, %xmm1
; AVX-NEXT:    vpsrlw $6, %xmm1, %xmm1
; AVX-NEXT:    vpmullw {{\.?LCPI[0-9]+_[0-9]+}}(%rip), %xmm1, %xmm2
; AVX-NEXT:    vpsubw %xmm2, %xmm0, %xmm0
; AVX-NEXT:    vpaddw %xmm1, %xmm0, %xmm0
; AVX-NEXT:    retq
  %1 = urem <4 x i16> %x, <i16 95, i16 95, i16 95, i16 95>
  %2 = udiv <4 x i16> %x, <i16 95, i16 95, i16 95, i16 95>
  %3 = add <4 x i16> %1, %2
  ret <4 x i16> %3
}

; Don't fold for divisors that are a power of two.
define <4 x i16> @dont_fold_urem_power_of_two(<4 x i16> %x) {
; SSE-LABEL: dont_fold_urem_power_of_two:
; SSE:       # %bb.0:
; SSE-NEXT:    pextrw $3, %xmm0, %eax
; SSE-NEXT:    imull $44151, %eax, %ecx # imm = 0xAC77
; SSE-NEXT:    shrl $22, %ecx
; SSE-NEXT:    imull $95, %ecx, %ecx
; SSE-NEXT:    subl %ecx, %eax
; SSE-NEXT:    pextrw $1, %xmm0, %ecx
; SSE-NEXT:    andl $31, %ecx
; SSE-NEXT:    movd %xmm0, %edx
; SSE-NEXT:    andl $63, %edx
; SSE-NEXT:    movd %edx, %xmm1
; SSE-NEXT:    pinsrw $1, %ecx, %xmm1
; SSE-NEXT:    pextrw $2, %xmm0, %ecx
; SSE-NEXT:    andl $7, %ecx
; SSE-NEXT:    pinsrw $2, %ecx, %xmm1
; SSE-NEXT:    pinsrw $3, %eax, %xmm1
; SSE-NEXT:    movdqa %xmm1, %xmm0
; SSE-NEXT:    retq
;
; AVX-LABEL: dont_fold_urem_power_of_two:
; AVX:       # %bb.0:
; AVX-NEXT:    vpextrw $3, %xmm0, %eax
; AVX-NEXT:    imull $44151, %eax, %ecx # imm = 0xAC77
; AVX-NEXT:    shrl $22, %ecx
; AVX-NEXT:    imull $95, %ecx, %ecx
; AVX-NEXT:    subl %ecx, %eax
; AVX-NEXT:    vpextrw $1, %xmm0, %ecx
; AVX-NEXT:    andl $31, %ecx
; AVX-NEXT:    vmovd %xmm0, %edx
; AVX-NEXT:    andl $63, %edx
; AVX-NEXT:    vmovd %edx, %xmm1
; AVX-NEXT:    vpinsrw $1, %ecx, %xmm1, %xmm1
; AVX-NEXT:    vpextrw $2, %xmm0, %ecx
; AVX-NEXT:    andl $7, %ecx
; AVX-NEXT:    vpinsrw $2, %ecx, %xmm1, %xmm0
; AVX-NEXT:    vpinsrw $3, %eax, %xmm0, %xmm0
; AVX-NEXT:    retq
  %1 = urem <4 x i16> %x, <i16 64, i16 32, i16 8, i16 95>
  ret <4 x i16> %1
}

; Don't fold if the divisor is one.
define <4 x i16> @dont_fold_urem_one(<4 x i16> %x) {
; SSE-LABEL: dont_fold_urem_one:
; SSE:       # %bb.0:
; SSE-NEXT:    pextrw $2, %xmm0, %eax
; SSE-NEXT:    imull $25645, %eax, %ecx # imm = 0x642D
; SSE-NEXT:    shrl $16, %ecx
; SSE-NEXT:    movl %eax, %edx
; SSE-NEXT:    subl %ecx, %edx
; SSE-NEXT:    movzwl %dx, %edx
; SSE-NEXT:    shrl %edx
; SSE-NEXT:    addl %ecx, %edx
; SSE-NEXT:    shrl $4, %edx
; SSE-NEXT:    leal (%rdx,%rdx,2), %ecx
; SSE-NEXT:    shll $3, %ecx
; SSE-NEXT:    subl %ecx, %edx
; SSE-NEXT:    addl %eax, %edx
; SSE-NEXT:    pextrw $1, %xmm0, %eax
; SSE-NEXT:    imull $51307, %eax, %ecx # imm = 0xC86B
; SSE-NEXT:    shrl $25, %ecx
; SSE-NEXT:    imull $654, %ecx, %ecx # imm = 0x28E
; SSE-NEXT:    subl %ecx, %eax
; SSE-NEXT:    pxor %xmm1, %xmm1
; SSE-NEXT:    pinsrw $1, %eax, %xmm1
; SSE-NEXT:    pinsrw $2, %edx, %xmm1
; SSE-NEXT:    pextrw $3, %xmm0, %eax
; SSE-NEXT:    imull $12375, %eax, %ecx # imm = 0x3057
; SSE-NEXT:    shrl $26, %ecx
; SSE-NEXT:    imull $5423, %ecx, %ecx # imm = 0x152F
; SSE-NEXT:    subl %ecx, %eax
; SSE-NEXT:    pinsrw $3, %eax, %xmm1
; SSE-NEXT:    movdqa %xmm1, %xmm0
; SSE-NEXT:    retq
;
; AVX-LABEL: dont_fold_urem_one:
; AVX:       # %bb.0:
; AVX-NEXT:    vpextrw $2, %xmm0, %eax
; AVX-NEXT:    imull $25645, %eax, %ecx # imm = 0x642D
; AVX-NEXT:    shrl $16, %ecx
; AVX-NEXT:    movl %eax, %edx
; AVX-NEXT:    subl %ecx, %edx
; AVX-NEXT:    movzwl %dx, %edx
; AVX-NEXT:    shrl %edx
; AVX-NEXT:    addl %ecx, %edx
; AVX-NEXT:    shrl $4, %edx
; AVX-NEXT:    leal (%rdx,%rdx,2), %ecx
; AVX-NEXT:    shll $3, %ecx
; AVX-NEXT:    subl %ecx, %edx
; AVX-NEXT:    addl %eax, %edx
; AVX-NEXT:    vpextrw $1, %xmm0, %eax
; AVX-NEXT:    imull $51307, %eax, %ecx # imm = 0xC86B
; AVX-NEXT:    shrl $25, %ecx
; AVX-NEXT:    imull $654, %ecx, %ecx # imm = 0x28E
; AVX-NEXT:    subl %ecx, %eax
; AVX-NEXT:    vpxor %xmm1, %xmm1, %xmm1
; AVX-NEXT:    vpinsrw $1, %eax, %xmm1, %xmm1
; AVX-NEXT:    vpinsrw $2, %edx, %xmm1, %xmm1
; AVX-NEXT:    vpextrw $3, %xmm0, %eax
; AVX-NEXT:    imull $12375, %eax, %ecx # imm = 0x3057
; AVX-NEXT:    shrl $26, %ecx
; AVX-NEXT:    imull $5423, %ecx, %ecx # imm = 0x152F
; AVX-NEXT:    subl %ecx, %eax
; AVX-NEXT:    vpinsrw $3, %eax, %xmm1, %xmm0
; AVX-NEXT:    retq
  %1 = urem <4 x i16> %x, <i16 1, i16 654, i16 23, i16 5423>
  ret <4 x i16> %1
}

; Don't fold if the divisor is 2^16.
define <4 x i16> @dont_fold_urem_i16_smax(<4 x i16> %x) {
; CHECK-LABEL: dont_fold_urem_i16_smax:
; CHECK:       # %bb.0:
; CHECK-NEXT:    retq
  %1 = urem <4 x i16> %x, <i16 1, i16 65536, i16 23, i16 5423>
  ret <4 x i16> %1
}

; Don't fold i64 urem.
define <4 x i64> @dont_fold_urem_i64(<4 x i64> %x) {
; SSE-LABEL: dont_fold_urem_i64:
; SSE:       # %bb.0:
; SSE-NEXT:    movq %xmm1, %rcx
; SSE-NEXT:    movabsq $7218291159277650633, %rdx # imm = 0x642C8590B21642C9
; SSE-NEXT:    movq %rcx, %rax
; SSE-NEXT:    mulq %rdx
; SSE-NEXT:    movq %rcx, %rax
; SSE-NEXT:    subq %rdx, %rax
; SSE-NEXT:    shrq %rax
; SSE-NEXT:    addq %rdx, %rax
; SSE-NEXT:    shrq $4, %rax
; SSE-NEXT:    leaq (%rax,%rax,2), %rdx
; SSE-NEXT:    shlq $3, %rdx
; SSE-NEXT:    subq %rdx, %rax
; SSE-NEXT:    addq %rcx, %rax
; SSE-NEXT:    movq %rax, %xmm2
; SSE-NEXT:    pextrq $1, %xmm1, %rcx
; SSE-NEXT:    movabsq $-4513890722074972339, %rdx # imm = 0xC15B704DCBCA2F4D
; SSE-NEXT:    movq %rcx, %rax
; SSE-NEXT:    mulq %rdx
; SSE-NEXT:    shrq $12, %rdx
; SSE-NEXT:    imulq $5423, %rdx, %rax # imm = 0x152F
; SSE-NEXT:    subq %rax, %rcx
; SSE-NEXT:    movq %rcx, %xmm1
; SSE-NEXT:    punpcklqdq {{.*#+}} xmm2 = xmm2[0],xmm1[0]
; SSE-NEXT:    pextrq $1, %xmm0, %rcx
; SSE-NEXT:    movq %rcx, %rax
; SSE-NEXT:    shrq %rax
; SSE-NEXT:    movabsq $7220743857598845893, %rdx # imm = 0x64353C48064353C5
; SSE-NEXT:    mulq %rdx
; SSE-NEXT:    shrq $7, %rdx
; SSE-NEXT:    imulq $654, %rdx, %rax # imm = 0x28E
; SSE-NEXT:    subq %rax, %rcx
; SSE-NEXT:    movq %rcx, %xmm0
; SSE-NEXT:    pslldq {{.*#+}} xmm0 = zero,zero,zero,zero,zero,zero,zero,zero,xmm0[0,1,2,3,4,5,6,7]
; SSE-NEXT:    movdqa %xmm2, %xmm1
; SSE-NEXT:    retq
;
; AVX1-LABEL: dont_fold_urem_i64:
; AVX1:       # %bb.0:
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm1
; AVX1-NEXT:    vmovq %xmm1, %rcx
; AVX1-NEXT:    movabsq $7218291159277650633, %rdx # imm = 0x642C8590B21642C9
; AVX1-NEXT:    movq %rcx, %rax
; AVX1-NEXT:    mulq %rdx
; AVX1-NEXT:    movq %rcx, %rax
; AVX1-NEXT:    subq %rdx, %rax
; AVX1-NEXT:    shrq %rax
; AVX1-NEXT:    addq %rdx, %rax
; AVX1-NEXT:    shrq $4, %rax
; AVX1-NEXT:    leaq (%rax,%rax,2), %rdx
; AVX1-NEXT:    shlq $3, %rdx
; AVX1-NEXT:    subq %rdx, %rax
; AVX1-NEXT:    addq %rcx, %rax
; AVX1-NEXT:    vmovq %rax, %xmm2
; AVX1-NEXT:    vpextrq $1, %xmm1, %rcx
; AVX1-NEXT:    movabsq $-4513890722074972339, %rdx # imm = 0xC15B704DCBCA2F4D
; AVX1-NEXT:    movq %rcx, %rax
; AVX1-NEXT:    mulq %rdx
; AVX1-NEXT:    shrq $12, %rdx
; AVX1-NEXT:    imulq $5423, %rdx, %rax # imm = 0x152F
; AVX1-NEXT:    subq %rax, %rcx
; AVX1-NEXT:    vmovq %rcx, %xmm1
; AVX1-NEXT:    vpunpcklqdq {{.*#+}} xmm1 = xmm2[0],xmm1[0]
; AVX1-NEXT:    vpextrq $1, %xmm0, %rcx
; AVX1-NEXT:    movq %rcx, %rax
; AVX1-NEXT:    shrq %rax
; AVX1-NEXT:    movabsq $7220743857598845893, %rdx # imm = 0x64353C48064353C5
; AVX1-NEXT:    mulq %rdx
; AVX1-NEXT:    shrq $7, %rdx
; AVX1-NEXT:    imulq $654, %rdx, %rax # imm = 0x28E
; AVX1-NEXT:    subq %rax, %rcx
; AVX1-NEXT:    vmovq %rcx, %xmm0
; AVX1-NEXT:    vpslldq {{.*#+}} xmm0 = zero,zero,zero,zero,zero,zero,zero,zero,xmm0[0,1,2,3,4,5,6,7]
; AVX1-NEXT:    vinsertf128 $1, %xmm1, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: dont_fold_urem_i64:
; AVX2:       # %bb.0:
; AVX2-NEXT:    vextracti128 $1, %ymm0, %xmm1
; AVX2-NEXT:    vmovq %xmm1, %rcx
; AVX2-NEXT:    movabsq $7218291159277650633, %rdx # imm = 0x642C8590B21642C9
; AVX2-NEXT:    movq %rcx, %rax
; AVX2-NEXT:    mulq %rdx
; AVX2-NEXT:    movq %rcx, %rax
; AVX2-NEXT:    subq %rdx, %rax
; AVX2-NEXT:    shrq %rax
; AVX2-NEXT:    addq %rdx, %rax
; AVX2-NEXT:    shrq $4, %rax
; AVX2-NEXT:    leaq (%rax,%rax,2), %rdx
; AVX2-NEXT:    shlq $3, %rdx
; AVX2-NEXT:    subq %rdx, %rax
; AVX2-NEXT:    addq %rcx, %rax
; AVX2-NEXT:    vmovq %rax, %xmm2
; AVX2-NEXT:    vpextrq $1, %xmm1, %rcx
; AVX2-NEXT:    movabsq $-4513890722074972339, %rdx # imm = 0xC15B704DCBCA2F4D
; AVX2-NEXT:    movq %rcx, %rax
; AVX2-NEXT:    mulq %rdx
; AVX2-NEXT:    shrq $12, %rdx
; AVX2-NEXT:    imulq $5423, %rdx, %rax # imm = 0x152F
; AVX2-NEXT:    subq %rax, %rcx
; AVX2-NEXT:    vmovq %rcx, %xmm1
; AVX2-NEXT:    vpunpcklqdq {{.*#+}} xmm1 = xmm2[0],xmm1[0]
; AVX2-NEXT:    vpextrq $1, %xmm0, %rcx
; AVX2-NEXT:    movq %rcx, %rax
; AVX2-NEXT:    shrq %rax
; AVX2-NEXT:    movabsq $7220743857598845893, %rdx # imm = 0x64353C48064353C5
; AVX2-NEXT:    mulq %rdx
; AVX2-NEXT:    shrq $7, %rdx
; AVX2-NEXT:    imulq $654, %rdx, %rax # imm = 0x28E
; AVX2-NEXT:    subq %rax, %rcx
; AVX2-NEXT:    vmovq %rcx, %xmm0
; AVX2-NEXT:    vpslldq {{.*#+}} xmm0 = zero,zero,zero,zero,zero,zero,zero,zero,xmm0[0,1,2,3,4,5,6,7]
; AVX2-NEXT:    vinserti128 $1, %xmm1, %ymm0, %ymm0
; AVX2-NEXT:    retq
  %1 = urem <4 x i64> %x, <i64 1, i64 654, i64 23, i64 5423>
  ret <4 x i64> %1
}
