; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py
; RUN: llc < %s -mtriple=x86_64-- -mattr=+sse2 | FileCheck %s --check-prefixes=SSE
; RUN: llc < %s -mtriple=x86_64-- -mattr=+avx2 | FileCheck %s --check-prefixes=AVX,AVX2
; RUN: llc < %s -mtriple=x86_64-- -mattr=+avx512f,+avx512bw | FileCheck %s --check-prefixes=AVX,AVX512

declare void @use_v8i1(<8 x i1>)
declare void @use_v8i8(<8 x i8>)

define <8 x i16> @cmp_ne_load_const(<8 x i8>* %x) nounwind {
; SSE-LABEL: cmp_ne_load_const:
; SSE:       # %bb.0:
; SSE-NEXT:    movq {{.*#+}} xmm0 = mem[0],zero
; SSE-NEXT:    pxor %xmm1, %xmm1
; SSE-NEXT:    pcmpeqb %xmm0, %xmm1
; SSE-NEXT:    pcmpeqd %xmm0, %xmm0
; SSE-NEXT:    pxor %xmm1, %xmm0
; SSE-NEXT:    punpcklbw {{.*#+}} xmm0 = xmm0[0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7]
; SSE-NEXT:    psraw $8, %xmm0
; SSE-NEXT:    retq
;
; AVX2-LABEL: cmp_ne_load_const:
; AVX2:       # %bb.0:
; AVX2-NEXT:    vmovq {{.*#+}} xmm0 = mem[0],zero
; AVX2-NEXT:    vpxor %xmm1, %xmm1, %xmm1
; AVX2-NEXT:    vpcmpeqb %xmm1, %xmm0, %xmm0
; AVX2-NEXT:    vpcmpeqd %xmm1, %xmm1, %xmm1
; AVX2-NEXT:    vpxor %xmm1, %xmm0, %xmm0
; AVX2-NEXT:    vpmovsxbw %xmm0, %xmm0
; AVX2-NEXT:    retq
;
; AVX512-LABEL: cmp_ne_load_const:
; AVX512:       # %bb.0:
; AVX512-NEXT:    vmovq {{.*#+}} xmm0 = mem[0],zero
; AVX512-NEXT:    vpxor %xmm1, %xmm1, %xmm1
; AVX512-NEXT:    vpcmpeqb %xmm1, %xmm0, %xmm0
; AVX512-NEXT:    vpternlogq $15, %zmm0, %zmm0, %zmm0
; AVX512-NEXT:    vpmovsxbw %xmm0, %xmm0
; AVX512-NEXT:    vzeroupper
; AVX512-NEXT:    retq
  %loadx = load <8 x i8>, <8 x i8>* %x
  %icmp = icmp ne <8 x i8> %loadx, zeroinitializer
  %sext = sext <8 x i1> %icmp to <8 x i16>
  ret <8 x i16> %sext
}

define <8 x i16> @cmp_ne_load_const_volatile(<8 x i8>* %x) nounwind {
; SSE-LABEL: cmp_ne_load_const_volatile:
; SSE:       # %bb.0:
; SSE-NEXT:    movq {{.*#+}} xmm0 = mem[0],zero
; SSE-NEXT:    pxor %xmm1, %xmm1
; SSE-NEXT:    pcmpeqb %xmm0, %xmm1
; SSE-NEXT:    pcmpeqd %xmm0, %xmm0
; SSE-NEXT:    pxor %xmm1, %xmm0
; SSE-NEXT:    punpcklbw {{.*#+}} xmm0 = xmm0[0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7]
; SSE-NEXT:    psraw $8, %xmm0
; SSE-NEXT:    retq
;
; AVX2-LABEL: cmp_ne_load_const_volatile:
; AVX2:       # %bb.0:
; AVX2-NEXT:    vmovq {{.*#+}} xmm0 = mem[0],zero
; AVX2-NEXT:    vpxor %xmm1, %xmm1, %xmm1
; AVX2-NEXT:    vpcmpeqb %xmm1, %xmm0, %xmm0
; AVX2-NEXT:    vpcmpeqd %xmm1, %xmm1, %xmm1
; AVX2-NEXT:    vpxor %xmm1, %xmm0, %xmm0
; AVX2-NEXT:    vpmovsxbw %xmm0, %xmm0
; AVX2-NEXT:    retq
;
; AVX512-LABEL: cmp_ne_load_const_volatile:
; AVX512:       # %bb.0:
; AVX512-NEXT:    vmovq {{.*#+}} xmm0 = mem[0],zero
; AVX512-NEXT:    vpxor %xmm1, %xmm1, %xmm1
; AVX512-NEXT:    vpcmpeqb %xmm1, %xmm0, %xmm0
; AVX512-NEXT:    vpternlogq $15, %zmm0, %zmm0, %zmm0
; AVX512-NEXT:    vpmovsxbw %xmm0, %xmm0
; AVX512-NEXT:    vzeroupper
; AVX512-NEXT:    retq
  %loadx = load volatile <8 x i8>, <8 x i8>* %x
  %icmp = icmp ne <8 x i8> %loadx, zeroinitializer
  %sext = sext <8 x i1> %icmp to <8 x i16>
  ret <8 x i16> %sext
}

define <8 x i16> @cmp_ne_load_const_extra_use1(<8 x i8>* %x) nounwind {
; SSE-LABEL: cmp_ne_load_const_extra_use1:
; SSE:       # %bb.0:
; SSE-NEXT:    subq $24, %rsp
; SSE-NEXT:    movq {{.*#+}} xmm0 = mem[0],zero
; SSE-NEXT:    movdqa %xmm0, (%rsp) # 16-byte Spill
; SSE-NEXT:    callq use_v8i8@PLT
; SSE-NEXT:    pxor %xmm0, %xmm0
; SSE-NEXT:    pcmpeqb (%rsp), %xmm0 # 16-byte Folded Reload
; SSE-NEXT:    pcmpeqd %xmm1, %xmm1
; SSE-NEXT:    pxor %xmm0, %xmm1
; SSE-NEXT:    punpcklbw {{.*#+}} xmm0 = xmm0[0],xmm1[0],xmm0[1],xmm1[1],xmm0[2],xmm1[2],xmm0[3],xmm1[3],xmm0[4],xmm1[4],xmm0[5],xmm1[5],xmm0[6],xmm1[6],xmm0[7],xmm1[7]
; SSE-NEXT:    psraw $8, %xmm0
; SSE-NEXT:    addq $24, %rsp
; SSE-NEXT:    retq
;
; AVX2-LABEL: cmp_ne_load_const_extra_use1:
; AVX2:       # %bb.0:
; AVX2-NEXT:    subq $24, %rsp
; AVX2-NEXT:    vmovq {{.*#+}} xmm0 = mem[0],zero
; AVX2-NEXT:    vmovdqa %xmm0, (%rsp) # 16-byte Spill
; AVX2-NEXT:    callq use_v8i8@PLT
; AVX2-NEXT:    vpxor %xmm0, %xmm0, %xmm0
; AVX2-NEXT:    vpcmpeqb (%rsp), %xmm0, %xmm0 # 16-byte Folded Reload
; AVX2-NEXT:    vpcmpeqd %xmm1, %xmm1, %xmm1
; AVX2-NEXT:    vpxor %xmm1, %xmm0, %xmm0
; AVX2-NEXT:    vpmovsxbw %xmm0, %xmm0
; AVX2-NEXT:    addq $24, %rsp
; AVX2-NEXT:    retq
;
; AVX512-LABEL: cmp_ne_load_const_extra_use1:
; AVX512:       # %bb.0:
; AVX512-NEXT:    subq $24, %rsp
; AVX512-NEXT:    vmovq {{.*#+}} xmm0 = mem[0],zero
; AVX512-NEXT:    vmovdqa %xmm0, (%rsp) # 16-byte Spill
; AVX512-NEXT:    callq use_v8i8@PLT
; AVX512-NEXT:    vpxor %xmm0, %xmm0, %xmm0
; AVX512-NEXT:    vpcmpeqb (%rsp), %xmm0, %xmm0 # 16-byte Folded Reload
; AVX512-NEXT:    vpternlogq $15, %zmm0, %zmm0, %zmm0
; AVX512-NEXT:    vpmovsxbw %xmm0, %xmm0
; AVX512-NEXT:    addq $24, %rsp
; AVX512-NEXT:    vzeroupper
; AVX512-NEXT:    retq
  %loadx = load <8 x i8>, <8 x i8>* %x
  call void @use_v8i8(<8 x i8> %loadx)
  %icmp = icmp ne <8 x i8> %loadx, zeroinitializer
  %sext = sext <8 x i1> %icmp to <8 x i16>
  ret <8 x i16> %sext
}

define <8 x i16> @cmp_ne_load_const_extra_use2(<8 x i8>* %x) nounwind {
; SSE-LABEL: cmp_ne_load_const_extra_use2:
; SSE:       # %bb.0:
; SSE-NEXT:    subq $24, %rsp
; SSE-NEXT:    movq {{.*#+}} xmm0 = mem[0],zero
; SSE-NEXT:    pxor %xmm1, %xmm1
; SSE-NEXT:    pcmpeqb %xmm0, %xmm1
; SSE-NEXT:    pcmpeqd %xmm0, %xmm0
; SSE-NEXT:    pxor %xmm1, %xmm0
; SSE-NEXT:    movdqa %xmm0, (%rsp) # 16-byte Spill
; SSE-NEXT:    punpcklbw {{.*#+}} xmm0 = xmm0[0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7]
; SSE-NEXT:    callq use_v8i1@PLT
; SSE-NEXT:    punpcklbw (%rsp), %xmm0 # 16-byte Folded Reload
; SSE-NEXT:    # xmm0 = xmm0[0],mem[0],xmm0[1],mem[1],xmm0[2],mem[2],xmm0[3],mem[3],xmm0[4],mem[4],xmm0[5],mem[5],xmm0[6],mem[6],xmm0[7],mem[7]
; SSE-NEXT:    psraw $8, %xmm0
; SSE-NEXT:    addq $24, %rsp
; SSE-NEXT:    retq
;
; AVX2-LABEL: cmp_ne_load_const_extra_use2:
; AVX2:       # %bb.0:
; AVX2-NEXT:    subq $24, %rsp
; AVX2-NEXT:    vmovq {{.*#+}} xmm0 = mem[0],zero
; AVX2-NEXT:    vpxor %xmm1, %xmm1, %xmm1
; AVX2-NEXT:    vpcmpeqb %xmm1, %xmm0, %xmm0
; AVX2-NEXT:    vpcmpeqd %xmm1, %xmm1, %xmm1
; AVX2-NEXT:    vpxor %xmm1, %xmm0, %xmm0
; AVX2-NEXT:    vmovdqa %xmm0, (%rsp) # 16-byte Spill
; AVX2-NEXT:    vpmovzxbw {{.*#+}} xmm0 = xmm0[0],zero,xmm0[1],zero,xmm0[2],zero,xmm0[3],zero,xmm0[4],zero,xmm0[5],zero,xmm0[6],zero,xmm0[7],zero
; AVX2-NEXT:    callq use_v8i1@PLT
; AVX2-NEXT:    vpmovsxbw (%rsp), %xmm0 # 16-byte Folded Reload
; AVX2-NEXT:    addq $24, %rsp
; AVX2-NEXT:    retq
;
; AVX512-LABEL: cmp_ne_load_const_extra_use2:
; AVX512:       # %bb.0:
; AVX512-NEXT:    subq $120, %rsp
; AVX512-NEXT:    vmovq {{.*#+}} xmm0 = mem[0],zero
; AVX512-NEXT:    vpxor %xmm1, %xmm1, %xmm1
; AVX512-NEXT:    vpcmpeqb %xmm1, %xmm0, %xmm0
; AVX512-NEXT:    vpternlogq $15, %zmm0, %zmm0, %zmm0
; AVX512-NEXT:    vmovdqu64 %zmm0, (%rsp) # 64-byte Spill
; AVX512-NEXT:    vpmovzxbw {{.*#+}} xmm0 = xmm0[0],zero,xmm0[1],zero,xmm0[2],zero,xmm0[3],zero,xmm0[4],zero,xmm0[5],zero,xmm0[6],zero,xmm0[7],zero
; AVX512-NEXT:    vzeroupper
; AVX512-NEXT:    callq use_v8i1@PLT
; AVX512-NEXT:    vpmovsxbw (%rsp), %xmm0 # 16-byte Folded Reload
; AVX512-NEXT:    addq $120, %rsp
; AVX512-NEXT:    retq
  %loadx = load <8 x i8>, <8 x i8>* %x
  %icmp = icmp ne <8 x i8> %loadx, zeroinitializer
  call void @use_v8i1(<8 x i1> %icmp)
  %sext = sext <8 x i1> %icmp to <8 x i16>
  ret <8 x i16> %sext
}

define <8 x i16> @cmp_ne_no_load_const(i64 %x) nounwind {
; SSE-LABEL: cmp_ne_no_load_const:
; SSE:       # %bb.0:
; SSE-NEXT:    movq %rdi, %xmm0
; SSE-NEXT:    pxor %xmm1, %xmm1
; SSE-NEXT:    pcmpeqb %xmm0, %xmm1
; SSE-NEXT:    pcmpeqd %xmm0, %xmm0
; SSE-NEXT:    pxor %xmm1, %xmm0
; SSE-NEXT:    punpcklbw {{.*#+}} xmm0 = xmm0[0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7]
; SSE-NEXT:    psraw $8, %xmm0
; SSE-NEXT:    retq
;
; AVX2-LABEL: cmp_ne_no_load_const:
; AVX2:       # %bb.0:
; AVX2-NEXT:    vmovq %rdi, %xmm0
; AVX2-NEXT:    vpxor %xmm1, %xmm1, %xmm1
; AVX2-NEXT:    vpcmpeqb %xmm1, %xmm0, %xmm0
; AVX2-NEXT:    vpcmpeqd %xmm1, %xmm1, %xmm1
; AVX2-NEXT:    vpxor %xmm1, %xmm0, %xmm0
; AVX2-NEXT:    vpmovsxbw %xmm0, %xmm0
; AVX2-NEXT:    retq
;
; AVX512-LABEL: cmp_ne_no_load_const:
; AVX512:       # %bb.0:
; AVX512-NEXT:    vmovq %rdi, %xmm0
; AVX512-NEXT:    vpxor %xmm1, %xmm1, %xmm1
; AVX512-NEXT:    vpcmpeqb %xmm1, %xmm0, %xmm0
; AVX512-NEXT:    vpternlogq $15, %zmm0, %zmm0, %zmm0
; AVX512-NEXT:    vpmovsxbw %xmm0, %xmm0
; AVX512-NEXT:    vzeroupper
; AVX512-NEXT:    retq
  %t = bitcast i64 %x to <8 x i8>
  %icmp = icmp ne <8 x i8> %t, zeroinitializer
  %sext = sext <8 x i1> %icmp to <8 x i16>
  ret <8 x i16> %sext
}

define <4 x i32> @cmp_ult_load_const(<4 x i8>* %x) nounwind {
; SSE-LABEL: cmp_ult_load_const:
; SSE:       # %bb.0:
; SSE-NEXT:    movd {{.*#+}} xmm0 = mem[0],zero,zero,zero
; SSE-NEXT:    movdqa {{.*#+}} xmm1 = <42,214,0,255,u,u,u,u,u,u,u,u,u,u,u,u>
; SSE-NEXT:    pmaxub %xmm0, %xmm1
; SSE-NEXT:    pcmpeqb %xmm0, %xmm1
; SSE-NEXT:    pcmpeqd %xmm0, %xmm0
; SSE-NEXT:    pxor %xmm1, %xmm0
; SSE-NEXT:    punpcklbw {{.*#+}} xmm0 = xmm0[0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7]
; SSE-NEXT:    punpcklwd {{.*#+}} xmm0 = xmm0[0,0,1,1,2,2,3,3]
; SSE-NEXT:    psrad $24, %xmm0
; SSE-NEXT:    retq
;
; AVX2-LABEL: cmp_ult_load_const:
; AVX2:       # %bb.0:
; AVX2-NEXT:    vmovd {{.*#+}} xmm0 = mem[0],zero,zero,zero
; AVX2-NEXT:    vpmaxub {{\.?LCPI[0-9]+_[0-9]+}}(%rip), %xmm0, %xmm1
; AVX2-NEXT:    vpcmpeqb %xmm1, %xmm0, %xmm0
; AVX2-NEXT:    vpcmpeqd %xmm1, %xmm1, %xmm1
; AVX2-NEXT:    vpxor %xmm1, %xmm0, %xmm0
; AVX2-NEXT:    vpmovsxbd %xmm0, %xmm0
; AVX2-NEXT:    retq
;
; AVX512-LABEL: cmp_ult_load_const:
; AVX512:       # %bb.0:
; AVX512-NEXT:    vmovd {{.*#+}} xmm0 = mem[0],zero,zero,zero
; AVX512-NEXT:    vpmaxub {{\.?LCPI[0-9]+_[0-9]+}}(%rip), %xmm0, %xmm1
; AVX512-NEXT:    vpcmpeqb %xmm1, %xmm0, %xmm0
; AVX512-NEXT:    vpternlogq $15, %zmm0, %zmm0, %zmm0
; AVX512-NEXT:    vpmovsxbd %xmm0, %xmm0
; AVX512-NEXT:    vzeroupper
; AVX512-NEXT:    retq
  %loadx = load <4 x i8>, <4 x i8>* %x
  %icmp = icmp ult <4 x i8> %loadx, <i8 42, i8 -42, i8 0, i8 -1>
  %sext = sext <4 x i1> %icmp to <4 x i32>
  ret <4 x i32> %sext
}

define <3 x i32> @cmp_ult_load_const_bad_type(<3 x i8>* %x) nounwind {
; SSE-LABEL: cmp_ult_load_const_bad_type:
; SSE:       # %bb.0:
; SSE-NEXT:    movd {{.*#+}} xmm0 = mem[0],zero,zero,zero
; SSE-NEXT:    movdqa {{.*#+}} xmm1 = <42,214,0,u,u,u,u,u,u,u,u,u,u,u,u,u>
; SSE-NEXT:    pmaxub %xmm0, %xmm1
; SSE-NEXT:    pcmpeqb %xmm0, %xmm1
; SSE-NEXT:    pcmpeqd %xmm0, %xmm0
; SSE-NEXT:    pxor %xmm1, %xmm0
; SSE-NEXT:    punpcklbw {{.*#+}} xmm0 = xmm0[0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7]
; SSE-NEXT:    punpcklwd {{.*#+}} xmm0 = xmm0[0,0,1,1,2,2,3,3]
; SSE-NEXT:    psrad $24, %xmm0
; SSE-NEXT:    retq
;
; AVX2-LABEL: cmp_ult_load_const_bad_type:
; AVX2:       # %bb.0:
; AVX2-NEXT:    vmovd {{.*#+}} xmm0 = mem[0],zero,zero,zero
; AVX2-NEXT:    vpmaxub {{\.?LCPI[0-9]+_[0-9]+}}(%rip), %xmm0, %xmm1
; AVX2-NEXT:    vpcmpeqb %xmm1, %xmm0, %xmm0
; AVX2-NEXT:    vpcmpeqd %xmm1, %xmm1, %xmm1
; AVX2-NEXT:    vpxor %xmm1, %xmm0, %xmm0
; AVX2-NEXT:    vpmovsxbd %xmm0, %xmm0
; AVX2-NEXT:    retq
;
; AVX512-LABEL: cmp_ult_load_const_bad_type:
; AVX512:       # %bb.0:
; AVX512-NEXT:    vmovd {{.*#+}} xmm0 = mem[0],zero,zero,zero
; AVX512-NEXT:    vpmaxub {{\.?LCPI[0-9]+_[0-9]+}}(%rip), %xmm0, %xmm1
; AVX512-NEXT:    vpcmpeqb %xmm1, %xmm0, %xmm0
; AVX512-NEXT:    vpternlogq $15, %zmm0, %zmm0, %zmm0
; AVX512-NEXT:    vpmovsxbd %xmm0, %xmm0
; AVX512-NEXT:    vzeroupper
; AVX512-NEXT:    retq
  %loadx = load <3 x i8>, <3 x i8>* %x
  %icmp = icmp ult <3 x i8> %loadx, <i8 42, i8 -42, i8 0>
  %sext = sext <3 x i1> %icmp to <3 x i32>
  ret <3 x i32> %sext
}

define <4 x i32> @cmp_slt_load_const(<4 x i8>* %x) nounwind {
; SSE-LABEL: cmp_slt_load_const:
; SSE:       # %bb.0:
; SSE-NEXT:    movd {{.*#+}} xmm0 = mem[0],zero,zero,zero
; SSE-NEXT:    movdqa {{.*#+}} xmm1 = <42,214,0,255,u,u,u,u,u,u,u,u,u,u,u,u>
; SSE-NEXT:    pcmpgtb %xmm0, %xmm1
; SSE-NEXT:    punpcklbw {{.*#+}} xmm0 = xmm0[0],xmm1[0],xmm0[1],xmm1[1],xmm0[2],xmm1[2],xmm0[3],xmm1[3],xmm0[4],xmm1[4],xmm0[5],xmm1[5],xmm0[6],xmm1[6],xmm0[7],xmm1[7]
; SSE-NEXT:    punpcklwd {{.*#+}} xmm0 = xmm0[0,0,1,1,2,2,3,3]
; SSE-NEXT:    psrad $24, %xmm0
; SSE-NEXT:    retq
;
; AVX-LABEL: cmp_slt_load_const:
; AVX:       # %bb.0:
; AVX-NEXT:    vmovd {{.*#+}} xmm0 = mem[0],zero,zero,zero
; AVX-NEXT:    vmovdqa {{.*#+}} xmm1 = <42,214,0,255,u,u,u,u,u,u,u,u,u,u,u,u>
; AVX-NEXT:    vpcmpgtb %xmm0, %xmm1, %xmm0
; AVX-NEXT:    vpmovsxbd %xmm0, %xmm0
; AVX-NEXT:    retq
  %loadx = load <4 x i8>, <4 x i8>* %x
  %icmp = icmp slt <4 x i8> %loadx, <i8 42, i8 -42, i8 0, i8 -1>
  %sext = sext <4 x i1> %icmp to <4 x i32>
  ret <4 x i32> %sext
}

define <2 x i64> @cmp_ne_zextload(<2 x i32>* %x, <2 x i32>* %y) nounwind {
; SSE-LABEL: cmp_ne_zextload:
; SSE:       # %bb.0:
; SSE-NEXT:    movq {{.*#+}} xmm0 = mem[0],zero
; SSE-NEXT:    movq {{.*#+}} xmm1 = mem[0],zero
; SSE-NEXT:    pcmpeqd %xmm0, %xmm1
; SSE-NEXT:    pcmpeqd %xmm0, %xmm0
; SSE-NEXT:    pxor %xmm1, %xmm0
; SSE-NEXT:    pxor %xmm1, %xmm1
; SSE-NEXT:    pcmpgtd %xmm0, %xmm1
; SSE-NEXT:    punpckldq {{.*#+}} xmm0 = xmm0[0],xmm1[0],xmm0[1],xmm1[1]
; SSE-NEXT:    retq
;
; AVX2-LABEL: cmp_ne_zextload:
; AVX2:       # %bb.0:
; AVX2-NEXT:    vmovq {{.*#+}} xmm0 = mem[0],zero
; AVX2-NEXT:    vmovq {{.*#+}} xmm1 = mem[0],zero
; AVX2-NEXT:    vpcmpeqd %xmm1, %xmm0, %xmm0
; AVX2-NEXT:    vpcmpeqd %xmm1, %xmm1, %xmm1
; AVX2-NEXT:    vpxor %xmm1, %xmm0, %xmm0
; AVX2-NEXT:    vpmovsxdq %xmm0, %xmm0
; AVX2-NEXT:    retq
;
; AVX512-LABEL: cmp_ne_zextload:
; AVX512:       # %bb.0:
; AVX512-NEXT:    vmovq {{.*#+}} xmm0 = mem[0],zero
; AVX512-NEXT:    vmovq {{.*#+}} xmm1 = mem[0],zero
; AVX512-NEXT:    vpcmpeqd %xmm1, %xmm0, %xmm0
; AVX512-NEXT:    vpternlogq $15, %zmm0, %zmm0, %zmm0
; AVX512-NEXT:    vpmovsxdq %xmm0, %xmm0
; AVX512-NEXT:    vzeroupper
; AVX512-NEXT:    retq
  %loadx = load <2 x i32>, <2 x i32>* %x
  %loady = load <2 x i32>, <2 x i32>* %y
  %icmp = icmp ne <2 x i32> %loadx, %loady
  %sext = sext <2 x i1> %icmp to <2 x i64>
  ret <2 x i64> %sext
}

define <8 x i16> @cmp_ugt_zextload(<8 x i8>* %x, <8 x i8>* %y) nounwind {
; SSE-LABEL: cmp_ugt_zextload:
; SSE:       # %bb.0:
; SSE-NEXT:    movq {{.*#+}} xmm0 = mem[0],zero
; SSE-NEXT:    movq {{.*#+}} xmm1 = mem[0],zero
; SSE-NEXT:    pminub %xmm0, %xmm1
; SSE-NEXT:    pcmpeqb %xmm0, %xmm1
; SSE-NEXT:    pcmpeqd %xmm0, %xmm0
; SSE-NEXT:    pxor %xmm1, %xmm0
; SSE-NEXT:    punpcklbw {{.*#+}} xmm0 = xmm0[0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7]
; SSE-NEXT:    psraw $8, %xmm0
; SSE-NEXT:    retq
;
; AVX2-LABEL: cmp_ugt_zextload:
; AVX2:       # %bb.0:
; AVX2-NEXT:    vmovq {{.*#+}} xmm0 = mem[0],zero
; AVX2-NEXT:    vmovq {{.*#+}} xmm1 = mem[0],zero
; AVX2-NEXT:    vpminub %xmm1, %xmm0, %xmm1
; AVX2-NEXT:    vpcmpeqb %xmm1, %xmm0, %xmm0
; AVX2-NEXT:    vpcmpeqd %xmm1, %xmm1, %xmm1
; AVX2-NEXT:    vpxor %xmm1, %xmm0, %xmm0
; AVX2-NEXT:    vpmovsxbw %xmm0, %xmm0
; AVX2-NEXT:    retq
;
; AVX512-LABEL: cmp_ugt_zextload:
; AVX512:       # %bb.0:
; AVX512-NEXT:    vmovq {{.*#+}} xmm0 = mem[0],zero
; AVX512-NEXT:    vmovq {{.*#+}} xmm1 = mem[0],zero
; AVX512-NEXT:    vpminub %xmm1, %xmm0, %xmm1
; AVX512-NEXT:    vpcmpeqb %xmm1, %xmm0, %xmm0
; AVX512-NEXT:    vpternlogq $15, %zmm0, %zmm0, %zmm0
; AVX512-NEXT:    vpmovsxbw %xmm0, %xmm0
; AVX512-NEXT:    vzeroupper
; AVX512-NEXT:    retq
  %loadx = load <8 x i8>, <8 x i8>* %x
  %loady = load <8 x i8>, <8 x i8>* %y
  %icmp = icmp ugt <8 x i8> %loadx, %loady
  %sext = sext <8 x i1> %icmp to <8 x i16>
  ret <8 x i16> %sext
}

define <8 x i16> @cmp_sgt_zextload(<8 x i8>* %x, <8 x i8>* %y) nounwind {
; SSE-LABEL: cmp_sgt_zextload:
; SSE:       # %bb.0:
; SSE-NEXT:    movq {{.*#+}} xmm0 = mem[0],zero
; SSE-NEXT:    movq {{.*#+}} xmm1 = mem[0],zero
; SSE-NEXT:    pcmpgtb %xmm1, %xmm0
; SSE-NEXT:    punpcklbw {{.*#+}} xmm0 = xmm0[0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7]
; SSE-NEXT:    psraw $8, %xmm0
; SSE-NEXT:    retq
;
; AVX-LABEL: cmp_sgt_zextload:
; AVX:       # %bb.0:
; AVX-NEXT:    vmovq {{.*#+}} xmm0 = mem[0],zero
; AVX-NEXT:    vmovq {{.*#+}} xmm1 = mem[0],zero
; AVX-NEXT:    vpcmpgtb %xmm1, %xmm0, %xmm0
; AVX-NEXT:    vpmovsxbw %xmm0, %xmm0
; AVX-NEXT:    retq
  %loadx = load <8 x i8>, <8 x i8>* %x
  %loady = load <8 x i8>, <8 x i8>* %y
  %icmp = icmp sgt <8 x i8> %loadx, %loady
  %sext = sext <8 x i1> %icmp to <8 x i16>
  ret <8 x i16> %sext
}

define <8 x i32> @cmp_ne_zextload_from_legal_op(<8 x i16>* %x, <8 x i16>* %y) {
; SSE-LABEL: cmp_ne_zextload_from_legal_op:
; SSE:       # %bb.0:
; SSE-NEXT:    movdqa (%rdi), %xmm0
; SSE-NEXT:    pcmpeqw (%rsi), %xmm0
; SSE-NEXT:    pcmpeqd %xmm1, %xmm1
; SSE-NEXT:    pxor %xmm0, %xmm1
; SSE-NEXT:    punpcklwd {{.*#+}} xmm0 = xmm0[0],xmm1[0],xmm0[1],xmm1[1],xmm0[2],xmm1[2],xmm0[3],xmm1[3]
; SSE-NEXT:    psrad $16, %xmm0
; SSE-NEXT:    punpckhwd {{.*#+}} xmm1 = xmm1[4,4,5,5,6,6,7,7]
; SSE-NEXT:    psrad $16, %xmm1
; SSE-NEXT:    retq
;
; AVX2-LABEL: cmp_ne_zextload_from_legal_op:
; AVX2:       # %bb.0:
; AVX2-NEXT:    vmovdqa (%rdi), %xmm0
; AVX2-NEXT:    vpcmpeqw (%rsi), %xmm0, %xmm0
; AVX2-NEXT:    vpcmpeqd %xmm1, %xmm1, %xmm1
; AVX2-NEXT:    vpxor %xmm1, %xmm0, %xmm0
; AVX2-NEXT:    vpmovsxwd %xmm0, %ymm0
; AVX2-NEXT:    retq
;
; AVX512-LABEL: cmp_ne_zextload_from_legal_op:
; AVX512:       # %bb.0:
; AVX512-NEXT:    vmovdqa (%rdi), %xmm0
; AVX512-NEXT:    vpcmpeqw (%rsi), %xmm0, %xmm0
; AVX512-NEXT:    vpternlogq $15, %zmm0, %zmm0, %zmm0
; AVX512-NEXT:    vpmovsxwd %xmm0, %ymm0
; AVX512-NEXT:    retq
  %loadx = load <8 x i16>, <8 x i16>* %x
  %loady = load <8 x i16>, <8 x i16>* %y
  %icmp = icmp ne <8 x i16> %loadx, %loady
  %sext = sext <8 x i1> %icmp to <8 x i32>
  ret <8 x i32> %sext
}

define <8 x i32> @PR50055(<8 x i8>* %src, <8 x i32>* %dst) nounwind {
; SSE-LABEL: PR50055:
; SSE:       # %bb.0:
; SSE-NEXT:    movq {{.*#+}} xmm2 = mem[0],zero
; SSE-NEXT:    pxor %xmm3, %xmm3
; SSE-NEXT:    movdqa %xmm2, %xmm1
; SSE-NEXT:    punpcklbw {{.*#+}} xmm1 = xmm1[0],xmm3[0],xmm1[1],xmm3[1],xmm1[2],xmm3[2],xmm1[3],xmm3[3],xmm1[4],xmm3[4],xmm1[5],xmm3[5],xmm1[6],xmm3[6],xmm1[7],xmm3[7]
; SSE-NEXT:    movdqa %xmm1, %xmm0
; SSE-NEXT:    punpcklwd {{.*#+}} xmm0 = xmm0[0],xmm3[0],xmm0[1],xmm3[1],xmm0[2],xmm3[2],xmm0[3],xmm3[3]
; SSE-NEXT:    punpckhwd {{.*#+}} xmm1 = xmm1[4],xmm3[4],xmm1[5],xmm3[5],xmm1[6],xmm3[6],xmm1[7],xmm3[7]
; SSE-NEXT:    pcmpeqb %xmm3, %xmm2
; SSE-NEXT:    pcmpeqd %xmm3, %xmm3
; SSE-NEXT:    pxor %xmm2, %xmm3
; SSE-NEXT:    punpcklbw {{.*#+}} xmm2 = xmm2[0],xmm3[0],xmm2[1],xmm3[1],xmm2[2],xmm3[2],xmm2[3],xmm3[3],xmm2[4],xmm3[4],xmm2[5],xmm3[5],xmm2[6],xmm3[6],xmm2[7],xmm3[7]
; SSE-NEXT:    punpcklwd {{.*#+}} xmm3 = xmm3[0],xmm2[0],xmm3[1],xmm2[1],xmm3[2],xmm2[2],xmm3[3],xmm2[3]
; SSE-NEXT:    psrad $24, %xmm3
; SSE-NEXT:    punpckhwd {{.*#+}} xmm2 = xmm2[4,4,5,5,6,6,7,7]
; SSE-NEXT:    psrad $24, %xmm2
; SSE-NEXT:    movdqa %xmm2, 16(%rsi)
; SSE-NEXT:    movdqa %xmm3, (%rsi)
; SSE-NEXT:    retq
;
; AVX2-LABEL: PR50055:
; AVX2:       # %bb.0:
; AVX2-NEXT:    vmovq {{.*#+}} xmm1 = mem[0],zero
; AVX2-NEXT:    vpmovzxbd {{.*#+}} ymm0 = xmm1[0],zero,zero,zero,xmm1[1],zero,zero,zero,xmm1[2],zero,zero,zero,xmm1[3],zero,zero,zero,xmm1[4],zero,zero,zero,xmm1[5],zero,zero,zero,xmm1[6],zero,zero,zero,xmm1[7],zero,zero,zero
; AVX2-NEXT:    vpxor %xmm2, %xmm2, %xmm2
; AVX2-NEXT:    vpcmpeqb %xmm2, %xmm1, %xmm1
; AVX2-NEXT:    vpcmpeqd %xmm2, %xmm2, %xmm2
; AVX2-NEXT:    vpxor %xmm2, %xmm1, %xmm1
; AVX2-NEXT:    vpmovsxbd %xmm1, %ymm1
; AVX2-NEXT:    vmovdqa %ymm1, (%rsi)
; AVX2-NEXT:    retq
;
; AVX512-LABEL: PR50055:
; AVX512:       # %bb.0:
; AVX512-NEXT:    vmovq {{.*#+}} xmm1 = mem[0],zero
; AVX512-NEXT:    vpmovzxbd {{.*#+}} ymm0 = xmm1[0],zero,zero,zero,xmm1[1],zero,zero,zero,xmm1[2],zero,zero,zero,xmm1[3],zero,zero,zero,xmm1[4],zero,zero,zero,xmm1[5],zero,zero,zero,xmm1[6],zero,zero,zero,xmm1[7],zero,zero,zero
; AVX512-NEXT:    vpxor %xmm2, %xmm2, %xmm2
; AVX512-NEXT:    vpcmpeqb %xmm2, %xmm1, %xmm1
; AVX512-NEXT:    vpternlogq $15, %zmm1, %zmm1, %zmm1
; AVX512-NEXT:    vpmovsxbd %xmm1, %ymm1
; AVX512-NEXT:    vmovdqa %ymm1, (%rsi)
; AVX512-NEXT:    retq
  %load = load <8 x i8>, <8 x i8>* %src
  %zext = zext <8 x i8> %load to <8 x i32>
  %icmp = icmp ne <8 x i8> %load, zeroinitializer
  %sext = sext <8 x i1> %icmp to <8 x i32>
  store <8 x i32> %sext, <8 x i32>* %dst
  ret <8 x i32> %zext
}
