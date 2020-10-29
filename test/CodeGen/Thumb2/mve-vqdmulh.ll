; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py
; RUN: llc -mtriple=thumbv8.1m.main-arm-none-eabi -mattr=+mve -verify-machineinstrs %s -o - | FileCheck %s

define arm_aapcs_vfpcc i32 @vqdmulh_i8(<16 x i8> %s0, <16 x i8> %s1) {
; CHECK-LABEL: vqdmulh_i8:
; CHECK:       @ %bb.0: @ %entry
; CHECK-NEXT:    .vsave {d8, d9, d10, d11}
; CHECK-NEXT:    vpush {d8, d9, d10, d11}
; CHECK-NEXT:    vmov.u8 r0, q0[12]
; CHECK-NEXT:    vmov.32 q2[0], r0
; CHECK-NEXT:    vmov.u8 r0, q0[13]
; CHECK-NEXT:    vmov.32 q2[1], r0
; CHECK-NEXT:    vmov.u8 r0, q0[14]
; CHECK-NEXT:    vmov.32 q2[2], r0
; CHECK-NEXT:    vmov.u8 r0, q0[15]
; CHECK-NEXT:    vmov.32 q2[3], r0
; CHECK-NEXT:    vmov.u8 r0, q1[12]
; CHECK-NEXT:    vmov.32 q3[0], r0
; CHECK-NEXT:    vmov.u8 r0, q1[13]
; CHECK-NEXT:    vmov.32 q3[1], r0
; CHECK-NEXT:    vmov.u8 r0, q1[14]
; CHECK-NEXT:    vmov.32 q3[2], r0
; CHECK-NEXT:    vmov.u8 r0, q1[15]
; CHECK-NEXT:    vmov.32 q3[3], r0
; CHECK-NEXT:    vmov.u8 r0, q0[4]
; CHECK-NEXT:    vmov.32 q4[0], r0
; CHECK-NEXT:    vmov.u8 r0, q0[5]
; CHECK-NEXT:    vmov.32 q4[1], r0
; CHECK-NEXT:    vmov.u8 r0, q0[6]
; CHECK-NEXT:    vmov.32 q4[2], r0
; CHECK-NEXT:    vmov.u8 r0, q0[7]
; CHECK-NEXT:    vmov.32 q4[3], r0
; CHECK-NEXT:    vmov.u8 r0, q1[4]
; CHECK-NEXT:    vmov.32 q5[0], r0
; CHECK-NEXT:    vmov.u8 r0, q1[5]
; CHECK-NEXT:    vmov.32 q5[1], r0
; CHECK-NEXT:    vmov.u8 r0, q1[6]
; CHECK-NEXT:    vmov.32 q5[2], r0
; CHECK-NEXT:    vmov.u8 r0, q1[7]
; CHECK-NEXT:    vmov.32 q5[3], r0
; CHECK-NEXT:    vmovlb.s8 q2, q2
; CHECK-NEXT:    vmovlb.s8 q3, q3
; CHECK-NEXT:    vmovlb.s8 q4, q4
; CHECK-NEXT:    vmovlb.s8 q5, q5
; CHECK-NEXT:    vmovlb.s16 q2, q2
; CHECK-NEXT:    vmovlb.s16 q3, q3
; CHECK-NEXT:    vmovlb.s16 q4, q4
; CHECK-NEXT:    vmovlb.s16 q5, q5
; CHECK-NEXT:    vmul.i32 q2, q3, q2
; CHECK-NEXT:    vmul.i32 q4, q5, q4
; CHECK-NEXT:    vshr.s32 q3, q2, #7
; CHECK-NEXT:    vmov.i32 q2, #0x7f
; CHECK-NEXT:    vshr.s32 q4, q4, #7
; CHECK-NEXT:    vmin.s32 q3, q3, q2
; CHECK-NEXT:    vmin.s32 q4, q4, q2
; CHECK-NEXT:    vmov.u8 r0, q0[8]
; CHECK-NEXT:    vadd.i32 q3, q4, q3
; CHECK-NEXT:    vmov.32 q4[0], r0
; CHECK-NEXT:    vmov.u8 r0, q0[9]
; CHECK-NEXT:    vmov.32 q4[1], r0
; CHECK-NEXT:    vmov.u8 r0, q0[10]
; CHECK-NEXT:    vmov.32 q4[2], r0
; CHECK-NEXT:    vmov.u8 r0, q0[11]
; CHECK-NEXT:    vmov.32 q4[3], r0
; CHECK-NEXT:    vmov.u8 r0, q1[8]
; CHECK-NEXT:    vmov.32 q5[0], r0
; CHECK-NEXT:    vmov.u8 r0, q1[9]
; CHECK-NEXT:    vmov.32 q5[1], r0
; CHECK-NEXT:    vmov.u8 r0, q1[10]
; CHECK-NEXT:    vmov.32 q5[2], r0
; CHECK-NEXT:    vmov.u8 r0, q1[11]
; CHECK-NEXT:    vmov.32 q5[3], r0
; CHECK-NEXT:    vmovlb.s8 q4, q4
; CHECK-NEXT:    vmovlb.s8 q5, q5
; CHECK-NEXT:    vmovlb.s16 q4, q4
; CHECK-NEXT:    vmovlb.s16 q5, q5
; CHECK-NEXT:    vmov.u8 r0, q0[0]
; CHECK-NEXT:    vmul.i32 q4, q5, q4
; CHECK-NEXT:    vmov.32 q5[0], r0
; CHECK-NEXT:    vmov.u8 r0, q0[1]
; CHECK-NEXT:    vshr.s32 q4, q4, #7
; CHECK-NEXT:    vmov.32 q5[1], r0
; CHECK-NEXT:    vmov.u8 r0, q0[2]
; CHECK-NEXT:    vmov.32 q5[2], r0
; CHECK-NEXT:    vmov.u8 r0, q0[3]
; CHECK-NEXT:    vmov.32 q5[3], r0
; CHECK-NEXT:    vmov.u8 r0, q1[0]
; CHECK-NEXT:    vmovlb.s8 q0, q5
; CHECK-NEXT:    vmov.32 q5[0], r0
; CHECK-NEXT:    vmov.u8 r0, q1[1]
; CHECK-NEXT:    vmovlb.s16 q0, q0
; CHECK-NEXT:    vmov.32 q5[1], r0
; CHECK-NEXT:    vmov.u8 r0, q1[2]
; CHECK-NEXT:    vmov.32 q5[2], r0
; CHECK-NEXT:    vmov.u8 r0, q1[3]
; CHECK-NEXT:    vmov.32 q5[3], r0
; CHECK-NEXT:    vmin.s32 q4, q4, q2
; CHECK-NEXT:    vmovlb.s8 q1, q5
; CHECK-NEXT:    vmovlb.s16 q1, q1
; CHECK-NEXT:    vmul.i32 q0, q1, q0
; CHECK-NEXT:    vshr.s32 q0, q0, #7
; CHECK-NEXT:    vmin.s32 q0, q0, q2
; CHECK-NEXT:    vadd.i32 q0, q0, q4
; CHECK-NEXT:    vadd.i32 q0, q0, q3
; CHECK-NEXT:    vaddv.u32 r0, q0
; CHECK-NEXT:    vpop {d8, d9, d10, d11}
; CHECK-NEXT:    bx lr
entry:
  %l2 = sext <16 x i8> %s0 to <16 x i32>
  %l5 = sext <16 x i8> %s1 to <16 x i32>
  %l6 = mul nsw <16 x i32> %l5, %l2
  %l7 = ashr <16 x i32> %l6, <i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7>
  %l8 = icmp slt <16 x i32> %l7, <i32 127, i32 127, i32 127, i32 127, i32 127, i32 127, i32 127, i32 127, i32 127, i32 127, i32 127, i32 127, i32 127, i32 127, i32 127, i32 127>
  %l9 = select <16 x i1> %l8, <16 x i32> %l7, <16 x i32> <i32 127, i32 127, i32 127, i32 127, i32 127, i32 127, i32 127, i32 127, i32 127, i32 127, i32 127, i32 127, i32 127, i32 127, i32 127, i32 127>
  %l10 = call i32 @llvm.vector.reduce.add.v16i32(<16 x i32> %l9)
  ret i32 %l10
}

define arm_aapcs_vfpcc <16 x i8> @vqdmulh_i8_b(<16 x i8> %s0, <16 x i8> %s1) {
; CHECK-LABEL: vqdmulh_i8_b:
; CHECK:       @ %bb.0: @ %entry
; CHECK-NEXT:    .vsave {d8, d9, d10, d11}
; CHECK-NEXT:    vpush {d8, d9, d10, d11}
; CHECK-NEXT:    vmov q2, q0
; CHECK-NEXT:    vmov.u8 r0, q0[0]
; CHECK-NEXT:    vmov.32 q0[0], r0
; CHECK-NEXT:    vmov.u8 r0, q2[1]
; CHECK-NEXT:    vmov.32 q0[1], r0
; CHECK-NEXT:    vmov.u8 r0, q2[2]
; CHECK-NEXT:    vmov.32 q0[2], r0
; CHECK-NEXT:    vmov.u8 r0, q2[3]
; CHECK-NEXT:    vmov.32 q0[3], r0
; CHECK-NEXT:    vmov.u8 r0, q1[0]
; CHECK-NEXT:    vmov.32 q3[0], r0
; CHECK-NEXT:    vmov.u8 r0, q1[1]
; CHECK-NEXT:    vmov.32 q3[1], r0
; CHECK-NEXT:    vmov.u8 r0, q1[2]
; CHECK-NEXT:    vmov.32 q3[2], r0
; CHECK-NEXT:    vmov.u8 r0, q1[3]
; CHECK-NEXT:    vmov.32 q3[3], r0
; CHECK-NEXT:    vmovlb.s8 q0, q0
; CHECK-NEXT:    vmovlb.s8 q3, q3
; CHECK-NEXT:    vmovlb.s16 q0, q0
; CHECK-NEXT:    vmovlb.s16 q3, q3
; CHECK-NEXT:    vmul.i32 q0, q3, q0
; CHECK-NEXT:    vmov.i32 q3, #0x7f
; CHECK-NEXT:    vshr.s32 q0, q0, #7
; CHECK-NEXT:    vmin.s32 q4, q0, q3
; CHECK-NEXT:    vmov r0, s16
; CHECK-NEXT:    vmov.8 q0[0], r0
; CHECK-NEXT:    vmov r0, s17
; CHECK-NEXT:    vmov.8 q0[1], r0
; CHECK-NEXT:    vmov r0, s18
; CHECK-NEXT:    vmov.8 q0[2], r0
; CHECK-NEXT:    vmov r0, s19
; CHECK-NEXT:    vmov.8 q0[3], r0
; CHECK-NEXT:    vmov.u8 r0, q2[4]
; CHECK-NEXT:    vmov.32 q4[0], r0
; CHECK-NEXT:    vmov.u8 r0, q2[5]
; CHECK-NEXT:    vmov.32 q4[1], r0
; CHECK-NEXT:    vmov.u8 r0, q2[6]
; CHECK-NEXT:    vmov.32 q4[2], r0
; CHECK-NEXT:    vmov.u8 r0, q2[7]
; CHECK-NEXT:    vmov.32 q4[3], r0
; CHECK-NEXT:    vmov.u8 r0, q1[4]
; CHECK-NEXT:    vmov.32 q5[0], r0
; CHECK-NEXT:    vmov.u8 r0, q1[5]
; CHECK-NEXT:    vmov.32 q5[1], r0
; CHECK-NEXT:    vmov.u8 r0, q1[6]
; CHECK-NEXT:    vmov.32 q5[2], r0
; CHECK-NEXT:    vmov.u8 r0, q1[7]
; CHECK-NEXT:    vmov.32 q5[3], r0
; CHECK-NEXT:    vmovlb.s8 q4, q4
; CHECK-NEXT:    vmovlb.s8 q5, q5
; CHECK-NEXT:    vmovlb.s16 q4, q4
; CHECK-NEXT:    vmovlb.s16 q5, q5
; CHECK-NEXT:    vmul.i32 q4, q5, q4
; CHECK-NEXT:    vshr.s32 q4, q4, #7
; CHECK-NEXT:    vmin.s32 q4, q4, q3
; CHECK-NEXT:    vmov r0, s16
; CHECK-NEXT:    vmov.8 q0[4], r0
; CHECK-NEXT:    vmov r0, s17
; CHECK-NEXT:    vmov.8 q0[5], r0
; CHECK-NEXT:    vmov r0, s18
; CHECK-NEXT:    vmov.8 q0[6], r0
; CHECK-NEXT:    vmov r0, s19
; CHECK-NEXT:    vmov.8 q0[7], r0
; CHECK-NEXT:    vmov.u8 r0, q2[8]
; CHECK-NEXT:    vmov.32 q4[0], r0
; CHECK-NEXT:    vmov.u8 r0, q2[9]
; CHECK-NEXT:    vmov.32 q4[1], r0
; CHECK-NEXT:    vmov.u8 r0, q2[10]
; CHECK-NEXT:    vmov.32 q4[2], r0
; CHECK-NEXT:    vmov.u8 r0, q2[11]
; CHECK-NEXT:    vmov.32 q4[3], r0
; CHECK-NEXT:    vmov.u8 r0, q1[8]
; CHECK-NEXT:    vmov.32 q5[0], r0
; CHECK-NEXT:    vmov.u8 r0, q1[9]
; CHECK-NEXT:    vmov.32 q5[1], r0
; CHECK-NEXT:    vmov.u8 r0, q1[10]
; CHECK-NEXT:    vmov.32 q5[2], r0
; CHECK-NEXT:    vmov.u8 r0, q1[11]
; CHECK-NEXT:    vmov.32 q5[3], r0
; CHECK-NEXT:    vmovlb.s8 q4, q4
; CHECK-NEXT:    vmovlb.s8 q5, q5
; CHECK-NEXT:    vmovlb.s16 q4, q4
; CHECK-NEXT:    vmovlb.s16 q5, q5
; CHECK-NEXT:    vmul.i32 q4, q5, q4
; CHECK-NEXT:    vshr.s32 q4, q4, #7
; CHECK-NEXT:    vmin.s32 q4, q4, q3
; CHECK-NEXT:    vmov r0, s16
; CHECK-NEXT:    vmov.8 q0[8], r0
; CHECK-NEXT:    vmov r0, s17
; CHECK-NEXT:    vmov.8 q0[9], r0
; CHECK-NEXT:    vmov r0, s18
; CHECK-NEXT:    vmov.8 q0[10], r0
; CHECK-NEXT:    vmov r0, s19
; CHECK-NEXT:    vmov.8 q0[11], r0
; CHECK-NEXT:    vmov.u8 r0, q2[12]
; CHECK-NEXT:    vmov.32 q4[0], r0
; CHECK-NEXT:    vmov.u8 r0, q2[13]
; CHECK-NEXT:    vmov.32 q4[1], r0
; CHECK-NEXT:    vmov.u8 r0, q2[14]
; CHECK-NEXT:    vmov.32 q4[2], r0
; CHECK-NEXT:    vmov.u8 r0, q2[15]
; CHECK-NEXT:    vmov.32 q4[3], r0
; CHECK-NEXT:    vmov.u8 r0, q1[12]
; CHECK-NEXT:    vmovlb.s8 q2, q4
; CHECK-NEXT:    vmov.32 q4[0], r0
; CHECK-NEXT:    vmov.u8 r0, q1[13]
; CHECK-NEXT:    vmovlb.s16 q2, q2
; CHECK-NEXT:    vmov.32 q4[1], r0
; CHECK-NEXT:    vmov.u8 r0, q1[14]
; CHECK-NEXT:    vmov.32 q4[2], r0
; CHECK-NEXT:    vmov.u8 r0, q1[15]
; CHECK-NEXT:    vmov.32 q4[3], r0
; CHECK-NEXT:    vmovlb.s8 q1, q4
; CHECK-NEXT:    vmovlb.s16 q1, q1
; CHECK-NEXT:    vmul.i32 q1, q1, q2
; CHECK-NEXT:    vshr.s32 q1, q1, #7
; CHECK-NEXT:    vmin.s32 q1, q1, q3
; CHECK-NEXT:    vmov r0, s4
; CHECK-NEXT:    vmov.8 q0[12], r0
; CHECK-NEXT:    vmov r0, s5
; CHECK-NEXT:    vmov.8 q0[13], r0
; CHECK-NEXT:    vmov r0, s6
; CHECK-NEXT:    vmov.8 q0[14], r0
; CHECK-NEXT:    vmov r0, s7
; CHECK-NEXT:    vmov.8 q0[15], r0
; CHECK-NEXT:    vpop {d8, d9, d10, d11}
; CHECK-NEXT:    bx lr
entry:
  %l2 = sext <16 x i8> %s0 to <16 x i32>
  %l5 = sext <16 x i8> %s1 to <16 x i32>
  %l6 = mul nsw <16 x i32> %l5, %l2
  %l7 = ashr <16 x i32> %l6, <i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7>
  %l8 = icmp slt <16 x i32> %l7, <i32 127, i32 127, i32 127, i32 127, i32 127, i32 127, i32 127, i32 127, i32 127, i32 127, i32 127, i32 127, i32 127, i32 127, i32 127, i32 127>
  %l9 = select <16 x i1> %l8, <16 x i32> %l7, <16 x i32> <i32 127, i32 127, i32 127, i32 127, i32 127, i32 127, i32 127, i32 127, i32 127, i32 127, i32 127, i32 127, i32 127, i32 127, i32 127, i32 127>
  %l10 = trunc <16 x i32> %l9 to <16 x i8>
  ret <16 x i8> %l10
}

define arm_aapcs_vfpcc i32 @vqdmulh_i16(<8 x i16> %s0, <8 x i16> %s1) {
; CHECK-LABEL: vqdmulh_i16:
; CHECK:       @ %bb.0: @ %entry
; CHECK-NEXT:    .vsave {d8, d9}
; CHECK-NEXT:    vpush {d8, d9}
; CHECK-NEXT:    vmov.u16 r0, q0[4]
; CHECK-NEXT:    vmov.32 q2[0], r0
; CHECK-NEXT:    vmov.u16 r0, q0[5]
; CHECK-NEXT:    vmov.32 q2[1], r0
; CHECK-NEXT:    vmov.u16 r0, q0[6]
; CHECK-NEXT:    vmov.32 q2[2], r0
; CHECK-NEXT:    vmov.u16 r0, q0[7]
; CHECK-NEXT:    vmov.32 q2[3], r0
; CHECK-NEXT:    vmov.u16 r0, q1[4]
; CHECK-NEXT:    vmov.32 q3[0], r0
; CHECK-NEXT:    vmov.u16 r0, q1[5]
; CHECK-NEXT:    vmov.32 q3[1], r0
; CHECK-NEXT:    vmov.u16 r0, q1[6]
; CHECK-NEXT:    vmov.32 q3[2], r0
; CHECK-NEXT:    vmov.u16 r0, q1[7]
; CHECK-NEXT:    vmov.32 q3[3], r0
; CHECK-NEXT:    vmov.u16 r0, q0[0]
; CHECK-NEXT:    vmov.32 q4[0], r0
; CHECK-NEXT:    vmov.u16 r0, q0[1]
; CHECK-NEXT:    vmov.32 q4[1], r0
; CHECK-NEXT:    vmov.u16 r0, q0[2]
; CHECK-NEXT:    vmov.32 q4[2], r0
; CHECK-NEXT:    vmov.u16 r0, q0[3]
; CHECK-NEXT:    vmov.32 q4[3], r0
; CHECK-NEXT:    vmov.u16 r0, q1[0]
; CHECK-NEXT:    vmov.32 q0[0], r0
; CHECK-NEXT:    vmov.u16 r0, q1[1]
; CHECK-NEXT:    vmov.32 q0[1], r0
; CHECK-NEXT:    vmov.u16 r0, q1[2]
; CHECK-NEXT:    vmov.32 q0[2], r0
; CHECK-NEXT:    vmov.u16 r0, q1[3]
; CHECK-NEXT:    vmov.32 q0[3], r0
; CHECK-NEXT:    vmullb.s16 q2, q3, q2
; CHECK-NEXT:    vmullb.s16 q0, q0, q4
; CHECK-NEXT:    vshr.s32 q3, q2, #15
; CHECK-NEXT:    vmov.i32 q2, #0x7fff
; CHECK-NEXT:    vshr.s32 q0, q0, #15
; CHECK-NEXT:    vmin.s32 q3, q3, q2
; CHECK-NEXT:    vmin.s32 q0, q0, q2
; CHECK-NEXT:    vadd.i32 q0, q0, q3
; CHECK-NEXT:    vaddv.u32 r0, q0
; CHECK-NEXT:    vpop {d8, d9}
; CHECK-NEXT:    bx lr
entry:
  %l2 = sext <8 x i16> %s0 to <8 x i32>
  %l5 = sext <8 x i16> %s1 to <8 x i32>
  %l6 = mul nsw <8 x i32> %l5, %l2
  %l7 = ashr <8 x i32> %l6, <i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15>
  %l8 = icmp slt <8 x i32> %l7, <i32 32767, i32 32767, i32 32767, i32 32767, i32 32767, i32 32767, i32 32767, i32 32767>
  %l9 = select <8 x i1> %l8, <8 x i32> %l7, <8 x i32> <i32 32767, i32 32767, i32 32767, i32 32767, i32 32767, i32 32767, i32 32767, i32 32767>
  %l10 = call i32 @llvm.vector.reduce.add.v8i32(<8 x i32> %l9)
  ret i32 %l10
}

define arm_aapcs_vfpcc <8 x i16> @vqdmulh_i16_b(<8 x i16> %s0, <8 x i16> %s1) {
; CHECK-LABEL: vqdmulh_i16_b:
; CHECK:       @ %bb.0: @ %entry
; CHECK-NEXT:    .vsave {d8, d9}
; CHECK-NEXT:    vpush {d8, d9}
; CHECK-NEXT:    vmov q2, q0
; CHECK-NEXT:    vmov.u16 r0, q0[0]
; CHECK-NEXT:    vmov.32 q0[0], r0
; CHECK-NEXT:    vmov.u16 r0, q2[1]
; CHECK-NEXT:    vmov.32 q0[1], r0
; CHECK-NEXT:    vmov.u16 r0, q2[2]
; CHECK-NEXT:    vmov.32 q0[2], r0
; CHECK-NEXT:    vmov.u16 r0, q2[3]
; CHECK-NEXT:    vmov.32 q0[3], r0
; CHECK-NEXT:    vmov.u16 r0, q1[0]
; CHECK-NEXT:    vmov.32 q3[0], r0
; CHECK-NEXT:    vmov.u16 r0, q1[1]
; CHECK-NEXT:    vmov.32 q3[1], r0
; CHECK-NEXT:    vmov.u16 r0, q1[2]
; CHECK-NEXT:    vmov.32 q3[2], r0
; CHECK-NEXT:    vmov.u16 r0, q1[3]
; CHECK-NEXT:    vmov.32 q3[3], r0
; CHECK-NEXT:    vmullb.s16 q0, q3, q0
; CHECK-NEXT:    vmov.i32 q3, #0x7fff
; CHECK-NEXT:    vshr.s32 q0, q0, #15
; CHECK-NEXT:    vmin.s32 q4, q0, q3
; CHECK-NEXT:    vmov r0, s16
; CHECK-NEXT:    vmov.16 q0[0], r0
; CHECK-NEXT:    vmov r0, s17
; CHECK-NEXT:    vmov.16 q0[1], r0
; CHECK-NEXT:    vmov r0, s18
; CHECK-NEXT:    vmov.16 q0[2], r0
; CHECK-NEXT:    vmov r0, s19
; CHECK-NEXT:    vmov.16 q0[3], r0
; CHECK-NEXT:    vmov.u16 r0, q2[4]
; CHECK-NEXT:    vmov.32 q4[0], r0
; CHECK-NEXT:    vmov.u16 r0, q2[5]
; CHECK-NEXT:    vmov.32 q4[1], r0
; CHECK-NEXT:    vmov.u16 r0, q2[6]
; CHECK-NEXT:    vmov.32 q4[2], r0
; CHECK-NEXT:    vmov.u16 r0, q2[7]
; CHECK-NEXT:    vmov.32 q4[3], r0
; CHECK-NEXT:    vmov.u16 r0, q1[4]
; CHECK-NEXT:    vmov.32 q2[0], r0
; CHECK-NEXT:    vmov.u16 r0, q1[5]
; CHECK-NEXT:    vmov.32 q2[1], r0
; CHECK-NEXT:    vmov.u16 r0, q1[6]
; CHECK-NEXT:    vmov.32 q2[2], r0
; CHECK-NEXT:    vmov.u16 r0, q1[7]
; CHECK-NEXT:    vmov.32 q2[3], r0
; CHECK-NEXT:    vmullb.s16 q1, q2, q4
; CHECK-NEXT:    vshr.s32 q1, q1, #15
; CHECK-NEXT:    vmin.s32 q1, q1, q3
; CHECK-NEXT:    vmov r0, s4
; CHECK-NEXT:    vmov.16 q0[4], r0
; CHECK-NEXT:    vmov r0, s5
; CHECK-NEXT:    vmov.16 q0[5], r0
; CHECK-NEXT:    vmov r0, s6
; CHECK-NEXT:    vmov.16 q0[6], r0
; CHECK-NEXT:    vmov r0, s7
; CHECK-NEXT:    vmov.16 q0[7], r0
; CHECK-NEXT:    vpop {d8, d9}
; CHECK-NEXT:    bx lr
entry:
  %l2 = sext <8 x i16> %s0 to <8 x i32>
  %l5 = sext <8 x i16> %s1 to <8 x i32>
  %l6 = mul nsw <8 x i32> %l5, %l2
  %l7 = ashr <8 x i32> %l6, <i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15>
  %l8 = icmp slt <8 x i32> %l7, <i32 32767, i32 32767, i32 32767, i32 32767, i32 32767, i32 32767, i32 32767, i32 32767>
  %l9 = select <8 x i1> %l8, <8 x i32> %l7, <8 x i32> <i32 32767, i32 32767, i32 32767, i32 32767, i32 32767, i32 32767, i32 32767, i32 32767>
  %l10 = trunc <8 x i32> %l9 to <8 x i16>
  ret <8 x i16> %l10
}

define arm_aapcs_vfpcc <8 x i16> @vqdmulh_i16_c(<8 x i16> %s0, <8 x i16> %s1) {
; CHECK-LABEL: vqdmulh_i16_c:
; CHECK:       @ %bb.0: @ %entry
; CHECK-NEXT:    .vsave {d8, d9}
; CHECK-NEXT:    vpush {d8, d9}
; CHECK-NEXT:    vmov q2, q0
; CHECK-NEXT:    vmov.u16 r0, q0[0]
; CHECK-NEXT:    vmov.32 q0[0], r0
; CHECK-NEXT:    vmov.u16 r0, q2[1]
; CHECK-NEXT:    vmov.32 q0[1], r0
; CHECK-NEXT:    vmov.u16 r0, q2[2]
; CHECK-NEXT:    vmov.32 q0[2], r0
; CHECK-NEXT:    vmov.u16 r0, q2[3]
; CHECK-NEXT:    vmov.32 q0[3], r0
; CHECK-NEXT:    vmov.u16 r0, q1[0]
; CHECK-NEXT:    vmov.32 q3[0], r0
; CHECK-NEXT:    vmov.u16 r0, q1[1]
; CHECK-NEXT:    vmov.32 q3[1], r0
; CHECK-NEXT:    vmov.u16 r0, q1[2]
; CHECK-NEXT:    vmov.32 q3[2], r0
; CHECK-NEXT:    vmov.u16 r0, q1[3]
; CHECK-NEXT:    vmov.32 q3[3], r0
; CHECK-NEXT:    vmullb.s16 q0, q3, q0
; CHECK-NEXT:    vmov.i32 q3, #0x7fff
; CHECK-NEXT:    vshl.i32 q0, q0, #10
; CHECK-NEXT:    vshr.s32 q0, q0, #10
; CHECK-NEXT:    vshr.s32 q0, q0, #15
; CHECK-NEXT:    vmin.s32 q4, q0, q3
; CHECK-NEXT:    vmov r0, s16
; CHECK-NEXT:    vmov.16 q0[0], r0
; CHECK-NEXT:    vmov r0, s17
; CHECK-NEXT:    vmov.16 q0[1], r0
; CHECK-NEXT:    vmov r0, s18
; CHECK-NEXT:    vmov.16 q0[2], r0
; CHECK-NEXT:    vmov r0, s19
; CHECK-NEXT:    vmov.16 q0[3], r0
; CHECK-NEXT:    vmov.u16 r0, q2[4]
; CHECK-NEXT:    vmov.32 q4[0], r0
; CHECK-NEXT:    vmov.u16 r0, q2[5]
; CHECK-NEXT:    vmov.32 q4[1], r0
; CHECK-NEXT:    vmov.u16 r0, q2[6]
; CHECK-NEXT:    vmov.32 q4[2], r0
; CHECK-NEXT:    vmov.u16 r0, q2[7]
; CHECK-NEXT:    vmov.32 q4[3], r0
; CHECK-NEXT:    vmov.u16 r0, q1[4]
; CHECK-NEXT:    vmov.32 q2[0], r0
; CHECK-NEXT:    vmov.u16 r0, q1[5]
; CHECK-NEXT:    vmov.32 q2[1], r0
; CHECK-NEXT:    vmov.u16 r0, q1[6]
; CHECK-NEXT:    vmov.32 q2[2], r0
; CHECK-NEXT:    vmov.u16 r0, q1[7]
; CHECK-NEXT:    vmov.32 q2[3], r0
; CHECK-NEXT:    vmullb.s16 q1, q2, q4
; CHECK-NEXT:    vshl.i32 q1, q1, #10
; CHECK-NEXT:    vshr.s32 q1, q1, #10
; CHECK-NEXT:    vshr.s32 q1, q1, #15
; CHECK-NEXT:    vmin.s32 q1, q1, q3
; CHECK-NEXT:    vmov r0, s4
; CHECK-NEXT:    vmov.16 q0[4], r0
; CHECK-NEXT:    vmov r0, s5
; CHECK-NEXT:    vmov.16 q0[5], r0
; CHECK-NEXT:    vmov r0, s6
; CHECK-NEXT:    vmov.16 q0[6], r0
; CHECK-NEXT:    vmov r0, s7
; CHECK-NEXT:    vmov.16 q0[7], r0
; CHECK-NEXT:    vpop {d8, d9}
; CHECK-NEXT:    bx lr
entry:
  %l2 = sext <8 x i16> %s0 to <8 x i22>
  %l5 = sext <8 x i16> %s1 to <8 x i22>
  %l6 = mul nsw <8 x i22> %l5, %l2
  %l7 = ashr <8 x i22> %l6, <i22 15, i22 15, i22 15, i22 15, i22 15, i22 15, i22 15, i22 15>
  %l8 = icmp slt <8 x i22> %l7, <i22 32767, i22 32767, i22 32767, i22 32767, i22 32767, i22 32767, i22 32767, i22 32767>
  %l9 = select <8 x i1> %l8, <8 x i22> %l7, <8 x i22> <i22 32767, i22 32767, i22 32767, i22 32767, i22 32767, i22 32767, i22 32767, i22 32767>
  %l10 = trunc <8 x i22> %l9 to <8 x i16>
  ret <8 x i16> %l10
}

define arm_aapcs_vfpcc i64 @vqdmulh_i32(<4 x i32> %s0, <4 x i32> %s1) {
; CHECK-LABEL: vqdmulh_i32:
; CHECK:       @ %bb.0: @ %entry
; CHECK-NEXT:    .save {r4, r5, r7, lr}
; CHECK-NEXT:    push {r4, r5, r7, lr}
; CHECK-NEXT:    .vsave {d8, d9, d10, d11}
; CHECK-NEXT:    vpush {d8, d9, d10, d11}
; CHECK-NEXT:    vmov.f32 s8, s0
; CHECK-NEXT:    mvn r12, #-2147483648
; CHECK-NEXT:    vmov.f32 s16, s4
; CHECK-NEXT:    vmov.f32 s18, s5
; CHECK-NEXT:    vmov.f32 s10, s1
; CHECK-NEXT:    vmov r0, s8
; CHECK-NEXT:    vmov r1, s16
; CHECK-NEXT:    vmov r7, s18
; CHECK-NEXT:    smull r2, r3, r1, r0
; CHECK-NEXT:    movs r0, #0
; CHECK-NEXT:    asrl r2, r3, #31
; CHECK-NEXT:    subs.w r1, r2, r12
; CHECK-NEXT:    vmov.32 q5[0], r2
; CHECK-NEXT:    sbcs r1, r3, #0
; CHECK-NEXT:    vmov.32 q5[1], r3
; CHECK-NEXT:    mov.w r1, #0
; CHECK-NEXT:    it lt
; CHECK-NEXT:    movlt r1, #1
; CHECK-NEXT:    cmp r1, #0
; CHECK-NEXT:    csetm r1, ne
; CHECK-NEXT:    vmov.32 q3[0], r1
; CHECK-NEXT:    vmov.32 q3[1], r1
; CHECK-NEXT:    vmov r1, s10
; CHECK-NEXT:    smull r4, r1, r7, r1
; CHECK-NEXT:    asrl r4, r1, #31
; CHECK-NEXT:    subs.w r7, r4, r12
; CHECK-NEXT:    vmov.32 q5[2], r4
; CHECK-NEXT:    sbcs r7, r1, #0
; CHECK-NEXT:    vmov.32 q5[3], r1
; CHECK-NEXT:    mov.w r7, #0
; CHECK-NEXT:    it lt
; CHECK-NEXT:    movlt r7, #1
; CHECK-NEXT:    cmp r7, #0
; CHECK-NEXT:    csetm r7, ne
; CHECK-NEXT:    vmov.32 q3[2], r7
; CHECK-NEXT:    vmov.32 q3[3], r7
; CHECK-NEXT:    adr r7, .LCPI5_0
; CHECK-NEXT:    vldrw.u32 q2, [r7]
; CHECK-NEXT:    vbic q4, q2, q3
; CHECK-NEXT:    vand q3, q5, q3
; CHECK-NEXT:    vorr q3, q3, q4
; CHECK-NEXT:    vmov r1, s14
; CHECK-NEXT:    vmov r7, s12
; CHECK-NEXT:    vmov r2, s15
; CHECK-NEXT:    vmov r3, s13
; CHECK-NEXT:    vmov.f32 s12, s2
; CHECK-NEXT:    vmov.f32 s14, s3
; CHECK-NEXT:    vmov.f32 s0, s6
; CHECK-NEXT:    vmov.f32 s2, s7
; CHECK-NEXT:    vmullb.s32 q1, q0, q3
; CHECK-NEXT:    vmov r5, s5
; CHECK-NEXT:    vmov r4, s6
; CHECK-NEXT:    adds.w lr, r7, r1
; CHECK-NEXT:    adcs r3, r2
; CHECK-NEXT:    vmov r2, s4
; CHECK-NEXT:    asrl r2, r5, #31
; CHECK-NEXT:    subs.w r7, r2, r12
; CHECK-NEXT:    sbcs r7, r5, #0
; CHECK-NEXT:    mov.w r7, #0
; CHECK-NEXT:    it lt
; CHECK-NEXT:    movlt r7, #1
; CHECK-NEXT:    cmp r7, #0
; CHECK-NEXT:    csetm r7, ne
; CHECK-NEXT:    vmov.32 q0[0], r7
; CHECK-NEXT:    vmov.32 q0[1], r7
; CHECK-NEXT:    vmov r7, s7
; CHECK-NEXT:    asrl r4, r7, #31
; CHECK-NEXT:    subs.w r1, r4, r12
; CHECK-NEXT:    sbcs r1, r7, #0
; CHECK-NEXT:    it lt
; CHECK-NEXT:    movlt r0, #1
; CHECK-NEXT:    cmp r0, #0
; CHECK-NEXT:    csetm r0, ne
; CHECK-NEXT:    vmov.32 q0[2], r0
; CHECK-NEXT:    vmov.32 q0[3], r0
; CHECK-NEXT:    vbic q1, q2, q0
; CHECK-NEXT:    vmov.32 q2[0], r2
; CHECK-NEXT:    vmov.32 q2[1], r5
; CHECK-NEXT:    vmov.32 q2[2], r4
; CHECK-NEXT:    vmov.32 q2[3], r7
; CHECK-NEXT:    vand q0, q2, q0
; CHECK-NEXT:    vorr q0, q0, q1
; CHECK-NEXT:    vmov r1, s0
; CHECK-NEXT:    vmov r0, s1
; CHECK-NEXT:    adds.w r1, r1, lr
; CHECK-NEXT:    adc.w r2, r3, r0
; CHECK-NEXT:    vmov r0, s2
; CHECK-NEXT:    vmov r3, s3
; CHECK-NEXT:    adds r0, r0, r1
; CHECK-NEXT:    adc.w r1, r2, r3
; CHECK-NEXT:    vpop {d8, d9, d10, d11}
; CHECK-NEXT:    pop {r4, r5, r7, pc}
; CHECK-NEXT:    .p2align 4
; CHECK-NEXT:  @ %bb.1:
; CHECK-NEXT:  .LCPI5_0:
; CHECK-NEXT:    .long 2147483647 @ 0x7fffffff
; CHECK-NEXT:    .long 0 @ 0x0
; CHECK-NEXT:    .long 2147483647 @ 0x7fffffff
; CHECK-NEXT:    .long 0 @ 0x0
entry:
  %l2 = sext <4 x i32> %s0 to <4 x i64>
  %l5 = sext <4 x i32> %s1 to <4 x i64>
  %l6 = mul nsw <4 x i64> %l5, %l2
  %l7 = ashr <4 x i64> %l6, <i64 31, i64 31, i64 31, i64 31>
  %l8 = icmp slt <4 x i64> %l7, <i64 2147483647, i64 2147483647, i64 2147483647, i64 2147483647>
  %l9 = select <4 x i1> %l8, <4 x i64> %l7, <4 x i64> <i64 2147483647, i64 2147483647, i64 2147483647, i64 2147483647>
  %l10 = call i64 @llvm.vector.reduce.add.v4i64(<4 x i64> %l9)
  ret i64 %l10
}

define arm_aapcs_vfpcc <4 x i32> @vqdmulh_i32_b(<4 x i32> %s0, <4 x i32> %s1) {
; CHECK-LABEL: vqdmulh_i32_b:
; CHECK:       @ %bb.0: @ %entry
; CHECK-NEXT:    .save {r5, lr}
; CHECK-NEXT:    push {r5, lr}
; CHECK-NEXT:    .vsave {d8, d9, d10, d11}
; CHECK-NEXT:    vpush {d8, d9, d10, d11}
; CHECK-NEXT:    vmov.f32 s8, s2
; CHECK-NEXT:    mvn r12, #-2147483648
; CHECK-NEXT:    vmov.f32 s16, s6
; CHECK-NEXT:    movs r3, #0
; CHECK-NEXT:    vmov.f32 s10, s3
; CHECK-NEXT:    vmov.f32 s18, s7
; CHECK-NEXT:    vmullb.s32 q3, q4, q2
; CHECK-NEXT:    vmov.f32 s2, s1
; CHECK-NEXT:    vmov r5, s13
; CHECK-NEXT:    vmov r2, s12
; CHECK-NEXT:    asrl r2, r5, #31
; CHECK-NEXT:    vmov.f32 s6, s5
; CHECK-NEXT:    subs.w r0, r2, r12
; CHECK-NEXT:    vmov.32 q5[0], r2
; CHECK-NEXT:    sbcs r0, r5, #0
; CHECK-NEXT:    vmov r5, s15
; CHECK-NEXT:    mov.w r0, #0
; CHECK-NEXT:    it lt
; CHECK-NEXT:    movlt r0, #1
; CHECK-NEXT:    cmp r0, #0
; CHECK-NEXT:    csetm r0, ne
; CHECK-NEXT:    vmov.32 q2[0], r0
; CHECK-NEXT:    vmov.32 q2[1], r0
; CHECK-NEXT:    vmov r0, s14
; CHECK-NEXT:    asrl r0, r5, #31
; CHECK-NEXT:    subs.w r1, r0, r12
; CHECK-NEXT:    vmov.32 q5[2], r0
; CHECK-NEXT:    sbcs r1, r5, #0
; CHECK-NEXT:    vmov r0, s0
; CHECK-NEXT:    mov.w r1, #0
; CHECK-NEXT:    it lt
; CHECK-NEXT:    movlt r1, #1
; CHECK-NEXT:    cmp r1, #0
; CHECK-NEXT:    csetm r1, ne
; CHECK-NEXT:    vmov.32 q2[2], r1
; CHECK-NEXT:    adr r1, .LCPI6_0
; CHECK-NEXT:    vldrw.u32 q3, [r1]
; CHECK-NEXT:    vmov r1, s4
; CHECK-NEXT:    vbic q4, q3, q2
; CHECK-NEXT:    vand q2, q5, q2
; CHECK-NEXT:    vorr q2, q2, q4
; CHECK-NEXT:    smull r2, r1, r1, r0
; CHECK-NEXT:    asrl r2, r1, #31
; CHECK-NEXT:    subs.w r0, r2, r12
; CHECK-NEXT:    sbcs r0, r1, #0
; CHECK-NEXT:    vmov r1, s6
; CHECK-NEXT:    mov.w r0, #0
; CHECK-NEXT:    vmov.32 q1[0], r2
; CHECK-NEXT:    it lt
; CHECK-NEXT:    movlt r0, #1
; CHECK-NEXT:    cmp r0, #0
; CHECK-NEXT:    csetm r0, ne
; CHECK-NEXT:    vmov.32 q4[0], r0
; CHECK-NEXT:    vmov.32 q4[1], r0
; CHECK-NEXT:    vmov r0, s2
; CHECK-NEXT:    smull r0, r1, r1, r0
; CHECK-NEXT:    asrl r0, r1, #31
; CHECK-NEXT:    subs.w r5, r0, r12
; CHECK-NEXT:    vmov.32 q1[2], r0
; CHECK-NEXT:    sbcs r1, r1, #0
; CHECK-NEXT:    it lt
; CHECK-NEXT:    movlt r3, #1
; CHECK-NEXT:    cmp r3, #0
; CHECK-NEXT:    csetm r1, ne
; CHECK-NEXT:    vmov.32 q4[2], r1
; CHECK-NEXT:    vbic q0, q3, q4
; CHECK-NEXT:    vand q1, q1, q4
; CHECK-NEXT:    vorr q0, q1, q0
; CHECK-NEXT:    vmov.f32 s1, s2
; CHECK-NEXT:    vmov.f32 s2, s8
; CHECK-NEXT:    vmov.f32 s3, s10
; CHECK-NEXT:    vpop {d8, d9, d10, d11}
; CHECK-NEXT:    pop {r5, pc}
; CHECK-NEXT:    .p2align 4
; CHECK-NEXT:  @ %bb.1:
; CHECK-NEXT:  .LCPI6_0:
; CHECK-NEXT:    .long 2147483647 @ 0x7fffffff
; CHECK-NEXT:    .long 0 @ 0x0
; CHECK-NEXT:    .long 2147483647 @ 0x7fffffff
; CHECK-NEXT:    .long 0 @ 0x0
entry:
  %l2 = sext <4 x i32> %s0 to <4 x i64>
  %l5 = sext <4 x i32> %s1 to <4 x i64>
  %l6 = mul nsw <4 x i64> %l5, %l2
  %l7 = ashr <4 x i64> %l6, <i64 31, i64 31, i64 31, i64 31>
  %l8 = icmp slt <4 x i64> %l7, <i64 2147483647, i64 2147483647, i64 2147483647, i64 2147483647>
  %l9 = select <4 x i1> %l8, <4 x i64> %l7, <4 x i64> <i64 2147483647, i64 2147483647, i64 2147483647, i64 2147483647>
  %l10 = trunc <4 x i64> %l9 to <4 x i32>
  ret <4 x i32> %l10
}




define void @vqdmulh_loop_i8(i8* nocapture readonly %x, i8* nocapture readonly %y, i8* noalias nocapture %z, i32 %n) local_unnamed_addr #0 {
; CHECK-LABEL: vqdmulh_loop_i8:
; CHECK:       @ %bb.0: @ %entry
; CHECK-NEXT:    .save {r7, lr}
; CHECK-NEXT:    push {r7, lr}
; CHECK-NEXT:    mov.w lr, #64
; CHECK-NEXT:    vmov.i32 q0, #0x7f
; CHECK-NEXT:    dls lr, lr
; CHECK-NEXT:  .LBB7_1: @ %vector.body
; CHECK-NEXT:    @ =>This Inner Loop Header: Depth=1
; CHECK-NEXT:    vldrb.s32 q1, [r0, #12]
; CHECK-NEXT:    vldrb.s32 q2, [r1, #12]
; CHECK-NEXT:    vmul.i32 q1, q2, q1
; CHECK-NEXT:    vldrb.s32 q2, [r1, #8]
; CHECK-NEXT:    vshr.s32 q1, q1, #7
; CHECK-NEXT:    vmin.s32 q1, q1, q0
; CHECK-NEXT:    vstrb.32 q1, [r2, #12]
; CHECK-NEXT:    vldrb.s32 q1, [r0, #8]
; CHECK-NEXT:    vmul.i32 q1, q2, q1
; CHECK-NEXT:    vldrb.s32 q2, [r1, #4]
; CHECK-NEXT:    vshr.s32 q1, q1, #7
; CHECK-NEXT:    vmin.s32 q1, q1, q0
; CHECK-NEXT:    vstrb.32 q1, [r2, #8]
; CHECK-NEXT:    vldrb.s32 q1, [r0, #4]
; CHECK-NEXT:    vmul.i32 q1, q2, q1
; CHECK-NEXT:    vldrb.s32 q2, [r1], #16
; CHECK-NEXT:    vshr.s32 q1, q1, #7
; CHECK-NEXT:    vmin.s32 q1, q1, q0
; CHECK-NEXT:    vstrb.32 q1, [r2, #4]
; CHECK-NEXT:    vldrb.s32 q1, [r0], #16
; CHECK-NEXT:    vmul.i32 q1, q2, q1
; CHECK-NEXT:    vshr.s32 q1, q1, #7
; CHECK-NEXT:    vmin.s32 q1, q1, q0
; CHECK-NEXT:    vstrb.32 q1, [r2], #16
; CHECK-NEXT:    le lr, .LBB7_1
; CHECK-NEXT:  @ %bb.2: @ %for.cond.cleanup
; CHECK-NEXT:    pop {r7, pc}
entry:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %entry
  %index = phi i32 [ 0, %entry ], [ %index.next, %vector.body ]
  %0 = getelementptr inbounds i8, i8* %x, i32 %index
  %1 = bitcast i8* %0 to <16 x i8>*
  %wide.load = load <16 x i8>, <16 x i8>* %1, align 1
  %2 = sext <16 x i8> %wide.load to <16 x i32>
  %3 = getelementptr inbounds i8, i8* %y, i32 %index
  %4 = bitcast i8* %3 to <16 x i8>*
  %wide.load26 = load <16 x i8>, <16 x i8>* %4, align 1
  %5 = sext <16 x i8> %wide.load26 to <16 x i32>
  %6 = mul nsw <16 x i32> %5, %2
  %7 = ashr <16 x i32> %6, <i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7>
  %8 = icmp slt <16 x i32> %7, <i32 127, i32 127, i32 127, i32 127, i32 127, i32 127, i32 127, i32 127, i32 127, i32 127, i32 127, i32 127, i32 127, i32 127, i32 127, i32 127>
  %9 = select <16 x i1> %8, <16 x i32> %7, <16 x i32> <i32 127, i32 127, i32 127, i32 127, i32 127, i32 127, i32 127, i32 127, i32 127, i32 127, i32 127, i32 127, i32 127, i32 127, i32 127, i32 127>
  %10 = trunc <16 x i32> %9 to <16 x i8>
  %11 = getelementptr inbounds i8, i8* %z, i32 %index
  %12 = bitcast i8* %11 to <16 x i8>*
  store <16 x i8> %10, <16 x i8>* %12, align 1
  %index.next = add i32 %index, 16
  %13 = icmp eq i32 %index.next, 1024
  br i1 %13, label %for.cond.cleanup, label %vector.body

for.cond.cleanup:                                 ; preds = %vector.body
  ret void
}

define void @vqdmulh_loop_i16(i16* nocapture readonly %x, i16* nocapture readonly %y, i16* noalias nocapture %z, i32 %n) {
; CHECK-LABEL: vqdmulh_loop_i16:
; CHECK:       @ %bb.0: @ %entry
; CHECK-NEXT:    .save {r7, lr}
; CHECK-NEXT:    push {r7, lr}
; CHECK-NEXT:    mov.w lr, #128
; CHECK-NEXT:    vmov.i32 q0, #0x7fff
; CHECK-NEXT:    dls lr, lr
; CHECK-NEXT:  .LBB8_1: @ %vector.body
; CHECK-NEXT:    @ =>This Inner Loop Header: Depth=1
; CHECK-NEXT:    vldrh.s32 q1, [r0, #8]
; CHECK-NEXT:    vldrh.s32 q2, [r1, #8]
; CHECK-NEXT:    vmul.i32 q1, q2, q1
; CHECK-NEXT:    vldrh.s32 q2, [r1], #16
; CHECK-NEXT:    vshr.s32 q1, q1, #15
; CHECK-NEXT:    vmin.s32 q1, q1, q0
; CHECK-NEXT:    vstrh.32 q1, [r2, #8]
; CHECK-NEXT:    vldrh.s32 q1, [r0], #16
; CHECK-NEXT:    vmul.i32 q1, q2, q1
; CHECK-NEXT:    vshr.s32 q1, q1, #15
; CHECK-NEXT:    vmin.s32 q1, q1, q0
; CHECK-NEXT:    vstrh.32 q1, [r2], #16
; CHECK-NEXT:    le lr, .LBB8_1
; CHECK-NEXT:  @ %bb.2: @ %for.cond.cleanup
; CHECK-NEXT:    pop {r7, pc}
entry:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %entry
  %index = phi i32 [ 0, %entry ], [ %index.next, %vector.body ]
  %0 = getelementptr inbounds i16, i16* %x, i32 %index
  %1 = bitcast i16* %0 to <8 x i16>*
  %wide.load = load <8 x i16>, <8 x i16>* %1, align 2
  %2 = sext <8 x i16> %wide.load to <8 x i32>
  %3 = getelementptr inbounds i16, i16* %y, i32 %index
  %4 = bitcast i16* %3 to <8 x i16>*
  %wide.load30 = load <8 x i16>, <8 x i16>* %4, align 2
  %5 = sext <8 x i16> %wide.load30 to <8 x i32>
  %6 = mul nsw <8 x i32> %5, %2
  %7 = ashr <8 x i32> %6, <i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15>
  %8 = icmp slt <8 x i32> %7, <i32 32767, i32 32767, i32 32767, i32 32767, i32 32767, i32 32767, i32 32767, i32 32767>
  %9 = select <8 x i1> %8, <8 x i32> %7, <8 x i32> <i32 32767, i32 32767, i32 32767, i32 32767, i32 32767, i32 32767, i32 32767, i32 32767>
  %10 = trunc <8 x i32> %9 to <8 x i16>
  %11 = getelementptr inbounds i16, i16* %z, i32 %index
  %12 = bitcast i16* %11 to <8 x i16>*
  store <8 x i16> %10, <8 x i16>* %12, align 2
  %index.next = add i32 %index, 8
  %13 = icmp eq i32 %index.next, 1024
  br i1 %13, label %for.cond.cleanup, label %vector.body

for.cond.cleanup:                                 ; preds = %vector.body
  ret void
}

define void @vqdmulh_loop_i32(i32* nocapture readonly %x, i32* nocapture readonly %y, i32* noalias nocapture %z, i32 %n) {
; CHECK-LABEL: vqdmulh_loop_i32:
; CHECK:       @ %bb.0: @ %entry
; CHECK-NEXT:    .save {r4, r5, r6, lr}
; CHECK-NEXT:    push {r4, r5, r6, lr}
; CHECK-NEXT:    .vsave {d8, d9, d10, d11}
; CHECK-NEXT:    vpush {d8, d9, d10, d11}
; CHECK-NEXT:    adr r3, .LCPI9_0
; CHECK-NEXT:    mov.w lr, #256
; CHECK-NEXT:    vldrw.u32 q0, [r3]
; CHECK-NEXT:    mvn r4, #-2147483648
; CHECK-NEXT:    dls lr, lr
; CHECK-NEXT:  .LBB9_1: @ %vector.body
; CHECK-NEXT:    @ =>This Inner Loop Header: Depth=1
; CHECK-NEXT:    vldrw.u32 q1, [r0], #16
; CHECK-NEXT:    vldrw.u32 q2, [r1], #16
; CHECK-NEXT:    vmov.f32 s12, s6
; CHECK-NEXT:    vmov.f32 s16, s10
; CHECK-NEXT:    vmov.f32 s14, s7
; CHECK-NEXT:    vmov.f32 s18, s11
; CHECK-NEXT:    vmullb.s32 q5, q4, q3
; CHECK-NEXT:    vmov.f32 s6, s5
; CHECK-NEXT:    vmov r3, s21
; CHECK-NEXT:    vmov r12, s20
; CHECK-NEXT:    asrl r12, r3, #31
; CHECK-NEXT:    vmov r6, s22
; CHECK-NEXT:    subs.w r5, r12, r4
; CHECK-NEXT:    vmov.f32 s10, s9
; CHECK-NEXT:    sbcs r3, r3, #0
; CHECK-NEXT:    mov.w r3, #0
; CHECK-NEXT:    it lt
; CHECK-NEXT:    movlt r3, #1
; CHECK-NEXT:    cmp r3, #0
; CHECK-NEXT:    csetm r3, ne
; CHECK-NEXT:    vmov.32 q3[0], r3
; CHECK-NEXT:    vmov.32 q3[1], r3
; CHECK-NEXT:    vmov r3, s23
; CHECK-NEXT:    asrl r6, r3, #31
; CHECK-NEXT:    vmov.32 q5[0], r12
; CHECK-NEXT:    subs r5, r6, r4
; CHECK-NEXT:    vmov.32 q5[2], r6
; CHECK-NEXT:    sbcs r3, r3, #0
; CHECK-NEXT:    vmov r6, s8
; CHECK-NEXT:    mov.w r3, #0
; CHECK-NEXT:    it lt
; CHECK-NEXT:    movlt r3, #1
; CHECK-NEXT:    cmp r3, #0
; CHECK-NEXT:    csetm r3, ne
; CHECK-NEXT:    vmov.32 q3[2], r3
; CHECK-NEXT:    vmov r3, s4
; CHECK-NEXT:    vbic q4, q0, q3
; CHECK-NEXT:    vand q3, q5, q3
; CHECK-NEXT:    vorr q3, q3, q4
; CHECK-NEXT:    smull r12, r3, r6, r3
; CHECK-NEXT:    asrl r12, r3, #31
; CHECK-NEXT:    subs.w r5, r12, r4
; CHECK-NEXT:    sbcs r3, r3, #0
; CHECK-NEXT:    vmov r5, s10
; CHECK-NEXT:    mov.w r3, #0
; CHECK-NEXT:    vmov.32 q2[0], r12
; CHECK-NEXT:    it lt
; CHECK-NEXT:    movlt r3, #1
; CHECK-NEXT:    cmp r3, #0
; CHECK-NEXT:    csetm r3, ne
; CHECK-NEXT:    vmov.32 q4[0], r3
; CHECK-NEXT:    vmov.32 q4[1], r3
; CHECK-NEXT:    vmov r3, s6
; CHECK-NEXT:    smull r6, r3, r5, r3
; CHECK-NEXT:    asrl r6, r3, #31
; CHECK-NEXT:    subs r5, r6, r4
; CHECK-NEXT:    vmov.32 q2[2], r6
; CHECK-NEXT:    sbcs r3, r3, #0
; CHECK-NEXT:    mov.w r3, #0
; CHECK-NEXT:    it lt
; CHECK-NEXT:    movlt r3, #1
; CHECK-NEXT:    cmp r3, #0
; CHECK-NEXT:    csetm r3, ne
; CHECK-NEXT:    vmov.32 q4[2], r3
; CHECK-NEXT:    vbic q1, q0, q4
; CHECK-NEXT:    vand q2, q2, q4
; CHECK-NEXT:    vorr q1, q2, q1
; CHECK-NEXT:    vmov.f32 s5, s6
; CHECK-NEXT:    vmov.f32 s6, s12
; CHECK-NEXT:    vmov.f32 s7, s14
; CHECK-NEXT:    vstrb.8 q1, [r2], #16
; CHECK-NEXT:    le lr, .LBB9_1
; CHECK-NEXT:  @ %bb.2: @ %for.cond.cleanup
; CHECK-NEXT:    vpop {d8, d9, d10, d11}
; CHECK-NEXT:    pop {r4, r5, r6, pc}
; CHECK-NEXT:    .p2align 4
; CHECK-NEXT:  @ %bb.3:
; CHECK-NEXT:  .LCPI9_0:
; CHECK-NEXT:    .long 2147483647 @ 0x7fffffff
; CHECK-NEXT:    .long 0 @ 0x0
; CHECK-NEXT:    .long 2147483647 @ 0x7fffffff
; CHECK-NEXT:    .long 0 @ 0x0
entry:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %entry
  %index = phi i32 [ 0, %entry ], [ %index.next, %vector.body ]
  %0 = getelementptr inbounds i32, i32* %x, i32 %index
  %1 = bitcast i32* %0 to <4 x i32>*
  %wide.load = load <4 x i32>, <4 x i32>* %1, align 4
  %2 = sext <4 x i32> %wide.load to <4 x i64>
  %3 = getelementptr inbounds i32, i32* %y, i32 %index
  %4 = bitcast i32* %3 to <4 x i32>*
  %wide.load30 = load <4 x i32>, <4 x i32>* %4, align 4
  %5 = sext <4 x i32> %wide.load30 to <4 x i64>
  %6 = mul nsw <4 x i64> %5, %2
  %7 = ashr <4 x i64> %6, <i64 31, i64 31, i64 31, i64 31>
  %8 = icmp slt <4 x i64> %7, <i64 2147483647, i64 2147483647, i64 2147483647, i64 2147483647>
  %9 = select <4 x i1> %8, <4 x i64> %7, <4 x i64> <i64 2147483647, i64 2147483647, i64 2147483647, i64 2147483647>
  %10 = trunc <4 x i64> %9 to <4 x i32>
  %11 = getelementptr inbounds i32, i32* %z, i32 %index
  %12 = bitcast i32* %11 to <4 x i32>*
  store <4 x i32> %10, <4 x i32>* %12, align 4
  %index.next = add i32 %index, 4
  %13 = icmp eq i32 %index.next, 1024
  br i1 %13, label %for.cond.cleanup, label %vector.body

for.cond.cleanup:                                 ; preds = %vector.body
  ret void
}

declare i64 @llvm.vector.reduce.add.v4i64(<4 x i64>)
declare i32 @llvm.vector.reduce.add.v8i32(<8 x i32>)
declare i32 @llvm.vector.reduce.add.v16i32(<16 x i32>)