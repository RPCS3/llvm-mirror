; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py
; RUN: llc -mtriple thumbv8m.main-arm-unknown-eabi --float-abi=soft -mattr=+vfp4d16sp < %s | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-VFPV4-SOFT
; RUN: llc -mtriple thumbv8.1m.main-arm-unknown-eabi --float-abi=soft -mattr=+fullfp16 < %s | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-FP16-SOFT
; RUN: llc -mtriple thumbv8m.main-arm-unknown-eabi --float-abi=hard -mattr=+vfp4d16sp < %s | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-VFPV4-HARD
; RUN: llc -mtriple thumbv8.1m.main-arm-unknown-eabi --float-abi=hard -mattr=+fullfp16 < %s | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-FP16-HARD

target triple = "thumbv8.1m.main-arm-unknown-eabi"

define float @add(float %a, float %b) {
; CHECK-VFPV4-SOFT-LABEL: add:
; CHECK-VFPV4-SOFT:       @ %bb.0: @ %entry
; CHECK-VFPV4-SOFT-NEXT:    vmov s0, r1
; CHECK-VFPV4-SOFT-NEXT:    vmov s2, r0
; CHECK-VFPV4-SOFT-NEXT:    vadd.f32 s0, s2, s0
; CHECK-VFPV4-SOFT-NEXT:    vmov r0, s0
; CHECK-VFPV4-SOFT-NEXT:    bx lr
;
; CHECK-FP16-SOFT-LABEL: add:
; CHECK-FP16-SOFT:       @ %bb.0: @ %entry
; CHECK-FP16-SOFT-NEXT:    vmov s0, r1
; CHECK-FP16-SOFT-NEXT:    vmov s2, r0
; CHECK-FP16-SOFT-NEXT:    vadd.f32 s0, s2, s0
; CHECK-FP16-SOFT-NEXT:    vmov r0, s0
; CHECK-FP16-SOFT-NEXT:    bx lr
;
; CHECK-VFPV4-HARD-LABEL: add:
; CHECK-VFPV4-HARD:       @ %bb.0: @ %entry
; CHECK-VFPV4-HARD-NEXT:    vadd.f32 s0, s0, s1
; CHECK-VFPV4-HARD-NEXT:    bx lr
;
; CHECK-FP16-HARD-LABEL: add:
; CHECK-FP16-HARD:       @ %bb.0: @ %entry
; CHECK-FP16-HARD-NEXT:    vadd.f32 s0, s0, s1
; CHECK-FP16-HARD-NEXT:    bx lr
entry:
  %add = fadd float %a, %b
  ret float %add
}

define half @addf16(half %a, half %b) {
; CHECK-VFPV4-SOFT-LABEL: addf16:
; CHECK-VFPV4-SOFT:       @ %bb.0: @ %entry
; CHECK-VFPV4-SOFT-NEXT:    vmov s0, r0
; CHECK-VFPV4-SOFT-NEXT:    vmov s2, r1
; CHECK-VFPV4-SOFT-NEXT:    vcvtb.f32.f16 s0, s0
; CHECK-VFPV4-SOFT-NEXT:    vcvtb.f32.f16 s2, s2
; CHECK-VFPV4-SOFT-NEXT:    vadd.f32 s0, s0, s2
; CHECK-VFPV4-SOFT-NEXT:    vcvtb.f16.f32 s0, s0
; CHECK-VFPV4-SOFT-NEXT:    vmov r0, s0
; CHECK-VFPV4-SOFT-NEXT:    bx lr
;
; CHECK-FP16-SOFT-LABEL: addf16:
; CHECK-FP16-SOFT:       @ %bb.0: @ %entry
; CHECK-FP16-SOFT-NEXT:    vmov.f16 s0, r1
; CHECK-FP16-SOFT-NEXT:    vmov.f16 s2, r0
; CHECK-FP16-SOFT-NEXT:    vadd.f16 s0, s2, s0
; CHECK-FP16-SOFT-NEXT:    vmov r0, s0
; CHECK-FP16-SOFT-NEXT:    bx lr
;
; CHECK-VFPV4-HARD-LABEL: addf16:
; CHECK-VFPV4-HARD:       @ %bb.0: @ %entry
; CHECK-VFPV4-HARD-NEXT:    vcvtb.f32.f16 s2, s1
; CHECK-VFPV4-HARD-NEXT:    vcvtb.f32.f16 s0, s0
; CHECK-VFPV4-HARD-NEXT:    vadd.f32 s0, s0, s2
; CHECK-VFPV4-HARD-NEXT:    vcvtb.f16.f32 s0, s0
; CHECK-VFPV4-HARD-NEXT:    bx lr
;
; CHECK-FP16-HARD-LABEL: addf16:
; CHECK-FP16-HARD:       @ %bb.0: @ %entry
; CHECK-FP16-HARD-NEXT:    vadd.f16 s0, s0, s1
; CHECK-FP16-HARD-NEXT:    bx lr
entry:
  %add = fadd half %a, %b
  ret half %add
}

define half @load_i16(i16 *%hp) {
; CHECK-VFPV4-SOFT-LABEL: load_i16:
; CHECK-VFPV4-SOFT:       @ %bb.0: @ %entry
; CHECK-VFPV4-SOFT-NEXT:    vmov.f32 s0, #1.000000e+00
; CHECK-VFPV4-SOFT-NEXT:    ldrh r0, [r0]
; CHECK-VFPV4-SOFT-NEXT:    vmov s2, r0
; CHECK-VFPV4-SOFT-NEXT:    vcvtb.f32.f16 s2, s2
; CHECK-VFPV4-SOFT-NEXT:    vadd.f32 s0, s2, s0
; CHECK-VFPV4-SOFT-NEXT:    vcvtb.f16.f32 s0, s0
; CHECK-VFPV4-SOFT-NEXT:    vmov r0, s0
; CHECK-VFPV4-SOFT-NEXT:    bx lr
;
; CHECK-FP16-SOFT-LABEL: load_i16:
; CHECK-FP16-SOFT:       @ %bb.0: @ %entry
; CHECK-FP16-SOFT-NEXT:    vldr.16 s2, [r0]
; CHECK-FP16-SOFT-NEXT:    vmov.f16 s0, #1.000000e+00
; CHECK-FP16-SOFT-NEXT:    vadd.f16 s0, s2, s0
; CHECK-FP16-SOFT-NEXT:    vmov r0, s0
; CHECK-FP16-SOFT-NEXT:    bx lr
;
; CHECK-VFPV4-HARD-LABEL: load_i16:
; CHECK-VFPV4-HARD:       @ %bb.0: @ %entry
; CHECK-VFPV4-HARD-NEXT:    vmov.f32 s0, #1.000000e+00
; CHECK-VFPV4-HARD-NEXT:    ldrh r0, [r0]
; CHECK-VFPV4-HARD-NEXT:    vmov s2, r0
; CHECK-VFPV4-HARD-NEXT:    vcvtb.f32.f16 s2, s2
; CHECK-VFPV4-HARD-NEXT:    vadd.f32 s0, s2, s0
; CHECK-VFPV4-HARD-NEXT:    vcvtb.f16.f32 s0, s0
; CHECK-VFPV4-HARD-NEXT:    bx lr
;
; CHECK-FP16-HARD-LABEL: load_i16:
; CHECK-FP16-HARD:       @ %bb.0: @ %entry
; CHECK-FP16-HARD-NEXT:    vldr.16 s2, [r0]
; CHECK-FP16-HARD-NEXT:    vmov.f16 s0, #1.000000e+00
; CHECK-FP16-HARD-NEXT:    vadd.f16 s0, s2, s0
; CHECK-FP16-HARD-NEXT:    bx lr
entry:
  %h = load i16, i16 *%hp, align 2
  %hc = bitcast i16 %h to half
  %add = fadd half %hc, 1.0
  ret half %add
}

define i16 @load_f16(half *%hp) {
; CHECK-LABEL: load_f16:
; CHECK:       @ %bb.0: @ %entry
; CHECK-NEXT:    ldrh r0, [r0]
; CHECK-NEXT:    adds r0, #1
; CHECK-NEXT:    bx lr
entry:
  %h = load half, half *%hp, align 2
  %hc = bitcast half %h to i16
  %add = add i16 %hc, 1
  ret i16 %add
}

define half @constcall() {
; CHECK-VFPV4-SOFT-LABEL: constcall:
; CHECK-VFPV4-SOFT:       @ %bb.0: @ %entry
; CHECK-VFPV4-SOFT-NEXT:    mov.w r0, #18688
; CHECK-VFPV4-SOFT-NEXT:    b ccc
;
; CHECK-FP16-SOFT-LABEL: constcall:
; CHECK-FP16-SOFT:       @ %bb.0: @ %entry
; CHECK-FP16-SOFT-NEXT:    mov.w r0, #18688
; CHECK-FP16-SOFT-NEXT:    b ccc
;
; CHECK-VFPV4-HARD-LABEL: constcall:
; CHECK-VFPV4-HARD:       @ %bb.0: @ %entry
; CHECK-VFPV4-HARD-NEXT:    vldr s0, .LCPI4_0
; CHECK-VFPV4-HARD-NEXT:    b ccc
; CHECK-VFPV4-HARD-NEXT:    .p2align 2
; CHECK-VFPV4-HARD-NEXT:  @ %bb.1:
; CHECK-VFPV4-HARD-NEXT:  .LCPI4_0:
; CHECK-VFPV4-HARD-NEXT:    .long 0x00004900 @ float 2.61874657E-41
;
; CHECK-FP16-HARD-LABEL: constcall:
; CHECK-FP16-HARD:       @ %bb.0: @ %entry
; CHECK-FP16-HARD-NEXT:    vldr s0, .LCPI4_0
; CHECK-FP16-HARD-NEXT:    b ccc
; CHECK-FP16-HARD-NEXT:    .p2align 2
; CHECK-FP16-HARD-NEXT:  @ %bb.1:
; CHECK-FP16-HARD-NEXT:  .LCPI4_0:
; CHECK-FP16-HARD-NEXT:    .long 0x00004900 @ float 2.61874657E-41
entry:
  %call = tail call fast half @ccc(half 0xH4900)
  ret half %call
}

define half @constret() {
; CHECK-VFPV4-SOFT-LABEL: constret:
; CHECK-VFPV4-SOFT:       @ %bb.0: @ %entry
; CHECK-VFPV4-SOFT-NEXT:    mov.w r0, #18688
; CHECK-VFPV4-SOFT-NEXT:    bx lr
;
; CHECK-FP16-SOFT-LABEL: constret:
; CHECK-FP16-SOFT:       @ %bb.0: @ %entry
; CHECK-FP16-SOFT-NEXT:    vmov.f16 s0, #1.000000e+01
; CHECK-FP16-SOFT-NEXT:    vmov r0, s0
; CHECK-FP16-SOFT-NEXT:    bx lr
;
; CHECK-VFPV4-HARD-LABEL: constret:
; CHECK-VFPV4-HARD:       @ %bb.0: @ %entry
; CHECK-VFPV4-HARD-NEXT:    vldr s0, .LCPI5_0
; CHECK-VFPV4-HARD-NEXT:    bx lr
; CHECK-VFPV4-HARD-NEXT:    .p2align 2
; CHECK-VFPV4-HARD-NEXT:  @ %bb.1:
; CHECK-VFPV4-HARD-NEXT:  .LCPI5_0:
; CHECK-VFPV4-HARD-NEXT:    .long 0x00004900 @ float 2.61874657E-41
;
; CHECK-FP16-HARD-LABEL: constret:
; CHECK-FP16-HARD:       @ %bb.0: @ %entry
; CHECK-FP16-HARD-NEXT:    vmov.f16 s0, #1.000000e+01
; CHECK-FP16-HARD-NEXT:    bx lr
entry:
  ret half 0xH4900
}

declare half @ccc(half)
