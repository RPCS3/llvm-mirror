; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py
; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+sve < %s 2>%t | FileCheck %s
; RUN: FileCheck --check-prefix=WARN --allow-empty %s <%t

; If this check fails please read test/CodeGen/AArch64/README for instructions on how to resolve it.
; WARN-NOT: warning

; LD1B

define <vscale x 16 x i8> @ld1b_lower_bound(<vscale x 16 x i8>* %a) {
; CHECK-LABEL: ld1b_lower_bound:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ptrue p0.b
; CHECK-NEXT:    ld1b { z0.b }, p0/z, [x0, #-8, mul vl]
; CHECK-NEXT:    ret
  %base = getelementptr <vscale x 16 x i8>, <vscale x 16 x i8>* %a, i64 -8
  %load = load <vscale x 16 x i8>, <vscale x 16 x i8>* %base
  ret <vscale x 16 x i8> %load
}

define <vscale x 16 x i8> @ld1b_inbound(<vscale x 16 x i8>* %a) {
; CHECK-LABEL: ld1b_inbound:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ptrue p0.b
; CHECK-NEXT:    ld1b { z0.b }, p0/z, [x0, #2, mul vl]
; CHECK-NEXT:    ret
  %base = getelementptr <vscale x 16 x i8>, <vscale x 16 x i8>* %a, i64 2
  %load = load <vscale x 16 x i8>, <vscale x 16 x i8>* %base
  ret <vscale x 16 x i8> %load
}

define <vscale x 16 x i8> @ld1b_upper_bound(<vscale x 16 x i8>* %a) {
; CHECK-LABEL: ld1b_upper_bound:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ptrue p0.b
; CHECK-NEXT:    ld1b { z0.b }, p0/z, [x0, #7, mul vl]
; CHECK-NEXT:    ret
  %base = getelementptr <vscale x 16 x i8>, <vscale x 16 x i8>* %a, i64 7
  %load = load <vscale x 16 x i8>, <vscale x 16 x i8>* %base
  ret <vscale x 16 x i8> %load
}

define <vscale x 16 x i8> @ld1b_out_of_upper_bound(<vscale x 16 x i8>* %a) {
; CHECK-LABEL: ld1b_out_of_upper_bound:
; CHECK:       // %bb.0:
; CHECK-NEXT:    rdvl x8, #8
; CHECK-NEXT:    ptrue p0.b
; CHECK-NEXT:    ld1b { z0.b }, p0/z, [x0, x8]
; CHECK-NEXT:    ret
  %base = getelementptr <vscale x 16 x i8>, <vscale x 16 x i8>* %a, i64 8
  %load = load <vscale x 16 x i8>, <vscale x 16 x i8>* %base
  ret <vscale x 16 x i8> %load
}

define <vscale x 16 x i8> @ld1b_out_of_lower_bound(<vscale x 16 x i8>* %a) {
; CHECK-LABEL: ld1b_out_of_lower_bound:
; CHECK:       // %bb.0:
; CHECK-NEXT:    rdvl x8, #-9
; CHECK-NEXT:    ptrue p0.b
; CHECK-NEXT:    ld1b { z0.b }, p0/z, [x0, x8]
; CHECK-NEXT:    ret
  %base = getelementptr <vscale x 16 x i8>, <vscale x 16 x i8>* %a, i64 -9
  %load = load <vscale x 16 x i8>, <vscale x 16 x i8>* %base
  ret <vscale x 16 x i8> %load
}

; LD1H

define <vscale x 8 x i16> @ld1h_inbound(<vscale x 8 x i16>* %a) {
; CHECK-LABEL: ld1h_inbound:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ptrue p0.h
; CHECK-NEXT:    ld1h { z0.h }, p0/z, [x0, #-2, mul vl]
; CHECK-NEXT:    ret
  %base = getelementptr <vscale x 8 x i16>, <vscale x 8 x i16>* %a, i64 -2
  %load = load <vscale x 8 x i16>, <vscale x 8 x i16>* %base
  ret <vscale x 8 x i16> %load
}

; LD1W

define <vscale x 4 x i32> @ld1s_inbound(<vscale x 4 x i32>* %a) {
; CHECK-LABEL: ld1s_inbound:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ptrue p0.s
; CHECK-NEXT:    ld1w { z0.s }, p0/z, [x0, #4, mul vl]
; CHECK-NEXT:    ret
  %base = getelementptr <vscale x 4 x i32>, <vscale x 4 x i32>* %a, i64 4
  %load = load <vscale x 4 x i32>, <vscale x 4 x i32>* %base
  ret <vscale x 4 x i32> %load
}

; LD1D

define <vscale x 2 x i64> @ld1d_inbound(<vscale x 2 x i64>* %a) {
; CHECK-LABEL: ld1d_inbound:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ptrue p0.d
; CHECK-NEXT:    ld1d { z0.d }, p0/z, [x0, #6, mul vl]
; CHECK-NEXT:    ret
  %base = getelementptr <vscale x 2 x i64>, <vscale x 2 x i64>* %a, i64 6
  %load = load <vscale x 2 x i64>, <vscale x 2 x i64>* %base
  ret <vscale x 2 x i64> %load
}

define void @load_nxv6f16(<vscale x 6 x half>* %a) {
; CHECK-LABEL: load_nxv6f16:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ptrue p0.d
; CHECK-NEXT:    ptrue p1.s
; CHECK-NEXT:    ld1h { z0.d }, p0/z, [x0, #2, mul vl]
; CHECK-NEXT:    ld1h { z0.s }, p1/z, [x0]
; CHECK-NEXT:    ret
  %val = load volatile <vscale x 6 x half>, <vscale x 6 x half>* %a
  ret void
}

define void @load_nxv6f32(<vscale x 6 x float>* %a) {
; CHECK-LABEL: load_nxv6f32:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ptrue p0.d
; CHECK-NEXT:    ptrue p1.s
; CHECK-NEXT:    ld1w { z0.d }, p0/z, [x0, #2, mul vl]
; CHECK-NEXT:    ld1w { z0.s }, p1/z, [x0]
; CHECK-NEXT:    ret
  %val = load volatile <vscale x 6 x float>, <vscale x 6 x float>* %a
  ret void
}

define void @load_nxv12f16(<vscale x 12 x half>* %a) {
; CHECK-LABEL: load_nxv12f16:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ptrue p0.s
; CHECK-NEXT:    ptrue p1.h
; CHECK-NEXT:    ld1h { z0.s }, p0/z, [x0, #2, mul vl]
; CHECK-NEXT:    ld1h { z0.h }, p1/z, [x0]
; CHECK-NEXT:    ret
  %val = load volatile <vscale x 12 x half>, <vscale x 12 x half>* %a
  ret void
}
