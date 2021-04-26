; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py

; RUN: llc -mtriple=riscv32 -target-abi=ilp32d -mattr=+experimental-v,+experimental-zfh,+f,+d -verify-machineinstrs -riscv-v-vector-bits-min=128 -riscv-v-fixed-length-vector-lmul-max=2 -verify-machineinstrs < %s | FileCheck %s --check-prefixes=CHECK
; RUN: llc -mtriple=riscv64 -target-abi=lp64d -mattr=+experimental-v,+experimental-zfh,+f,+d -verify-machineinstrs -riscv-v-vector-bits-min=128 -riscv-v-fixed-length-vector-lmul-max=2 -verify-machineinstrs < %s | FileCheck %s --check-prefixes=CHECK
; RUN: llc -mtriple=riscv32 -target-abi=ilp32d -mattr=+experimental-v,+experimental-zfh,+f,+d -verify-machineinstrs -riscv-v-vector-bits-min=128 -riscv-v-fixed-length-vector-lmul-max=1 -verify-machineinstrs < %s | FileCheck %s --check-prefixes=CHECK
; RUN: llc -mtriple=riscv64 -target-abi=lp64d -mattr=+experimental-v,+experimental-zfh,+f,+d -verify-machineinstrs -riscv-v-vector-bits-min=128 -riscv-v-fixed-length-vector-lmul-max=1 -verify-machineinstrs < %s | FileCheck %s --check-prefixes=CHECK

; Tests that a floating-point build_vector doesn't try and generate a VID
; instruction
define void @buildvec_no_vid_v4f32(<4 x float>* %x) {
; CHECK-LABEL: buildvec_no_vid_v4f32:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lui a1, %hi(.LCPI0_0)
; CHECK-NEXT:    addi a1, a1, %lo(.LCPI0_0)
; CHECK-NEXT:    vsetivli a2, 4, e32,m1,ta,mu
; CHECK-NEXT:    vle32.v v25, (a1)
; CHECK-NEXT:    vse32.v v25, (a0)
; CHECK-NEXT:    ret
  store <4 x float> <float 0.0, float 4.0, float 0.0, float 2.0>, <4 x float>* %x
  ret void
}

define void @buildvec_dominant0_v4f32(<4 x float>* %x) {
; CHECK-LABEL: buildvec_dominant0_v4f32:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetivli a1, 4, e32,m1,ta,mu
; CHECK-NEXT:    lui a1, %hi(.LCPI1_0)
; CHECK-NEXT:    addi a1, a1, %lo(.LCPI1_0)
; CHECK-NEXT:    vlse32.v v25, (a1), zero
; CHECK-NEXT:    fmv.w.x ft0, zero
; CHECK-NEXT:    vfmv.s.f v26, ft0
; CHECK-NEXT:    vsetivli a1, 3, e32,m1,tu,mu
; CHECK-NEXT:    vslideup.vi v25, v26, 2
; CHECK-NEXT:    vsetivli a1, 4, e32,m1,ta,mu
; CHECK-NEXT:    vse32.v v25, (a0)
; CHECK-NEXT:    ret
  store <4 x float> <float 2.0, float 2.0, float 0.0, float 2.0>, <4 x float>* %x
  ret void
}

define void @buildvec_dominant1_v4f32(<4 x float>* %x, float %f) {
; CHECK-LABEL: buildvec_dominant1_v4f32:
; CHECK:       # %bb.0:
; CHECK-NEXT:    fmv.w.x ft0, zero
; CHECK-NEXT:    vsetivli a1, 4, e32,m1,ta,mu
; CHECK-NEXT:    vfmv.s.f v25, ft0
; CHECK-NEXT:    vfmv.v.f v26, fa0
; CHECK-NEXT:    vsetivli a1, 2, e32,m1,tu,mu
; CHECK-NEXT:    vslideup.vi v26, v25, 1
; CHECK-NEXT:    vsetivli a1, 4, e32,m1,ta,mu
; CHECK-NEXT:    vse32.v v26, (a0)
; CHECK-NEXT:    ret
  %v0 = insertelement <4 x float> undef, float %f, i32 0
  %v1 = insertelement <4 x float> %v0, float 0.0, i32 1
  %v2 = insertelement <4 x float> %v1, float %f, i32 2
  %v3 = insertelement <4 x float> %v2, float %f, i32 3
  store <4 x float> %v3, <4 x float>* %x
  ret void
}

define void @buildvec_dominant2_v4f32(<4 x float>* %x, float %f) {
; CHECK-LABEL: buildvec_dominant2_v4f32:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lui a1, %hi(.LCPI3_0)
; CHECK-NEXT:    flw ft0, %lo(.LCPI3_0)(a1)
; CHECK-NEXT:    vsetivli a1, 4, e32,m1,ta,mu
; CHECK-NEXT:    vfmv.s.f v25, ft0
; CHECK-NEXT:    vfmv.v.f v26, fa0
; CHECK-NEXT:    vsetivli a1, 2, e32,m1,tu,mu
; CHECK-NEXT:    vslideup.vi v26, v25, 1
; CHECK-NEXT:    vsetivli a1, 4, e32,m1,ta,mu
; CHECK-NEXT:    vse32.v v26, (a0)
; CHECK-NEXT:    ret
  %v0 = insertelement <4 x float> undef, float %f, i32 0
  %v1 = insertelement <4 x float> %v0, float 2.0, i32 1
  %v2 = insertelement <4 x float> %v1, float %f, i32 2
  %v3 = insertelement <4 x float> %v2, float %f, i32 3
  store <4 x float> %v3, <4 x float>* %x
  ret void
}

define void @buildvec_merge0_v4f32(<4 x float>* %x, float %f) {
; CHECK-LABEL: buildvec_merge0_v4f32:
; CHECK:       # %bb.0:
; CHECK-NEXT:    addi a1, zero, 6
; CHECK-NEXT:    vsetivli a2, 1, e8,m1,ta,mu
; CHECK-NEXT:    lui a2, %hi(.LCPI4_0)
; CHECK-NEXT:    flw ft0, %lo(.LCPI4_0)(a2)
; CHECK-NEXT:    vmv.s.x v0, a1
; CHECK-NEXT:    vsetivli a1, 4, e32,m1,ta,mu
; CHECK-NEXT:    vfmv.v.f v25, fa0
; CHECK-NEXT:    vfmerge.vfm v25, v25, ft0, v0
; CHECK-NEXT:    vse32.v v25, (a0)
; CHECK-NEXT:    ret
  %v0 = insertelement <4 x float> undef, float %f, i32 0
  %v1 = insertelement <4 x float> %v0, float 2.0, i32 1
  %v2 = insertelement <4 x float> %v1, float 2.0, i32 2
  %v3 = insertelement <4 x float> %v2, float %f, i32 3
  store <4 x float> %v3, <4 x float>* %x
  ret void
}
