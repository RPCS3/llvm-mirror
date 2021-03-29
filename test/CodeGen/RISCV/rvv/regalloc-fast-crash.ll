; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py
; RUN: llc -mtriple=riscv32 -mattr=+d,+experimental-zvlsseg,+experimental-zfh,+m \
; RUN:     -regalloc=fast -verify-machineinstrs < %s | FileCheck %s

; This test previously crashed with an error "ran out of registers during register allocation"

declare void @llvm.riscv.vsseg2.mask.nxv16i16(<vscale x 16 x i16>,<vscale x 16 x i16>, i16*, <vscale x 16 x i1>, i32)

define void @test_vsseg2_mask_nxv16i16(<vscale x 16 x i16> %val, i16* %base, <vscale x 16 x i1> %mask, i32 %vl) {
; CHECK-LABEL: test_vsseg2_mask_nxv16i16:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    vmv4r.v v4, v8
; CHECK-NEXT:    vsetvli a1, a1, e16,m4,ta,mu
; CHECK-NEXT:    vsseg2e16.v v4, (a0), v0.t
; CHECK-NEXT:    ret
entry:
  tail call void @llvm.riscv.vsseg2.mask.nxv16i16(<vscale x 16 x i16> %val,<vscale x 16 x i16> %val, i16* %base, <vscale x 16 x i1> %mask, i32 %vl)
  ret void
}
