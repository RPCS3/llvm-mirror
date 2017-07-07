; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py
; RUN: llc -mtriple=powerpc64-unknown-linux-gnu -mcpu=pwr8 < %s | FileCheck %s
; RUN: llc -mtriple=powerpc64le-unknown-linux-gnu -mcpu=pwr8 < %s | FileCheck %s

define zeroext i32 @ReverseBits(i32 zeroext %n) {
; CHECK-LABEL: ReverseBits:
; CHECK:       # BB#0: # %entry
; CHECK-NEXT:    lis 4, -21846
; CHECK-NEXT:    lis 5, 21845
; CHECK-NEXT:    slwi 6, 3, 1
; CHECK-NEXT:    srwi 3, 3, 1
; CHECK-NEXT:    lis 7, -13108
; CHECK-NEXT:    lis 8, 13107
; CHECK-NEXT:    ori 4, 4, 43690
; CHECK-NEXT:    ori 5, 5, 21845
; CHECK-NEXT:    lis 10, -3856
; CHECK-NEXT:    lis 11, 3855
; CHECK-NEXT:    and 3, 3, 5
; CHECK-NEXT:    and 4, 6, 4
; CHECK-NEXT:    ori 5, 8, 13107
; CHECK-NEXT:    or 3, 3, 4
; CHECK-NEXT:    ori 4, 7, 52428
; CHECK-NEXT:    slwi 9, 3, 2
; CHECK-NEXT:    srwi 3, 3, 2
; CHECK-NEXT:    and 3, 3, 5
; CHECK-NEXT:    and 4, 9, 4
; CHECK-NEXT:    ori 5, 11, 3855
; CHECK-NEXT:    or 3, 3, 4
; CHECK-NEXT:    ori 4, 10, 61680
; CHECK-NEXT:    slwi 12, 3, 4
; CHECK-NEXT:    srwi 3, 3, 4
; CHECK-NEXT:    and 4, 12, 4
; CHECK-NEXT:    and 3, 3, 5
; CHECK-NEXT:    or 3, 3, 4
; CHECK-NEXT:    rotlwi 4, 3, 24
; CHECK-NEXT:    rlwimi 4, 3, 8, 8, 15
; CHECK-NEXT:    rlwimi 4, 3, 8, 24, 31
; CHECK-NEXT:    rldicl 3, 4, 0, 32
; CHECK-NEXT:    clrldi 3, 3, 32
; CHECK-NEXT:    blr
entry:
  %shr = lshr i32 %n, 1
  %and = and i32 %shr, 1431655765
  %and1 = shl i32 %n, 1
  %shl = and i32 %and1, -1431655766
  %or = or i32 %and, %shl
  %shr2 = lshr i32 %or, 2
  %and3 = and i32 %shr2, 858993459
  %and4 = shl i32 %or, 2
  %shl5 = and i32 %and4, -858993460
  %or6 = or i32 %and3, %shl5
  %shr7 = lshr i32 %or6, 4
  %and8 = and i32 %shr7, 252645135
  %and9 = shl i32 %or6, 4
  %shl10 = and i32 %and9, -252645136
  %or11 = or i32 %and8, %shl10
  %shr13 = lshr i32 %or11, 24
  %and14 = lshr i32 %or11, 8
  %shr15 = and i32 %and14, 65280
  %and17 = shl i32 %or11, 8
  %shl18 = and i32 %and17, 16711680
  %shl21 = shl i32 %or11, 24
  %or16 = or i32 %shl21, %shr13
  %or19 = or i32 %or16, %shr15
  %or22 = or i32 %or19, %shl18
  ret i32 %or22
}
