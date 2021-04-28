; NOTE: Assertions have been autogenerated by utils/update_test_checks.py
; RUN: opt -instcombine -S < %s | FileCheck %s

define i16 @sext_shl_trunc_same_size(i16 %x, i32 %y) {
; CHECK-LABEL: @sext_shl_trunc_same_size(
; CHECK-NEXT:    [[CONV:%.*]] = sext i16 [[X:%.*]] to i32
; CHECK-NEXT:    [[SHL:%.*]] = shl i32 [[CONV]], [[Y:%.*]]
; CHECK-NEXT:    [[T:%.*]] = trunc i32 [[SHL]] to i16
; CHECK-NEXT:    ret i16 [[T]]
;
  %conv = sext i16 %x to i32
  %shl = shl i32 %conv, %y
  %t = trunc i32 %shl to i16
  ret i16 %t
}

define i5 @sext_shl_trunc_smaller(i16 %x, i32 %y) {
; CHECK-LABEL: @sext_shl_trunc_smaller(
; CHECK-NEXT:    [[CONV:%.*]] = sext i16 [[X:%.*]] to i32
; CHECK-NEXT:    [[SHL:%.*]] = shl i32 [[CONV]], [[Y:%.*]]
; CHECK-NEXT:    [[T:%.*]] = trunc i32 [[SHL]] to i5
; CHECK-NEXT:    ret i5 [[T]]
;
  %conv = sext i16 %x to i32
  %shl = shl i32 %conv, %y
  %t = trunc i32 %shl to i5
  ret i5 %t
}

define i17 @sext_shl_trunc_larger(i16 %x, i32 %y) {
; CHECK-LABEL: @sext_shl_trunc_larger(
; CHECK-NEXT:    [[CONV:%.*]] = sext i16 [[X:%.*]] to i32
; CHECK-NEXT:    [[SHL:%.*]] = shl i32 [[CONV]], [[Y:%.*]]
; CHECK-NEXT:    [[T:%.*]] = trunc i32 [[SHL]] to i17
; CHECK-NEXT:    ret i17 [[T]]
;
  %conv = sext i16 %x to i32
  %shl = shl i32 %conv, %y
  %t = trunc i32 %shl to i17
  ret i17 %t
}

define i32 @sext_shl_mask(i16 %x, i32 %y) {
; CHECK-LABEL: @sext_shl_mask(
; CHECK-NEXT:    [[CONV:%.*]] = sext i16 [[X:%.*]] to i32
; CHECK-NEXT:    [[SHL:%.*]] = shl i32 [[CONV]], [[Y:%.*]]
; CHECK-NEXT:    [[T:%.*]] = and i32 [[SHL]], 65535
; CHECK-NEXT:    ret i32 [[T]]
;
  %conv = sext i16 %x to i32
  %shl = shl i32 %conv, %y
  %t = and i32 %shl, 65535
  ret i32 %t
}

define i32 @sext_shl_mask_higher(i16 %x, i32 %y) {
; CHECK-LABEL: @sext_shl_mask_higher(
; CHECK-NEXT:    [[CONV:%.*]] = sext i16 [[X:%.*]] to i32
; CHECK-NEXT:    [[SHL:%.*]] = shl i32 [[CONV]], [[Y:%.*]]
; CHECK-NEXT:    [[T:%.*]] = and i32 [[SHL]], 65536
; CHECK-NEXT:    ret i32 [[T]]
;
  %conv = sext i16 %x to i32
  %shl = shl i32 %conv, %y
  %t = and i32 %shl, 65536
  ret i32 %t
}

define i32 @set_shl_mask(i32 %x, i32 %y) {
; CHECK-LABEL: @set_shl_mask(
; CHECK-NEXT:    [[Z:%.*]] = or i32 [[X:%.*]], 196609
; CHECK-NEXT:    [[S:%.*]] = shl i32 [[Z]], [[Y:%.*]]
; CHECK-NEXT:    [[R:%.*]] = and i32 [[S]], 65536
; CHECK-NEXT:    ret i32 [[R]]
;
  %z = or i32 %x, 196609
  %s = shl i32 %z, %y
  %r = and i32 %s, 65536
  ret i32 %r
}
