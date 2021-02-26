; NOTE: Assertions have been autogenerated by utils/update_test_checks.py
; RUN: opt < %s -nary-reassociate -S | FileCheck %s
; RUN: opt < %s -passes='nary-reassociate' -S | FileCheck %s

declare i32 @llvm.smin.i32(i32 %a, i32 %b)

; m1 = smin(a,b) ; has side uses
; m2 = smin(smin((b,c), a) -> m2 = smin(m1, c)
define i32 @smin_test1(i32 %a, i32 %b, i32 %c) {
; CHECK-LABEL: @smin_test1(
; CHECK-NEXT:    [[C1:%.*]] = icmp slt i32 [[A:%.*]], [[B:%.*]]
; CHECK-NEXT:    [[SMIN1:%.*]] = select i1 [[C1]], i32 [[A]], i32 [[B]]
; CHECK-NEXT:    [[C2:%.*]] = icmp slt i32 [[B]], [[C:%.*]]
; CHECK-NEXT:    [[SMIN2:%.*]] = select i1 [[C2]], i32 [[B]], i32 [[C]]
; CHECK-NEXT:    [[C3:%.*]] = icmp slt i32 [[SMIN2]], [[A]]
; CHECK-NEXT:    [[SMIN3:%.*]] = select i1 [[C3]], i32 [[SMIN2]], i32 [[A]]
; CHECK-NEXT:    [[RES:%.*]] = add i32 [[SMIN1]], [[SMIN3]]
; CHECK-NEXT:    ret i32 [[RES]]
;
  %c1 = icmp slt i32 %a, %b
  %smin1 = select i1 %c1, i32 %a, i32 %b
  %c2 = icmp slt i32 %b, %c
  %smin2 = select i1 %c2, i32 %b, i32 %c
  %c3 = icmp slt i32 %smin2, %a
  %smin3 = select i1 %c3, i32 %smin2, i32 %a
  %res = add i32 %smin1, %smin3
  ret i32 %res
}

; m1 = smin(a,b) ; has side uses
; m2 = smin(b, (smin(a, c))) -> m2 = smin(m1, c)
define i32 @smin_test2(i32 %a, i32 %b, i32 %c) {
; CHECK-LABEL: @smin_test2(
; CHECK-NEXT:    [[C1:%.*]] = icmp slt i32 [[A:%.*]], [[B:%.*]]
; CHECK-NEXT:    [[SMIN1:%.*]] = select i1 [[C1]], i32 [[A]], i32 [[B]]
; CHECK-NEXT:    [[C2:%.*]] = icmp slt i32 [[A]], [[C:%.*]]
; CHECK-NEXT:    [[SMIN2:%.*]] = select i1 [[C2]], i32 [[A]], i32 [[C]]
; CHECK-NEXT:    [[C3:%.*]] = icmp slt i32 [[B]], [[SMIN2]]
; CHECK-NEXT:    [[SMIN3:%.*]] = select i1 [[C3]], i32 [[B]], i32 [[SMIN2]]
; CHECK-NEXT:    [[RES:%.*]] = add i32 [[SMIN1]], [[SMIN3]]
; CHECK-NEXT:    ret i32 [[RES]]
;
  %c1 = icmp slt i32 %a, %b
  %smin1 = select i1 %c1, i32 %a, i32 %b
  %c2 = icmp slt i32 %a, %c
  %smin2 = select i1 %c2, i32 %a, i32 %c
  %c3 = icmp slt i32 %b, %smin2
  %smin3 = select i1 %c3, i32 %b, i32 %smin2
  %res = add i32 %smin1, %smin3
  ret i32 %res
}

; Same test as smin_test1 but uses @llvm.smin intrinsic
define i32 @smin_test3(i32 %a, i32 %b, i32 %c) {
; CHECK-LABEL: @smin_test3(
; CHECK-NEXT:    [[SMIN1:%.*]] = call i32 @llvm.smin.i32(i32 [[A:%.*]], i32 [[B:%.*]])
; CHECK-NEXT:    [[SMIN2:%.*]] = call i32 @llvm.smin.i32(i32 [[B]], i32 [[C:%.*]])
; CHECK-NEXT:    [[SMIN3:%.*]] = call i32 @llvm.smin.i32(i32 [[SMIN2]], i32 [[A]])
; CHECK-NEXT:    [[RES:%.*]] = add i32 [[SMIN1]], [[SMIN3]]
; CHECK-NEXT:    ret i32 [[RES]]
;
  %smin1 = call i32 @llvm.smin.i32(i32 %a, i32 %b)
  %smin2 = call i32 @llvm.smin.i32(i32 %b, i32 %c)
  %smin3 = call i32 @llvm.smin.i32(i32 %smin2, i32 %a)
  %res = add i32 %smin1, %smin3
  ret i32 %res
}

; m1 = smin(a,b) ; has side uses
; m2 = smin(smin_or_eq((b,c), a) -> m2 = smin(m1, c)
define i32 @umin_test4(i32 %a, i32 %b, i32 %c) {
; CHECK-LABEL: @umin_test4(
; CHECK-NEXT:    [[C1:%.*]] = icmp slt i32 [[A:%.*]], [[B:%.*]]
; CHECK-NEXT:    [[SMIN1:%.*]] = select i1 [[C1]], i32 [[A]], i32 [[B]]
; CHECK-NEXT:    [[C2:%.*]] = icmp sle i32 [[B]], [[C:%.*]]
; CHECK-NEXT:    [[SMIN_OR_EQ2:%.*]] = select i1 [[C2]], i32 [[B]], i32 [[C]]
; CHECK-NEXT:    [[C3:%.*]] = icmp slt i32 [[SMIN_OR_EQ2]], [[A]]
; CHECK-NEXT:    [[SMIN3:%.*]] = select i1 [[C3]], i32 [[SMIN_OR_EQ2]], i32 [[A]]
; CHECK-NEXT:    [[RES:%.*]] = add i32 [[SMIN1]], [[SMIN3]]
; CHECK-NEXT:    ret i32 [[RES]]
;
  %c1 = icmp slt i32 %a, %b
  %smin1 = select i1 %c1, i32 %a, i32 %b
  %c2 = icmp sle i32 %b, %c
  %smin_or_eq2 = select i1 %c2, i32 %b, i32 %c
  %c3 = icmp slt i32 %smin_or_eq2, %a
  %smin3 = select i1 %c3, i32 %smin_or_eq2, i32 %a
  %res = add i32 %smin1, %smin3
  ret i32 %res
}

; m1 = smin_or_eq(a,b) ; has side uses
; m2 = smin_or_eq(smin((b,c), a) -> m2 = smin(m1, c)
define i32 @smin_test5(i32 %a, i32 %b, i32 %c) {
; CHECK-LABEL: @smin_test5(
; CHECK-NEXT:    [[C1:%.*]] = icmp sle i32 [[A:%.*]], [[B:%.*]]
; CHECK-NEXT:    [[SMIN_OR_EQ1:%.*]] = select i1 [[C1]], i32 [[A]], i32 [[B]]
; CHECK-NEXT:    [[C2:%.*]] = icmp slt i32 [[B]], [[C:%.*]]
; CHECK-NEXT:    [[SMIN2:%.*]] = select i1 [[C2]], i32 [[B]], i32 [[C]]
; CHECK-NEXT:    [[C3:%.*]] = icmp sle i32 [[SMIN2]], [[A]]
; CHECK-NEXT:    [[SMIN_OR_EQ3:%.*]] = select i1 [[C3]], i32 [[SMIN2]], i32 [[A]]
; CHECK-NEXT:    [[RES:%.*]] = add i32 [[SMIN_OR_EQ1]], [[SMIN_OR_EQ3]]
; CHECK-NEXT:    ret i32 [[RES]]
;
  %c1 = icmp sle i32 %a, %b
  %smin_or_eq1 = select i1 %c1, i32 %a, i32 %b
  %c2 = icmp slt i32 %b, %c
  %smin2 = select i1 %c2, i32 %b, i32 %c
  %c3 = icmp sle i32 %smin2, %a
  %smin_or_eq3 = select i1 %c3, i32 %smin2, i32 %a
  %res = add i32 %smin_or_eq1, %smin_or_eq3
  ret i32 %res
}

; m1 = smin(a,b) ; has side uses
; m2 = smin(umin((b,c), a) ; check that signed and unsigned mins are not mixed
define i32 @smin_test6(i32 %a, i32 %b, i32 %c) {
; CHECK-LABEL: @smin_test6(
; CHECK-NEXT:    [[C1:%.*]] = icmp slt i32 [[A:%.*]], [[B:%.*]]
; CHECK-NEXT:    [[SMIN1:%.*]] = select i1 [[C1]], i32 [[A]], i32 [[B]]
; CHECK-NEXT:    [[C2:%.*]] = icmp ult i32 [[B]], [[C:%.*]]
; CHECK-NEXT:    [[UMIN2:%.*]] = select i1 [[C2]], i32 [[B]], i32 [[C]]
; CHECK-NEXT:    [[C3:%.*]] = icmp slt i32 [[UMIN2]], [[A]]
; CHECK-NEXT:    [[SMIN3:%.*]] = select i1 [[C3]], i32 [[UMIN2]], i32 [[A]]
; CHECK-NEXT:    [[RES:%.*]] = add i32 [[SMIN1]], [[SMIN3]]
; CHECK-NEXT:    ret i32 [[RES]]
;
  %c1 = icmp slt i32 %a, %b
  %smin1 = select i1 %c1, i32 %a, i32 %b
  %c2 = icmp ult i32 %b, %c
  %umin2 = select i1 %c2, i32 %b, i32 %c
  %c3 = icmp slt i32 %umin2, %a
  %smin3 = select i1 %c3, i32 %umin2, i32 %a
  %res = add i32 %smin1, %smin3
  ret i32 %res
}

; m1 = smin(a,b) ; has side uses
; m2 = smin(smax((b,c), a) ; check that min and max are not mixed
define i32 @smin_test7(i32 %a, i32 %b, i32 %c) {
; CHECK-LABEL: @smin_test7(
; CHECK-NEXT:    [[C1:%.*]] = icmp slt i32 [[A:%.*]], [[B:%.*]]
; CHECK-NEXT:    [[SMIN1:%.*]] = select i1 [[C1]], i32 [[A]], i32 [[B]]
; CHECK-NEXT:    [[C2:%.*]] = icmp sgt i32 [[B]], [[C:%.*]]
; CHECK-NEXT:    [[SMAX2:%.*]] = select i1 [[C2]], i32 [[B]], i32 [[C]]
; CHECK-NEXT:    [[C3:%.*]] = icmp slt i32 [[SMAX2]], [[A]]
; CHECK-NEXT:    [[SMIN3:%.*]] = select i1 [[C3]], i32 [[SMAX2]], i32 [[A]]
; CHECK-NEXT:    [[RES:%.*]] = add i32 [[SMIN1]], [[SMIN3]]
; CHECK-NEXT:    ret i32 [[RES]]
;
  %c1 = icmp slt i32 %a, %b
  %smin1 = select i1 %c1, i32 %a, i32 %b
  %c2 = icmp sgt i32 %b, %c
  %smax2 = select i1 %c2, i32 %b, i32 %c
  %c3 = icmp slt i32 %smax2, %a
  %smin3 = select i1 %c3, i32 %smax2, i32 %a
  %res = add i32 %smin1, %smin3
  ret i32 %res
}
