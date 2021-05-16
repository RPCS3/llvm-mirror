; NOTE: Assertions have been autogenerated by utils/update_test_checks.py
; RUN: opt -S -codegenprepare < %s | FileCheck %s

target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "thumbv8.1m.main-none-eabi"

define i32 @lshr3_then(i32 %a) {
; CHECK-LABEL: @lshr3_then(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[C:%.*]] = icmp ult i32 [[A:%.*]], 8
; CHECK-NEXT:    br i1 [[C]], label [[THEN:%.*]], label [[ELSE:%.*]]
; CHECK:       then:
; CHECK-NEXT:    ret i32 0
; CHECK:       else:
; CHECK-NEXT:    [[L:%.*]] = lshr i32 [[A]], 3
; CHECK-NEXT:    ret i32 [[L]]
;
entry:
  %c = icmp ult i32 %a, 8
  br i1 %c, label %then, label %else

then:
  ret i32 0

else:
  %l = lshr i32 %a, 3
  ret i32 %l
}

define i32 @lshr5_else(i32 %a) {
; CHECK-LABEL: @lshr5_else(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[C:%.*]] = icmp ult i32 [[A:%.*]], 32
; CHECK-NEXT:    br i1 [[C]], label [[THEN:%.*]], label [[ELSE:%.*]]
; CHECK:       then:
; CHECK-NEXT:    [[L:%.*]] = lshr i32 [[A]], 5
; CHECK-NEXT:    ret i32 [[L]]
; CHECK:       else:
; CHECK-NEXT:    ret i32 0
;
entry:
  %c = icmp ult i32 %a, 32
  br i1 %c, label %then, label %else

then:
  %l = lshr i32 %a, 5
  ret i32 %l

else:
  ret i32 0
}

define i32 @lshr2_entry(i32 %a) {
; CHECK-LABEL: @lshr2_entry(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[L:%.*]] = lshr i32 [[A:%.*]], 1
; CHECK-NEXT:    [[C:%.*]] = icmp ult i32 [[A]], 2
; CHECK-NEXT:    br i1 [[C]], label [[THEN:%.*]], label [[ELSE:%.*]]
; CHECK:       then:
; CHECK-NEXT:    ret i32 [[L]]
; CHECK:       else:
; CHECK-NEXT:    ret i32 0
;
entry:
  %l = lshr i32 %a, 1
  %c = icmp ult i32 %a, 2
  br i1 %c, label %then, label %else

then:
  ret i32 %l

else:
  ret i32 0
}

define i32 @lshr5mismatch(i32 %a) {
; CHECK-LABEL: @lshr5mismatch(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[C:%.*]] = icmp ult i32 [[A:%.*]], 17
; CHECK-NEXT:    br i1 [[C]], label [[THEN:%.*]], label [[ELSE:%.*]]
; CHECK:       then:
; CHECK-NEXT:    [[L:%.*]] = lshr i32 [[A]], 5
; CHECK-NEXT:    ret i32 [[L]]
; CHECK:       else:
; CHECK-NEXT:    ret i32 0
;
entry:
  %c = icmp ult i32 %a, 17
  br i1 %c, label %then, label %else

then:
  %l = lshr i32 %a, 5
  ret i32 %l

else:
  ret i32 0
}

define i32 @ashr5_else(i32 %a) {
; CHECK-LABEL: @ashr5_else(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[C:%.*]] = icmp ult i32 [[A:%.*]], 32
; CHECK-NEXT:    br i1 [[C]], label [[THEN:%.*]], label [[ELSE:%.*]]
; CHECK:       then:
; CHECK-NEXT:    [[L:%.*]] = ashr i32 [[A]], 5
; CHECK-NEXT:    ret i32 [[L]]
; CHECK:       else:
; CHECK-NEXT:    ret i32 0
;
entry:
  %c = icmp ult i32 %a, 32
  br i1 %c, label %then, label %else

then:
  %l = ashr i32 %a, 5
  ret i32 %l

else:
  ret i32 0
}

define i32 @add10_else(i32 %a) {
; CHECK-LABEL: @add10_else(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[C:%.*]] = icmp eq i32 [[A:%.*]], 10
; CHECK-NEXT:    br i1 [[C]], label [[THEN:%.*]], label [[ELSE:%.*]]
; CHECK:       then:
; CHECK-NEXT:    ret i32 0
; CHECK:       else:
; CHECK-NEXT:    [[L:%.*]] = add i32 [[A]], 10
; CHECK-NEXT:    ret i32 [[L]]
;
entry:
  %c = icmp eq i32 %a, 10
  br i1 %c, label %then, label %else

then:
  ret i32 0

else:
  %l = add i32 %a, 10
  ret i32 %l
}

define i32 @addm10_then(i32 %a) {
; CHECK-LABEL: @addm10_then(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[C:%.*]] = icmp eq i32 [[A:%.*]], 10
; CHECK-NEXT:    br i1 [[C]], label [[THEN:%.*]], label [[ELSE:%.*]]
; CHECK:       then:
; CHECK-NEXT:    [[L:%.*]] = add i32 [[A]], -10
; CHECK-NEXT:    ret i32 [[L]]
; CHECK:       else:
; CHECK-NEXT:    ret i32 0
;
entry:
  %c = icmp eq i32 %a, 10
  br i1 %c, label %then, label %else

then:
  %l = add i32 %a, -10
  ret i32 %l

else:
  ret i32 0
}

define i32 @add_missmatch(i32 %a) {
; CHECK-LABEL: @add_missmatch(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[C:%.*]] = icmp eq i32 [[A:%.*]], 10
; CHECK-NEXT:    br i1 [[C]], label [[THEN:%.*]], label [[ELSE:%.*]]
; CHECK:       then:
; CHECK-NEXT:    [[L:%.*]] = add i32 [[A]], 10
; CHECK-NEXT:    ret i32 [[L]]
; CHECK:       else:
; CHECK-NEXT:    ret i32 0
;
entry:
  %c = icmp eq i32 %a, 10
  br i1 %c, label %then, label %else

then:
  %l = add i32 %a, 10
  ret i32 %l

else:
  ret i32 0
}

define i32 @sub10_else(i32 %a) {
; CHECK-LABEL: @sub10_else(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[C:%.*]] = icmp eq i32 [[A:%.*]], 10
; CHECK-NEXT:    br i1 [[C]], label [[THEN:%.*]], label [[ELSE:%.*]]
; CHECK:       then:
; CHECK-NEXT:    ret i32 0
; CHECK:       else:
; CHECK-NEXT:    [[L:%.*]] = sub i32 [[A]], 10
; CHECK-NEXT:    ret i32 [[L]]
;
entry:
  %c = icmp eq i32 %a, 10
  br i1 %c, label %then, label %else

then:
  ret i32 0

else:
  %l = sub i32 %a, 10
  ret i32 %l
}

define i32 @subm10_then(i32 %a) {
; CHECK-LABEL: @subm10_then(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[C:%.*]] = icmp eq i32 [[A:%.*]], -10
; CHECK-NEXT:    br i1 [[C]], label [[THEN:%.*]], label [[ELSE:%.*]]
; CHECK:       then:
; CHECK-NEXT:    [[L:%.*]] = sub i32 [[A]], -10
; CHECK-NEXT:    ret i32 [[L]]
; CHECK:       else:
; CHECK-NEXT:    ret i32 0
;
entry:
  %c = icmp eq i32 %a, -10
  br i1 %c, label %then, label %else

then:
  %l = sub i32 %a, -10
  ret i32 %l

else:
  ret i32 0
}

define i64 @lshr64(i64 %a) {
; CHECK-LABEL: @lshr64(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[C:%.*]] = icmp ult i64 [[A:%.*]], 1099511627776
; CHECK-NEXT:    br i1 [[C]], label [[THEN:%.*]], label [[ELSE:%.*]]
; CHECK:       then:
; CHECK-NEXT:    ret i64 0
; CHECK:       else:
; CHECK-NEXT:    [[L:%.*]] = lshr i64 [[A]], 40
; CHECK-NEXT:    ret i64 [[L]]
;
entry:
  %c = icmp ult i64 %a, 1099511627776
  br i1 %c, label %then, label %else

then:
  ret i64 0

else:
  %l = lshr i64 %a, 40
  ret i64 %l
}

define i128 @lshr128(i128 %a) {
; CHECK-LABEL: @lshr128(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[C:%.*]] = icmp ult i128 [[A:%.*]], 36893488147419103232
; CHECK-NEXT:    br i1 [[C]], label [[THEN:%.*]], label [[ELSE:%.*]]
; CHECK:       then:
; CHECK-NEXT:    ret i128 0
; CHECK:       else:
; CHECK-NEXT:    [[L:%.*]] = lshr i128 [[A]], 65
; CHECK-NEXT:    ret i128 [[L]]
;
entry:
  %c = icmp ult i128 %a, 36893488147419103232
  br i1 %c, label %then, label %else

then:
  ret i128 0

else:
  %l = lshr i128 %a, 65
  ret i128 %l
}

define i32 @addm1_dom(i32 %a) {
; CHECK-LABEL: @addm1_dom(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[C1:%.*]] = icmp eq i32 [[A:%.*]], 100
; CHECK-NEXT:    br i1 [[C1]], label [[IF:%.*]], label [[ELSE:%.*]]
; CHECK:       if:
; CHECK-NEXT:    [[C:%.*]] = icmp eq i32 [[A]], -1
; CHECK-NEXT:    br i1 [[C]], label [[THEN:%.*]], label [[ELSE]]
; CHECK:       then:
; CHECK-NEXT:    ret i32 0
; CHECK:       else:
; CHECK-NEXT:    [[L:%.*]] = add i32 [[A]], 1
; CHECK-NEXT:    ret i32 [[L]]
;
entry:
  %c1 = icmp eq i32 %a, 100
  br i1 %c1, label %if, label %else

if:
  %c = icmp eq i32 %a, -1
  br i1 %c, label %then, label %else

then:
  ret i32 0

else:
  %l = add i32 %a, 1
  ret i32 %l
}

declare void @other()
