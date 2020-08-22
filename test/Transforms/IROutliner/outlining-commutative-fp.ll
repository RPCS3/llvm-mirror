; NOTE: Assertions have been autogenerated by utils/update_test_checks.py
; RUN: opt -S -verify -iroutliner < %s | FileCheck %s

; This test checks that floating point commutative instructions are not treated
; as commutative.  Even though an ffadd is technically commutative, the order
; of operands still needs to be enforced since the process of fadding floating
; point values requires the order to be the same.

; We make sure that we outline the identical regions from the first two
; functions, but not the third.  this is because the operands are in a different
; order in a floating point instruction in this section.

define void @outline_from_fadd1() {
; CHECK-LABEL: @outline_from_fadd1(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[A:%.*]] = alloca double, align 4
; CHECK-NEXT:    [[B:%.*]] = alloca double, align 4
; CHECK-NEXT:    [[C:%.*]] = alloca double, align 4
; CHECK-NEXT:    call void @outlined_ir_func_0(double* [[A]], double* [[B]], double* [[C]])
; CHECK-NEXT:    ret void
;
entry:
  %a = alloca double, align 4
  %b = alloca double, align 4
  %c = alloca double, align 4
  store double 2.0, double* %a, align 4
  store double 3.0, double* %b, align 4
  store double 4.0, double* %c, align 4
  %al = load double, double* %a
  %bl = load double, double* %b
  %cl = load double, double* %c
  %0 = fadd double %al, %bl
  %1 = fadd double %al, %cl
  %2 = fadd double %bl, %cl
  ret void
}

define void @outline_from_fadd2.0() {
; CHECK-LABEL: @outline_from_fadd2.0(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[A:%.*]] = alloca double, align 4
; CHECK-NEXT:    [[B:%.*]] = alloca double, align 4
; CHECK-NEXT:    [[C:%.*]] = alloca double, align 4
; CHECK-NEXT:    call void @outlined_ir_func_0(double* [[A]], double* [[B]], double* [[C]])
; CHECK-NEXT:    ret void
;
entry:
  %a = alloca double, align 4
  %b = alloca double, align 4
  %c = alloca double, align 4
  store double 2.0, double* %a, align 4
  store double 3.0, double* %b, align 4
  store double 4.0, double* %c, align 4
  %al = load double, double* %a
  %bl = load double, double* %b
  %cl = load double, double* %c
  %0 = fadd double %al, %bl
  %1 = fadd double %al, %cl
  %2 = fadd double %bl, %cl
  ret void
}

define void @outline_from_flipped_fadd3.0() {
; CHECK-LABEL: @outline_from_flipped_fadd3.0(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[A:%.*]] = alloca double, align 4
; CHECK-NEXT:    [[B:%.*]] = alloca double, align 4
; CHECK-NEXT:    [[C:%.*]] = alloca double, align 4
; CHECK-NEXT:    store double 2.000000e+00, double* [[A]], align 4
; CHECK-NEXT:    store double 3.000000e+00, double* [[B]], align 4
; CHECK-NEXT:    store double 4.000000e+00, double* [[C]], align 4
; CHECK-NEXT:    [[AL:%.*]] = load double, double* [[A]], align 8
; CHECK-NEXT:    [[BL:%.*]] = load double, double* [[B]], align 8
; CHECK-NEXT:    [[CL:%.*]] = load double, double* [[C]], align 8
; CHECK-NEXT:    [[TMP0:%.*]] = fadd double [[BL]], [[AL]]
; CHECK-NEXT:    [[TMP1:%.*]] = fadd double [[CL]], [[AL]]
; CHECK-NEXT:    [[TMP2:%.*]] = fadd double [[CL]], [[BL]]
; CHECK-NEXT:    ret void
;
entry:
  %a = alloca double, align 4
  %b = alloca double, align 4
  %c = alloca double, align 4
  store double 2.0, double* %a, align 4
  store double 3.0, double* %b, align 4
  store double 4.0, double* %c, align 4
  %al = load double, double* %a
  %bl = load double, double* %b
  %cl = load double, double* %c
  %0 = fadd double %bl, %al
  %1 = fadd double %cl, %al
  %2 = fadd double %cl, %bl
  ret void
}

; CHECK: define internal void @outlined_ir_func_0(double* [[ARG0:%.*]], double* [[ARG1:%.*]], double* [[ARG2:%.*]]) #0 {
; CHECK: entry_to_outline:
; CHECK-NEXT:    store double 2.000000e+00, double* [[ARG0]], align 4
; CHECK-NEXT:    store double 3.000000e+00, double* [[ARG1]], align 4
; CHECK-NEXT:    store double 4.000000e+00, double* [[ARG2]], align 4
; CHECK-NEXT:    [[AL:%.*]] = load double, double* [[ARG0]], align 8
; CHECK-NEXT:    [[BL:%.*]] = load double, double* [[ARG1]], align 8
; CHECK-NEXT:    [[CL:%.*]] = load double, double* [[ARG2]], align 8
; CHECK-NEXT:    [[TMP0:%.*]] = fadd double [[AL]], [[BL]]
; CHECK-NEXT:    [[TMP1:%.*]] = fadd double [[AL]], [[CL]]
; CHECK-NEXT:    [[TMP2:%.*]] = fadd double [[BL]], [[CL]]

