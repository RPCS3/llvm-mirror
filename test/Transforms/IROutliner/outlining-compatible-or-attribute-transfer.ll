; NOTE: Assertions have been autogenerated by utils/update_test_checks.py
; RUN: opt -S -verify -iroutliner < "%s" | FileCheck "%s"

; This has two compatible regions.  We have attributes that should be transferred 
; even if it is on only one of the regions.

; This includes the attributes no-jump-tables, profile-sample-accurate,
; speculative_load_hardening, and noimplicitfloat.  When instance of similarity
; has these attributes can we say that the outlined function can have these
; attributes since that is the more general case.

define void @outline_attrs1() #0 {
; CHECK-LABEL: @outline_attrs1(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[A:%.*]] = alloca i32, align 4
; CHECK-NEXT:    [[B:%.*]] = alloca i32, align 4
; CHECK-NEXT:    [[C:%.*]] = alloca i32, align 4
; CHECK-NEXT:    call void @outlined_ir_func_0(i32* [[A]], i32* [[B]], i32* [[C]])
; CHECK-NEXT:    ret void
;
entry:
  %a = alloca i32, align 4
  %b = alloca i32, align 4
  %c = alloca i32, align 4
  store i32 2, i32* %a, align 4
  store i32 3, i32* %b, align 4
  store i32 4, i32* %c, align 4
  %al = load i32, i32* %a
  %bl = load i32, i32* %b
  %cl = load i32, i32* %c
  ret void
}

define void @outline_attrs2() {
; CHECK-LABEL: @outline_attrs2(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[A:%.*]] = alloca i32, align 4
; CHECK-NEXT:    [[B:%.*]] = alloca i32, align 4
; CHECK-NEXT:    [[C:%.*]] = alloca i32, align 4
; CHECK-NEXT:    call void @outlined_ir_func_0(i32* [[A]], i32* [[B]], i32* [[C]])
; CHECK-NEXT:    ret void
;
entry:
  %a = alloca i32, align 4
  %b = alloca i32, align 4
  %c = alloca i32, align 4
  store i32 2, i32* %a, align 4
  store i32 3, i32* %b, align 4
  store i32 4, i32* %c, align 4
  %al = load i32, i32* %a
  %bl = load i32, i32* %b
  %cl = load i32, i32* %c
  ret void
}

attributes #0 = { "no-jump-tables"="true" "profile-sample-accurate"="true" "speculative_load_hardening" "noimplicitfloat"="true" "use-sample-profile"="true"}

; CHECK: define internal void @outlined_ir_func_0(i32* [[ARG0:%.*]], i32* [[ARG1:%.*]], i32* [[ARG2:%.*]]) [[ATTR:#[0-9]+]] {
; CHECK: entry_to_outline:
; CHECK-NEXT:    store i32 2, i32* [[ARG0]], align 4
; CHECK-NEXT:    store i32 3, i32* [[ARG1]], align 4
; CHECK-NEXT:    store i32 4, i32* [[ARG2]], align 4
; CHECK-NEXT:    [[AL:%.*]] = load i32, i32* [[ARG0]], align 4
; CHECK-NEXT:    [[BL:%.*]] = load i32, i32* [[ARG1]], align 4
; CHECK-NEXT:    [[CL:%.*]] = load i32, i32* [[ARG2]], align 4

; CHECK: attributes [[ATTR]] = { minsize optsize "no-jump-tables"="true" "noimplicitfloat"="true" "profile-sample-accurate"="true" "speculative_load_hardening" "use-sample-profile"="true" }
