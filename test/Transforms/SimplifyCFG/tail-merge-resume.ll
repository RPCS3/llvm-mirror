; NOTE: Assertions have been autogenerated by utils/update_test_checks.py
; RUN: opt -simplifycfg -simplifycfg-require-and-preserve-domtree=1 -sink-common-insts  -S < %s | FileCheck %s

; Test that we tail merge resume blocks and phi operands properly.

declare void @foo()
declare void @bar()
declare void @baz()
declare void @qux()
declare void @quux()
declare void @quuz()

define void @merge_simple(i1 %cond) personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
; CHECK-LABEL: @merge_simple(
; CHECK-NEXT:    invoke void @foo()
; CHECK-NEXT:    to label [[INVOKE_CONT:%.*]] unwind label [[LPAD:%.*]]
; CHECK:       invoke.cont:
; CHECK-NEXT:    invoke void @bar()
; CHECK-NEXT:    to label [[INVOKE_CONT2:%.*]] unwind label [[LPAD2:%.*]]
; CHECK:       invoke.cont2:
; CHECK-NEXT:    ret void
; CHECK:       lpad:
; CHECK-NEXT:    [[LP:%.*]] = landingpad { i8*, i32 }
; CHECK-NEXT:    cleanup
; CHECK-NEXT:    call void @baz()
; CHECK-NEXT:    br i1 [[COND:%.*]], label [[RESUME0:%.*]], label [[RESUME1:%.*]]
; CHECK:       lpad2:
; CHECK-NEXT:    [[LP2:%.*]] = landingpad { i8*, i32 }
; CHECK-NEXT:    cleanup
; CHECK-NEXT:    call void @quuz()
; CHECK-NEXT:    resume { i8*, i32 } [[LP2]]
; CHECK:       resume0:
; CHECK-NEXT:    call void @qux()
; CHECK-NEXT:    resume { i8*, i32 } [[LP]]
; CHECK:       resume1:
; CHECK-NEXT:    call void @quux()
; CHECK-NEXT:    resume { i8*, i32 } [[LP]]
;
  invoke void @foo() to label %invoke.cont unwind label %lpad

invoke.cont:
  invoke void @bar()to label %invoke.cont2 unwind label %lpad2
  ret void

invoke.cont2:
  ret void

lpad:
  %lp = landingpad { i8*, i32 } cleanup
  call void @baz()
  br i1 %cond, label %resume0, label %resume1

lpad2:
  %lp2 = landingpad { i8*, i32 } cleanup
  call void @quuz()
  resume { i8*, i32 } %lp2

resume0:
  call void @qux()
  resume { i8*, i32 } %lp

resume1:
  call void @quux()
  resume { i8*, i32 } %lp
}

declare dso_local i32 @__gxx_personality_v0(...)
