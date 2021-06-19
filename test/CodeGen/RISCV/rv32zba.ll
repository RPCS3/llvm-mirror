; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py
; RUN: llc -mtriple=riscv32 -mattr=+m -verify-machineinstrs < %s \
; RUN:   | FileCheck %s -check-prefix=RV32I
; RUN: llc -mtriple=riscv32 -mattr=+m,+experimental-b -verify-machineinstrs < %s \
; RUN:   | FileCheck %s -check-prefix=RV32IB
; RUN: llc -mtriple=riscv32 -mattr=+m,+experimental-zba -verify-machineinstrs < %s \
; RUN:   | FileCheck %s -check-prefix=RV32IBA

define signext i16 @sh1add(i64 %0, i16* %1) {
; RV32I-LABEL: sh1add:
; RV32I:       # %bb.0:
; RV32I-NEXT:    slli a0, a0, 1
; RV32I-NEXT:    add a0, a2, a0
; RV32I-NEXT:    lh a0, 0(a0)
; RV32I-NEXT:    ret
;
; RV32IB-LABEL: sh1add:
; RV32IB:       # %bb.0:
; RV32IB-NEXT:    sh1add a0, a0, a2
; RV32IB-NEXT:    lh a0, 0(a0)
; RV32IB-NEXT:    ret
;
; RV32IBA-LABEL: sh1add:
; RV32IBA:       # %bb.0:
; RV32IBA-NEXT:    sh1add a0, a0, a2
; RV32IBA-NEXT:    lh a0, 0(a0)
; RV32IBA-NEXT:    ret
  %3 = getelementptr inbounds i16, i16* %1, i64 %0
  %4 = load i16, i16* %3
  ret i16 %4
}

define i32 @sh2add(i64 %0, i32* %1) {
; RV32I-LABEL: sh2add:
; RV32I:       # %bb.0:
; RV32I-NEXT:    slli a0, a0, 2
; RV32I-NEXT:    add a0, a2, a0
; RV32I-NEXT:    lw a0, 0(a0)
; RV32I-NEXT:    ret
;
; RV32IB-LABEL: sh2add:
; RV32IB:       # %bb.0:
; RV32IB-NEXT:    sh2add a0, a0, a2
; RV32IB-NEXT:    lw a0, 0(a0)
; RV32IB-NEXT:    ret
;
; RV32IBA-LABEL: sh2add:
; RV32IBA:       # %bb.0:
; RV32IBA-NEXT:    sh2add a0, a0, a2
; RV32IBA-NEXT:    lw a0, 0(a0)
; RV32IBA-NEXT:    ret
  %3 = getelementptr inbounds i32, i32* %1, i64 %0
  %4 = load i32, i32* %3
  ret i32 %4
}

define i64 @sh3add(i64 %0, i64* %1) {
; RV32I-LABEL: sh3add:
; RV32I:       # %bb.0:
; RV32I-NEXT:    slli a0, a0, 3
; RV32I-NEXT:    add a1, a2, a0
; RV32I-NEXT:    lw a0, 0(a1)
; RV32I-NEXT:    lw a1, 4(a1)
; RV32I-NEXT:    ret
;
; RV32IB-LABEL: sh3add:
; RV32IB:       # %bb.0:
; RV32IB-NEXT:    sh3add a1, a0, a2
; RV32IB-NEXT:    lw a0, 0(a1)
; RV32IB-NEXT:    lw a1, 4(a1)
; RV32IB-NEXT:    ret
;
; RV32IBA-LABEL: sh3add:
; RV32IBA:       # %bb.0:
; RV32IBA-NEXT:    sh3add a1, a0, a2
; RV32IBA-NEXT:    lw a0, 0(a1)
; RV32IBA-NEXT:    lw a1, 4(a1)
; RV32IBA-NEXT:    ret
  %3 = getelementptr inbounds i64, i64* %1, i64 %0
  %4 = load i64, i64* %3
  ret i64 %4
}

define i32 @addmul6(i32 %a, i32 %b) {
; RV32I-LABEL: addmul6:
; RV32I:       # %bb.0:
; RV32I-NEXT:    addi a2, zero, 6
; RV32I-NEXT:    mul a0, a0, a2
; RV32I-NEXT:    add a0, a0, a1
; RV32I-NEXT:    ret
;
; RV32IB-LABEL: addmul6:
; RV32IB:       # %bb.0:
; RV32IB-NEXT:    sh1add a0, a0, a0
; RV32IB-NEXT:    sh1add a0, a0, a1
; RV32IB-NEXT:    ret
;
; RV32IBA-LABEL: addmul6:
; RV32IBA:       # %bb.0:
; RV32IBA-NEXT:    sh1add a0, a0, a0
; RV32IBA-NEXT:    sh1add a0, a0, a1
; RV32IBA-NEXT:    ret
  %c = mul i32 %a, 6
  %d = add i32 %c, %b
  ret i32 %d
}

define i32 @addmul10(i32 %a, i32 %b) {
; RV32I-LABEL: addmul10:
; RV32I:       # %bb.0:
; RV32I-NEXT:    addi a2, zero, 10
; RV32I-NEXT:    mul a0, a0, a2
; RV32I-NEXT:    add a0, a0, a1
; RV32I-NEXT:    ret
;
; RV32IB-LABEL: addmul10:
; RV32IB:       # %bb.0:
; RV32IB-NEXT:    sh2add a0, a0, a0
; RV32IB-NEXT:    sh1add a0, a0, a1
; RV32IB-NEXT:    ret
;
; RV32IBA-LABEL: addmul10:
; RV32IBA:       # %bb.0:
; RV32IBA-NEXT:    sh2add a0, a0, a0
; RV32IBA-NEXT:    sh1add a0, a0, a1
; RV32IBA-NEXT:    ret
  %c = mul i32 %a, 10
  %d = add i32 %c, %b
  ret i32 %d
}

define i32 @addmul12(i32 %a, i32 %b) {
; RV32I-LABEL: addmul12:
; RV32I:       # %bb.0:
; RV32I-NEXT:    addi a2, zero, 12
; RV32I-NEXT:    mul a0, a0, a2
; RV32I-NEXT:    add a0, a0, a1
; RV32I-NEXT:    ret
;
; RV32IB-LABEL: addmul12:
; RV32IB:       # %bb.0:
; RV32IB-NEXT:    sh1add a0, a0, a0
; RV32IB-NEXT:    sh2add a0, a0, a1
; RV32IB-NEXT:    ret
;
; RV32IBA-LABEL: addmul12:
; RV32IBA:       # %bb.0:
; RV32IBA-NEXT:    sh1add a0, a0, a0
; RV32IBA-NEXT:    sh2add a0, a0, a1
; RV32IBA-NEXT:    ret
  %c = mul i32 %a, 12
  %d = add i32 %c, %b
  ret i32 %d
}

define i32 @addmul18(i32 %a, i32 %b) {
; RV32I-LABEL: addmul18:
; RV32I:       # %bb.0:
; RV32I-NEXT:    addi a2, zero, 18
; RV32I-NEXT:    mul a0, a0, a2
; RV32I-NEXT:    add a0, a0, a1
; RV32I-NEXT:    ret
;
; RV32IB-LABEL: addmul18:
; RV32IB:       # %bb.0:
; RV32IB-NEXT:    sh3add a0, a0, a0
; RV32IB-NEXT:    sh1add a0, a0, a1
; RV32IB-NEXT:    ret
;
; RV32IBA-LABEL: addmul18:
; RV32IBA:       # %bb.0:
; RV32IBA-NEXT:    sh3add a0, a0, a0
; RV32IBA-NEXT:    sh1add a0, a0, a1
; RV32IBA-NEXT:    ret
  %c = mul i32 %a, 18
  %d = add i32 %c, %b
  ret i32 %d
}

define i32 @addmul20(i32 %a, i32 %b) {
; RV32I-LABEL: addmul20:
; RV32I:       # %bb.0:
; RV32I-NEXT:    addi a2, zero, 20
; RV32I-NEXT:    mul a0, a0, a2
; RV32I-NEXT:    add a0, a0, a1
; RV32I-NEXT:    ret
;
; RV32IB-LABEL: addmul20:
; RV32IB:       # %bb.0:
; RV32IB-NEXT:    sh2add a0, a0, a0
; RV32IB-NEXT:    sh2add a0, a0, a1
; RV32IB-NEXT:    ret
;
; RV32IBA-LABEL: addmul20:
; RV32IBA:       # %bb.0:
; RV32IBA-NEXT:    sh2add a0, a0, a0
; RV32IBA-NEXT:    sh2add a0, a0, a1
; RV32IBA-NEXT:    ret
  %c = mul i32 %a, 20
  %d = add i32 %c, %b
  ret i32 %d
}

define i32 @addmul24(i32 %a, i32 %b) {
; RV32I-LABEL: addmul24:
; RV32I:       # %bb.0:
; RV32I-NEXT:    addi a2, zero, 24
; RV32I-NEXT:    mul a0, a0, a2
; RV32I-NEXT:    add a0, a0, a1
; RV32I-NEXT:    ret
;
; RV32IB-LABEL: addmul24:
; RV32IB:       # %bb.0:
; RV32IB-NEXT:    sh1add a0, a0, a0
; RV32IB-NEXT:    sh3add a0, a0, a1
; RV32IB-NEXT:    ret
;
; RV32IBA-LABEL: addmul24:
; RV32IBA:       # %bb.0:
; RV32IBA-NEXT:    sh1add a0, a0, a0
; RV32IBA-NEXT:    sh3add a0, a0, a1
; RV32IBA-NEXT:    ret
  %c = mul i32 %a, 24
  %d = add i32 %c, %b
  ret i32 %d
}

define i32 @addmul36(i32 %a, i32 %b) {
; RV32I-LABEL: addmul36:
; RV32I:       # %bb.0:
; RV32I-NEXT:    addi a2, zero, 36
; RV32I-NEXT:    mul a0, a0, a2
; RV32I-NEXT:    add a0, a0, a1
; RV32I-NEXT:    ret
;
; RV32IB-LABEL: addmul36:
; RV32IB:       # %bb.0:
; RV32IB-NEXT:    sh3add a0, a0, a0
; RV32IB-NEXT:    sh2add a0, a0, a1
; RV32IB-NEXT:    ret
;
; RV32IBA-LABEL: addmul36:
; RV32IBA:       # %bb.0:
; RV32IBA-NEXT:    sh3add a0, a0, a0
; RV32IBA-NEXT:    sh2add a0, a0, a1
; RV32IBA-NEXT:    ret
  %c = mul i32 %a, 36
  %d = add i32 %c, %b
  ret i32 %d
}

define i32 @addmul40(i32 %a, i32 %b) {
; RV32I-LABEL: addmul40:
; RV32I:       # %bb.0:
; RV32I-NEXT:    addi a2, zero, 40
; RV32I-NEXT:    mul a0, a0, a2
; RV32I-NEXT:    add a0, a0, a1
; RV32I-NEXT:    ret
;
; RV32IB-LABEL: addmul40:
; RV32IB:       # %bb.0:
; RV32IB-NEXT:    sh2add a0, a0, a0
; RV32IB-NEXT:    sh3add a0, a0, a1
; RV32IB-NEXT:    ret
;
; RV32IBA-LABEL: addmul40:
; RV32IBA:       # %bb.0:
; RV32IBA-NEXT:    sh2add a0, a0, a0
; RV32IBA-NEXT:    sh3add a0, a0, a1
; RV32IBA-NEXT:    ret
  %c = mul i32 %a, 40
  %d = add i32 %c, %b
  ret i32 %d
}

define i32 @addmul72(i32 %a, i32 %b) {
; RV32I-LABEL: addmul72:
; RV32I:       # %bb.0:
; RV32I-NEXT:    addi a2, zero, 72
; RV32I-NEXT:    mul a0, a0, a2
; RV32I-NEXT:    add a0, a0, a1
; RV32I-NEXT:    ret
;
; RV32IB-LABEL: addmul72:
; RV32IB:       # %bb.0:
; RV32IB-NEXT:    sh3add a0, a0, a0
; RV32IB-NEXT:    sh3add a0, a0, a1
; RV32IB-NEXT:    ret
;
; RV32IBA-LABEL: addmul72:
; RV32IBA:       # %bb.0:
; RV32IBA-NEXT:    sh3add a0, a0, a0
; RV32IBA-NEXT:    sh3add a0, a0, a1
; RV32IBA-NEXT:    ret
  %c = mul i32 %a, 72
  %d = add i32 %c, %b
  ret i32 %d
}

define i32 @mul96(i32 %a) {
; RV32I-LABEL: mul96:
; RV32I:       # %bb.0:
; RV32I-NEXT:    addi a1, zero, 96
; RV32I-NEXT:    mul a0, a0, a1
; RV32I-NEXT:    ret
;
; RV32IB-LABEL: mul96:
; RV32IB:       # %bb.0:
; RV32IB-NEXT:    addi a1, zero, 96
; RV32IB-NEXT:    mul a0, a0, a1
; RV32IB-NEXT:    ret
;
; RV32IBA-LABEL: mul96:
; RV32IBA:       # %bb.0:
; RV32IBA-NEXT:    addi a1, zero, 96
; RV32IBA-NEXT:    mul a0, a0, a1
; RV32IBA-NEXT:    ret
  %c = mul i32 %a, 96
  ret i32 %c
}

define i32 @mul160(i32 %a) {
; RV32I-LABEL: mul160:
; RV32I:       # %bb.0:
; RV32I-NEXT:    addi a1, zero, 160
; RV32I-NEXT:    mul a0, a0, a1
; RV32I-NEXT:    ret
;
; RV32IB-LABEL: mul160:
; RV32IB:       # %bb.0:
; RV32IB-NEXT:    addi a1, zero, 160
; RV32IB-NEXT:    mul a0, a0, a1
; RV32IB-NEXT:    ret
;
; RV32IBA-LABEL: mul160:
; RV32IBA:       # %bb.0:
; RV32IBA-NEXT:    addi a1, zero, 160
; RV32IBA-NEXT:    mul a0, a0, a1
; RV32IBA-NEXT:    ret
  %c = mul i32 %a, 160
  ret i32 %c
}

define i32 @mul288(i32 %a) {
; RV32I-LABEL: mul288:
; RV32I:       # %bb.0:
; RV32I-NEXT:    addi a1, zero, 288
; RV32I-NEXT:    mul a0, a0, a1
; RV32I-NEXT:    ret
;
; RV32IB-LABEL: mul288:
; RV32IB:       # %bb.0:
; RV32IB-NEXT:    addi a1, zero, 288
; RV32IB-NEXT:    mul a0, a0, a1
; RV32IB-NEXT:    ret
;
; RV32IBA-LABEL: mul288:
; RV32IBA:       # %bb.0:
; RV32IBA-NEXT:    addi a1, zero, 288
; RV32IBA-NEXT:    mul a0, a0, a1
; RV32IBA-NEXT:    ret
  %c = mul i32 %a, 288
  ret i32 %c
}
