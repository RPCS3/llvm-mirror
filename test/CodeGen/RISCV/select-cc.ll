; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py
; RUN: llc -mtriple=riscv32 -disable-block-placement -verify-machineinstrs < %s \
; RUN:   | FileCheck -check-prefix=RV32I %s
; RUN: llc -mtriple=riscv32 -mattr=+experimental-zbt -disable-block-placement -verify-machineinstrs < %s \
; RUN:   | FileCheck -check-prefix=RV32IBT %s

define i32 @foo(i32 %a, i32 *%b) nounwind {
; RV32I-LABEL: foo:
; RV32I:       # %bb.0:
; RV32I-NEXT:    lw a2, 0(a1)
; RV32I-NEXT:    beq a0, a2, .LBB0_2
; RV32I-NEXT:  # %bb.1:
; RV32I-NEXT:    mv a0, a2
; RV32I-NEXT:  .LBB0_2:
; RV32I-NEXT:    lw a2, 0(a1)
; RV32I-NEXT:    bne a0, a2, .LBB0_4
; RV32I-NEXT:  # %bb.3:
; RV32I-NEXT:    mv a0, a2
; RV32I-NEXT:  .LBB0_4:
; RV32I-NEXT:    lw a2, 0(a1)
; RV32I-NEXT:    bltu a2, a0, .LBB0_6
; RV32I-NEXT:  # %bb.5:
; RV32I-NEXT:    mv a0, a2
; RV32I-NEXT:  .LBB0_6:
; RV32I-NEXT:    lw a2, 0(a1)
; RV32I-NEXT:    bgeu a0, a2, .LBB0_8
; RV32I-NEXT:  # %bb.7:
; RV32I-NEXT:    mv a0, a2
; RV32I-NEXT:  .LBB0_8:
; RV32I-NEXT:    lw a2, 0(a1)
; RV32I-NEXT:    bltu a0, a2, .LBB0_10
; RV32I-NEXT:  # %bb.9:
; RV32I-NEXT:    mv a0, a2
; RV32I-NEXT:  .LBB0_10:
; RV32I-NEXT:    lw a2, 0(a1)
; RV32I-NEXT:    bgeu a2, a0, .LBB0_12
; RV32I-NEXT:  # %bb.11:
; RV32I-NEXT:    mv a0, a2
; RV32I-NEXT:  .LBB0_12:
; RV32I-NEXT:    lw a2, 0(a1)
; RV32I-NEXT:    blt a2, a0, .LBB0_14
; RV32I-NEXT:  # %bb.13:
; RV32I-NEXT:    mv a0, a2
; RV32I-NEXT:  .LBB0_14:
; RV32I-NEXT:    lw a2, 0(a1)
; RV32I-NEXT:    bge a0, a2, .LBB0_16
; RV32I-NEXT:  # %bb.15:
; RV32I-NEXT:    mv a0, a2
; RV32I-NEXT:  .LBB0_16:
; RV32I-NEXT:    lw a2, 0(a1)
; RV32I-NEXT:    blt a0, a2, .LBB0_18
; RV32I-NEXT:  # %bb.17:
; RV32I-NEXT:    mv a0, a2
; RV32I-NEXT:  .LBB0_18:
; RV32I-NEXT:    lw a2, 0(a1)
; RV32I-NEXT:    bge a2, a0, .LBB0_20
; RV32I-NEXT:  # %bb.19:
; RV32I-NEXT:    mv a0, a2
; RV32I-NEXT:  .LBB0_20:
; RV32I-NEXT:    lw a2, 0(a1)
; RV32I-NEXT:    addi a3, zero, 1
; RV32I-NEXT:    blt a2, a3, .LBB0_22
; RV32I-NEXT:  # %bb.21:
; RV32I-NEXT:    mv a0, a2
; RV32I-NEXT:  .LBB0_22:
; RV32I-NEXT:    lw a1, 0(a1)
; RV32I-NEXT:    addi a3, zero, -1
; RV32I-NEXT:    blt a3, a2, .LBB0_24
; RV32I-NEXT:  # %bb.23:
; RV32I-NEXT:    mv a0, a1
; RV32I-NEXT:  .LBB0_24:
; RV32I-NEXT:    ret
;
; RV32IBT-LABEL: foo:
; RV32IBT:       # %bb.0:
; RV32IBT-NEXT:    lw a2, 0(a1)
; RV32IBT-NEXT:    lw a3, 0(a1)
; RV32IBT-NEXT:    xor a4, a0, a2
; RV32IBT-NEXT:    cmov a0, a4, a2, a0
; RV32IBT-NEXT:    lw a2, 0(a1)
; RV32IBT-NEXT:    xor a4, a0, a3
; RV32IBT-NEXT:    cmov a0, a4, a0, a3
; RV32IBT-NEXT:    lw a3, 0(a1)
; RV32IBT-NEXT:    sltu a4, a2, a0
; RV32IBT-NEXT:    cmov a0, a4, a0, a2
; RV32IBT-NEXT:    lw a2, 0(a1)
; RV32IBT-NEXT:    sltu a4, a0, a3
; RV32IBT-NEXT:    cmov a0, a4, a3, a0
; RV32IBT-NEXT:    lw a3, 0(a1)
; RV32IBT-NEXT:    sltu a4, a0, a2
; RV32IBT-NEXT:    cmov a0, a4, a0, a2
; RV32IBT-NEXT:    lw a2, 0(a1)
; RV32IBT-NEXT:    sltu a4, a3, a0
; RV32IBT-NEXT:    cmov a0, a4, a3, a0
; RV32IBT-NEXT:    lw a3, 0(a1)
; RV32IBT-NEXT:    slt a4, a2, a0
; RV32IBT-NEXT:    cmov a0, a4, a0, a2
; RV32IBT-NEXT:    lw a2, 0(a1)
; RV32IBT-NEXT:    slt a4, a0, a3
; RV32IBT-NEXT:    cmov a0, a4, a3, a0
; RV32IBT-NEXT:    lw a3, 0(a1)
; RV32IBT-NEXT:    slt a4, a0, a2
; RV32IBT-NEXT:    lw a5, 0(a1)
; RV32IBT-NEXT:    cmov a0, a4, a0, a2
; RV32IBT-NEXT:    slt a2, a3, a0
; RV32IBT-NEXT:    cmov a0, a2, a3, a0
; RV32IBT-NEXT:    slti a2, a5, 1
; RV32IBT-NEXT:    lw a1, 0(a1)
; RV32IBT-NEXT:    cmov a0, a2, a0, a5
; RV32IBT-NEXT:    addi a2, zero, -1
; RV32IBT-NEXT:    slt a2, a2, a5
; RV32IBT-NEXT:    cmov a0, a2, a0, a1
; RV32IBT-NEXT:    ret
  %val1 = load volatile i32, i32* %b
  %tst1 = icmp eq i32 %a, %val1
  %val2 = select i1 %tst1, i32 %a, i32 %val1

  %val3 = load volatile i32, i32* %b
  %tst2 = icmp ne i32 %val2, %val3
  %val4 = select i1 %tst2, i32 %val2, i32 %val3

  %val5 = load volatile i32, i32* %b
  %tst3 = icmp ugt i32 %val4, %val5
  %val6 = select i1 %tst3, i32 %val4, i32 %val5

  %val7 = load volatile i32, i32* %b
  %tst4 = icmp uge i32 %val6, %val7
  %val8 = select i1 %tst4, i32 %val6, i32 %val7

  %val9 = load volatile i32, i32* %b
  %tst5 = icmp ult i32 %val8, %val9
  %val10 = select i1 %tst5, i32 %val8, i32 %val9

  %val11 = load volatile i32, i32* %b
  %tst6 = icmp ule i32 %val10, %val11
  %val12 = select i1 %tst6, i32 %val10, i32 %val11

  %val13 = load volatile i32, i32* %b
  %tst7 = icmp sgt i32 %val12, %val13
  %val14 = select i1 %tst7, i32 %val12, i32 %val13

  %val15 = load volatile i32, i32* %b
  %tst8 = icmp sge i32 %val14, %val15
  %val16 = select i1 %tst8, i32 %val14, i32 %val15

  %val17 = load volatile i32, i32* %b
  %tst9 = icmp slt i32 %val16, %val17
  %val18 = select i1 %tst9, i32 %val16, i32 %val17

  %val19 = load volatile i32, i32* %b
  %tst10 = icmp sle i32 %val18, %val19
  %val20 = select i1 %tst10, i32 %val18, i32 %val19

  %val21 = load volatile i32, i32* %b
  %tst11 = icmp slt i32 %val21, 1
  %val22 = select i1 %tst11, i32 %val20, i32 %val21

  %val23 = load volatile i32, i32* %b
  %tst12 = icmp sgt i32 %val21, -1
  %val24 = select i1 %tst12, i32 %val22, i32 %val23

  ret i32 %val24
}
