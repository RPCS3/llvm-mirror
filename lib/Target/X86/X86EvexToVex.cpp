//===- X86EvexToVex.cpp ---------------------------------------------------===//
// Compress EVEX instructions to VEX encoding when possible to reduce code size
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// This file defines the pass that goes over all AVX-512 instructions which
/// are encoded using the EVEX prefix and if possible replaces them by their
/// corresponding VEX encoding which is usually shorter by 2 bytes.
/// EVEX instructions may be encoded via the VEX prefix when the AVX-512
/// instruction has a corresponding AVX/AVX2 opcode, when vector length 
/// accessed by instruction is less than 512 bits and when it does not use 
//  the xmm or the mask registers or xmm/ymm registers with indexes higher than 15.
/// The pass applies code reduction on the generated code for AVX-512 instrs.
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/X86BaseInfo.h"
#include "MCTargetDesc/X86InstComments.h"
#include "X86.h"
#include "X86InstrInfo.h"
#include "X86Subtarget.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/MC/MCInstrDesc.h"
#include "llvm/Pass.h"
#include <cassert>
#include <cstdint>

using namespace llvm;

// Including the generated EVEX2VEX tables.
struct X86EvexToVexCompressTableEntry {
  uint16_t EvexOpcode;
  uint16_t VexOpcode;

  bool operator<(const X86EvexToVexCompressTableEntry &RHS) const {
    return EvexOpcode < RHS.EvexOpcode;
  }

  friend bool operator<(const X86EvexToVexCompressTableEntry &TE,
                        unsigned Opc) {
    return TE.EvexOpcode < Opc;
  }
};
#include "X86GenEVEX2VEXTables.inc"

#define EVEX2VEX_DESC "Compressing EVEX instrs to VEX encoding when possible"
#define EVEX2VEX_NAME "x86-evex-to-vex-compress"

#define DEBUG_TYPE EVEX2VEX_NAME

namespace {

class EvexToVexInstPass : public MachineFunctionPass {

  /// For EVEX instructions that can be encoded using VEX encoding, replace
  /// them by the VEX encoding in order to reduce size.
  bool CompressEvexToVexImpl(MachineInstr &MI) const;

public:
  static char ID;

  EvexToVexInstPass() : MachineFunctionPass(ID) { }

  StringRef getPassName() const override { return EVEX2VEX_DESC; }

  /// Loop over all of the basic blocks, replacing EVEX instructions
  /// by equivalent VEX instructions when possible for reducing code size.
  bool runOnMachineFunction(MachineFunction &MF) override;

  // This pass runs after regalloc and doesn't support VReg operands.
  MachineFunctionProperties getRequiredProperties() const override {
    return MachineFunctionProperties().set(
        MachineFunctionProperties::Property::NoVRegs);
  }

private:
  /// Machine instruction info used throughout the class.
  const X86InstrInfo *TII = nullptr;

  const X86Subtarget *ST = nullptr;
};

} // end anonymous namespace

char EvexToVexInstPass::ID = 0;

bool EvexToVexInstPass::runOnMachineFunction(MachineFunction &MF) {
  TII = MF.getSubtarget<X86Subtarget>().getInstrInfo();

  ST = &MF.getSubtarget<X86Subtarget>();
  if (!ST->hasAVX512())
    return false;

  bool Changed = false;

  /// Go over all basic blocks in function and replace
  /// EVEX encoded instrs by VEX encoding when possible.
  for (MachineBasicBlock &MBB : MF) {

    // Traverse the basic block.
    for (MachineInstr &MI : MBB)
      Changed |= CompressEvexToVexImpl(MI);
  }

  return Changed;
}

static bool usesExtendedRegister(const MachineInstr &MI) {
  auto isHiRegIdx = [](unsigned Reg) {
    // Check for XMM register with indexes between 16 - 31.
    if (Reg >= X86::XMM16 && Reg <= X86::XMM31)
      return true;

    // Check for YMM register with indexes between 16 - 31.
    if (Reg >= X86::YMM16 && Reg <= X86::YMM31)
      return true;

    return false;
  };

  // Check that operands are not ZMM regs or
  // XMM/YMM regs with hi indexes between 16 - 31.
  for (const MachineOperand &MO : MI.explicit_operands()) {
    if (!MO.isReg())
      continue;

    Register Reg = MO.getReg();

    assert(!(Reg >= X86::ZMM0 && Reg <= X86::ZMM31) &&
           "ZMM instructions should not be in the EVEX->VEX tables");

    if (isHiRegIdx(Reg))
      return true;
  }

  return false;
}

static bool usesDisp8Compression(const MachineInstr &MI) {
  unsigned Opc = MI.getOpcode();
  int disp = 0;
  int stride = 0;
  switch (Opc) {
  case X86::VADDPDZ128rm:
  case X86::VADDPSZ128rm:
  case X86::VADDSDZrm:
  case X86::VADDSDZrm_Int:
  case X86::VADDSSZrm:
  case X86::VADDSSZrm_Int:
  case X86::VAESDECLASTZ128rm:
  case X86::VAESDECZ128rm:
  case X86::VAESENCLASTZ128rm:
  case X86::VAESENCZ128rm:
  case X86::VALIGNDZ128rmi:
  case X86::VALIGNQZ128rmi:
  case X86::VANDNPDZ128rm:
  case X86::VANDNPSZ128rm:
  case X86::VANDPDZ128rm:
  case X86::VANDPSZ128rm:
  case X86::VCVTSI2SDZrm:
  case X86::VCVTSI2SDZrm_Int:
  case X86::VCVTSI2SSZrm:
  case X86::VCVTSI2SSZrm_Int:
  case X86::VCVTSI642SDZrm:
  case X86::VCVTSI642SDZrm_Int:
  case X86::VCVTSI642SSZrm:
  case X86::VCVTSI642SSZrm_Int:
  case X86::VCVTSS2SDZrm:
  case X86::VCVTSS2SDZrm_Int:
  case X86::VDIVPDZ128rm:
  case X86::VDIVPSZ128rm:
  case X86::VDIVSDZrm:
  case X86::VDIVSDZrm_Int:
  case X86::VDIVSSZrm:
  case X86::VDIVSSZrm_Int:
  case X86::VGF2P8AFFINEINVQBZ128rmi:
  case X86::VGF2P8AFFINEQBZ128rmi:
  case X86::VGF2P8MULBZ128rm:
  case X86::VINSERTPSZrm:
  case X86::VMAXCPDZ128rm:
  case X86::VMAXCPSZ128rm:
  case X86::VMAXCSDZrm:
  case X86::VMAXCSSZrm:
  case X86::VMAXPDZ128rm:
  case X86::VMAXPSZ128rm:
  case X86::VMAXSDZrm:
  case X86::VMAXSDZrm_Int:
  case X86::VMAXSSZrm:
  case X86::VMAXSSZrm_Int:
  case X86::VMINCPDZ128rm:
  case X86::VMINCPSZ128rm:
  case X86::VMINCSDZrm:
  case X86::VMINCSSZrm:
  case X86::VMINPDZ128rm:
  case X86::VMINPSZ128rm:
  case X86::VMINSDZrm:
  case X86::VMINSDZrm_Int:
  case X86::VMINSSZrm:
  case X86::VMINSSZrm_Int:
  case X86::VMOVHPDZ128rm:
  case X86::VMOVHPSZ128rm:
  case X86::VMOVLPDZ128rm:
  case X86::VMOVLPSZ128rm:
  case X86::VMULPDZ128rm:
  case X86::VMULPSZ128rm:
  case X86::VMULSDZrm:
  case X86::VMULSDZrm_Int:
  case X86::VMULSSZrm:
  case X86::VMULSSZrm_Int:
  case X86::VORPDZ128rm:
  case X86::VORPSZ128rm:
  case X86::VPACKSSDWZ128rm:
  case X86::VPACKSSWBZ128rm:
  case X86::VPACKUSDWZ128rm:
  case X86::VPACKUSWBZ128rm:
  case X86::VPADDBZ128rm:
  case X86::VPADDDZ128rm:
  case X86::VPADDQZ128rm:
  case X86::VPADDSBZ128rm:
  case X86::VPADDSWZ128rm:
  case X86::VPADDUSBZ128rm:
  case X86::VPADDUSWZ128rm:
  case X86::VPADDWZ128rm:
  case X86::VPALIGNRZ128rmi:
  case X86::VPANDDZ128rm:
  case X86::VPANDNDZ128rm:
  case X86::VPANDNQZ128rm:
  case X86::VPANDQZ128rm:
  case X86::VPAVGBZ128rm:
  case X86::VPAVGWZ128rm:
  case X86::VPCLMULQDQZ128rm:
  case X86::VPERMILPDZ128rm:
  case X86::VPERMILPSZ128rm:
  case X86::VPINSRBZrm:
  case X86::VPINSRDZrm:
  case X86::VPINSRQZrm:
  case X86::VPINSRWZrm:
  case X86::VPMADDUBSWZ128rm:
  case X86::VPMADDWDZ128rm:
  case X86::VPMAXSBZ128rm:
  case X86::VPMAXSDZ128rm:
  case X86::VPMAXSWZ128rm:
  case X86::VPMAXUBZ128rm:
  case X86::VPMAXUDZ128rm:
  case X86::VPMAXUWZ128rm:
  case X86::VPMINSBZ128rm:
  case X86::VPMINSDZ128rm:
  case X86::VPMINSWZ128rm:
  case X86::VPMINUBZ128rm:
  case X86::VPMINUDZ128rm:
  case X86::VPMINUWZ128rm:
  case X86::VPMULDQZ128rm:
  case X86::VPMULHRSWZ128rm:
  case X86::VPMULHUWZ128rm:
  case X86::VPMULHWZ128rm:
  case X86::VPMULLDZ128rm:
  case X86::VPMULLWZ128rm:
  case X86::VPMULUDQZ128rm:
  case X86::VPORDZ128rm:
  case X86::VPORQZ128rm:
  case X86::VPSADBWZ128rm:
  case X86::VPSHUFBZ128rm:
  case X86::VPSLLDZ128rm:
  case X86::VPSLLQZ128rm:
  case X86::VPSLLVDZ128rm:
  case X86::VPSLLVQZ128rm:
  case X86::VPSLLWZ128rm:
  case X86::VPSRADZ128rm:
  case X86::VPSRAVDZ128rm:
  case X86::VPSRAWZ128rm:
  case X86::VPSRLDZ128rm:
  case X86::VPSRLQZ128rm:
  case X86::VPSRLVDZ128rm:
  case X86::VPSRLVQZ128rm:
  case X86::VPSRLWZ128rm:
  case X86::VPSUBBZ128rm:
  case X86::VPSUBDZ128rm:
  case X86::VPSUBQZ128rm:
  case X86::VPSUBSBZ128rm:
  case X86::VPSUBSWZ128rm:
  case X86::VPSUBUSBZ128rm:
  case X86::VPSUBUSWZ128rm:
  case X86::VPSUBWZ128rm:
  case X86::VPUNPCKHBWZ128rm:
  case X86::VPUNPCKHDQZ128rm:
  case X86::VPUNPCKHQDQZ128rm:
  case X86::VPUNPCKHWDZ128rm:
  case X86::VPUNPCKLBWZ128rm:
  case X86::VPUNPCKLDQZ128rm:
  case X86::VPUNPCKLQDQZ128rm:
  case X86::VPUNPCKLWDZ128rm:
  case X86::VPXORDZ128rm:
  case X86::VPXORQZ128rm:
  case X86::VSHUFPDZ128rmi:
  case X86::VSHUFPSZ128rmi:
  case X86::VSUBPDZ128rm:
  case X86::VSUBPSZ128rm:
  case X86::VSUBSDZrm:
  case X86::VSUBSDZrm_Int:
  case X86::VSUBSSZrm:
  case X86::VSUBSSZrm_Int:
  case X86::VUNPCKHPDZ128rm:
  case X86::VUNPCKLPDZ128rm:
  case X86::VUNPCKLPSZ128rm:
  case X86::VXORPDZ128rm:
  case X86::VXORPSZ128rm:
  case X86::VFMADD132PDZ128m:
  case X86::VFMADD132PSZ128m:
  case X86::VFMADD132SDZm:
  case X86::VFMADD132SDZm_Int:
  case X86::VFMADD132SSZm:
  case X86::VFMADD132SSZm_Int:
  case X86::VFMADD213PDZ128m:
  case X86::VFMADD213PSZ128m:
  case X86::VFMADD213SDZm:
  case X86::VFMADD213SDZm_Int:
  case X86::VFMADD213SSZm:
  case X86::VFMADD213SSZm_Int:
  case X86::VFMADD231PDZ128m:
  case X86::VFMADD231PSZ128m:
  case X86::VFMADD231SDZm:
  case X86::VFMADD231SDZm_Int:
  case X86::VFMADD231SSZm:
  case X86::VFMADD231SSZm_Int:
  case X86::VFMADDSUB132PDZ128m:
  case X86::VFMADDSUB132PSZ128m:
  case X86::VFMADDSUB213PDZ128m:
  case X86::VFMADDSUB213PSZ128m:
  case X86::VFMADDSUB231PDZ128m:
  case X86::VFMADDSUB231PSZ128m:
  case X86::VFMSUB132PDZ128m:
  case X86::VFMSUB132PSZ128m:
  case X86::VFMSUB132SDZm:
  case X86::VFMSUB132SDZm_Int:
  case X86::VFMSUB132SSZm:
  case X86::VFMSUB132SSZm_Int:
  case X86::VFMSUB213PDZ128m:
  case X86::VFMSUB213PDZ128r:
  case X86::VFMSUB213PSZ128m:
  case X86::VFMSUB213SDZm:
  case X86::VFMSUB213SDZm_Int:
  case X86::VFMSUB213SSZm:
  case X86::VFMSUB213SSZm_Int:
  case X86::VFMSUB231PDZ128m:
  case X86::VFMSUB231PSZ128m:
  case X86::VFMSUB231SDZm:
  case X86::VFMSUB231SDZm_Int:
  case X86::VFMSUB231SSZm:
  case X86::VFMSUB231SSZm_Int:
  case X86::VFMSUBADD132PDZ128m:
  case X86::VFMSUBADD132PSZ128m:
  case X86::VFMSUBADD213PDZ128m:
  case X86::VFMSUBADD231PDZ128m:
  case X86::VFMSUBADD231PSZ128m:
  case X86::VFNMADD132PDZ128m:
  case X86::VFNMADD132PSZ128m:
  case X86::VFNMADD132SDZm:
  case X86::VFNMADD132SDZm_Int:
  case X86::VFNMADD132SSZm:
  case X86::VFNMADD132SSZm_Int:
  case X86::VFNMADD213PDZ128m:
  case X86::VFNMADD213PSZ128m:
  case X86::VFNMADD213SDZm:
  case X86::VFNMADD213SDZm_Int:
  case X86::VFNMADD213SSZm:
  case X86::VFNMADD213SSZm_Int:
  case X86::VFNMADD231PDZ128m:
  case X86::VFNMADD231PSZ128m:
  case X86::VFNMADD231SDZm:
  case X86::VFNMADD231SDZm_Int:
  case X86::VFNMADD231SSZm:
  case X86::VFNMADD231SSZm_Int:
  case X86::VFNMSUB132PDZ128m:
  case X86::VFNMSUB132SDZm:
  case X86::VFNMSUB132SDZm_Int:
  case X86::VFNMSUB132SSZm:
  case X86::VFNMSUB132SSZm_Int:
  case X86::VFNMSUB213PDZ128m:
  case X86::VFNMSUB213PSZ128m:
  case X86::VFNMSUB213SDZm:
  case X86::VFNMSUB213SDZm_Int:
  case X86::VFNMSUB213SSZm:
  case X86::VFNMSUB213SSZm_Int:
  case X86::VFNMSUB231PDZ128m:
  case X86::VFNMSUB231PSZ128m:
  case X86::VFNMSUB231SDZm:
  case X86::VFNMSUB231SDZm_Int:
  case X86::VFNMSUB231SSZm:
  case X86::VFNMSUB231SSZm_Int:
  case X86::VPDPBUSDSZ128m:
  case X86::VPDPBUSDZ128m:
  case X86::VPDPWSSDSZ128m:
  case X86::VPDPWSSDZ128m:
  case X86::VRNDSCALESDZm:
  case X86::VRNDSCALESDZm_Int:
  case X86::VRNDSCALESSZm:
  case X86::VRNDSCALESSZm_Int:
  case X86::VSQRTSDZm:
  case X86::VSQRTSDZm_Int:
  case X86::VSQRTSSZm:
  case X86::VSQRTSSZm_Int:
    disp = MI.getOperand(2 + X86::AddrDisp).getImm();
    stride = 16;
    break;
  case X86::VBROADCASTI32X2Z128rm:
  case X86::VBROADCASTSSZ128rm:
  case X86::VPBROADCASTBZ128rm:
  case X86::VPBROADCASTDZ128rm:
  case X86::VPBROADCASTQZ128rm:
  case X86::VPBROADCASTWZ128rm:
  case X86::VCOMISDZrm:
  case X86::VCOMISDZrm_Int:
  case X86::VCOMISSZrm:
  case X86::VCOMISSZrm_Int:
  case X86::VCVTDQ2PDZ128rm:
  case X86::VCVTDQ2PSZ128rm:
  case X86::VCVTPD2DQZ128rm:
  case X86::VCVTPD2PSZ128rm:
  case X86::VCVTPH2PSZ128rm:
  case X86::VCVTPS2DQZ128rm:
  case X86::VCVTPS2PDZ128rm:
  case X86::VCVTSD2SI64Zrm:
  case X86::VCVTSD2SI64Zrm_Int:
  case X86::VCVTSD2SIZrm:
  case X86::VCVTSD2SIZrm_Int:
  case X86::VCVTSD2SSZrm:
  case X86::VCVTSD2SSZrm_Int:
  case X86::VCVTSS2SI64Zrm:
  case X86::VCVTSS2SI64Zrm_Int:
  case X86::VCVTSS2SIZrm:
  case X86::VCVTSS2SIZrm_Int:
  case X86::VCVTTPD2DQZ128rm:
  case X86::VCVTTPS2DQZ128rm:
  case X86::VCVTTSD2SI64Zrm:
  case X86::VCVTTSD2SI64Zrm_Int:
  case X86::VCVTTSD2SIZrm:
  case X86::VCVTTSD2SIZrm_Int:
  case X86::VCVTTSS2SI64Zrm:
  case X86::VCVTTSS2SI64Zrm_Int:
  case X86::VCVTTSS2SIZrm:
  case X86::VCVTTSS2SIZrm_Int:
  case X86::VMOV64toPQIZrm:
  case X86::VMOVAPDZ128rm:
  case X86::VMOVAPSZ128rm:
  case X86::VMOVDDUPZ128rm:
  case X86::VMOVDI2PDIZrm:
  case X86::VMOVDQA32Z128rm:
  case X86::VMOVDQA64Z128rm:
  case X86::VMOVDQU16Z128rm:
  case X86::VMOVDQU32Z128rm:
  case X86::VMOVDQU64Z128rm:
  case X86::VMOVDQU8Z128rm:
  case X86::VMOVNTDQAZ128rm:
  case X86::VMOVQI2PQIZrm:
  case X86::VMOVSDZrm:
  case X86::VMOVSDZrm_alt:
  case X86::VMOVSHDUPZ128rm:
  case X86::VMOVSLDUPZ128rm:
  case X86::VMOVSSZrm:
  case X86::VMOVSSZrm_alt:
  case X86::VMOVUPDZ128rm:
  case X86::VMOVUPSZ128rm:
  case X86::VPABSBZ128rm:
  case X86::VPABSDZ128rm:
  case X86::VPABSWZ128rm:
  case X86::VPMOVSXBDZ128rm:
  case X86::VPMOVSXBQZ128rm:
  case X86::VPMOVSXBWZ128rm:
  case X86::VPMOVSXDQZ128rm:
  case X86::VPMOVSXWDZ128rm:
  case X86::VPMOVSXWQZ128rm:
  case X86::VPMOVZXBDZ128rm:
  case X86::VPMOVZXBQZ128rm:
  case X86::VPMOVZXBWZ128rm:
  case X86::VPMOVZXDQZ128rm:
  case X86::VPMOVZXWDZ128rm:
  case X86::VPMOVZXWQZ128rm:
  case X86::VRNDSCALEPDZ128rmi:
  case X86::VRNDSCALEPSZ128rmi:
  case X86::VUCOMISDZrm:
  case X86::VUCOMISDZrm_Int:
  case X86::VUCOMISSZrm:
  case X86::VUCOMISSZrm_Int:
  case X86::VPSHUFDZ128mi:
  case X86::VPSHUFHWZ128mi:
  case X86::VPSHUFLWZ128mi:
  case X86::VSQRTPDZ128m:
  case X86::VSQRTPSZ128m:
  case X86::VPERMILPDZ128mi:
  case X86::VPERMILPSZ128mi:
    disp = MI.getOperand(1 + X86::AddrDisp).getImm();
    stride = 16;
    break;
  case X86::VCVTPS2PHZ128mr:
  case X86::VEXTRACTPSZmr:
  case X86::VMOVAPDZ128mr:
  case X86::VMOVAPSZ128mr:
  case X86::VMOVDQA32Z128mr:
  case X86::VMOVDQA64Z128mr:
  case X86::VMOVDQU16Z128mr:
  case X86::VMOVDQU32Z128mr:
  case X86::VMOVDQU64Z128mr:
  case X86::VMOVDQU8Z128mr:
  case X86::VMOVHPDZ128mr:
  case X86::VMOVHPSZ128mr:
  case X86::VMOVLPDZ128mr:
  case X86::VMOVLPSZ128mr:
  case X86::VMOVNTDQZ128mr:
  case X86::VMOVNTPDZ128mr:
  case X86::VMOVNTPSZ128mr:
  case X86::VMOVPDI2DIZmr:
  case X86::VMOVPQI2QIZmr:
  case X86::VMOVPQIto64Zmr:
  case X86::VMOVSDZmr:
  case X86::VMOVSSZmr:
  case X86::VMOVUPDZ128mr:
  case X86::VMOVUPSZ128mr:
  case X86::VPEXTRBZmr:
  case X86::VPEXTRDZmr:
  case X86::VPEXTRQZmr:
  case X86::VPEXTRWZmr:
    disp = MI.getOperand(0 + X86::AddrDisp).getImm();
    stride = 16;
    break;
  case X86::VADDPDZ256rm:
  case X86::VADDPSZ256rm:
  case X86::VAESDECLASTZ256rm:
  case X86::VAESDECZ256rm:
  case X86::VAESENCLASTZ256rm:
  case X86::VAESENCZ256rm:
  case X86::VANDNPDZ256rm:
  case X86::VANDNPSZ256rm:
  case X86::VANDPDZ256rm:
  case X86::VANDPSZ256rm:
  case X86::VDIVPDZ256rm:
  case X86::VDIVPSZ256rm:
  case X86::VGF2P8AFFINEINVQBZ256rmi:
  case X86::VGF2P8AFFINEQBZ256rmi:
  case X86::VGF2P8MULBZ256rm:
  case X86::VINSERTF32x4Z256rm:
  case X86::VINSERTF64x2Z256rm:
  case X86::VINSERTI32x4Z256rm:
  case X86::VINSERTI64x2Z256rm:
  case X86::VMAXCPDZ256rm:
  case X86::VMAXCPSZ256rm:
  case X86::VMAXPDZ256rm:
  case X86::VMAXPSZ256rm:
  case X86::VMINCPDZ256rm:
  case X86::VMINCPSZ256rm:
  case X86::VMINPDZ256rm:
  case X86::VMINPSZ256rm:
  case X86::VMULPDZ256rm:
  case X86::VMULPSZ256rm:
  case X86::VORPDZ256rm:
  case X86::VORPSZ256rm:
  case X86::VPACKSSDWZ256rm:
  case X86::VPACKSSWBZ256rm:
  case X86::VPACKUSDWZ256rm:
  case X86::VPACKUSWBZ256rm:
  case X86::VPADDBZ256rm:
  case X86::VPADDDZ256rm:
  case X86::VPADDQZ256rm:
  case X86::VPADDSBZ256rm:
  case X86::VPADDSWZ256rm:
  case X86::VPADDUSBZ256rm:
  case X86::VPADDUSWZ256rm:
  case X86::VPADDWZ256rm:
  case X86::VPALIGNRZ256rmi:
  case X86::VPANDDZ256rm:
  case X86::VPANDNDZ256rm:
  case X86::VPANDNQZ256rm:
  case X86::VPANDQZ256rm:
  case X86::VPAVGBZ256rm:
  case X86::VPAVGWZ256rm:
  case X86::VPCLMULQDQZ256rm:
  case X86::VPERMDZ256rm:
  case X86::VPERMILPDZ256rm:
  case X86::VPERMILPSZ256rm:
  case X86::VPERMPSZ256rm:
  case X86::VPMADDUBSWZ256rm:
  case X86::VPMADDWDZ256rm:
  case X86::VPMAXSBZ256rm:
  case X86::VPMAXSDZ256rm:
  case X86::VPMAXSWZ256rm:
  case X86::VPMAXUBZ256rm:
  case X86::VPMAXUDZ256rm:
  case X86::VPMAXUWZ256rm:
  case X86::VPMINSBZ256rm:
  case X86::VPMINSDZ256rm:
  case X86::VPMINSWZ256rm:
  case X86::VPMINUBZ256rm:
  case X86::VPMINUDZ256rm:
  case X86::VPMINUWZ256rm:
  case X86::VPMULDQZ256rm:
  case X86::VPMULHRSWZ256rm:
  case X86::VPMULHUWZ256rm:
  case X86::VPMULHWZ256rm:
  case X86::VPMULLDZ256rm:
  case X86::VPMULLWZ256rm:
  case X86::VPMULUDQZ256rm:
  case X86::VPORDZ256rm:
  case X86::VPORQZ256rm:
  case X86::VPSADBWZ256rm:
  case X86::VPSHUFBZ256rm:
  case X86::VPSLLDZ256rm:
  case X86::VPSLLQZ256rm:
  case X86::VPSLLVDZ256rm:
  case X86::VPSLLVQZ256rm:
  case X86::VPSLLWZ256rm:
  case X86::VPSRADZ256rm:
  case X86::VPSRAVDZ256rm:
  case X86::VPSRAWZ256rm:
  case X86::VPSRLVDZ256rm:
  case X86::VPSRLVQZ256rm:
  case X86::VPSRLWZ256rm:
  case X86::VPSUBBZ256rm:
  case X86::VPSUBDZ256rm:
  case X86::VPSUBQZ256rm:
  case X86::VPSUBSBZ256rm:
  case X86::VPSUBSWZ256rm:
  case X86::VPSUBUSBZ256rm:
  case X86::VPSUBUSWZ256rm:
  case X86::VPSUBWZ256rm:
  case X86::VPUNPCKHBWZ256rm:
  case X86::VPUNPCKHDQZ256rm:
  case X86::VPUNPCKHQDQZ256rm:
  case X86::VPUNPCKHWDZ256rm:
  case X86::VPUNPCKLBWZ256rm:
  case X86::VPUNPCKLDQZ256rm:
  case X86::VPUNPCKLQDQZ256rm:
  case X86::VPUNPCKLWDZ256rm:
  case X86::VPXORDZ256rm:
  case X86::VPXORQZ256rm:
  case X86::VSHUFF32X4Z256rmi:
  case X86::VSHUFF64X2Z256rmi:
  case X86::VSHUFI32X4Z256rmi:
  case X86::VSHUFI64X2Z256rmi:
  case X86::VSHUFPDZ256rmi:
  case X86::VSHUFPSZ256rmi:
  case X86::VSUBPDZ256rm:
  case X86::VSUBPSZ256rm:
  case X86::VUNPCKHPDZ256rm:
  case X86::VUNPCKHPSZ256rm:
  case X86::VUNPCKLPDZ256rm:
  case X86::VUNPCKLPSZ256rm:
  case X86::VXORPDZ256rm:
  case X86::VXORPSZ256rm:
  case X86::VFMADD132PDZ256m:
  case X86::VFMADD132PSZ256m:
  case X86::VFMADD213PDZ256m:
  case X86::VFMADD213PSZ256m:
  case X86::VFMADD231PDZ256m:
  case X86::VFMADD231PSZ256m:
  case X86::VFMADDSUB132PDZ256m:
  case X86::VFMADDSUB132PSZ256m:
  case X86::VFMADDSUB213PDZ256m:
  case X86::VFMADDSUB213PSZ256m:
  case X86::VFMADDSUB231PSZ256m:
  case X86::VFMSUB132PDZ256m:
  case X86::VFMSUB132PSZ256m:
  case X86::VFMSUB213PDZ256m:
  case X86::VFMSUB213PSZ256m:
  case X86::VFMSUB231PDZ256m:
  case X86::VFMSUB231PSZ256m:
  case X86::VFMSUBADD132PDZ256m:
  case X86::VFMSUBADD132PSZ256m:
  case X86::VFMSUBADD213PDZ256m:
  case X86::VFMSUBADD213PSZ256m:
  case X86::VFMSUBADD231PDZ256m:
  case X86::VFMSUBADD231PSZ256m:
  case X86::VFNMADD132PDZ256m:
  case X86::VFNMADD132PSZ256m:
  case X86::VFNMADD213PDZ256m:
  case X86::VFNMADD213PSZ256m:
  case X86::VFNMADD231PDZ256m:
  case X86::VFNMADD231PSZ256m:
  case X86::VFNMSUB132PDZ256m:
  case X86::VFNMSUB132PSZ256m:
  case X86::VFNMSUB213PDZ256m:
  case X86::VFNMSUB213PSZ256m:
  case X86::VFNMSUB231PDZ256m:
  case X86::VFNMSUB231PSZ256m:
  case X86::VPDPBUSDSZ256m:
  case X86::VPDPBUSDZ256m:
  case X86::VPDPWSSDSZ256m:
  case X86::VPDPWSSDZ256m:
	disp = MI.getOperand(2 + X86::AddrDisp).getImm();
    stride = 32;
    break;
  case X86::VBROADCASTF32X2Z256rm:
  case X86::VBROADCASTF32X4Z256rm:
  case X86::VBROADCASTF64X2Z128rm:
  case X86::VBROADCASTI32X2Z256rm:
  case X86::VBROADCASTI32X4Z256rm:
  case X86::VBROADCASTI64X2Z128rm:
  case X86::VBROADCASTSDZ256rm:
  case X86::VBROADCASTSSZ256rm:
  case X86::VCVTDQ2PDZ256rm:
  case X86::VCVTDQ2PSZ256rm:
  case X86::VCVTPD2DQZ256rm:
  case X86::VCVTPD2PSZ256rm:
  case X86::VCVTPH2PSZ256rm:
  case X86::VCVTPS2DQZ256rm:
  case X86::VCVTPS2PDZ256rm:
  case X86::VCVTTPD2DQZ256rm:
  case X86::VCVTTPS2DQZ256rm:
  case X86::VMOVAPDZ256rm:
  case X86::VMOVAPSZ256rm:
  case X86::VMOVDDUPZ256rm:
  case X86::VMOVDQA32Z256rm:
  case X86::VMOVDQA64Z256rm:
  case X86::VMOVDQU16Z256rm:
  case X86::VMOVDQU32Z256rm:
  case X86::VMOVDQU64Z256rm:
  case X86::VMOVDQU8Z256rm:
  case X86::VMOVNTDQAZ256rm:
  case X86::VMOVSHDUPZ256rm:
  case X86::VMOVSLDUPZ256rm:
  case X86::VMOVUPDZ256rm:
  case X86::VMOVUPSZ256rm:
  case X86::VPABSBZ256rm:
  case X86::VPABSDZ256rm:
  case X86::VPABSWZ256rm:
  case X86::VPBROADCASTBZ256rm:
  case X86::VPBROADCASTDZ256rm:
  case X86::VPBROADCASTQZ256rm:
  case X86::VPBROADCASTWZ256rm:
  case X86::VPMOVSXBDZ256rm:
  case X86::VPMOVSXBQZ256rm:
  case X86::VPMOVSXBWZ256rm:
  case X86::VPMOVSXDQZ256rm:
  case X86::VPMOVSXWDZ256rm:
  case X86::VPMOVZXBDZ256rm:
  case X86::VPMOVZXBQZ256rm:
  case X86::VPMOVZXBWZ256rm:
  case X86::VPMOVZXDQZ256rm:
  case X86::VPMOVZXWDZ256rm:
  case X86::VPMOVZXWQZ256rm:
  case X86::VRNDSCALEPDZ256rmi:
  case X86::VRNDSCALEPSZ256rmi:
  case X86::VPERMPDZ256mi:
  case X86::VPERMQZ256mi:
  case X86::VPSHUFDZ256mi:
  case X86::VPSHUFHWZ256mi:
  case X86::VPSHUFLWZ256mi:
  case X86::VSQRTPDZ256m:
  case X86::VSQRTPSZ256m:
  case X86::VPERMILPDZ256mi:
  case X86::VPERMILPSZ256mi:
    disp = MI.getOperand(1 + X86::AddrDisp).getImm();
    stride = 32;
    break;
  case X86::VCVTPS2PHZ256mr:
  case X86::VEXTRACTF32x4Z256mr:
  case X86::VEXTRACTF64x2Z256mr:
  case X86::VEXTRACTI32x4Z256mr:
  case X86::VEXTRACTI64x2Z256mr:
  case X86::VMOVAPDZ256mr:
  case X86::VMOVAPSZ256mr:
  case X86::VMOVDQA32Z256mr:
  case X86::VMOVDQA64Z256mr:
  case X86::VMOVDQU16Z256mr:
  case X86::VMOVDQU32Z256mr:
  case X86::VMOVDQU64Z256mr:
  case X86::VMOVDQU8Z256mr:
  case X86::VMOVNTDQZ256mr:
  case X86::VMOVNTPDZ256mr:
  case X86::VMOVNTPSZ256mr:
  case X86::VMOVUPDZ256mr:
  case X86::VMOVUPSZ256mr:
    disp = MI.getOperand(0 + X86::AddrDisp).getImm();
    stride = 32;
    break;
  }
  // Don't convert Evex encoding to Vex if disp8 encoding will be smaller
  if ((disp < -128 || disp > 127) && !(disp % stride) && (disp >= stride * -128 && disp <= stride * 127))
    return true;

  return false;
}

// Do any custom cleanup needed to finalize the conversion.
static bool performCustomAdjustments(MachineInstr &MI, unsigned NewOpc,
                                     const X86Subtarget *ST) {
  (void)NewOpc;
  unsigned Opc = MI.getOpcode();
  switch (Opc) {
  case X86::VALIGNDZ128rri:
  case X86::VALIGNDZ128rmi:
  case X86::VALIGNQZ128rri:
  case X86::VALIGNQZ128rmi: {
    assert((NewOpc == X86::VPALIGNRrri || NewOpc == X86::VPALIGNRrmi) &&
           "Unexpected new opcode!");
    unsigned Scale = (Opc == X86::VALIGNQZ128rri ||
                      Opc == X86::VALIGNQZ128rmi) ? 8 : 4;
    MachineOperand &Imm = MI.getOperand(MI.getNumExplicitOperands()-1);
    Imm.setImm(Imm.getImm() * Scale);
    break;
  }
  case X86::VSHUFF32X4Z256rmi:
  case X86::VSHUFF32X4Z256rri:
  case X86::VSHUFF64X2Z256rmi:
  case X86::VSHUFF64X2Z256rri:
  case X86::VSHUFI32X4Z256rmi:
  case X86::VSHUFI32X4Z256rri:
  case X86::VSHUFI64X2Z256rmi:
  case X86::VSHUFI64X2Z256rri: {
    assert((NewOpc == X86::VPERM2F128rr || NewOpc == X86::VPERM2I128rr ||
            NewOpc == X86::VPERM2F128rm || NewOpc == X86::VPERM2I128rm) &&
           "Unexpected new opcode!");
    MachineOperand &Imm = MI.getOperand(MI.getNumExplicitOperands()-1);
    int64_t ImmVal = Imm.getImm();
    // Set bit 5, move bit 1 to bit 4, copy bit 0.
    Imm.setImm(0x20 | ((ImmVal & 2) << 3) | (ImmVal & 1));
    break;
  }
  case X86::VRNDSCALEPDZ128rri:
  case X86::VRNDSCALEPDZ128rmi:
  case X86::VRNDSCALEPSZ128rri:
  case X86::VRNDSCALEPSZ128rmi:
  case X86::VRNDSCALEPDZ256rri:
  case X86::VRNDSCALEPDZ256rmi:
  case X86::VRNDSCALEPSZ256rri:
  case X86::VRNDSCALEPSZ256rmi:
  case X86::VRNDSCALESDZr:
  case X86::VRNDSCALESDZm:
  case X86::VRNDSCALESSZr:
  case X86::VRNDSCALESSZm:
  case X86::VRNDSCALESDZr_Int:
  case X86::VRNDSCALESDZm_Int:
  case X86::VRNDSCALESSZr_Int:
  case X86::VRNDSCALESSZm_Int:
    const MachineOperand &Imm = MI.getOperand(MI.getNumExplicitOperands()-1);
    int64_t ImmVal = Imm.getImm();
    // Ensure that only bits 3:0 of the immediate are used.
    if ((ImmVal & 0xf) != ImmVal)
      return false;
    break;
  }

  return true;
}


// For EVEX instructions that can be encoded using VEX encoding
// replace them by the VEX encoding in order to reduce size.
bool EvexToVexInstPass::CompressEvexToVexImpl(MachineInstr &MI) const {
  // VEX format.
  // # of bytes: 0,2,3  1      1      0,1   0,1,2,4  0,1
  //  [Prefixes] [VEX]  OPCODE ModR/M [SIB] [DISP]  [IMM]
  //
  // EVEX format.
  //  # of bytes: 4    1      1      1      4       / 1         1
  //  [Prefixes]  EVEX Opcode ModR/M [SIB] [Disp32] / [Disp8*N] [Immediate]

  const MCInstrDesc &Desc = MI.getDesc();

  // Check for EVEX instructions only.
  if ((Desc.TSFlags & X86II::EncodingMask) != X86II::EVEX)
    return false;

  // Check for EVEX instructions with mask or broadcast as in these cases
  // the EVEX prefix is needed in order to carry this information
  // thus preventing the transformation to VEX encoding.
  if (Desc.TSFlags & (X86II::EVEX_K | X86II::EVEX_B))
    return false;

  // Check for EVEX instructions with L2 set. These instructions are 512-bits
  // and can't be converted to VEX.
  if (Desc.TSFlags & X86II::EVEX_L2)
    return false;

#ifndef NDEBUG
  // Make sure the tables are sorted.
  static std::atomic<bool> TableChecked(false);
  if (!TableChecked.load(std::memory_order_relaxed)) {
    assert(llvm::is_sorted(X86EvexToVex128CompressTable) &&
           "X86EvexToVex128CompressTable is not sorted!");
    assert(llvm::is_sorted(X86EvexToVex256CompressTable) &&
           "X86EvexToVex256CompressTable is not sorted!");
    TableChecked.store(true, std::memory_order_relaxed);
  }
#endif

  // Use the VEX.L bit to select the 128 or 256-bit table.
  ArrayRef<X86EvexToVexCompressTableEntry> Table =
    (Desc.TSFlags & X86II::VEX_L) ? makeArrayRef(X86EvexToVex256CompressTable)
                                  : makeArrayRef(X86EvexToVex128CompressTable);

  const auto *I = llvm::lower_bound(Table, MI.getOpcode());
  if (I == Table.end() || I->EvexOpcode != MI.getOpcode())
    return false;

  unsigned NewOpc = I->VexOpcode;

  if (usesExtendedRegister(MI))
    return false;

  if (usesDisp8Compression(MI))
    return false;

  if (!CheckVEXInstPredicate(MI, ST))
    return false;

  if (!performCustomAdjustments(MI, NewOpc, ST))
    return false;

  MI.setDesc(TII->get(NewOpc));
  MI.setAsmPrinterFlag(X86::AC_EVEX_2_VEX);
  return true;
}

INITIALIZE_PASS(EvexToVexInstPass, EVEX2VEX_NAME, EVEX2VEX_DESC, false, false)

FunctionPass *llvm::createX86EvexToVexInsts() {
  return new EvexToVexInstPass();
}
