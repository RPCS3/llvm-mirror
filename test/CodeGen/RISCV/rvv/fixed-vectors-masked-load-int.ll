; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py
; RUN: llc -mtriple=riscv32 -mattr=+m,+experimental-v -verify-machineinstrs -riscv-v-vector-bits-min=128 -verify-machineinstrs < %s | FileCheck %s --check-prefixes=CHECK,RV32
; RUN: llc -mtriple=riscv64 -mattr=+m,+experimental-v -verify-machineinstrs -riscv-v-vector-bits-min=128 -verify-machineinstrs < %s | FileCheck %s --check-prefixes=CHECK,RV64

define void @masked_load_v1i8(<1 x i8>* %a, <1 x i8>* %m_ptr, <1 x i8>* %res_ptr) nounwind {
; CHECK-LABEL: masked_load_v1i8:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetivli a3, 1, e8,m1,ta,mu
; CHECK-NEXT:    vle8.v v25, (a1)
; CHECK-NEXT:    vmseq.vi v0, v25, 0
; CHECK-NEXT:    vle8.v v25, (a0), v0.t
; CHECK-NEXT:    vse8.v v25, (a2)
; CHECK-NEXT:    ret
  %m = load <1 x i8>, <1 x i8>* %m_ptr
  %mask = icmp eq <1 x i8> %m, zeroinitializer
  %load = call <1 x i8> @llvm.masked.load.v1i8(<1 x i8>* %a, i32 8, <1 x i1> %mask, <1 x i8> undef)
  store <1 x i8> %load, <1 x i8>* %res_ptr
  ret void
}
declare <1 x i8> @llvm.masked.load.v1i8(<1 x i8>*, i32, <1 x i1>, <1 x i8>)

define void @masked_load_v1i16(<1 x i16>* %a, <1 x i16>* %m_ptr, <1 x i16>* %res_ptr) nounwind {
; CHECK-LABEL: masked_load_v1i16:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetivli a3, 1, e16,m1,ta,mu
; CHECK-NEXT:    vle16.v v25, (a1)
; CHECK-NEXT:    vmseq.vi v0, v25, 0
; CHECK-NEXT:    vle16.v v25, (a0), v0.t
; CHECK-NEXT:    vse16.v v25, (a2)
; CHECK-NEXT:    ret
  %m = load <1 x i16>, <1 x i16>* %m_ptr
  %mask = icmp eq <1 x i16> %m, zeroinitializer
  %load = call <1 x i16> @llvm.masked.load.v1i16(<1 x i16>* %a, i32 8, <1 x i1> %mask, <1 x i16> undef)
  store <1 x i16> %load, <1 x i16>* %res_ptr
  ret void
}
declare <1 x i16> @llvm.masked.load.v1i16(<1 x i16>*, i32, <1 x i1>, <1 x i16>)

define void @masked_load_v1i32(<1 x i32>* %a, <1 x i32>* %m_ptr, <1 x i32>* %res_ptr) nounwind {
; CHECK-LABEL: masked_load_v1i32:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetivli a3, 1, e32,m1,ta,mu
; CHECK-NEXT:    vle32.v v25, (a1)
; CHECK-NEXT:    vmseq.vi v0, v25, 0
; CHECK-NEXT:    vle32.v v25, (a0), v0.t
; CHECK-NEXT:    vse32.v v25, (a2)
; CHECK-NEXT:    ret
  %m = load <1 x i32>, <1 x i32>* %m_ptr
  %mask = icmp eq <1 x i32> %m, zeroinitializer
  %load = call <1 x i32> @llvm.masked.load.v1i32(<1 x i32>* %a, i32 8, <1 x i1> %mask, <1 x i32> undef)
  store <1 x i32> %load, <1 x i32>* %res_ptr
  ret void
}
declare <1 x i32> @llvm.masked.load.v1i32(<1 x i32>*, i32, <1 x i1>, <1 x i32>)

define void @masked_load_v1i64(<1 x i64>* %a, <1 x i64>* %m_ptr, <1 x i64>* %res_ptr) nounwind {
; RV32-LABEL: masked_load_v1i64:
; RV32:       # %bb.0:
; RV32-NEXT:    vsetivli a3, 1, e64,m1,ta,mu
; RV32-NEXT:    vle64.v v25, (a1)
; RV32-NEXT:    vsetivli a1, 2, e32,m1,ta,mu
; RV32-NEXT:    vmv.v.i v26, 0
; RV32-NEXT:    vsetivli a1, 1, e64,m1,ta,mu
; RV32-NEXT:    vmseq.vv v0, v25, v26
; RV32-NEXT:    vle64.v v25, (a0), v0.t
; RV32-NEXT:    vse64.v v25, (a2)
; RV32-NEXT:    ret
;
; RV64-LABEL: masked_load_v1i64:
; RV64:       # %bb.0:
; RV64-NEXT:    vsetivli a3, 1, e64,m1,ta,mu
; RV64-NEXT:    vle64.v v25, (a1)
; RV64-NEXT:    vmseq.vi v0, v25, 0
; RV64-NEXT:    vle64.v v25, (a0), v0.t
; RV64-NEXT:    vse64.v v25, (a2)
; RV64-NEXT:    ret
  %m = load <1 x i64>, <1 x i64>* %m_ptr
  %mask = icmp eq <1 x i64> %m, zeroinitializer
  %load = call <1 x i64> @llvm.masked.load.v1i64(<1 x i64>* %a, i32 8, <1 x i1> %mask, <1 x i64> undef)
  store <1 x i64> %load, <1 x i64>* %res_ptr
  ret void
}
declare <1 x i64> @llvm.masked.load.v1i64(<1 x i64>*, i32, <1 x i1>, <1 x i64>)

define void @masked_load_v2i8(<2 x i8>* %a, <2 x i8>* %m_ptr, <2 x i8>* %res_ptr) nounwind {
; CHECK-LABEL: masked_load_v2i8:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetivli a3, 2, e8,m1,ta,mu
; CHECK-NEXT:    vle8.v v25, (a1)
; CHECK-NEXT:    vmseq.vi v0, v25, 0
; CHECK-NEXT:    vle8.v v25, (a0), v0.t
; CHECK-NEXT:    vse8.v v25, (a2)
; CHECK-NEXT:    ret
  %m = load <2 x i8>, <2 x i8>* %m_ptr
  %mask = icmp eq <2 x i8> %m, zeroinitializer
  %load = call <2 x i8> @llvm.masked.load.v2i8(<2 x i8>* %a, i32 8, <2 x i1> %mask, <2 x i8> undef)
  store <2 x i8> %load, <2 x i8>* %res_ptr
  ret void
}
declare <2 x i8> @llvm.masked.load.v2i8(<2 x i8>*, i32, <2 x i1>, <2 x i8>)

define void @masked_load_v2i16(<2 x i16>* %a, <2 x i16>* %m_ptr, <2 x i16>* %res_ptr) nounwind {
; CHECK-LABEL: masked_load_v2i16:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetivli a3, 2, e16,m1,ta,mu
; CHECK-NEXT:    vle16.v v25, (a1)
; CHECK-NEXT:    vmseq.vi v0, v25, 0
; CHECK-NEXT:    vle16.v v25, (a0), v0.t
; CHECK-NEXT:    vse16.v v25, (a2)
; CHECK-NEXT:    ret
  %m = load <2 x i16>, <2 x i16>* %m_ptr
  %mask = icmp eq <2 x i16> %m, zeroinitializer
  %load = call <2 x i16> @llvm.masked.load.v2i16(<2 x i16>* %a, i32 8, <2 x i1> %mask, <2 x i16> undef)
  store <2 x i16> %load, <2 x i16>* %res_ptr
  ret void
}
declare <2 x i16> @llvm.masked.load.v2i16(<2 x i16>*, i32, <2 x i1>, <2 x i16>)

define void @masked_load_v2i32(<2 x i32>* %a, <2 x i32>* %m_ptr, <2 x i32>* %res_ptr) nounwind {
; CHECK-LABEL: masked_load_v2i32:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetivli a3, 2, e32,m1,ta,mu
; CHECK-NEXT:    vle32.v v25, (a1)
; CHECK-NEXT:    vmseq.vi v0, v25, 0
; CHECK-NEXT:    vle32.v v25, (a0), v0.t
; CHECK-NEXT:    vse32.v v25, (a2)
; CHECK-NEXT:    ret
  %m = load <2 x i32>, <2 x i32>* %m_ptr
  %mask = icmp eq <2 x i32> %m, zeroinitializer
  %load = call <2 x i32> @llvm.masked.load.v2i32(<2 x i32>* %a, i32 8, <2 x i1> %mask, <2 x i32> undef)
  store <2 x i32> %load, <2 x i32>* %res_ptr
  ret void
}
declare <2 x i32> @llvm.masked.load.v2i32(<2 x i32>*, i32, <2 x i1>, <2 x i32>)

define void @masked_load_v2i64(<2 x i64>* %a, <2 x i64>* %m_ptr, <2 x i64>* %res_ptr) nounwind {
; RV32-LABEL: masked_load_v2i64:
; RV32:       # %bb.0:
; RV32-NEXT:    vsetivli a3, 2, e64,m1,ta,mu
; RV32-NEXT:    vle64.v v25, (a1)
; RV32-NEXT:    vsetivli a1, 4, e32,m1,ta,mu
; RV32-NEXT:    vmv.v.i v26, 0
; RV32-NEXT:    vsetivli a1, 2, e64,m1,ta,mu
; RV32-NEXT:    vmseq.vv v0, v25, v26
; RV32-NEXT:    vle64.v v25, (a0), v0.t
; RV32-NEXT:    vse64.v v25, (a2)
; RV32-NEXT:    ret
;
; RV64-LABEL: masked_load_v2i64:
; RV64:       # %bb.0:
; RV64-NEXT:    vsetivli a3, 2, e64,m1,ta,mu
; RV64-NEXT:    vle64.v v25, (a1)
; RV64-NEXT:    vmseq.vi v0, v25, 0
; RV64-NEXT:    vle64.v v25, (a0), v0.t
; RV64-NEXT:    vse64.v v25, (a2)
; RV64-NEXT:    ret
  %m = load <2 x i64>, <2 x i64>* %m_ptr
  %mask = icmp eq <2 x i64> %m, zeroinitializer
  %load = call <2 x i64> @llvm.masked.load.v2i64(<2 x i64>* %a, i32 8, <2 x i1> %mask, <2 x i64> undef)
  store <2 x i64> %load, <2 x i64>* %res_ptr
  ret void
}
declare <2 x i64> @llvm.masked.load.v2i64(<2 x i64>*, i32, <2 x i1>, <2 x i64>)

define void @masked_load_v4i8(<4 x i8>* %a, <4 x i8>* %m_ptr, <4 x i8>* %res_ptr) nounwind {
; CHECK-LABEL: masked_load_v4i8:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetivli a3, 4, e8,m1,ta,mu
; CHECK-NEXT:    vle8.v v25, (a1)
; CHECK-NEXT:    vmseq.vi v0, v25, 0
; CHECK-NEXT:    vle8.v v25, (a0), v0.t
; CHECK-NEXT:    vse8.v v25, (a2)
; CHECK-NEXT:    ret
  %m = load <4 x i8>, <4 x i8>* %m_ptr
  %mask = icmp eq <4 x i8> %m, zeroinitializer
  %load = call <4 x i8> @llvm.masked.load.v4i8(<4 x i8>* %a, i32 8, <4 x i1> %mask, <4 x i8> undef)
  store <4 x i8> %load, <4 x i8>* %res_ptr
  ret void
}
declare <4 x i8> @llvm.masked.load.v4i8(<4 x i8>*, i32, <4 x i1>, <4 x i8>)

define void @masked_load_v4i16(<4 x i16>* %a, <4 x i16>* %m_ptr, <4 x i16>* %res_ptr) nounwind {
; CHECK-LABEL: masked_load_v4i16:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetivli a3, 4, e16,m1,ta,mu
; CHECK-NEXT:    vle16.v v25, (a1)
; CHECK-NEXT:    vmseq.vi v0, v25, 0
; CHECK-NEXT:    vle16.v v25, (a0), v0.t
; CHECK-NEXT:    vse16.v v25, (a2)
; CHECK-NEXT:    ret
  %m = load <4 x i16>, <4 x i16>* %m_ptr
  %mask = icmp eq <4 x i16> %m, zeroinitializer
  %load = call <4 x i16> @llvm.masked.load.v4i16(<4 x i16>* %a, i32 8, <4 x i1> %mask, <4 x i16> undef)
  store <4 x i16> %load, <4 x i16>* %res_ptr
  ret void
}
declare <4 x i16> @llvm.masked.load.v4i16(<4 x i16>*, i32, <4 x i1>, <4 x i16>)

define void @masked_load_v4i32(<4 x i32>* %a, <4 x i32>* %m_ptr, <4 x i32>* %res_ptr) nounwind {
; CHECK-LABEL: masked_load_v4i32:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetivli a3, 4, e32,m1,ta,mu
; CHECK-NEXT:    vle32.v v25, (a1)
; CHECK-NEXT:    vmseq.vi v0, v25, 0
; CHECK-NEXT:    vle32.v v25, (a0), v0.t
; CHECK-NEXT:    vse32.v v25, (a2)
; CHECK-NEXT:    ret
  %m = load <4 x i32>, <4 x i32>* %m_ptr
  %mask = icmp eq <4 x i32> %m, zeroinitializer
  %load = call <4 x i32> @llvm.masked.load.v4i32(<4 x i32>* %a, i32 8, <4 x i1> %mask, <4 x i32> undef)
  store <4 x i32> %load, <4 x i32>* %res_ptr
  ret void
}
declare <4 x i32> @llvm.masked.load.v4i32(<4 x i32>*, i32, <4 x i1>, <4 x i32>)

define void @masked_load_v4i64(<4 x i64>* %a, <4 x i64>* %m_ptr, <4 x i64>* %res_ptr) nounwind {
; RV32-LABEL: masked_load_v4i64:
; RV32:       # %bb.0:
; RV32-NEXT:    vsetivli a3, 4, e64,m2,ta,mu
; RV32-NEXT:    vle64.v v26, (a1)
; RV32-NEXT:    vsetivli a1, 8, e32,m2,ta,mu
; RV32-NEXT:    vmv.v.i v28, 0
; RV32-NEXT:    vsetivli a1, 4, e64,m2,ta,mu
; RV32-NEXT:    vmseq.vv v0, v26, v28
; RV32-NEXT:    vle64.v v26, (a0), v0.t
; RV32-NEXT:    vse64.v v26, (a2)
; RV32-NEXT:    ret
;
; RV64-LABEL: masked_load_v4i64:
; RV64:       # %bb.0:
; RV64-NEXT:    vsetivli a3, 4, e64,m2,ta,mu
; RV64-NEXT:    vle64.v v26, (a1)
; RV64-NEXT:    vmseq.vi v0, v26, 0
; RV64-NEXT:    vle64.v v26, (a0), v0.t
; RV64-NEXT:    vse64.v v26, (a2)
; RV64-NEXT:    ret
  %m = load <4 x i64>, <4 x i64>* %m_ptr
  %mask = icmp eq <4 x i64> %m, zeroinitializer
  %load = call <4 x i64> @llvm.masked.load.v4i64(<4 x i64>* %a, i32 8, <4 x i1> %mask, <4 x i64> undef)
  store <4 x i64> %load, <4 x i64>* %res_ptr
  ret void
}
declare <4 x i64> @llvm.masked.load.v4i64(<4 x i64>*, i32, <4 x i1>, <4 x i64>)

define void @masked_load_v8i8(<8 x i8>* %a, <8 x i8>* %m_ptr, <8 x i8>* %res_ptr) nounwind {
; CHECK-LABEL: masked_load_v8i8:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetivli a3, 8, e8,m1,ta,mu
; CHECK-NEXT:    vle8.v v25, (a1)
; CHECK-NEXT:    vmseq.vi v0, v25, 0
; CHECK-NEXT:    vle8.v v25, (a0), v0.t
; CHECK-NEXT:    vse8.v v25, (a2)
; CHECK-NEXT:    ret
  %m = load <8 x i8>, <8 x i8>* %m_ptr
  %mask = icmp eq <8 x i8> %m, zeroinitializer
  %load = call <8 x i8> @llvm.masked.load.v8i8(<8 x i8>* %a, i32 8, <8 x i1> %mask, <8 x i8> undef)
  store <8 x i8> %load, <8 x i8>* %res_ptr
  ret void
}
declare <8 x i8> @llvm.masked.load.v8i8(<8 x i8>*, i32, <8 x i1>, <8 x i8>)

define void @masked_load_v8i16(<8 x i16>* %a, <8 x i16>* %m_ptr, <8 x i16>* %res_ptr) nounwind {
; CHECK-LABEL: masked_load_v8i16:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetivli a3, 8, e16,m1,ta,mu
; CHECK-NEXT:    vle16.v v25, (a1)
; CHECK-NEXT:    vmseq.vi v0, v25, 0
; CHECK-NEXT:    vle16.v v25, (a0), v0.t
; CHECK-NEXT:    vse16.v v25, (a2)
; CHECK-NEXT:    ret
  %m = load <8 x i16>, <8 x i16>* %m_ptr
  %mask = icmp eq <8 x i16> %m, zeroinitializer
  %load = call <8 x i16> @llvm.masked.load.v8i16(<8 x i16>* %a, i32 8, <8 x i1> %mask, <8 x i16> undef)
  store <8 x i16> %load, <8 x i16>* %res_ptr
  ret void
}
declare <8 x i16> @llvm.masked.load.v8i16(<8 x i16>*, i32, <8 x i1>, <8 x i16>)

define void @masked_load_v8i32(<8 x i32>* %a, <8 x i32>* %m_ptr, <8 x i32>* %res_ptr) nounwind {
; CHECK-LABEL: masked_load_v8i32:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetivli a3, 8, e32,m2,ta,mu
; CHECK-NEXT:    vle32.v v26, (a1)
; CHECK-NEXT:    vmseq.vi v0, v26, 0
; CHECK-NEXT:    vle32.v v26, (a0), v0.t
; CHECK-NEXT:    vse32.v v26, (a2)
; CHECK-NEXT:    ret
  %m = load <8 x i32>, <8 x i32>* %m_ptr
  %mask = icmp eq <8 x i32> %m, zeroinitializer
  %load = call <8 x i32> @llvm.masked.load.v8i32(<8 x i32>* %a, i32 8, <8 x i1> %mask, <8 x i32> undef)
  store <8 x i32> %load, <8 x i32>* %res_ptr
  ret void
}
declare <8 x i32> @llvm.masked.load.v8i32(<8 x i32>*, i32, <8 x i1>, <8 x i32>)

define void @masked_load_v8i64(<8 x i64>* %a, <8 x i64>* %m_ptr, <8 x i64>* %res_ptr) nounwind {
; RV32-LABEL: masked_load_v8i64:
; RV32:       # %bb.0:
; RV32-NEXT:    vsetivli a3, 8, e64,m4,ta,mu
; RV32-NEXT:    vle64.v v28, (a1)
; RV32-NEXT:    vsetivli a1, 16, e32,m4,ta,mu
; RV32-NEXT:    vmv.v.i v8, 0
; RV32-NEXT:    vsetivli a1, 8, e64,m4,ta,mu
; RV32-NEXT:    vmseq.vv v0, v28, v8
; RV32-NEXT:    vle64.v v28, (a0), v0.t
; RV32-NEXT:    vse64.v v28, (a2)
; RV32-NEXT:    ret
;
; RV64-LABEL: masked_load_v8i64:
; RV64:       # %bb.0:
; RV64-NEXT:    vsetivli a3, 8, e64,m4,ta,mu
; RV64-NEXT:    vle64.v v28, (a1)
; RV64-NEXT:    vmseq.vi v0, v28, 0
; RV64-NEXT:    vle64.v v28, (a0), v0.t
; RV64-NEXT:    vse64.v v28, (a2)
; RV64-NEXT:    ret
  %m = load <8 x i64>, <8 x i64>* %m_ptr
  %mask = icmp eq <8 x i64> %m, zeroinitializer
  %load = call <8 x i64> @llvm.masked.load.v8i64(<8 x i64>* %a, i32 8, <8 x i1> %mask, <8 x i64> undef)
  store <8 x i64> %load, <8 x i64>* %res_ptr
  ret void
}
declare <8 x i64> @llvm.masked.load.v8i64(<8 x i64>*, i32, <8 x i1>, <8 x i64>)

define void @masked_load_v16i8(<16 x i8>* %a, <16 x i8>* %m_ptr, <16 x i8>* %res_ptr) nounwind {
; CHECK-LABEL: masked_load_v16i8:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetivli a3, 16, e8,m1,ta,mu
; CHECK-NEXT:    vle8.v v25, (a1)
; CHECK-NEXT:    vmseq.vi v0, v25, 0
; CHECK-NEXT:    vle8.v v25, (a0), v0.t
; CHECK-NEXT:    vse8.v v25, (a2)
; CHECK-NEXT:    ret
  %m = load <16 x i8>, <16 x i8>* %m_ptr
  %mask = icmp eq <16 x i8> %m, zeroinitializer
  %load = call <16 x i8> @llvm.masked.load.v16i8(<16 x i8>* %a, i32 8, <16 x i1> %mask, <16 x i8> undef)
  store <16 x i8> %load, <16 x i8>* %res_ptr
  ret void
}
declare <16 x i8> @llvm.masked.load.v16i8(<16 x i8>*, i32, <16 x i1>, <16 x i8>)

define void @masked_load_v16i16(<16 x i16>* %a, <16 x i16>* %m_ptr, <16 x i16>* %res_ptr) nounwind {
; CHECK-LABEL: masked_load_v16i16:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetivli a3, 16, e16,m2,ta,mu
; CHECK-NEXT:    vle16.v v26, (a1)
; CHECK-NEXT:    vmseq.vi v0, v26, 0
; CHECK-NEXT:    vle16.v v26, (a0), v0.t
; CHECK-NEXT:    vse16.v v26, (a2)
; CHECK-NEXT:    ret
  %m = load <16 x i16>, <16 x i16>* %m_ptr
  %mask = icmp eq <16 x i16> %m, zeroinitializer
  %load = call <16 x i16> @llvm.masked.load.v16i16(<16 x i16>* %a, i32 8, <16 x i1> %mask, <16 x i16> undef)
  store <16 x i16> %load, <16 x i16>* %res_ptr
  ret void
}
declare <16 x i16> @llvm.masked.load.v16i16(<16 x i16>*, i32, <16 x i1>, <16 x i16>)

define void @masked_load_v16i32(<16 x i32>* %a, <16 x i32>* %m_ptr, <16 x i32>* %res_ptr) nounwind {
; CHECK-LABEL: masked_load_v16i32:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetivli a3, 16, e32,m4,ta,mu
; CHECK-NEXT:    vle32.v v28, (a1)
; CHECK-NEXT:    vmseq.vi v0, v28, 0
; CHECK-NEXT:    vle32.v v28, (a0), v0.t
; CHECK-NEXT:    vse32.v v28, (a2)
; CHECK-NEXT:    ret
  %m = load <16 x i32>, <16 x i32>* %m_ptr
  %mask = icmp eq <16 x i32> %m, zeroinitializer
  %load = call <16 x i32> @llvm.masked.load.v16i32(<16 x i32>* %a, i32 8, <16 x i1> %mask, <16 x i32> undef)
  store <16 x i32> %load, <16 x i32>* %res_ptr
  ret void
}
declare <16 x i32> @llvm.masked.load.v16i32(<16 x i32>*, i32, <16 x i1>, <16 x i32>)

define void @masked_load_v16i64(<16 x i64>* %a, <16 x i64>* %m_ptr, <16 x i64>* %res_ptr) nounwind {
; RV32-LABEL: masked_load_v16i64:
; RV32:       # %bb.0:
; RV32-NEXT:    vsetivli a3, 16, e64,m8,ta,mu
; RV32-NEXT:    vle64.v v8, (a1)
; RV32-NEXT:    addi a1, zero, 32
; RV32-NEXT:    vsetvli a1, a1, e32,m8,ta,mu
; RV32-NEXT:    vmv.v.i v16, 0
; RV32-NEXT:    vsetivli a1, 16, e64,m8,ta,mu
; RV32-NEXT:    vmseq.vv v0, v8, v16
; RV32-NEXT:    vle64.v v8, (a0), v0.t
; RV32-NEXT:    vse64.v v8, (a2)
; RV32-NEXT:    ret
;
; RV64-LABEL: masked_load_v16i64:
; RV64:       # %bb.0:
; RV64-NEXT:    vsetivli a3, 16, e64,m8,ta,mu
; RV64-NEXT:    vle64.v v8, (a1)
; RV64-NEXT:    vmseq.vi v0, v8, 0
; RV64-NEXT:    vle64.v v8, (a0), v0.t
; RV64-NEXT:    vse64.v v8, (a2)
; RV64-NEXT:    ret
  %m = load <16 x i64>, <16 x i64>* %m_ptr
  %mask = icmp eq <16 x i64> %m, zeroinitializer
  %load = call <16 x i64> @llvm.masked.load.v16i64(<16 x i64>* %a, i32 8, <16 x i1> %mask, <16 x i64> undef)
  store <16 x i64> %load, <16 x i64>* %res_ptr
  ret void
}
declare <16 x i64> @llvm.masked.load.v16i64(<16 x i64>*, i32, <16 x i1>, <16 x i64>)

define void @masked_load_v32i8(<32 x i8>* %a, <32 x i8>* %m_ptr, <32 x i8>* %res_ptr) nounwind {
; CHECK-LABEL: masked_load_v32i8:
; CHECK:       # %bb.0:
; CHECK-NEXT:    addi a3, zero, 32
; CHECK-NEXT:    vsetvli a3, a3, e8,m2,ta,mu
; CHECK-NEXT:    vle8.v v26, (a1)
; CHECK-NEXT:    vmseq.vi v0, v26, 0
; CHECK-NEXT:    vle8.v v26, (a0), v0.t
; CHECK-NEXT:    vse8.v v26, (a2)
; CHECK-NEXT:    ret
  %m = load <32 x i8>, <32 x i8>* %m_ptr
  %mask = icmp eq <32 x i8> %m, zeroinitializer
  %load = call <32 x i8> @llvm.masked.load.v32i8(<32 x i8>* %a, i32 8, <32 x i1> %mask, <32 x i8> undef)
  store <32 x i8> %load, <32 x i8>* %res_ptr
  ret void
}
declare <32 x i8> @llvm.masked.load.v32i8(<32 x i8>*, i32, <32 x i1>, <32 x i8>)

define void @masked_load_v32i16(<32 x i16>* %a, <32 x i16>* %m_ptr, <32 x i16>* %res_ptr) nounwind {
; CHECK-LABEL: masked_load_v32i16:
; CHECK:       # %bb.0:
; CHECK-NEXT:    addi a3, zero, 32
; CHECK-NEXT:    vsetvli a3, a3, e16,m4,ta,mu
; CHECK-NEXT:    vle16.v v28, (a1)
; CHECK-NEXT:    vmseq.vi v0, v28, 0
; CHECK-NEXT:    vle16.v v28, (a0), v0.t
; CHECK-NEXT:    vse16.v v28, (a2)
; CHECK-NEXT:    ret
  %m = load <32 x i16>, <32 x i16>* %m_ptr
  %mask = icmp eq <32 x i16> %m, zeroinitializer
  %load = call <32 x i16> @llvm.masked.load.v32i16(<32 x i16>* %a, i32 8, <32 x i1> %mask, <32 x i16> undef)
  store <32 x i16> %load, <32 x i16>* %res_ptr
  ret void
}
declare <32 x i16> @llvm.masked.load.v32i16(<32 x i16>*, i32, <32 x i1>, <32 x i16>)

define void @masked_load_v32i32(<32 x i32>* %a, <32 x i32>* %m_ptr, <32 x i32>* %res_ptr) nounwind {
; CHECK-LABEL: masked_load_v32i32:
; CHECK:       # %bb.0:
; CHECK-NEXT:    addi a3, zero, 32
; CHECK-NEXT:    vsetvli a3, a3, e32,m8,ta,mu
; CHECK-NEXT:    vle32.v v8, (a1)
; CHECK-NEXT:    vmseq.vi v0, v8, 0
; CHECK-NEXT:    vle32.v v8, (a0), v0.t
; CHECK-NEXT:    vse32.v v8, (a2)
; CHECK-NEXT:    ret
  %m = load <32 x i32>, <32 x i32>* %m_ptr
  %mask = icmp eq <32 x i32> %m, zeroinitializer
  %load = call <32 x i32> @llvm.masked.load.v32i32(<32 x i32>* %a, i32 8, <32 x i1> %mask, <32 x i32> undef)
  store <32 x i32> %load, <32 x i32>* %res_ptr
  ret void
}
declare <32 x i32> @llvm.masked.load.v32i32(<32 x i32>*, i32, <32 x i1>, <32 x i32>)

define void @masked_load_v32i64(<32 x i64>* %a, <32 x i64>* %m_ptr, <32 x i64>* %res_ptr) nounwind {
; RV32-LABEL: masked_load_v32i64:
; RV32:       # %bb.0:
; RV32-NEXT:    addi sp, sp, -16
; RV32-NEXT:    csrr a3, vlenb
; RV32-NEXT:    slli a3, a3, 3
; RV32-NEXT:    sub sp, sp, a3
; RV32-NEXT:    addi a3, a1, 128
; RV32-NEXT:    vsetivli a4, 16, e64,m8,ta,mu
; RV32-NEXT:    vle64.v v8, (a3)
; RV32-NEXT:    addi a3, sp, 16
; RV32-NEXT:    vs8r.v v8, (a3) # Unknown-size Folded Spill
; RV32-NEXT:    vle64.v v16, (a1)
; RV32-NEXT:    addi a1, zero, 32
; RV32-NEXT:    vsetvli a1, a1, e32,m8,ta,mu
; RV32-NEXT:    vmv.v.i v8, 0
; RV32-NEXT:    vsetivli a1, 16, e64,m8,ta,mu
; RV32-NEXT:    vmseq.vv v25, v16, v8
; RV32-NEXT:    addi a1, sp, 16
; RV32-NEXT:    vl8re8.v v16, (a1) # Unknown-size Folded Reload
; RV32-NEXT:    vmseq.vv v0, v16, v8
; RV32-NEXT:    addi a1, a0, 128
; RV32-NEXT:    vle64.v v8, (a1), v0.t
; RV32-NEXT:    vmv1r.v v0, v25
; RV32-NEXT:    vle64.v v16, (a0), v0.t
; RV32-NEXT:    vse64.v v16, (a2)
; RV32-NEXT:    addi a0, a2, 128
; RV32-NEXT:    vse64.v v8, (a0)
; RV32-NEXT:    csrr a0, vlenb
; RV32-NEXT:    slli a0, a0, 3
; RV32-NEXT:    add sp, sp, a0
; RV32-NEXT:    addi sp, sp, 16
; RV32-NEXT:    ret
;
; RV64-LABEL: masked_load_v32i64:
; RV64:       # %bb.0:
; RV64-NEXT:    addi a3, a1, 128
; RV64-NEXT:    vsetivli a4, 16, e64,m8,ta,mu
; RV64-NEXT:    vle64.v v8, (a1)
; RV64-NEXT:    vle64.v v16, (a3)
; RV64-NEXT:    vmseq.vi v25, v8, 0
; RV64-NEXT:    vmseq.vi v0, v16, 0
; RV64-NEXT:    addi a1, a0, 128
; RV64-NEXT:    vle64.v v8, (a1), v0.t
; RV64-NEXT:    vmv1r.v v0, v25
; RV64-NEXT:    vle64.v v16, (a0), v0.t
; RV64-NEXT:    vse64.v v16, (a2)
; RV64-NEXT:    addi a0, a2, 128
; RV64-NEXT:    vse64.v v8, (a0)
; RV64-NEXT:    ret
  %m = load <32 x i64>, <32 x i64>* %m_ptr
  %mask = icmp eq <32 x i64> %m, zeroinitializer
  %load = call <32 x i64> @llvm.masked.load.v32i64(<32 x i64>* %a, i32 8, <32 x i1> %mask, <32 x i64> undef)
  store <32 x i64> %load, <32 x i64>* %res_ptr
  ret void
}
declare <32 x i64> @llvm.masked.load.v32i64(<32 x i64>*, i32, <32 x i1>, <32 x i64>)

define void @masked_load_v64i8(<64 x i8>* %a, <64 x i8>* %m_ptr, <64 x i8>* %res_ptr) nounwind {
; CHECK-LABEL: masked_load_v64i8:
; CHECK:       # %bb.0:
; CHECK-NEXT:    addi a3, zero, 64
; CHECK-NEXT:    vsetvli a3, a3, e8,m4,ta,mu
; CHECK-NEXT:    vle8.v v28, (a1)
; CHECK-NEXT:    vmseq.vi v0, v28, 0
; CHECK-NEXT:    vle8.v v28, (a0), v0.t
; CHECK-NEXT:    vse8.v v28, (a2)
; CHECK-NEXT:    ret
  %m = load <64 x i8>, <64 x i8>* %m_ptr
  %mask = icmp eq <64 x i8> %m, zeroinitializer
  %load = call <64 x i8> @llvm.masked.load.v64i8(<64 x i8>* %a, i32 8, <64 x i1> %mask, <64 x i8> undef)
  store <64 x i8> %load, <64 x i8>* %res_ptr
  ret void
}
declare <64 x i8> @llvm.masked.load.v64i8(<64 x i8>*, i32, <64 x i1>, <64 x i8>)

define void @masked_load_v64i16(<64 x i16>* %a, <64 x i16>* %m_ptr, <64 x i16>* %res_ptr) nounwind {
; CHECK-LABEL: masked_load_v64i16:
; CHECK:       # %bb.0:
; CHECK-NEXT:    addi a3, zero, 64
; CHECK-NEXT:    vsetvli a3, a3, e16,m8,ta,mu
; CHECK-NEXT:    vle16.v v8, (a1)
; CHECK-NEXT:    vmseq.vi v0, v8, 0
; CHECK-NEXT:    vle16.v v8, (a0), v0.t
; CHECK-NEXT:    vse16.v v8, (a2)
; CHECK-NEXT:    ret
  %m = load <64 x i16>, <64 x i16>* %m_ptr
  %mask = icmp eq <64 x i16> %m, zeroinitializer
  %load = call <64 x i16> @llvm.masked.load.v64i16(<64 x i16>* %a, i32 8, <64 x i1> %mask, <64 x i16> undef)
  store <64 x i16> %load, <64 x i16>* %res_ptr
  ret void
}
declare <64 x i16> @llvm.masked.load.v64i16(<64 x i16>*, i32, <64 x i1>, <64 x i16>)

define void @masked_load_v64i32(<64 x i32>* %a, <64 x i32>* %m_ptr, <64 x i32>* %res_ptr) nounwind {
; CHECK-LABEL: masked_load_v64i32:
; CHECK:       # %bb.0:
; CHECK-NEXT:    addi a3, a1, 128
; CHECK-NEXT:    addi a4, zero, 32
; CHECK-NEXT:    vsetvli a4, a4, e32,m8,ta,mu
; CHECK-NEXT:    vle32.v v8, (a1)
; CHECK-NEXT:    vle32.v v16, (a3)
; CHECK-NEXT:    vmseq.vi v25, v8, 0
; CHECK-NEXT:    vmseq.vi v0, v16, 0
; CHECK-NEXT:    addi a1, a0, 128
; CHECK-NEXT:    vle32.v v8, (a1), v0.t
; CHECK-NEXT:    vmv1r.v v0, v25
; CHECK-NEXT:    vle32.v v16, (a0), v0.t
; CHECK-NEXT:    vse32.v v16, (a2)
; CHECK-NEXT:    addi a0, a2, 128
; CHECK-NEXT:    vse32.v v8, (a0)
; CHECK-NEXT:    ret
  %m = load <64 x i32>, <64 x i32>* %m_ptr
  %mask = icmp eq <64 x i32> %m, zeroinitializer
  %load = call <64 x i32> @llvm.masked.load.v64i32(<64 x i32>* %a, i32 8, <64 x i1> %mask, <64 x i32> undef)
  store <64 x i32> %load, <64 x i32>* %res_ptr
  ret void
}
declare <64 x i32> @llvm.masked.load.v64i32(<64 x i32>*, i32, <64 x i1>, <64 x i32>)

define void @masked_load_v128i8(<128 x i8>* %a, <128 x i8>* %m_ptr, <128 x i8>* %res_ptr) nounwind {
; CHECK-LABEL: masked_load_v128i8:
; CHECK:       # %bb.0:
; CHECK-NEXT:    addi a3, zero, 128
; CHECK-NEXT:    vsetvli a3, a3, e8,m8,ta,mu
; CHECK-NEXT:    vle8.v v8, (a1)
; CHECK-NEXT:    vmseq.vi v0, v8, 0
; CHECK-NEXT:    vle8.v v8, (a0), v0.t
; CHECK-NEXT:    vse8.v v8, (a2)
; CHECK-NEXT:    ret
  %m = load <128 x i8>, <128 x i8>* %m_ptr
  %mask = icmp eq <128 x i8> %m, zeroinitializer
  %load = call <128 x i8> @llvm.masked.load.v128i8(<128 x i8>* %a, i32 8, <128 x i1> %mask, <128 x i8> undef)
  store <128 x i8> %load, <128 x i8>* %res_ptr
  ret void
}
declare <128 x i8> @llvm.masked.load.v128i8(<128 x i8>*, i32, <128 x i1>, <128 x i8>)

define void @masked_load_v256i8(<256 x i8>* %a, <256 x i8>* %m_ptr, <256 x i8>* %res_ptr) nounwind {
; CHECK-LABEL: masked_load_v256i8:
; CHECK:       # %bb.0:
; CHECK-NEXT:    addi a3, a1, 128
; CHECK-NEXT:    addi a4, zero, 128
; CHECK-NEXT:    vsetvli a4, a4, e8,m8,ta,mu
; CHECK-NEXT:    vle8.v v8, (a1)
; CHECK-NEXT:    vle8.v v16, (a3)
; CHECK-NEXT:    vmseq.vi v25, v8, 0
; CHECK-NEXT:    vmseq.vi v0, v16, 0
; CHECK-NEXT:    addi a1, a0, 128
; CHECK-NEXT:    vle8.v v8, (a1), v0.t
; CHECK-NEXT:    vmv1r.v v0, v25
; CHECK-NEXT:    vle8.v v16, (a0), v0.t
; CHECK-NEXT:    vse8.v v16, (a2)
; CHECK-NEXT:    addi a0, a2, 128
; CHECK-NEXT:    vse8.v v8, (a0)
; CHECK-NEXT:    ret
  %m = load <256 x i8>, <256 x i8>* %m_ptr
  %mask = icmp eq <256 x i8> %m, zeroinitializer
  %load = call <256 x i8> @llvm.masked.load.v256i8(<256 x i8>* %a, i32 8, <256 x i1> %mask, <256 x i8> undef)
  store <256 x i8> %load, <256 x i8>* %res_ptr
  ret void
}
declare <256 x i8> @llvm.masked.load.v256i8(<256 x i8>*, i32, <256 x i1>, <256 x i8>)
