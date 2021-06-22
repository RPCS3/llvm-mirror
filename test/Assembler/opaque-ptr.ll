; RUN: llvm-as < %s | llvm-dis | llvm-as | llvm-dis | FileCheck %s
; RUN: verify-uselistorder %s

; CHECK: define ptr @f(ptr %a) {
; CHECK:     %b = bitcast ptr %a to ptr
; CHECK:     ret ptr %b
define ptr @f(ptr %a) {
    %b = bitcast ptr %a to ptr
    ret ptr %b
}

; CHECK: define ptr @g(ptr addrspace(2) %a) {
; CHECK:     %b = addrspacecast ptr addrspace(2) %a to ptr
; CHECK:     ret ptr %b
define ptr @g(ptr addrspace(2) %a) {
    %b = addrspacecast ptr addrspace(2) %a to ptr addrspace(0)
    ret ptr addrspace(0) %b
}

; CHECK: define ptr addrspace(2) @g2(ptr %a) {
; CHECK:     %b = addrspacecast ptr %a to ptr addrspace(2)
; CHECK:     ret ptr addrspace(2) %b
define ptr addrspace(2) @g2(ptr addrspace(0) %a) {
    %b = addrspacecast ptr addrspace(0) %a to ptr addrspace(2)
    ret ptr addrspace(2) %b
}

; CHECK: define i32 @load(ptr %a)
; CHECK:     %i = load i32, ptr %a
; CHECK:     ret i32 %i
define i32 @load(ptr %a) {
    %i = load i32, ptr %a
    ret i32 %i
}

; CHECK: define void @store(ptr %a, i32 %i)
; CHECK:     store i32 %i, ptr %a
; CHECK:     ret void
define void @store(ptr %a, i32 %i) {
    store i32 %i, ptr %a
    ret void
}

; CHECK: define ptr @gep(ptr %a)
; CHECK:     %res = getelementptr i8, ptr %a, i32 2
; CHECK:     ret ptr %res
define ptr @gep(ptr %a) {
  %res = getelementptr i8, ptr %a, i32 2
  ret ptr %res
}

; CHECK: define <2 x ptr> @gep_vec1(ptr %a)
; CHECK:     %res = getelementptr i8, ptr %a, <2 x i32> <i32 1, i32 2>
; CHECK:     ret <2 x ptr> %res
define <2 x ptr> @gep_vec1(ptr %a) {
  %res = getelementptr i8, ptr %a, <2 x i32> <i32 1, i32 2>
  ret <2 x ptr> %res
}

; CHECK: define <2 x ptr> @gep_vec2(<2 x ptr> %a)
; CHECK:     %res = getelementptr i8, <2 x ptr> %a, i32 2
; CHECK:     ret <2 x ptr> %res
define <2 x ptr> @gep_vec2(<2 x ptr> %a) {
  %res = getelementptr i8, <2 x ptr> %a, i32 2
  ret <2 x ptr> %res
}

; CHECK: define ptr @gep_constexpr(ptr %a)
; CHECK:     ret ptr getelementptr (i16, ptr null, i32 3)
define ptr @gep_constexpr(ptr %a) {
  ret ptr getelementptr (i16, ptr null, i32 3)
}

; CHECK: define <2 x ptr> @gep_constexpr_vec1(ptr %a)
; CHECK:     ret <2 x ptr> getelementptr (i16, ptr null, <2 x i32> <i32 3, i32 4>)
define <2 x ptr> @gep_constexpr_vec1(ptr %a) {
  ret <2 x ptr> getelementptr (i16, ptr null, <2 x i32> <i32 3, i32 4>)
}

; CHECK: define <2 x ptr> @gep_constexpr_vec2(<2 x ptr> %a)
; CHECK:     ret <2 x ptr> getelementptr (i16, <2 x ptr> zeroinitializer, <2 x i32> <i32 3, i32 3>)
define <2 x ptr> @gep_constexpr_vec2(<2 x ptr> %a) {
  ret <2 x ptr> getelementptr (i16, <2 x ptr> zeroinitializer, i32 3)
}

; CHECK: define void @cmpxchg(ptr %p, i32 %a, i32 %b)
; CHECK:     %val_success = cmpxchg ptr %p, i32 %a, i32 %b acq_rel monotonic
; CHECK:     ret void
define void @cmpxchg(ptr %p, i32 %a, i32 %b) {
    %val_success = cmpxchg ptr %p, i32 %a, i32 %b acq_rel monotonic
    ret void
}

; CHECK: define void @atomicrmw(ptr %a, i32 %i)
; CHECK:     %b = atomicrmw add ptr %a, i32 %i acquire
; CHECK:     ret void
define void @atomicrmw(ptr %a, i32 %i) {
    %b = atomicrmw add ptr %a, i32 %i acquire
    ret void
}

; CHECK: define void @call(ptr %p)
; CHECK:     call void %p()
; CHECK:     ret void
define void @call(ptr %p) {
  call void %p()
  ret void
}

; CHECK: define void @call_arg(ptr %p, i32 %a)
; CHECK:     call void %p(i32 %a)
; CHECK:     ret void
define void @call_arg(ptr %p, i32 %a) {
  call void %p(i32 %a)
  ret void
}
