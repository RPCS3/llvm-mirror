; NOTE: Assertions have been autogenerated by utils/update_test_checks.py UTC_ARGS: --function-signature --check-attributes --check-globals
; RUN: opt -attributor -enable-new-pm=0 -attributor-manifest-internal  -attributor-max-iterations-verify -attributor-annotate-decl-cs -attributor-max-iterations=9 -S < %s | FileCheck %s --check-prefixes=CHECK,NOT_CGSCC_NPM,NOT_CGSCC_OPM,NOT_TUNIT_NPM,IS__TUNIT____,IS________OPM,IS__TUNIT_OPM
; RUN: opt -aa-pipeline=basic-aa -passes=attributor -attributor-manifest-internal  -attributor-max-iterations-verify -attributor-annotate-decl-cs -attributor-max-iterations=9 -S < %s | FileCheck %s --check-prefixes=CHECK,NOT_CGSCC_OPM,NOT_CGSCC_NPM,NOT_TUNIT_OPM,IS__TUNIT____,IS________NPM,IS__TUNIT_NPM
; RUN: opt -attributor-cgscc -enable-new-pm=0 -attributor-manifest-internal  -attributor-annotate-decl-cs -S < %s | FileCheck %s --check-prefixes=CHECK,NOT_TUNIT_NPM,NOT_TUNIT_OPM,NOT_CGSCC_NPM,IS__CGSCC____,IS________OPM,IS__CGSCC_OPM
; RUN: opt -aa-pipeline=basic-aa -passes=attributor-cgscc -attributor-manifest-internal  -attributor-annotate-decl-cs -S < %s | FileCheck %s --check-prefixes=CHECK,NOT_TUNIT_NPM,NOT_TUNIT_OPM,NOT_CGSCC_OPM,IS__CGSCC____,IS________NPM,IS__CGSCC_NPM

declare noalias i8* @malloc(i64)

declare void @nocapture_func_frees_pointer(i8* nocapture)

declare void @func_throws(...)

declare void @sync_func(i8* %p)

declare void @sync_will_return(i8* %p) willreturn nounwind

declare void @no_sync_func(i8* nocapture %p) nofree nosync willreturn

declare void @nofree_func(i8* nocapture %p) nofree  nosync willreturn

declare void @foo(i32* %p)

declare void @foo_nounw(i32* %p) nounwind nofree

declare void @usei8(i8)

declare i32 @no_return_call() noreturn

declare void @free(i8* nocapture)

declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture) nounwind

define void @h2s_value_simplify_interaction(i1 %c, i8* %A) {
; IS________OPM-LABEL: define {{[^@]+}}@h2s_value_simplify_interaction
; IS________OPM-SAME: (i1 [[C:%.*]], i8* nocapture nofree [[A:%.*]]) {
; IS________OPM-NEXT:  entry:
; IS________OPM-NEXT:    [[M:%.*]] = tail call noalias i8* @malloc(i64 noundef 4)
; IS________OPM-NEXT:    br i1 [[C]], label [[T:%.*]], label [[F:%.*]]
; IS________OPM:       t:
; IS________OPM-NEXT:    br i1 false, label [[DEAD:%.*]], label [[F2:%.*]]
; IS________OPM:       f:
; IS________OPM-NEXT:    br label [[J:%.*]]
; IS________OPM:       f2:
; IS________OPM-NEXT:    [[C1:%.*]] = bitcast i8* [[M]] to i32*
; IS________OPM-NEXT:    [[C2:%.*]] = bitcast i32* [[C1]] to i8*
; IS________OPM-NEXT:    [[L:%.*]] = load i8, i8* [[C2]], align 1
; IS________OPM-NEXT:    call void @usei8(i8 [[L]])
; IS________OPM-NEXT:    call void @no_sync_func(i8* nocapture nofree noundef [[C2]]) #[[ATTR5:[0-9]+]]
; IS________OPM-NEXT:    br label [[J]]
; IS________OPM:       dead:
; IS________OPM-NEXT:    unreachable
; IS________OPM:       j:
; IS________OPM-NEXT:    [[PHI:%.*]] = phi i8* [ [[M]], [[F]] ], [ null, [[F2]] ]
; IS________OPM-NEXT:    tail call void @no_sync_func(i8* nocapture nofree noundef [[PHI]]) #[[ATTR5]]
; IS________OPM-NEXT:    ret void
;
; IS________NPM-LABEL: define {{[^@]+}}@h2s_value_simplify_interaction
; IS________NPM-SAME: (i1 [[C:%.*]], i8* nocapture nofree [[A:%.*]]) {
; IS________NPM-NEXT:  entry:
; IS________NPM-NEXT:    [[TMP0:%.*]] = alloca i8, i64 4, align 1
; IS________NPM-NEXT:    br i1 [[C]], label [[T:%.*]], label [[F:%.*]]
; IS________NPM:       t:
; IS________NPM-NEXT:    br i1 false, label [[DEAD:%.*]], label [[F2:%.*]]
; IS________NPM:       f:
; IS________NPM-NEXT:    br label [[J:%.*]]
; IS________NPM:       f2:
; IS________NPM-NEXT:    [[L:%.*]] = load i8, i8* [[TMP0]], align 1
; IS________NPM-NEXT:    call void @usei8(i8 [[L]])
; IS________NPM-NEXT:    call void @no_sync_func(i8* nocapture nofree noundef [[TMP0]]) #[[ATTR6:[0-9]+]]
; IS________NPM-NEXT:    br label [[J]]
; IS________NPM:       dead:
; IS________NPM-NEXT:    unreachable
; IS________NPM:       j:
; IS________NPM-NEXT:    [[PHI:%.*]] = phi i8* [ [[TMP0]], [[F]] ], [ null, [[F2]] ]
; IS________NPM-NEXT:    tail call void @no_sync_func(i8* nocapture nofree noundef [[PHI]]) #[[ATTR6]]
; IS________NPM-NEXT:    ret void
;
entry:
  %add = add i64 2, 2
  %m = tail call noalias i8* @malloc(i64 %add)
  br i1 %c, label %t, label %f
t:
  br i1 false, label %dead, label %f2
f:
  br label %j
f2:
  %c1 = bitcast i8* %m to i32*
  %c2 = bitcast i32* %c1 to i8*
  %l = load i8, i8* %c2
  call void @usei8(i8 %l)
  call void @no_sync_func(i8* noundef %c2) nounwind
  br label %j
dead:
  br label %j
j:
  %phi = phi i8* [ %m, %f ], [ null, %f2 ], [ %A, %dead ]
  tail call void @no_sync_func(i8* noundef %phi) nounwind
  ;tail call void @free(i8* %m)
  ret void
}

define void @nofree_arg_only(i8* %p1, i8* %p2) {
; CHECK-LABEL: define {{[^@]+}}@nofree_arg_only
; CHECK-SAME: (i8* nocapture nofree [[P1:%.*]], i8* nocapture [[P2:%.*]]) {
; CHECK-NEXT:    tail call void @free(i8* nocapture [[P2]])
; CHECK-NEXT:    tail call void @nofree_func(i8* nocapture nofree [[P1]])
; CHECK-NEXT:    ret void
;
  tail call void @free(i8* %p2)
  tail call void @nofree_func(i8* %p1)
  ret void
}

; TEST 1 - negative, pointer freed in another function.

define void @test1() {
; CHECK-LABEL: define {{[^@]+}}@test1() {
; CHECK-NEXT:    [[TMP1:%.*]] = tail call noalias i8* @malloc(i64 noundef 4)
; CHECK-NEXT:    tail call void @nocapture_func_frees_pointer(i8* noalias nocapture [[TMP1]])
; CHECK-NEXT:    tail call void (...) @func_throws()
; CHECK-NEXT:    tail call void @free(i8* noalias nocapture [[TMP1]])
; CHECK-NEXT:    ret void
;
  %1 = tail call noalias i8* @malloc(i64 4)
  tail call void @nocapture_func_frees_pointer(i8* %1)
  tail call void (...) @func_throws()
  tail call void @free(i8* %1)
  ret void
}

; TEST 2 - negative, call to a sync function.

define void @test2() {
; CHECK-LABEL: define {{[^@]+}}@test2() {
; CHECK-NEXT:    [[TMP1:%.*]] = tail call noalias i8* @malloc(i64 noundef 4)
; CHECK-NEXT:    tail call void @sync_func(i8* [[TMP1]])
; CHECK-NEXT:    tail call void @free(i8* nocapture [[TMP1]])
; CHECK-NEXT:    ret void
;
  %1 = tail call noalias i8* @malloc(i64 4)
  tail call void @sync_func(i8* %1)
  tail call void @free(i8* %1)
  ret void
}

; TEST 3 - 1 malloc, 1 free

define void @test3() {
; IS________OPM-LABEL: define {{[^@]+}}@test3() {
; IS________OPM-NEXT:    [[TMP1:%.*]] = tail call noalias i8* @malloc(i64 noundef 4)
; IS________OPM-NEXT:    tail call void @no_sync_func(i8* noalias nocapture nofree [[TMP1]])
; IS________OPM-NEXT:    tail call void @free(i8* noalias nocapture [[TMP1]])
; IS________OPM-NEXT:    ret void
;
; IS________NPM-LABEL: define {{[^@]+}}@test3() {
; IS________NPM-NEXT:    [[TMP1:%.*]] = alloca i8, i64 4, align 1
; IS________NPM-NEXT:    tail call void @no_sync_func(i8* noalias nocapture nofree [[TMP1]])
; IS________NPM-NEXT:    ret void
;
  %1 = tail call noalias i8* @malloc(i64 4)
  tail call void @no_sync_func(i8* %1)
  tail call void @free(i8* %1)
  ret void
}

define void @test3a(i8* %p) {
; IS________OPM-LABEL: define {{[^@]+}}@test3a
; IS________OPM-SAME: (i8* nocapture [[P:%.*]]) {
; IS________OPM-NEXT:    [[TMP1:%.*]] = tail call noalias i8* @malloc(i64 noundef 4)
; IS________OPM-NEXT:    tail call void @nofree_arg_only(i8* nocapture nofree [[TMP1]], i8* nocapture [[P]])
; IS________OPM-NEXT:    tail call void @free(i8* noalias nocapture [[TMP1]])
; IS________OPM-NEXT:    ret void
;
; IS________NPM-LABEL: define {{[^@]+}}@test3a
; IS________NPM-SAME: (i8* nocapture [[P:%.*]]) {
; IS________NPM-NEXT:    [[TMP1:%.*]] = alloca i8, i64 4, align 1
; IS________NPM-NEXT:    tail call void @nofree_arg_only(i8* noalias nocapture nofree [[TMP1]], i8* nocapture [[P]])
; IS________NPM-NEXT:    ret void
;
  %1 = tail call noalias i8* @malloc(i64 4)
  tail call void @nofree_arg_only(i8* %1, i8* %p)
  tail call void @free(i8* %1)
  ret void
}

declare noalias i8* @aligned_alloc(i64, i64)

define void @test3b(i8* %p) {
; IS________OPM-LABEL: define {{[^@]+}}@test3b
; IS________OPM-SAME: (i8* nocapture [[P:%.*]]) {
; IS________OPM-NEXT:    [[TMP1:%.*]] = tail call noalias i8* @aligned_alloc(i64 noundef 32, i64 noundef 128)
; IS________OPM-NEXT:    tail call void @nofree_arg_only(i8* nocapture nofree [[TMP1]], i8* nocapture [[P]])
; IS________OPM-NEXT:    tail call void @free(i8* noalias nocapture [[TMP1]])
; IS________OPM-NEXT:    ret void
;
; IS________NPM-LABEL: define {{[^@]+}}@test3b
; IS________NPM-SAME: (i8* nocapture [[P:%.*]]) {
; IS________NPM-NEXT:    [[TMP1:%.*]] = alloca i8, i64 128, align 32
; IS________NPM-NEXT:    tail call void @nofree_arg_only(i8* noalias nocapture nofree [[TMP1]], i8* nocapture [[P]])
; IS________NPM-NEXT:    ret void
;
  %1 = tail call noalias i8* @aligned_alloc(i64 32, i64 128)
  tail call void @nofree_arg_only(i8* %1, i8* %p)
  tail call void @free(i8* %1)
  ret void
}

; leave alone non-constant alignments.
define void @test3c(i64 %alignment) {
; CHECK-LABEL: define {{[^@]+}}@test3c
; CHECK-SAME: (i64 [[ALIGNMENT:%.*]]) {
; CHECK-NEXT:    [[TMP1:%.*]] = tail call noalias i8* @aligned_alloc(i64 [[ALIGNMENT]], i64 noundef 128)
; CHECK-NEXT:    tail call void @free(i8* noalias nocapture [[TMP1]])
; CHECK-NEXT:    ret void
;
  %1 = tail call noalias i8* @aligned_alloc(i64 %alignment, i64 128)
  tail call void @free(i8* %1)
  ret void
}

declare noalias i8* @calloc(i64, i64)

define void @test0() {
; IS________OPM-LABEL: define {{[^@]+}}@test0() {
; IS________OPM-NEXT:    [[TMP1:%.*]] = tail call noalias i8* @calloc(i64 noundef 2, i64 noundef 4)
; IS________OPM-NEXT:    tail call void @no_sync_func(i8* noalias nocapture nofree [[TMP1]])
; IS________OPM-NEXT:    tail call void @free(i8* noalias nocapture [[TMP1]])
; IS________OPM-NEXT:    ret void
;
; IS________NPM-LABEL: define {{[^@]+}}@test0() {
; IS________NPM-NEXT:    [[TMP1:%.*]] = alloca i8, i64 8, align 1
; IS________NPM-NEXT:    [[CALLOC_BC:%.*]] = bitcast i8* [[TMP1]] to i8*
; IS________NPM-NEXT:    call void @llvm.memset.p0i8.i64(i8* [[CALLOC_BC]], i8 0, i64 8, i1 false)
; IS________NPM-NEXT:    tail call void @no_sync_func(i8* noalias nocapture nofree [[TMP1]])
; IS________NPM-NEXT:    ret void
;
  %1 = tail call noalias i8* @calloc(i64 2, i64 4)
  tail call void @no_sync_func(i8* %1)
  tail call void @free(i8* %1)
  ret void
}

; TEST 4
define void @test4() {
; IS________OPM-LABEL: define {{[^@]+}}@test4() {
; IS________OPM-NEXT:    [[TMP1:%.*]] = tail call noalias i8* @malloc(i64 noundef 4)
; IS________OPM-NEXT:    tail call void @nofree_func(i8* noalias nocapture nofree [[TMP1]])
; IS________OPM-NEXT:    ret void
;
; IS________NPM-LABEL: define {{[^@]+}}@test4() {
; IS________NPM-NEXT:    [[TMP1:%.*]] = alloca i8, i64 4, align 1
; IS________NPM-NEXT:    tail call void @nofree_func(i8* noalias nocapture nofree [[TMP1]])
; IS________NPM-NEXT:    ret void
;
  %1 = tail call noalias i8* @malloc(i64 4)
  tail call void @nofree_func(i8* %1)
  ret void
}

; TEST 5 - not all exit paths have a call to free, but all uses of malloc
; are in nofree functions and are not captured

define void @test5(i32, i8* %p) {
; IS________OPM-LABEL: define {{[^@]+}}@test5
; IS________OPM-SAME: (i32 [[TMP0:%.*]], i8* nocapture [[P:%.*]]) {
; IS________OPM-NEXT:    [[TMP2:%.*]] = tail call noalias i8* @malloc(i64 noundef 4)
; IS________OPM-NEXT:    [[TMP3:%.*]] = icmp eq i32 [[TMP0]], 0
; IS________OPM-NEXT:    br i1 [[TMP3]], label [[TMP5:%.*]], label [[TMP4:%.*]]
; IS________OPM:       4:
; IS________OPM-NEXT:    tail call void @nofree_func(i8* noalias nocapture nofree [[TMP2]])
; IS________OPM-NEXT:    br label [[TMP6:%.*]]
; IS________OPM:       5:
; IS________OPM-NEXT:    tail call void @nofree_arg_only(i8* nocapture nofree [[TMP2]], i8* nocapture [[P]])
; IS________OPM-NEXT:    tail call void @free(i8* noalias nocapture [[TMP2]])
; IS________OPM-NEXT:    br label [[TMP6]]
; IS________OPM:       6:
; IS________OPM-NEXT:    ret void
;
; IS________NPM-LABEL: define {{[^@]+}}@test5
; IS________NPM-SAME: (i32 [[TMP0:%.*]], i8* nocapture [[P:%.*]]) {
; IS________NPM-NEXT:    [[TMP2:%.*]] = alloca i8, i64 4, align 1
; IS________NPM-NEXT:    [[TMP3:%.*]] = icmp eq i32 [[TMP0]], 0
; IS________NPM-NEXT:    br i1 [[TMP3]], label [[TMP5:%.*]], label [[TMP4:%.*]]
; IS________NPM:       4:
; IS________NPM-NEXT:    tail call void @nofree_func(i8* noalias nocapture nofree [[TMP2]])
; IS________NPM-NEXT:    br label [[TMP6:%.*]]
; IS________NPM:       5:
; IS________NPM-NEXT:    tail call void @nofree_arg_only(i8* noalias nocapture nofree [[TMP2]], i8* nocapture [[P]])
; IS________NPM-NEXT:    br label [[TMP6]]
; IS________NPM:       6:
; IS________NPM-NEXT:    ret void
;
  %2 = tail call noalias i8* @malloc(i64 4)
  %3 = icmp eq i32 %0, 0
  br i1 %3, label %5, label %4

4:                                                ; preds = %1
  tail call void @nofree_func(i8* %2)
  br label %6

5:                                                ; preds = %1
  tail call void @nofree_arg_only(i8* %2, i8* %p)
  tail call void @free(i8* %2)
  br label %6

6:                                                ; preds = %5, %4
  ret void
}

; TEST 6 - all exit paths have a call to free

define void @test6(i32) {
; IS________OPM-LABEL: define {{[^@]+}}@test6
; IS________OPM-SAME: (i32 [[TMP0:%.*]]) {
; IS________OPM-NEXT:    [[TMP2:%.*]] = tail call noalias i8* @malloc(i64 noundef 4)
; IS________OPM-NEXT:    [[TMP3:%.*]] = icmp eq i32 [[TMP0]], 0
; IS________OPM-NEXT:    br i1 [[TMP3]], label [[TMP5:%.*]], label [[TMP4:%.*]]
; IS________OPM:       4:
; IS________OPM-NEXT:    tail call void @nofree_func(i8* noalias nocapture nofree [[TMP2]])
; IS________OPM-NEXT:    tail call void @free(i8* noalias nocapture [[TMP2]])
; IS________OPM-NEXT:    br label [[TMP6:%.*]]
; IS________OPM:       5:
; IS________OPM-NEXT:    tail call void @free(i8* noalias nocapture [[TMP2]])
; IS________OPM-NEXT:    br label [[TMP6]]
; IS________OPM:       6:
; IS________OPM-NEXT:    ret void
;
; IS________NPM-LABEL: define {{[^@]+}}@test6
; IS________NPM-SAME: (i32 [[TMP0:%.*]]) {
; IS________NPM-NEXT:    [[TMP2:%.*]] = alloca i8, i64 4, align 1
; IS________NPM-NEXT:    [[TMP3:%.*]] = icmp eq i32 [[TMP0]], 0
; IS________NPM-NEXT:    br i1 [[TMP3]], label [[TMP5:%.*]], label [[TMP4:%.*]]
; IS________NPM:       4:
; IS________NPM-NEXT:    tail call void @nofree_func(i8* noalias nocapture nofree [[TMP2]])
; IS________NPM-NEXT:    br label [[TMP6:%.*]]
; IS________NPM:       5:
; IS________NPM-NEXT:    br label [[TMP6]]
; IS________NPM:       6:
; IS________NPM-NEXT:    ret void
;
  %2 = tail call noalias i8* @malloc(i64 4)
  %3 = icmp eq i32 %0, 0
  br i1 %3, label %5, label %4

4:                                                ; preds = %1
  tail call void @nofree_func(i8* %2)
  tail call void @free(i8* %2)
  br label %6

5:                                                ; preds = %1
  tail call void @free(i8* %2)
  br label %6

6:                                                ; preds = %5, %4
  ret void
}

; TEST 7 - free is dead.

define void @test7() {
; IS________OPM: Function Attrs: noreturn
; IS________OPM-LABEL: define {{[^@]+}}@test7
; IS________OPM-SAME: () #[[ATTR3:[0-9]+]] {
; IS________OPM-NEXT:    [[TMP1:%.*]] = tail call noalias i8* @malloc(i64 noundef 4)
; IS________OPM-NEXT:    [[TMP2:%.*]] = tail call i32 @no_return_call() #[[ATTR3]]
; IS________OPM-NEXT:    unreachable
;
; IS________NPM: Function Attrs: noreturn
; IS________NPM-LABEL: define {{[^@]+}}@test7
; IS________NPM-SAME: () #[[ATTR3:[0-9]+]] {
; IS________NPM-NEXT:    [[TMP1:%.*]] = alloca i8, i64 4, align 1
; IS________NPM-NEXT:    [[TMP2:%.*]] = tail call i32 @no_return_call() #[[ATTR3]]
; IS________NPM-NEXT:    unreachable
;
  %1 = tail call noalias i8* @malloc(i64 4)
  tail call i32 @no_return_call()
  tail call void @free(i8* %1)
  ret void
}

; TEST 8 - Negative: bitcast pointer used in capture function

define void @test8() {
; CHECK-LABEL: define {{[^@]+}}@test8() {
; CHECK-NEXT:    [[TMP1:%.*]] = tail call noalias i8* @malloc(i64 noundef 4)
; CHECK-NEXT:    tail call void @no_sync_func(i8* noalias nocapture nofree [[TMP1]])
; CHECK-NEXT:    [[TMP2:%.*]] = bitcast i8* [[TMP1]] to i32*
; CHECK-NEXT:    store i32 10, i32* [[TMP2]], align 4
; CHECK-NEXT:    tail call void @foo(i32* noundef align 4 [[TMP2]])
; CHECK-NEXT:    tail call void @free(i8* nocapture noundef nonnull align 4 dereferenceable(4) [[TMP1]])
; CHECK-NEXT:    ret void
;
  %1 = tail call noalias i8* @malloc(i64 4)
  tail call void @no_sync_func(i8* %1)
  %2 = bitcast i8* %1 to i32*
  store i32 10, i32* %2
  %3 = load i32, i32* %2
  tail call void @foo(i32* %2)
  tail call void @free(i8* %1)
  ret void
}

; TEST 9 - FIXME: malloc should be converted.
define void @test9() {
; IS________OPM-LABEL: define {{[^@]+}}@test9() {
; IS________OPM-NEXT:    [[TMP1:%.*]] = tail call noalias i8* @malloc(i64 noundef 4)
; IS________OPM-NEXT:    tail call void @no_sync_func(i8* noalias nocapture nofree [[TMP1]])
; IS________OPM-NEXT:    [[TMP2:%.*]] = bitcast i8* [[TMP1]] to i32*
; IS________OPM-NEXT:    store i32 10, i32* [[TMP2]], align 4
; IS________OPM-NEXT:    tail call void @foo_nounw(i32* nofree noundef align 4 [[TMP2]]) #[[ATTR5]]
; IS________OPM-NEXT:    tail call void @free(i8* nocapture noundef nonnull align 4 dereferenceable(4) [[TMP1]])
; IS________OPM-NEXT:    ret void
;
; IS________NPM-LABEL: define {{[^@]+}}@test9() {
; IS________NPM-NEXT:    [[TMP1:%.*]] = tail call noalias i8* @malloc(i64 noundef 4)
; IS________NPM-NEXT:    tail call void @no_sync_func(i8* noalias nocapture nofree [[TMP1]])
; IS________NPM-NEXT:    [[TMP2:%.*]] = bitcast i8* [[TMP1]] to i32*
; IS________NPM-NEXT:    store i32 10, i32* [[TMP2]], align 4
; IS________NPM-NEXT:    tail call void @foo_nounw(i32* nofree noundef align 4 [[TMP2]]) #[[ATTR6]]
; IS________NPM-NEXT:    tail call void @free(i8* nocapture noundef nonnull align 4 dereferenceable(4) [[TMP1]])
; IS________NPM-NEXT:    ret void
;
  %1 = tail call noalias i8* @malloc(i64 4)
  tail call void @no_sync_func(i8* %1)
  %2 = bitcast i8* %1 to i32*
  store i32 10, i32* %2
  %3 = load i32, i32* %2
  tail call void @foo_nounw(i32* %2)
  tail call void @free(i8* %1)
  ret void
}

; TEST 10 - 1 malloc, 1 free

define i32 @test10() {
; IS________OPM-LABEL: define {{[^@]+}}@test10() {
; IS________OPM-NEXT:    [[TMP1:%.*]] = tail call noalias i8* @malloc(i64 noundef 4)
; IS________OPM-NEXT:    tail call void @no_sync_func(i8* noalias nocapture nofree [[TMP1]])
; IS________OPM-NEXT:    [[TMP2:%.*]] = bitcast i8* [[TMP1]] to i32*
; IS________OPM-NEXT:    store i32 10, i32* [[TMP2]], align 4
; IS________OPM-NEXT:    [[TMP3:%.*]] = load i32, i32* [[TMP2]], align 4
; IS________OPM-NEXT:    tail call void @free(i8* noalias nocapture noundef nonnull align 4 dereferenceable(4) [[TMP1]])
; IS________OPM-NEXT:    ret i32 [[TMP3]]
;
; IS________NPM-LABEL: define {{[^@]+}}@test10() {
; IS________NPM-NEXT:    [[TMP1:%.*]] = alloca i8, i64 4, align 1
; IS________NPM-NEXT:    tail call void @no_sync_func(i8* noalias nocapture nofree [[TMP1]])
; IS________NPM-NEXT:    [[TMP2:%.*]] = bitcast i8* [[TMP1]] to i32*
; IS________NPM-NEXT:    store i32 10, i32* [[TMP2]], align 4
; IS________NPM-NEXT:    [[TMP3:%.*]] = load i32, i32* [[TMP2]], align 4
; IS________NPM-NEXT:    ret i32 [[TMP3]]
;
  %1 = tail call noalias i8* @malloc(i64 4)
  tail call void @no_sync_func(i8* %1)
  %2 = bitcast i8* %1 to i32*
  store i32 10, i32* %2
  %3 = load i32, i32* %2
  tail call void @free(i8* %1)
  ret i32 %3
}

define i32 @test_lifetime() {
; IS________OPM-LABEL: define {{[^@]+}}@test_lifetime() {
; IS________OPM-NEXT:    [[TMP1:%.*]] = tail call noalias i8* @malloc(i64 noundef 4)
; IS________OPM-NEXT:    tail call void @no_sync_func(i8* noalias nocapture nofree [[TMP1]])
; IS________OPM-NEXT:    call void @llvm.lifetime.start.p0i8(i64 noundef 4, i8* noalias nocapture nofree noundef nonnull align 4 dereferenceable(4) [[TMP1]])
; IS________OPM-NEXT:    [[TMP2:%.*]] = bitcast i8* [[TMP1]] to i32*
; IS________OPM-NEXT:    store i32 10, i32* [[TMP2]], align 4
; IS________OPM-NEXT:    [[TMP3:%.*]] = load i32, i32* [[TMP2]], align 4
; IS________OPM-NEXT:    tail call void @free(i8* noalias nocapture noundef nonnull align 4 dereferenceable(4) [[TMP1]])
; IS________OPM-NEXT:    ret i32 [[TMP3]]
;
; IS________NPM-LABEL: define {{[^@]+}}@test_lifetime() {
; IS________NPM-NEXT:    [[TMP1:%.*]] = alloca i8, i64 4, align 1
; IS________NPM-NEXT:    tail call void @no_sync_func(i8* noalias nocapture nofree [[TMP1]])
; IS________NPM-NEXT:    call void @llvm.lifetime.start.p0i8(i64 noundef 4, i8* noalias nocapture nofree noundef nonnull align 4 dereferenceable(4) [[TMP1]])
; IS________NPM-NEXT:    [[TMP2:%.*]] = bitcast i8* [[TMP1]] to i32*
; IS________NPM-NEXT:    store i32 10, i32* [[TMP2]], align 4
; IS________NPM-NEXT:    [[TMP3:%.*]] = load i32, i32* [[TMP2]], align 4
; IS________NPM-NEXT:    ret i32 [[TMP3]]
;
  %1 = tail call noalias i8* @malloc(i64 4)
  tail call void @no_sync_func(i8* %1)
  call void @llvm.lifetime.start.p0i8(i64 4, i8* %1)
  %2 = bitcast i8* %1 to i32*
  store i32 10, i32* %2
  %3 = load i32, i32* %2
  tail call void @free(i8* %1)
  ret i32 %3
}

; TEST 11

define void @test11() {
; IS________OPM-LABEL: define {{[^@]+}}@test11() {
; IS________OPM-NEXT:    [[TMP1:%.*]] = tail call noalias i8* @malloc(i64 noundef 4)
; IS________OPM-NEXT:    tail call void @sync_will_return(i8* [[TMP1]]) #[[ATTR5]]
; IS________OPM-NEXT:    tail call void @free(i8* nocapture [[TMP1]])
; IS________OPM-NEXT:    ret void
;
; IS________NPM-LABEL: define {{[^@]+}}@test11() {
; IS________NPM-NEXT:    [[TMP1:%.*]] = alloca i8, i64 4, align 1
; IS________NPM-NEXT:    tail call void @sync_will_return(i8* [[TMP1]]) #[[ATTR6]]
; IS________NPM-NEXT:    ret void
;
  %1 = tail call noalias i8* @malloc(i64 4)
  tail call void @sync_will_return(i8* %1)
  tail call void @free(i8* %1)
  ret void
}

; TEST 12
define i32 @irreducible_cfg(i32 %0) {
; IS________OPM-LABEL: define {{[^@]+}}@irreducible_cfg
; IS________OPM-SAME: (i32 [[TMP0:%.*]]) {
; IS________OPM-NEXT:    [[TMP2:%.*]] = call noalias i8* @malloc(i64 noundef 4)
; IS________OPM-NEXT:    [[TMP3:%.*]] = bitcast i8* [[TMP2]] to i32*
; IS________OPM-NEXT:    store i32 10, i32* [[TMP3]], align 4
; IS________OPM-NEXT:    [[TMP4:%.*]] = icmp eq i32 [[TMP0]], 1
; IS________OPM-NEXT:    br i1 [[TMP4]], label [[TMP5:%.*]], label [[TMP7:%.*]]
; IS________OPM:       5:
; IS________OPM-NEXT:    [[TMP6:%.*]] = add nsw i32 [[TMP0]], 5
; IS________OPM-NEXT:    br label [[TMP13:%.*]]
; IS________OPM:       7:
; IS________OPM-NEXT:    br label [[TMP8:%.*]]
; IS________OPM:       8:
; IS________OPM-NEXT:    [[DOT0:%.*]] = phi i32 [ [[TMP14:%.*]], [[TMP13]] ], [ 1, [[TMP7]] ]
; IS________OPM-NEXT:    [[TMP9:%.*]] = load i32, i32* [[TMP3]], align 4
; IS________OPM-NEXT:    [[TMP10:%.*]] = add nsw i32 [[TMP9]], -1
; IS________OPM-NEXT:    store i32 [[TMP10]], i32* [[TMP3]], align 4
; IS________OPM-NEXT:    [[TMP11:%.*]] = icmp ne i32 [[TMP9]], 0
; IS________OPM-NEXT:    br i1 [[TMP11]], label [[TMP12:%.*]], label [[TMP15:%.*]]
; IS________OPM:       12:
; IS________OPM-NEXT:    br label [[TMP13]]
; IS________OPM:       13:
; IS________OPM-NEXT:    [[DOT1:%.*]] = phi i32 [ [[TMP6]], [[TMP5]] ], [ [[DOT0]], [[TMP12]] ]
; IS________OPM-NEXT:    [[TMP14]] = add nsw i32 [[DOT1]], 1
; IS________OPM-NEXT:    br label [[TMP8]]
; IS________OPM:       15:
; IS________OPM-NEXT:    [[TMP16:%.*]] = load i32, i32* [[TMP3]], align 4
; IS________OPM-NEXT:    [[TMP17:%.*]] = bitcast i32* [[TMP3]] to i8*
; IS________OPM-NEXT:    call void @free(i8* nocapture noundef [[TMP17]])
; IS________OPM-NEXT:    [[TMP18:%.*]] = load i32, i32* [[TMP3]], align 4
; IS________OPM-NEXT:    ret i32 [[TMP18]]
;
; IS________NPM-LABEL: define {{[^@]+}}@irreducible_cfg
; IS________NPM-SAME: (i32 [[TMP0:%.*]]) {
; IS________NPM-NEXT:    [[TMP2:%.*]] = alloca i8, i64 4, align 1
; IS________NPM-NEXT:    [[TMP3:%.*]] = bitcast i8* [[TMP2]] to i32*
; IS________NPM-NEXT:    store i32 10, i32* [[TMP3]], align 4
; IS________NPM-NEXT:    [[TMP4:%.*]] = icmp eq i32 [[TMP0]], 1
; IS________NPM-NEXT:    br i1 [[TMP4]], label [[TMP5:%.*]], label [[TMP7:%.*]]
; IS________NPM:       5:
; IS________NPM-NEXT:    [[TMP6:%.*]] = add nsw i32 [[TMP0]], 5
; IS________NPM-NEXT:    br label [[TMP13:%.*]]
; IS________NPM:       7:
; IS________NPM-NEXT:    br label [[TMP8:%.*]]
; IS________NPM:       8:
; IS________NPM-NEXT:    [[DOT0:%.*]] = phi i32 [ [[TMP14:%.*]], [[TMP13]] ], [ 1, [[TMP7]] ]
; IS________NPM-NEXT:    [[TMP9:%.*]] = load i32, i32* [[TMP3]], align 4
; IS________NPM-NEXT:    [[TMP10:%.*]] = add nsw i32 [[TMP9]], -1
; IS________NPM-NEXT:    store i32 [[TMP10]], i32* [[TMP3]], align 4
; IS________NPM-NEXT:    [[TMP11:%.*]] = icmp ne i32 [[TMP9]], 0
; IS________NPM-NEXT:    br i1 [[TMP11]], label [[TMP12:%.*]], label [[TMP15:%.*]]
; IS________NPM:       12:
; IS________NPM-NEXT:    br label [[TMP13]]
; IS________NPM:       13:
; IS________NPM-NEXT:    [[DOT1:%.*]] = phi i32 [ [[TMP6]], [[TMP5]] ], [ [[DOT0]], [[TMP12]] ]
; IS________NPM-NEXT:    [[TMP14]] = add nsw i32 [[DOT1]], 1
; IS________NPM-NEXT:    br label [[TMP8]]
; IS________NPM:       15:
; IS________NPM-NEXT:    [[TMP16:%.*]] = load i32, i32* [[TMP3]], align 4
; IS________NPM-NEXT:    ret i32 [[TMP16]]
;
  %2 = call noalias i8* @malloc(i64 4)
  %3 = bitcast i8* %2 to i32*
  store i32 10, i32* %3, align 4
  %4 = icmp eq i32 %0, 1
  br i1 %4, label %5, label %7

5:                                                ; preds = %1
  %6 = add nsw i32 %0, 5
  br label %13

7:                                                ; preds = %1
  br label %8

8:                                                ; preds = %13, %7
  %.0 = phi i32 [ %14, %13 ], [ 1, %7 ]
  %9 = load i32, i32* %3, align 4
  %10 = add nsw i32 %9, -1
  store i32 %10, i32* %3, align 4
  %11 = icmp ne i32 %9, 0
  br i1 %11, label %12, label %15

12:                                               ; preds = %8
  br label %13

13:                                               ; preds = %12, %5
  %.1 = phi i32 [ %6, %5 ], [ %.0, %12 ]
  %14 = add nsw i32 %.1, 1
  br label %8

15:                                               ; preds = %8
  %16 = load i32, i32* %3, align 4
  %17 = bitcast i32* %3 to i8*
  call void @free(i8* %17)
  %18 = load i32, i32* %3, align 4
  ret i32 %18
}


define i32 @malloc_in_loop(i32 %0) {
; IS________OPM-LABEL: define {{[^@]+}}@malloc_in_loop
; IS________OPM-SAME: (i32 [[TMP0:%.*]]) {
; IS________OPM-NEXT:    [[TMP2:%.*]] = alloca i32, align 4
; IS________OPM-NEXT:    [[TMP3:%.*]] = alloca i32*, align 8
; IS________OPM-NEXT:    store i32 [[TMP0]], i32* [[TMP2]], align 4
; IS________OPM-NEXT:    br label [[TMP4:%.*]]
; IS________OPM:       4:
; IS________OPM-NEXT:    [[TMP5:%.*]] = load i32, i32* [[TMP2]], align 4
; IS________OPM-NEXT:    [[TMP6:%.*]] = add nsw i32 [[TMP5]], -1
; IS________OPM-NEXT:    store i32 [[TMP6]], i32* [[TMP2]], align 4
; IS________OPM-NEXT:    [[TMP7:%.*]] = icmp sgt i32 [[TMP6]], 0
; IS________OPM-NEXT:    br i1 [[TMP7]], label [[TMP8:%.*]], label [[TMP11:%.*]]
; IS________OPM:       8:
; IS________OPM-NEXT:    [[TMP9:%.*]] = call noalias i8* @malloc(i64 noundef 4)
; IS________OPM-NEXT:    [[TMP10:%.*]] = bitcast i8* [[TMP9]] to i32*
; IS________OPM-NEXT:    store i32 1, i32* [[TMP10]], align 8
; IS________OPM-NEXT:    br label [[TMP4]]
; IS________OPM:       11:
; IS________OPM-NEXT:    ret i32 5
;
; IS________NPM-LABEL: define {{[^@]+}}@malloc_in_loop
; IS________NPM-SAME: (i32 [[TMP0:%.*]]) {
; IS________NPM-NEXT:    [[TMP2:%.*]] = alloca i32, align 4
; IS________NPM-NEXT:    [[TMP3:%.*]] = alloca i32*, align 8
; IS________NPM-NEXT:    store i32 [[TMP0]], i32* [[TMP2]], align 4
; IS________NPM-NEXT:    br label [[TMP4:%.*]]
; IS________NPM:       4:
; IS________NPM-NEXT:    [[TMP5:%.*]] = load i32, i32* [[TMP2]], align 4
; IS________NPM-NEXT:    [[TMP6:%.*]] = add nsw i32 [[TMP5]], -1
; IS________NPM-NEXT:    store i32 [[TMP6]], i32* [[TMP2]], align 4
; IS________NPM-NEXT:    [[TMP7:%.*]] = icmp sgt i32 [[TMP6]], 0
; IS________NPM-NEXT:    br i1 [[TMP7]], label [[TMP8:%.*]], label [[TMP11:%.*]]
; IS________NPM:       8:
; IS________NPM-NEXT:    [[TMP9:%.*]] = alloca i8, i64 4, align 1
; IS________NPM-NEXT:    [[TMP10:%.*]] = bitcast i8* [[TMP9]] to i32*
; IS________NPM-NEXT:    store i32 1, i32* [[TMP10]], align 8
; IS________NPM-NEXT:    br label [[TMP4]]
; IS________NPM:       11:
; IS________NPM-NEXT:    ret i32 5
;
  %2 = alloca i32, align 4
  %3 = alloca i32*, align 8
  store i32 %0, i32* %2, align 4
  br label %4

4:                                                ; preds = %8, %1
  %5 = load i32, i32* %2, align 4
  %6 = add nsw i32 %5, -1
  store i32 %6, i32* %2, align 4
  %7 = icmp sgt i32 %6, 0
  br i1 %7, label %8, label %11

8:                                                ; preds = %4
  %9 = call noalias i8* @malloc(i64 4)
  %10 = bitcast i8* %9 to i32*
  store i32 1, i32* %10, align 8
  br label %4

11:                                               ; preds = %4
  ret i32 5
}

; Malloc/Calloc too large
define i32 @test13() {
; CHECK-LABEL: define {{[^@]+}}@test13() {
; CHECK-NEXT:    [[TMP1:%.*]] = tail call noalias i8* @malloc(i64 noundef 256)
; CHECK-NEXT:    tail call void @no_sync_func(i8* noalias nocapture nofree [[TMP1]])
; CHECK-NEXT:    [[TMP2:%.*]] = bitcast i8* [[TMP1]] to i32*
; CHECK-NEXT:    store i32 10, i32* [[TMP2]], align 4
; CHECK-NEXT:    [[TMP3:%.*]] = load i32, i32* [[TMP2]], align 4
; CHECK-NEXT:    tail call void @free(i8* noalias nocapture noundef nonnull align 4 dereferenceable(4) [[TMP1]])
; CHECK-NEXT:    ret i32 [[TMP3]]
;
  %1 = tail call noalias i8* @malloc(i64 256)
  tail call void @no_sync_func(i8* %1)
  %2 = bitcast i8* %1 to i32*
  store i32 10, i32* %2
  %3 = load i32, i32* %2
  tail call void @free(i8* %1)
  ret i32 %3
}

define i32 @test_sle() {
; CHECK-LABEL: define {{[^@]+}}@test_sle() {
; CHECK-NEXT:    [[TMP1:%.*]] = tail call noalias i8* @malloc(i64 noundef -1)
; CHECK-NEXT:    tail call void @no_sync_func(i8* noalias nocapture nofree [[TMP1]])
; CHECK-NEXT:    [[TMP2:%.*]] = bitcast i8* [[TMP1]] to i32*
; CHECK-NEXT:    store i32 10, i32* [[TMP2]], align 4
; CHECK-NEXT:    [[TMP3:%.*]] = load i32, i32* [[TMP2]], align 4
; CHECK-NEXT:    tail call void @free(i8* noalias nocapture noundef nonnull align 4 dereferenceable(4) [[TMP1]])
; CHECK-NEXT:    ret i32 [[TMP3]]
;
  %1 = tail call noalias i8* @malloc(i64 -1)
  tail call void @no_sync_func(i8* %1)
  %2 = bitcast i8* %1 to i32*
  store i32 10, i32* %2
  %3 = load i32, i32* %2
  tail call void @free(i8* %1)
  ret i32 %3
}

define i32 @test_overflow() {
; CHECK-LABEL: define {{[^@]+}}@test_overflow() {
; CHECK-NEXT:    [[TMP1:%.*]] = tail call noalias i8* @calloc(i64 noundef 65537, i64 noundef 65537)
; CHECK-NEXT:    tail call void @no_sync_func(i8* noalias nocapture nofree [[TMP1]])
; CHECK-NEXT:    [[TMP2:%.*]] = bitcast i8* [[TMP1]] to i32*
; CHECK-NEXT:    store i32 10, i32* [[TMP2]], align 4
; CHECK-NEXT:    [[TMP3:%.*]] = load i32, i32* [[TMP2]], align 4
; CHECK-NEXT:    tail call void @free(i8* noalias nocapture noundef nonnull align 4 dereferenceable(4) [[TMP1]])
; CHECK-NEXT:    ret i32 [[TMP3]]
;
  %1 = tail call noalias i8* @calloc(i64 65537, i64 65537)
  tail call void @no_sync_func(i8* %1)
  %2 = bitcast i8* %1 to i32*
  store i32 10, i32* %2
  %3 = load i32, i32* %2
  tail call void @free(i8* %1)
  ret i32 %3
}

define void @test14() {
; CHECK-LABEL: define {{[^@]+}}@test14() {
; CHECK-NEXT:    [[TMP1:%.*]] = tail call noalias i8* @calloc(i64 noundef 64, i64 noundef 4)
; CHECK-NEXT:    tail call void @no_sync_func(i8* noalias nocapture nofree [[TMP1]])
; CHECK-NEXT:    tail call void @free(i8* noalias nocapture [[TMP1]])
; CHECK-NEXT:    ret void
;
  %1 = tail call noalias i8* @calloc(i64 64, i64 4)
  tail call void @no_sync_func(i8* %1)
  tail call void @free(i8* %1)
  ret void
}

define void @test15(i64 %S) {
; CHECK-LABEL: define {{[^@]+}}@test15
; CHECK-SAME: (i64 [[S:%.*]]) {
; CHECK-NEXT:    [[TMP1:%.*]] = tail call noalias i8* @malloc(i64 [[S]])
; CHECK-NEXT:    tail call void @no_sync_func(i8* noalias nocapture nofree [[TMP1]])
; CHECK-NEXT:    tail call void @free(i8* noalias nocapture [[TMP1]])
; CHECK-NEXT:    ret void
;
  %1 = tail call noalias i8* @malloc(i64 %S)
  tail call void @no_sync_func(i8* %1)
  tail call void @free(i8* %1)
  ret void
}

define void @test16a(i8 %v, i8** %P) {
; IS________OPM-LABEL: define {{[^@]+}}@test16a
; IS________OPM-SAME: (i8 [[V:%.*]], i8** nocapture nofree readnone [[P:%.*]]) {
; IS________OPM-NEXT:    [[TMP1:%.*]] = tail call noalias i8* @malloc(i64 noundef 4)
; IS________OPM-NEXT:    store i8 [[V]], i8* [[TMP1]], align 1
; IS________OPM-NEXT:    tail call void @no_sync_func(i8* noalias nocapture nofree noundef nonnull dereferenceable(1) [[TMP1]])
; IS________OPM-NEXT:    tail call void @free(i8* noalias nocapture noundef nonnull dereferenceable(1) [[TMP1]])
; IS________OPM-NEXT:    ret void
;
; IS________NPM-LABEL: define {{[^@]+}}@test16a
; IS________NPM-SAME: (i8 [[V:%.*]], i8** nocapture nofree readnone [[P:%.*]]) {
; IS________NPM-NEXT:    [[TMP1:%.*]] = alloca i8, i64 4, align 1
; IS________NPM-NEXT:    store i8 [[V]], i8* [[TMP1]], align 1
; IS________NPM-NEXT:    tail call void @no_sync_func(i8* noalias nocapture nofree noundef nonnull dereferenceable(1) [[TMP1]])
; IS________NPM-NEXT:    ret void
;
  %1 = tail call noalias i8* @malloc(i64 4)
  store i8 %v, i8* %1
  tail call void @no_sync_func(i8* %1)
  tail call void @free(i8* nonnull dereferenceable(1) %1)
  ret void
}

define void @test16b(i8 %v, i8** %P) {
; CHECK-LABEL: define {{[^@]+}}@test16b
; CHECK-SAME: (i8 [[V:%.*]], i8** nocapture nofree writeonly [[P:%.*]]) {
; CHECK-NEXT:    [[TMP1:%.*]] = tail call noalias i8* @malloc(i64 noundef 4)
; CHECK-NEXT:    store i8* [[TMP1]], i8** [[P]], align 8
; CHECK-NEXT:    tail call void @no_sync_func(i8* nocapture nofree [[TMP1]])
; CHECK-NEXT:    tail call void @free(i8* nocapture [[TMP1]])
; CHECK-NEXT:    ret void
;
  %1 = tail call noalias i8* @malloc(i64 4)
  store i8* %1, i8** %P
  tail call void @no_sync_func(i8* %1)
  tail call void @free(i8* %1)
  ret void
}

define void @test16c(i8 %v, i8** %P) {
; IS________OPM-LABEL: define {{[^@]+}}@test16c
; IS________OPM-SAME: (i8 [[V:%.*]], i8** nocapture nofree writeonly [[P:%.*]]) {
; IS________OPM-NEXT:    [[TMP1:%.*]] = tail call noalias i8* @malloc(i64 noundef 4)
; IS________OPM-NEXT:    store i8* [[TMP1]], i8** [[P]], align 8
; IS________OPM-NEXT:    tail call void @no_sync_func(i8* nocapture nofree [[TMP1]]) #[[ATTR5]]
; IS________OPM-NEXT:    tail call void @free(i8* nocapture [[TMP1]])
; IS________OPM-NEXT:    ret void
;
; IS________NPM-LABEL: define {{[^@]+}}@test16c
; IS________NPM-SAME: (i8 [[V:%.*]], i8** nocapture nofree writeonly [[P:%.*]]) {
; IS________NPM-NEXT:    [[TMP1:%.*]] = alloca i8, i64 4, align 1
; IS________NPM-NEXT:    store i8* [[TMP1]], i8** [[P]], align 8
; IS________NPM-NEXT:    tail call void @no_sync_func(i8* nocapture nofree [[TMP1]]) #[[ATTR6]]
; IS________NPM-NEXT:    ret void
;
  %1 = tail call noalias i8* @malloc(i64 4)
  store i8* %1, i8** %P
  tail call void @no_sync_func(i8* %1) nounwind
  tail call void @free(i8* %1)
  ret void
}

define void @test16d(i8 %v, i8** %P) {
; CHECK-LABEL: define {{[^@]+}}@test16d
; CHECK-SAME: (i8 [[V:%.*]], i8** nocapture nofree writeonly [[P:%.*]]) {
; CHECK-NEXT:    [[TMP1:%.*]] = tail call noalias i8* @malloc(i64 noundef 4)
; CHECK-NEXT:    store i8* [[TMP1]], i8** [[P]], align 8
; CHECK-NEXT:    ret void
;
  %1 = tail call noalias i8* @malloc(i64 4)
  store i8* %1, i8** %P
  ret void
}
;.
; IS________OPM: attributes #[[ATTR0:[0-9]+]] = { nounwind willreturn }
; IS________OPM: attributes #[[ATTR1:[0-9]+]] = { nofree nosync willreturn }
; IS________OPM: attributes #[[ATTR2:[0-9]+]] = { nofree nounwind }
; IS________OPM: attributes #[[ATTR3]] = { noreturn }
; IS________OPM: attributes #[[ATTR4:[0-9]+]] = { argmemonly nofree nosync nounwind willreturn }
; IS________OPM: attributes #[[ATTR5]] = { nounwind }
;.
; IS________NPM: attributes #[[ATTR0:[0-9]+]] = { nounwind willreturn }
; IS________NPM: attributes #[[ATTR1:[0-9]+]] = { nofree nosync willreturn }
; IS________NPM: attributes #[[ATTR2:[0-9]+]] = { nofree nounwind }
; IS________NPM: attributes #[[ATTR3]] = { noreturn }
; IS________NPM: attributes #[[ATTR4:[0-9]+]] = { argmemonly nofree nosync nounwind willreturn }
; IS________NPM: attributes #[[ATTR5:[0-9]+]] = { argmemonly nofree nounwind willreturn writeonly }
; IS________NPM: attributes #[[ATTR6]] = { nounwind }
;.
