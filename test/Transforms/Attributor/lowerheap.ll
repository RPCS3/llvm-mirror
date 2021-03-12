; NOTE: Assertions have been autogenerated by utils/update_test_checks.py UTC_ARGS: --function-signature --check-attributes --check-globals
; RUN: opt -max-heap-to-stack-size=-1 -attributor -enable-new-pm=0 -attributor-manifest-internal  -attributor-max-iterations-verify -attributor-annotate-decl-cs -attributor-max-iterations=1 -S < %s | FileCheck %s --check-prefixes=CHECK,NOT_CGSCC_NPM,NOT_CGSCC_OPM,NOT_TUNIT_NPM,IS__TUNIT____,IS________OPM,IS__TUNIT_OPM
; RUN: opt -max-heap-to-stack-size=-1 -aa-pipeline=basic-aa -passes=attributor -attributor-manifest-internal  -attributor-max-iterations-verify -attributor-annotate-decl-cs -attributor-max-iterations=1 -S < %s | FileCheck %s --check-prefixes=CHECK,NOT_CGSCC_OPM,NOT_CGSCC_NPM,NOT_TUNIT_OPM,IS__TUNIT____,IS________NPM,IS__TUNIT_NPM
; RUN: opt -max-heap-to-stack-size=-1 -attributor-cgscc -enable-new-pm=0 -attributor-manifest-internal  -attributor-annotate-decl-cs -S < %s | FileCheck %s --check-prefixes=CHECK,NOT_TUNIT_NPM,NOT_TUNIT_OPM,NOT_CGSCC_NPM,IS__CGSCC____,IS________OPM,IS__CGSCC_OPM
; RUN: opt -max-heap-to-stack-size=-1 -aa-pipeline=basic-aa -passes=attributor-cgscc -attributor-manifest-internal  -attributor-annotate-decl-cs -S < %s | FileCheck %s --check-prefixes=CHECK,NOT_TUNIT_NPM,NOT_TUNIT_OPM,NOT_CGSCC_OPM,IS__CGSCC____,IS________NPM,IS__CGSCC_NPM

declare i64 @subfn(i8*) #0

declare noalias i8* @malloc(i64)
declare noalias i8* @calloc(i64, i64)
declare void @free(i8*)

define i64 @f(i64 %len) {
; IS________OPM-LABEL: define {{[^@]+}}@f
; IS________OPM-SAME: (i64 [[LEN:%.*]]) {
; IS________OPM-NEXT:  entry:
; IS________OPM-NEXT:    [[MEM:%.*]] = call noalias i8* @malloc(i64 [[LEN]])
; IS________OPM-NEXT:    [[RES:%.*]] = call i64 @subfn(i8* [[MEM]]) #[[ATTR1:[0-9]+]]
; IS________OPM-NEXT:    call void @free(i8* [[MEM]])
; IS________OPM-NEXT:    ret i64 [[RES]]
;
; IS________NPM-LABEL: define {{[^@]+}}@f
; IS________NPM-SAME: (i64 [[LEN:%.*]]) {
; IS________NPM-NEXT:  entry:
; IS________NPM-NEXT:    [[TMP0:%.*]] = alloca i8, i64 [[LEN]], align 1
; IS________NPM-NEXT:    [[RES:%.*]] = call i64 @subfn(i8* [[TMP0]]) #[[ATTR2:[0-9]+]]
; IS________NPM-NEXT:    ret i64 [[RES]]
;
entry:
  %mem = call i8* @malloc(i64 %len)
  %res = call i64 @subfn(i8* %mem)
  call void @free(i8* %mem)
  ret i64 %res
}


define i64 @g(i64 %len) {
; IS________OPM-LABEL: define {{[^@]+}}@g
; IS________OPM-SAME: (i64 [[LEN:%.*]]) {
; IS________OPM-NEXT:  entry:
; IS________OPM-NEXT:    [[MEM:%.*]] = call noalias i8* @calloc(i64 [[LEN]], i64 noundef 8)
; IS________OPM-NEXT:    [[RES:%.*]] = call i64 @subfn(i8* [[MEM]]) #[[ATTR1]]
; IS________OPM-NEXT:    call void @free(i8* [[MEM]])
; IS________OPM-NEXT:    ret i64 [[RES]]
;
; IS________NPM-LABEL: define {{[^@]+}}@g
; IS________NPM-SAME: (i64 [[LEN:%.*]]) {
; IS________NPM-NEXT:  entry:
; IS________NPM-NEXT:    [[H2S_CALLOC_SIZE:%.*]] = mul i64 [[LEN]], 8
; IS________NPM-NEXT:    [[TMP0:%.*]] = alloca i8, i64 [[H2S_CALLOC_SIZE]], align 1
; IS________NPM-NEXT:    [[CALLOC_BC:%.*]] = bitcast i8* [[TMP0]] to i8*
; IS________NPM-NEXT:    call void @llvm.memset.p0i8.i64(i8* [[CALLOC_BC]], i8 0, i64 [[H2S_CALLOC_SIZE]], i1 false)
; IS________NPM-NEXT:    [[RES:%.*]] = call i64 @subfn(i8* [[TMP0]]) #[[ATTR2]]
; IS________NPM-NEXT:    ret i64 [[RES]]
;
entry:
  %mem = call i8* @calloc(i64 %len, i64 8)
  %res = call i64 @subfn(i8* %mem)
  call void @free(i8* %mem)
  ret i64 %res
}

attributes #0 = { nounwind willreturn }
;.
; IS________OPM: attributes #[[ATTR0:[0-9]+]] = { nounwind willreturn }
; IS________OPM: attributes #[[ATTR1]] = { nounwind }
;.
; IS________NPM: attributes #[[ATTR0:[0-9]+]] = { nounwind willreturn }
; IS________NPM: attributes #[[ATTR1:[0-9]+]] = { argmemonly nofree nosync nounwind willreturn writeonly }
; IS________NPM: attributes #[[ATTR2]] = { nounwind }
;.
