; NOTE: Assertions have been autogenerated by utils/update_test_checks.py
; RUN: opt < %s -instcombine -S | FileCheck %s

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"

@a = common global [60 x i8] zeroinitializer, align 1
@b = common global [60 x i8] zeroinitializer, align 1
@.str = private constant [12 x i8] c"abcdefghijk\00"

%struct.__va_list_tag = type { i32, i32, i8*, i8* }

define i8* @test_memccpy() {
; CHECK-LABEL: @test_memccpy(
; CHECK-NEXT:    [[MEMCCPY:%.*]] = call i8* @memccpy(i8* noalias writeonly getelementptr inbounds ([60 x i8], [60 x i8]* @a, i64 0, i64 0), i8* noalias nocapture readonly getelementptr inbounds ([60 x i8], [60 x i8]* @b, i64 0, i64 0), i32 0, i64 60) #[[ATTR0:[0-9]+]]
; CHECK-NEXT:    ret i8* [[MEMCCPY]]
;
  %dst = getelementptr inbounds [60 x i8], [60 x i8]* @a, i32 0, i32 0
  %src = getelementptr inbounds [60 x i8], [60 x i8]* @b, i32 0, i32 0
  %ret = call i8* @__memccpy_chk(i8* %dst, i8* %src, i32 0, i64 60, i64 -1)
  ret i8* %ret
}

define i8* @test_not_memccpy() {
; CHECK-LABEL: @test_not_memccpy(
; CHECK-NEXT:    [[RET:%.*]] = call i8* @__memccpy_chk(i8* getelementptr inbounds ([60 x i8], [60 x i8]* @a, i64 0, i64 0), i8* getelementptr inbounds ([60 x i8], [60 x i8]* @b, i64 0, i64 0), i32 0, i64 60, i64 59)
; CHECK-NEXT:    ret i8* [[RET]]
;
  %dst = getelementptr inbounds [60 x i8], [60 x i8]* @a, i32 0, i32 0
  %src = getelementptr inbounds [60 x i8], [60 x i8]* @b, i32 0, i32 0
  %ret = call i8* @__memccpy_chk(i8* %dst, i8* %src, i32 0, i64 60, i64 59)
  ret i8* %ret
}

define i8* @test_mempcpy() {
; CHECK-LABEL: @test_mempcpy(
; CHECK-NEXT:    call void @llvm.memcpy.p0i8.p0i8.i64(i8* noundef nonnull align 1 dereferenceable(15) getelementptr inbounds ([60 x i8], [60 x i8]* @a, i64 0, i64 0), i8* noundef nonnull align 1 dereferenceable(15) getelementptr inbounds ([60 x i8], [60 x i8]* @b, i64 0, i64 0), i64 15, i1 false)
; CHECK-NEXT:    ret i8* getelementptr inbounds ([60 x i8], [60 x i8]* @a, i64 0, i64 15)
;
  %dst = getelementptr inbounds [60 x i8], [60 x i8]* @a, i32 0, i32 0
  %src = getelementptr inbounds [60 x i8], [60 x i8]* @b, i32 0, i32 0
  %ret = call i8* @__mempcpy_chk(i8* %dst, i8* %src, i64 15, i64 -1)
  ret i8* %ret
}

define i8* @test_not_mempcpy() {
; CHECK-LABEL: @test_not_mempcpy(
; CHECK-NEXT:    [[RET:%.*]] = call i8* @__mempcpy_chk(i8* getelementptr inbounds ([60 x i8], [60 x i8]* @a, i64 0, i64 0), i8* getelementptr inbounds ([60 x i8], [60 x i8]* @b, i64 0, i64 0), i64 60, i64 59)
; CHECK-NEXT:    ret i8* [[RET]]
;
  %dst = getelementptr inbounds [60 x i8], [60 x i8]* @a, i32 0, i32 0
  %src = getelementptr inbounds [60 x i8], [60 x i8]* @b, i32 0, i32 0
  %ret = call i8* @__mempcpy_chk(i8* %dst, i8* %src, i64 60, i64 59)
  ret i8* %ret
}

define i32 @test_snprintf() {
; CHECK-LABEL: @test_snprintf(
; CHECK-NEXT:    [[SNPRINTF:%.*]] = call noundef i32 (i8*, i64, i8*, ...) @snprintf(i8* noalias nocapture noundef nonnull writeonly dereferenceable(1) getelementptr inbounds ([60 x i8], [60 x i8]* @a, i64 0, i64 0), i64 noundef 60, i8* nocapture noundef readonly getelementptr inbounds ([60 x i8], [60 x i8]* @b, i64 0, i64 0)) #[[ATTR1:[0-9]+]]
; CHECK-NEXT:    ret i32 [[SNPRINTF]]
;
  %dst = getelementptr inbounds [60 x i8], [60 x i8]* @a, i32 0, i32 0
  %fmt = getelementptr inbounds [60 x i8], [60 x i8]* @b, i32 0, i32 0
  %ret = call i32 (i8*, i64, i32, i64, i8*, ...) @__snprintf_chk(i8* %dst, i64 60, i32 0, i64 -1, i8* %fmt)
  ret i32 %ret
}

define i32 @test_not_snprintf() {
; CHECK-LABEL: @test_not_snprintf(
; CHECK-NEXT:    [[RET:%.*]] = call i32 (i8*, i64, i32, i64, i8*, ...) @__snprintf_chk(i8* getelementptr inbounds ([60 x i8], [60 x i8]* @a, i64 0, i64 0), i64 60, i32 0, i64 59, i8* getelementptr inbounds ([60 x i8], [60 x i8]* @b, i64 0, i64 0))
; CHECK-NEXT:    [[IGN:%.*]] = call i32 (i8*, i64, i32, i64, i8*, ...) @__snprintf_chk(i8* getelementptr inbounds ([60 x i8], [60 x i8]* @a, i64 0, i64 0), i64 60, i32 1, i64 -1, i8* getelementptr inbounds ([60 x i8], [60 x i8]* @b, i64 0, i64 0))
; CHECK-NEXT:    ret i32 [[RET]]
;
  %dst = getelementptr inbounds [60 x i8], [60 x i8]* @a, i32 0, i32 0
  %fmt = getelementptr inbounds [60 x i8], [60 x i8]* @b, i32 0, i32 0
  %ret = call i32 (i8*, i64, i32, i64, i8*, ...) @__snprintf_chk(i8* %dst, i64 60, i32 0, i64 59, i8* %fmt)
  %ign = call i32 (i8*, i64, i32, i64, i8*, ...) @__snprintf_chk(i8* %dst, i64 60, i32 1, i64 -1, i8* %fmt)
  ret i32 %ret
}

define i32 @test_sprintf() {
; CHECK-LABEL: @test_sprintf(
; CHECK-NEXT:    [[SPRINTF:%.*]] = call noundef i32 (i8*, i8*, ...) @sprintf(i8* noalias nocapture noundef nonnull writeonly dereferenceable(1) getelementptr inbounds ([60 x i8], [60 x i8]* @a, i64 0, i64 0), i8* nocapture noundef nonnull readonly dereferenceable(1) getelementptr inbounds ([60 x i8], [60 x i8]* @b, i64 0, i64 0)) #[[ATTR1]]
; CHECK-NEXT:    ret i32 [[SPRINTF]]
;
  %dst = getelementptr inbounds [60 x i8], [60 x i8]* @a, i32 0, i32 0
  %fmt = getelementptr inbounds [60 x i8], [60 x i8]* @b, i32 0, i32 0
  %ret = call i32 (i8*, i32, i64, i8*, ...) @__sprintf_chk(i8* %dst, i32 0, i64 -1, i8* %fmt)
  ret i32 %ret
}

define i32 @test_not_sprintf() {
; CHECK-LABEL: @test_not_sprintf(
; CHECK-NEXT:    [[RET:%.*]] = call i32 (i8*, i32, i64, i8*, ...) @__sprintf_chk(i8* getelementptr inbounds ([60 x i8], [60 x i8]* @a, i64 0, i64 0), i32 0, i64 59, i8* getelementptr inbounds ([60 x i8], [60 x i8]* @b, i64 0, i64 0))
; CHECK-NEXT:    [[IGNORED:%.*]] = call i32 (i8*, i32, i64, i8*, ...) @__sprintf_chk(i8* getelementptr inbounds ([60 x i8], [60 x i8]* @a, i64 0, i64 0), i32 1, i64 -1, i8* getelementptr inbounds ([60 x i8], [60 x i8]* @b, i64 0, i64 0))
; CHECK-NEXT:    ret i32 [[RET]]
;
  %dst = getelementptr inbounds [60 x i8], [60 x i8]* @a, i32 0, i32 0
  %fmt = getelementptr inbounds [60 x i8], [60 x i8]* @b, i32 0, i32 0
  %ret = call i32 (i8*, i32, i64, i8*, ...) @__sprintf_chk(i8* %dst, i32 0, i64 59, i8* %fmt)
  %ignored = call i32 (i8*, i32, i64, i8*, ...) @__sprintf_chk(i8* %dst, i32 1, i64 -1, i8* %fmt)
  ret i32 %ret
}

define i8* @test_strcat() {
; CHECK-LABEL: @test_strcat(
; CHECK-NEXT:    [[STRCAT:%.*]] = call i8* @strcat(i8* noalias noundef nonnull returned dereferenceable(1) getelementptr inbounds ([60 x i8], [60 x i8]* @a, i64 0, i64 0), i8* noalias nocapture noundef nonnull readonly dereferenceable(1) getelementptr inbounds ([60 x i8], [60 x i8]* @b, i64 0, i64 0)) #[[ATTR0]]
; CHECK-NEXT:    ret i8* getelementptr inbounds ([60 x i8], [60 x i8]* @a, i64 0, i64 0)
;
  %dst = getelementptr inbounds [60 x i8], [60 x i8]* @a, i32 0, i32 0
  %src = getelementptr inbounds [60 x i8], [60 x i8]* @b, i32 0, i32 0
  %ret = call i8* @__strcat_chk(i8* %dst, i8* %src, i64 -1)
  ret i8* %ret
}

define i8* @test_not_strcat() {
; CHECK-LABEL: @test_not_strcat(
; CHECK-NEXT:    [[RET:%.*]] = call i8* @__strcat_chk(i8* getelementptr inbounds ([60 x i8], [60 x i8]* @a, i64 0, i64 0), i8* getelementptr inbounds ([60 x i8], [60 x i8]* @b, i64 0, i64 0), i64 0)
; CHECK-NEXT:    ret i8* [[RET]]
;
  %dst = getelementptr inbounds [60 x i8], [60 x i8]* @a, i32 0, i32 0
  %src = getelementptr inbounds [60 x i8], [60 x i8]* @b, i32 0, i32 0
  %ret = call i8* @__strcat_chk(i8* %dst, i8* %src, i64 0)
  ret i8* %ret
}

define i64 @test_strlcat() {
; CHECK-LABEL: @test_strlcat(
; CHECK-NEXT:    [[STRLCAT:%.*]] = call i64 @strlcat(i8* getelementptr inbounds ([60 x i8], [60 x i8]* @a, i64 0, i64 0), i8* getelementptr inbounds ([60 x i8], [60 x i8]* @b, i64 0, i64 0), i64 22) #[[ATTR2:[0-9]+]]
; CHECK-NEXT:    ret i64 [[STRLCAT]]
;
  %dst = getelementptr inbounds [60 x i8], [60 x i8]* @a, i32 0, i32 0
  %src = getelementptr inbounds [60 x i8], [60 x i8]* @b, i32 0, i32 0
  %ret = call i64 @__strlcat_chk(i8* %dst, i8* %src, i64 22, i64 -1)
  ret i64 %ret
}

define i64 @test_not_strlcat() {
; CHECK-LABEL: @test_not_strlcat(
; CHECK-NEXT:    [[RET:%.*]] = call i64 @__strlcat_chk(i8* getelementptr inbounds ([60 x i8], [60 x i8]* @a, i64 0, i64 0), i8* getelementptr inbounds ([60 x i8], [60 x i8]* @b, i64 0, i64 0), i64 22, i64 0)
; CHECK-NEXT:    ret i64 [[RET]]
;
  %dst = getelementptr inbounds [60 x i8], [60 x i8]* @a, i32 0, i32 0
  %src = getelementptr inbounds [60 x i8], [60 x i8]* @b, i32 0, i32 0
  %ret = call i64 @__strlcat_chk(i8* %dst, i8* %src, i64 22, i64 0)
  ret i64 %ret
}

define i8* @test_strncat() {
; CHECK-LABEL: @test_strncat(
; CHECK-NEXT:    [[STRNCAT:%.*]] = call i8* @strncat(i8* noalias noundef nonnull returned dereferenceable(1) getelementptr inbounds ([60 x i8], [60 x i8]* @a, i64 0, i64 0), i8* noalias nocapture noundef nonnull readonly dereferenceable(1) getelementptr inbounds ([60 x i8], [60 x i8]* @b, i64 0, i64 0), i64 22) #[[ATTR0]]
; CHECK-NEXT:    ret i8* getelementptr inbounds ([60 x i8], [60 x i8]* @a, i64 0, i64 0)
;
  %dst = getelementptr inbounds [60 x i8], [60 x i8]* @a, i32 0, i32 0
  %src = getelementptr inbounds [60 x i8], [60 x i8]* @b, i32 0, i32 0
  %ret = call i8* @__strncat_chk(i8* %dst, i8* %src, i64 22, i64 -1)
  ret i8* %ret
}

define i8* @test_not_strncat() {
; CHECK-LABEL: @test_not_strncat(
; CHECK-NEXT:    [[RET:%.*]] = call i8* @__strncat_chk(i8* getelementptr inbounds ([60 x i8], [60 x i8]* @a, i64 0, i64 0), i8* getelementptr inbounds ([60 x i8], [60 x i8]* @b, i64 0, i64 0), i64 22, i64 3)
; CHECK-NEXT:    ret i8* [[RET]]
;
  %dst = getelementptr inbounds [60 x i8], [60 x i8]* @a, i32 0, i32 0
  %src = getelementptr inbounds [60 x i8], [60 x i8]* @b, i32 0, i32 0
  %ret = call i8* @__strncat_chk(i8* %dst, i8* %src, i64 22, i64 3)
  ret i8* %ret
}

define i64 @test_strlcpy() {
; CHECK-LABEL: @test_strlcpy(
; CHECK-NEXT:    [[STRLCPY:%.*]] = call i64 @strlcpy(i8* getelementptr inbounds ([60 x i8], [60 x i8]* @a, i64 0, i64 0), i8* getelementptr inbounds ([60 x i8], [60 x i8]* @b, i64 0, i64 0), i64 22) #[[ATTR2]]
; CHECK-NEXT:    ret i64 [[STRLCPY]]
;
  %dst = getelementptr inbounds [60 x i8], [60 x i8]* @a, i32 0, i32 0
  %src = getelementptr inbounds [60 x i8], [60 x i8]* @b, i32 0, i32 0
  %ret = call i64 @__strlcpy_chk(i8* %dst, i8* %src, i64 22, i64 -1)
  ret i64 %ret
}

define i64 @test_not_strlcpy() {
; CHECK-LABEL: @test_not_strlcpy(
; CHECK-NEXT:    [[RET:%.*]] = call i64 @__strlcpy_chk(i8* getelementptr inbounds ([60 x i8], [60 x i8]* @a, i64 0, i64 0), i8* getelementptr inbounds ([60 x i8], [60 x i8]* @b, i64 0, i64 0), i64 22, i64 2)
; CHECK-NEXT:    ret i64 [[RET]]
;
  %dst = getelementptr inbounds [60 x i8], [60 x i8]* @a, i32 0, i32 0
  %src = getelementptr inbounds [60 x i8], [60 x i8]* @b, i32 0, i32 0
  %ret = call i64 @__strlcpy_chk(i8* %dst, i8* %src, i64 22, i64 2)
  ret i64 %ret
}

define i32 @test_vsnprintf() {
; CHECK-LABEL: @test_vsnprintf(
; CHECK-NEXT:    [[VSNPRINTF:%.*]] = call noundef i32 @vsnprintf(i8* nocapture noundef getelementptr inbounds ([60 x i8], [60 x i8]* @a, i64 0, i64 0), i64 noundef 4, i8* nocapture noundef readonly getelementptr inbounds ([60 x i8], [60 x i8]* @b, i64 0, i64 0), %struct.__va_list_tag* noundef null) #[[ATTR1]]
; CHECK-NEXT:    ret i32 [[VSNPRINTF]]
;
  ; ret i32
  %dst = getelementptr inbounds [60 x i8], [60 x i8]* @a, i32 0, i32 0
  %src = getelementptr inbounds [60 x i8], [60 x i8]* @b, i32 0, i32 0
  %ret = call i32 @__vsnprintf_chk(i8* %dst, i64 4, i32 0, i64 -1, i8* %src, %struct.__va_list_tag* null)
  ret i32 %ret
}

define i32 @test_not_vsnprintf() {
; CHECK-LABEL: @test_not_vsnprintf(
; CHECK-NEXT:    [[RET:%.*]] = call i32 @__vsnprintf_chk(i8* getelementptr inbounds ([60 x i8], [60 x i8]* @a, i64 0, i64 0), i64 4, i32 0, i64 3, i8* getelementptr inbounds ([60 x i8], [60 x i8]* @b, i64 0, i64 0), %struct.__va_list_tag* null)
; CHECK-NEXT:    [[IGN:%.*]] = call i32 @__vsnprintf_chk(i8* getelementptr inbounds ([60 x i8], [60 x i8]* @a, i64 0, i64 0), i64 4, i32 1, i64 -1, i8* getelementptr inbounds ([60 x i8], [60 x i8]* @b, i64 0, i64 0), %struct.__va_list_tag* null)
; CHECK-NEXT:    ret i32 [[RET]]
;
  ; ret i32
  %dst = getelementptr inbounds [60 x i8], [60 x i8]* @a, i32 0, i32 0
  %src = getelementptr inbounds [60 x i8], [60 x i8]* @b, i32 0, i32 0
  %ret = call i32 @__vsnprintf_chk(i8* %dst, i64 4, i32 0, i64 3, i8* %src, %struct.__va_list_tag* null)
  %ign = call i32 @__vsnprintf_chk(i8* %dst, i64 4, i32 1, i64 -1, i8* %src, %struct.__va_list_tag* null)
  ret i32 %ret
}

define i32 @test_vsprintf() {
; CHECK-LABEL: @test_vsprintf(
; CHECK-NEXT:    [[VSPRINTF:%.*]] = call noundef i32 @vsprintf(i8* nocapture noundef getelementptr inbounds ([60 x i8], [60 x i8]* @a, i64 0, i64 0), i8* nocapture noundef readonly getelementptr inbounds ([60 x i8], [60 x i8]* @b, i64 0, i64 0), %struct.__va_list_tag* noundef null) #[[ATTR1]]
; CHECK-NEXT:    ret i32 [[VSPRINTF]]
;
  ; ret i32
  %dst = getelementptr inbounds [60 x i8], [60 x i8]* @a, i32 0, i32 0
  %src = getelementptr inbounds [60 x i8], [60 x i8]* @b, i32 0, i32 0
  %ret = call i32 @__vsprintf_chk(i8* %dst, i32 0, i64 -1, i8* %src, %struct.__va_list_tag* null)
  ret i32 %ret
}

define i32 @test_not_vsprintf() {
; CHECK-LABEL: @test_not_vsprintf(
; CHECK-NEXT:    [[RET:%.*]] = call i32 @__vsprintf_chk(i8* getelementptr inbounds ([60 x i8], [60 x i8]* @a, i64 0, i64 0), i32 0, i64 3, i8* getelementptr inbounds ([60 x i8], [60 x i8]* @b, i64 0, i64 0), %struct.__va_list_tag* null)
; CHECK-NEXT:    [[IGN:%.*]] = call i32 @__vsprintf_chk(i8* getelementptr inbounds ([60 x i8], [60 x i8]* @a, i64 0, i64 0), i32 1, i64 -1, i8* getelementptr inbounds ([60 x i8], [60 x i8]* @b, i64 0, i64 0), %struct.__va_list_tag* null)
; CHECK-NEXT:    ret i32 [[RET]]
;
  ; ret i32
  %dst = getelementptr inbounds [60 x i8], [60 x i8]* @a, i32 0, i32 0
  %src = getelementptr inbounds [60 x i8], [60 x i8]* @b, i32 0, i32 0
  %ret = call i32 @__vsprintf_chk(i8* %dst, i32 0, i64 3, i8* %src, %struct.__va_list_tag* null)
  %ign = call i32 @__vsprintf_chk(i8* %dst, i32 1, i64 -1, i8* %src, %struct.__va_list_tag* null)
  ret i32 %ret
}

declare i8* @__mempcpy_chk(i8*, i8*, i64, i64)
declare i8* @__memccpy_chk(i8*, i8*, i32, i64, i64)
declare i32 @__snprintf_chk(i8*, i64, i32, i64, i8*, ...)
declare i32 @__sprintf_chk(i8*, i32, i64, i8*, ...)
declare i8* @__strcat_chk(i8*, i8*, i64)
declare i64 @__strlcat_chk(i8*, i8*, i64, i64)
declare i8* @__strncat_chk(i8*, i8*, i64, i64)
declare i64 @__strlcpy_chk(i8*, i8*, i64, i64)
declare i32 @__vsnprintf_chk(i8*, i64, i32, i64, i8*, %struct.__va_list_tag*)
declare i32 @__vsprintf_chk(i8*, i32, i64, i8*, %struct.__va_list_tag*)
