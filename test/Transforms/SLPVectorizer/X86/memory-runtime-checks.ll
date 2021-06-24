; NOTE: Assertions have been autogenerated by utils/update_test_checks.py
; RUN: opt -scoped-noalias-aa -slp-vectorizer -mtriple=x86_64-apple-darwin -enable-new-pm=false -S %s | FileCheck %s

define void @version_multiple(i32* nocapture %out_block, i32* nocapture readonly %counter) {
; CHECK-LABEL: @version_multiple(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[TMP0:%.*]] = load i32, i32* [[COUNTER:%.*]], align 4
; CHECK-NEXT:    [[TMP1:%.*]] = load i32, i32* [[OUT_BLOCK:%.*]], align 4
; CHECK-NEXT:    [[XOR:%.*]] = xor i32 [[TMP1]], [[TMP0]]
; CHECK-NEXT:    store i32 [[XOR]], i32* [[OUT_BLOCK]], align 4
; CHECK-NEXT:    [[ARRAYIDX_1:%.*]] = getelementptr inbounds i32, i32* [[COUNTER]], i64 1
; CHECK-NEXT:    [[TMP2:%.*]] = load i32, i32* [[ARRAYIDX_1]], align 4
; CHECK-NEXT:    [[ARRAYIDX2_1:%.*]] = getelementptr inbounds i32, i32* [[OUT_BLOCK]], i64 1
; CHECK-NEXT:    [[TMP3:%.*]] = load i32, i32* [[ARRAYIDX2_1]], align 4
; CHECK-NEXT:    [[XOR_1:%.*]] = xor i32 [[TMP3]], [[TMP2]]
; CHECK-NEXT:    store i32 [[XOR_1]], i32* [[ARRAYIDX2_1]], align 4
; CHECK-NEXT:    [[ARRAYIDX_2:%.*]] = getelementptr inbounds i32, i32* [[COUNTER]], i64 2
; CHECK-NEXT:    [[TMP4:%.*]] = load i32, i32* [[ARRAYIDX_2]], align 4
; CHECK-NEXT:    [[ARRAYIDX2_2:%.*]] = getelementptr inbounds i32, i32* [[OUT_BLOCK]], i64 2
; CHECK-NEXT:    [[TMP5:%.*]] = load i32, i32* [[ARRAYIDX2_2]], align 4
; CHECK-NEXT:    [[XOR_2:%.*]] = xor i32 [[TMP5]], [[TMP4]]
; CHECK-NEXT:    store i32 [[XOR_2]], i32* [[ARRAYIDX2_2]], align 4
; CHECK-NEXT:    [[ARRAYIDX_3:%.*]] = getelementptr inbounds i32, i32* [[COUNTER]], i64 3
; CHECK-NEXT:    [[TMP6:%.*]] = load i32, i32* [[ARRAYIDX_3]], align 4
; CHECK-NEXT:    [[ARRAYIDX2_3:%.*]] = getelementptr inbounds i32, i32* [[OUT_BLOCK]], i64 3
; CHECK-NEXT:    [[TMP7:%.*]] = load i32, i32* [[ARRAYIDX2_3]], align 4
; CHECK-NEXT:    [[XOR_3:%.*]] = xor i32 [[TMP7]], [[TMP6]]
; CHECK-NEXT:    store i32 [[XOR_3]], i32* [[ARRAYIDX2_3]], align 4
; CHECK-NEXT:    ret void
;
entry:
  %0 = load i32, i32* %counter, align 4
  %1 = load i32, i32* %out_block, align 4
  %xor = xor i32 %1, %0
  store i32 %xor, i32* %out_block, align 4
  %arrayidx.1 = getelementptr inbounds i32, i32* %counter, i64 1
  %2 = load i32, i32* %arrayidx.1, align 4
  %arrayidx2.1 = getelementptr inbounds i32, i32* %out_block, i64 1
  %3 = load i32, i32* %arrayidx2.1, align 4
  %xor.1 = xor i32 %3, %2
  store i32 %xor.1, i32* %arrayidx2.1, align 4
  %arrayidx.2 = getelementptr inbounds i32, i32* %counter, i64 2
  %4 = load i32, i32* %arrayidx.2, align 4
  %arrayidx2.2 = getelementptr inbounds i32, i32* %out_block, i64 2
  %5 = load i32, i32* %arrayidx2.2, align 4
  %xor.2 = xor i32 %5, %4
  store i32 %xor.2, i32* %arrayidx2.2, align 4
  %arrayidx.3 = getelementptr inbounds i32, i32* %counter, i64 3
  %6 = load i32, i32* %arrayidx.3, align 4
  %arrayidx2.3 = getelementptr inbounds i32, i32* %out_block, i64 3
  %7 = load i32, i32* %arrayidx2.3, align 4
  %xor.3 = xor i32 %7, %6
  store i32 %xor.3, i32* %arrayidx2.3, align 4
  ret void
}

declare void @use(<8 x float>)
define void @delete_pointer_bound(float* %a, float* %b, i1 %c) #0 {
; CHECK-LABEL: @delete_pointer_bound(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[TMP0:%.*]] = insertelement <2 x float*> poison, float* [[B:%.*]], i32 0
; CHECK-NEXT:    [[TMP1:%.*]] = insertelement <2 x float*> [[TMP0]], float* [[B]], i32 1
; CHECK-NEXT:    [[TMP2:%.*]] = getelementptr float, <2 x float*> [[TMP1]], <2 x i64> <i64 10, i64 14>
; CHECK-NEXT:    br i1 [[C:%.*]], label [[THEN:%.*]], label [[ELSE:%.*]]
; CHECK:       else:
; CHECK-NEXT:    [[TMP3:%.*]] = call <2 x float> @llvm.masked.gather.v2f32.v2p0f32(<2 x float*> [[TMP2]], i32 4, <2 x i1> <i1 true, i1 true>, <2 x float> undef)
; CHECK-NEXT:    [[SHUFFLE:%.*]] = shufflevector <2 x float> [[TMP3]], <2 x float> poison, <4 x i32> <i32 0, i32 0, i32 1, i32 1>
; CHECK-NEXT:    [[TMP4:%.*]] = shufflevector <4 x float> [[SHUFFLE]], <4 x float> poison, <8 x i32> <i32 0, i32 1, i32 2, i32 undef, i32 undef, i32 3, i32 undef, i32 undef>
; CHECK-NEXT:    [[I71:%.*]] = shufflevector <8 x float> undef, <8 x float> [[TMP4]], <8 x i32> <i32 0, i32 1, i32 8, i32 9, i32 10, i32 5, i32 6, i32 13>
; CHECK-NEXT:    call void @use(<8 x float> [[I71]])
; CHECK-NEXT:    ret void
; CHECK:       then:
; CHECK-NEXT:    [[A_8:%.*]] = getelementptr inbounds float, float* [[A:%.*]], i64 8
; CHECK-NEXT:    store float 0.000000e+00, float* [[A_8]], align 4
; CHECK-NEXT:    [[TMP5:%.*]] = extractelement <2 x float*> [[TMP2]], i32 1
; CHECK-NEXT:    [[L6:%.*]] = load float, float* [[TMP5]], align 4
; CHECK-NEXT:    [[A_5:%.*]] = getelementptr inbounds float, float* [[A]], i64 5
; CHECK-NEXT:    store float [[L6]], float* [[A_5]], align 4
; CHECK-NEXT:    [[A_6:%.*]] = getelementptr inbounds float, float* [[A]], i64 6
; CHECK-NEXT:    store float 0.000000e+00, float* [[A_6]], align 4
; CHECK-NEXT:    [[A_7:%.*]] = getelementptr inbounds float, float* [[A]], i64 7
; CHECK-NEXT:    store float 0.000000e+00, float* [[A_7]], align 4
; CHECK-NEXT:    ret void
;
entry:
  %b.10 = getelementptr inbounds float, float* %b, i64 10
  %b.14 = getelementptr inbounds float, float* %b, i64 14
  br i1 %c, label %then, label %else

else:
  %l0 = load float, float* %b.10, align 4
  %l1 = load float, float* %b.14, align 4
  %i2 = insertelement <8 x float> undef, float %l0, i32 2
  %i3 = insertelement <8 x float> %i2, float %l0, i32 3
  %i4 = insertelement <8 x float> %i3, float %l1, i32 4
  %i7 = insertelement <8 x float> %i4, float %l1, i32 7
  call void @use(<8 x float> %i7)
  ret void

then:
  %a.8 = getelementptr inbounds float, float* %a, i64 8
  store float 0.0, float* %a.8, align 4
  %l6 = load float, float* %b.14, align 4
  %a.5 = getelementptr inbounds float, float* %a, i64 5
  store float %l6, float* %a.5, align 4
  %a.6 = getelementptr inbounds float, float* %a, i64 6
  store float 0.0, float* %a.6, align 4
  %a.7 = getelementptr inbounds float, float* %a, i64 7
  store float 0.0, float* %a.7, align 4
  ret void
}

%struct.zot = type { i16, i16, i16, i32, float, float, float, %struct.quux*, %struct.zot*, %struct.wombat*, %struct.wombat.0 }
%struct.quux = type { i16, %struct.quux*, %struct.quux* }
%struct.wombat = type { i32, i16, i8, i8, %struct.eggs* }
%struct.eggs = type { float, i8, %struct.ham }
%struct.ham = type { [2 x double], [8 x i8] }
%struct.wombat.0 = type { %struct.bar }
%struct.bar = type { [3 x double], [3 x double], double, double, i16, [3 x double]*, i32, [3 x double] }

define double @preserve_loop_info(%struct.zot* %arg) {
; CHECK-LABEL: @preserve_loop_info(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[TMP:%.*]] = alloca [3 x double], align 16
; CHECK-NEXT:    br label [[OUTER_HEADER:%.*]]
; CHECK:       outer.header:
; CHECK-NEXT:    br label [[INNER:%.*]]
; CHECK:       inner:
; CHECK-NEXT:    br i1 undef, label [[OUTER_LATCH:%.*]], label [[INNER]]
; CHECK:       outer.latch:
; CHECK-NEXT:    br i1 undef, label [[BB:%.*]], label [[OUTER_HEADER]]
; CHECK:       bb:
; CHECK-NEXT:    [[TMP5:%.*]] = load [3 x double]*, [3 x double]** undef, align 8
; CHECK-NEXT:    [[TMP6:%.*]] = getelementptr inbounds [3 x double], [3 x double]* [[TMP]], i64 0, i64 0
; CHECK-NEXT:    [[TMP7:%.*]] = getelementptr inbounds [3 x double], [3 x double]* [[TMP]], i64 0, i64 1
; CHECK-NEXT:    br label [[LOOP_3HEADER:%.*]]
; CHECK:       loop.3header:
; CHECK-NEXT:    br i1 undef, label [[LOOP_3LATCH:%.*]], label [[BB9:%.*]]
; CHECK:       bb9:
; CHECK-NEXT:    [[TMP10:%.*]] = getelementptr inbounds [3 x double], [3 x double]* [[TMP5]], i64 undef, i64 1
; CHECK-NEXT:    store double undef, double* [[TMP6]], align 16
; CHECK-NEXT:    [[TMP12:%.*]] = load double, double* [[TMP10]], align 8
; CHECK-NEXT:    store double [[TMP12]], double* [[TMP7]], align 8
; CHECK-NEXT:    br label [[LOOP_3LATCH]]
; CHECK:       loop.3latch:
; CHECK-NEXT:    br i1 undef, label [[BB14:%.*]], label [[LOOP_3HEADER]]
; CHECK:       bb14:
; CHECK-NEXT:    [[TMP15:%.*]] = call double undef(double* [[TMP6]], %struct.zot* [[ARG:%.*]])
; CHECK-NEXT:    ret double undef
;
entry:
  %tmp = alloca [3 x double], align 16
  br label %outer.header

outer.header:                                              ; preds = %bb3, %bb
  br label %inner

inner:
  br i1 undef, label %outer.latch, label %inner

outer.latch:                                              ; preds = %bb16
  br i1 undef, label %bb, label %outer.header

bb:                                              ; preds = %bb3
  %tmp5 = load [3 x double]*, [3 x double]** undef, align 8
  %tmp6 = getelementptr inbounds [3 x double], [3 x double]* %tmp, i64 0, i64 0
  %tmp7 = getelementptr inbounds [3 x double], [3 x double]* %tmp, i64 0, i64 1
  br label %loop.3header

loop.3header:                                              ; preds = %bb13, %bb4
  br i1 undef, label %loop.3latch, label %bb9

bb9:                                              ; preds = %bb8
  %tmp10 = getelementptr inbounds [3 x double], [3 x double]* %tmp5, i64 undef, i64 1
  store double undef, double* %tmp6, align 16
  %tmp12 = load double, double* %tmp10, align 8
  store double %tmp12, double* %tmp7, align 8
  br label %loop.3latch

loop.3latch:                                             ; preds = %bb11, %bb8
  br i1 undef, label %bb14, label %loop.3header

bb14:                                             ; preds = %bb13
  %tmp15 = call double undef(double* %tmp6, %struct.zot* %arg)
  ret double undef
}

attributes #0 = { "target-features"="+avx2" }
