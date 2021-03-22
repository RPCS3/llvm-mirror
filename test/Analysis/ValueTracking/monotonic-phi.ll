; NOTE: Assertions have been autogenerated by utils/update_test_checks.py
; RUN: opt -instsimplify -S < %s | FileCheck %s

define i1 @test1(i8 %p, i8* %pq, i8 %n, i8 %r) {
; CHECK-LABEL: @test1(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br label [[LOOP:%.*]]
; CHECK:       loop:
; CHECK-NEXT:    [[A:%.*]] = phi i8 [ 1, [[ENTRY:%.*]] ], [ [[NEXT:%.*]], [[LOOP]] ]
; CHECK-NEXT:    [[NEXT]] = add nsw i8 [[A]], 1
; CHECK-NEXT:    [[CMP1:%.*]] = icmp eq i8 [[A]], [[N:%.*]]
; CHECK-NEXT:    br i1 [[CMP1]], label [[EXIT:%.*]], label [[LOOP]]
; CHECK:       exit:
; CHECK-NEXT:    ret i1 false
;
entry:
  br label %loop
loop:
  %A = phi i8 [ 1, %entry ], [ %next, %loop ]
  %next = add nsw i8 %A, 1
  %cmp1 = icmp eq i8 %A, %n
  br i1 %cmp1, label %exit, label %loop
exit:
  %add = or i8 %A, %r
  %cmp = icmp eq i8 %add, 0
  ret i1 %cmp
}

define i1 @test2(i8 %p, i8* %pq, i8 %n, i8 %r) {
; CHECK-LABEL: @test2(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br label [[LOOP:%.*]]
; CHECK:       loop:
; CHECK-NEXT:    [[A:%.*]] = phi i8 [ 1, [[ENTRY:%.*]] ], [ [[NEXT:%.*]], [[LOOP]] ]
; CHECK-NEXT:    [[NEXT]] = add i8 [[A]], 1
; CHECK-NEXT:    [[CMP1:%.*]] = icmp eq i8 [[A]], [[N:%.*]]
; CHECK-NEXT:    br i1 [[CMP1]], label [[EXIT:%.*]], label [[LOOP]]
; CHECK:       exit:
; CHECK-NEXT:    [[ADD:%.*]] = or i8 [[A]], [[R:%.*]]
; CHECK-NEXT:    [[CMP:%.*]] = icmp eq i8 [[ADD]], 0
; CHECK-NEXT:    ret i1 [[CMP]]
;
entry:
  br label %loop
loop:
  %A = phi i8 [ 1, %entry ], [ %next, %loop ]
  %next = add i8 %A, 1
  %cmp1 = icmp eq i8 %A, %n
  br i1 %cmp1, label %exit, label %loop
exit:
  %add = or i8 %A, %r
  %cmp = icmp eq i8 %add, 0
  ret i1 %cmp
}

define i1 @test3(i8 %p, i8* %pq, i8 %n, i8 %r) {
; CHECK-LABEL: @test3(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br label [[LOOP:%.*]]
; CHECK:       loop:
; CHECK-NEXT:    [[A:%.*]] = phi i8 [ 1, [[ENTRY:%.*]] ], [ [[NEXT:%.*]], [[LOOP]] ]
; CHECK-NEXT:    [[NEXT]] = add nuw i8 [[A]], 1
; CHECK-NEXT:    [[CMP1:%.*]] = icmp eq i8 [[A]], [[N:%.*]]
; CHECK-NEXT:    br i1 [[CMP1]], label [[EXIT:%.*]], label [[LOOP]]
; CHECK:       exit:
; CHECK-NEXT:    ret i1 false
;
entry:
  br label %loop
loop:
  %A = phi i8 [ 1, %entry ], [ %next, %loop ]
  %next = add nuw i8 %A, 1
  %cmp1 = icmp eq i8 %A, %n
  br i1 %cmp1, label %exit, label %loop
exit:
  %add = or i8 %A, %r
  %cmp = icmp eq i8 %add, 0
  ret i1 %cmp
}

define i1 @test4(i8 %p, i8* %pq, i8 %n, i8 %r) {
; CHECK-LABEL: @test4(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br label [[LOOP:%.*]]
; CHECK:       loop:
; CHECK-NEXT:    [[A:%.*]] = phi i8 [ 0, [[ENTRY:%.*]] ], [ [[NEXT:%.*]], [[LOOP]] ]
; CHECK-NEXT:    [[NEXT]] = add nuw i8 [[A]], 1
; CHECK-NEXT:    [[CMP1:%.*]] = icmp eq i8 [[A]], [[N:%.*]]
; CHECK-NEXT:    br i1 [[CMP1]], label [[EXIT:%.*]], label [[LOOP]]
; CHECK:       exit:
; CHECK-NEXT:    [[ADD:%.*]] = or i8 [[A]], [[R:%.*]]
; CHECK-NEXT:    [[CMP:%.*]] = icmp eq i8 [[ADD]], 0
; CHECK-NEXT:    ret i1 [[CMP]]
;
entry:
  br label %loop
loop:
  %A = phi i8 [ 0, %entry ], [ %next, %loop ]
  %next = add nuw i8 %A, 1
  %cmp1 = icmp eq i8 %A, %n
  br i1 %cmp1, label %exit, label %loop
exit:
  %add = or i8 %A, %r
  %cmp = icmp eq i8 %add, 0
  ret i1 %cmp
}

define i1 @test5(i8 %p, i8* %pq, i8 %n, i8 %r) {
; CHECK-LABEL: @test5(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br label [[LOOP:%.*]]
; CHECK:       loop:
; CHECK-NEXT:    [[A:%.*]] = phi i8 [ -2, [[ENTRY:%.*]] ], [ [[NEXT:%.*]], [[LOOP]] ]
; CHECK-NEXT:    [[NEXT]] = add nuw i8 [[A]], 1
; CHECK-NEXT:    [[CMP1:%.*]] = icmp eq i8 [[A]], [[N:%.*]]
; CHECK-NEXT:    br i1 [[CMP1]], label [[EXIT:%.*]], label [[LOOP]]
; CHECK:       exit:
; CHECK-NEXT:    [[ADD:%.*]] = or i8 [[A]], [[R:%.*]]
; CHECK-NEXT:    [[CMP:%.*]] = icmp eq i8 [[ADD]], 0
; CHECK-NEXT:    ret i1 [[CMP]]
;
entry:
  br label %loop
loop:
  %A = phi i8 [ -2, %entry ], [ %next, %loop ]
  %next = add nuw i8 %A, 1
  %cmp1 = icmp eq i8 %A, %n
  br i1 %cmp1, label %exit, label %loop
exit:
  %add = or i8 %A, %r
  %cmp = icmp eq i8 %add, 0
  ret i1 %cmp
}

define i1 @test6(i8 %p, i8* %pq, i8 %n, i8 %r) {
; CHECK-LABEL: @test6(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br label [[LOOP:%.*]]
; CHECK:       loop:
; CHECK-NEXT:    [[A:%.*]] = phi i8 [ 2, [[ENTRY:%.*]] ], [ [[NEXT:%.*]], [[LOOP]] ]
; CHECK-NEXT:    [[NEXT]] = mul nsw i8 [[A]], 2
; CHECK-NEXT:    [[CMP1:%.*]] = icmp eq i8 [[A]], [[N:%.*]]
; CHECK-NEXT:    br i1 [[CMP1]], label [[EXIT:%.*]], label [[LOOP]]
; CHECK:       exit:
; CHECK-NEXT:    ret i1 false
;
entry:
  br label %loop
loop:
  %A = phi i8 [ 2, %entry ], [ %next, %loop ]
  %next = mul nsw i8 %A, 2
  %cmp1 = icmp eq i8 %A, %n
  br i1 %cmp1, label %exit, label %loop
exit:
  %cmp = icmp eq i8 %A, 0
  ret i1 %cmp
}

define i1 @test7(i8 %p, i8* %pq, i8 %n, i8 %r) {
; CHECK-LABEL: @test7(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br label [[LOOP:%.*]]
; CHECK:       loop:
; CHECK-NEXT:    [[A:%.*]] = phi i8 [ 2, [[ENTRY:%.*]] ], [ [[NEXT:%.*]], [[LOOP]] ]
; CHECK-NEXT:    [[NEXT]] = mul i8 [[A]], 2
; CHECK-NEXT:    [[CMP1:%.*]] = icmp eq i8 [[A]], [[N:%.*]]
; CHECK-NEXT:    br i1 [[CMP1]], label [[EXIT:%.*]], label [[LOOP]]
; CHECK:       exit:
; CHECK-NEXT:    [[CMP:%.*]] = icmp eq i8 [[A]], 0
; CHECK-NEXT:    ret i1 [[CMP]]
;
entry:
  br label %loop
loop:
  %A = phi i8 [ 2, %entry ], [ %next, %loop ]
  %next = mul i8 %A, 2
  %cmp1 = icmp eq i8 %A, %n
  br i1 %cmp1, label %exit, label %loop
exit:
  %cmp = icmp eq i8 %A, 0
  ret i1 %cmp
}

define i1 @test8(i8 %p, i8* %pq, i8 %n, i8 %r) {
; CHECK-LABEL: @test8(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br label [[LOOP:%.*]]
; CHECK:       loop:
; CHECK-NEXT:    [[A:%.*]] = phi i8 [ 2, [[ENTRY:%.*]] ], [ [[NEXT:%.*]], [[LOOP]] ]
; CHECK-NEXT:    [[NEXT]] = mul nuw i8 [[A]], 2
; CHECK-NEXT:    [[CMP1:%.*]] = icmp eq i8 [[A]], [[N:%.*]]
; CHECK-NEXT:    br i1 [[CMP1]], label [[EXIT:%.*]], label [[LOOP]]
; CHECK:       exit:
; CHECK-NEXT:    ret i1 false
;
entry:
  br label %loop
loop:
  %A = phi i8 [ 2, %entry ], [ %next, %loop ]
  %next = mul nuw i8 %A, 2
  %cmp1 = icmp eq i8 %A, %n
  br i1 %cmp1, label %exit, label %loop
exit:
  %cmp = icmp eq i8 %A, 0
  ret i1 %cmp
}

define i1 @test9(i8 %p, i8* %pq, i8 %n, i8 %r) {
; CHECK-LABEL: @test9(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br label [[LOOP:%.*]]
; CHECK:       loop:
; CHECK-NEXT:    [[A:%.*]] = phi i8 [ 0, [[ENTRY:%.*]] ], [ [[NEXT:%.*]], [[LOOP]] ]
; CHECK-NEXT:    [[NEXT]] = mul nuw i8 [[A]], 2
; CHECK-NEXT:    [[CMP1:%.*]] = icmp eq i8 [[A]], [[N:%.*]]
; CHECK-NEXT:    br i1 [[CMP1]], label [[EXIT:%.*]], label [[LOOP]]
; CHECK:       exit:
; CHECK-NEXT:    [[CMP:%.*]] = icmp eq i8 [[A]], 0
; CHECK-NEXT:    ret i1 [[CMP]]
;
entry:
  br label %loop
loop:
  %A = phi i8 [ 0, %entry ], [ %next, %loop ]
  %next = mul nuw i8 %A, 2
  %cmp1 = icmp eq i8 %A, %n
  br i1 %cmp1, label %exit, label %loop
exit:
  %cmp = icmp eq i8 %A, 0
  ret i1 %cmp
}

define i1 @test10(i8 %p, i8* %pq, i8 %n, i8 %r) {
; CHECK-LABEL: @test10(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br label [[LOOP:%.*]]
; CHECK:       loop:
; CHECK-NEXT:    [[A:%.*]] = phi i8 [ 2, [[ENTRY:%.*]] ], [ [[NEXT:%.*]], [[LOOP]] ]
; CHECK-NEXT:    [[NEXT]] = mul nuw i8 [[A]], -2
; CHECK-NEXT:    [[CMP1:%.*]] = icmp eq i8 [[A]], [[N:%.*]]
; CHECK-NEXT:    br i1 [[CMP1]], label [[EXIT:%.*]], label [[LOOP]]
; CHECK:       exit:
; CHECK-NEXT:    [[CMP:%.*]] = icmp eq i8 [[A]], 0
; CHECK-NEXT:    ret i1 [[CMP]]
;
entry:
  br label %loop
loop:
  %A = phi i8 [ 2, %entry ], [ %next, %loop ]
  %next = mul nuw i8 %A, -2
  %cmp1 = icmp eq i8 %A, %n
  br i1 %cmp1, label %exit, label %loop
exit:
  %cmp = icmp eq i8 %A, 0
  ret i1 %cmp
}

define i1 @test11(i8 %p, i8* %pq, i8 %n, i8 %r) {
; CHECK-LABEL: @test11(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br label [[LOOP:%.*]]
; CHECK:       loop:
; CHECK-NEXT:    [[A:%.*]] = phi i8 [ -2, [[ENTRY:%.*]] ], [ [[NEXT:%.*]], [[LOOP]] ]
; CHECK-NEXT:    [[NEXT]] = mul nuw i8 [[A]], 2
; CHECK-NEXT:    [[CMP1:%.*]] = icmp eq i8 [[A]], [[N:%.*]]
; CHECK-NEXT:    br i1 [[CMP1]], label [[EXIT:%.*]], label [[LOOP]]
; CHECK:       exit:
; CHECK-NEXT:    ret i1 false
;
entry:
  br label %loop
loop:
  %A = phi i8 [ -2, %entry ], [ %next, %loop ]
  %next = mul nuw i8 %A, 2
  %cmp1 = icmp eq i8 %A, %n
  br i1 %cmp1, label %exit, label %loop
exit:
  %cmp = icmp eq i8 %A, 0
  ret i1 %cmp
}
