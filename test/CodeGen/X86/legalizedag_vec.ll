; RUN: llvm-as < %s | llc -march=x86 -mattr=sse2 -disable-mmx -o %t -f
; RUN: grep divdi3  %t | count 2


; Test case for r63760 where we generate a legalization assert that an illegal
; type has been inserted by LegalizeDAG after LegalizeType has run. With sse2,
; v2i64 is a legal type but with mmx disabled, i64 is an illegal type. When
; legalizing the divide in LegalizeDAG, we scalarize the vector divide and make
; two 64 bit divide library calls which introduces i64 nodes that needs to be
; promoted.

define <2 x i64> @test_long_div(<2 x i64> %num, <2 x i64> %div) {
  %div.r = sdiv <2 x i64> %num, %div
  ret <2 x i64>  %div.r
}                                     