// RUN: %llvmgcc -S -g --emit-llvm %s -o - | grep "\~A"
// XFAIL: darwin
class A {
  int i;
public:
  A() { i = 0; }
 ~A() { i = 42; }
};

A a;

