//===- DbgInfoPrinter.cpp - Print debug info in a human readable form -----==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass that prints instructions, and associated debug
// info:
//   - source/line/col information
//   - original variable name
//   - original type name
//
//===----------------------------------------------------------------------===//
#include "llvm/Pass.h"
#include "llvm/Function.h"
#include "llvm/Module.h"
#include "llvm/Value.h"
#include "llvm/IntrinsicInst.h"
#include "llvm/Assembly/Writer.h"
#include "llvm/Analysis/DebugInfo.h"
#include "llvm/Analysis/Passes.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/Support/CFG.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

static cl::opt<bool>
PrintDirectory("print-fullpath", cl::desc("Print fullpath when printing debug info"), cl::Hidden);

namespace {
	struct VISIBILITY_HIDDEN PrintDbgInfo : public FunctionPass {
    private:
      raw_ostream &Out;
      void printStopPoint(const DbgStopPointInst *DSI);
      void printFuncStart(const DbgFuncStartInst *FS);
      void printVariableDeclaration(const Value *V);
		public:
			static char ID; // Pass identification
			PrintDbgInfo() : FunctionPass(&ID), Out(outs()) {}

			virtual bool runOnFunction(Function &F);
			virtual void getAnalysisUsage(AnalysisUsage &AU) const {
				AU.setPreservesAll();
			}

	};
	char PrintDbgInfo::ID = 0;
	static RegisterPass<PrintDbgInfo> X("print-dbginfo",
      "Print debug info in human readable form");
}

FunctionPass *llvm::createDbgInfoPrinterPass() { return new PrintDbgInfo(); }

void PrintDbgInfo::printVariableDeclaration(const Value *V)
{
  std::string DisplayName, File, Directory, Type;
  unsigned LineNo;
  if (getLocationInfo(V, DisplayName, Type, LineNo, File, Directory)) {
    Out << "; ";
    WriteAsOperand(Out, V, false, 0);
    Out << " is variable " << DisplayName
      << " of type " << Type << " declared at ";
    if (PrintDirectory) {
      Out << Directory << "/";
    }
    Out << File << ":" << LineNo << "\n";
  }
}

void PrintDbgInfo::printStopPoint(const DbgStopPointInst *DSI)
{
  if (PrintDirectory) {
    std::string dir;
		GetConstantStringInfo(DSI->getDirectory(), dir);
    Out << dir << "/";
  }
  std::string file;
  GetConstantStringInfo(DSI->getFileName(), file);
  Out << file << ":" << DSI->getLine();
  if (unsigned Col = DSI->getColumn()) {
    Out << ":" << Col;
  }
}

void PrintDbgInfo::printFuncStart(const DbgFuncStartInst *FS)
{
  DISubprogram Subprogram(cast<GlobalVariable>(FS->getSubprogram()));
  std::string Res1, Res2;
  Out << ";fully qualified function name: " << Subprogram.getDisplayName(Res1)
    << " return type: " << Subprogram.getType().getName(Res2)
    << " at line " << Subprogram.getLineNumber()
    << "\n\n";
}

bool PrintDbgInfo::runOnFunction(Function &F)
{
  if (F.isDeclaration())
    return false;
  Out << "function " << F.getName() << "\n\n";
  for (Function::iterator I = F.begin(), E = F.end(); I != E; ++I) {
    BasicBlock *BB = I;
    if (I != F.begin() && (pred_begin(BB) == pred_end(BB)))
      // Skip dead blocks.
      continue;
    const DbgStopPointInst *DSI = findBBStopPoint(BB);
    Out << BB->getName();
    Out << ":";
    if (DSI) {
      Out << "; (";
      printStopPoint(DSI);
      Out << ")";
    }
    Out << "\n";
    // A dbgstoppoint's information is valid until we encounter a new one.
    const DbgStopPointInst *LastDSP = DSI;
    bool printed = DSI != 0;
    for (BasicBlock::const_iterator i = BB->begin(), e = BB->end(); i != e; ++i) {
      if (isa<DbgInfoIntrinsic>(i)) {
        if ((DSI = dyn_cast<DbgStopPointInst>(i))) {

          if (DSI->getContext() == LastDSP->getContext() &&
              DSI->getLineValue() == LastDSP->getLineValue() &&
              DSI->getColumnValue() == LastDSP->getColumnValue()) {
            // Don't print same location twice.
            continue;
          }
          LastDSP = cast<DbgStopPointInst>(i);
          // Don't print consecutive stoppoints, use a flag
          // to know which one we printed.
          printed = false;

        } else if (const DbgFuncStartInst *FS = dyn_cast<DbgFuncStartInst>(i)) {
          printFuncStart(FS);
        }
      } else {
        if (!printed && LastDSP) {
          Out << "; ";
          printStopPoint(LastDSP);
          Out << "\n";
          printed = true;
        }
        Out << *i;
        printVariableDeclaration(i);
        if (const User *U = dyn_cast<User>(i)) {
          for(unsigned i=0;i<U->getNumOperands();i++) {
            printVariableDeclaration(U->getOperand(i));
          }
        }
      }
    }
  }
  return false;
}
