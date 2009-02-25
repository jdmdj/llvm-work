//===-- PrologEpilogInserter.cpp - Insert Prolog/Epilog code in function --===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass is responsible for finalizing the functions frame layout, saving
// callee saved registers, and for emitting prolog & epilog code for the
// function.
//
// This pass must be run after register allocation.  After this pass is
// executed, it is illegal to construct MO_FrameIndex operands.
//
// This pass implements a shrink wrapping variant of prolog/epilog insertion.
//
// This implementations of shrink wrapping is done in two parts:
//   1. Callee-saved register spills are placed in only the basic blocks
//      where such registers are actually written by the function.
//   2. Prologues and epilogues are placed only in the basic blocks that
//      contain function calls.
//
// Shrink wrapping uses an analysis similar to the one in GVNPRE to determine
// which basic blocks require callee-saved register save/restore code.
//
// This pass uses MachineDominators and MachineLoopInfo. Loop information
// is used to prevent shrink wrapping of callee-saved register save/restore
// code into loops.
//
// Part 1. of the implementation is straightforward: the code to save/restore
// code insertion done in PEI is usable with only a single modification: a
// map of basic blocks and used callee-saved registers is used to place the
// save/restore code instead of simply inserting it into the entry block and
// all exiting blocks.
//
// Part 2. is only complicated by the fact that the prologue/epilogue emitters
// provided by TargetRegisterInfo insert code perforce into the entry and exit
// blocks. To permit shrink wrapping of prologue/epilogue code, methods must be
// provided that take a list of basic blocks for code insertion. Since this is
// a more extensive code change, part 2. will be done later.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "shrink-wrap"

#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/RegisterScavenging.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/Target/TargetFrameInfo.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/ADT/SparseBitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
#include "llvm/ADT/STLExtras.h"
#include <climits>
#include <sstream>

using namespace llvm;

// Shrink Wrapping:
static cl::opt<bool>
ShrinkWrapping("shrink-wrap",
               cl::desc("Apply shrink wrapping to callee-saved register spills/restores"));

namespace {
  struct VISIBILITY_HIDDEN PEI : public MachineFunctionPass {
    static char ID;
    PEI() : MachineFunctionPass(&ID) {}

    const char *getPassName() const {
      return "Prolog/Epilog Insertion & Frame Finalization";
    }

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesCFG();
      AU.addRequired<MachineLoopInfo>();
      AU.addRequired<MachineDominatorTree>();
      AU.addPreserved<MachineLoopInfo>();
      AU.addPreserved<MachineDominatorTree>();
      MachineFunctionPass::getAnalysisUsage(AU);
    }

    virtual void releaseMemory() {
      /// TODO -- release memory used by analysis.
    }

    /// runOnMachineFunction - Insert prolog/epilog code and replace abstract
    /// frame indexes with appropriate references.
    ///
    bool runOnMachineFunction(MachineFunction &Fn) {
      const TargetRegisterInfo *TRI = Fn.getTarget().getRegisterInfo();
      RS = TRI->requiresRegisterScavenging(Fn) ? new RegScavenger() : NULL;

      // Get MachineModuleInfo so that we can track the construction of the
      // frame.
      if (MachineModuleInfo *MMI = getAnalysisIfAvailable<MachineModuleInfo>())
        Fn.getFrameInfo()->setMachineModuleInfo(MMI);

      // Allow the target machine to make some adjustments to the function
      // e.g. UsedPhysRegs before calculateCalleeSavedRegisters.
      TRI->processFunctionBeforeCalleeSavedScan(Fn, RS);

      // Scan the function for modified callee saved registers and insert spill
      // code for any callee saved registers that are modified.  Also calculate
      // the MaxCallFrameSize and HasCalls variables for the function's frame
      // information and eliminates call frame pseudo instructions.
      calculateCalleeSavedRegisters(Fn);

      // ShrinkWrap: Determine placement of CSR spill/restore code.
      if (ShrinkWrapping)
        placeCSRSpillsAndRestores(Fn);

      // Add the code to save and restore the callee saved registers
      saveCalleeSavedRegisters(Fn);

      // Allow the target machine to make final modifications to the function
      // before the frame layout is finalized.
      TRI->processFunctionBeforeFrameFinalized(Fn);

      // Calculate actual frame offsets for all of the abstract stack objects...
      calculateFrameObjectOffsets(Fn);

      // Add prolog and epilog code to the function.  This function is required
      // to align the stack frame as necessary for any stack variables or
      // called functions.  Because of this, calculateCalleeSavedRegisters
      // must be called before this function in order to set the HasCalls
      // and MaxCallFrameSize variables.
      insertPrologEpilogCode(Fn);

      // Replace all MO_FrameIndex operands with physical register references
      // and actual offsets.
      //
      replaceFrameIndices(Fn);

      delete RS;
      return true;
    }

  private:
    RegScavenger *RS;

    // MinCSFrameIndex, MaxCSFrameIndex - Keeps the range of callee saved
    // stack frame indexes.
    unsigned MinCSFrameIndex, MaxCSFrameIndex;

    // Analysis info for placing callee saved register saves/restores.

    // CSRegSet contains indices into the Callee Saved Register Info
    // vector built by calculateCalleeSavedRegisters() and accessed
    // via Fn.getFrameInfo()->getCalleeSavedInfo().
    typedef SparseBitVector<> CSRegSet;

    // CSRegBlockMap maps MachineBasicBlocks to sets of callee
    // saved register indices.
    typedef DenseMap<MachineBasicBlock*, CSRegSet> CSRegBlockMap;

    // Set and maps for callee saved registers:
    //  used in function (UsedCSRegs)
    //  used in a basic block (CSRUsed)
    //  to be spilled at the entry to a basic block (CSRSave)
    //  to be restored at the end of a basic block (CSRRestore)
    CSRegSet UsedCSRegs;
    CSRegBlockMap CSRUsed;
    CSRegBlockMap CSRSave;
    CSRegBlockMap CSRRestore;

    void placeCSRSpillsAndRestores(MachineFunction &Fn);

    void calculateCalleeSavedRegisters(MachineFunction &Fn);
    void saveCalleeSavedRegisters(MachineFunction &Fn);
    void calculateFrameObjectOffsets(MachineFunction &Fn);
    void replaceFrameIndices(MachineFunction &Fn);
    void insertPrologEpilogCode(MachineFunction &Fn);

    // Convienences used by placeCSRSpillsAndRestores().

    void clearCSRegSet(CSRegSet& s) {
      CSRegSet emptySet;
      s &= emptySet;
    }

    void clearSets() {
      clearCSRegSet(UsedCSRegs);
      CSRUsed.clear();
      CSRSave.clear();
      CSRRestore.clear();
    }

    // Because SparseBitVector has no _infix_ operator|&-()...
    CSRegSet CSRegSetUnion(const CSRegSet& s1, const CSRegSet& s2) {
      CSRegSet result(s1);
      result |= s2;
      return result;
    }

    CSRegSet CSRegSetIntersection(const CSRegSet& s1, const CSRegSet& s2) {
      CSRegSet result(s1);
      result &= s2;
      return result;
    }

    CSRegSet CSRegSetDifference(const CSRegSet& s1, const CSRegSet& s2) {
      CSRegSet result;
      result.intersectWithComplement(s1, s2);
      return result;
    }

    MachineBasicBlock* getTopLevelLoopPreheader(MachineLoop* LP) {
      assert(LP && "Machine loop is NULL.");
      MachineBasicBlock* PHDR = LP->getLoopPreheader();
      MachineLoop* PLP = LP->getParentLoop();
      while (PLP) {
        PHDR = PLP->getLoopPreheader();
        PLP = PLP->getParentLoop();
      }
      return PHDR;
    }

    MachineLoop* getTopLevelLoopParent(MachineLoop *LP) {
      if (LP == 0)
        return 0;
      MachineLoop* PLP = LP->getParentLoop();
      while (PLP) {
        LP = PLP;
        PLP = PLP->getParentLoop();
      }
      return LP;
    }

    // Debugging methods.
    std::string stringifyCSRegSet(const CSRegSet& s, MachineFunction &Fn) {
      const TargetRegisterInfo* TRI = Fn.getTarget().getRegisterInfo();
      const std::vector<CalleeSavedInfo> CSI =
        Fn.getFrameInfo()->getCalleeSavedInfo();

      unsigned reg;
      std::ostringstream srep;
      if (CSI.size() == 0) {
        srep << "[]";
        return srep.str();
      }
      srep << "[";
      CSRegSet::iterator I = s.begin(), E = s.end();
      if (I != E) {
        reg = CSI[*I].getReg();
        srep << TRI->getName(reg);
        for (++I; I != E; ++I) {
          reg = CSI[*I].getReg();
          srep << ",";
          srep << TRI->getName(reg);
        }
      }
      srep << "]";
      return srep.str();
    }

  };
  char PEI::ID = 0;
}


/// createPrologEpilogCodeInserter - This function returns a pass that inserts
/// prolog and epilog code, and eliminates abstract frame references.
///
FunctionPass *llvm::createPrologEpilogCodeInserter() { return new PEI(); }


/// placeCSRSpillsAndRestores - determine which MBBs of the function
/// need save, restore code for callee-saved registers by doing a DF analysis
/// similar to the one used in code motion (GVNPRE). This produces maps of MBBs
/// to sets of registers (CSRs) for saves and restores. MachineLoopInfo
/// is used to ensure that CSR save/restore code is not placed inside loops.
///
void PEI::placeCSRSpillsAndRestores(MachineFunction &Fn) {

  // Sets used to compute spill, restore placement sets.
  CSRegBlockMap AnticIn, AnticOut;
  CSRegBlockMap AvailIn, AvailOut;
  const std::vector<CalleeSavedInfo> CSI =
    Fn.getFrameInfo()->getCalleeSavedInfo();

  DOUT << "Place CSR spills/restores for "
       << Fn.getFunction()->getName() << "\n";

  // Clear Save, Restore sets.
  clearSets();

  // If no CSRs used, we are done.
  if (CSI.empty()) {
    DOUT << Fn.getFunction()->getName()
         << " uses no callee-saved registers.\n";
    return;
  }

  MachineDominatorTree &DT = getAnalysis<MachineDominatorTree>();
  MachineLoopInfo &LI = getAnalysis<MachineLoopInfo>();

  const TargetRegisterInfo *TRI = Fn.getTarget().getRegisterInfo();
  bool allCSRegUsesInEntryBlock = true;

  // DEBUG -- dump loop info
  DOUT << "Top level loops of " << Fn.getFunction()->getName() << ":\n";
  for (MachineLoopInfo::iterator
         I = LI.begin(), E = LI.end(); I != E; ++I) {
    MachineLoop* LP = *I;
    MachineBasicBlock* HDR = LP->getHeader();
    MachineBasicBlock* PHDR = LP->getLoopPreheader();
    SmallVector<MachineBasicBlock*, 4> loopExits;

    LP->getExitingBlocks(loopExits);

    DOUT << " Loop with header "
         << HDR->getBasicBlock()->getName();
    if (PHDR) {
      DOUT << " and preheader " << PHDR->getBasicBlock()->getName();
    }
    DOUT << " has exiting blocks:";
    for (unsigned i = 0; i < loopExits.size(); ++i) {
      DOUT << " " << loopExits[i]->getBasicBlock()->getName();
    }
    loopExits.clear();
    LP->getExitBlocks(loopExits);
    DOUT << " and exit blocks:";
    for (unsigned i = 0; i < loopExits.size(); ++i) {
      DOUT << " " << loopExits[i]->getBasicBlock()->getName();
    }
    DOUT << "\n";
  }
  // DEBUG -- dump loop info

  // 0. Initialize UsedCSRegs set, CSRUsed map.
  //    At the same time, put entry block directly into
  //    CSRSave, CSRRestore sets if any CSRs are used.
  //
  //    Quick exit option:
  //    If there is at least one use in entry block,
  //    revert to default behavior, skip the placement
  //    step and put all saves in entry, restores in
  //    return blocks.
  SmallVector<MachineBasicBlock*, 4> returnBlocks;
  MachineBasicBlock* entryBlock;
  for (MachineFunction::iterator MBB = Fn.begin(), E = Fn.end();
       MBB != E; ++MBB)
    // Save entry block, return blocks.
    if (MBB->pred_size() == 0)
      entryBlock = MBB;
    else
      if (!MBB->empty() && MBB->back().getDesc().isReturn())
        returnBlocks.push_back(MBB);
  assert(entryBlock && "Function has no entry block?");
  for (MachineFunction::iterator MBB = Fn.begin(), E = Fn.end();
       MBB != E; ++MBB) {
    for (MachineBasicBlock::iterator I = MBB->begin(); I != MBB->end(); ++I) {
      for (unsigned i = 0, e = CSI.size(); i != e; ++i) {
        unsigned Reg = CSI[i].getReg();
        // If instruction I reads or modifies Reg, add it to UsedCSRegs,
        // CSRUsed map for the current block.
        if (I->readsRegister(Reg, TRI) || I->modifiesRegister(Reg, TRI)) {
          UsedCSRegs.set(i);
          CSRUsed[MBB].set(i);
          // Short-circuit analysis for entry, return blocks:
          // if a CSR is used in the entry block, add it directly
          // to CSRSave[entryBlock] and to returnBlocks.
          // Check for a CSR use in a non-entry block.
          if (MBB->getNumber() == entryBlock->getNumber()) {
            CSRSave[MBB].set(i);
            for (unsigned ii = 0; ii < returnBlocks.size(); ++ii)
              CSRRestore[returnBlocks[ii]].set(i);
          } else
            allCSRegUsesInEntryBlock = false;
        }
      }
    }
    DOUT << "CSRUsed[" << MBB->getBasicBlock()->getName() << "] = "
         << stringifyCSRegSet(CSRUsed[MBB], Fn) << "\n";
  }

  DOUT << "UsedCSRegs = " << stringifyCSRegSet(UsedCSRegs, Fn) << "\n";

  // If all uses of CSRegs are in the entry block, nothing to do:
  // spills go in entry block, restores go in exiting blocks.
  if (allCSRegUsesInEntryBlock) {
    DOUT << "All uses of CSRegs are in entry block, nothing to do.\n";
    return;
  }

  DOUT << "Computing {ANTIC,AVAIL}_{IN,OUT} sets\n";

  // Calculate AnticIn, AnticOut using post-order traversal of MCFG.
  for (po_iterator<MachineBasicBlock*> MBBI = po_begin(Fn.getBlockNumbered(0)),
         MBBE = po_end(Fn.getBlockNumbered(0)); MBBI != MBBE; ++MBBI) {
    MachineBasicBlock* MBB = *MBBI;

    // AnticOut[MBB]: intersection of AnticIn[succ(MBB)]...
    for (MachineBasicBlock::succ_iterator SI = MBB->succ_begin(),
           SE = MBB->succ_end(); SI != SE; ++SI) {
      MachineBasicBlock* SUCC = *SI;
      AnticOut[MBB] &= AnticIn[SUCC];
    }
    // AnticIn[MBB] = CSRUsed[MBB] u AnticOut[MBB]
    AnticIn[MBB] = CSRegSetUnion(CSRUsed[MBB], AnticOut[MBB]);

    DOUT << "ANTIC_IN[" << MBB->getBasicBlock()->getName() << "] = "
         << stringifyCSRegSet(AnticIn[MBB], Fn)
         << "  ANTIC_OUT[" << MBB->getBasicBlock()->getName() << "] = "
         << stringifyCSRegSet(AnticOut[MBB], Fn) << "\n";
  }

  DOUT << "-----------------------------------------------------------\n";

  // Calculate Avail{In,Out} via top-down walk of Machine dominator tree.
  for (df_iterator<MachineDomTreeNode*> DI = df_begin(DT.getRootNode()),
         E = df_end(DT.getRootNode()); DI != E; ++DI) {

    MachineBasicBlock* MBB = DI->getBlock();

    // AvailOut[MBB]: intersection of AvailIn[pred(MBB)]...
    for (MachineBasicBlock::pred_iterator PI = MBB->pred_begin(),
           PE = MBB->pred_end(); PI != PE; ++PI) {
      MachineBasicBlock* PRED = *PI;
      AvailIn[MBB] &= AvailOut[PRED];
    }
    // AvailIn[MBB] = CSRUsed[MBB] u AvailOut[MBB]
    AvailOut[MBB] = CSRegSetUnion(CSRUsed[MBB], AvailIn[MBB]);

    DOUT << "AVAIL_IN[" << MBB->getBasicBlock()->getName() << "] = "
               << stringifyCSRegSet(AvailIn[MBB], Fn)
               << "  AVAIL_OUT[" << MBB->getBasicBlock()->getName() << "] = "
               << stringifyCSRegSet(AvailOut[MBB], Fn) << "\n";
  }

  DOUT << "-----------------------------------------------------------\n";

  // Calculate CSRSave via top-down walk of Machine dominator tree.
  for (df_iterator<MachineDomTreeNode*> DI = df_begin(DT.getRootNode()),
         E = df_end(DT.getRootNode()); DI != E; ++DI) {

    MachineBasicBlock* MBB = DI->getBlock();

    // Entry block saves are recorded in UsedCSRegs pass above.
    if (MBB->pred_size() > 0) {
      // Intersect (CSRegs - AnticIn[P]) for all predecessors P of MBB
      CSRegSet anticInPreds;
      MachineBasicBlock::pred_iterator PI = MBB->pred_begin(),
        PE = MBB->pred_end();
      if (PI != PE) {
        MachineBasicBlock* PRED = *PI;
        anticInPreds = CSRegSetDifference(UsedCSRegs, AnticOut[PRED]);
        for (++PI; PI != PE; ++PI) {
          PRED = *PI;
          anticInPreds &= CSRegSetDifference(UsedCSRegs, AnticOut[PRED]);
        }
      }
      // CSRSave[MBB] = (AnticIn[MBB] - AvailIn[MBB]) ^ anticInPreds
      CSRSave[MBB] = CSRegSetIntersection(CSRegSetDifference(AnticIn[MBB],
                                                             AvailIn[MBB]),
                                          anticInPreds);
      // Remove the CSRs that are saved in the entry block
      if (! CSRSave[MBB].empty() && ! CSRSave[entryBlock].empty())
        CSRSave[MBB] = CSRegSetDifference(CSRSave[MBB], CSRSave[entryBlock]);
      // Move saves inside loops to the preheaders of the outermost (top level)
      // containing loops.
      if (! CSRSave[MBB].empty()) {
        if (MachineLoop* LP = LI.getLoopFor(MBB)) {
          CSRegSet emptySet;
          MachineBasicBlock* LPH = getTopLevelLoopPreheader(LP);
          assert(LPH && "Loop has no top level preheader?");

          DOUT << "Moving saves of "
               << stringifyCSRegSet(CSRSave[MBB], Fn)
               << " from " << MBB->getBasicBlock()->getName()
               << " to " << LPH->getBasicBlock()->getName() << "\n";
          // Add CSRegSet from MBB to LPH, empty out MBB's CSRegSet.
          CSRSave[LPH] |= CSRSave[MBB];
          // If saves moved to entry block, add restores to returns.
          if (LPH->getNumber() == entryBlock->getNumber())
            for (unsigned i = 0; i < returnBlocks.size(); ++i)
              CSRRestore[returnBlocks[i]] |= CSRSave[MBB];
          CSRSave[MBB] &= emptySet;
        }
      }
    }

    if (! CSRSave[MBB].empty())
      DOUT << "SAVE[" << MBB->getBasicBlock()->getName() << "] = "
           << stringifyCSRegSet(CSRSave[MBB], Fn) << "\n";
  }

  DOUT << "-----------------------------------------------------------\n";

  // Calculate CSRRestore using post-order traversal of Machine-CFG.
  for (po_iterator<MachineBasicBlock*> MBBI = po_begin(Fn.getBlockNumbered(0)),
         MBBE = po_end(Fn.getBlockNumbered(0)); MBBI != MBBE; ++MBBI) {

    MachineBasicBlock* MBB = *MBBI;

    // May already be set for return blocks.
    if (CSRRestore[MBB].empty() && MBB->pred_size() > 0) {
      // Intersect (CSRegs - AvailOut[S]) for all successors S of MBB
      CSRegSet availOutPreds;
      MachineBasicBlock::succ_iterator SI = MBB->succ_begin(),
        SE = MBB->succ_end();
      if (SI != SE) {
        MachineBasicBlock* SUCC = *SI;
        availOutPreds = CSRegSetDifference(UsedCSRegs, AvailOut[SUCC]);
        for (++SI; SI != SE; ++SI) {
          SUCC = *SI;
          availOutPreds &= CSRegSetDifference(UsedCSRegs, AvailOut[SUCC]);
        }
      }
      // CSRRestore[MBB] = (AvailOut[MBB] - AnticOut[MBB]) ^ availOutPreds
      CSRRestore[MBB] = CSRegSetIntersection(CSRegSetDifference(AvailOut[MBB],
                                                                AnticOut[MBB]),
                                             availOutPreds);
      // Remove the CSRs that are restored in the return blocks.
      // Lest this be confusing, note that:
      // CSRSave[entryBlock] == CSRRestore[B] for all B in returnBlocks.
      if (! CSRRestore[MBB].empty() && ! CSRSave[entryBlock].empty())
        CSRRestore[MBB] = CSRegSetDifference(CSRRestore[MBB],
                                             CSRSave[entryBlock]);
      // Move restores inside loops to the exits of the outermost (top level)
      // containing loops.
      if (! CSRRestore[MBB].empty()) {
        if (MachineLoop* LP = LI.getLoopFor(MBB)) {
          CSRegSet emptySet;

          LP = getTopLevelLoopParent(LP);
          assert(LP && "Loop with no top level parent?");

          SmallVector<MachineBasicBlock*, 4> exitBlocks;

          LP->getExitBlocks(exitBlocks);
          assert(exitBlocks.size() > 0 &&
                 "Loop has no top level exit blocks?");
          for (unsigned i = 0; i < exitBlocks.size(); ++i) {
            MachineBasicBlock* EXB = exitBlocks[i];

            DOUT << "Moving restores of "
                 << stringifyCSRegSet(CSRRestore[MBB], Fn)
                 << " from " << MBB->getBasicBlock()->getName()
                 << " to " << EXB->getBasicBlock()->getName() << "\n";

            // Add CSRegSet from MBB to LPE, empty out MBB's CSRegSet.
            CSRRestore[EXB] |= CSRRestore[MBB];
          }
          CSRRestore[MBB] &= emptySet;
        }
      }
    }
    if (! CSRRestore[MBB].empty())
      DOUT << "RESTORE[" << MBB->getBasicBlock()->getName() << "] = "
           << stringifyCSRegSet(CSRRestore[MBB], Fn) << "\n";
  }

  // DEBUG -- dump final Save, Restore maps.
  DOUT << "-----------------------------------------------------------\n";
  DOUT << "Final SAVE, RESTORE:\n";
  DOUT << "-----------------------------------------------------------\n";
  for (MachineFunction::iterator MBB = Fn.begin(), E = Fn.end();
       MBB != E; ++MBB) {
    if (! CSRSave[MBB].empty()) {
      DOUT << "SAVE[" << MBB->getBasicBlock()->getName() << "] = "
           << stringifyCSRegSet(CSRSave[MBB], Fn);
      if (CSRRestore[MBB].empty())
        DOUT << "\n";
    }
    if (! CSRRestore[MBB].empty()) {
      if (! CSRSave[MBB].empty())
        DOUT << "    ";
      DOUT << "RESTORE[" << MBB->getBasicBlock()->getName() << "] = "
           << stringifyCSRegSet(CSRRestore[MBB], Fn) << "\n";
    }
  }
}


/// calculateCalleeSavedRegisters - Scan the function for modified callee saved
/// registers.  Also calculate the MaxCallFrameSize and HasCalls variables for
/// the function's frame information and eliminates call frame pseudo
/// instructions.
///
void PEI::calculateCalleeSavedRegisters(MachineFunction &Fn) {
  const TargetRegisterInfo *RegInfo = Fn.getTarget().getRegisterInfo();
  const TargetFrameInfo *TFI = Fn.getTarget().getFrameInfo();

  // Get the callee saved register list...
  const unsigned *CSRegs = RegInfo->getCalleeSavedRegs(&Fn);

  // Get the function call frame set-up and tear-down instruction opcode
  int FrameSetupOpcode   = RegInfo->getCallFrameSetupOpcode();
  int FrameDestroyOpcode = RegInfo->getCallFrameDestroyOpcode();

  // These are used to keep track the callee-save area. Initialize them.
  MinCSFrameIndex = INT_MAX;
  MaxCSFrameIndex = 0;

  // Early exit for targets which have no callee saved registers and no call
  // frame setup/destroy pseudo instructions.
  if ((CSRegs == 0 || CSRegs[0] == 0) &&
      FrameSetupOpcode == -1 && FrameDestroyOpcode == -1)
    return;

  unsigned MaxCallFrameSize = 0;
  bool HasCalls = false;

  std::vector<MachineBasicBlock::iterator> FrameSDOps;
  for (MachineFunction::iterator BB = Fn.begin(), E = Fn.end(); BB != E; ++BB)
    for (MachineBasicBlock::iterator I = BB->begin(); I != BB->end(); ++I)
      if (I->getOpcode() == FrameSetupOpcode ||
          I->getOpcode() == FrameDestroyOpcode) {
        assert(I->getNumOperands() >= 1 && "Call Frame Setup/Destroy Pseudo"
               " instructions should have a single immediate argument!");
        unsigned Size = I->getOperand(0).getImm();
        if (Size > MaxCallFrameSize) MaxCallFrameSize = Size;
        HasCalls = true;
        FrameSDOps.push_back(I);
      }

  MachineFrameInfo *FFI = Fn.getFrameInfo();
  FFI->setHasCalls(HasCalls);
  FFI->setMaxCallFrameSize(MaxCallFrameSize);

  for (unsigned i = 0, e = FrameSDOps.size(); i != e; ++i) {
    MachineBasicBlock::iterator I = FrameSDOps[i];
    // If call frames are not being included as part of the stack frame,
    // and there is no dynamic allocation (therefore referencing frame slots
    // off sp), leave the pseudo ops alone. We'll eliminate them later.
    if (RegInfo->hasReservedCallFrame(Fn) || RegInfo->hasFP(Fn))
      RegInfo->eliminateCallFramePseudoInstr(Fn, *I->getParent(), I);
  }

  // Now figure out which *callee saved* registers are modified by the current
  // function, thus needing to be saved and restored in the prolog/epilog.
  //
  const TargetRegisterClass* const *CSRegClasses =
    RegInfo->getCalleeSavedRegClasses(&Fn);
  std::vector<CalleeSavedInfo> CSI;
  for (unsigned i = 0; CSRegs[i]; ++i) {
    unsigned Reg = CSRegs[i];
    if (Fn.getRegInfo().isPhysRegUsed(Reg)) {
        // If the reg is modified, save it!
      CSI.push_back(CalleeSavedInfo(Reg, CSRegClasses[i]));
    } else {
      for (const unsigned *AliasSet = RegInfo->getAliasSet(Reg);
           *AliasSet; ++AliasSet) {  // Check alias registers too.
        if (Fn.getRegInfo().isPhysRegUsed(*AliasSet)) {
          CSI.push_back(CalleeSavedInfo(Reg, CSRegClasses[i]));
          break;
        }
      }
    }
  }

  if (CSI.empty())
    return;   // Early exit if no callee saved registers are modified!

  unsigned NumFixedSpillSlots;
  const std::pair<unsigned,int> *FixedSpillSlots =
    TFI->getCalleeSavedSpillSlots(NumFixedSpillSlots);

  // Now that we know which registers need to be saved and restored, allocate
  // stack slots for them.
  for (unsigned i = 0, e = CSI.size(); i != e; ++i) {
    unsigned Reg = CSI[i].getReg();
    const TargetRegisterClass *RC = CSI[i].getRegClass();

    // Check to see if this physreg must be spilled to a particular stack slot
    // on this target.
    const std::pair<unsigned,int> *FixedSlot = FixedSpillSlots;
    while (FixedSlot != FixedSpillSlots+NumFixedSpillSlots &&
           FixedSlot->first != Reg)
      ++FixedSlot;

    int FrameIdx;
    if (FixedSlot == FixedSpillSlots+NumFixedSpillSlots) {
      // Nope, just spill it anywhere convenient.
      unsigned Align = RC->getAlignment();
      unsigned StackAlign = TFI->getStackAlignment();
      // We may not be able to sastify the desired alignment specification of
      // the TargetRegisterClass if the stack alignment is smaller.
      // Use the min.
      Align = std::min(Align, StackAlign);
      FrameIdx = FFI->CreateStackObject(RC->getSize(), Align);
      if ((unsigned)FrameIdx < MinCSFrameIndex) MinCSFrameIndex = FrameIdx;
      if ((unsigned)FrameIdx > MaxCSFrameIndex) MaxCSFrameIndex = FrameIdx;
    } else {
      // Spill it to the stack where we must.
      FrameIdx = FFI->CreateFixedObject(RC->getSize(), FixedSlot->second);
    }
    CSI[i].setFrameIdx(FrameIdx);
  }

  FFI->setCalleeSavedInfo(CSI);
}

/// saveCalleeSavedRegisters -  Insert spill code for any callee saved
/// registers that are modified in the function.
///
void PEI::saveCalleeSavedRegisters(MachineFunction &Fn) {
  // Get callee saved register information.
  MachineFrameInfo *FFI = Fn.getFrameInfo();
  const std::vector<CalleeSavedInfo> &CSI = FFI->getCalleeSavedInfo();

  // Early exit if no callee saved registers are modified!
  if (CSI.empty())
    return;

  const TargetInstrInfo &TII = *Fn.getTarget().getInstrInfo();

  // Shrink Wrapping:
  //   Use CSRSave, CSRRestore maps to determine the blocks into which
  //   spill/restore code should be inserted, and the registers to be
  //   spilled/restored.
  if (ShrinkWrapping) {
    const TargetRegisterInfo *TRI = Fn.getTarget().getRegisterInfo();
    MachineBasicBlock::iterator I;
    std::vector<CalleeSavedInfo> blockCSI;

    DOUT << "Emitting spill code for CSRs in function "
         << Fn.getFunction()->getName() << "\n";

    for (CSRegBlockMap::iterator
           BI = CSRSave.begin(), BE = CSRSave.end(); BI != BE; ++BI) {
      MachineBasicBlock* MBB = BI->first;
      CSRegSet save = BI->second;

      if (! save.empty()) {
        DOUT << "CSRSave[" << MBB->getBasicBlock()->getName() << "] = "
             << stringifyCSRegSet(save, Fn) << "\n";

        blockCSI.clear();
        for (CSRegSet::iterator RI = save.begin(),
               RE = save.end(); RI != RE; ++RI) {
          unsigned reg = CSI[*RI].getReg();
          DOUT << "adding save of "
               << reg << "(" << TRI->getName(reg) << ") to blockCSI for "
               << MBB->getBasicBlock()->getName() << "\n";
          blockCSI.push_back(CSI[*RI]);
        }
        assert(blockCSI.size() > 0 &&
               "Could not find callee saved register info");
        I = MBB->begin();
        if (!TII.spillCalleeSavedRegisters(*MBB, I, blockCSI)) {
          for (unsigned i = 0, e = blockCSI.size(); i != e; ++i) {
            // Add the callee-saved register as live-in.
            // It's killed at the spill.
            MBB->addLiveIn(blockCSI[i].getReg());

            // Insert the spill to the stack frame.
            TII.storeRegToStackSlot(*MBB, I, blockCSI[i].getReg(),
                                    true,
                                    blockCSI[i].getFrameIdx(),
                                    blockCSI[i].getRegClass());
          }
        }
      }
    }
    // Use CSRRestore to add code to restore the callee-save registers in
    // each block.
    for (CSRegBlockMap::iterator
           BI = CSRRestore.begin(), BE = CSRRestore.end(); BI != BE; ++BI) {
      MachineBasicBlock* MBB = BI->first;
      CSRegSet restore = BI->second;

      if (! restore.empty()) {
        DOUT << "CSRRestore[" << MBB->getBasicBlock()->getName() << "] = "
             << stringifyCSRegSet(restore, Fn) << "\n";

        blockCSI.clear();
        for (CSRegSet::iterator RI = restore.begin(),
               RE = restore.end(); RI != RE; ++RI) {
          unsigned reg = CSI[*RI].getReg();
          DOUT << "adding restore of "
               << reg << "(" << TRI->getName(reg) << ") to blockCSI for "
               << MBB->getBasicBlock()->getName() << "\n";
          blockCSI.push_back(CSI[*RI]);
        }
        assert(blockCSI.size() > 0 &&
               "Could not find callee saved register info");

        I = MBB->end();
        --I;

        // EXP append restore to block unless it ends in a
        // barrier terminator instruction.

        // Skip over all terminator instructions, which are part of the
        // return sequence.
        if (I->getDesc().isCall()) {
          ++I;
        } else {
          MachineBasicBlock::iterator I2 = I;
          while (I2 != MBB->begin() && (--I2)->getDesc().isTerminator())
            I = I2;
        }

        bool AtStart = I == MBB->begin();
        MachineBasicBlock::iterator BeforeI = I;
        if (!AtStart)
          --BeforeI;

        // DEBUG
        MachineInstr* MI = BeforeI;
        DOUT << "adding restore after ";
        MI->dump();
        // DEBUG

        // Restore all registers immediately before the return and any
        // terminators that preceed it.
        if (!TII.restoreCalleeSavedRegisters(*MBB, I, blockCSI)) {
          for (unsigned i = 0, e = blockCSI.size(); i != e; ++i) {
            TII.loadRegFromStackSlot(*MBB, I, blockCSI[i].getReg(),
                                     blockCSI[i].getFrameIdx(),
                                     blockCSI[i].getRegClass());
            assert(I != MBB->begin() &&
                   "loadRegFromStackSlot didn't insert any code!");
            // Insert in reverse order.  loadRegFromStackSlot can insert
            // multiple instructions.
            if (AtStart)
              I = MBB->begin();
            else {
              I = BeforeI;
              ++I;
            }
          }
        }
      }
    }
  } else {

    // Now that we have a stack slot for each register to be saved,
    // insert spill code into the entry block.
    MachineBasicBlock *MBB = Fn.begin();
    MachineBasicBlock::iterator I = MBB->begin();

    if (!TII.spillCalleeSavedRegisters(*MBB, I, CSI)) {
      for (unsigned i = 0, e = CSI.size(); i != e; ++i) {
        // Add the callee-saved register as live-in. It's killed at the spill.
        MBB->addLiveIn(CSI[i].getReg());

        // Insert the spill to the stack frame.
        TII.storeRegToStackSlot(*MBB, I, CSI[i].getReg(), true,
                                CSI[i].getFrameIdx(), CSI[i].getRegClass());
      }
    }

    // Add code to restore the callee-save registers in each exiting block.
    for (MachineFunction::iterator FI = Fn.begin(), E = Fn.end();
         FI != E; ++FI)
      // If last instruction is a return instruction, add an epilogue.
      if (!FI->empty() && FI->back().getDesc().isReturn()) {
        MBB = FI;
        I = MBB->end(); --I;

        // Skip over all terminator instructions, which are part of the return
        // sequence.
        MachineBasicBlock::iterator I2 = I;
        while (I2 != MBB->begin() && (--I2)->getDesc().isTerminator())
          I = I2;

        bool AtStart = I == MBB->begin();
        MachineBasicBlock::iterator BeforeI = I;
        if (!AtStart)
          --BeforeI;

        // Restore all registers immediately before the return and any
        // terminators that preceed it.
        if (!TII.restoreCalleeSavedRegisters(*MBB, I, CSI)) {
          for (unsigned i = 0, e = CSI.size(); i != e; ++i) {
            TII.loadRegFromStackSlot(*MBB, I, CSI[i].getReg(),
                                     CSI[i].getFrameIdx(),
                                     CSI[i].getRegClass());
            assert(I != MBB->begin() &&
                   "loadRegFromStackSlot didn't insert any code!");
            // Insert in reverse order.  loadRegFromStackSlot can insert
            // multiple instructions.
            if (AtStart)
              I = MBB->begin();
            else {
              I = BeforeI;
              ++I;
            }
          }
        }
      }
  }
}

/// AdjustStackOffset - Helper function used to adjust the stack frame offset.
static inline void
AdjustStackOffset(MachineFrameInfo *FFI, int FrameIdx,
                  bool StackGrowsDown, int64_t &Offset,
                  unsigned &MaxAlign) {
  // If stack grows down, we need to add size of find the lowest address of the
  // object.
  if (StackGrowsDown)
    Offset += FFI->getObjectSize(FrameIdx);

  unsigned Align = FFI->getObjectAlignment(FrameIdx);

  // If the alignment of this object is greater than that of the stack, then
  // increase the stack alignment to match.
  MaxAlign = std::max(MaxAlign, Align);

  // Adjust to alignment boundary.
  Offset = (Offset + Align - 1) / Align * Align;

  if (StackGrowsDown) {
    FFI->setObjectOffset(FrameIdx, -Offset); // Set the computed offset
  } else {
    FFI->setObjectOffset(FrameIdx, Offset);
    Offset += FFI->getObjectSize(FrameIdx);
  }
}

/// calculateFrameObjectOffsets - Calculate actual frame offsets for all of the
/// abstract stack objects.
///
void PEI::calculateFrameObjectOffsets(MachineFunction &Fn) {
  const TargetFrameInfo &TFI = *Fn.getTarget().getFrameInfo();

  bool StackGrowsDown =
    TFI.getStackGrowthDirection() == TargetFrameInfo::StackGrowsDown;

  // Loop over all of the stack objects, assigning sequential addresses...
  MachineFrameInfo *FFI = Fn.getFrameInfo();

  unsigned MaxAlign = FFI->getMaxAlignment();

  // Start at the beginning of the local area.
  // The Offset is the distance from the stack top in the direction
  // of stack growth -- so it's always nonnegative.
  int64_t Offset = TFI.getOffsetOfLocalArea();
  if (StackGrowsDown)
    Offset = -Offset;
  assert(Offset >= 0
         && "Local area offset should be in direction of stack growth");

  // If there are fixed sized objects that are preallocated in the local area,
  // non-fixed objects can't be allocated right at the start of local area.
  // We currently don't support filling in holes in between fixed sized
  // objects, so we adjust 'Offset' to point to the end of last fixed sized
  // preallocated object.
  for (int i = FFI->getObjectIndexBegin(); i != 0; ++i) {
    int64_t FixedOff;
    if (StackGrowsDown) {
      // The maximum distance from the stack pointer is at lower address of
      // the object -- which is given by offset. For down growing stack
      // the offset is negative, so we negate the offset to get the distance.
      FixedOff = -FFI->getObjectOffset(i);
    } else {
      // The maximum distance from the start pointer is at the upper
      // address of the object.
      FixedOff = FFI->getObjectOffset(i) + FFI->getObjectSize(i);
    }
    if (FixedOff > Offset) Offset = FixedOff;
  }

  // First assign frame offsets to stack objects that are used to spill
  // callee saved registers.
  if (StackGrowsDown) {
    for (unsigned i = MinCSFrameIndex; i <= MaxCSFrameIndex; ++i) {
      // If stack grows down, we need to add size of find the lowest
      // address of the object.
      Offset += FFI->getObjectSize(i);

      unsigned Align = FFI->getObjectAlignment(i);
      // If the alignment of this object is greater than that of the stack,
      // then increase the stack alignment to match.
      MaxAlign = std::max(MaxAlign, Align);
      // Adjust to alignment boundary
      Offset = (Offset+Align-1)/Align*Align;

      FFI->setObjectOffset(i, -Offset);        // Set the computed offset
    }
  } else {
    int MaxCSFI = MaxCSFrameIndex, MinCSFI = MinCSFrameIndex;
    for (int i = MaxCSFI; i >= MinCSFI ; --i) {
      unsigned Align = FFI->getObjectAlignment(i);
      // If the alignment of this object is greater than that of the stack,
      // then increase the stack alignment to match.
      MaxAlign = std::max(MaxAlign, Align);
      // Adjust to alignment boundary
      Offset = (Offset+Align-1)/Align*Align;

      FFI->setObjectOffset(i, Offset);
      Offset += FFI->getObjectSize(i);
    }
  }

  // Make sure the special register scavenging spill slot is closest to the
  // frame pointer if a frame pointer is required.
  const TargetRegisterInfo *RegInfo = Fn.getTarget().getRegisterInfo();
  if (RS && RegInfo->hasFP(Fn)) {
    int SFI = RS->getScavengingFrameIndex();
    if (SFI >= 0)
      AdjustStackOffset(FFI, SFI, StackGrowsDown, Offset, MaxAlign);
  }

  // Make sure that the stack protector comes before the local variables on the
  // stack.
  if (FFI->getStackProtectorIndex() >= 0)
    AdjustStackOffset(FFI, FFI->getStackProtectorIndex(), StackGrowsDown,
                      Offset, MaxAlign);

  // Then assign frame offsets to stack objects that are not used to spill
  // callee saved registers.
  for (unsigned i = 0, e = FFI->getObjectIndexEnd(); i != e; ++i) {
    if (i >= MinCSFrameIndex && i <= MaxCSFrameIndex)
      continue;
    if (RS && (int)i == RS->getScavengingFrameIndex())
      continue;
    if (FFI->isDeadObjectIndex(i))
      continue;
    if (FFI->getStackProtectorIndex() == (int)i)
      continue;

    AdjustStackOffset(FFI, i, StackGrowsDown, Offset, MaxAlign);
  }

  // Make sure the special register scavenging spill slot is closest to the
  // stack pointer.
  if (RS && !RegInfo->hasFP(Fn)) {
    int SFI = RS->getScavengingFrameIndex();
    if (SFI >= 0)
      AdjustStackOffset(FFI, SFI, StackGrowsDown, Offset, MaxAlign);
  }

  // Round up the size to a multiple of the alignment, but only if there are
  // calls or alloca's in the function.  This ensures that any calls to
  // subroutines have their stack frames suitable aligned.
  // Also do this if we need runtime alignment of the stack.  In this case
  // offsets will be relative to SP not FP; round up the stack size so this
  // works.
  if (!RegInfo->targetHandlesStackFrameRounding() &&
      (FFI->hasCalls() || FFI->hasVarSizedObjects() ||
       (RegInfo->needsStackRealignment(Fn) &&
        FFI->getObjectIndexEnd() != 0))) {
    // If we have reserved argument space for call sites in the function
    // immediately on entry to the current function, count it as part of the
    // overall stack size.
    if (RegInfo->hasReservedCallFrame(Fn))
      Offset += FFI->getMaxCallFrameSize();

    unsigned AlignMask = std::max(TFI.getStackAlignment(),MaxAlign) - 1;
    Offset = (Offset + AlignMask) & ~uint64_t(AlignMask);
  }

  // Update frame info to pretend that this is part of the stack...
  FFI->setStackSize(Offset+TFI.getOffsetOfLocalArea());

  // Remember the required stack alignment in case targets need it to perform
  // dynamic stack alignment.
  FFI->setMaxAlignment(MaxAlign);
}


/// insertPrologEpilogCode - Scan the function for modified callee saved
/// registers, insert spill code for these callee saved registers, then add
/// prolog and epilog code to the function.
///
void PEI::insertPrologEpilogCode(MachineFunction &Fn) {
  const TargetRegisterInfo *TRI = Fn.getTarget().getRegisterInfo();

  // EXP adjust stack size to account for CSR spills that have been
  // moved by shrink wrapping out of the entry block.
  uint64_t stackSize = Fn.getFrameInfo()->getStackSize();
  bool adjustedStack = false;
  if (UsedCSRegs.count()) {
    const std::vector<CalleeSavedInfo> &CSI =
      Fn.getFrameInfo()->getCalleeSavedInfo();
    unsigned stackAdj = 0;
    MachineFunction::iterator BB = Fn.begin();
    if (BB->pred_size() == 0 && CSRSave[BB].empty()) {
      for (CSRegSet::iterator RI = UsedCSRegs.begin(),
             RE = UsedCSRegs.end(); RI != RE; ++RI) {
        stackAdj += CSI[*RI].getRegClass()->getSize();
      }
      DOUT << "Adjusting stack for pro/epi in function "
           << Fn.getFunction()->getName() << " to "
           << stackSize+stackAdj << "\n";
      Fn.getFrameInfo()->setStackSize(stackSize + stackAdj);
      adjustedStack = true;
    }
  }

  // Add prologue to the function...
  TRI->emitPrologue(Fn);

  // Add epilogue to restore the callee-save registers in each exiting block
  for (MachineFunction::iterator I = Fn.begin(), E = Fn.end(); I != E; ++I) {
    // If last instruction is a return instruction, add an epilogue
    if (!I->empty() && I->back().getDesc().isReturn())
      TRI->emitEpilogue(Fn, *I);
  }

  // EXP restore stack size.
  if (adjustedStack) {
    DOUT << "Restoring stack for pro/epi in function "
         << Fn.getFunction()->getName() << " to " << stackSize << "\n";
    Fn.getFrameInfo()->setStackSize(stackSize);
  }
}


/// replaceFrameIndices - Replace all MO_FrameIndex operands with physical
/// register references and actual offsets.
///
void PEI::replaceFrameIndices(MachineFunction &Fn) {
  if (!Fn.getFrameInfo()->hasStackObjects()) return; // Nothing to do?

  const TargetMachine &TM = Fn.getTarget();
  assert(TM.getRegisterInfo() && "TM::getRegisterInfo() must be implemented!");
  const TargetRegisterInfo &TRI = *TM.getRegisterInfo();
  const TargetFrameInfo *TFI = TM.getFrameInfo();
  bool StackGrowsDown =
    TFI->getStackGrowthDirection() == TargetFrameInfo::StackGrowsDown;
  int FrameSetupOpcode   = TRI.getCallFrameSetupOpcode();
  int FrameDestroyOpcode = TRI.getCallFrameDestroyOpcode();

  DOUT << "Replacing frame indices for: "
       << Fn.getFunction()->getName() << "\n";

  // TODO -- here: save stack adj for spill in BB of CSRs that
  //   will be used in other BBs. When the using BBs are processed,
  //   find the stack adj in the StackAdj map.
  uint64_t savedStackSize = Fn.getFrameInfo()->getStackSize();
#if 0
  DenseMap<MachineBasicBlock*, int> StackAdj;
  for (MachineFunction::iterator BB = Fn.begin(),
         E = Fn.end(); BB != E; ++BB) {
    uint64_t adjStackSize = 0;
    bool inEntryBlock = BB->pred_size() == 0;
    if (UsedCSRegs.count()) {
      int stackAdj = 0;
      unsigned numCSR = CSRSave[BB].count();
      DOUT << "MBB: " << BB->getBasicBlock()->getName()
           << " has " << numCSR << " CSR saves\n";
      if (! inEntryBlock && (numCSR > 0)) {
        stackAdj = numCSR * 4; // FIXME: slot size.
        StackAdj[BB] = stackAdj;
      }
    }
  }
#endif

  for (MachineFunction::iterator BB = Fn.begin(),
         E = Fn.end(); BB != E; ++BB) {
    int SPAdj = 0;  // SP offset due to call frame setup / destroy.
    if (RS) RS->enterBasicBlock(BB);

    uint64_t adjStackSize = 0;
    bool inEntryBlock = BB->pred_size() == 0;
    if (UsedCSRegs.count() && ! inEntryBlock) {
      const std::vector<CalleeSavedInfo> &CSI =
        Fn.getFrameInfo()->getCalleeSavedInfo();
      int stackAdj = 0;

      for (CSRegSet::iterator RI = CSRSave[BB].begin(),
             RE = CSRSave[BB].end(); RI != RE; ++RI) {
        stackAdj += CSI[*RI].getRegClass()->getSize();
      }

      if (stackAdj > 0) {
        adjStackSize = savedStackSize + stackAdj;
        DOUT << "Adjusting stack size " << savedStackSize
             << " by " << stackAdj << " to " << adjStackSize << "\n";
      }
    }

    for (MachineBasicBlock::iterator I = BB->begin(); I != BB->end(); ) {
      MachineInstr *MI = I;

      if (I->getOpcode() == TargetInstrInfo::DECLARE) {
        // Ignore it.
        ++I;
        continue;
      }

      if (I->getOpcode() == FrameSetupOpcode ||
          I->getOpcode() == FrameDestroyOpcode) {
        // Remember how much SP has been adjusted to create the call
        // frame.
        int Size = I->getOperand(0).getImm();

        if ((!StackGrowsDown && I->getOpcode() == FrameSetupOpcode) ||
            (StackGrowsDown && I->getOpcode() == FrameDestroyOpcode))
          Size = -Size;

        SPAdj += Size;

        MachineBasicBlock::iterator PrevI = BB->end();
        if (I != BB->begin()) PrevI = prior(I);
        TRI.eliminateCallFramePseudoInstr(Fn, *BB, I);

        // Visit the instructions created by eliminateCallFramePseudoInstr().
        if (PrevI == BB->end())
          I = BB->begin();     // The replaced instr was the first in the block.
        else
          I = next(PrevI);
        continue;
      }

      bool DoIncr = true;

      for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i)
        if (MI->getOperand(i).isFI()) {
          // Some instructions (e.g. inline asm instructions) can have
          // multiple frame indices and/or cause eliminateFrameIndex
          // to insert more than one instruction. We need the register
          // scavenger to go through all of these instructions so that
          // it can update its register information. We keep the
          // iterator at the point before insertion so that we can
          // revisit them in full.
          bool AtBeginning = (I == BB->begin());
          if (!AtBeginning) --I;

          // If this instruction has a FrameIndex operand, we need to
          // use that target machine register info object to eliminate
          // it.

          // EXP adjust stack size to account for CSR spills in non-entry
          // blocks, so that FI operands will get correct stack offsets.
          bool adjustedStack = false;
          if (adjStackSize != 0) {
            Fn.getFrameInfo()->setStackSize(adjStackSize);
            adjustedStack = true;
            DOUT << "Adjusted stack size to " << adjStackSize << " in ";
            MI->dump();
          }

          TRI.eliminateFrameIndex(MI, SPAdj, RS);

          // EXP restore stack size.
          if (adjustedStack) {
            Fn.getFrameInfo()->setStackSize(savedStackSize);
            DOUT << "Restored stack size to " << savedStackSize << "\n";
          }

          // Reset the iterator if we were at the beginning of the BB.
          if (AtBeginning) {
            I = BB->begin();
            DoIncr = false;
          }

          MI = 0;
          break;
        }

      if (DoIncr && I != BB->end()) ++I;

      // Update register states.
      if (RS && MI) RS->forward(MI);
    }

    assert(SPAdj == 0 && "Unbalanced call frame setup / destroy pairs?");
  }
}
