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
// This pass implements a shrink wrapping variant of prolog/epilog insertion:
// - Places callee saved register (CSR) spills and restores in the CFG to
//   tightly surround uses so that execution paths that do not use CSRs do not
//   pay the spill/restore penalty.
//
// - Avoiding placment of spills/restores in loops: if a CSR is used inside a
//   loop(nest), the spills are placed in the loop preheader, and restores are
//   placed in the loop exit nodes (the successors of the loop _exiting_ nodes).
//
// - Covering paths without CSR uses: e.g. if a restore is placed in a join
//   block, a matching spill is added to the end of all immediate predecessor
//   blocks that are not reached by a spill. Similarly for saves placed in
//   branch blocks.
//
// Shrink wrapping uses an analysis similar to the one in GVNPRE to determine
// which basic blocks require callee-saved register save/restore code.
//
// This pass uses MachineDominators and MachineLoopInfo. Loop information
// is used to prevent shrink wrapping of callee-saved register save/restore
// code into loops.
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
               cl::desc("Shrink wrap callee-saved register spills/restores"));

static cl::opt<std::string>
ShrinkWrapFunc("shrink-wrap-func",
               cl::desc("Shrink wrap the specified function"),
               cl::value_desc("funcname"),
               cl::init(""));

namespace {
  struct VISIBILITY_HIDDEN PEI : public MachineFunctionPass {
    static char ID;
    PEI() : MachineFunctionPass(&ID) {}

    const char *getPassName() const {
      return "Prolog/Epilog Insertion & Frame Finalization";
    }

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesCFG();
      if (ShrinkWrapping || ShrinkWrapFunc != "") {
        AU.addRequired<MachineLoopInfo>();
        AU.addRequired<MachineDominatorTree>();
      }
      AU.addPreserved<MachineLoopInfo>();
      AU.addPreserved<MachineDominatorTree>();
      MachineFunctionPass::getAnalysisUsage(AU);
    }

    /// runOnMachineFunction - Insert prolog/epilog code and replace abstract
    /// frame indexes with appropriate references.
    ///
    bool runOnMachineFunction(MachineFunction &Fn) {
      const TargetRegisterInfo *TRI = Fn.getTarget().getRegisterInfo();
      RS = TRI->requiresRegisterScavenging(Fn) ? new RegScavenger() : NULL;

      MF = &Fn;
      // DEBUG
      MFName = MF->getFunction()->getName();

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

      // Determine placement of CSR spill/restore code:
      //  - with shrink wrapping, place spills and restores to tightly
      //    enclose regions in the Machine CFG of the function where
      //    they are used. Without shrink wrapping
      //  - default (no shrink wrapping), place all spills in the
      //    entry block, all restores in return blocks.
      placeCSRSpillsAndRestores(Fn);

      // Add the code to save and restore the callee saved registers
      insertCSRSpillsAndRestores(Fn);

      // Allow the target machine to make final modifications to the function
      // before the frame layout is finalized.
      TRI->processFunctionBeforeFrameFinalized(Fn);

      // Calculate actual frame offsets for all abstract stack objects...
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

    // Machine function handle.
    MachineFunction* MF;

    // DEBUG
    std::string MFName;

    // Analysis info for spill/restore placement.
    // "CSR": "callee saved register".

    // CSRegSet contains indices into the Callee Saved Register Info
    // vector built by calculateCalleeSavedRegisters() and accessed
    // via MF.getFrameInfo()->getCalleeSavedInfo().
    typedef SparseBitVector<> CSRegSet;

    // CSRegBlockMap maps MachineBasicBlocks to sets of callee
    // saved register indices.
    typedef DenseMap<MachineBasicBlock*, CSRegSet> CSRegBlockMap;

    // Set and maps for computing CSR spill/restore placement:
    //  used in function (UsedCSRegs)
    //  used in a basic block (CSRUsed)
    //  anticipatable in a basic block (Antic{In,Out})
    //  available in a basic block (Avail{In,Out})
    //  to be spilled at the entry to a basic block (CSRSave)
    //  to be restored at the end of a basic block (CSRRestore)

    CSRegSet UsedCSRegs;
    CSRegBlockMap CSRUsed;
    CSRegBlockMap AnticIn, AnticOut;
    CSRegBlockMap AvailIn, AvailOut;
    CSRegBlockMap CSRSave;
    CSRegBlockMap CSRRestore;

    // Entry and return blocks of the current function.
    MachineBasicBlock* EntryBlock;
    SmallVector<MachineBasicBlock*, 4> ReturnBlocks;

    // Map of MBBs to top level MachineLoops.
    DenseMap<MachineBasicBlock*, MachineLoop*> TLLoops;

    // Flag to control shrink wrapping per-function:
    // may choose to skip shrink wrapping for certain
    // functions.
    bool ShrinkWrapThisFunction;

    bool calculateSets(MachineFunction &Fn);
    bool calcAnticInOut(MachineBasicBlock* MBB);
    bool calcAvailInOut(MachineBasicBlock* MBB);
    void calculateAnticAvail(MachineFunction &Fn);
    bool addCSRUsesForCoverBlock(MachineBasicBlock* MBB,
                                 SmallVector<MachineBasicBlock*, 4>& blks);
    void placeSpillsAndRestores(MachineFunction &Fn);
    void placeCSRSpillsAndRestores(MachineFunction &Fn);
    void calculateCalleeSavedRegisters(MachineFunction &Fn);
    void insertCSRSpillsAndRestores(MachineFunction &Fn);
    void calculateFrameObjectOffsets(MachineFunction &Fn);
    void replaceFrameIndices(MachineFunction &Fn);
    void insertPrologEpilogCode(MachineFunction &Fn);

    // Initialize all DFA sets, called before iterations.
    void initDFASets();

    // Initialize all shrink wrapping data.
    void initShrinkWrappingInfo();

    // Convienences for dealing with machine loops.
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

    void propagateUsesAroundLoop(MachineBasicBlock* MBB, MachineLoop* LP);

    void getLiveInCSRegs(MachineBasicBlock* MBB, CSRegSet& LiveInCSRegs) {
      const std::vector<CalleeSavedInfo> CSI =
        MF->getFrameInfo()->getCalleeSavedInfo();

      LiveInCSRegs.clear();
      if (! MBB->livein_empty()) {
        for (MachineBasicBlock::const_livein_iterator I = MBB->livein_begin(),
               E = MBB->livein_end(); I != E; ++I) {
          unsigned reg = *I;
          for (unsigned i = 0, e = CSI.size(); i != e; ++i)
            if (CSI[i].getReg() == reg) {
              LiveInCSRegs.set(i);
              break;
            }
        }
      }
    }


#ifndef NDEBUG
    // Debugging methods.
    std::string getBasicBlockName(const MachineBasicBlock* MBB);
    std::string stringifyCSRegSet(const CSRegSet& s);
    void dumpSet(const CSRegSet& s);
    void dumpUsed(MachineBasicBlock* MBB);
    void dumpAllUsed();
    void dumpSets(MachineBasicBlock* MBB);
    void dumpSets1(MachineBasicBlock* MBB);
    void dumpAllSets();
    void dumpSRSets();
#endif

  };
  char PEI::ID = 0;
}

#ifndef NDEBUG
// Debugging methods.
std::string PEI::getBasicBlockName(const MachineBasicBlock* MBB) {
  std::ostringstream name;
  if (MBB) {
    if (MBB->getBasicBlock())
      name << MBB->getBasicBlock()->getName();
    else
      name << "_MBB_" << MBB->getNumber();
  }
  return name.str();
}

std::string PEI::stringifyCSRegSet(const CSRegSet& s) {
  const TargetRegisterInfo* TRI = MF->getTarget().getRegisterInfo();
  const std::vector<CalleeSavedInfo> CSI =
    MF->getFrameInfo()->getCalleeSavedInfo();

  std::ostringstream srep;
  if (CSI.size() == 0) {
    srep << "[]";
    return srep.str();
  }
  srep << "[";
  CSRegSet::iterator I = s.begin(), E = s.end();
  if (I != E) {
    unsigned reg = CSI[*I].getReg();
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

void PEI::dumpSet(const CSRegSet& s) {
  DOUT << stringifyCSRegSet(s) << "\n";
}

void PEI::dumpUsed(MachineBasicBlock* MBB) {
  if (MBB) {
    DOUT << "CSRUsed[" << getBasicBlockName(MBB) << "] = "
         << stringifyCSRegSet(CSRUsed[MBB])  << "\n";
  }
}

void PEI::dumpAllUsed() {
    for (MachineFunction::iterator MBBI = MF->begin(), MBBE = MF->end();
         MBBI != MBBE; ++MBBI) {
      MachineBasicBlock* MBB = MBBI;
      dumpUsed(MBB);
    }
}

void PEI::dumpSets(MachineBasicBlock* MBB) {
  if (MBB) {
    DOUT << getBasicBlockName(MBB)           << " | "
         << stringifyCSRegSet(CSRUsed[MBB])  << " | "
         << stringifyCSRegSet(AnticIn[MBB])  << " | "
         << stringifyCSRegSet(AnticOut[MBB]) << " | "
         << stringifyCSRegSet(AvailIn[MBB])  << " | "
         << stringifyCSRegSet(AvailOut[MBB]) << "\n";
  }
}

void PEI::dumpSets1(MachineBasicBlock* MBB) {
  if (MBB) {
    DOUT << getBasicBlockName(MBB)             << " | "
         << stringifyCSRegSet(CSRUsed[MBB])    << " | "
         << stringifyCSRegSet(AnticIn[MBB])    << " | "
         << stringifyCSRegSet(AnticOut[MBB])   << " | "
         << stringifyCSRegSet(AvailIn[MBB])    << " | "
         << stringifyCSRegSet(AvailOut[MBB])   << " | "
         << stringifyCSRegSet(CSRSave[MBB])    << " | "
         << stringifyCSRegSet(CSRRestore[MBB]) << "\n";
  }
}

void PEI::dumpAllSets() {
    for (MachineFunction::iterator MBBI = MF->begin(), MBBE = MF->end();
         MBBI != MBBE; ++MBBI) {
      MachineBasicBlock* MBB = MBBI;
      dumpSets1(MBB);
    }
}

void PEI::dumpSRSets() {
  for (MachineFunction::iterator MBB = MF->begin(), E = MF->end();
       MBB != E; ++MBB) {
    if (! CSRSave[MBB].empty()) {
      DOUT << "SAVE[" << getBasicBlockName(MBB) << "] = "
           << stringifyCSRegSet(CSRSave[MBB]);
      if (CSRRestore[MBB].empty())
        DOUT << "\n";
    }
    if (! CSRRestore[MBB].empty()) {
      if (! CSRSave[MBB].empty())
        DOUT << "    ";
      DOUT << "RESTORE[" << getBasicBlockName(MBB) << "] = "
           << stringifyCSRegSet(CSRRestore[MBB]) << "\n";
    }
  }
}
#endif

// Initialize all DFA sets, called before iterations.
void PEI::initDFASets() {
  CSRSave.clear();
  CSRRestore.clear();
  AnticIn.clear();
  AnticOut.clear();
  AvailIn.clear();
  AvailOut.clear();
}

// Initialize all shrink wrapping data.
void PEI::initShrinkWrappingInfo() {
  EntryBlock = 0;
  if (! ReturnBlocks.empty())
    ReturnBlocks.clear();
  UsedCSRegs.clear();
  CSRUsed.clear();
  TLLoops.clear();
  initDFASets();
  // DEBUG: enable or disable shrink wrapping for the current function
  // as requested.
  ShrinkWrapThisFunction = false;
  if (ShrinkWrapFunc != "") {
    if (MFName == ShrinkWrapFunc) {
      ShrinkWrapThisFunction = true;
    }
#ifndef NDEBUG
    if (! ShrinkWrapThisFunction)
      DOUT << "DISABLED: " << MFName
           << ": shrink-wrap-func: not selected\n";
#endif
  } else {
    ShrinkWrapThisFunction = ShrinkWrapping;
  }
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
/// This function computes the maps of MBBs -> CSRs to spill and restore
/// in CSRSave, CSRRestore.
///
/// If shrink wrapping is not being performed, place all spills in
/// the entry block, all restores in return blocks. In this case,
/// CSRSave has a single mapping, CSRRestore has mappings for each
/// return block.
///
void PEI::placeCSRSpillsAndRestores(MachineFunction &Fn) {

#ifndef NDEBUG
  DOUT << "Place CSR spills/restores for "
       << MF->getFunction()->getName() << "\n";
#endif

  initShrinkWrappingInfo();

  if (calculateSets(Fn))
    placeSpillsAndRestores(Fn);
}

/// calculateAnticAvail - helper for computing the data flow
/// sets required for determining spill/restore placements.
///

bool PEI::calcAnticInOut(MachineBasicBlock* MBB) {
  bool changed = false;

  // AnticOut[MBB] = INTERSECT(AnticIn[S] for S in SUCCESSORS(MBB))
  SmallVector<MachineBasicBlock*, 4> successors;
  for (MachineBasicBlock::succ_iterator SI = MBB->succ_begin(),
         SE = MBB->succ_end(); SI != SE; ++SI) {
    MachineBasicBlock* SUCC = *SI;
    if (SUCC != MBB)
      successors.push_back(SUCC);
  }

  unsigned i = 0, e = successors.size();
  if (i != e) {
    CSRegSet prevAnticOut = AnticOut[MBB];
    MachineBasicBlock* SUCC = successors[i];

    AnticOut[MBB] = AnticIn[SUCC];
    for (++i; i != e; ++i) {
      AnticOut[MBB] &= AnticIn[SUCC];
    }
    if (prevAnticOut != AnticOut[MBB])
      changed = true;
  }

  // AnticIn[MBB] = UNION(CSRUsed[MBB], AnticOut[MBB]);
  CSRegSet prevAnticIn = AnticIn[MBB];
  AnticIn[MBB] = CSRUsed[MBB] | AnticOut[MBB];
  if (prevAnticIn |= AnticIn[MBB])
    changed = true;
  return changed;
}

bool PEI::calcAvailInOut(MachineBasicBlock* MBB) {
  bool changed = false;

  // AvailIn[MBB] = INTERSECT(AvailOut[P] for P in PREDECESSORS(MBB))
  SmallVector<MachineBasicBlock*, 4> predecessors;
  for (MachineBasicBlock::pred_iterator PI = MBB->pred_begin(),
         PE = MBB->pred_end(); PI != PE; ++PI) {
    MachineBasicBlock* PRED = *PI;
    if (PRED != MBB)
      predecessors.push_back(PRED);
  }

  unsigned i = 0, e = predecessors.size();
  if (i != e) {
    CSRegSet prevAvailIn = AvailIn[MBB];
    MachineBasicBlock* PRED = predecessors[i];

    AvailIn[MBB] = AvailOut[PRED];
    for (++i; i != e; ++i) {
      AvailIn[MBB] &= AvailOut[PRED];
    }
    if (prevAvailIn != AvailIn[MBB])
      changed = true;
  }

  // AvailOut[MBB] = UNION(CSRUsed[MBB], AvailIn[MBB]);
  CSRegSet prevAvailOut = AvailOut[MBB];
  AvailOut[MBB] = CSRUsed[MBB] | AvailIn[MBB];
  if (prevAvailOut |= AvailOut[MBB])
    changed = true;
  return changed;
}

void PEI::calculateAnticAvail(MachineFunction &Fn) {

  // Initialize data flow sets.
  initDFASets();

  // Calulate Antic{In,Out} and Avail{In,Out} iteratively on the MCFG.
  bool changed = true;
  unsigned iterations = 0;
  while (changed) {
    changed = false;
    for (MachineFunction::iterator MBBI = Fn.begin(), MBBE = Fn.end();
         MBBI != MBBE; ++MBBI) {
      MachineBasicBlock* MBB = MBBI;

      // Calculate anticipated regs at MBB from
      // anticipated at successors of MBB.
      changed |= calcAnticInOut(MBB);

      // Calculate available regs at MBB from
      // available at predecessors of MBB.
      changed |= calcAvailInOut(MBB);
    }
    ++iterations;
  }

#ifndef NDEBUG
  DOUT << "-----------------------------------------------------------\n";
  DOUT << " Antic/Avail Sets:\n";
  DOUT << "-----------------------------------------------------------\n";
  DOUT << "iterations = " << iterations << "\n";
  DOUT << "-----------------------------------------------------------\n";
  DOUT << "MBB | USED | ANTIC_IN | ANTIC_OUT | AVAIL_IN | AVAIL_OUT\n";
  DOUT << "-----------------------------------------------------------\n";
  for (MachineFunction::iterator MBBI = Fn.begin(), MBBE = Fn.end();
       MBBI != MBBE; ++MBBI) {
    MachineBasicBlock* MBB = MBBI;
    dumpSets(MBB);
  }
#endif
}

void PEI::propagateUsesAroundLoop(MachineBasicBlock* MBB, MachineLoop* LP) {
  if (! MBB || !LP)
    return;

  std::vector<MachineBasicBlock*> loopBlocks = LP->getBlocks();
  for (unsigned i = 0, e = loopBlocks.size(); i != e; ++i) {
    MachineBasicBlock* LBB = loopBlocks[i];
    if (LBB == MBB)
      continue;
    if (CSRUsed[LBB].contains(CSRUsed[MBB]))
      continue;
    CSRUsed[LBB] |= CSRUsed[MBB];
  }
}

/// calculateSets - helper function for placeCSRSpillsAndRestores,
/// collect the CSRs used in this function, develop the DF sets that
/// describe the minimal regions in the Machine CFG around which spills,
/// restores must be placed.
///
/// This function decides if shrink wrapping should actually be done:
///   if all CSR uses are in the entry block, no shrink wrapping is possible,
///   so ShrinkWrapping is turned off (for the current function) and the
///   function returns false.
///
bool PEI::calculateSets(MachineFunction &Fn) {

  // Sets used to compute spill, restore placement sets.
  const std::vector<CalleeSavedInfo> CSI =
    Fn.getFrameInfo()->getCalleeSavedInfo();

  // If no CSRs used, we are done.
  if (CSI.empty()) {
#ifndef NDEBUG
    if (ShrinkWrapThisFunction)
      DOUT << "DISABLED: " << Fn.getFunction()->getName()
           << ": uses no callee-saved registers\n";
#endif
    return false;
  }

  // Save refs to entry and return blocks.
  EntryBlock = Fn.begin();
  for (MachineFunction::iterator MBB = Fn.begin(), E = Fn.end();
       MBB != E; ++MBB)
    if (!MBB->empty() && MBB->back().getDesc().isReturn())
      ReturnBlocks.push_back(MBB);

  // Limit shrink wrapping to functions with fewer than 1K MBBs.
  if (Fn.size() > 1000) {
#ifndef NDEBUG
    if (ShrinkWrapThisFunction)
      DOUT << "DISABLED: " << Fn.getFunction()->getName()
           << ": too large (" << Fn.size() << " MBBs)\n";
#endif
    ShrinkWrapThisFunction = false;
    return false;
  }

  // Initialize UsedCSRegs set, CSRUsed map.
  // At the same time, put entry block directly into
  // CSRSave, CSRRestore sets if any CSRs are used.
  //
  // Quick exit option (not implemented):
  //   Given N CSR uses in entry block,
  //   revert to default behavior, skip the placement
  //   step and put all saves in entry, restores in
  //   return blocks.

  // If not shrink wrapping at this point, just set
  // CSR{Save,Restore}[] and UsedCSRegs as necessary.
  if (! ShrinkWrapThisFunction) {
    for (unsigned inx = 0, e = CSI.size(); inx != e; ++inx) {
      UsedCSRegs.set(inx);
      CSRSave[EntryBlock].set(inx);
      for (unsigned ri = 0, re = ReturnBlocks.size(); ri != re; ++ri)
        CSRRestore[ReturnBlocks[ri]].set(inx);
    }
    return false;
  }

  // Walk instructions in all MBBs, create basic sets, choose
  // whether or not to shrink wrap this function.
  MachineLoopInfo &LI = getAnalysis<MachineLoopInfo>();
  MachineDominatorTree &DT = getAnalysis<MachineDominatorTree>();
  const TargetRegisterInfo *TRI = Fn.getTarget().getRegisterInfo();

  bool allCSRUsesInEntryBlock = true;
  bool someCSRUsesInEntryBlock = false;
  for (MachineFunction::iterator MBBI = Fn.begin(), MBBE = Fn.end();
       MBBI != MBBE; ++MBBI) {
    MachineBasicBlock* MBB = MBBI;
    for (MachineBasicBlock::iterator I = MBB->begin(); I != MBB->end(); ++I) {
      for (unsigned inx = 0, e = CSI.size(); inx != e; ++inx) {
        unsigned Reg = CSI[inx].getReg();
        // If instruction I reads or modifies Reg, add it to UsedCSRegs,
        // CSRUsed map for the current block.
        for (unsigned opInx = 0, opEnd = I->getNumOperands();
             opInx != opEnd; ++opInx) {
          const MachineOperand &MO = I->getOperand(opInx);
          if (! (MO.isReg() && (MO.isUse() || MO.isDef())))
            continue;
          unsigned MOReg = MO.getReg();
          if (!MOReg)
            continue;
          if (MOReg == Reg ||
              (TargetRegisterInfo::isPhysicalRegister(MOReg) &&
               TargetRegisterInfo::isPhysicalRegister(Reg) &&
               TRI->isSubRegister(Reg, MOReg))) {
            // CSR Reg is defined/used in block MBB.
            UsedCSRegs.set(inx);
            CSRUsed[MBB].set(inx);
            // Check for uses in EntryBlock.
            if (MBB == EntryBlock)
              someCSRUsesInEntryBlock = true;
            else
              allCSRUsesInEntryBlock = false;
          }
        }
      }
    }

    if (CSRUsed[MBB].empty())
      continue;
    // Propagate CSRUsed[MBB] in loops
    if (MachineLoop* LP = LI.getLoopFor(MBB)) {

      // Add top level loop to work list.
      MachineBasicBlock* HDR = getTopLevelLoopPreheader(LP);
      MachineLoop* PLP = getTopLevelLoopParent(LP);

      if (! HDR) {
        HDR = PLP->getHeader();
        assert(HDR->pred_size() > 0 && "Loop header has no predecessors?");
        MachineBasicBlock::pred_iterator PI = HDR->pred_begin();
        HDR = *PI;
      }
      TLLoops[HDR] = PLP;

      // My loop.
      propagateUsesAroundLoop(MBB, LP);
      // My parent loops.
      for (MachineLoop* PLP = LP->getParentLoop(); PLP != 0;
           PLP = PLP->getParentLoop()) {
        propagateUsesAroundLoop(MBB, PLP);
      }
      // My child loops.
      const std::vector<MachineLoop*> &children = LP->getSubLoops();
      for (unsigned i = 0, e = children.size(); i != e; ++i) {
        MachineLoop* CLP = children[i];
        propagateUsesAroundLoop(MBB, CLP);
      }
    }
  }

  if (ShrinkWrapThisFunction) {
    if (allCSRUsesInEntryBlock) {
#ifndef NDEBUG
      DOUT << "DISABLED: " << Fn.getFunction()->getName()
           << ": all CSRs used in EntryBlock\n";
#endif
      ShrinkWrapThisFunction = false;
    } else {
      allCSRUsesInEntryBlock = true;
      for (MachineBasicBlock::succ_iterator SI = EntryBlock->succ_begin(),
             SE = EntryBlock->succ_end(); SI != SE; ++SI) {
        MachineBasicBlock* SUCC = *SI;
        if (CSRUsed[SUCC] != UsedCSRegs)
          allCSRUsesInEntryBlock = false;
      }
      if (allCSRUsesInEntryBlock) {
#ifndef NDEBUG
        DOUT << "DISABLED: " << Fn.getFunction()->getName()
             << ": all CSRs used in imm successors of EntryBlock\n";
#endif
        ShrinkWrapThisFunction = false;
      }
    }

    if (ShrinkWrapThisFunction) {
      // Check if MBB uses CSRs and dominates all exit nodes.
      // Such nodes are equiv. to the entry node w.r.t.
      // CSR uses: every path through the function must
      // pass through this node. If each CSR is used at least
      // once by these nodes, shrink wrapping is disabled.
      CSRegSet CSRUsedInChokePoints;
      for (MachineFunction::iterator MBBI = Fn.begin(), MBBE = Fn.end();
           MBBI != MBBE; ++MBBI) {
        MachineBasicBlock* MBB = MBBI;
        if (MBB == EntryBlock || CSRUsed[MBB].empty() || MBB->succ_size() < 1)
          continue;
        bool dominatesExitNodes = true;
        for (unsigned ri = 0, re = ReturnBlocks.size(); ri != re; ++ri)
          if (! DT.dominates(MBB, ReturnBlocks[ri])) {
            dominatesExitNodes = false;
            break;
          }
        if (dominatesExitNodes) {
          CSRUsedInChokePoints |= CSRUsed[MBB];
          if (CSRUsedInChokePoints == UsedCSRegs) {
#ifndef NDEBUG
            DOUT << "DISABLED: " << Fn.getFunction()->getName()
                 << ": all CSRs used in choke point(s) at "
                 << getBasicBlockName(MBB) << "\n";
#endif
            ShrinkWrapThisFunction = false;
            break;
          }
        }
      }
    }
  }

#ifndef NDEBUG
  DOUT << "-----------------------------------------------------------\n";
  DOUT << "UsedCSRegs = " << stringifyCSRegSet(UsedCSRegs) << "\n";
  DOUT << "-----------------------------------------------------------\n";
  dumpAllUsed();
  DOUT << "-----------------------------------------------------------\n";
#endif

  // Early exit:
  // 1. Not asked to do shrink wrapping => just "place" all spills(restores)
  //    in the entry(return) block(s), already done above.
  // 2. All CSR uses in entry block => same as case 1, but say we will
  //    not shrink wrap the current function.
  if (! ShrinkWrapThisFunction) {
    CSRSave[EntryBlock] = UsedCSRegs;
    for (unsigned ri = 0, re = ReturnBlocks.size(); ri != re; ++ri)
      CSRRestore[ReturnBlocks[ri]] = UsedCSRegs;
    return false;
  }

  // Build DF sets to determine minimal regions in the MCFG around which
  // CSRs must be spilled and restored.
  calculateAnticAvail(Fn);

  return true;
}

/// addCSRUsesForCoverBlock - helper for placeSpillsAndRestores() which
/// adds uses of CSRs saved in branch point MBBs to any successor blocks
//  connected to regions with no uses of the saved CSRs.
///
bool PEI::addCSRUsesForCoverBlock(MachineBasicBlock* MBB,
                                  SmallVector<MachineBasicBlock*, 4>& blks) {

  if (MBB->succ_size() < 2 && MBB->pred_size() < 2) {
    bool processThisBlock = false;
    for (MachineBasicBlock::succ_iterator SI = MBB->succ_begin(),
           SE = MBB->succ_end(); SI != SE; ++SI) {
      MachineBasicBlock* SUCC = *SI;
      if (SUCC->succ_size() == 0 && SUCC->pred_size() > 1) {
        processThisBlock = true;
        break;
      }
    }
    if (! processThisBlock)
      return false;
  }

  bool addedUses = false;
  bool propagateSpills = (! CSRSave[MBB].empty());
  bool propagateRestores = (! CSRRestore[MBB].empty());
  // Add uses to all immed. successors of MBB.
  for (MachineBasicBlock::succ_iterator SI = MBB->succ_begin(),
         SE = MBB->succ_end(); SI != SE; ++SI) {
    MachineBasicBlock* SUCC = *SI;

    // Self-loop
    if (SUCC == MBB)
      continue;
    if (propagateSpills && ! CSRUsed[SUCC].contains(CSRSave[MBB])) {
      CSRUsed[SUCC] |= CSRSave[MBB];
      addedUses = true;
      blks.push_back(SUCC);
#ifndef NDEBUG
      DOUT << "Adding uses for " << stringifyCSRegSet(CSRSave[MBB])
           << " from "           << getBasicBlockName(MBB)
           << " to successor "   << getBasicBlockName(SUCC) << "\n";
#endif
    }
    if (propagateRestores &&
        ! CSRUsed[SUCC].contains(CSRRestore[MBB])) {
      CSRUsed[SUCC] |= CSRRestore[MBB];
      addedUses = true;
      blks.push_back(SUCC);
#ifndef NDEBUG
      DOUT << "Adding uses for " << stringifyCSRegSet(CSRRestore[MBB])
           << " from "           << getBasicBlockName(MBB)
           << " to successor "   << getBasicBlockName(SUCC) << "\n";
#endif
    }
  }
  for (MachineBasicBlock::pred_iterator PI = MBB->pred_begin(),
         PE = MBB->pred_end(); PI != PE; ++PI) {
    MachineBasicBlock* PRED = *PI;
    // Self-loop
    if (PRED == MBB)
      continue;
    if (propagateSpills && ! CSRUsed[PRED].contains(CSRSave[MBB])) {
      CSRUsed[PRED] |= CSRSave[MBB];
      addedUses = true;
      blks.push_back(PRED);
#ifndef NDEBUG
      DOUT << "Adding uses for " << stringifyCSRegSet(CSRSave[MBB])
           << " from "           << getBasicBlockName(MBB)
           << " to predecessor " << getBasicBlockName(PRED) << "\n";
#endif
    }
    if (propagateRestores &&
        ! CSRUsed[PRED].contains(CSRRestore[MBB])) {
      CSRUsed[PRED] |= CSRRestore[MBB];
      addedUses = true;
      blks.push_back(PRED);
#ifndef NDEBUG
      DOUT << "Adding uses for " << stringifyCSRegSet(CSRRestore[MBB])
           << " from "           << getBasicBlockName(MBB)
           << " to predecessor " << getBasicBlockName(PRED) << "\n";
#endif
    }
  }
  return addedUses;
}

/// placeSpillsAndRestores - decide which MBBs need spills, restores
/// of CSRs.
///
void PEI::placeSpillsAndRestores(MachineFunction &Fn) {

#ifndef NDEBUG
  DOUT << "-----------------------------------------------------------\n";
#endif

  SmallVector<MachineBasicBlock*, 4> coverBlocks;
  bool changed = true;
  unsigned iterations = 0;
  while (changed) {
    changed = false;

    // Calculate CSR{Save,Restore} using Antic, Avail on the Machine-CFG.
    for (MachineFunction::iterator MBBI = Fn.begin(), MBBE = Fn.end();
         MBBI != MBBE; ++MBBI) {
      MachineBasicBlock* MBB = MBBI;

      // Intersect (CSRegs - AnticIn[P]) for all predecessors P of MBB
      CSRegSet anticInPreds;
      MachineBasicBlock::pred_iterator PI = MBB->pred_begin(),
        PE = MBB->pred_end();
      if (PI != PE) {
        MachineBasicBlock* PRED = *PI;
        anticInPreds = UsedCSRegs - AnticIn[PRED];
        for (++PI; PI != PE; ++PI) {
          PRED = *PI;
          anticInPreds &= (UsedCSRegs - AnticIn[PRED]);
        }
      } else {
        // Take care of uses in entry blocks (which have no predecessors).
        // This is necessary because the DFA formulation assumes the
        // entry and (multiple) exit nodes cannot have CSR uses, which
        // is not the case in the real world.
        anticInPreds = UsedCSRegs;
      }

      // Compute spills required at MBB:
      CSRSave[MBB] |= (AnticIn[MBB] - AvailIn[MBB]) & anticInPreds;

      if (! CSRSave[MBB].empty()) {
        if (MBB == EntryBlock) {
          for (unsigned ri = 0, re = ReturnBlocks.size(); ri != re; ++ri)
            CSRRestore[ReturnBlocks[ri]] |= CSRSave[MBB];
        } else {
          // EXP -- reset all regs spilled in MBB that are
          // also spilled in EntryBlock
          if (CSRSave[EntryBlock].intersects(CSRSave[MBB])) {
            CSRSave[MBB] = CSRSave[MBB] - CSRSave[EntryBlock];
          }
        }
      }
      // Remember this block for adding restores to successor
      // blocks for multi-entry region.
      if (! CSRSave[MBB].empty() && MBB->succ_size() > 1)
        coverBlocks.push_back(MBB);

#ifndef NDEBUG
      if (! CSRSave[MBB].empty())
        DOUT << "SAVE[" << getBasicBlockName(MBB) << "] = "
             << stringifyCSRegSet(CSRSave[MBB]) << "\n";
#endif

      // Intersect (CSRegs - AvailOut[S]) for all successors S of MBB
      CSRegSet availOutSucc;
      MachineBasicBlock::succ_iterator SI = MBB->succ_begin(),
        SE = MBB->succ_end();
      if (SI != SE) {
        MachineBasicBlock* SUCC = *SI;

        availOutSucc = UsedCSRegs - AvailOut[SUCC];
        for (++SI; SI != SE; ++SI) {
          SUCC = *SI;
          availOutSucc &= (UsedCSRegs - AvailOut[SUCC]);
        }
      } else {
        if (! CSRUsed[MBB].empty() || ! AvailOut[MBB].empty()) {
          // Take care of uses in return blocks (which have no successors).
          // This is necessary because the DFA formulation assumes the
          // entry and (multiple) exit nodes cannot have CSR uses, which
          // is not the case in the real world.
          availOutSucc = UsedCSRegs;
        }
      }
      // Compute restores required at MBB:
      CSRRestore[MBB] |= (AvailOut[MBB] - AnticOut[MBB]) & availOutSucc;

      // Postprocess restore placements at MBB.
      // Remove the CSRs that are restored in the return blocks.
      // Lest this be confusing, note that:
      // CSRSave[EntryBlock] == CSRRestore[B] for all B in ReturnBlocks.
      if (MBB->succ_size() && ! CSRRestore[MBB].empty()) {
        if (! CSRSave[EntryBlock].empty())
          CSRRestore[MBB] = CSRRestore[MBB] - CSRSave[EntryBlock];
      }
      // Remember this block for adding saves to predecessor
      // blocks for multi-entry region.
      if (! CSRRestore[MBB].empty())
        coverBlocks.push_back(MBB);
#ifndef NDEBUG
      if (! CSRRestore[MBB].empty())
        DOUT << "RESTORE[" << getBasicBlockName(MBB) << "] = "
             << stringifyCSRegSet(CSRRestore[MBB]) << "\n";
#endif
    }

    // Place restores for top level loops where needed.
    // Take advantage of partial top level loops:
    //   Given TLLoop L:
    //   attempt to place restores in EXB in exitingBLocks(L)
    //   along the exit paths of EXB if there are no
    //   restores of CSRUsed[getHeader(L)] on paths leaving L.
    for (DenseMap<MachineBasicBlock*, MachineLoop*>::iterator
           I = TLLoops.begin(), E = TLLoops.end(); I != E; ++I) {
      MachineBasicBlock* MBB = I->first;
      MachineLoop* LP = I->second;
      MachineBasicBlock* HDR = LP->getHeader();
      SmallVector<MachineBasicBlock*, 4> exitBlocks;
      CSRegSet loopSpills;

      loopSpills = CSRSave[MBB];
      if (CSRSave[MBB].empty()) {
        loopSpills = CSRUsed[HDR];
        assert(!loopSpills.empty() && "No CSRs used in loop?");
      } else if (CSRRestore[MBB].contains(CSRSave[MBB]))
        continue;

      LP->getExitBlocks(exitBlocks);
      assert(exitBlocks.size() > 0 && "Loop has no top level exit blocks?");
      for (unsigned i = 0, e = exitBlocks.size(); i != e; ++i) {
        MachineBasicBlock* EXB = exitBlocks[i];
        if (! CSRRestore[EXB].contains(loopSpills) &&
            ! CSRUsed[EXB].contains(loopSpills)) {
          CSRUsed[EXB] |= loopSpills;
          changed = true;
#ifndef NDEBUG
          DOUT << "Adding uses " << stringifyCSRegSet(loopSpills)
               << " for loop "   << getBasicBlockName(MBB)
               << " to "         << getBasicBlockName(EXB) << "\n";
#endif
          if (EXB->succ_size() > 1 || EXB->pred_size() > 1)
            coverBlocks.push_back(EXB);
        }
      }
    }

    // Add uses for spills, restores related to branch, join points.
    SmallVector<MachineBasicBlock*, 4> tmpBlocks;
    while (! coverBlocks.empty()) {
      MachineBasicBlock* MBB = coverBlocks.pop_back_val();
      changed |= addCSRUsesForCoverBlock(MBB, tmpBlocks);
    }
    if (! tmpBlocks.empty()) {
      coverBlocks = tmpBlocks;
      tmpBlocks.clear();
    }

#if 0
    // Check for all CSRs used in EntryBlock.
    if (CSRUsed[EntryBlock] == UsedCSRegs) {
#ifndef NDEBUG
      DOUT << "-----------------------------------------------------------\n";
      DOUT << "DISABLED: " << Fn.getFunction()->getName()
           << ": all CSRs used in EntryBlock\n";
#endif
      ShrinkWrapThisFunction = false;
      break;
    }

    // Check for all CSRs restored in all exit blocks, in which
    // case we can skip the rest of the placement phase.
    bool allReturnsRestoreAllCSRs = true;
    for (unsigned ri = 0, re = ReturnBlocks.size(); ri != re; ++ri) {
      if (CSRRestore[ReturnBlocks[ri]] != UsedCSRegs) {
        allReturnsRestoreAllCSRs = false;
        break;
      }
    }
    if (allReturnsRestoreAllCSRs) {
#ifndef NDEBUG
      DOUT << "-----------------------------------------------------------\n";
      DOUT << "DISABLED: " << Fn.getFunction()->getName()
           << ": all CSRs restored in all return blocks\n";
#endif
      ShrinkWrapThisFunction = false;
      break;
    }
#endif

    // If things changed, iterate the DFA...
    if (changed) {
      calculateAnticAvail(Fn);
    }
    ++iterations;
  }

  // EXP -- post-process top level loops.
  for (DenseMap<MachineBasicBlock*, MachineLoop*>::iterator
         I = TLLoops.begin(), E = TLLoops.end(); I != E; ++I) {
    MachineBasicBlock* MBB = I->first;
    MachineLoop* LP = I->second;
    MachineBasicBlock* HDR = LP->getHeader();
    SmallVector<MachineBasicBlock*, 4> exitBlocks;
    CSRegSet loopSpills;

    loopSpills = CSRSave[MBB];
    if (CSRSave[MBB].empty()) {
      loopSpills = CSRUsed[HDR];
      assert(!loopSpills.empty() && "No CSRs used in loop?");
    } else if (CSRRestore[MBB].contains(CSRSave[MBB]))
      continue;

    LP->getExitBlocks(exitBlocks);
    assert(exitBlocks.size() > 0 && "Loop has no top level exit blocks?");
    for (unsigned i = 0, e = exitBlocks.size(); i != e; ++i) {
      MachineBasicBlock* EXB = exitBlocks[i];
      if (! CSRRestore[EXB].contains(loopSpills)) {
        // Check for restores of loopSpills below EXB.
        bool needsRestore = true;
        for (df_iterator<MachineBasicBlock*> BI = df_begin(EXB),
               BE = df_end(EXB); BI != BE; ++BI) {
          MachineBasicBlock* SBB = *BI;
          if (CSRRestore[SBB].contains(loopSpills)) {
            needsRestore = false;
            break;
          }
        }
        if (needsRestore) {
          CSRRestore[EXB] |= loopSpills;
#ifndef NDEBUG
          DOUT << "Adding restores " << stringifyCSRegSet(loopSpills)
               << " for loop "       << getBasicBlockName(MBB)
               << " to "             << getBasicBlockName(EXB) << "\n";
#endif
        }
      }
    }
  }

#if 0
  if (! ShrinkWrapThisFunction) {
    CSRSave.clear();
    CSRRestore.clear();
    CSRSave[EntryBlock] = UsedCSRegs;
    for (unsigned ri = 0, re = ReturnBlocks.size(); ri != re; ++ri)
      CSRRestore[ReturnBlocks[ri]] = UsedCSRegs;
  }
#endif

#ifndef NDEBUG
  if (ShrinkWrapThisFunction) {
    DOUT << "-----------------------------------------------------------\n";
    DOUT << "ENABLED: " << Fn.getFunction()->getName() << "\n";
  }
  DOUT << "-----------------------------------------------------------\n";
  DOUT << ">>>> Final SAVE, RESTORE:\n";
  DOUT << "-----------------------------------------------------------\n";
  DOUT << "iterations = " << iterations << "\n";
  DOUT << "-----------------------------------------------------------\n";
  dumpSRSets();
  DOUT << "-----------------------------------------------------------\n";
  DOUT << "<<<< Final SAVE, RESTORE\n";
  DOUT << "-----------------------------------------------------------\n";
#endif
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

/// insertCSRSpillsAndRestores - Insert spill and restore code for
/// callee saved registers used in the function, handling shrink wrapping.
///
void PEI::insertCSRSpillsAndRestores(MachineFunction &Fn) {
  // Get callee saved register information.
  MachineFrameInfo *FFI = Fn.getFrameInfo();
  const std::vector<CalleeSavedInfo> &CSI = FFI->getCalleeSavedInfo();

  // Early exit if no callee saved registers are modified!
  if (CSI.empty())
    return;

  const TargetInstrInfo &TII = *Fn.getTarget().getInstrInfo();
  MachineBasicBlock::iterator I;

#ifndef NDEBUG
  if (ShrinkWrapThisFunction)
    DOUT << "Inserting CSR spills/restores in function "
         << Fn.getFunction()->getName() << "\n";
#endif

  if (! ShrinkWrapThisFunction) {
    // Spill using target interface.
    I = EntryBlock->begin();
    if (!TII.spillCalleeSavedRegisters(*EntryBlock, I, CSI)) {
      for (unsigned i = 0, e = CSI.size(); i != e; ++i) {
        // Add the callee-saved register as live-in.
        // It's killed at the spill.
        EntryBlock->addLiveIn(CSI[i].getReg());

        // Insert the spill to the stack frame.
        TII.storeRegToStackSlot(*EntryBlock, I, CSI[i].getReg(), true,
                                CSI[i].getFrameIdx(), CSI[i].getRegClass());
      }
    }

    // Restore using target interface.
    for (unsigned ri = 0, re = ReturnBlocks.size(); ri != re; ++ri) {
      MachineBasicBlock* MBB = ReturnBlocks[ri];
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
    return;
  }

  // Insert spills.
  std::vector<CalleeSavedInfo> blockCSI;
  for (CSRegBlockMap::iterator BI = CSRSave.begin(),
         BE = CSRSave.end(); BI != BE; ++BI) {
    MachineBasicBlock* MBB = BI->first;
    CSRegSet save = BI->second;

    if (save.empty())
      continue;

#ifndef NDEBUG
    DOUT << "Spilling " << stringifyCSRegSet(save)
         << " in " << getBasicBlockName(MBB) << "\n";
#endif

    blockCSI.clear();
    for (CSRegSet::iterator RI = save.begin(),
           RE = save.end(); RI != RE; ++RI) {
      blockCSI.push_back(CSI[*RI]);
    }
    assert(blockCSI.size() > 0 &&
           "Could not collect callee saved register info");

    I = MBB->begin();

    // When shrink wrapping, use stack slot stores/loads.
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

#ifndef NDEBUG
  DOUT << "-----------------------------------------------------------\n";
#endif

  for (CSRegBlockMap::iterator BI = CSRRestore.begin(),
         BE = CSRRestore.end(); BI != BE; ++BI) {
    MachineBasicBlock* MBB = BI->first;
    CSRegSet restore = BI->second;

    if (restore.empty())
      continue;
#ifndef NDEBUG
    DOUT << "Restoring " << stringifyCSRegSet(restore)
         << " in " << getBasicBlockName(MBB) << "\n";
#endif

    blockCSI.clear();
    for (CSRegSet::iterator RI = restore.begin(),
           RE = restore.end(); RI != RE; ++RI) {
      blockCSI.push_back(CSI[*RI]);
    }
    assert(blockCSI.size() > 0 &&
           "Could not find callee saved register info");

    // If MBB is empty and needs restores, insert at the _beginning_.
    if (MBB->empty()) {
      I = MBB->begin();
    } else {
      I = MBB->end();
      --I;

      // Skip over all terminator instructions, which are part of the
      // return sequence.
      if (! I->getDesc().isTerminator()) {
        ++I;
      } else {
        MachineBasicBlock::iterator I2 = I;
        while (I2 != MBB->begin() && (--I2)->getDesc().isTerminator())
          I = I2;
      }
    }

    bool AtStart = I == MBB->begin();
    MachineBasicBlock::iterator BeforeI = I;
    if (!AtStart)
      --BeforeI;

    // Restore all registers immediately before the return and any
    // terminators that preceed it.
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
#ifndef NDEBUG
  DOUT << "-----------------------------------------------------------\n";
#endif
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

  // Add prologue to the function...
  TRI->emitPrologue(Fn);

  // Add epilogue to restore the callee-save registers in each exiting block
  for (MachineFunction::iterator I = Fn.begin(), E = Fn.end(); I != E; ++I) {
    // If last instruction is a return instruction, add an epilogue
    if (!I->empty() && I->back().getDesc().isReturn())
      TRI->emitEpilogue(Fn, *I);
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

  for (MachineFunction::iterator BB = Fn.begin(),
         E = Fn.end(); BB != E; ++BB) {
    int SPAdj = 0;  // SP offset due to call frame setup / destroy.
    if (RS) RS->enterBasicBlock(BB);

    for (MachineBasicBlock::iterator I = BB->begin(); I != BB->end(); ) {
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

      MachineInstr *MI = I;
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

          TRI.eliminateFrameIndex(MI, SPAdj, RS);

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
