import {SourceInfo} from 'org_xprof/frontend/app/common/interfaces/source_info.jsonpb_decls';

/** Profile is the top-level data that summarizes a program. */
export interface Profile {
  byCategory?: Node;
  byProgram?: Node;
  deviceType?: string;
  aggDvfsTimeScaleMultiplier?: number;
  byCategoryExcludeIdle?: Node;
  byProgramExcludeIdle?: Node;
  byProvenance?: Node;
  byProvenanceExcludeIdle?: Node;
}

/** An entry in the profile tree. (An instruction, or set of instructions). */
export interface Node {
  name?: string;
  metrics?: Metrics;
  children?: Node[];
  category?: Node.InstructionCategory;
  xla?: Node.XLAInstruction;
  numChildren?: /* int32 */ number;
}

/** An entry in the profile tree. (An instruction, or set of instructions). */
export namespace Node {
  export interface InstructionCategory {}
  export interface XLAInstruction {
    op?: string;
    expression?: string;
    provenance?: string;
    category?: string;
    layout?: Node.XLAInstruction.LayoutAnalysis;
    computationPrimitiveSize?: /* uint32 */ number;
    programId?: /* uint64 */ string;
    sourceInfo?: SourceInfo;
    xprofKernelMetadata?: string;
  }
  export namespace XLAInstruction {
    export interface LayoutAnalysis {
      dimensions?: Node.XLAInstruction.LayoutAnalysis.Dimension[];
    }
    export namespace LayoutAnalysis {
      export interface Dimension {
        size?: /* int32 */ number;
        alignment?: /* int32 */ number;
        semantics?: string;
      }
    }
  }
}

/**
 *  Measurements of an operation (or aggregated set of operations).
 *  Metrics are always "total" rather than "self".
 */
export interface Metrics {
  flops?: /* double */ number;
  uncappedFlops?: /* double */ number;
  bf16Flops?: /* double */ number;
  bandwidthUtils?: /* double */ number[];
  rawTime?: /* double */ number;
  normalizedTimePs?: /* double */ number;
  rawFlops?: /* double */ number;
  rawBytesAccessedArray?: /* double */ number[];
  occurrences?: /* uint32 */ number;
  avgTimePs?: /* double */ number;
}
