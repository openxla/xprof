
/** The base interface for a heap object. */
export interface HeapObject {
  instructionName?: string;
  logicalBufferId?: number;
  unpaddedSizeMiB?: number;
  tfOpName?: string;
  opcode?: string;
  sourceInfo?: string;
  sizeMiB?: number;
  color?: number;
  shape?: string;
  groupName?: string;
}
