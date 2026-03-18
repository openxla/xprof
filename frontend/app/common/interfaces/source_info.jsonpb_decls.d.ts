/** The base interface for a source information. */
export interface SourceInfo {
  fileName?: string;
  lineNumber?: /* int32 */ number;
  stackFrame?: string;
}
