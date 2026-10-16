import {
  Address,
  Content,
} from 'org_xprof/frontend/app/services/source_code_service/source_code_service_interface';

/** Definitions for the current compiler pass used to generate IR. */
export enum CompilerPass {
  HLO_OPTIMIZED = 'HLO(optimized)',
  HLO_UNOPTIMIZED = 'HLO(original)',
  MOSAIC_ORIGINAL = 'Mosaic(original)',
}

/**
 * Maximum line prefix length inspected when searching for HLO op definitions
 * (`<op> =`) to build the op-name index. HLO op names on the left-hand side of
 * an assignment always appear near the start of a line, so capping the search
 * prefix avoids scanning 100K-char lines without affecting displayed IR text.
 */
const OP_NAME_SEARCH_MAX_LINE_PREFIX_LENGTH = 256;

/**
 * Encapsulates an IR text document (HLO or Mosaic MLIR) for a specific compiler
 * pass, providing lazy line splitting and an O(1) operation-to-line index.
 */
export class IrText {
  private linesCache?: string[];
  private opLineIndexCache?: Map<string, number>;
  private firstKernelLineIndexCache?: number;

  constructor(
    readonly text: string,
    readonly compilerPass: CompilerPass = CompilerPass.HLO_OPTIMIZED,
  ) {}

  get length(): number {
    return this.text.length;
  }

  /** Lazily splits and caches the lines of the IR text. */
  get lines(): string[] {
    if (!this.linesCache) {
      this.linesCache = this.text.split('\n');
    }
    return this.linesCache;
  }

  /**
   * Returns the 0-based focus line index for the given op name, or 0 if not
   * found.
   */
  getFocusLineIndex(opName: string): number {
    switch (this.compilerPass) {
      case CompilerPass.HLO_OPTIMIZED: {
        if (!opName) return 0;
        return this.opLineIndex.get(opName) ?? 0;
      }
      case CompilerPass.MOSAIC_ORIGINAL: {
        if (this.firstKernelLineIndexCache === undefined) {
          // Assumptions: the MLIR text contains the keyword 'kernel' in the
          // kernel definition line.
          const index = this.lines.findIndex((line) => line.includes('kernel'));
          this.firstKernelLineIndexCache = index !== -1 ? index : 0;
        }
        return this.firstKernelLineIndexCache;
      }
      case CompilerPass.HLO_UNOPTIMIZED:
      default:
        return 0;
    }
  }

  /**
   * Lazily builds and caches a map from HLO operation name to its 0-based
   * definition line index in a single pass over `this.lines`.
   */
  private get opLineIndex(): Map<string, number> {
    if (this.opLineIndexCache) {
      return this.opLineIndexCache;
    }
    const index = new Map<string, number>();
    const lines = this.lines;
    for (let i = 0; i < lines.length; i++) {
      const line = lines[i];
      if (!line) continue;
      const prefix =
        line.length > OP_NAME_SEARCH_MAX_LINE_PREFIX_LENGTH
          ? line.slice(0, OP_NAME_SEARCH_MAX_LINE_PREFIX_LENGTH)
          : line;
      const eqIdx = prefix.indexOf(' =');
      if (eqIdx > 0) {
        let start = eqIdx - 1;
        // Walk backwards over the op identifier, stopping at whitespace or '%'.
        while (
          start >= 0 &&
          prefix.charCodeAt(start) > 32 &&
          prefix.charCodeAt(start) !== 37
        ) {
          start--;
        }
        const opName = prefix.slice(start + 1, eqIdx);
        if (opName && !index.has(opName)) {
          index.set(opName, i);
        }
      }
    }
    this.opLineIndexCache = index;
    return index;
  }

  /**
   * Returns the windowed lines around the focus line for display.
   */
  getLinesForDisplay(opName: string, contextWindow: number): string[] {
    // Return the entire HLO text for HLO unoptimized pass, as the mapping from
    // the selected HLO op to its original text line is not implemented yet.
    if (this.compilerPass === CompilerPass.HLO_UNOPTIMIZED) {
      return this.lines;
    }
    const focusIndex = this.getFocusLineIndex(opName);
    const minLineIndex = Math.max(0, focusIndex - contextWindow / 2);
    const maxLineIndex = Math.min(
      this.lines.length - 1,
      focusIndex + contextWindow / 2,
    );
    return this.lines.slice(minLineIndex, maxLineIndex + 1);
  }

  /**
   * Returns the 1-based focus line number within the displayed window.
   */
  getFocusLineNumberForDisplay(opName: string, contextWindow: number): number {
    const focusIndex = this.getFocusLineIndex(opName);
    const minLineIndex = Math.max(0, focusIndex - contextWindow / 2);
    return focusIndex - minLineIndex + 1;
  }

  /**
   * Returns the `Content` frame for `<source-code-editor>`.
   */
  getDisplayFrame(opName: string, contextWindow: number): Content | undefined {
    const lines = this.getLinesForDisplay(opName, contextWindow);
    const focusLineNumber = this.getFocusLineNumberForDisplay(
      opName,
      contextWindow,
    );
    const address = new Address(
      'ir_text',
      focusLineNumber,
      focusLineNumber - 1,
      Math.max(0, lines.length - focusLineNumber),
    );
    return new Content(address, lines, []);
  }
}
