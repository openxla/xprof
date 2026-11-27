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
 * Extracts the trailing numeric program ID from an HLO module name formatted as
 * `<name>(<program_id>)`, or returns an empty string if absent.
 */
export function extractProgramIdFromHloModuleName(moduleName?: string): string {
  const match = (moduleName ?? '').trim().match(/\((\d+)\)$/);
  return match ? match[1] : '';
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
 * pass, providing lazy line splitting, an O(1) operation-to-line index, and
 * cached display frames.
 */
export class IrText {
  private linesCache?: string[];
  private opLineIndexCache?: Map<string, number>;
  private firstKernelLineIndexCache?: number;
  private lastFrameFocusIndex?: number;
  private lastContextWindow?: number;
  private cachedFrame?: Content;

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
    return this.findFocusLineIndex(opName) ?? 0;
  }

  /**
   * Returns the 0-based focus line index for the given op name, or undefined
   * when no line is mapped for the current pass or operation.
   */
  private findFocusLineIndex(opName: string): number | undefined {
    switch (this.compilerPass) {
      case CompilerPass.HLO_OPTIMIZED: {
        if (!opName) return undefined;
        return this.opLineIndex.get(opName);
      }
      case CompilerPass.MOSAIC_ORIGINAL: {
        if (this.firstKernelLineIndexCache === undefined) {
          // Assumptions: the MLIR text contains the keyword 'kernel' in the
          // kernel definition line.
          this.firstKernelLineIndexCache = this.lines.findIndex((line) =>
            line.includes('kernel'),
          );
        }
        return this.firstKernelLineIndexCache !== -1
          ? this.firstKernelLineIndexCache
          : undefined;
      }
      case CompilerPass.HLO_UNOPTIMIZED:
      default:
        return undefined;
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
    const halfWindow = Math.floor(Math.max(0, contextWindow) / 2);
    const minLineIndex = Math.max(0, focusIndex - halfWindow);
    const maxLineIndex = Math.min(
      this.lines.length - 1,
      focusIndex + halfWindow,
    );
    return this.lines.slice(minLineIndex, maxLineIndex + 1);
  }

  /**
   * Returns the 1-based file line number of the first line in the displayed
   * window.
   */
  getFirstLineNumberForDisplay(opName: string, contextWindow: number): number {
    if (this.compilerPass === CompilerPass.HLO_UNOPTIMIZED) {
      return 1;
    }
    const focusIndex = this.getFocusLineIndex(opName);
    const halfWindow = Math.floor(Math.max(0, contextWindow) / 2);
    return Math.max(0, focusIndex - halfWindow) + 1;
  }

  /**
   * Returns the 1-based absolute focus line number in the IR file, or
   * undefined when the operation is not mapped to a specific line.
   */
  getFocusLineNumberForDisplay(opName: string): number | undefined {
    const focusIndex = this.findFocusLineIndex(opName);
    return focusIndex !== undefined ? focusIndex + 1 : undefined;
  }

  /**
   * Returns the cached `Content` frame for `<source-code-editor>`, preserving
   * object identity across Angular change detection cycles.
   */
  getDisplayFrame(opName: string, contextWindow: number): Content | undefined {
    const focusIndex = this.getFocusLineIndex(opName);
    if (
      this.lastFrameFocusIndex !== focusIndex ||
      this.lastContextWindow !== contextWindow ||
      !this.cachedFrame
    ) {
      this.lastFrameFocusIndex = focusIndex;
      this.lastContextWindow = contextWindow;
      const lines = this.getLinesForDisplay(opName, contextWindow);
      const firstLineNumber = this.getFirstLineNumberForDisplay(
        opName,
        contextWindow,
      );
      const focusLineNumber =
        this.getFocusLineNumberForDisplay(opName) ?? firstLineNumber;
      const linesBefore = focusLineNumber - firstLineNumber;
      const linesAfter = Math.max(0, lines.length - 1 - linesBefore);
      const address = new Address(
        'ir_text',
        focusLineNumber,
        linesBefore,
        linesAfter,
      );
      this.cachedFrame = new Content(address, lines, []);
    }
    return this.cachedFrame;
  }
}
