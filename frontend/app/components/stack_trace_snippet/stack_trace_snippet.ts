import {Component, Input, OnChanges, SimpleChanges} from '@angular/core';
import {Address} from 'org_xprof/frontend/app/services/source_code_service/source_code_service_interface';

/**
 * A component to display a snippet of source code corresponding to a given
 * stack trace.
 */
@Component({
  standalone: false,
  selector: 'stack-trace-snippet',
  templateUrl: './stack_trace_snippet.ng.html',
  styleUrls: ['./stack_trace_snippet.scss'],
})
export class StackTraceSnippet implements OnChanges {
  /**
   * The source location of the HLO operation.
   *
   * This is a string representation of the first frame of the stack trace.
   * Sometimes the stack trace is not available, but the source location is
   * available. Whenever that is the case, we treat this input as a stack-trace
   * with a single frame and use it instead of the `stackTrace` input.
   *
   * The expected format is the same as each line in the stack trace string.
   * Example:
   *
   *   /full/path/to/file.py:100
   */
  @Input() sourceFileAndLineNumber: string|undefined = undefined;
  @Input() stackTrace: string|undefined = undefined;
  /**
   * The number of lines to show around the stack frame.
   */
  @Input() sourceContextWindow = 40;
  sourceCodeSnippetAddresses: readonly Address[] = [];

  ngOnChanges(changes: SimpleChanges) {
    if (changes['sourceFileAndLineNumber'] || changes['stackTrace']) {
      this.parseAddresses();
    }
  }

  trackByIndex(index: number, item: Address): number {
    return index;
  }

  /**
   * Returns true if the `stackTrace` is not available, but the
   * `sourceFileAndLineNumber` is available.
   */
  get usingSourceFileAndLineNumber(): boolean {
    return !this.stackTrace && !!this.sourceFileAndLineNumber;
  }

  private parseAddresses() {
    this.sourceCodeSnippetAddresses = parseAddresses(
        this.stackTrace || this.sourceFileAndLineNumber || '',
        this.sourceContextWindow);
  }
}

/**
 * Parses a stack trace string into a list of stack frame addresses.
 *
 * Each line in the stack trace string is expected to be in the following
 * format.
 *
 * [<white_space>]<file>:<line>[:<column>][<white_space>]
 *
 * Example:
 *
 *   /usr/local/google/home/user/src/main.cc:100:5
 */
function parseAddresses(value: string, sourceContextWindow = 20): Address[] {
  const result: Address[] = [];
  const linesBefore = sourceContextWindow / 2;
  const linesAfter = sourceContextWindow / 2;
  const framePattern = /^\s*([^:]+?)(?:\:(-?\d+))?(?:\:-?\d+)?\s*$/;
  const lines = value.trim().split('\n');
  for (const line of lines) {
    const match = line.match(framePattern);
    if (match) {
      const fileName = match[1];
      const lineNumberStr = match[2];
      let lineNumber = Number(lineNumberStr);
      // Since in the current implementation, we only show a few lines around
      // the line number, we need `lineNumber` to be valid. In fact, `Address`
      // constructor throws an error if `lineNumber` is not positive.
      //
      // An alternative implementation is to show top of the file and let the
      // user scroll down. Since I've not yet seen a case where the file name
      // is valid but the line number is invalid, I've opted for the current
      // implementation which is simpler.
      if (!isNaN(lineNumber) && lineNumber > 0) {
        lineNumber = Math.floor(lineNumber);
        result.push(new Address(fileName, lineNumber, linesBefore, linesAfter));
      }
    }
  }
  return result;
}
