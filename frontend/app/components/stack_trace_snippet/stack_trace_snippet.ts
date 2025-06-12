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
  @Input() stackTrace: string|undefined = undefined;
  addresses: readonly Address[] = [];

  ngOnChanges(changes: SimpleChanges) {
    if (changes['stackTrace']) {
      this.stackTraceChanged();
    }
  }

  private stackTraceChanged() {
    this.addresses = parseAddresses(this.stackTrace || '');
  }
}

/**
 * Parses a stack trace string into a list of stack frame addresses.
 *
 * Each line in the stack trace string is expected to be in the following
 * format:
 *
 * [<white_space>]<file>[:<line>[:<column>]][<white_space>]
 *
 * Example:
 *
 *   /usr/local/google/home/user/src/main.cc:100:5
 */
function parseAddresses(value: string): Address[] {
  const result: Address[] = [];
  const linesBefore = 5;
  const linesAfter = 5;
  const framePattern = /^\s*(.+?)(?:\:(\d+))?(?:\:\d+)?\s*$/;
  const lines = value.trim().split('\n');
  for (const line of lines) {
    const match = line.match(framePattern);
    if (match) {
      const fileName = match[1];
      const lineNumberStr = match[2];
      let lineNumber = Number(lineNumberStr);
      if (isNaN(lineNumber)) {
        lineNumber = -1;
      } else {
        lineNumber = Math.floor(lineNumber);
      }
      result.push(new Address(fileName, lineNumber, linesBefore, linesAfter));
    }
  }
  return result;
}
