/**
 * Utilities for pretty-printing HLO instructions shown in the trace viewer v2
 * event details JSON tree.
 *
 * The "Start Stack Trace" arg of an XLA op holds the full HLO expression as a
 * single, very long line, e.g.
 * `%x = (bf16[...], f32[...]) custom-call(bf16[...] %a, ...),
 * operand_layout_constraints={...}, frontend_attributes={...}`.
 * Rendering it verbatim is hard to read, so we reformat it by expanding operand
 * lists and attribute dictionaries onto multiple indented lines, while keeping
 * shapes and layouts (e.g. `f32[256,8,128]{2,1,0}`, `T(8,128)`) inline.
 *
 * The formatter is a pure string function and only ever adds line breaks,
 * indentation and (for `(`) a separating space, so it cannot corrupt the
 * expression. Lines that do not look like HLO instructions are returned
 * unchanged by `prettyPrintHloStackTrace`.
 */

/** Indentation unit used for each nesting level of pretty-printed HLO. */
const INDENT = '  ';

/** A pending bracket group tracked on the parser stack. */
interface BracketFrame {
  /** The opening bracket character (`(`, `[` or `{`). */
  bracket: string;
  /** Whether the group is expanded across multiple lines. */
  expanded: boolean;
}

/** Returns the last non-space character in `s` before `pos` (or ''). */
function previousNonSpace(s: string, pos: number): string {
  let j = pos - 1;
  while (j >= 0 && s[j] === ' ') j--;
  return j >= 0 ? s[j] : '';
}

/**
 * Returns whether the bracket `ch` at `pos` opens an expandable group (an
 * operand list or attribute dictionary) rather than an inline construct such as
 * a tile spec `T(8,128)` or a layout `{2,1,0}` that immediately follows a shape.
 */
function opensExpandableGroup(s: string, ch: string, pos: number): boolean {
  const prev = previousNonSpace(s, pos);
  if (ch === '(') return !/[A-Z\]})]/.test(prev);
  if (ch === '{') return !/[\]})]/.test(prev);
  return false;
}

/**
 * Returns the index of the closing bracket matching the opener at `pos`,
 * ignoring brackets inside double-quoted strings.
 */
function closeIndex(s: string, pos: number): number {
  const open = s[pos];
  const close = open === '(' ? ')' : '}';
  let depth = 1;
  let j = pos + 1;
  let inString = false;
  while (j < s.length && depth > 0) {
    if (s[j] === '"' && !inString) inString = true;
    else if (s[j] === '"' && inString && s[j - 1] !== '\\') inString = false;
    else if (!inString) {
      if (s[j] === open) depth++;
      else if (s[j] === close) depth--;
    }
    if (depth > 0) j++;
  }
  return j;
}

/**
 * Pretty-prints a single HLO instruction, expanding operand lists and attribute
 * dictionaries onto multiple indented lines. Shapes (`[...]`) and layouts
 * (`{...}` following a shape) always stay inline.
 */
export function prettyPrintHloOp(input: string): string {
  const s = input.trim().replace(/\s+/g, ' ');
  const stack: BracketFrame[] = [];
  let result = '';
  let i = 0;
  let depth = 0;

  while (i < s.length) {
    const ch = s[i];

    // Copy string literals verbatim.
    if (ch === '"') {
      let j = i + 1;
      while (j < s.length) {
        if (s[j] === '\\') {
          j += 2;
          continue;
        }
        if (s[j] === '"') break;
        j++;
      }
      result += s.substring(i, j + 1);
      i = j + 1;
      continue;
    }

    // Shapes in square brackets are always kept inline.
    if (ch === '[') {
      stack.push({bracket: '[', expanded: false});
      result += ch;
      i++;
      continue;
    }
    if (ch === ']') {
      while (stack.length && stack[stack.length - 1].bracket !== '[') {
        stack.pop();
      }
      if (stack.length) stack.pop();
      result += ch;
      i++;
      continue;
    }

    // Inside a non-expanded group everything is copied inline.
    if (stack.length && !stack[stack.length - 1].expanded) {
      if (ch === '(' || ch === '{') {
        stack.push({bracket: ch, expanded: false});
      } else if (ch === ')' || ch === '}') {
        if (stack.length) stack.pop();
      }
      result += ch;
      i++;
      continue;
    }

    // Opening bracket at an expandable position.
    if (ch === '(' || ch === '{') {
      if (opensExpandableGroup(s, ch, i)) {
        const close = closeIndex(s, i);
        // Keep empty groups inline, e.g. `custom-call()`.
        if (s.substring(i + 1, close).trim() === '') {
          result += ch + s[close];
          i = close + 1;
          continue;
        }
        depth++;
        stack.push({bracket: ch, expanded: true});
        // Separate an opcode from its expanded operand list, e.g. `fusion (`.
        if (ch === '(' && /[a-z\-_]/.test(previousNonSpace(s, i))) result += ' ';
        result += ch + '\n' + INDENT.repeat(depth);
        i++;
        while (i < s.length && s[i] === ' ') i++;
        continue;
      }
      stack.push({bracket: ch, expanded: false});
      result += ch;
      i++;
      continue;
    }

    // Closing bracket.
    if (ch === ')' || ch === '}') {
      if (stack.length && stack[stack.length - 1].expanded) {
        depth--;
        stack.pop();
        result = result.replace(/\s+$/, '');
        result += '\n' + INDENT.repeat(depth) + ch;
        i++;
        continue;
      }
      if (stack.length) stack.pop();
      result += ch;
      i++;
      continue;
    }

    // Commas separate operands / attributes.
    if (ch === ',') {
      const top = stack.length ? stack[stack.length - 1] : undefined;
      if (top && !top.expanded) {
        result += ch;
        i++;
        continue;
      }
      if ((top && top.expanded) || depth === 0) {
        result += top ? ',\n' + INDENT.repeat(depth) : ',\n';
        i++;
        while (i < s.length && s[i] === ' ') i++;
        continue;
      }
      result += ch;
      i++;
      continue;
    }

    result += ch;
    i++;
  }
  return result;
}

/**
 * Pretty-prints the value of the "Start Stack Trace" arg. The value may contain
 * several stack frames joined by newlines; for XLA ops each frame holds a full
 * (single-line) HLO expression. Each HLO line is formatted independently and
 * lines that do not look like HLO instructions are returned unchanged.
 */
export function prettyPrintHloStackTrace(value: string): string {
  return value
    .split('\n')
    .map((line) => (line.includes(' = ') ? prettyPrintHloOp(line) : line))
    .join('\n');
}

/**
 * Key of the arg that holds the HLO stack trace. Must stay in sync with
 * `STACK_TRACE_ARG_KEY` in
 * //third_party/xprof/frontend/app/components/trace_viewer/utils.ts.
 */
const STACK_TRACE_ARG_KEY = 'Start Stack Trace';

/**
 * Returns a shallow copy of `args` with the HLO stack trace value pretty-printed
 * for display in the JSON tree. Returns the original `args` reference unchanged
 * when there is no string stack trace to format, so no allocation happens for
 * events without an HLO stack trace.
 */
export function formatHloArgsForJsonTree(
  args: Record<string, unknown>,
): Record<string, unknown> {
  const stackTrace = args[STACK_TRACE_ARG_KEY];
  if (typeof stackTrace !== 'string') {
    return args;
  }
  return {
    ...args,
    [STACK_TRACE_ARG_KEY]: prettyPrintHloStackTrace(stackTrace),
  };
}
