import {
  ChangeDetectionStrategy,
  Component,
  DestroyRef,
  effect,
  ElementRef,
  inject,
  input,
  output,
  signal,
  untracked,
  viewChild,
} from '@angular/core';
import {
  formatCountOf,
  type LineGutter,
} from 'org_xprof/frontend/app/components/static_kernel_viewer/kernel_source_model';

/** A line of the source list. */
export interface SourceLinesRow {
  /** 1-based line number. */
  readonly line: number;
  /** Code of the line as HTML, or '' when only the line's stats are known. */
  readonly html: string;
  /** Share of the schedule of the line, if it has instructions. */
  readonly gutter?: LineGutter;
  /** Pallas primitives of the line, listed instead of missing code. */
  readonly primitives?: readonly string[];
  readonly instructions?: number;
}

/** Bounds of a line on the screen, in viewport coordinates. */
export interface SourceLinesBounds {
  readonly top: number;
  readonly bottom: number;
  readonly left: number;
  readonly right: number;
}

/** The line under the pointer, and where it is on the screen. */
export interface SourceLinesHover {
  readonly line: number;
  readonly bounds: SourceLinesBounds;
  /** Whether the pointer is over the gutter, rather than the code. */
  readonly inGutter: boolean;
  /** Horizontal position of the pointer, in viewport coordinates. */
  readonly x: number;
}

/** Asks the list to scroll a line into view. */
export interface SourceLinesReveal {
  readonly line: number;
  /** Distinguishes repeated requests for the same line. */
  readonly nonce: number;
  /** Whether to flash the line, e.g. after a timeline selection. */
  readonly flash: boolean;
}

/** How long a line flashes after it is revealed. */
const FLASH_MS = 1600;

/** Height of a row of the list, in pixels; see the stylesheet. */
const ROW_HEIGHT = 20;

/**
 * A plain list of source lines with the share of the schedule of each line.
 * It shows the lines that have instructions when the code itself is not
 * available, and the code where the Monaco editor is not available.
 */
@Component({
  standalone: true,
  selector: 'kernel-source-lines',
  templateUrl: './kernel_source_lines.ng.html',
  styleUrls: ['./kernel_source_lines.scss'],
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class KernelSourceLines {
  readonly rows = input<readonly SourceLinesRow[]>([]);
  /** Whether the rows list only the lines with instructions, without code. */
  readonly linesOnly = input(false);
  readonly selectedLine = input<number | undefined>(undefined);
  readonly hoveredLine = input<number | undefined>(undefined);
  readonly reveal = input<SourceLinesReveal | undefined>(undefined);
  readonly ariaLabel = input('Lines of the HLO op source');

  readonly lineHover = output<SourceLinesHover | undefined>();
  readonly lineClick = output<number>();
  /** Emits on Escape, to clear the selected line. */
  readonly clear = output<void>();

  /** Line of the keyboard cursor. */
  protected readonly cursorLine = signal<number | undefined>(undefined);
  protected readonly flashLine = signal<number | undefined>(undefined);
  protected readonly formatCountOf = formatCountOf;

  private readonly list = viewChild.required<ElementRef<HTMLElement>>('list');
  private lastRevealNonce = -1;
  /** Line and zone (gutter or code) under the pointer, e.g. `12:code`. */
  private hoverKey = '';
  private flashTimer?: ReturnType<typeof setTimeout>;
  private scrollTimer?: ReturnType<typeof setTimeout>;

  constructor() {
    effect(() => {
      const reveal = this.reveal();
      const rows = this.rows();
      untracked(() => {
        if (
          !reveal ||
          reveal.nonce === this.lastRevealNonce ||
          !rows.some((row) => row.line === reveal.line)
        ) {
          return;
        }
        this.lastRevealNonce = reveal.nonce;
        this.cursorLine.set(reveal.line);
        this.scrollToLine(reveal.line);
        if (reveal.flash) {
          this.flashLine.set(reveal.line);
          clearTimeout(this.flashTimer);
          this.flashTimer = setTimeout(() => {
            this.flashLine.set(undefined);
          }, FLASH_MS);
        }
      });
    });

    inject(DestroyRef).onDestroy(() => {
      clearTimeout(this.flashTimer);
      clearTimeout(this.scrollTimer);
    });
  }

  protected onRowMove(row: SourceLinesRow, event: MouseEvent): void {
    const target = event.currentTarget;
    if (!row.gutter || event.buttons !== 0 || !(target instanceof Element)) {
      this.onListLeave();
      return;
    }
    const gutter = target.querySelector('.gutter')?.getBoundingClientRect();
    const inGutter =
      gutter !== undefined &&
      event.clientX >= gutter.left &&
      event.clientX <= gutter.right;
    const key = `${row.line}:${inGutter ? 'gutter' : 'code'}`;
    if (key === this.hoverKey) {
      return;
    }
    this.hoverKey = key;
    const rect = target.getBoundingClientRect();
    const list = this.list().nativeElement.getBoundingClientRect();
    this.lineHover.emit({
      line: row.line,
      bounds: {
        top: rect.top,
        bottom: rect.bottom,
        left: list.left,
        right: list.right,
      },
      inGutter,
      x: event.clientX,
    });
  }

  protected onListLeave(): void {
    if (this.hoverKey === '') {
      return;
    }
    this.hoverKey = '';
    this.lineHover.emit(undefined);
  }

  protected onRowClick(row: SourceLinesRow): void {
    this.cursorLine.set(row.line);
    this.lineClick.emit(row.line);
  }

  /** Handles keys of the focused list, like a list box. */
  protected onKeydown(event: KeyboardEvent): void {
    const rows = this.rows();
    if (rows.length === 0 || event.ctrlKey || event.metaKey || event.altKey) {
      return;
    }
    const current = this.cursorLine() ?? this.selectedLine();
    let index = rows.findIndex((row) => row.line === current);
    const page = Math.max(
      1,
      Math.floor(this.list().nativeElement.clientHeight / ROW_HEIGHT) - 1,
    );
    switch (event.key) {
      case 'ArrowDown':
        index = index < 0 ? 0 : Math.min(rows.length - 1, index + 1);
        break;
      case 'ArrowUp':
        index = index < 0 ? 0 : Math.max(0, index - 1);
        break;
      case 'PageDown':
        index = Math.min(rows.length - 1, Math.max(0, index) + page);
        break;
      case 'PageUp':
        index = Math.max(0, index - page);
        break;
      case 'Home':
        index = 0;
        break;
      case 'End':
        index = rows.length - 1;
        break;
      case 'Enter':
      case ' ':
        if (current !== undefined) {
          event.preventDefault();
          event.stopPropagation();
          this.lineClick.emit(current);
        }
        return;
      case 'Escape':
        if (this.selectedLine() !== undefined) {
          event.stopPropagation();
          this.clear.emit();
        }
        return;
      default:
        return;
    }
    // Keeps the keys from also panning the timeline.
    event.preventDefault();
    event.stopPropagation();
    const line = rows[index]?.line;
    if (line !== undefined) {
      this.cursorLine.set(line);
      this.scrollToLine(line, false);
    }
  }

  protected rowLabel(row: SourceLinesRow): string {
    const parts = [`Line ${row.line}`];
    if (row.gutter) {
      parts.push(`${row.gutter.shareLabel} of the schedule`);
    }
    if (row.instructions) {
      parts.push(formatCountOf(row.instructions, 'instruction'));
    }
    return parts.join(', ');
  }

  private scrollToLine(line: number, center = true): void {
    clearTimeout(this.scrollTimer);
    // Rows render once the inputs they depend on settle.
    this.scrollTimer = setTimeout(() => {
      const list = this.list().nativeElement;
      const row = list.querySelector<HTMLElement>(`[data-line="${line}"]`);
      if (!row) {
        return;
      }
      const top = row.offsetTop;
      const bottom = top + row.offsetHeight;
      const margin = ROW_HEIGHT * 2;
      if (
        top >= list.scrollTop + margin &&
        bottom <= list.scrollTop + list.clientHeight - margin
      ) {
        return;
      }
      list.scrollTo({
        top: center
          ? Math.max(0, top - list.clientHeight / 3)
          : top < list.scrollTop + margin
            ? Math.max(0, top - margin)
            : bottom - list.clientHeight + margin,
        behavior: center ? 'smooth' : 'auto',
      });
    });
  }
}
