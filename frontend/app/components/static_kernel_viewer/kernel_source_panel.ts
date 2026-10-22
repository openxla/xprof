import {NgTemplateOutlet} from '@angular/common';
import {
  ChangeDetectionStrategy,
  Component,
  computed,
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
import {MatButtonModule} from '@angular/material/button';
import {MatDividerModule} from '@angular/material/divider';
import {MatIconModule} from '@angular/material/icon';
import {MatMenuModule} from '@angular/material/menu';
import {MatProgressBarModule} from '@angular/material/progress-bar';
import {MatTooltipModule} from '@angular/material/tooltip';
import {
  KernelLineCard,
  SHARE_DESCRIPTION,
  type StackFrameView,
} from 'org_xprof/frontend/app/components/static_kernel_viewer/kernel_line_card';
import {
  KernelSourceLines,
  type SourceLinesHover,
  type SourceLinesRow,
} from 'org_xprof/frontend/app/components/static_kernel_viewer/kernel_source_lines';
import {
  describeSourceVersion,
  escapeHtml,
  fileGutters,
  findSourceFile,
  findSourceLine,
  formatCount,
  formatCountOf,
  formatFraction,
  formatShare,
  hotspots,
  isSamePath,
  type KernelSourceSummary,
  type RunEnvironment,
  type SourceFileStats,
  type SourceLineStats,
  type SourceLocation,
  splitPath,
  stripMarkup,
  toDepotPath,
} from 'org_xprof/frontend/app/components/static_kernel_viewer/kernel_source_model';
import {
  Address,
  SOURCE_CODE_SERVICE_INTERFACE_TOKEN,
} from 'org_xprof/frontend/app/services/source_code_service/source_code_service_interface';
import {Subscription} from 'rxjs';

/** Loading state of the source attribution of a kernel. */
export type SummaryState = 'loading' | 'ready' | 'error' | 'unsupported';

/** Asks the panel to reveal a source location, e.g. of a timeline selection. */
export interface SourceRevealRequest {
  readonly location: SourceLocation;
  /** Distinguishes repeated requests for the same location. */
  readonly nonce: number;
}

/** A range of bundles of the schedule, `end` exclusive. */
export interface BundleRange {
  readonly start: number;
  readonly end: number;
}

/** The selected source line, as the timeline needs it. */
export interface SourceSelection {
  /** Timeline search query that matches exactly the line's instructions. */
  readonly query: string;
  /** Bundles that hold the line's instructions. */
  readonly range: BundleRange;
  /** Short name of the line, e.g. `flash_attention.py:51`. */
  readonly label: string;
}

/** Position of the highlighted instruction among those of the line. */
export interface SearchPosition {
  /** 0-based index of the highlighted instruction, or -1 if none is. */
  readonly index: number;
  readonly count: number;
}

/** Display options of the code. */
export interface SourceViewOptions {
  /** Monaco theme, e.g. `vs` or `vs-dark`. */
  readonly theme: string;
  /** Font size of the code, in pixels. */
  readonly fontSize: number;
  readonly minimap: boolean;
  readonly wordWrap: boolean;
  readonly stickyScroll: boolean;
}

/** A display option of the code that can be turned on and off. */
export type SourceViewToggle = 'minimap' | 'wordWrap' | 'stickyScroll';

/** Built-in actions of the editor that the panel's menu runs. */
type EditorAction =
  | 'actions.find'
  | 'editor.action.gotoLine'
  | 'editor.foldAll'
  | 'editor.unfoldAll';

/** A line of the list of hotspots. */
export interface HotspotItem {
  readonly key: string;
  readonly path: string;
  readonly line: number;
  /** Short name of the line, e.g. `flash_attention.py:51`. */
  readonly label: string;
  /** Trimmed code of the line, if the file is loaded. */
  readonly code: string;
  readonly share: string;
  /** Length of the bar, relative to the busiest line, in percent. */
  readonly barPercent: number;
  readonly segments: ReadonlyArray<{
    readonly color: string;
    readonly percent: number;
  }>;
}

/** The hover card of a line, and where it shows in the panel. */
interface HoverCard {
  readonly stats: SourceLineStats;
  readonly width: number;
  readonly maxHeight: number;
  readonly left: number;
  /** Offset of the top of the card, when it shows below the line. */
  readonly top?: number;
  /** Offset of the bottom of the card, when it shows above the line. */
  readonly bottom?: number;
}

/** The last row of the list of hotspots: the share without a source line. */
interface UnattributedRow {
  readonly share: string;
  /** Length of the bar, relative to the busiest line, in percent. */
  readonly barPercent: number;
  /** Short description, e.g. `4 instructions`. */
  readonly description: string;
  /** Longer description for a tooltip. */
  readonly tooltip: string;
}

/** Content of a source file, as the code view needs it. */
interface FileContent {
  readonly state: 'loading' | 'ready' | 'error' | 'unavailable';
  /** Syntax highlighted lines, for the plain list. */
  readonly html: readonly string[];
  /** Plain lines, for the editor. */
  readonly text: readonly string[];
  /** Why the file could not be loaded. */
  readonly message?: string;
}

/** Where the shown code comes from, for the chip in the header. */
interface SourceOrigin {
  readonly kind: 'local' | 'latest' | 'workspace' | 'changelist';
  readonly label: string;
  readonly icon: string;
  readonly description: string;
}

/** A request to scroll the code view to a line. */
interface ViewReveal {
  readonly path: string;
  readonly line: number;
  readonly nonce: number;
  readonly flash: boolean;
}

/** Monaco themes that the code can show in, and their backgrounds. */
export const SOURCE_THEMES: ReadonlyArray<{
  readonly id: string;
  readonly label: string;
  readonly swatch: string;
}> = [
  {id: 'vs', label: 'Light', swatch: '#fffffe'},
  {id: 'vs-dark', label: 'Dark', swatch: '#1e1e1e'},
  {id: 'hc-light', label: 'High contrast light', swatch: '#ffffff'},
  {id: 'hc-black', label: 'High contrast dark', swatch: '#000000'},
  {id: 'solarized-light', label: 'Solarized light', swatch: '#fdf6e3'},
  {id: 'monokai', label: 'Monokai', swatch: '#272822'},
];

/** Sizes of the text of the code. */
export const TEXT_SIZES: ReadonlyArray<{
  readonly fontSize: number;
  readonly label: string;
}> = [
  {fontSize: 12, label: 'Small'},
  {fontSize: 13, label: 'Default'},
  {fontSize: 15, label: 'Large'},
];

/** Display options of the code that the menu turns on and off. */
const VIEW_TOGGLES: ReadonlyArray<{
  readonly option: SourceViewToggle;
  readonly label: string;
  readonly icon: string;
}> = [
  {option: 'minimap', label: 'Minimap', icon: 'map'},
  {option: 'wordWrap', label: 'Word wrap', icon: 'wrap_text'},
  {option: 'stickyScroll', label: 'Sticky scroll', icon: 'vertical_align_top'},
];

/** Label of the modifier key of keyboard shortcuts. */
const MOD_KEY = /Mac|iPhone|iPad/.test(navigator.userAgent) ? '⌘' : 'Ctrl';

/** Whether the code shows in the Monaco editor, rather than a plain list. */
const HAS_EDITOR = false;

/** Local storage key of the Monaco theme, shared with other code views. */
const THEME_STORAGE_KEY = 'xprof_monaco_editor_theme';

/** Local storage key of the other display options of the code. */
const OPTIONS_STORAGE_KEY = 'xprof_kernel_source_view_options';

/** Local storage key of whether the details of lines are expanded. */
const DETAILS_STORAGE_KEY = 'xprof_kernel_source_details_expanded';

/** Local storage key of the height of the details, in percent of the panel. */
const DETAILS_HEIGHT_STORAGE_KEY = 'xprof_kernel_source_details_height';

/** Height of the details that the user can resize them to, in percent. */
const MIN_DETAILS_HEIGHT_PERCENT = 15;
const MAX_DETAILS_HEIGHT_PERCENT = 75;

/** How far the arrow keys move the details resizer, in percent. */
const DETAILS_HEIGHT_KEY_STEP_PERCENT = 4;

/** Lines to request from the source code service: the whole file. */
const MAX_FILE_LINES = 100_000;

/** Number of lines in the list of hotspots. */
const HOTSPOT_LIST_LENGTH = 6;

/** How long the copy buttons show their confirmation. */
const COPIED_FEEDBACK_MS = 1500;

/** How long the pointer rests on a line before its card shows. */
const HOVER_DELAY_GUTTER_MS = 180;
const HOVER_DELAY_CODE_MS = 650;

/** How long the card stays after the pointer leaves its line or the card. */
const HOVER_HIDE_DELAY_MS = 150;

/** Geometry of the hover card, in pixels; see the line card stylesheet. */
const CARD_WIDTH = 300;
const CARD_MARGIN = 8;
const CARD_GAP = 6;
const CARD_MIN_HEIGHT = 160;

/**
 * Estimates the height of the hover card of `stats`, in pixels, from the
 * sections it shows, so the panel can pick a side of the line where the whole
 * card fits.
 */
function estimateCardHeight(stats: SourceLineStats): number {
  // Header, schedule footprint, footer, and vertical padding and gaps.
  let height = 152;
  if (stats.units.length > 0) {
    height += 37 + Math.floor((stats.units.length - 1) / 3) * 18;
  }
  if (stats.spills + stats.fills > 0) {
    height += 36;
  }
  return height;
}

/** Short labels of the languages of source files, by file extension. */
const LANGUAGE_BADGES: ReadonlyMap<string, string> = new Map([
  ['py', 'PY'],
  ['cc', 'C++'],
  ['cpp', 'C++'],
  ['h', 'C++'],
  ['mlir', 'MLIR'],
  ['ts', 'TS'],
  ['js', 'JS'],
]);

const LOADING_CONTENT: FileContent = {state: 'loading', html: [], text: []};

const DEFAULT_OPTIONS: SourceViewOptions = {
  theme: 'vs',
  fontSize: 13,
  minimap: true,
  wordWrap: false,
  stickyScroll: true,
};

function readStorage(key: string): string | null {
  try {
    return window.localStorage.getItem(key);
  } catch {
    // Storage can be unavailable, e.g. in sandboxed iframes.
    return null;
  }
}

function writeStorage(key: string, value: string): void {
  try {
    window.localStorage.setItem(key, value);
  } catch {
    // Storage can be unavailable, e.g. in sandboxed iframes.
  }
}

function removeStorage(key: string): void {
  try {
    window.localStorage.removeItem(key);
  } catch {
    // Storage can be unavailable, e.g. in sandboxed iframes.
  }
}

/** Reads the display options of the code that the user picked before. */
function readOptions(): SourceViewOptions {
  let stored: Record<string, unknown> = {};
  try {
    const parsed: unknown = JSON.parse(
      readStorage(OPTIONS_STORAGE_KEY) ?? '{}',
    );
    if (typeof parsed === 'object' && parsed !== null) {
      stored = parsed as Record<string, unknown>;
    }
  } catch {
    // Starts over from the defaults if the stored options are malformed.
  }
  const flag = (key: SourceViewToggle): boolean => {
    const value = stored[key];
    return typeof value === 'boolean' ? value : DEFAULT_OPTIONS[key];
  };
  const storedTheme = readStorage(THEME_STORAGE_KEY);
  return {
    theme:
      SOURCE_THEMES.find((theme) => theme.id === storedTheme)?.id ??
      DEFAULT_OPTIONS.theme,
    fontSize:
      TEXT_SIZES.find((size) => size.fontSize === stored['fontSize'])
        ?.fontSize ?? DEFAULT_OPTIONS.fontSize,
    minimap: flag('minimap'),
    wordWrap: flag('wordWrap'),
    stickyScroll: flag('stickyScroll'),
  };
}

function writeOptions(options: SourceViewOptions): void {
  writeStorage(THEME_STORAGE_KEY, options.theme);
  writeStorage(
    OPTIONS_STORAGE_KEY,
    JSON.stringify({
      fontSize: options.fontSize,
      minimap: options.minimap,
      wordWrap: options.wordWrap,
      stickyScroll: options.stickyScroll,
    }),
  );
}

function toRange(stats: SourceLineStats): BundleRange {
  return {start: stats.firstBundle, end: stats.lastBundle + 1};
}

/** Widens a bundle range a bit, so that its edges stay visible. */
function padRange(range: BundleRange): BundleRange {
  const padding = Math.max(2, Math.round((range.end - range.start) * 0.04));
  return {start: Math.max(0, range.start - padding), end: range.end + padding};
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), Math.max(min, max));
}

function clampDetailsHeight(percent: number): number {
  return clamp(percent, MIN_DETAILS_HEIGHT_PERCENT, MAX_DETAILS_HEIGHT_PERCENT);
}

/**
 * Reads the height of the details that the user resized them to, or undefined
 * if they fit their content.
 */
function readDetailsHeight(): number | undefined {
  const stored = readStorage(DETAILS_HEIGHT_STORAGE_KEY);
  const percent = stored === null ? NaN : Number(stored);
  return Number.isFinite(percent) ? clampDetailsHeight(percent) : undefined;
}

function describeLoadError(error: unknown, path: string): string {
  const name = splitPath(path).name;
  if (typeof error === 'object' && error !== null) {
    const status = (error as Record<string, unknown>)['status'];
    if (status === 401 || status === 403) {
      return `You don't have access to ${name} from this XProf server.`;
    }
    if (status === 404) {
      return `${name} wasn't found at the profiled version.`;
    }
  }
  return `Couldn't load ${name}.`;
}

/**
 * Source panel of the Static Kernel Viewer.
 *
 * Shows the source of a kernel next to its static schedule, with the share of
 * the schedule that each line accounts for in a gutter. Hovering a line shows
 * its stats in a card, and selecting a line lights up its instructions on the
 * timeline. Selecting an instruction on the timeline reveals its line.
 */
@Component({
  standalone: true,
  selector: 'kernel-source-panel',
  templateUrl: './kernel_source_panel.ng.html',
  styleUrls: ['./kernel_source_panel.scss'],
  changeDetection: ChangeDetectionStrategy.OnPush,
  imports: [
    KernelLineCard,
    KernelSourceLines,
    MatButtonModule,
    MatDividerModule,
    MatIconModule,
    MatMenuModule,
    MatProgressBarModule,
    MatTooltipModule,
    NgTemplateOutlet,
  ],
})
export class KernelSourcePanel {
  /** Session of the profile. */
  readonly sessionId = input('');
  /** Identifies the kernel, so that the selected line resets with it. */
  readonly kernelKey = input('');
  /** Source attribution of the kernel. */
  readonly summary = input<KernelSourceSummary | undefined>(undefined);
  readonly summaryState = input<SummaryState>('loading');
  /** Which version of the code the profile was built from, if recorded. */
  readonly runEnvironment = input<RunEnvironment | undefined>(undefined);
  /** Asks to reveal the line of a timeline selection. */
  readonly reveal = input<SourceRevealRequest | undefined>(undefined);
  /** Position of the highlighted instruction of the selected line. */
  readonly searchPosition = input<SearchPosition | undefined>(undefined);

  /** Emits the selected line, or undefined once it is cleared. */
  readonly selectionChange = output<SourceSelection | undefined>();
  /** Emits a bundle range to show on the timeline. */
  readonly zoom = output<BundleRange>();
  /** Emits 1 or -1 to highlight the next or previous instruction. */
  readonly step = output<number>();
  /** Emits to load the source attribution again, after an error. */
  readonly retry = output<void>();
  /** Emits to close the panel. */
  readonly closed = output<void>();

  /** The selected line of the shown file, whose instructions are lit. */
  readonly selectedLine = signal<number | undefined>(undefined);

  /** Path of the shown file, or '' for the busiest file. */
  protected readonly filePath = signal('');
  /** The attributed line under the pointer. */
  protected readonly hoveredLine = signal<number | undefined>(undefined);
  protected readonly hoverCard = signal<HoverCard | undefined>(undefined);
  protected readonly viewReveal = signal<ViewReveal | undefined>(undefined);
  protected readonly options = signal<SourceViewOptions>(readOptions());
  protected readonly isDetailsExpanded = signal(
    readStorage(DETAILS_STORAGE_KEY) !== 'false',
  );
  /**
   * Height of the details, in percent of the panel, or undefined while they
   * fit their content.
   */
  protected readonly detailsHeight = signal<number | undefined>(
    readDetailsHeight(),
  );
  /** Whether the user is dragging the details resizer. */
  protected readonly isResizingDetails = signal(false);
  /** Height of the details, in whole percent, for assistive technologies. */
  protected readonly detailsHeightValue = computed(() => {
    const height = this.detailsHeight();
    return height !== undefined ? Math.round(height) : null;
  });
  protected readonly minDetailsHeight = MIN_DETAILS_HEIGHT_PERCENT;
  protected readonly maxDetailsHeight = MAX_DETAILS_HEIGHT_PERCENT;
  protected readonly codeSearchUrl = signal<string | undefined>(undefined);
  protected readonly copied = signal<'' | 'path' | 'location'>('');
  /** Whether the Monaco editor failed to load, so the list shows the code. */
  protected readonly editorFailed = signal(false);

  protected readonly themes = SOURCE_THEMES;
  protected readonly textSizes = TEXT_SIZES;
  protected readonly toggles = VIEW_TOGGLES;
  protected readonly modKey = MOD_KEY;
  protected readonly skeletonWidths = [62, 84, 46, 72, 90, 54, 68, 40];
  protected readonly formatCount = formatCount;
  protected readonly shareDescription = SHARE_DESCRIPTION;

  private readonly fileInput =
    viewChild.required<ElementRef<HTMLInputElement>>('fileInput');
  private readonly panelElement =
    viewChild.required<ElementRef<HTMLElement>>('panel');
  private readonly detailsElement =
    viewChild<ElementRef<HTMLElement>>('details');

  /** Contents of the files of the session, by path. */
  private readonly contents = signal<ReadonlyMap<string, FileContent>>(
    new Map(),
  );
  /** Files that the user opened from disk, by source path. */
  private readonly localFiles = signal<ReadonlyMap<string, FileContent>>(
    new Map(),
  );
  private readonly subscriptions = new Map<string, Subscription>();
  private linkSubscription?: Subscription;
  private copiedTimer?: ReturnType<typeof setTimeout>;
  private hoverTimer?: ReturnType<typeof setTimeout>;
  private lastRevealNonce = -1;
  private viewRevealNonce = 0;
  private lastSessionId = '';
  private lastKernelKey = '';
  /** A line that was just clicked, whose card stays hidden until left. */
  private suppressedHoverLine = 0;
  private isPointerInCard = false;
  /** Where a drag of the details resizer started. */
  private detailsResizeStart?: {y: number; percent: number};
  private readonly detailsResizeMoveListener = (event: PointerEvent) => {
    this.onDetailsResizerPointerMove(event);
  };
  private readonly detailsResizeEndListener = () => {
    this.endDetailsResize();
  };

  private readonly host = inject<ElementRef<HTMLElement>>(ElementRef);
  private readonly sourceCodeService = inject(
    SOURCE_CODE_SERVICE_INTERFACE_TOKEN,
    {optional: true},
  );

  protected readonly files = computed<readonly SourceFileStats[]>(
    () => this.summary()?.files ?? [],
  );

  /** Path of the shown file. */
  protected readonly resolvedPath = computed(() => {
    const requested = this.filePath();
    const files = this.files();
    if (requested && files.some((file) => isSamePath(file.path, requested))) {
      return requested;
    }
    return files[0]?.path ?? '';
  });

  protected readonly file = computed(() =>
    findSourceFile(this.summary(), this.resolvedPath()),
  );
  protected readonly fileName = computed(
    () => splitPath(this.resolvedPath()).name,
  );
  protected readonly fileDirectory = computed(
    () => splitPath(this.resolvedPath()).directory,
  );

  /** Short language label of the shown file, e.g. `PY`. */
  protected readonly fileBadge = computed(() => {
    const name = this.fileName();
    const dot = name.lastIndexOf('.');
    const extension = dot > 0 ? name.slice(dot + 1).toLowerCase() : '';
    return (
      LANGUAGE_BADGES.get(extension) ??
      (extension.slice(0, 3).toUpperCase() || '</>')
    );
  });

  protected readonly content = computed<FileContent>(() => {
    const path = this.resolvedPath();
    return (
      this.localFiles().get(path) ??
      this.contents().get(path) ??
      LOADING_CONTENT
    );
  });

  protected readonly isLocalCopy = computed(() =>
    this.localFiles().has(this.resolvedPath()),
  );

  /** Where the shown code comes from, for the chip in the header. */
  protected readonly origin = computed<SourceOrigin | undefined>(() => {
    if (this.isLocalCopy()) {
      return {
        kind: 'local',
        label: 'Local copy',
        icon: 'upload_file',
        description:
          `${this.fileName()} from your computer, which may differ from ` +
          'the code that was profiled.',
      };
    }
    const version = describeSourceVersion(this.runEnvironment());
    if (version) {
      return {
        kind: version.kind,
        label: version.label,
        icon: version.kind === 'workspace' ? 'cloud_sync' : 'commit',
        description: version.description,
      };
    }
    if (this.content().state === 'ready') {
      return {
        kind: 'latest',
        label: 'Latest',
        icon: 'update',
        description:
          "This profile doesn't record which version of the code it ran, " +
          'so this is the latest version of the file.',
      };
    }
    return undefined;
  });

  /** Whether the code is unavailable, so only lines with instructions show. */
  protected readonly linesOnly = computed(() => {
    const state = this.content().state;
    return state === 'error' || state === 'unavailable';
  });

  /** Whether the code shows in the Monaco editor. */
  protected readonly showsEditor = computed(
    () => HAS_EDITOR && !this.editorFailed(),
  );

  protected readonly gutters = computed(() => fileGutters(this.file()));

  /** Content of the shown file, for the editor. */
  protected readonly text = computed(() => this.content().text.join('\n'));

  /** Rows of the plain list of the code. */
  protected readonly codeRows = computed<readonly SourceLinesRow[]>(() => {
    const gutters = this.gutters();
    const file = this.file();
    return this.content().html.map((html, index) => {
      const line = index + 1;
      const gutter = gutters.get(line);
      return {
        line,
        html,
        ...(gutter
          ? {gutter, instructions: findSourceLine(file, line)?.instructions}
          : {}),
      };
    });
  });

  /** Rows of the lines with instructions, when the code is not loaded. */
  protected readonly attributedRows = computed<readonly SourceLinesRow[]>(
    () => {
      const gutters = this.gutters();
      return (this.file()?.lines ?? []).map((stats) => ({
        line: stats.line,
        html: '',
        gutter: gutters.get(stats.line),
        primitives: stats.primitives,
        instructions: stats.instructions,
      }));
    },
  );

  protected readonly selectedStats = computed(() => {
    const line = this.selectedLine();
    return line !== undefined ? findSourceLine(this.file(), line) : undefined;
  });

  /** Share of the schedule of the selected line, e.g. `8.6%`. */
  protected readonly selectedShare = computed(() => {
    const stats = this.selectedStats();
    return stats ? formatShare(stats.share) || '<1%' : '';
  });

  /** Lines of all files, from the largest share of the schedule down. */
  private readonly rankedLines = computed(() =>
    hotspots(this.summary(), Number.MAX_SAFE_INTEGER),
  );

  protected readonly hotspotItems = computed<readonly HotspotItem[]>(() => {
    const ranked = this.rankedLines().slice(0, HOTSPOT_LIST_LENGTH);
    const max = ranked[0]?.line.share ?? 0;
    return ranked.map(({file, line}) => ({
      key: `${file.path}:${line.line}`,
      path: file.path,
      line: line.line,
      label: `${file.name}:${line.line}`,
      code: this.codeOf(file.path, line.line),
      share: formatShare(line.share) || '<1%',
      barPercent: max > 0 ? Math.max(4, (line.share / max) * 100) : 0,
      segments: line.units.map((unit) => ({
        color: unit.unit.color,
        percent: unit.fraction * 100,
      })),
    }));
  });

  /** How much of the schedule the source map covers, e.g. `92%`. */
  protected readonly coverage = computed(() => {
    const summary = this.summary();
    if (!summary || summary.instructions === 0) {
      return '';
    }
    return formatFraction(
      summary.attributedInstructions / summary.instructions,
    );
  });

  /** The share of the schedule without a source line, below the hotspots. */
  protected readonly unattributedRow = computed<UnattributedRow | undefined>(
    () => {
      const unattributed = this.summary()?.unattributed;
      if (!unattributed || unattributed.instructions === 0) {
        return undefined;
      }
      const max = this.rankedLines()[0]?.line.share ?? 0;
      const spills = unattributed.spills + unattributed.fills;
      return {
        share: formatShare(unattributed.share) || '<1%',
        barPercent:
          max > 0 ? clamp((unattributed.share / max) * 100, 4, 100) : 0,
        description:
          formatCountOf(unattributed.instructions, 'instruction') +
          (spills > 0 ? `, incl. ${formatCount(spills)} spills and fills` : ''),
        tooltip:
          "Instructions that the source map doesn't attribute to a line, " +
          "e.g. of the HLO op's prologue and epilogue.",
      };
    },
  );

  /** Position of the highlighted instruction, e.g. `3 / 1,548`. */
  protected readonly stepLabel = computed(() => {
    const position = this.searchPosition();
    if (!position || position.count === 0) {
      return '';
    }
    return position.index >= 0
      ? `${formatCount(position.index + 1)} / ${formatCount(position.count)}`
      : `${formatCount(position.count)} lit`;
  });

  /** Empty state of the panel, or '' if it shows a file. */
  protected readonly emptyState = computed<
    'loading' | 'unsupported' | 'error' | 'no-map' | ''
  >(() => {
    if (this.files().length > 0) {
      return '';
    }
    const state = this.summaryState();
    return state === 'ready' ? 'no-map' : state;
  });

  /** Explains why the code of the file is not shown, or ''. */
  protected readonly contentMessage = computed(() => {
    const content = this.content();
    if (content.state === 'unavailable') {
      return "This XProf server doesn't serve source code.";
    }
    return content.state === 'error'
      ? (content.message ?? "Couldn't load the file.")
      : '';
  });

  constructor() {
    // Resets the panel when the session or the kernel changes.
    effect(() => {
      const sessionId = this.sessionId();
      const kernelKey = this.kernelKey();
      untracked(() => {
        if (sessionId !== this.lastSessionId) {
          this.lastSessionId = sessionId;
          this.resetSession();
        }
        if (kernelKey !== this.lastKernelKey) {
          this.lastKernelKey = kernelKey;
          this.filePath.set('');
          this.viewReveal.set(undefined);
          this.hideHoverCard();
          this.clearSelection();
        }
      });
    });

    // Loads the shown file.
    effect(() => {
      const path = this.resolvedPath();
      const sessionId = this.sessionId();
      untracked(() => {
        this.load(sessionId, path);
      });
    });

    // Links the shown file to Code Search, at the selected or busiest line.
    effect(() => {
      const sessionId = this.sessionId();
      const path = this.resolvedPath();
      const line =
        this.selectedLine() ?? this.busiestLine(this.file())?.line ?? 1;
      untracked(() => {
        this.updateCodeSearchLink(sessionId, path, line);
      });
    });

    // Reveals the line of a timeline selection, once the lines are known.
    effect(() => {
      const request = this.reveal();
      const summary = this.summary();
      untracked(() => {
        if (!request || !summary || request.nonce === this.lastRevealNonce) {
          return;
        }
        this.lastRevealNonce = request.nonce;
        this.revealLocation(request.location);
      });
    });

    // Brings the busiest line of a newly shown file into view, unless a line
    // of the file was already revealed.
    effect(() => {
      const path = this.resolvedPath();
      const file = this.file();
      const isLoading = this.content().state === 'loading';
      untracked(() => {
        const current = this.viewReveal();
        if (
          !file ||
          isLoading ||
          (current !== undefined && isSamePath(current.path, path))
        ) {
          return;
        }
        const busiest = this.busiestLine(file);
        if (busiest) {
          this.revealInView(busiest.line, false);
        }
      });
    });

    inject(DestroyRef).onDestroy(() => {
      for (const subscription of this.subscriptions.values()) {
        subscription.unsubscribe();
      }
      this.linkSubscription?.unsubscribe();
      clearTimeout(this.copiedTimer);
      clearTimeout(this.hoverTimer);
      this.stopDetailsResizeListeners();
    });
  }

  /** Selects the next (`delta` = 1) or previous (-1) busiest line. */
  stepHotspot(delta: number): void {
    const ranked = this.rankedLines();
    if (ranked.length === 0) {
      return;
    }
    const path = this.resolvedPath();
    const selected = this.selectedLine();
    const index =
      selected === undefined
        ? -1
        : ranked.findIndex(
            ({file, line}) =>
              line.line === selected && isSamePath(file.path, path),
          );
    const next =
      index < 0
        ? delta > 0
          ? 0
          : ranked.length - 1
        : (index + delta + ranked.length) % ranked.length;
    const {file, line} = ranked[next];
    this.showLine(file.path, line.line, true);
  }

  /** Clears the selected line, which turns off its timeline highlight. */
  clearSelection(): void {
    if (this.selectedLine() === undefined) {
      return;
    }
    this.selectedLine.set(undefined);
    this.selectionChange.emit(undefined);
  }

  /** Shows `path`, e.g. from the file menu. */
  protected selectFile(path: string): void {
    if (isSamePath(path, this.resolvedPath())) {
      return;
    }
    this.clearSelection();
    this.showFile(path);
  }

  /** Selects a line, or clears the selection if the line is selected. */
  protected onLineClick(line: number): void {
    this.hideHoverCard();
    this.suppressedHoverLine = line;
    if (this.selectedLine() === line) {
      this.clearSelection();
      return;
    }
    this.selectLine(line);
  }

  protected selectHotspot(item: HotspotItem): void {
    this.showLine(item.path, item.line, true);
  }

  /** Shows the line of a frame of the call stack. */
  protected selectFrame(frame: StackFrameView): void {
    if (!frame.path || !frame.line) {
      return;
    }
    const file = findSourceFile(this.summary(), frame.path);
    if (!file) {
      return;
    }
    if (findSourceLine(file, frame.line)) {
      this.showLine(file.path, frame.line, true);
      return;
    }
    this.hideHoverCard();
    // Callers without instructions of their own can still be looked at.
    if (!isSamePath(file.path, this.resolvedPath())) {
      this.selectFile(file.path);
    }
    this.revealInView(frame.line, true);
  }

  /** Shows the bundles of `stats` on the timeline. */
  protected zoomToRange(stats: SourceLineStats): void {
    this.zoom.emit(padRange(toRange(stats)));
  }

  /** Shows the bundles of the selected line on the timeline. */
  protected zoomToSelection(): void {
    const stats = this.selectedStats();
    if (stats) {
      this.zoomToRange(stats);
    }
  }

  /** Shows a slice of the schedule footprint of the line on the timeline. */
  protected zoomToBin(bin: number): void {
    const summary = this.summary();
    const bins = summary?.histogramBins ?? 0;
    if (!summary || bins === 0 || summary.bundles === 0) {
      return;
    }
    const start = Math.floor((bin * summary.bundles) / bins);
    const end = Math.max(
      start + 1,
      Math.ceil(((bin + 1) * summary.bundles) / bins),
    );
    this.zoom.emit({start, end});
  }

  /** Copies `path:line` of `line` in the shown file. */
  protected copyLineLocation(line: number): void {
    this.copy(`${toDepotPath(this.resolvedPath())}:${line}`, 'location');
  }

  /** Copies `path:line` of the selected or hovered line. */
  protected copyLocation(): void {
    const line = this.selectedLine() ?? this.hoveredLine();
    if (line !== undefined) {
      this.copyLineLocation(line);
    }
  }

  /** Copies the path of the shown file. */
  protected copyPath(): void {
    this.copy(toDepotPath(this.resolvedPath()), 'path');
  }

  /** Opens the file picker to show a local copy of the file. */
  protected pickLocalFile(): void {
    this.fileInput().nativeElement.click();
  }

  /** Shows a file that the user opened from disk in place of the source. */
  protected onLocalFile(event: Event): void {
    const target = event.target;
    if (!(target instanceof HTMLInputElement)) {
      return;
    }
    const file = target.files?.[0];
    target.value = '';
    if (!file) {
      return;
    }
    const path = this.resolvedPath();
    file.text().then(
      (text) => {
        const next = new Map(this.localFiles());
        next.set(
          path,
          this.highlightText(
            text,
          ),
        );
        this.localFiles.set(next);
      },
      () => {
        // The file could not be read; the current view stays.
      },
    );
  }

  /** Goes back from a local copy to the profiled version of the file. */
  protected closeLocalCopy(): void {
    const next = new Map(this.localFiles());
    next.delete(this.resolvedPath());
    this.localFiles.set(next);
  }

  protected onCardEnter(): void {
    this.isPointerInCard = true;
    clearTimeout(this.hoverTimer);
    const card = this.hoverCard();
    if (card) {
      this.hoveredLine.set(card.stats.line);
    }
  }

  protected onCardLeave(): void {
    this.isPointerInCard = false;
    this.scheduleHideHoverCard();
  }

  protected onLineHover(hover: SourceLinesHover | undefined): void {
    if (this.isPointerInCard) {
      return;
    }
    clearTimeout(this.hoverTimer);
    if (!hover) {
      // Keeps the card while the pointer crosses from the gutter into the
      // code of the same line, or into the card itself.
      this.scheduleHideHoverCard();
      return;
    }
    this.hoveredLine.set(hover.line);
    // The expanded details already show the stats of the selected line.
    if (hover.line === this.selectedLine() && this.isDetailsExpanded()) {
      this.hoverCard.set(undefined);
      return;
    }
    if (hover.line === this.suppressedHoverLine) {
      return;
    }
    this.suppressedHoverLine = 0;
    if (this.hoverCard()) {
      this.showHoverCard(hover);
      return;
    }
    this.hoverTimer = setTimeout(
      () => {
        this.showHoverCard(hover);
      },
      hover.inGutter ? HOVER_DELAY_GUTTER_MS : HOVER_DELAY_CODE_MS,
    );
  }

  /** Handles the keyboard shortcuts and context menu of the editor. */
  protected onEditorCommand(command: string): void {
    switch (command) {
      case 'next-hotspot':
        this.stepHotspot(1);
        break;
      case 'previous-hotspot':
        this.stepHotspot(-1);
        break;
      case 'next-instruction':
        this.step.emit(1);
        break;
      case 'previous-instruction':
        this.step.emit(-1);
        break;
      case 'clear':
        this.clearSelection();
        break;
      case 'copy-location':
        this.copyLocation();
        break;
      default:
        break;
    }
  }

  protected onEditorLoadError(): void {
    this.editorFailed.set(true);
  }

  protected runEditorAction(action: EditorAction): void {
  }

  protected toggleDetails(): void {
    const expanded = !this.isDetailsExpanded();
    this.isDetailsExpanded.set(expanded);
    writeStorage(DETAILS_STORAGE_KEY, String(expanded));
  }

  protected onDetailsResizerPointerDown(event: PointerEvent): void {
    const details = this.detailsElement()?.nativeElement;
    const panelHeight = this.panelHeight();
    if (event.button !== 0 || !details || panelHeight <= 0) {
      return;
    }
    event.preventDefault();
    this.hideHoverCard();
    this.detailsResizeStart = {
      y: event.clientY,
      percent: (details.getBoundingClientRect().height / panelHeight) * 100,
    };
    this.isResizingDetails.set(true);
    window.addEventListener('pointermove', this.detailsResizeMoveListener);
    window.addEventListener('pointerup', this.detailsResizeEndListener);
    window.addEventListener('pointercancel', this.detailsResizeEndListener);
  }

  /** Resizes the details with the arrow keys, like a slider. */
  protected onDetailsResizerKeydown(event: KeyboardEvent): void {
    const current = this.currentDetailsHeight();
    const step = DETAILS_HEIGHT_KEY_STEP_PERCENT;
    const heights: Record<string, number> = {
      'ArrowUp': current + step,
      'ArrowDown': current - step,
      'Home': MAX_DETAILS_HEIGHT_PERCENT,
      'End': MIN_DETAILS_HEIGHT_PERCENT,
    };
    const height = heights[event.key];
    if (height === undefined) {
      return;
    }
    event.preventDefault();
    event.stopPropagation();
    this.detailsHeight.set(clampDetailsHeight(height));
    this.saveDetailsHeight();
  }

  /** Lets the details fit their content again. */
  protected resetDetailsHeight(): void {
    this.detailsHeight.set(undefined);
    this.saveDetailsHeight();
  }

  private onDetailsResizerPointerMove(event: PointerEvent): void {
    const start = this.detailsResizeStart;
    const panelHeight = this.panelHeight();
    if (!start || panelHeight <= 0) {
      return;
    }
    const delta = ((start.y - event.clientY) / panelHeight) * 100;
    this.detailsHeight.set(clampDetailsHeight(start.percent + delta));
  }

  private endDetailsResize(): void {
    if (!this.detailsResizeStart) {
      return;
    }
    this.detailsResizeStart = undefined;
    this.isResizingDetails.set(false);
    this.stopDetailsResizeListeners();
    this.saveDetailsHeight();
  }

  private stopDetailsResizeListeners(): void {
    window.removeEventListener('pointermove', this.detailsResizeMoveListener);
    window.removeEventListener('pointerup', this.detailsResizeEndListener);
    window.removeEventListener('pointercancel', this.detailsResizeEndListener);
  }

  /** Height of the details, in percent of the panel. */
  private currentDetailsHeight(): number {
    const height = this.detailsHeight();
    if (height !== undefined) {
      return height;
    }
    const details = this.detailsElement()?.nativeElement;
    const panelHeight = this.panelHeight();
    return details && panelHeight > 0
      ? clampDetailsHeight(
          (details.getBoundingClientRect().height / panelHeight) * 100,
        )
      : MIN_DETAILS_HEIGHT_PERCENT;
  }

  protected setTheme(theme: string): void {
    this.updateOptions({...this.options(), theme});
  }

  protected setFontSize(fontSize: number): void {
    this.updateOptions({...this.options(), fontSize});
  }

  protected toggleOption(option: SourceViewToggle): void {
    const options = this.options();
    this.updateOptions({...options, [option]: !options[option]});
  }

  /** Formats the share of the schedule of a file, for the file menu. */
  protected shareOf(file: SourceFileStats): string {
    return formatShare(file.share) || '<1%';
  }

  protected hideHoverCard(): void {
    clearTimeout(this.hoverTimer);
    this.isPointerInCard = false;
    this.hoverCard.set(undefined);
    this.hoveredLine.set(undefined);
  }

  private scheduleHideHoverCard(): void {
    clearTimeout(this.hoverTimer);
    this.hoverTimer = setTimeout(() => {
      if (!this.isPointerInCard) {
        this.suppressedHoverLine = 0;
        this.hideHoverCard();
      }
    }, HOVER_HIDE_DELAY_MS);
  }

  private updateOptions(options: SourceViewOptions): void {
    this.options.set(options);
    writeOptions(options);
  }

  private panelHeight(): number {
    return this.panelElement().nativeElement.getBoundingClientRect().height;
  }

  private saveDetailsHeight(): void {
    const height = this.detailsHeight();
    if (height === undefined) {
      removeStorage(DETAILS_HEIGHT_STORAGE_KEY);
    } else {
      writeStorage(DETAILS_HEIGHT_STORAGE_KEY, String(Math.round(height)));
    }
  }

  private resetSession(): void {
    for (const subscription of this.subscriptions.values()) {
      subscription.unsubscribe();
    }
    this.subscriptions.clear();
    this.contents.set(new Map());
    this.localFiles.set(new Map());
  }

  private showFile(path: string): void {
    this.hideHoverCard();
    this.filePath.set(path);
  }

  /** Selects a line of the shown file, which lights up its instructions. */
  private selectLine(line: number): void {
    const stats = findSourceLine(this.file(), line);
    if (!stats) {
      this.clearSelection();
      return;
    }
    this.selectedLine.set(line);
    this.selectionChange.emit({
      query: stats.query,
      range: toRange(stats),
      label: `${this.fileName()}:${line}`,
    });
  }

  /** Shows a line with instructions of any file, selects it, and reveals it. */
  private showLine(path: string, line: number, flash: boolean): void {
    const file = findSourceFile(this.summary(), path);
    if (!file || !findSourceLine(file, line)) {
      return;
    }
    if (!isSamePath(file.path, this.resolvedPath())) {
      // The new selection replaces the one in the other file.
      this.selectedLine.set(undefined);
      this.showFile(file.path);
    }
    this.hideHoverCard();
    this.selectLine(line);
    this.revealInView(line, flash);
  }

  private revealLocation(location: SourceLocation): void {
    const file = findSourceFile(this.summary(), location.path);
    if (file) {
      if (!isSamePath(file.path, this.resolvedPath())) {
        this.selectFile(file.path);
      }
    } else if (!isSamePath(location.path, this.resolvedPath())) {
      // The panel only shows the files that the kernel was lowered from.
      return;
    }
    this.revealInView(location.line, true);
  }

  private revealInView(line: number, flash: boolean): void {
    this.viewReveal.set({
      path: this.resolvedPath(),
      line,
      nonce: ++this.viewRevealNonce,
      flash,
    });
  }

  private showHoverCard(hover: SourceLinesHover): void {
    const stats = findSourceLine(this.file(), hover.line);
    if (!stats) {
      return;
    }
    const panel = this.host.nativeElement.getBoundingClientRect();
    const width = Math.max(
      0,
      Math.min(CARD_WIDTH, panel.width - 2 * CARD_MARGIN),
    );
    const left = clamp(
      hover.x - panel.left - 28,
      CARD_MARGIN,
      Math.max(CARD_MARGIN, panel.width - width - CARD_MARGIN),
    );
    const estimatedHeight = estimateCardHeight(stats);
    const topLimit = Math.max(panel.top, 0) + CARD_MARGIN;
    const viewportHeight =
      document.documentElement.clientHeight || window.innerHeight;
    const bottomLimit =
      (viewportHeight > topLimit + CARD_MIN_HEIGHT
        ? Math.min(panel.bottom, viewportHeight)
        : panel.bottom) - CARD_MARGIN;
    // Shows the card above the line, like code editors do, when the whole card
    // fits above or when there is more room above than below.
    const spaceAbove = hover.bounds.top - CARD_GAP - topLimit;
    const spaceBelow = bottomLimit - (hover.bounds.bottom + CARD_GAP);
    const above =
      spaceAbove >= estimatedHeight ||
      (spaceBelow < estimatedHeight && spaceAbove > spaceBelow);
    const availableSpace = Math.floor(above ? spaceAbove : spaceBelow);
    const maxVisibleHeight = Math.max(0, Math.floor(bottomLimit - topLimit));
    const maxHeight = Math.max(
      Math.min(CARD_MIN_HEIGHT, maxVisibleHeight),
      Math.min(maxVisibleHeight, availableSpace),
    );
    const minTop = topLimit - panel.top;
    const minBottom = panel.bottom - bottomLimit;
    this.hoverCard.set({
      stats,
      width,
      maxHeight,
      left,
      ...(above
        ? {
            bottom: clamp(
              panel.bottom - hover.bounds.top + CARD_GAP,
              minBottom,
              Math.max(minBottom, panel.height - minTop - maxHeight),
            ),
          }
        : {
            top: clamp(
              hover.bounds.bottom - panel.top + CARD_GAP,
              minTop,
              Math.max(minTop, panel.height - minBottom - maxHeight),
            ),
          }),
    });
  }

  /** Returns the line of `file` with the largest share of the schedule. */
  private busiestLine(
    file: SourceFileStats | undefined,
  ): SourceLineStats | undefined {
    let busiest: SourceLineStats | undefined = undefined;
    for (const line of file?.lines ?? []) {
      if (!busiest || line.share > busiest.share) {
        busiest = line;
      }
    }
    return busiest;
  }

  /** Returns the trimmed code of a line, if its file is loaded. */
  private codeOf(path: string, line: number): string {
    const content = this.localFiles().get(path) ?? this.contents().get(path);
    return content?.text[line - 1]?.trim() ?? '';
  }

  private copy(text: string, what: 'path' | 'location'): void {
    if (!text || !navigator.clipboard) {
      return;
    }
    navigator.clipboard.writeText(text).then(
      () => {
        this.copied.set(what);
        clearTimeout(this.copiedTimer);
        this.copiedTimer = setTimeout(() => {
          this.copied.set('');
        }, COPIED_FEEDBACK_MS);
      },
      () => {
        // Clipboard access can be denied; there is nothing else to do.
      },
    );
  }

  private setContent(path: string, content: FileContent): void {
    const next = new Map(this.contents());
    next.set(path, content);
    this.contents.set(next);
  }

  private load(sessionId: string, path: string): void {
    if (!sessionId || !path || this.contents().has(path)) {
      return;
    }
    const service = this.sourceCodeService;
    if (!service?.isCodeFetchEnabled()) {
      this.setContent(path, {state: 'unavailable', html: [], text: []});
      return;
    }
    this.setContent(path, LOADING_CONTENT);
    const address = new Address(toDepotPath(path), 1, 0, MAX_FILE_LINES);
    const subscription = service.loadContent(sessionId, address).subscribe({
      next: (content) => {
        const html = [...content.lines];
        this.setContent(path, {
          state: 'ready',
          html,
          text: html.map(stripMarkup),
        });
      },
      error: (error: unknown) => {
        this.setContent(path, {
          state: 'error',
          html: [],
          text: [],
          message: describeLoadError(error, path),
        });
      },
    });
    this.subscriptions.set(path, subscription);
  }

  private highlightText(
    text: string,
  ): FileContent {
    const normalized = text.replace(/\r\n?/g, '\n');
    const lines = normalized.split('\n');
    if (lines[lines.length - 1] === '') {
      lines.pop();
    }
    const html = lines.map(escapeHtml);
    return {state: 'ready', html, text: lines};
  }

  private updateCodeSearchLink(
    sessionId: string,
    path: string,
    line: number,
  ): void {
    this.linkSubscription?.unsubscribe();
    this.linkSubscription = undefined;
    const service = this.sourceCodeService;
    if (!service || !sessionId || !path) {
      this.codeSearchUrl.set(undefined);
      return;
    }
    this.linkSubscription = service
      .codeSearchLink(sessionId, toDepotPath(path), line, '')
      .subscribe({
        next: (url) => {
          this.codeSearchUrl.set(url || undefined);
        },
        error: () => {
          this.codeSearchUrl.set(undefined);
        },
      });
  }
}
