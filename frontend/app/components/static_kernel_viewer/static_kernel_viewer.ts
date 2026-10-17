import {Location} from '@angular/common';
import {
  AfterViewInit,
  ChangeDetectionStrategy,
  ChangeDetectorRef,
  Component,
  ElementRef,
  inject,
  OnDestroy,
  OnInit,
  ViewChild,
} from '@angular/core';
import {ActivatedRoute, Params} from '@angular/router';
import {Store} from '@ngrx/store';
import {DEFAULT_HOST} from 'org_xprof/frontend/app/common/constants/constants';
import {
  KernelMinimap,
  MinimapRegionMarker,
} from 'org_xprof/frontend/app/components/static_kernel_viewer/kernel_minimap';
import {KernelSelectionPanel} from 'org_xprof/frontend/app/components/static_kernel_viewer/kernel_selection_panel';
import {
  DetailsKind,
  hasSeenKernelViewerTour,
  KernelViewerTour,
  KernelViewerTourSummary,
  TourKernel,
} from 'org_xprof/frontend/app/components/static_kernel_viewer/kernel_viewer_tour';
import {
  LOADING_STATUS_UPDATE_EVENT_NAME,
  shutdownTraceViewerV2,
  TraceViewerV2LoadingStatus,
  traceViewerV2Main,
  TraceViewerV2Module,
} from 'org_xprof/frontend/app/components/trace_viewer_v2/main';
import {
  DATA_SERVICE_INTERFACE_TOKEN,
  DataServiceV2Interface,
} from 'org_xprof/frontend/app/services/data_service_v2/data_service_v2_interface';
import {setCurrentToolStateAction} from 'org_xprof/frontend/app/store/actions';
import {combineLatest, firstValueFrom, ReplaySubject} from 'rxjs';
import {takeUntil} from 'rxjs/operators';

/** Response structure from the Kernel Viewer backend list request. */
declare interface KernelListResponse {
  // tslint:disable-next-line:enforce-name-casing
  readonly hlo_modules?: readonly string[];
  // tslint:disable-next-line:enforce-name-casing
  readonly module_kernels?: Record<string, readonly string[]>;
  readonly kernels?: readonly string[];
}

/** Statistics of one functional unit in the static schedule of a kernel. */
export interface UnitStats {
  /** Timeline lane of the unit, e.g. `MXU`. */
  readonly unit: string;
  /** Number of instructions scheduled on the unit. */
  readonly instructions: number;
  /** Mean per-bundle utilization between 0 and 1, if known. */
  readonly utilization?: number;
}

/**
 * Summary of the static schedule of a kernel, which the list request reports
 * in `kernel_stats[module][kernel]`.
 */
export interface KernelStats {
  /** Length of the schedule. */
  readonly bundles: number;
  /** Number of scheduled instructions. */
  readonly instructions: number;
  /** Functional units with instructions or utilization, busiest first. */
  readonly units: readonly UnitStats[];
  /** Whether `units` report utilization, which needs a bundle profile. */
  readonly hasUtilization: boolean;
  /** Number of vector register spills, if known. */
  readonly vectorSpills?: number;
  /** Number of vector register fills, if known. */
  readonly vectorFills?: number;
}

/** A labeled value in the details card of a kernel. */
export interface KernelFact {
  readonly label: string;
  readonly value: string;
  /** Whether the value points at a likely performance problem. */
  readonly warning: boolean;
}

/** How a kind of kernel, e.g. Pallas, is presented. */
export interface KernelKind {
  /** Kind reported by the backend, e.g. `pallas`. */
  readonly id: string;
  /** Label of the kind, e.g. `Pallas`. */
  readonly label: string;
  /** Label of one kernel of the kind, e.g. `Pallas kernel`. */
  readonly noun: string;
  /** Material icon of the kind. */
  readonly icon: string;
  /** Accent color of the kind. */
  readonly color: string;
  /** Position of the kind in filters and sections. */
  readonly order: number;
}

/** How the kernel rail orders the kernels within each section. */
export type KernelSort = 'bundles' | 'utilization' | 'name';

/** How the kernel rail divides the kernels into sections. */
export type KernelGrouping = 'module' | 'kind' | 'none';

/** An option of a kernel rail filter. */
export interface KernelFilterOption {
  /** Value of the option, e.g. a kind or a functional unit. */
  readonly id: string;
  readonly label: string;
  /** Number of kernels that the option shows, given the other filters. */
  readonly count: number;
  readonly icon?: string;
  readonly color?: string;
}

/** A run of a kernel or module name, which may match the search terms. */
export interface NameSegment {
  readonly text: string;
  readonly match: boolean;
}

/** A bar of the functional unit chart in the details card of a kernel. */
export interface UnitBar {
  readonly unit: string;
  /** Color of the functional unit. */
  readonly color: string;
  /** Length of the bar, between 0 and 1. */
  readonly fraction: number;
  readonly label: string;
}

/** A text or keyboard shortcut token in the footer of a details card. */
export interface HintPart {
  readonly text: string;
  readonly isKey: boolean;
}

/** A kernel that can be opened in the viewer. */
export interface KernelEntry {
  /** HLO module of the kernel, or '' if the backend reports no modules. */
  readonly module: string;
  /** Kernel name, as reported by the backend. */
  readonly kernel: string;
  /**
   * Kernel name split after each `_` and `-`, where the details card may wrap
   * it, e.g. `['flash_', 'attention_', 'bwd.3']`.
   */
  readonly nameParts: readonly string[];
  /** Zero-based position in the kernel list. */
  readonly index: number;
  /** Short label shown while the kernel rail is collapsed. */
  readonly monogram: string;
  /** Color of the kernel's module group. */
  readonly color: string;
  /** Statistics of the kernel's schedule, if the backend reports them. */
  readonly stats?: KernelStats;
  /** Kind of the kernel, e.g. Pallas, if the backend reports it. */
  readonly kind?: KernelKind;
  /** Compact schedule length, e.g. `38.4K`, or '' without statistics. */
  readonly bundlesLabel: string;
  /** Schedule length, e.g. `38,412 bundles`, or '' without statistics. */
  readonly bundlesText: string;
  /** Busiest functional unit and its utilization, e.g. `MXU 82%`, or ''. */
  readonly busiestUnit: string;
  /** Busiest functional unit, e.g. `MXU`, or ''. */
  readonly boundUnit: string;
  /** Color of the busiest functional unit, or ''. */
  readonly busiestUnitColor: string;
  /** Whether the schedule spills vector registers to memory. */
  readonly hasSpills: boolean;
  /** Module and schedule length, shown below the kernel name in its tab. */
  readonly subtitle: string;
  /** Facts shown in the details card. */
  readonly facts: readonly KernelFact[];
  /** Title of the functional unit chart in the details card. */
  readonly chartTitle: string;
  /** Functional unit chart in the details card. */
  readonly bars: readonly UnitBar[];
  /** Accessible description of the kernel and its statistics. */
  readonly description: string;
  /** Lowercase kernel name, module, kind and busiest unit, for the search. */
  readonly searchText: string;
}

/** The kernels of one HLO module, shown like a browser tab group. */
export interface KernelGroup {
  readonly module: string;
  readonly color: string;
  readonly entries: readonly KernelEntry[];
}

/**
 * A section of the kernel rail: the kernels of an HLO module or of a kind, or
 * all kernels if the rail is not grouped.
 */
export interface KernelSection {
  /** Identifies the section across updates, e.g. `module:jit_step`. */
  readonly id: string;
  /** Header of the section, or '' for a section without header. */
  readonly label: string;
  readonly color: string;
  /** Whether `label` is the name of an HLO module. */
  readonly isModule: boolean;
  /** Material icon of a kind section. */
  readonly icon?: string;
  /** Kernels that pass the filters, in display order. */
  readonly entries: readonly KernelEntry[];
}

/** A kernel opened in a tab of the kernel bar. */
export interface KernelTab {
  /** Identifies the tab while its kernel changes. */
  readonly id: number;
  /** Kernel shown in the tab. Stepping through kernels replaces it. */
  entry: KernelEntry;
  /** When the tab was last active, to close the least recently used tab. */
  lastUsed: number;
}

/** Details card of a hovered kernel row or tab. */
export interface KernelCard {
  readonly entry: KernelEntry;
  /** Whether the card belongs to a row of the kernel rail or to a tab. */
  readonly source: 'rail' | 'tab';
  /** Footer, e.g. the shortcut that switches to the kernel's tab. */
  readonly hint: string;
  /** `hint`, split into plain text and keyboard shortcut tokens. */
  readonly hintParts: readonly HintPart[];
  /** Position relative to the component, in pixels. */
  readonly left: number;
  readonly top?: number;
  readonly bottom?: number;
}

/** Context menu popup opened by right-clicking a kernel row or tab. */
export interface KernelContextMenu {
  readonly entry: KernelEntry;
  /** Whether the menu was opened from a row of the kernel rail or a tab. */
  readonly source: 'rail' | 'tab';
  /** Position relative to the component, in pixels. */
  readonly left: number;
  readonly top: number;
}

/** Colors assigned to module groups, in order (browser tab group colors). */
const GROUP_COLORS = [
  '#1a73e8',
  '#d93025',
  '#e37400',
  '#188038',
  '#d01884',
  '#9334e6',
  '#007b83',
  '#e8710a',
  '#5f6368',
];

/** Colors of functional units, matched by uppercase prefix. */
const UNIT_COLORS: ReadonlyArray<readonly [string, string]> = [
  ['MXU', '#7c4dff'],
  ['VPU', '#1a73e8'],
  ['SALU', '#188038'],
  ['VLD', '#00838f'],
  ['VST', '#c2185b'],
  ['XLU', '#e37400'],
  ['EUP', '#00796b'],
  ['DMA', '#e8710a'],
];

/** Kinds of kernel that the backend reports, see `parseKernelKind`. */
const KERNEL_KINDS: ReadonlyMap<string, KernelKind> = new Map([
  [
    'pallas',
    {
      id: 'pallas',
      label: 'Pallas',
      noun: 'Pallas kernel',
      icon: 'function',
      color: '#8430ce',
      order: 0,
    },
  ],
  [
    'custom_call',
    {
      id: 'custom_call',
      label: 'Custom call',
      noun: 'custom call',
      icon: 'extension',
      color: '#b06000',
      order: 1,
    },
  ],
]);

/** Color and icon of kinds that this version of the viewer does not know. */
const OTHER_KIND_COLOR = '#5f6368';
const OTHER_KIND_ICON = 'category';

/** Local storage keys of the kernel rail's filter, sort and group preferences. */
const KIND_FILTER_STORAGE_KEY = 'xprof_static_kernel_viewer_kind_filter';
const UNIT_FILTER_STORAGE_KEY = 'xprof_static_kernel_viewer_unit_filter';
const SPILLS_ONLY_STORAGE_KEY = 'xprof_static_kernel_viewer_spills_only';
const SORT_STORAGE_KEY = 'xprof_static_kernel_viewer_sort';
const GROUPING_STORAGE_KEY = 'xprof_static_kernel_viewer_grouping';

const KERNEL_SORTS: readonly KernelSort[] = ['bundles', 'utilization', 'name'];
const KERNEL_GROUPINGS: readonly KernelGrouping[] = ['module', 'kind', 'none'];

/** Sorts kernel names like people do, e.g. `fusion.2` before `fusion.10`. */
const NAME_COLLATOR = new Intl.Collator(undefined, {
  numeric: true,
  sensitivity: 'base',
});

/** Delay before the collapsed kernel rail expands on hover. */
const RAIL_EXPAND_DELAY_MS = 150;

/** Delay before the hover-expanded kernel rail collapses again. */
const RAIL_COLLAPSE_DELAY_MS = 250;

/** Duration of the kernel rail width transition, see the stylesheet. */
const RAIL_TRANSITION_MS = 200;

/** How long the copy link button shows its confirmation. */
const LINK_COPIED_FEEDBACK_MS = 1500;

/** Local storage key of the kernel rail pin preference. */
const RAIL_PINNED_STORAGE_KEY = 'xprof_static_kernel_viewer_rail_pinned';

/** Maximum number of tabs. Opening another closes the least recently used. */
const MAX_OPEN_TABS = 8;

/** How long a tab opened in the background stays highlighted. */
const TAB_HIGHLIGHT_MS = 900;

/** Delay before the details card of a hovered kernel appears. */
const CARD_SHOW_DELAY_MS = 400;

/** Delay before the details card hides, so it can move to a neighbor. */
const CARD_HIDE_DELAY_MS = 100;

/** Width of the details card, see the stylesheet. */
const CARD_WIDTH = 320;

/** Distance between the details card and its kernel. */
const CARD_GAP = 8;

/** Approximate dimensions of the kernel context menu, for viewport clamping. */
const CONTEXT_MENU_WIDTH = 180;
const CONTEXT_MENU_HEIGHT = 44;

/**
 * Delay before the tour starts on the first visit, so that the kernel list
 * and tabs settle first.
 */
const TOUR_AUTO_START_DELAY_MS = 900;

/** How long the tour button pulses as the tour closes; see the stylesheet. */
const TOUR_BUTTON_PULSE_MS = 1100;

/** Keys that only modify other keys. */
const MODIFIER_KEYS = new Set(['Alt', 'Control', 'Meta', 'Shift']);

const IS_APPLE_PLATFORM = /Mac|iPhone|iPad/.test(navigator.userAgent);

/** Label of the modifier key that opens kernels in background tabs. */
const BACKGROUND_CLICK_KEY = IS_APPLE_PLATFORM ? '⌘' : 'Ctrl';

const NUMBER_FORMAT = new Intl.NumberFormat();
const COMPACT_NUMBER_FORMAT = new Intl.NumberFormat(undefined, {
  notation: 'compact',
  maximumFractionDigits: 1,
});
const RATIO_FORMAT = new Intl.NumberFormat(undefined, {
  maximumFractionDigits: 1,
});
const PERCENT_FORMAT = new Intl.NumberFormat(undefined, {
  style: 'percent',
  maximumFractionDigits: 0,
});

/**
 * Returns a two-letter label for a kernel, built from its first two words,
 * e.g. `add_add_fusion.114` -> `Aa` and `MLA-bd-bq_1` -> `Mb`.
 */
export function kernelMonogram(kernel: string): string {
  const [first = '', second = ''] =
    kernel.match(/[A-Z]?[a-z]+|[A-Z]+(?![a-z])/g) ?? [];
  if (!first) {
    return kernel.slice(0, 2).toUpperCase() || '?';
  }
  const secondLetter = (second || first.slice(1)).charAt(0);
  return first.charAt(0).toUpperCase() + secondLetter.toLowerCase();
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function readNumber(value: unknown): number | undefined {
  return typeof value === 'number' && Number.isFinite(value)
    ? value
    : undefined;
}

/**
 * Parses the schedule summary of a kernel, e.g. `{"bundles": 4,
 * "instructions": 5, "unit_instructions": {"MXU": 1, "VPU": 4},
 * "unit_utilization": {"MXU": 0.25}, "vector_spills": 0, "vector_fills": 0}`.
 * Returns undefined if the summary has no schedule length.
 */
export function parseKernelStats(summary: unknown): KernelStats | undefined {
  if (!isRecord(summary)) {
    return undefined;
  }
  const bundles = readNumber(summary['bundles']);
  if (bundles === undefined) {
    return undefined;
  }
  const unitInstructions = isRecord(summary['unit_instructions'])
    ? summary['unit_instructions']
    : {};
  const unitUtilization = isRecord(summary['unit_utilization'])
    ? summary['unit_utilization']
    : undefined;
  const unitNames = new Set([
    ...Object.keys(unitInstructions),
    ...Object.keys(unitUtilization ?? {}),
  ]);
  const units: UnitStats[] = [];
  for (const unit of unitNames) {
    const instructions = readNumber(unitInstructions[unit]) ?? 0;
    const utilization =
      unitUtilization === undefined
        ? undefined
        : (readNumber(unitUtilization[unit]) ?? 0);
    if (instructions > 0 || (utilization ?? 0) > 0) {
      units.push({unit, instructions, utilization});
    }
  }
  units.sort(
    (a, b) =>
      (b.utilization ?? 0) - (a.utilization ?? 0) ||
      b.instructions - a.instructions,
  );
  return {
    bundles,
    instructions: readNumber(summary['instructions']) ?? 0,
    units,
    hasUtilization: unitUtilization !== undefined,
    vectorSpills: readNumber(summary['vector_spills']),
    vectorFills: readNumber(summary['vector_fills']),
  };
}

function formatPercent(fraction: number): string {
  // Tiny shares would round to 0%, which reads as unused.
  return fraction > 0 && fraction < 0.005
    ? `<${PERCENT_FORMAT.format(0.01)}`
    : PERCENT_FORMAT.format(fraction);
}

/**
 * Returns how to present the `kind` of a kernel's schedule summary, e.g.
 * `{"kind": "pallas", ...}`, or undefined if the summary reports no kind.
 * Kinds that this viewer does not know get a label derived from their name.
 */
export function parseKernelKind(summary: unknown): KernelKind | undefined {
  const id = isRecord(summary) ? summary['kind'] : undefined;
  if (typeof id !== 'string' || !id.trim()) {
    return undefined;
  }
  const known = KERNEL_KINDS.get(id);
  if (known) {
    return known;
  }
  const words = id.trim().replace(/[_-]+/g, ' ');
  const label = words.charAt(0).toUpperCase() + words.slice(1);
  return {
    id,
    label,
    noun: `${label} kernel`,
    icon: OTHER_KIND_ICON,
    color: OTHER_KIND_COLOR,
    order: KERNEL_KINDS.size,
  };
}

/** Terms of a kernel search. */
export interface SearchTerms {
  /** Lowercase terms that each kernel must contain. */
  readonly include: readonly string[];
  /** Lowercase terms, written with a leading `-`, that kernels must not contain. */
  readonly exclude: readonly string[];
}

/**
 * Splits a kernel search into terms, e.g. `attn fwd -mla` finds kernels that
 * contain `attn` and `fwd` but not `mla`. Terms are plain text, not patterns.
 */
export function parseSearchTerms(query: string): SearchTerms {
  const include: string[] = [];
  const exclude: string[] = [];
  for (const term of query.toLowerCase().split(/\s+/)) {
    if (term.length > 1 && term.startsWith('-')) {
      exclude.push(term.slice(1));
    } else if (term && term !== '-') {
      include.push(term);
    }
  }
  return {include, exclude};
}

function matchesSearchTerms(entry: KernelEntry, terms: SearchTerms): boolean {
  return (
    terms.include.every((term) => entry.searchText.includes(term)) &&
    !terms.exclude.some((term) => entry.searchText.includes(term))
  );
}

/**
 * Splits `text` into runs that do and do not contain any of `terms`, so that
 * matches can be highlighted, e.g. `flash_attn` with `att` gives `flash_`,
 * `att` (a match) and `n`.
 */
export function splitMatches(
  text: string,
  terms: readonly string[],
): NameSegment[] {
  const lower = text.toLowerCase();
  const ranges: Array<[number, number]> = [];
  for (const term of terms) {
    for (let i = lower.indexOf(term); i >= 0; i = lower.indexOf(term, i + 1)) {
      ranges.push([i, i + term.length]);
    }
  }
  if (ranges.length === 0) {
    return [{text, match: false}];
  }
  ranges.sort((a, b) => a[0] - b[0] || b[1] - a[1]);
  const segments: NameSegment[] = [];
  let end = 0;
  for (const [start, stop] of ranges) {
    if (stop <= end) {
      continue;
    }
    const from = Math.max(start, end);
    if (from > end) {
      segments.push({text: text.slice(end, from), match: false});
    }
    const last = segments[segments.length - 1];
    if (last?.match && from === end) {
      segments[segments.length - 1] = {
        text: last.text + text.slice(from, stop),
        match: true,
      };
    } else {
      segments.push({text: text.slice(from, stop), match: true});
    }
    end = stop;
  }
  if (end < text.length) {
    segments.push({text: text.slice(end), match: false});
  }
  return segments;
}

/** Mean utilization of the busiest unit of `entry`, or -1 if unknown. */
function busiestUtilization(entry: KernelEntry): number {
  return entry.boundUnit ? (entry.stats?.units[0]?.utilization ?? 0) : -1;
}

function compareBundles(a: KernelEntry, b: KernelEntry): number {
  return (b.stats?.bundles ?? -1) - (a.stats?.bundles ?? -1);
}

/** Orders kernels by `sort`, keeping the backend's order for ties. */
function kernelComparator(
  sort: KernelSort,
): (a: KernelEntry, b: KernelEntry) => number {
  switch (sort) {
    case 'utilization':
      return (a, b) =>
        busiestUtilization(b) - busiestUtilization(a) ||
        compareBundles(a, b) ||
        a.index - b.index;
    case 'name':
      return (a, b) =>
        NAME_COLLATOR.compare(a.kernel, b.kernel) || a.index - b.index;
    default:
      return (a, b) => compareBundles(a, b) || a.index - b.index;
  }
}

function readPreference<T extends string>(
  key: string,
  values: readonly T[],
  fallback: T,
): T {
  try {
    const value = window.localStorage.getItem(key);
    return values.find((option) => option === value) ?? fallback;
  } catch {
    // Storage can be unavailable, e.g. in sandboxed iframes.
    return fallback;
  }
}

function readStringPreference(key: string): string {
  try {
    return window.localStorage.getItem(key)?.trim() ?? '';
  } catch {
    // Storage can be unavailable, e.g. in sandboxed iframes.
    return '';
  }
}

function readListPreference(key: string): string[] {
  try {
    const raw = window.localStorage.getItem(key);
    if (!raw) {
      return [];
    }
    const parsed: unknown = JSON.parse(raw);
    return Array.isArray(parsed)
      ? parsed.filter(
          (item): item is string =>
            typeof item === 'string' && item.trim() !== '',
        )
      : [];
  } catch {
    // Storage can be unavailable or hold an older format.
    return [];
  }
}

function readBooleanPreference(key: string): boolean {
  try {
    return window.localStorage.getItem(key) === 'true';
  } catch {
    // Storage can be unavailable, e.g. in sandboxed iframes.
    return false;
  }
}

function writePreference(key: string, value: string) {
  try {
    window.localStorage.setItem(key, value);
  } catch {
    // Storage can be unavailable, e.g. in sandboxed iframes.
  }
}

function writeListPreference(key: string, values: readonly string[]) {
  writePreference(key, JSON.stringify(values));
}

function kernelFacts(stats: KernelStats): KernelFact[] {
  let instructions = NUMBER_FORMAT.format(stats.instructions);
  if (stats.bundles > 0) {
    const perBundle = RATIO_FORMAT.format(stats.instructions / stats.bundles);
    instructions += ` · ${perBundle} per bundle`;
  }
  const facts: KernelFact[] = [
    {
      label: 'Bundles',
      value: NUMBER_FORMAT.format(stats.bundles),
      warning: false,
    },
    {label: 'Instructions', value: instructions, warning: false},
  ];
  if (stats.vectorSpills !== undefined || stats.vectorFills !== undefined) {
    const spills = stats.vectorSpills ?? 0;
    const fills = stats.vectorFills ?? 0;
    facts.push({
      label: 'Vector spills / fills',
      value:
        spills + fills > 0
          ? `${NUMBER_FORMAT.format(spills)} / ${NUMBER_FORMAT.format(fills)}`
          : 'None',
      warning: spills + fills > 0,
    });
  }
  return facts;
}

function unitColor(unit: string): string {
  const upper = unit.toUpperCase();
  for (const [prefix, color] of UNIT_COLORS) {
    if (upper.startsWith(prefix)) {
      return color;
    }
  }
  return 'var(--mat-sys-primary)';
}

function buildCardHint(parts: readonly HintPart[]): {
  hint: string;
  hintParts: readonly HintPart[];
} {
  return {
    hint: parts.map((part) => part.text).join(''),
    hintParts: parts,
  };
}

/**
 * Charts the mean utilization of each functional unit if the schedule reports
 * it, or else the share of instructions of each unit.
 */
function unitChart(stats?: KernelStats): {
  chartTitle: string;
  bars: readonly UnitBar[];
} {
  const units = stats?.units ?? [];
  const utilized = units.filter((unit) => (unit.utilization ?? 0) > 0);
  if (utilized.length > 0) {
    return {
      chartTitle: 'Mean utilization',
      bars: utilized.map(({unit, utilization = 0}) => ({
        unit,
        color: unitColor(unit),
        fraction: Math.min(utilization, 1),
        label: formatPercent(utilization),
      })),
    };
  }
  const total = units.reduce((sum, unit) => sum + unit.instructions, 0);
  if (total === 0) {
    return {chartTitle: '', bars: []};
  }
  return {
    chartTitle: 'Instruction mix',
    bars: units
      .filter((unit) => unit.instructions > 0)
      .map(({unit, instructions}) => ({
        unit,
        color: unitColor(unit),
        fraction: instructions / total,
        label: formatPercent(instructions / total),
      })),
  };
}

function createEntry(
  module: string,
  kernel: string,
  index: number,
  color: string,
  stats?: KernelStats,
  kind?: KernelKind,
): KernelEntry {
  const bundlesText = stats
    ? `${NUMBER_FORMAT.format(stats.bundles)} ${
        stats.bundles === 1 ? 'bundle' : 'bundles'
      }`
    : '';
  // A busiest unit is only meaningful with utilization from a bundle profile.
  const busiest = stats?.hasUtilization ? stats.units[0] : undefined;
  const busiestUnit = busiest?.utilization
    ? `${busiest.unit} ${formatPercent(busiest.utilization)}`
    : '';
  const boundUnit = busiestUnit && busiest ? busiest.unit : '';
  const busiestUnitColor = boundUnit ? unitColor(boundUnit) : '';
  const hasSpills = (stats?.vectorSpills ?? 0) + (stats?.vectorFills ?? 0) > 0;
  const description = [
    kernel,
    module,
    kind?.label ?? '',
    bundlesText,
    busiestUnit ? `busiest unit ${busiestUnit}` : '',
    hasSpills ? 'spills vector registers' : '',
  ];
  return {
    module,
    kernel,
    nameParts: kernel.match(/[^-_]+[-_]*|[-_]+/g) ?? [kernel],
    index,
    monogram: kernelMonogram(kernel),
    color,
    stats,
    kind,
    bundlesLabel: stats ? COMPACT_NUMBER_FORMAT.format(stats.bundles) : '',
    bundlesText,
    busiestUnit,
    boundUnit,
    busiestUnitColor,
    hasSpills,
    subtitle: [module, bundlesText].filter(Boolean).join(' · '),
    facts: stats ? kernelFacts(stats) : [],
    ...unitChart(stats),
    description: description.filter(Boolean).join(', '),
    // Newlines keep search terms from matching across fields.
    searchText: [kernel, module, kind?.label, kind?.id, boundUnit]
      .filter(Boolean)
      .join('\n')
      .toLowerCase(),
  };
}

/** An HLO module group before its kernels are assigned display indices. */
interface DraftGroup {
  readonly module: string;
  readonly kernels: ReadonlyArray<{
    readonly kernel: string;
    readonly stats?: KernelStats;
    readonly kind?: KernelKind;
  }>;
  readonly maxBundles: number;
}

function buildDraftGroup(
  module: string,
  kernelNames: readonly string[],
  moduleStats: unknown,
): DraftGroup | undefined {
  if (kernelNames.length === 0) {
    return undefined;
  }
  const statsByKernel = isRecord(moduleStats) ? moduleStats : {};
  const kernels = kernelNames
    .map((kernel) => ({
      kernel,
      stats: parseKernelStats(statsByKernel[kernel]),
      kind: parseKernelKind(statsByKernel[kernel]),
    }))
    .sort((a, b) => (b.stats?.bundles ?? -1) - (a.stats?.bundles ?? -1));
  const maxBundles = kernels[0]?.stats?.bundles ?? -1;
  return {module, kernels, maxBundles};
}

/**
 * Groups the kernels of a list response by HLO module. Kernels within each
 * module are sorted by descending bundle count, and modules are ranked by the
 * maximum bundle count of their kernels.
 */
function buildKernelGroups(response: unknown): KernelGroup[] {
  const list = (isRecord(response) ? response : {}) as KernelListResponse;
  const stats =
    isRecord(response) && isRecord(response['kernel_stats'])
      ? response['kernel_stats']
      : {};
  const moduleKernels = list.module_kernels ?? {};
  const modules = new Set([
    ...(list.hlo_modules ?? []),
    ...Object.keys(moduleKernels),
  ]);
  const namedGroups: DraftGroup[] = [];
  const groupedKernels = new Set<string>();
  for (const module of modules) {
    const kernels = moduleKernels[module] ?? [];
    for (const kernel of kernels) {
      groupedKernels.add(kernel);
    }
    const group = buildDraftGroup(module, kernels, stats[module]);
    if (group) {
      namedGroups.push(group);
    }
  }
  namedGroups.sort((a, b) => b.maxBundles - a.maxBundles);
  // Older backends only report a flat list of kernels.
  const ungrouped = buildDraftGroup(
    '',
    (list.kernels ?? []).filter((kernel) => !groupedKernels.has(kernel)),
    stats[''],
  );
  const allGroups = ungrouped ? [...namedGroups, ungrouped] : namedGroups;
  let index = 0;
  return allGroups.map((group, groupIndex) => {
    const color = GROUP_COLORS[groupIndex % GROUP_COLORS.length];
    const entries = group.kernels.map(({kernel, stats: entryStats, kind}) =>
      createEntry(group.module, kernel, index++, color, entryStats, kind),
    );
    return {module: group.module, color, entries};
  });
}

function isEditableTarget(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) {
    return false;
  }
  return (
    target.isContentEditable ||
    target.tagName === 'INPUT' ||
    target.tagName === 'TEXTAREA' ||
    target.tagName === 'SELECT'
  );
}

/**
 * Returns the longest of the region or scope `markers` of the minimap. The
 * minimap merges runs of tiny events into one marker, which is as long as the
 * run but selects only one of its events, so these come last.
 */
function longestMarker(
  markers: readonly MinimapRegionMarker[],
): MinimapRegionMarker | undefined {
  const events = markers.filter((marker) => marker.count === 1);
  return (events.length > 0 ? events : markers).reduce<
    MinimapRegionMarker | undefined
  >(
    (best, marker) =>
      best && best.durationUs >= marker.durationUs ? best : marker,
    undefined,
  );
}

/** Scrolls `container` horizontally just enough to show all of `child`. */
function revealHorizontally(container: HTMLElement, child: HTMLElement) {
  const start = child.offsetLeft;
  const end = start + child.offsetWidth;
  if (start < container.scrollLeft) {
    container.scrollLeft = start;
  } else if (end > container.scrollLeft + container.clientWidth) {
    container.scrollLeft = end - container.clientWidth;
  }
}

function readRailPinned(): boolean {
  try {
    return window.localStorage.getItem(RAIL_PINNED_STORAGE_KEY) === 'true';
  } catch {
    // Storage can be unavailable, e.g. in sandboxed iframes.
    return false;
  }
}

function writeRailPinned(pinned: boolean) {
  try {
    window.localStorage.setItem(RAIL_PINNED_STORAGE_KEY, String(pinned));
  } catch {
    // Storage can be unavailable, e.g. in sandboxed iframes.
  }
}

/**
 * Component for the Static Kernel Viewer tool page.
 *
 * Kernels are listed in a rail on the left, grouped by HLO module like
 * vertical browser tabs. The rail stays collapsed to a narrow strip so the
 * timeline gets most of the width; it expands over the timeline on hover and
 * can be pinned open. Kernels open in tabs above the timeline.
 *
 * The rail narrows down the kernels by search terms, by kind (e.g. Pallas
 * kernels or custom calls), by busiest functional unit and by vector register
 * spills, and can sort and group them differently.
 */
@Component({
  changeDetection: ChangeDetectionStrategy.Default,
  standalone: false,
  selector: 'static-kernel-viewer',
  templateUrl: './static_kernel_viewer.ng.html',
  styleUrls: ['./static_kernel_viewer.scss'],
})
export class StaticKernelViewer implements OnInit, AfterViewInit, OnDestroy {
  readonly tool = 'kernel_viewer';
  sessionId = '';
  host = DEFAULT_HOST;

  /** Kernels grouped by HLO module, in display order. */
  groups: KernelGroup[] = [];
  /** All kernels in display order. */
  entries: KernelEntry[] = [];
  /** Sections of the rail, with the kernels that pass the filters. */
  sections: KernelSection[] = [];
  /** Number of kernels in `sections`. */
  visibleCount = 0;
  /** Kinds of the listed kernels, with counts given the other filters. */
  kindOptions: KernelFilterOption[] = [];
  /** Number of kernels of any kind that pass the other filters. */
  allKindsCount = 0;
  /** Busiest units of the listed kernels, with counts. */
  unitOptions: KernelFilterOption[] = [];
  /** Number of kernels with spills that pass the other filters. */
  spillsCount = 0;
  /** Kind of the kernels the rail shows, or '' for all kinds. */
  kindFilter = readStringPreference(KIND_FILTER_STORAGE_KEY);
  /** Busiest units of the kernels the rail shows, or empty for all units. */
  unitFilter: readonly string[] = readListPreference(UNIT_FILTER_STORAGE_KEY);
  /** Whether the rail only shows kernels that spill vector registers. */
  spillsOnly = readBooleanPreference(SPILLS_ONLY_STORAGE_KEY);
  sortOrder: KernelSort = readPreference(
    SORT_STORAGE_KEY,
    KERNEL_SORTS,
    'bundles',
  );
  grouping: KernelGrouping = readPreference(
    GROUPING_STORAGE_KEY,
    KERNEL_GROUPINGS,
    'module',
  );
  /** Whether the panel of unit, spill, sort and grouping options is open. */
  isFilterPanelOpen = false;
  readonly sortOptions: ReadonlyArray<{id: KernelSort; label: string}> = [
    {id: 'bundles', label: 'Bundles'},
    {id: 'utilization', label: 'Utilization'},
    {id: 'name', label: 'Name'},
  ];
  readonly groupingOptions: ReadonlyArray<{
    id: KernelGrouping;
    label: string;
  }> = [
    {id: 'module', label: 'Module'},
    {id: 'kind', label: 'Kind'},
    {id: 'none', label: 'None'},
  ];
  /** Kernel that Enter opens from the search box, moved with the arrow keys. */
  cursorEntry: KernelEntry | undefined = undefined;
  /** Open kernel tabs, in display order. */
  tabs: KernelTab[] = [];
  /** The tab whose kernel the timeline shows. */
  activeTab: KernelTab | undefined = undefined;
  /** A tab that was just opened in the background, or switched to. */
  highlightedTab: KernelTab | undefined = undefined;
  /** Details card of the hovered kernel row or tab. */
  hoverCard: KernelCard | undefined = undefined;
  /** Context menu of a right-clicked kernel row or tab. */
  contextMenu: KernelContextMenu | undefined = undefined;
  url = '';
  loadingKernels = false;
  loadFailed = false;
  filterQuery = '';

  /** Whether the kernel rail is docked open next to the timeline. */
  isRailPinned = readRailPinned();
  /** Whether the unpinned kernel rail is expanded over the timeline. */
  isRailPeeking = false;
  /** The Clipboard API is only available in secure contexts. */
  readonly canCopyLink = navigator.clipboard !== undefined;
  linkCopied = false;

  /**
   * Whether to show the Angular kernel event tooltip on the timeline. The
   * built-in Trace Viewer event tooltip is always off (see onInitializeWasm).
   */
  enableEventTooltip = false;

  traceViewerModule: TraceViewerV2Module | null = null;
  isSelectionPanelOpen = false;

  /** Whether the guided tour is showing. */
  isTourOpen = false;
  /** Whether the tour holds the kernel rail open while it explains it. */
  isRailHeldOpen = false;
  /** Whether the tour button shows that the tour can be replayed from it. */
  isTourButtonPulsing = false;

  @ViewChild('searchInput') searchInput?: ElementRef<HTMLInputElement>;
  @ViewChild(KernelSelectionPanel) selectionPanel?: KernelSelectionPanel;
  @ViewChild(KernelMinimap) minimap?: KernelMinimap;
  @ViewChild(KernelViewerTour) tour?: KernelViewerTour;
  @ViewChild('tourButton', {read: ElementRef})
  tourButton?: ElementRef<HTMLElement>;

  /** IDs of the sections whose kernels are hidden, see `KernelSection`. */
  private readonly collapsedSections = new Set<string>();
  /** Kinds of the listed kernels, in filter order. */
  private kinds: readonly KernelKind[] = [];
  /** Busiest units of the listed kernels, most common first. */
  private units: readonly string[] = [];
  /** Kernels that pass the filters, in display order. */
  private navigableEntries: readonly KernelEntry[] = [];
  /** All kernels in display order, ignoring the filters. */
  private orderedEntries: readonly KernelEntry[] = [];
  private searchTerms: SearchTerms = {include: [], exclude: []};
  /** Highlighted runs of kernel and module names, by name. */
  private readonly nameSegmentCache = new Map<string, readonly NameSegment[]>();
  private listKey = '';
  private listRequestId = 0;
  private requestedKernel = '';
  private requestedModule = '';
  private isPointerInRail = false;
  private isFocusInRail = false;
  /** Whether the last focused element was focused with the keyboard. */
  private isKeyboardFocus = false;
  private nextTabId = 0;
  private tabClock = 0;
  /** Number of tabs opened in the background since the last tab switch. */
  private backgroundTabCount = 0;
  private railTimer: ReturnType<typeof setTimeout> | undefined = undefined;
  private cardTimer: ReturnType<typeof setTimeout> | undefined = undefined;
  private highlightTimer: ReturnType<typeof setTimeout> | undefined = undefined;
  private linkCopiedTimer: ReturnType<typeof setTimeout> | undefined =
    undefined;
  private tourTimer: ReturnType<typeof setTimeout> | undefined = undefined;
  private tourButtonTimer: ReturnType<typeof setTimeout> | undefined =
    undefined;
  /** Whether the tour was considered for this visit, so it starts once. */
  private hasConsideredTour = false;
  private isInitializing = false;
  private isDestroyed = false;
  // Kernel trace URL currently loaded into the WASM instance.
  private lastLoadedUrl = '';
  // Set on a viewer load error so that re-selecting the same kernel retries.
  private lastLoadFailed = false;

  private readonly loadingStatusUpdateListener = (event: Event) => {
    const detail = (event as CustomEvent<{status?: unknown}>).detail;
    if (detail?.status === TraceViewerV2LoadingStatus.ERROR) {
      this.lastLoadFailed = true;
    }
  };

  private readonly dataService: DataServiceV2Interface = inject(
    DATA_SERVICE_INTERFACE_TOKEN,
  );
  private readonly location = inject(Location, {optional: true});
  private readonly changeDetectorRef = inject(ChangeDetectorRef);
  private readonly elementRef = inject<ElementRef<HTMLElement>>(ElementRef);
  private readonly route = inject(ActivatedRoute);
  private readonly store = inject<Store<{}>>(Store);
  private readonly destroyed = new ReplaySubject<void>(1);
  private readonly keydownListener = (event: KeyboardEvent) => {
    this.onWindowKeydown(event);
  };
  private readonly pointerDownListener = (event: MouseEvent) => {
    this.onWindowPointerOrClick(event);
  };
  private readonly windowClickListener = (event: MouseEvent) => {
    this.onWindowPointerOrClick(event);
  };
  private readonly windowContextMenuListener = (event: MouseEvent) => {
    this.onWindowContextMenu(event);
  };

  get isRailExpanded(): boolean {
    return this.isRailPinned || this.isRailPeeking || this.isRailHeldOpen;
  }

  /** Whether search terms narrow down the kernel rail. */
  get hasQuery(): boolean {
    return this.filterQuery.trim() !== '';
  }

  /** Whether the kind, unit or spill filters narrow down the kernel rail. */
  get hasFacetFilters(): boolean {
    return (
      this.kindFilter !== '' || this.unitFilter.length > 0 || this.spillsOnly
    );
  }

  /** Number of active filters in the filter panel. */
  get panelFilterCount(): number {
    return this.unitFilter.length + (this.spillsOnly ? 1 : 0);
  }

  /** Whether the search or the filters narrow down the kernel rail. */
  get isFiltering(): boolean {
    return this.hasQuery || this.hasFacetFilters;
  }

  /** Whether the search, filters, sort order or grouping differ from default. */
  get hasNonDefaultSettings(): boolean {
    return (
      this.isFiltering ||
      this.sortOrder !== 'bundles' ||
      this.grouping !== 'module'
    );
  }

  /** Whether the kernels are of more than one kind, so kinds can be told apart. */
  get hasKindFilter(): boolean {
    return this.kindOptions.length > 1;
  }

  /** Whether the backend reports the kind of any kernel. */
  get hasKinds(): boolean {
    return this.kinds.length > 0;
  }

  /** Whether any kernel spills vector registers. */
  get hasSpillingKernels(): boolean {
    return this.entries.some((entry) => entry.hasSpills);
  }

  /** Announces the number of kernels that pass the search and filters. */
  get resultsStatus(): string {
    if (!this.isFiltering) {
      return '';
    }
    const count = this.visibleCount;
    return `${count} of ${this.entries.length} ${
      this.entries.length === 1 ? 'kernel' : 'kernels'
    } shown`;
  }

  /** Whether the search box has focus, which shows `cursorEntry`. */
  get isSearchFocused(): boolean {
    const input = this.searchInput?.nativeElement;
    return input !== undefined && document.activeElement === input;
  }

  /** The kernel that the timeline shows. */
  get selectedEntry(): KernelEntry | undefined {
    return this.activeTab?.entry;
  }

  /**
   * One-based position of the selected kernel among the kernels that pass the
   * filters, or 0 if none is selected or it is filtered out.
   */
  get selectedPosition(): number {
    const entry = this.selectedEntry;
    return entry ? this.navigableEntries.indexOf(entry) + 1 : 0;
  }

  /** Number of kernels that the stepper steps through. */
  get navigableCount(): number {
    return this.navigableEntries.length;
  }

  /** Position of the selected kernel in the stepper, e.g. `3 / 75`. */
  get positionLabel(): string {
    const position = this.selectedPosition;
    return `${position > 0 ? position : '–'} / ${this.navigableCount}`;
  }

  /** Whether the stepper can open another kernel. */
  get canStep(): boolean {
    const count = this.navigableEntries.length;
    return (
      count > 1 ||
      (count === 1 && this.navigableEntries[0] !== this.selectedEntry)
    );
  }

  /** The kind that `kindFilter` selects, if any. */
  get selectedKind(): KernelKind | undefined {
    return this.kinds.find((option) => option.id === this.kindFilter);
  }

  /** What the stepper steps through, e.g. `Pallas kernel`. */
  get stepNoun(): string {
    if (!this.isFiltering) {
      return 'kernel';
    }
    const kind = this.selectedKind;
    return kind && !this.hasQuery && this.panelFilterCount === 0
      ? kind.noun
      : 'matching kernel';
  }

  ngOnInit(): void {
    this.store.dispatch(setCurrentToolStateAction({currentTool: this.tool}));
    window.addEventListener(
      LOADING_STATUS_UPDATE_EVENT_NAME,
      this.loadingStatusUpdateListener,
    );

    combineLatest([this.route.params, this.route.queryParams])
      .pipe(takeUntil(this.destroyed))
      .subscribe(([params, queryParams]: [Params, Params]) => {
        this.onRouteChange(params, queryParams);
      });
  }

  ngAfterViewInit(): void {
    // The trace viewer container registers its own keydown listener in its
    // ngAfterViewInit, which runs before this one. Registering afterwards lets
    // the shortcuts below skip keys the container already handled.
    window.addEventListener('keydown', this.keydownListener);
    window.addEventListener('mousedown', this.pointerDownListener);
    window.addEventListener('click', this.windowClickListener);
    window.addEventListener('contextmenu', this.windowContextMenuListener);
  }

  ngOnDestroy(): void {
    this.isDestroyed = true;
    window.removeEventListener('keydown', this.keydownListener);
    window.removeEventListener('mousedown', this.pointerDownListener);
    window.removeEventListener('click', this.windowClickListener);
    window.removeEventListener('contextmenu', this.windowContextMenuListener);
    clearTimeout(this.railTimer);
    clearTimeout(this.cardTimer);
    clearTimeout(this.highlightTimer);
    clearTimeout(this.linkCopiedTimer);
    clearTimeout(this.tourTimer);
    clearTimeout(this.tourButtonTimer);
    this.destroyed.next();
    this.destroyed.complete();
    window.removeEventListener(
      LOADING_STATUS_UPDATE_EVENT_NAME,
      this.loadingStatusUpdateListener,
    );
    if (this.traceViewerModule !== null) {
      shutdownTraceViewerV2();
      this.traceViewerModule = null;
    }
  }

  async onInitializeWasm(): Promise<void> {
    if (this.isInitializing || this.traceViewerModule !== null) {
      return;
    }
    this.isInitializing = true;
    try {
      this.traceViewerModule = await traceViewerV2Main();
      if (this.isDestroyed) {
        if (this.traceViewerModule !== null) {
          shutdownTraceViewerV2();
          this.traceViewerModule = null;
        }
      } else {
        // The kernel backend ignores time ranges and always returns the whole
        // kernel, so incremental refetching would only re-download it.
        this.traceViewerModule?.SetIncrementalFetchEnabled?.(false);
        // The built-in event tooltip reads `name (duration)`, but kernel
        // events are measured in bundles, not time, and already show their
        // name. The counter tooltip (utilization values) is unaffected.
        this.traceViewerModule?.SetEventTooltipEnabled?.(false);
        this.lastLoadedUrl = '';
        this.lastLoadFailed = false;
        this.loadKernelTrace(this.url);
      }
    } catch (error) {
      console.error('Failed to initialize Trace Viewer V2 WASM module:', error);
    } finally {
      this.isInitializing = false;
    }
  }

  async loadKernelList(): Promise<void> {
    if (!this.sessionId) {
      return;
    }
    const requestId = ++this.listRequestId;
    this.loadingKernels = true;
    this.loadFailed = false;
    let groups: KernelGroup[] = [];
    let failed = false;
    try {
      const response: unknown = await firstValueFrom(
        this.dataService.getData(
          this.sessionId,
          this.tool,
          this.host,
          new Map<string, string>([['request_type', 'list']]),
        ),
      );
      groups = buildKernelGroups(response);
    } catch (error) {
      console.error('Failed to fetch kernel list:', error);
      failed = true;
    }
    if (this.isDestroyed || requestId !== this.listRequestId) {
      return;
    }
    this.loadingKernels = false;
    this.loadFailed = failed;
    this.setKernelGroups(groups);
    this.changeDetectorRef.markForCheck();
  }

  /**
   * Shows `entry` in the active tab, or switches to the tab that already shows
   * it. Opens the first tab if there is none.
   */
  selectEntry(entry: KernelEntry | undefined): void {
    if (!entry) {
      return;
    }
    const tab = this.findTab(entry);
    if (tab || !this.activeTab) {
      this.openEntry(entry);
      return;
    }
    this.activeTab.entry = entry;
    this.activeTab.lastUsed = ++this.tabClock;
    this.showEntry(entry);
  }

  /**
   * Opens `entry` in a new tab next to the active tab, or switches to the tab
   * that already shows it. A `background` tab opens without switching to it.
   */
  openEntry(entry: KernelEntry | undefined, background = false): void {
    if (!entry) {
      return;
    }
    let tab = this.findTab(entry);
    if (!tab) {
      if (this.tabs.length >= MAX_OPEN_TABS) {
        this.closeLeastRecentlyUsedTab();
      }
      tab = {id: this.nextTabId++, entry, lastUsed: ++this.tabClock};
      // Like links opened in the background, consecutive tabs open in order
      // after the active tab.
      const activeIndex = this.activeTab
        ? this.tabs.indexOf(this.activeTab)
        : -1;
      const index = Math.min(
        activeIndex + 1 + this.backgroundTabCount,
        this.tabs.length,
      );
      this.tabs.splice(index, 0, tab);
      if (background) {
        this.backgroundTabCount++;
      }
    }
    if (background) {
      this.highlightTab(tab);
    } else {
      this.activateTab(tab);
    }
  }

  /** Shows the kernel of `tab`; re-activating it only retries a failed load. */
  activateTab(tab: KernelTab | undefined): void {
    if (!tab || (tab === this.activeTab && !this.lastLoadFailed)) {
      return;
    }
    this.activeTab = tab;
    this.backgroundTabCount = 0;
    tab.lastUsed = ++this.tabClock;
    this.showEntry(tab.entry);
  }

  /**
   * Closes `tab`, and switches to its right neighbor if it was active. The
   * last tab stays open.
   */
  closeTab(tab: KernelTab): void {
    const index = this.tabs.indexOf(tab);
    if (index < 0 || this.tabs.length < 2) {
      return;
    }
    this.hideCard();
    this.closeContextMenu();
    this.tabs.splice(index, 1);
    if (tab === this.activeTab) {
      this.activeTab = undefined;
      this.activateTab(this.tabs[index] ?? this.tabs[index - 1]);
    }
  }

  /** Whether a tab shows `entry`. */
  isEntryOpen(entry: KernelEntry): boolean {
    return this.findTab(entry) !== undefined;
  }

  /**
   * Opens the next (`delta` = 1) or previous (`delta` = -1) kernel among the
   * kernels that pass the filters, in display order. From a kernel that is
   * filtered out, steps to the nearest kernel that passes them.
   */
  stepKernel(delta: number): void {
    const list = this.navigableEntries;
    const count = list.length;
    if (count === 0) {
      return;
    }
    this.hideCard();
    this.closeContextMenu();
    const selected = this.selectedEntry;
    const current = selected ? list.indexOf(selected) : -1;
    let next: KernelEntry | undefined = undefined;
    if (current >= 0) {
      next = list[(((current + delta) % count) + count) % count];
    } else if (selected && delta !== 0) {
      const order = this.orderedEntries;
      const position = order.indexOf(selected);
      next =
        delta > 0
          ? (list.find((entry) => order.indexOf(entry) > position) ?? list[0])
          : ([...list]
              .reverse()
              .find((entry) => order.indexOf(entry) < position) ??
            list[count - 1]);
    } else if (!selected) {
      next = delta > 0 ? list[0] : list[count - 1];
    }
    this.selectEntry(next);
  }

  onKernelRowClick(entry: KernelEntry, event: MouseEvent): void {
    this.hideCard();
    this.closeContextMenu();
    this.openEntry(entry, event.ctrlKey || event.metaKey);
  }

  /** Opens kernels in background tabs on middle click, like links. */
  onKernelRowAuxClick(entry: KernelEntry, event: MouseEvent): void {
    if (event.button === 1) {
      event.preventDefault();
      this.hideCard();
      this.closeContextMenu();
      this.openEntry(entry, true);
    }
  }

  onKernelRowContextMenu(entry: KernelEntry, event: MouseEvent): void {
    this.openContextMenu(entry, 'rail', event);
  }

  onTabClick(tab: KernelTab): void {
    this.hideCard();
    this.closeContextMenu();
    this.activateTab(tab);
  }

  /** Closes tabs on middle click, like browser tabs. */
  onTabAuxClick(tab: KernelTab, event: MouseEvent): void {
    if (event.button === 1) {
      event.preventDefault();
      this.closeTab(tab);
    }
  }

  onTabContextMenu(tab: KernelTab, event: MouseEvent): void {
    this.openContextMenu(tab.entry, 'tab', event);
  }

  /** Copies the kernel name of `entry` (or the context menu target) to the clipboard. */
  copyKernelName(
    entry: KernelEntry | undefined = this.contextMenu?.entry,
  ): void {
    this.closeContextMenu();
    if (!entry || !navigator.clipboard?.writeText) {
      return;
    }
    navigator.clipboard.writeText(entry.kernel).then(
      () => {},
      () => {
        // Clipboard access can be denied; there is nothing else to do.
      },
    );
  }

  /** Closes the kernel context menu. */
  closeContextMenu(): void {
    const wasRailMenu = this.contextMenu?.source === 'rail';
    this.contextMenu = undefined;
    if (wasRailMenu && !this.isPointerInRail && !this.isFocusInRail) {
      this.setRailPeeking(false, RAIL_COLLAPSE_DELAY_MS);
    }
  }

  onCloseTabClick(tab: KernelTab): void {
    const restoreFocus = this.isKeyboardFocus;
    this.closeTab(tab);
    if (restoreFocus) {
      // The focused close button is gone, so focus the tab that replaced it.
      setTimeout(() => {
        this.elementRef.nativeElement
          .querySelector<HTMLElement>('.kernel-tab.active .tab-button')
          ?.focus();
      });
    }
  }

  /** Keeps middle clicks on kernels and tabs from starting to autoscroll. */
  onMouseDown(event: MouseEvent): void {
    if (event.button === 1) {
      event.preventDefault();
    }
  }

  /** Shows the details card of a hovered or keyboard-focused kernel row. */
  onKernelRowHover(entry: KernelEntry, event: Event): void {
    const row = event.currentTarget;
    if (!(row instanceof HTMLElement) || !this.isHoverOrKeyboardFocus(event)) {
      return;
    }
    // Rows only show cards once the rail has expanded to show their names.
    this.scheduleCard(() =>
      this.isRailExpanded ? this.cardBesideRow(entry, row) : undefined,
    );
  }

  /** Shows the details card of a hovered or keyboard-focused tab. */
  onTabHover(tab: KernelTab, event: Event): void {
    const target = event.currentTarget;
    const tabElement =
      target instanceof HTMLElement ? target.closest('.kernel-tab') : null;
    if (!tabElement || !this.isHoverOrKeyboardFocus(event)) {
      return;
    }
    this.scheduleCard(() => this.cardBelowTab(tab, tabElement));
  }

  /** Hides the details card. */
  hideCard(): void {
    clearTimeout(this.cardTimer);
    this.cardTimer = undefined;
    this.hoverCard = undefined;
  }

  /** Hides the details card soon, unless the pointer moves to a neighbor. */
  hideCardSoon(): void {
    clearTimeout(this.cardTimer);
    this.cardTimer = setTimeout(() => {
      this.hideCard();
      this.changeDetectorRef.markForCheck();
    }, CARD_HIDE_DELAY_MS);
  }

  onRailScroll(): void {
    this.hideCardFrom('rail');
    this.closeContextMenuFrom('rail');
  }

  onTabListScroll(): void {
    this.hideCardFrom('tab');
    this.closeContextMenuFrom('tab');
  }

  onFilterChange(query: string): void {
    this.filterQuery = query;
    this.hideCardFrom('rail');
    this.closeContextMenuFrom('rail');
    this.updateSections();
    // Enter opens the top match, so point it out while searching.
    this.cursorEntry = this.hasQuery ? this.renderedEntries()[0] : undefined;
  }

  /** Clears the search terms. The kind, unit and spill filters stay. */
  clearFilter(): void {
    this.onFilterChange('');
  }

  /** Shows only kernels of `kind`, or kernels of all kinds for ''. */
  setKindFilter(kind: string): void {
    this.kindFilter = kind;
    writePreference(KIND_FILTER_STORAGE_KEY, kind);
    this.onFacetFilterChange();
  }

  /** Shows only kernels whose busiest unit is one of `units`, if any. */
  setUnitFilter(units: readonly string[]): void {
    this.unitFilter = units;
    writeListPreference(UNIT_FILTER_STORAGE_KEY, units);
    this.onFacetFilterChange();
  }

  /** Whether the rail shows the kernels whose busiest unit is `unit`. */
  isUnitSelected(unit: string): boolean {
    return this.unitFilter.includes(unit);
  }

  /** Adds `unit` to, or removes it from, the units that the rail shows. */
  toggleUnitFilter(unit: string): void {
    this.setUnitFilter(
      this.isUnitSelected(unit)
        ? this.unitFilter.filter((option) => option !== unit)
        : [...this.unitFilter, unit],
    );
  }

  /** Shows only kernels that spill vector registers, or all kernels. */
  setSpillsOnly(spillsOnly: boolean): void {
    this.spillsOnly = spillsOnly;
    writePreference(SPILLS_ONLY_STORAGE_KEY, String(spillsOnly));
    this.onFacetFilterChange();
  }

  /** Resets the filters, search terms, sort order and grouping to default. */
  clearAllFilters(): void {
    this.kindFilter = '';
    this.unitFilter = [];
    this.spillsOnly = false;
    this.sortOrder = 'bundles';
    this.grouping = 'module';
    writePreference(KIND_FILTER_STORAGE_KEY, '');
    writeListPreference(UNIT_FILTER_STORAGE_KEY, []);
    writePreference(SPILLS_ONLY_STORAGE_KEY, 'false');
    writePreference(SORT_STORAGE_KEY, 'bundles');
    writePreference(GROUPING_STORAGE_KEY, 'module');
    this.clearFilter();
  }

  setSortOrder(sortOrder: KernelSort): void {
    this.sortOrder = sortOrder;
    writePreference(SORT_STORAGE_KEY, sortOrder);
    this.onFacetFilterChange();
  }

  setGrouping(grouping: KernelGrouping): void {
    this.grouping = grouping;
    writePreference(GROUPING_STORAGE_KEY, grouping);
    this.onFacetFilterChange();
  }

  toggleFilterPanel(): void {
    this.isFilterPanelOpen = !this.isFilterPanelOpen;
    this.hideCardFrom('rail');
    this.closeContextMenuFrom('rail');
  }

  /** Shows or hides the search cursor as the search box gains or loses focus. */
  onSearchFocusChange(): void {
    this.changeDetectorRef.markForCheck();
  }

  onSearchKeydown(event: KeyboardEvent): void {
    if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
      event.preventDefault();
      this.moveCursor(event.key === 'ArrowDown' ? 1 : -1);
    } else if (event.key === 'Enter') {
      event.preventDefault();
      this.openEntry(this.cursorEntry, event.ctrlKey || event.metaKey);
      this.dismissRail();
    } else if (event.key === 'Escape') {
      event.preventDefault();
      if (this.contextMenu) {
        this.closeContextMenu();
      } else if (this.filterQuery) {
        this.clearFilter();
      } else {
        this.dismissRail();
      }
    }
  }

  /** Whether `entry` is the kernel that Enter opens from the search box. */
  isCursor(entry: KernelEntry): boolean {
    return this.isSearchFocused && entry === this.cursorEntry;
  }

  /** ID of the row of `entry`, for `aria-activedescendant`. */
  rowId(entry: KernelEntry): string {
    return `skv-kernel-row-${entry.index}`;
  }

  /**
   * Splits a kernel or module name into runs that do and do not match the
   * search terms, for highlighting.
   */
  nameSegments(name: string): readonly NameSegment[] {
    let segments = this.nameSegmentCache.get(name);
    if (!segments) {
      segments = splitMatches(name, this.searchTerms.include);
      this.nameSegmentCache.set(name, segments);
    }
    return segments;
  }

  /** Whether the kernels of `section` are listed; searching expands all. */
  isGroupExpanded(section: KernelSection): boolean {
    return this.hasQuery || !this.collapsedSections.has(section.id);
  }

  toggleGroup(section: KernelSection): void {
    this.hideCardFrom('rail');
    this.closeContextMenuFrom('rail');
    if (this.collapsedSections.has(section.id)) {
      this.collapsedSections.delete(section.id);
    } else {
      this.collapsedSections.add(section.id);
    }
    if (
      this.cursorEntry &&
      !this.renderedEntries().includes(this.cursorEntry)
    ) {
      this.cursorEntry = undefined;
    }
  }

  togglePin(): void {
    this.isRailPinned = !this.isRailPinned;
    // Collapse right away when unpinning, like a browser's vertical tabs: the
    // rail only expands again once the pointer re-enters it.
    this.isRailPeeking = false;
    clearTimeout(this.railTimer);
    if (!this.isRailPinned) {
      this.clearFilter();
      this.hideCardFrom('rail');
      this.closeContextMenuFrom('rail');
    }
    writeRailPinned(this.isRailPinned);
    this.notifyResize();
  }

  onRailMouseEnter(): void {
    this.isPointerInRail = true;
    if (this.contextMenu?.source !== 'rail') {
      this.setRailPeeking(true, RAIL_EXPAND_DELAY_MS);
    }
  }

  onRailMouseLeave(): void {
    this.isPointerInRail = false;
    if (this.contextMenu?.source === 'rail') {
      clearTimeout(this.railTimer);
      this.railTimer = undefined;
      return;
    }
    if (!this.isFocusInRail) {
      this.setRailPeeking(false, RAIL_COLLAPSE_DELAY_MS);
    }
  }

  onRailFocusIn(event: FocusEvent): void {
    // Keep the rail open while typing a search or navigating with the
    // keyboard, but not after a mouse click on a kernel.
    const target = event.target;
    this.isFocusInRail =
      target instanceof HTMLElement &&
      (target.tagName === 'INPUT' || target.matches(':focus-visible'));
    if (this.isFocusInRail) {
      this.setRailPeeking(true, 0);
    }
  }

  onRailFocusOut(event: FocusEvent): void {
    const rail = event.currentTarget;
    const next = event.relatedTarget;
    if (rail instanceof Node && next instanceof Node && rail.contains(next)) {
      return;
    }
    this.isFocusInRail = false;
    if (!this.isPointerInRail && this.contextMenu?.source !== 'rail') {
      this.setRailPeeking(false, RAIL_COLLAPSE_DELAY_MS);
    }
  }

  onLayoutFocusIn(event: FocusEvent): void {
    const target = event.target;
    this.isKeyboardFocus =
      target instanceof HTMLElement &&
      (isEditableTarget(target) || target.matches(':focus-visible'));
  }

  /** Expands the kernel rail and focuses its search box. */
  focusSearch(): void {
    this.hideCard();
    this.closeContextMenu();
    this.setRailPeeking(true, 0);
    this.searchInput?.nativeElement.focus();
    this.searchInput?.nativeElement.select();
  }

  /** Copies the current browser URL to the clipboard. */
  copyKernelLink(): void {
    if (!this.selectedEntry || !this.canCopyLink) {
      return;
    }
    navigator.clipboard.writeText(window.location.href).then(
      () => {
        if (this.isDestroyed) {
          return;
        }
        this.linkCopied = true;
        clearTimeout(this.linkCopiedTimer);
        this.linkCopiedTimer = setTimeout(() => {
          this.linkCopied = false;
          this.changeDetectorRef.markForCheck();
        }, LINK_COPIED_FEEDBACK_MS);
        this.changeDetectorRef.markForCheck();
      },
      () => {
        // Clipboard access can be denied; there is nothing else to do.
      },
    );
  }

  /** Starts the guided tour of the viewer. */
  startTour(): void {
    const tour = this.tour;
    if (!tour) {
      return;
    }
    clearTimeout(this.tourTimer);
    this.hideCard();
    this.closeContextMenu();
    this.isTourOpen = true;
    tour.start(this.buildTourSummary(), {
      returnFocus: this.tourButton?.nativeElement,
    });
  }

  /** Holds the kernel rail open while the tour explains it. */
  onTourStepChange(stepId: string): void {
    const holdRail = stepId === 'rail' || stepId === 'search';
    if (holdRail === this.isRailHeldOpen) {
      return;
    }
    this.isRailHeldOpen = holdRail;
    if (!holdRail) {
      this.collapseRailAfterTour();
    }
    this.changeDetectorRef.markForCheck();
  }

  /** Lets go of the rail, and shows where the tour can be replayed from. */
  onTourClosed(): void {
    this.isTourOpen = false;
    if (this.isRailHeldOpen) {
      this.isRailHeldOpen = false;
      this.collapseRailAfterTour();
    }
    this.isTourButtonPulsing = true;
    clearTimeout(this.tourButtonTimer);
    this.tourButtonTimer = setTimeout(() => {
      this.isTourButtonPulsing = false;
      this.changeDetectorRef.markForCheck();
    }, TOUR_BUTTON_PULSE_MS);
    this.changeDetectorRef.markForCheck();
  }

  trackBySection(index: number, section: KernelSection): string {
    return section.id;
  }

  trackByOption(index: number, option: KernelFilterOption): string {
    return option.id;
  }

  trackByEntry(index: number, entry: KernelEntry): string {
    return `${entry.module}/${entry.kernel}`;
  }

  trackByTab(index: number, tab: KernelTab): number {
    return tab.id;
  }

  private onRouteChange(params: Params, queryParams: Params): void {
    const searchParams = this.dataService.getSearchParams?.();
    const getParam = (key: string): string | undefined => {
      const fromRoute = queryParams[key];
      if (typeof fromRoute === 'string' && fromRoute) {
        return fromRoute;
      }
      const fromSearch = searchParams?.get(key);
      return fromSearch ? fromSearch : undefined;
    };

    this.sessionId =
      params['sessionId'] ??
      getParam('run') ??
      getParam('sessionId') ??
      this.sessionId;
    this.host = getParam('host') ?? DEFAULT_HOST;
    const hasRouteKernelOrModule =
      queryParams['kernel_name'] !== undefined ||
      queryParams['kernel'] !== undefined ||
      queryParams['hlo_module'] !== undefined;
    this.requestedKernel = hasRouteKernelOrModule
      ? (queryParams['kernel_name'] ?? queryParams['kernel'] ?? '')
      : (getParam('kernel_name') ?? getParam('kernel') ?? '');
    this.requestedModule = hasRouteKernelOrModule
      ? (queryParams['hlo_module'] ?? '')
      : (getParam('hlo_module') ?? '');

    // Only refetch the kernel list when the profile changes. Other query
    // parameter changes at most switch the open kernel.
    const listKey = `${this.sessionId}|${this.host}`;
    if (listKey !== this.listKey) {
      this.listKey = listKey;
      void this.loadKernelList();
    } else {
      this.selectEntry(this.findRequestedEntry());
    }
  }

  private setKernelGroups(groups: KernelGroup[]): void {
    this.groups = groups;
    this.entries = groups.flatMap((group) => [...group.entries]);
    this.collapsedSections.clear();
    this.kinds = this.listKinds();
    this.units = this.listUnits();
    // Only apply stored filters that this profile's kernels offer.
    const savedKind = readStringPreference(KIND_FILTER_STORAGE_KEY);
    this.kindFilter =
      this.kinds.length > 1 && this.kinds.some((kind) => kind.id === savedKind)
        ? savedKind
        : '';
    this.unitFilter = readListPreference(UNIT_FILTER_STORAGE_KEY).filter(
      (unit) => this.units.includes(unit),
    );
    this.spillsOnly =
      readBooleanPreference(SPILLS_ONLY_STORAGE_KEY) && this.hasSpillingKernels;
    this.cursorEntry = undefined;
    this.updateSections();
    this.tabs = [];
    this.activeTab = undefined;
    this.highlightedTab = undefined;
    this.backgroundTabCount = 0;
    this.hideCard();
    this.closeContextMenu();
    this.url = '';
    const initialEntry =
      this.findRequestedEntry() ??
      this.navigableEntries[0] ??
      this.orderedEntries[0];
    if (initialEntry) {
      this.selectEntry(initialEntry);
    } else {
      this.updateUrlQueryParams();
    }
    this.considerTour();
  }

  /** Returns the kinds of the listed kernels, in filter order. */
  private listKinds(): KernelKind[] {
    const kinds = new Map<string, KernelKind>();
    for (const entry of this.entries) {
      if (entry.kind) {
        kinds.set(entry.kind.id, entry.kind);
      }
    }
    return [...kinds.values()].sort(
      (a, b) => a.order - b.order || a.label.localeCompare(b.label),
    );
  }

  /** Returns the busiest units of the listed kernels, most common first. */
  private listUnits(): string[] {
    const counts = new Map<string, number>();
    for (const entry of this.entries) {
      if (entry.boundUnit) {
        counts.set(entry.boundUnit, (counts.get(entry.boundUnit) ?? 0) + 1);
      }
    }
    return [...counts.keys()].sort(
      (a, b) =>
        (counts.get(b) ?? 0) - (counts.get(a) ?? 0) || a.localeCompare(b),
    );
  }

  /** Starts the tour soon on the first visit to a profile with kernels. */
  private considerTour(): void {
    if (
      this.hasConsideredTour ||
      this.entries.length === 0 ||
      hasSeenKernelViewerTour()
    ) {
      return;
    }
    this.hasConsideredTour = true;
    this.tourTimer = setTimeout(() => {
      if (!this.isDestroyed && !this.isTourOpen) {
        this.startTour();
        this.changeDetectorRef.markForCheck();
      }
    }, TOUR_AUTO_START_DELAY_MS);
  }

  /** Summarizes the profile for the tour, which features its kernels. */
  private buildTourSummary(): KernelViewerTourSummary {
    const topKernels: TourKernel[] = this.entries
      .filter((entry) => entry.stats !== undefined)
      .sort((a, b) => (b.stats?.bundles ?? 0) - (a.stats?.bundles ?? 0))
      .slice(0, 4)
      .map((entry) => ({
        name: entry.kernel,
        module: entry.module,
        monogram: entry.monogram,
        color: entry.color,
        bundles: entry.stats?.bundles ?? 0,
        bundlesLabel: entry.bundlesLabel,
        busiestUnit: entry.busiestUnit,
        busiestUnitColor: entry.busiestUnitColor,
      }));
    return {
      kernelCount: this.entries.length,
      moduleCount: this.groups.filter((group) => group.module).length,
      topKernels,
      backgroundClickKey: BACKGROUND_CLICK_KEY,
      canCopyLink: this.canCopyLink,
    };
  }

  /** Collapses the rail once the tour lets go of it, unless it is in use. */
  private collapseRailAfterTour(): void {
    if (this.isRailExpanded) {
      return;
    }
    // Like when the rail collapses on its own, the search only narrows down
    // the rail while it is open.
    this.clearFilter();
    this.hideCardFrom('rail');
    this.closeContextMenuFrom('rail');
    this.releasePointerFocus('.kernel-rail');
  }

  private findRequestedEntry(): KernelEntry | undefined {
    if (!this.requestedKernel) {
      return undefined;
    }
    const matches = this.entries.filter(
      (entry) => entry.kernel === this.requestedKernel,
    );
    return (
      matches.find((entry) => entry.module === this.requestedModule) ??
      matches[0]
    );
  }

  private findTab(entry: KernelEntry): KernelTab | undefined {
    return this.tabs.find((tab) => tab.entry === entry);
  }

  onSelectionPanelOpenChange(isOpen: boolean): void {
    if (this.isSelectionPanelOpen === isOpen) {
      return;
    }
    this.isSelectionPanelOpen = isOpen;
    this.changeDetectorRef.markForCheck();
    this.notifyResize();
  }

  onSelectTimelineEvent(eventIndex: number): void {
    this.traceViewerModule?.application
      ?.instance?.()
      ?.selectEvent?.(eventIndex);
  }

  /**
   * Opens the details panel for the tour, which teaches it to users who never
   * clicked a block. For an instruction, selects the most frequent instruction
   * of the longest region or scope, or else of the first block of the
   * timeline. For a region, selects the longest region.
   */
  onTourOpenDetails(kind: DetailsKind): void {
    const timeline = this.traceViewerModule?.application?.instance?.();
    // The minimap reports the regions once the trace shows.
    const state = this.minimap?.state;
    if (!timeline?.selectEvent || !state) {
      return;
    }
    if (kind === 'region') {
      // Only blocks of the Regions lane show their details as a region.
      const region = longestMarker(state.regions);
      if (region) {
        timeline.selectEvent(region.eventIndex);
      }
      return;
    }
    const longest = longestMarker([
      ...state.regions,
      ...state.pallasPrimitives,
    ]);
    timeline.selectEvent(longest?.eventIndex ?? 0);
    // Trace Viewer v2 reports the selection right away, and the details of a
    // region or scope list its most frequent instructions.
    const top = this.selectionPanel?.content?.topInstructions?.find(
      (item) => item.firstEventIndex >= 0,
    );
    if (top) {
      timeline.selectEvent(top.firstEventIndex);
    }
  }

  onSearchTimelineQuery(query: string): void {
    this.traceViewerModule?.application?.instance?.()?.setSearchQuery?.(query);
  }

  onVisibleRangeChange(range: {startUs: number; endUs: number}): void {
    this.traceViewerModule?.application
      ?.instance?.()
      ?.setVisibleRange?.(range.startUs, range.endUs);
  }
  /** Loads the trace of `entry` into the timeline. */
  private showEntry(entry: KernelEntry): void {
    this.selectionPanel?.close();
    this.collapsedSections.delete(this.sectionIdOf(entry));
    this.updateUrlQueryParams();
    const queryParamsMap = new Map<string, string>([
      ['request_type', 'trace'],
      ['kernel_name', entry.kernel],
    ]);
    // Kernel names are only unique within a module.
    if (entry.module) {
      queryParamsMap.set('hlo_module', entry.module);
    }
    this.url = this.dataService.getDataUrl(
      this.sessionId,
      this.tool,
      this.host,
      queryParamsMap,
    );
    this.loadKernelTrace(this.url);
    this.scrollIntoView('.kernel-row.active', '.kernel-tab.active');
  }

  /**
   * Loads `url` into the WASM viewer. Skips a URL that is already loaded
   * (unless its load failed) and resets the viewer's per-dataset state before
   * loading a different one.
   */
  private loadKernelTrace(url: string): void {
    const module = this.traceViewerModule;
    if (!url || !module?.loadTraceData) {
      return;
    }
    if (url === this.lastLoadedUrl && !this.lastLoadFailed) {
      return;
    }
    if (this.lastLoadedUrl) {
      module.ResetDataset?.();
    }
    this.lastLoadedUrl = url;
    this.lastLoadFailed = false;
    void module.loadTraceData(url);
  }

  private updateUrlQueryParams(): void {
    const entry = this.selectedEntry;
    const applyParams = (params: URLSearchParams) => {
      if (entry?.module) {
        params.set('hlo_module', entry.module);
      } else {
        params.delete('hlo_module');
      }

      if (entry?.kernel) {
        params.set('kernel_name', entry.kernel);
      } else {
        params.delete('kernel_name');
      }
      params.delete('kernel');
    };

    if (this.dataService.getSearchParams && this.dataService.setSearchParams) {
      const searchParams = this.dataService.getSearchParams();
      applyParams(searchParams);
      this.dataService.setSearchParams(searchParams);
    }

    if (this.location) {
      const url = new URL(this.location.path(), window.location.origin);
      const searchParams = new URLSearchParams(url.search);
      applyParams(searchParams);
      const newSearch = searchParams.toString();
      const currentSearch = url.search.startsWith('?')
        ? url.search.slice(1)
        : url.search;
      if (currentSearch !== newSearch) {
        this.location.replaceState(url.pathname, decodeURIComponent(newSearch));
      }
    }
  }

  private closeLeastRecentlyUsedTab(): void {
    let leastRecent: KernelTab | undefined = undefined;
    for (const tab of this.tabs) {
      if (
        tab !== this.activeTab &&
        (!leastRecent || tab.lastUsed < leastRecent.lastUsed)
      ) {
        leastRecent = tab;
      }
    }
    if (leastRecent) {
      this.closeTab(leastRecent);
    }
  }

  /** Briefly highlights `tab`, e.g. after opening it in the background. */
  private highlightTab(tab: KernelTab): void {
    this.highlightedTab = tab;
    clearTimeout(this.highlightTimer);
    this.highlightTimer = setTimeout(() => {
      this.highlightedTab = undefined;
      this.changeDetectorRef.markForCheck();
    }, TAB_HIGHLIGHT_MS);
    this.scrollIntoView(undefined, '.kernel-tab.highlighted');
  }

  /**
   * Applies the search terms and the kind, unit and spill filters to the
   * kernels, and lays the kernels that pass them out in sections. The count of
   * each filter option takes all other filters into account, so it tells how
   * many kernels the option would show.
   */
  private updateSections(): void {
    this.searchTerms = parseSearchTerms(this.filterQuery);
    this.nameSegmentCache.clear();
    const kindFilter = this.kindFilter;
    const unitFilter = new Set(this.unitFilter);
    const matching = this.entries.filter((entry) =>
      matchesSearchTerms(entry, this.searchTerms),
    );
    const passesKind = (entry: KernelEntry) =>
      !kindFilter || entry.kind?.id === kindFilter;
    const passesUnit = (entry: KernelEntry) =>
      unitFilter.size === 0 || unitFilter.has(entry.boundUnit);
    const passesSpills = (entry: KernelEntry) =>
      !this.spillsOnly || entry.hasSpills;

    const kindCounts = new Map<string, number>();
    const unitCounts = new Map<string, number>();
    const visible = new Set<KernelEntry>();
    this.allKindsCount = 0;
    this.spillsCount = 0;
    for (const entry of matching) {
      const kind = passesKind(entry);
      const unit = passesUnit(entry);
      const spills = passesSpills(entry);
      if (unit && spills) {
        this.allKindsCount++;
        if (entry.kind) {
          kindCounts.set(
            entry.kind.id,
            (kindCounts.get(entry.kind.id) ?? 0) + 1,
          );
        }
      }
      if (kind && spills && entry.boundUnit) {
        unitCounts.set(
          entry.boundUnit,
          (unitCounts.get(entry.boundUnit) ?? 0) + 1,
        );
      }
      if (kind && unit && entry.hasSpills) {
        this.spillsCount++;
      }
      if (kind && unit && spills) {
        visible.add(entry);
      }
    }
    this.kindOptions = this.kinds.map((kind) => ({
      id: kind.id,
      label: kind.label,
      icon: kind.icon,
      color: kind.color,
      count: kindCounts.get(kind.id) ?? 0,
    }));
    this.unitOptions = this.units.map((unit) => ({
      id: unit,
      label: unit,
      color: unitColor(unit),
      count: unitCounts.get(unit) ?? 0,
    }));
    this.sections = this.buildSections((entry) => visible.has(entry));
    this.visibleCount = visible.size;
    this.navigableEntries = this.sections.flatMap((section) => [
      ...section.entries,
    ]);
    this.orderedEntries = this.isFiltering
      ? this.buildSections(() => true).flatMap((section) => [
          ...section.entries,
        ])
      : this.navigableEntries;
  }

  /** Lays out the kernels that pass `include` in sections, in display order. */
  private buildSections(
    include: (entry: KernelEntry) => boolean,
  ): KernelSection[] {
    const compare = kernelComparator(this.sortOrder);
    if (this.grouping === 'none') {
      const entries = this.entries.filter(include).sort(compare);
      return entries.length > 0
        ? [
            {
              id: 'all',
              label: '',
              color: 'var(--mat-sys-outline)',
              isModule: false,
              entries,
            },
          ]
        : [];
    }
    if (this.grouping === 'kind') {
      const kinds: Array<KernelKind | undefined> = [...this.kinds, undefined];
      return kinds
        .map((kind) => ({
          id: `kind:${kind?.id ?? ''}`,
          label: kind?.label ?? 'Other',
          color: kind?.color ?? OTHER_KIND_COLOR,
          isModule: false,
          icon: kind?.icon ?? OTHER_KIND_ICON,
          entries: this.entries
            .filter((entry) => entry.kind?.id === kind?.id && include(entry))
            .sort(compare),
        }))
        .filter((section) => section.entries.length > 0);
    }
    const groups =
      this.sortOrder === 'name'
        ? [...this.groups].sort(
            (a, b) =>
              Number(!a.module) - Number(!b.module) ||
              NAME_COLLATOR.compare(a.module, b.module),
          )
        : this.groups;
    return groups
      .map((group) => ({
        id: `module:${group.module}`,
        label: group.module,
        color: group.color,
        isModule: true,
        entries: group.entries.filter(include).sort(compare),
      }))
      .filter((section) => section.entries.length > 0);
  }

  /** ID of the section that lists `entry`, see `KernelSection`. */
  private sectionIdOf(entry: KernelEntry): string {
    switch (this.grouping) {
      case 'kind':
        return `kind:${entry.kind?.id ?? ''}`;
      case 'none':
        return 'all';
      default:
        return `module:${entry.module}`;
    }
  }

  /** Kernels whose rows are showing, i.e. not in collapsed sections. */
  private renderedEntries(): KernelEntry[] {
    return this.sections
      .filter((section) => this.isGroupExpanded(section))
      .flatMap((section) => [...section.entries]);
  }

  /** Re-applies the filters after a kind, unit, spill, sort or group change. */
  private onFacetFilterChange(): void {
    this.hideCardFrom('rail');
    this.closeContextMenuFrom('rail');
    this.updateSections();
    const rendered = this.renderedEntries();
    if (this.cursorEntry && !rendered.includes(this.cursorEntry)) {
      this.cursorEntry = this.hasQuery ? rendered[0] : undefined;
    }
  }

  /** Moves the kernel that Enter opens to the next or previous row. */
  private moveCursor(delta: number): void {
    const rendered = this.renderedEntries();
    const count = rendered.length;
    if (count === 0) {
      return;
    }
    const current = this.cursorEntry ? rendered.indexOf(this.cursorEntry) : -1;
    const next =
      current < 0
        ? delta > 0
          ? 0
          : count - 1
        : (((current + delta) % count) + count) % count;
    this.cursorEntry = rendered[next];
    this.hideCardFrom('rail');
    this.scrollIntoView('.kernel-row.cursor');
  }

  private setRailPeeking(peeking: boolean, delayMs: number): void {
    clearTimeout(this.railTimer);
    this.railTimer = undefined;
    if (this.isRailPinned || this.isRailPeeking === peeking) {
      return;
    }
    const apply = () => {
      this.isRailPeeking = peeking;
      if (!peeking) {
        // The search only narrows down the rail while it is open.
        this.clearFilter();
        this.hideCardFrom('rail');
        this.closeContextMenuFrom('rail');
        this.releasePointerFocus('.kernel-rail');
      }
      this.changeDetectorRef.markForCheck();
    };
    if (delayMs > 0) {
      this.railTimer = setTimeout(apply, delayMs);
    } else {
      apply();
    }
  }

  private dismissRail(): void {
    this.isFocusInRail = false;
    this.searchInput?.nativeElement.blur();
    this.clearFilter();
    clearTimeout(this.railTimer);
    this.isRailPeeking = false;
    this.hideCardFrom('rail');
    this.closeContextMenuFrom('rail');
  }

  /** Whether `event` is a pointer hover, or focus from the keyboard. */
  private isHoverOrKeyboardFocus(event: Event): boolean {
    const target = event.target;
    return (
      event.type !== 'focus' ||
      (target instanceof HTMLElement && target.matches(':focus-visible'))
    );
  }

  /** Shows a details card, right away if another card is showing. */
  private scheduleCard(build: () => KernelCard | undefined): void {
    clearTimeout(this.cardTimer);
    this.cardTimer = undefined;
    // The tour covers the page, and its card takes the place of details cards.
    if (this.isTourOpen || this.contextMenu) {
      return;
    }
    const show = () => {
      this.cardTimer = undefined;
      this.hoverCard = build();
      this.changeDetectorRef.markForCheck();
    };
    if (this.hoverCard) {
      show();
    } else {
      this.cardTimer = setTimeout(show, CARD_SHOW_DELAY_MS);
    }
  }

  private hideCardFrom(source: KernelCard['source']): void {
    if (this.hoverCard?.source === source) {
      this.hideCard();
    }
  }

  private openContextMenu(
    entry: KernelEntry,
    source: KernelContextMenu['source'],
    event: MouseEvent,
  ): void {
    event.preventDefault();
    this.hideCard();
    if (source === 'rail') {
      clearTimeout(this.railTimer);
      this.railTimer = undefined;
    } else if (this.contextMenu?.source === 'rail') {
      this.closeContextMenu();
    }
    const host = this.elementRef.nativeElement.getBoundingClientRect();
    const target =
      event.currentTarget instanceof HTMLElement ? event.currentTarget : null;
    const targetRect = target?.getBoundingClientRect();
    const hasPointerPosition = event.clientX !== 0 || event.clientY !== 0;
    const rawLeft = hasPointerPosition
      ? event.clientX - host.left
      : (targetRect?.left ?? host.left) - host.left + CARD_GAP;
    let rawTop = event.clientY - host.top;
    if (!hasPointerPosition) {
      const belowTop = (targetRect?.bottom ?? host.top) - host.top + 4;
      const aboveTop =
        (targetRect?.top ?? host.top) - host.top - CONTEXT_MENU_HEIGHT - 4;
      rawTop =
        belowTop + CONTEXT_MENU_HEIGHT + CARD_GAP > host.height &&
        aboveTop >= CARD_GAP
          ? aboveTop
          : belowTop;
    }
    const left =
      host.width > CONTEXT_MENU_WIDTH + 2 * CARD_GAP
        ? Math.min(
            Math.max(rawLeft, CARD_GAP),
            host.width - CONTEXT_MENU_WIDTH - CARD_GAP,
          )
        : Math.max(0, rawLeft);
    const top =
      host.height > CONTEXT_MENU_HEIGHT + 2 * CARD_GAP
        ? Math.min(
            Math.max(rawTop, CARD_GAP),
            host.height - CONTEXT_MENU_HEIGHT - CARD_GAP,
          )
        : Math.max(0, rawTop);
    this.contextMenu = {entry, source, left, top};
  }

  private closeContextMenuFrom(source: KernelContextMenu['source']): void {
    if (this.contextMenu?.source === source) {
      this.closeContextMenu();
    }
  }

  private isInsideContextMenu(target: EventTarget | null): boolean {
    if (!(target instanceof Node)) {
      return false;
    }
    return Boolean(
      this.elementRef.nativeElement
        .querySelector('.kernel-context-menu')
        ?.contains(target),
    );
  }

  private onWindowPointerOrClick(event: MouseEvent): void {
    if (!this.contextMenu || this.isInsideContextMenu(event.target)) {
      return;
    }
    this.closeContextMenu();
    this.changeDetectorRef.markForCheck();
  }

  private onWindowContextMenu(event: MouseEvent): void {
    if (!this.contextMenu) {
      return;
    }
    if (
      event.target instanceof Element &&
      this.elementRef.nativeElement.contains(event.target) &&
      event.target.closest('.kernel-row, .kernel-tab, .kernel-context-menu')
    ) {
      return;
    }
    this.closeContextMenu();
    this.changeDetectorRef.markForCheck();
  }

  private cardBesideRow(entry: KernelEntry, row: HTMLElement): KernelCard {
    const host = this.elementRef.nativeElement.getBoundingClientRect();
    const rowRect = row.getBoundingClientRect();
    const railRight =
      row.closest('.kernel-rail')?.getBoundingClientRect().right ??
      rowRect.right;
    const left = railRight - host.left + CARD_GAP;
    const top = rowRect.top - host.top;
    const bottom = host.bottom - rowRect.bottom;
    const tab = this.findTab(entry);
    let hintParts: readonly HintPart[] = [
      {text: BACKGROUND_CLICK_KEY, isKey: true},
      {text: '+click to open in a background tab', isKey: false},
    ];
    if (tab === this.activeTab) {
      hintParts = [];
    } else if (tab) {
      hintParts = [
        {
          text: `Already open in tab ${this.tabs.indexOf(tab) + 1}`,
          isKey: false,
        },
      ];
    }
    // Grow the card away from the closer edge, so that it fits.
    return top < bottom
      ? {entry, source: 'rail', ...buildCardHint(hintParts), left, top}
      : {entry, source: 'rail', ...buildCardHint(hintParts), left, bottom};
  }

  private cardBelowTab(tab: KernelTab, tabElement: Element): KernelCard {
    const host = this.elementRef.nativeElement.getBoundingClientRect();
    const rect = tabElement.getBoundingClientRect();
    const maxLeft = Math.max(CARD_GAP, host.width - CARD_WIDTH - CARD_GAP);
    const hintParts: readonly HintPart[] =
      this.tabs.length > 1
        ? [
            {text: '[', isKey: true},
            {text: ' and ', isKey: false},
            {text: ']', isKey: true},
            {text: ' switch tabs', isKey: false},
          ]
        : [];
    return {
      entry: tab.entry,
      source: 'tab',
      ...buildCardHint(hintParts),
      left: Math.min(Math.max(rect.left - host.left, CARD_GAP), maxLeft),
      top: rect.bottom - host.top + CARD_GAP,
    };
  }

  /**
   * Blurs a kernel row or tab that was focused by a click, so that its focus
   * ring does not appear once keys are pressed. Keyboard focus stays.
   */
  private releasePointerFocus(scope: string): void {
    const focused = document.activeElement;
    if (
      this.isKeyboardFocus ||
      !(focused instanceof HTMLElement) ||
      isEditableTarget(focused) ||
      !focused.closest(scope) ||
      !this.elementRef.nativeElement.contains(focused)
    ) {
      return;
    }
    focused.blur();
  }

  private onWindowKeydown(event: KeyboardEvent): void {
    const wasContextMenuOpen = this.contextMenu !== undefined;
    if (
      this.contextMenu &&
      this.isInsideContextMenu(event.target) &&
      (event.key === 'Enter' || event.key === ' ') &&
      !event.ctrlKey &&
      !event.metaKey &&
      !event.altKey
    ) {
      event.preventDefault();
      this.copyKernelName(this.contextMenu.entry);
      this.changeDetectorRef.markForCheck();
      return;
    }
    if (
      this.contextMenu &&
      (event.key === 'ArrowDown' || event.key === 'ArrowUp') &&
      !event.ctrlKey &&
      !event.metaKey &&
      !event.altKey &&
      !isEditableTarget(event.target)
    ) {
      event.preventDefault();
      this.elementRef.nativeElement
        .querySelector<HTMLElement>('.copy-kernel-name-button')
        ?.focus();
      return;
    }
    if (!MODIFIER_KEYS.has(event.key)) {
      this.hideCard();
      if (event.key !== 'Tab') {
        this.closeContextMenu();
        this.releasePointerFocus('.kernel-rail, .kernel-bar');
      }
      this.changeDetectorRef.markForCheck();
    }
    if (wasContextMenuOpen && event.key === 'Escape') {
      event.preventDefault();
      return;
    }
    // The WASM viewer's capture listener calls preventDefault() on every key
    // once initialized, even keys it does not use (like K, [ and ]).
    if (
      (event.defaultPrevented && this.traceViewerModule === null) ||
      isEditableTarget(event.target) ||
      this.isHelpDialogOpen()
    ) {
      return;
    }
    if (event.ctrlKey || event.metaKey || event.altKey) {
      return;
    }
    if (
      (event.key === '[' || event.key === ']') &&
      !this.isSourceFrameNavOpen()
    ) {
      event.preventDefault();
      // Holding the key down would queue up a trace load per repeat.
      if (!event.repeat) {
        this.stepTab(event.key === ']' ? 1 : -1);
        this.changeDetectorRef.markForCheck();
      }
    } else if (event.key === 'k' || event.key === 'K') {
      event.preventDefault();
      this.focusSearch();
      this.changeDetectorRef.markForCheck();
    } else if (event.key === 'Escape' && this.isSelectionPanelOpen) {
      this.selectionPanel?.close();
      this.changeDetectorRef.markForCheck();
    }
  }

  /** Whether the shortcut help dialog (?) of Trace Viewer v2 is open. */
  private isHelpDialogOpen(): boolean {
    const dialog = this.elementRef.nativeElement.querySelector(
      'trace-viewer-help-dialog',
    ) as (Element & {open?: boolean}) | null;
    return Boolean(dialog?.open);
  }

  /** Whether the hovered source tooltip is cycling stack frames with [ / ]. */
  private isSourceFrameNavOpen(): boolean {
    return Boolean(
      this.elementRef.nativeElement.querySelector(
        'trace-viewer-container .hovered-tooltip-frame-nav',
      ),
    );
  }

  /** Switches to the next (`delta` = 1) or previous (`delta` = -1) open tab. */
  private stepTab(delta: number): void {
    const count = this.tabs.length;
    if (!this.activeTab || count < 2) {
      return;
    }
    const index = this.tabs.indexOf(this.activeTab);
    this.activateTab(this.tabs[(((index + delta) % count) + count) % count]);
  }

  /** Scrolls the matching kernel row and tab into view once rendered. */
  private scrollIntoView(rowSelector?: string, tabSelector?: string): void {
    setTimeout(() => {
      const host = this.elementRef.nativeElement;
      if (rowSelector) {
        host.querySelector(rowSelector)?.scrollIntoView({block: 'nearest'});
      }
      const tabList = host.querySelector<HTMLElement>('.tab-list');
      const tab = tabSelector
        ? tabList?.querySelector<HTMLElement>(tabSelector)
        : undefined;
      if (tabList && tab) {
        revealHorizontally(tabList, tab);
      }
    });
  }

  /**
   * Dispatches resize events immediately and after the rail width transition
   * to ensure the canvas recalculates its layout.
   */
  private notifyResize(): void {
    window.dispatchEvent(new Event('resize'));
    setTimeout(() => {
      window.dispatchEvent(new Event('resize'));
    }, RAIL_TRANSITION_MS + 10);
  }
}
