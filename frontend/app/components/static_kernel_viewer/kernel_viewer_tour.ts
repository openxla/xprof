import {
  ChangeDetectionStrategy,
  ChangeDetectorRef,
  Component,
  computed,
  ElementRef,
  inject,
  NgZone,
  OnDestroy,
  OnInit,
  output,
  signal,
  ViewChild,
} from '@angular/core';
import {MatButtonModule} from '@angular/material/button';
import {MatIconModule} from '@angular/material/icon';
import {
  DEFAULT_ACCENT_COLOR,
  FUNCTIONAL_UNITS_BY_LANE,
} from 'org_xprof/frontend/app/components/static_kernel_viewer/kernel_event_tooltip';
import {MINIMAP_UPDATED_EVENT_NAME} from 'org_xprof/frontend/app/components/static_kernel_viewer/kernel_minimap';
import {
  buildKernelSelectionFromEvent,
  buildKernelSelectionFromRange,
  KernelSelectionContent,
} from 'org_xprof/frontend/app/components/static_kernel_viewer/kernel_selection_panel';
import {
  BlockSighting,
  Box,
  buildDemoScript,
  bundleAt,
  DemoContext,
  DemoPose,
  DemoScript,
  GhostBox,
  hasDemo,
  locateSighting,
  loopDuration,
  Point,
  sampleDemo,
  Size,
  staticDemoPose,
  TimelineMapping,
} from 'org_xprof/frontend/app/components/static_kernel_viewer/kernel_viewer_tour_demo';
import {
  DEFAULT_DRAWER_SIZE_PERCENT,
  EVENT_HOVERED_EVENT_NAME,
  EVENT_SELECTED_EVENT_NAME,
  EVENTS_SELECTED_EVENT_NAME,
} from 'org_xprof/frontend/app/components/trace_viewer_container/trace_viewer_container';
import {
  LOADING_STATUS_UPDATE_EVENT_NAME,
  TraceViewerV2LoadingStatus,
} from 'org_xprof/frontend/app/components/trace_viewer_v2/main';

/** How the tour ended. */
export type TourCloseReason = 'completed' | 'dismissed';

/**
 * Where the tour card sits relative to its spotlight. `inside` puts it in a
 * corner of the spotlight, away from what the step points at.
 */
export type TourPlacement = 'right' | 'bottom' | 'left' | 'top' | 'inside';

/**
 * Illustration at the top of a tour card. `legend`, `fields`, `panel` and
 * `region` are live: they follow what the user points at and clicks.
 */
export type TourVisual =
  | 'welcome'
  | 'rail'
  | 'search'
  | 'tabs'
  | 'share'
  | 'legend'
  | 'keys'
  | 'fields'
  | 'panel'
  | 'region'
  | 'minimap'
  | 'done';

/**
 * A hands-on task of a tour step, done when the user presses one of its keys,
 * or does its gesture.
 */
export interface TourTask {
  /** What to do, e.g. `Try it: press`, followed by `keyLabels`. */
  readonly prompt: string;
  /** `KeyboardEvent.key` values that complete the task, in lowercase. */
  readonly keys?: readonly string[];
  /** Keys shown after the prompt, e.g. `['K']`. */
  readonly keyLabels?: readonly string[];
  /**
   * Gesture that completes the task: pointing at a block of the timeline,
   * selecting one, or moving the lens of the minimap.
   */
  readonly gesture?: 'hover' | 'select' | 'minimap';
  /** Shown once the task is done. */
  readonly success: string;
}

/** A step of the tour. */
export interface TourStep {
  readonly id: string;
  /** Short label above the title. */
  readonly eyebrow: string;
  readonly title: string;
  readonly body: string;
  readonly visual: TourVisual;
  /**
   * Selectors of the elements to spotlight. The backdrop cuts out the union of
   * the matching elements. The card is centered if no element matches.
   */
  readonly spotlight: readonly string[];
  /** Selectors of the elements to spotlight if `spotlight` matches nothing. */
  readonly fallback?: readonly string[];
  /** Selectors of the elements to ring in the spotlight, if not all of it. */
  readonly highlight?: readonly string[];
  /** Selectors of the elements that the card should not cover. */
  readonly keepClear?: readonly string[];
  /** Placements of the card to try, in order of preference. */
  readonly placements: readonly TourPlacement[];
  /** Space between the spotlight and its elements, in pixels. */
  readonly padding?: number;
  readonly task?: TourTask;
}

/** A kernel of the profile, as the tour illustrates it. */
export interface TourKernel {
  readonly name: string;
  readonly module: string;
  /** Two-letter label of the kernel, e.g. `Fa`. */
  readonly monogram: string;
  /** Color of the kernel's module group. */
  readonly color: string;
  /** Length of the kernel's schedule. */
  readonly bundles: number;
  /** Compact schedule length, e.g. `38.4K`. */
  readonly bundlesLabel: string;
  /** Busiest functional unit and its utilization, e.g. `MXU 82%`, or ''. */
  readonly busiestUnit: string;
  /** Color of the busiest functional unit, or ''. */
  readonly busiestUnitColor: string;
}

/** What the tour knows about the profile, to personalize its steps. */
export interface KernelViewerTourSummary {
  readonly kernelCount: number;
  readonly moduleCount: number;
  /** The largest kernels with schedule statistics, largest first. */
  readonly topKernels: readonly TourKernel[];
  /** Label of the modifier key that opens kernels in background tabs. */
  readonly backgroundClickKey: string;
  /** Whether the viewer can copy links to kernels. */
  readonly canCopyLink: boolean;
}

/** A kernel row of the illustrations, in ranked order. */
export interface RailRow extends TourKernel {
  /** Length of the row's size meter, between 0 and 1. */
  readonly fraction: number;
  /** How many rows away from its ranked position the row starts sorting. */
  readonly from: number;
}

/** Where the card goes, and on which side of its spotlight. */
export interface CardPlacement {
  readonly x: number;
  readonly y: number;
  readonly side: TourPlacement | 'center';
}

/** Local storage key that records that the user has seen the tour. */
export const KERNEL_VIEWER_TOUR_STORAGE_KEY = 'xprof_kernel_viewer_tour';

/** Version of the tour. Bump it to show a reworked tour to everyone again. */
const TOUR_VERSION = '1';

/** The canvas of the timeline, below the minimap. */
const CANVAS_SELECTOR = 'trace-viewer-container .canvas-container canvas';

/** The details panel of the selected block. */
const PANEL_SELECTOR = 'kernel-selection-panel .selection-panel';

/** The track of the minimap, which spans the bundles like the timeline does. */
const MINIMAP_TRACK_SELECTOR = 'kernel-minimap .minimap-track';

/** The timeline and its details drawer. */
const CONTENT_SELECTOR = 'trace-viewer-container .main-content-area';

/** Height of the bundle ruler at the top of the canvas, in pixels. */
const RULER_HEIGHT = 28;

/** Width of the labels of the lanes of the timeline, if the minimap is off. */
const DEFAULT_LABEL_WIDTH = 250;

/** Space between the spotlight and the elements it shows, in pixels. */
const DEFAULT_PADDING = 8;

/** Space between the ring and the element it highlights, in pixels. */
const RING_PADDING = 4;

/** Corner radii, clockwise from the top left corner, in pixels. */
export type Corners = readonly [number, number, number, number];

const SQUARE: Corners = [0, 0, 0, 0];

/** Distance between the card and its spotlight, in pixels. */
const CARD_GAP = 16;

/** Minimum distance between the card and the edges of the viewport. */
const VIEWPORT_MARGIN = 16;

/** Distance between a card placed inside its spotlight and its edges. */
const INSIDE_INSET = 20;

/** Minimum distance between the arrow of the card and its corners. */
const ARROW_INSET = 24;

/** How long the tour takes to close, see the stylesheet. */
const CLOSE_DURATION_MS = 320;

/**
 * How long the spotlight keeps tracking its elements after a layout change,
 * which covers transitions such as the kernel rail expanding.
 */
const KEEP_ALIVE_MS = 260;

/** Delay before the demo of a step plays, so that the user can read first. */
const DEMO_START_DELAY_MS = 1100;

/**
 * How long the user's pointer has to rest before the demo plays again, so that
 * it does not get in the way of the user trying things out.
 */
const DEMO_IDLE_MS = 2600;

/** How long a gesture on the minimap may take to move its lens. */
const MINIMAP_GESTURE_MS = 2500;

/** Distance between the demo pointer and its caption, in pixels. */
const CAPTION_OFFSET: Point = {x: 16, y: 20};

/** Longest frame that the springs integrate, in seconds. */
const MAX_FRAME_SECONDS = 1 / 30;

/** Longest step of the spring integration, in seconds. */
const MAX_SPRING_STEP_SECONDS = 1 / 120;

/** Stiffness and damping of the springs that move the tour around. */
const SPOTLIGHT_SPRING = {stiffness: 300, damping: 30};
const RING_SPRING = {stiffness: 340, damping: 32};
const CARD_SPRING = {stiffness: 260, damping: 28};

const NUMBER_FORMAT = new Intl.NumberFormat();

const EMPTY_SUMMARY: KernelViewerTourSummary = {
  kernelCount: 0,
  moduleCount: 0,
  topKernels: [],
  backgroundClickKey: 'Ctrl',
  canCopyLink: false,
};

/** Kernels that the illustrations show if the profile has no statistics. */
const EXAMPLE_KERNELS: readonly TourKernel[] = [
  {
    name: 'flash_attention_fwd',
    module: 'jit_train_step',
    monogram: 'Fa',
    color: '#1a73e8',
    bundles: 38412,
    bundlesLabel: '38.4K',
    busiestUnit: 'MXU 82%',
    busiestUnitColor: '#7c4dff',
  },
  {
    name: 'gmm_prefill',
    module: 'jit_train_step',
    monogram: 'Gp',
    color: '#1a73e8',
    bundles: 21730,
    bundlesLabel: '21.7K',
    busiestUnit: 'MXU 74%',
    busiestUnitColor: '#7c4dff',
  },
  {
    name: 'ragged_dot',
    module: 'jit_eval_step',
    monogram: 'Rd',
    color: '#d93025',
    bundles: 9140,
    bundlesLabel: '9.1K',
    busiestUnit: 'VPU 61%',
    busiestUnitColor: '#1a73e8',
  },
  {
    name: 'add_add_fusion.114',
    module: 'jit_eval_step',
    monogram: 'Aa',
    color: '#d93025',
    bundles: 1203,
    bundlesLabel: '1.2K',
    busiestUnit: 'VPU 48%',
    busiestUnitColor: '#1a73e8',
  },
];

/**
 * Positions that the rows of the rail illustration start from before they
 * sort into ranked order, by number of rows.
 */
const SHUFFLES: ReadonlyArray<readonly number[]> = [
  [],
  [0],
  [1, 0],
  [2, 0, 1],
  [2, 0, 3, 1],
];

/** What each functional unit does, as the field sheet explains it. */
const UNIT_DESCRIPTIONS: ReadonlyMap<string, string> = new Map([
  ['MXU', 'Matrix multiply unit'],
  ['XLU', 'Cross-lane unit: transpose, permute, reduce'],
  ['SALU', 'Scalar ALU: control flow, addressing, DMA issue'],
  ['VPU', 'Vector ALU'],
  ['EUP', 'Extended unary pipeline: exp, rcp, tanh'],
  ['VLD', 'Vector load: VMEM → vregs'],
  ['VST', 'Vector store: vregs → VMEM'],
  ['DMA', 'DMA engine: HBM ↔ VMEM'],
  ['Other', 'No dedicated functional unit'],
]);

function unitColor(unit: string): string {
  return FUNCTIONAL_UNITS_BY_LANE.get(unit)?.color ?? '#5f6368';
}

/** A block of a lane in the timeline illustrations, in percent of the lane. */
interface LaneBlock {
  readonly start: number;
  readonly width: number;
}

/** A lane of a functional unit in the timeline illustrations. */
interface Lane {
  readonly unit: string;
  readonly color: string;
  readonly blocks: readonly LaneBlock[];
}

function lane(unit: string, blocks: readonly LaneBlock[]): Lane {
  return {unit, color: unitColor(unit), blocks};
}

/** A software-pipelined schedule, as the welcome illustration draws it. */
const WELCOME_LANES: readonly Lane[] = [
  lane('MXU', [
    {start: 13, width: 15},
    {start: 38, width: 16},
    {start: 63, width: 16},
  ]),
  lane('VPU', [
    {start: 5, width: 7},
    {start: 29, width: 8},
    {start: 55, width: 7},
    {start: 81, width: 13},
  ]),
  lane('VLD', [
    {start: 1, width: 10},
    {start: 27, width: 9},
    {start: 53, width: 8},
  ]),
  lane('VST', [
    {start: 30, width: 5},
    {start: 56, width: 5},
    {start: 83, width: 11},
  ]),
  lane('SALU', [
    {start: 0, width: 3},
    {start: 25, width: 2},
    {start: 51, width: 2},
    {start: 77, width: 2},
  ]),
];

/** Lanes of the navigation and minimap illustrations. */
const NAVIGATION_LANES: readonly Lane[] = WELCOME_LANES.slice(0, 4);

/** A region of the minimap illustration, in percent of its width. */
interface MinimapRegion {
  readonly name: string;
  readonly start: number;
  readonly width: number;
}

/** Regions of the minimap illustration: a pipelined loop between two ends. */
const MINIMAP_REGIONS: readonly MinimapRegion[] = [
  {name: 'prologue', start: 0, width: 12},
  {name: 'loop.body', start: 12.5, width: 74},
  {name: 'epilogue', start: 87, width: 13},
];

/** Utilization of the bins of the minimap illustration, in percent. */
const MINIMAP_UTILIZATION: readonly number[] = [
  30, 42, 58, 74, 90, 86, 70, 92, 96, 88, 72, 90, 94, 84, 68, 88, 92, 80, 62,
  46, 34, 22,
];

/** Sections of the timeline of a kernel, from top to bottom. */
export type TimelineSection =
  | 'utilization'
  | 'memory'
  | 'jax'
  | 'regions'
  | 'units';

/** A row of the timeline legend. */
export interface TimelineRow {
  readonly section: TimelineSection;
  readonly label: string;
  /** Material icon of the row. */
  readonly icon: string;
  readonly color: string;
  /** What the row shows, in a few words. */
  readonly meaning: string;
}

/** The rows of the timeline of a kernel, in the order that it shows them. */
export const TIMELINE_ROWS: readonly TimelineRow[] = [
  {
    section: 'jax',
    label: 'Scopes',
    icon: 'code',
    color: '#d01884',
    meaning: 'Named scopes, Pallas primitives',
  },
  {
    section: 'regions',
    label: 'Regions',
    icon: 'account_tree',
    color: DEFAULT_ACCENT_COLOR,
    meaning: 'Loops and branches',
  },
  {
    section: 'units',
    label: 'Unit lanes',
    icon: 'view_week',
    color: '#7c4dff',
    meaning: 'One block per instruction',
  },
  {
    section: 'memory',
    label: 'Memory',
    icon: 'memory',
    color: '#e8710a',
    meaning: 'VMEM and live registers',
  },
  {
    section: 'utilization',
    label: 'Utilization',
    icon: 'show_chart',
    color: '#12a4af',
    meaning: 'How busy each unit is',
  },
];

/**
 * A metric of the details panel, which the details steps number and explain:
 * its value for the selection, where the panel shows it, and what it means.
 */
export interface PanelMetric {
  readonly label: string;
  /** Its value for the selection, e.g. `VPU 75%`, or '' if unknown. */
  readonly value: string;
  /** Color of the value, e.g. of its functional unit. */
  readonly color: string;
  /** What it means, in a few words. */
  readonly meaning: string;
  /** Selector of the element of the details panel that shows it. */
  readonly selector: string;
  /** What the demo pointer says as it points at the metric. */
  readonly caption: string;
}

/** A button of the header of the details panel, which the panel step lists. */
export interface PanelAction {
  readonly icons: readonly string[];
  readonly label: string;
}

/** What the details steps make the details panel show. */
export type DetailsKind = 'instruction' | 'region';

/** The most metrics that a details step explains, one per numbered marker. */
const MAX_PANEL_METRICS = 6;

/** Returns the selector of card `n` of the details panel, from 1. */
function panelCard(mode: 'single' | 'range', n: number): string {
  return `${PANEL_SELECTOR} .${mode}-mode-grid > .panel-card:nth-child(${n})`;
}

const LANE_ACTION: PanelAction = {
  icons: ['chevron_left', 'chevron_right'],
  label: 'Along the lane',
};
const BACK_ACTION: PanelAction = {icons: ['undo'], label: 'Back'};

/** Explains the metrics that the details panel shows for an instruction. */
function instructionMetrics(
  content: KernelSelectionContent | undefined,
): PanelMetric[] {
  const color = content?.unit?.color ?? DEFAULT_ACCENT_COLOR;
  const metrics: PanelMetric[] = [
    {
      label: 'Bundle',
      value: content
        ? [content.bundles, content.hexBundle].filter(Boolean).join(' ')
        : '',
      color,
      meaning: 'The VLIW bundle it issues in, also in hex',
      selector: `${PANEL_SELECTOR} .panel-header .bundle-pill`,
      caption: 'The bundle it issues in',
    },
  ];
  const crumbs = content?.regionBreadcrumbs ?? [];
  if (!content || crumbs.length > 0) {
    metrics.push({
      label: 'Region',
      value: crumbs[crumbs.length - 1]?.name ?? '',
      color: DEFAULT_ACCENT_COLOR,
      meaning: 'The loops or other regions that it runs in',
      selector: `${PANEL_SELECTOR} .header-breadcrumb-row`,
      caption: 'The regions around it',
    });
  }
  // Only some compilers record the dataflow of instructions.
  const register = content?.output?.register;
  if (register) {
    metrics.push({
      label: 'Produces',
      value: register,
      color,
      meaning: 'The register that it writes',
      selector: `${panelCard('single', 1)} .register-pill.interactive-pill`,
      caption: 'The register it writes',
    });
  }
  const consumers = content?.consumers?.length ?? 0;
  if (consumers > 0) {
    metrics.push({
      label: 'Consumers',
      value: `→ ${NUMBER_FORMAT.format(consumers)}`,
      color,
      meaning: 'Later ops that read its result',
      selector: `${panelCard('single', 1)} .consumer-list`,
      caption: 'Ops that read its result',
    });
  }
  if (!content || content.bundleStrip) {
    metrics.push({
      label: 'In this bundle',
      value: content?.activeLanesSummary ?? '',
      color: '#7c4dff',
      meaning: 'Ops issued with it, and the units they use',
      selector: `${panelCard('single', 2)} .bundle-strip`,
      caption: 'Everything issued with it',
    });
  }
  const busiest = content?.utilizationBars?.[0];
  if (!content || busiest) {
    metrics.push({
      label: 'Static utilization',
      value:
        busiest && busiest.percent > 0
          ? `${busiest.unit.label} ${busiest.formattedPercent}`
          : '',
      color: busiest?.unit.color ?? '#12a4af',
      meaning: 'How much of each unit is in use, as scheduled',
      selector: `${panelCard('single', 3)} .util-bars-list`,
      caption: 'How busy each unit is here',
    });
  }
  if (!content || content.schedulePosition) {
    metrics.push({
      label: 'Schedule position',
      value: content?.schedulePosition?.formattedPercent ?? '',
      color: DEFAULT_ACCENT_COLOR,
      meaning: 'How far into the HLO op this bundle is',
      selector: `${panelCard('single', 3)} .card-section-footer`,
      caption: 'Where it sits in the HLO op',
    });
  }
  return metrics.slice(0, MAX_PANEL_METRICS);
}

/**
 * Explains the metrics that the details panel shows for a region, or for a
 * measured range, which it sums up the same way.
 */
function regionMetrics(
  content: KernelSelectionContent | undefined,
): PanelMetric[] {
  const metrics: PanelMetric[] = [];
  const mix = content?.instructionMix?.[0];
  if (!content || mix) {
    metrics.push({
      label: 'Instruction mix',
      value: mix ? `${mix.unit.label} ${mix.formattedPercent}` : '',
      color: mix?.unit.color ?? DEFAULT_ACCENT_COLOR,
      meaning: 'Share of its instructions on each unit',
      selector: `${panelCard('range', 1)} .stacked-mix-bar`,
      caption: 'Which units its instructions use',
    });
  }
  if (!content || content.density) {
    metrics.push({
      label: 'Density',
      value: content?.density ?? '',
      color: DEFAULT_ACCENT_COLOR,
      meaning: 'Instructions per bundle, on average',
      selector: `${panelCard('range', 1)} .section-badge`,
      caption: 'How full its bundles are',
    });
  }
  // Measured ranges show the mean and the peak of each unit.
  const range = content?.utilizationMetrics?.[0];
  const mean = content?.utilizationBars?.[0];
  if (range) {
    metrics.push({
      label: 'Utilization',
      value: `${range.unit.label} ${range.formattedMean}`,
      color: range.unit.color,
      meaning: 'Mean and peak of each unit across the span',
      selector: `${panelCard('range', 2)} .range-util-metrics-list`,
      caption: 'Each unit’s mean and peak',
    });
  } else if (!content || mean) {
    metrics.push({
      label: 'Mean utilization',
      value: mean ? `${mean.unit.label} ${mean.formattedPercent}` : '',
      color: mean?.unit.color ?? '#12a4af',
      meaning: 'Static utilization, averaged over its bundles',
      selector: `${panelCard('range', 2)} .util-bars-list`,
      caption: 'How busy each unit is, on average',
    });
  }
  // Measured ranges sum up the memory and register counters too.
  const resource = content?.resourceCounterMetrics?.[0];
  if (resource) {
    metrics.push({
      label: 'Memory & registers',
      value: `${resource.shortLabel} ${resource.formattedMean}`,
      color: resource.accentColor,
      meaning: 'Mean and peak of each memory or register counter',
      selector: `${panelCard('range', 2)} .resource-metrics-grid`,
      caption: 'Memory and register pressure',
    });
  }
  const children = content?.childRegions?.length ?? 0;
  if (!content || children > 0 || content.parentRegion) {
    metrics.push({
      label: 'Hierarchy',
      value:
        children > 0
          ? `${NUMBER_FORMAT.format(children)} ` +
            (children === 1 ? 'child' : 'children')
          : (content?.parentRegion?.name ?? ''),
      color: DEFAULT_ACCENT_COLOR,
      meaning: 'Its parent and child regions, a click away',
      // The region hierarchy comes after the memory counters, if any.
      selector: `${panelCard('range', 2)} .hierarchy-block:last-child`,
      caption: 'Move up, or into a child region',
    });
  }
  const top = content?.topInstructions?.[0];
  if (!content || top) {
    metrics.push({
      label: 'Top instructions',
      value: top ? `${top.name} ${top.formattedShare}` : '',
      color: top?.unit?.color ?? DEFAULT_ACCENT_COLOR,
      meaning: 'Its most frequent ops; click one to jump to it',
      selector: `${panelCard('range', 3)} .top-instruction-row`,
      caption: 'Its most frequent ops',
    });
  }
  return metrics.slice(0, MAX_PANEL_METRICS);
}

/**
 * Explains the metrics that the details panel shows for a point of a counter
 * track, e.g. of the utilization of a unit, followed by its bundle.
 */
function counterMetrics(content: KernelSelectionContent): PanelMetric[] {
  const details = content.counterDetails;
  const color = content.unit?.color ?? content.accentColor;
  const kpi = `${panelCard('single', 1)} .counter-kpi-box`;
  const metrics: PanelMetric[] = [];
  if (details) {
    metrics.push({
      label: 'At bundle',
      value: details.formattedCurrent,
      color,
      meaning: 'Its value in this bundle',
      selector: `${kpi}.primary-kpi`,
      caption: 'Its value in this bundle',
    });
    let next = `${kpi}.primary-kpi + .counter-kpi-box`;
    if (details.formattedMean) {
      metrics.push({
        label: 'Op mean',
        value: details.formattedMean,
        color,
        meaning: 'Its average over the whole HLO op',
        selector: next,
        caption: 'Its op-wide average',
      });
      next = `${next} + .counter-kpi-box`;
    }
    if (details.formattedPeak) {
      metrics.push({
        label: 'Op peak',
        value: details.formattedPeak,
        color,
        meaning: 'Its highest value in the HLO op, and where',
        selector: next,
        caption: 'Its op-wide peak',
      });
    }
    metrics.push({
      label: 'Holds for',
      value: details.stepDurationLabel,
      color,
      meaning: 'The bundles until its value changes',
      selector: `${panelCard('single', 1)} .counter-meta-grid`,
      caption: 'How long this value holds',
    });
  }
  // The other cards show the bundle, like for an instruction.
  const bundleMetrics = new Set(['In this bundle', 'Static utilization']);
  for (const metric of instructionMetrics(content)) {
    if (bundleMetrics.has(metric.label)) {
      metrics.push(metric);
    }
  }
  return metrics.slice(0, MAX_PANEL_METRICS);
}

/**
 * Returns the metrics that the details panel shows for `content`, or that it
 * shows for `kind` of selection if its content is unknown.
 */
export function buildPanelMetrics(
  content: KernelSelectionContent | undefined,
  kind: DetailsKind,
): PanelMetric[] {
  if (!content) {
    return kind === 'region'
      ? regionMetrics(undefined)
      : instructionMetrics(undefined);
  }
  if (content.kind === 'counter') {
    return counterMetrics(content);
  }
  return content.mode === 'range'
    ? regionMetrics(content)
    : instructionMetrics(content);
}

const FILTER_ACTION: PanelAction = {
  icons: ['filter_alt'],
  label: 'Filter by unit',
};
const SEARCH_ACTION: PanelAction = {icons: ['search'], label: 'Highlight all'};

/** Returns the buttons of the header of the details panel for `content`. */
export function buildPanelActions(
  content: KernelSelectionContent | undefined,
  kind: DetailsKind,
): PanelAction[] {
  if (content?.kind === 'counter') {
    return [BACK_ACTION];
  }
  // Measured ranges have no neighbors along a lane.
  if (content?.kind === 'range') {
    return [FILTER_ACTION, BACK_ACTION];
  }
  const isRange = content ? content.mode === 'range' : kind === 'region';
  return isRange
    ? [LANE_ACTION, FILTER_ACTION, BACK_ACTION]
    : [LANE_ACTION, BACK_ACTION, SEARCH_ACTION];
}

/** Whether `content` is the details of an instruction of a unit lane. */
function isInstructionDetails(
  content: KernelSelectionContent | undefined,
): boolean {
  return (
    content?.mode === 'single' &&
    content.kind === 'instruction' &&
    content.unit !== undefined
  );
}

/** Whether `content` is the details of a region. */
function isRegionDetails(content: KernelSelectionContent | undefined) {
  return content?.kind === 'region';
}

/** A region around a selected block, which the region step can select. */
export interface RegionRef {
  readonly index: number;
  readonly name: string;
  /** Length of the region, in bundles. */
  readonly length: number;
}

/**
 * Reads the regions around a selected block, from the outermost to the
 * innermost, from the detail of an `eventselected` event of Trace Viewer v2.
 */
export function readRegionAncestors(detail: unknown): RegionRef[] {
  if (typeof detail !== 'object' || detail === null) {
    return [];
  }
  const ancestors = (detail as Record<string, unknown>)['regionAncestors'];
  if (!Array.isArray(ancestors)) {
    return [];
  }
  const regions: RegionRef[] = [];
  for (const ancestor of ancestors) {
    if (typeof ancestor !== 'object' || ancestor === null) {
      continue;
    }
    const fields = ancestor as Record<string, unknown>;
    const index = fields['eventIndex'];
    const name = fields['name'];
    const length = fields['durationUs'];
    if (
      typeof index === 'number' &&
      index >= 0 &&
      typeof name === 'string' &&
      typeof length === 'number'
    ) {
      regions.push({index, name, length});
    }
  }
  return regions;
}

/** Regions shorter than this sum up too few instructions to explain well. */
const MIN_REGION_BUNDLES = 16;

/**
 * Picks the region to explain around a block: the innermost one that spans a
 * few bundles at least, e.g. the loop that the block runs in.
 */
export function pickRegionAround(
  ancestors: readonly RegionRef[],
): RegionRef | undefined {
  for (let i = ancestors.length - 1; i >= 0; i--) {
    if (ancestors[i].length >= MIN_REGION_BUNDLES) {
      return ancestors[i];
    }
  }
  return ancestors[ancestors.length - 1];
}

const REGIONS_LANE = 'Regions';
const JAX_LANES = new Set(['Named Scope', 'Pallas Primitives']);
const MEMORY_LANES = new Set([
  'VMEM Scoped',
  'VMEM Spill',
  'Scalar Reg',
  'Vector Reg',
  'Predicate Reg',
  'Vector Mask Reg',
]);

/**
 * A block of the timeline, as Trace Viewer v2 reports it when the user points
 * at or clicks it.
 */
export interface TimelineBlock {
  /** Index of the event in Trace Viewer v2, which can select it. */
  readonly index: number;
  readonly name: string;
  /** First bundle of the block. */
  readonly start: number;
  /** Length of the block, in bundles. */
  readonly length: number;
  /** Lane of the block, e.g. `MXU`, `Regions` or `Named Scope`, or ''. */
  readonly lane: string;
  /** Fields that the compiler recorded for the block, on selection. */
  readonly args: Readonly<Record<string, string>>;
}

/**
 * Reads the detail of an `eventhovered` or `eventselected` event of Trace
 * Viewer v2. Returns undefined if no block is hovered or selected.
 */
export function readTimelineBlock(detail: unknown): TimelineBlock | undefined {
  if (typeof detail !== 'object' || detail === null) {
    return undefined;
  }
  // The detail comes from WASM, so its fields are read by their exact names.
  const fields = detail as Record<string, unknown>;
  const index = fields['eventIndex'];
  const trackName = fields['trackName'];
  const trackType = fields['trackType'];
  const isCounterOrProcess =
    trackType === 'counter' || trackType === 'process';
  const name =
    typeof fields['name'] === 'string' && fields['name']
      ? fields['name']
      : isCounterOrProcess && typeof trackName === 'string'
        ? trackName
        : undefined;
  const start = fields['startUs'];
  const length = fields['durationUs'];
  if (
    typeof index !== 'number' ||
    (index < 0 && !(index === -1 && isCounterOrProcess)) ||
    !name ||
    typeof start !== 'number' ||
    typeof length !== 'number'
  ) {
    return undefined;
  }
  const args: Record<string, string> = {};
  const rawArgs = fields['args'];
  if (typeof rawArgs === 'object' && rawArgs !== null) {
    for (const [key, value] of Object.entries(rawArgs)) {
      if (typeof value === 'string' || typeof value === 'number') {
        args[key] = String(value);
      }
    }
  }
  return {
    index,
    name,
    start,
    length,
    lane:
      typeof trackName === 'string' && trackName
        ? trackName
        : laneFromArgs(args),
    args,
  };
}

/**
 * Reads where the pointer was on the canvas, relative to its top left corner,
 * from the detail of an `eventhovered` event of Trace Viewer v2.
 */
export function readPointerOffset(detail: unknown): Point | undefined {
  if (typeof detail !== 'object' || detail === null) {
    return undefined;
  }
  const fields = detail as Record<string, unknown>;
  const x = fields['mouse_x'];
  const y = fields['mouse_y'];
  return typeof x === 'number' &&
    typeof y === 'number' &&
    Number.isFinite(x) &&
    Number.isFinite(y)
    ? {x, y}
    : undefined;
}

/** The bundles that the timeline shows, as the minimap reports them. */
export interface VisibleRange {
  readonly start: number;
  readonly end: number;
}

/** Reads the visible bundles from the detail of a `minimap_updated` event. */
export function readVisibleRange(detail: unknown): VisibleRange | undefined {
  if (typeof detail !== 'object' || detail === null) {
    return undefined;
  }
  const fields = detail as Record<string, unknown>;
  const start = fields['visibleStartUs'];
  const end = fields['visibleEndUs'];
  return typeof start === 'number' &&
    typeof end === 'number' &&
    Number.isFinite(start) &&
    Number.isFinite(end) &&
    end > start
    ? {start, end}
    : undefined;
}

/**
 * Reads the visible bundles from the `lens` of the minimap, which spans them
 * on its `track`: the minimap only reports them as they change.
 */
export function readLensRange(
  lens: Element,
  lensBox: Box,
  track: Box,
): VisibleRange | undefined {
  const first = Number(lens.getAttribute('aria-valuemin'));
  const last = Number(lens.getAttribute('aria-valuemax'));
  const start = Number(lens.getAttribute('aria-valuenow'));
  if (
    !Number.isFinite(first) ||
    !Number.isFinite(last) ||
    !Number.isFinite(start) ||
    last <= first ||
    track.w <= 0
  ) {
    return undefined;
  }
  const end = Math.min(last, start + (lensBox.w / track.w) * (last - first));
  return end > start ? {start, end} : undefined;
}

/**
 * Infers the lane of a selected block from its fields, since Trace Viewer v2
 * only names the lane of hovered blocks.
 */
function laneFromArgs(args: Readonly<Record<string, string>>): string {
  if (args['unit']) {
    return args['unit'];
  }
  if (args['depth'] !== undefined) {
    return args['source'] !== undefined ? 'Pallas Primitives' : 'Named Scope';
  }
  return args['limit_bundle'] !== undefined ? REGIONS_LANE : '';
}

/** Returns the section of the timeline that `lane` belongs to, if known. */
export function sectionOf(lane: string): TimelineSection | undefined {
  if (FUNCTIONAL_UNITS_BY_LANE.has(lane)) {
    return 'units';
  }
  if (lane === REGIONS_LANE) {
    return 'regions';
  }
  if (lane === 'JAX Source Code' || JAX_LANES.has(lane)) {
    return 'jax';
  }
  if (lane === 'Utilization' || lane.endsWith(' utilization')) {
    return 'utilization';
  }
  return lane === 'Memory' || MEMORY_LANES.has(lane) ? 'memory' : undefined;
}

/** Kinds of blocks, which have different fields. */
export type BlockKind = 'instruction' | 'region' | 'scope';

/**
 * Who selected a block: the user, or the tour, which selects the block that
 * the user pointed at, or else one that the viewer picks.
 */
export type SelectionOrigin = 'user' | 'pointed' | 'picked';

/** Says who selected a block, e.g. `Picked by the tour`. */
function captionFor(origin: SelectionOrigin): string {
  switch (origin) {
    case 'pointed':
      return 'The block you pointed at';
    case 'picked':
      return 'Picked by the tour';
    default:
      return 'Your selection';
  }
}

/** A field of a block: its name in the trace, its value, and its meaning. */
export interface BlockField {
  readonly key: string;
  readonly value: string;
  readonly meaning: string;
}

/** The fields of a block, as the tour shows them. */
export interface FieldSheet {
  readonly kind: BlockKind;
  /** Name of the block, or '' until the user selects one. */
  readonly name: string;
  /** Lane of the block, e.g. `MXU`, or '' until the user selects one. */
  readonly lane: string;
  /** Color of the lane. */
  readonly color: string;
  readonly fields: readonly BlockField[];
}

/** Placeholder of a field that has no value yet. */
const NO_VALUE = '—';

const SCOPE_COLOR = '#d01884';

const FIELD_MEANINGS = {
  opcode: 'The operation the compiler emitted',
  unit: 'The functional unit that issues it',
  bundle: 'The VLIW bundle that it issues in',
  ordinal: 'Program order, before scheduling',
  instructionDur: 'Always 1 bundle: not its latency',
  regionName: 'A loop, branch, or other region',
  startBundle: 'Its first bundle',
  limitBundle: 'The bundle after its last one',
  regionDur: 'Its length, in bundles',
  regionOrdinal: 'Its place in program order',
  scopeName: 'A named scope or Pallas primitive',
  depth: 'How deeply it is nested',
  source: 'Where it is in your Python code',
};

/** Formats a number, or a string that holds one, e.g. `12,345`. */
function formatCount(value: string | number | undefined): string {
  if (value === undefined || value === '') {
    return NO_VALUE;
  }
  const count = Number(value);
  return Number.isFinite(count) ? NUMBER_FORMAT.format(count) : String(value);
}

/** Returns the bundles that a block covers, e.g. `bundles 8–23`. */
function bundlesOf(start: number, length: number): string {
  if (length <= 1) {
    return `bundle ${NUMBER_FORMAT.format(start)}`;
  }
  const last = NUMBER_FORMAT.format(start + length - 1);
  return `bundles ${NUMBER_FORMAT.format(start)}–${last}`;
}

/** Shortens a source location to its file name, e.g. `kernel.py:42`. */
function shortSource(source: string): string {
  return source.split('/').pop() || source;
}

/**
 * Returns the fields of `block` with their meanings, or the fields of an
 * instruction without values until the user selects a block.
 */
export function buildFieldSheet(block: TimelineBlock | undefined): FieldSheet {
  if (!block) {
    return {
      kind: 'instruction',
      name: '',
      lane: '',
      color: DEFAULT_ACCENT_COLOR,
      fields: [
        {key: 'opcode', value: NO_VALUE, meaning: FIELD_MEANINGS.opcode},
        {key: 'unit', value: NO_VALUE, meaning: FIELD_MEANINGS.unit},
        {key: 'bundle', value: NO_VALUE, meaning: FIELD_MEANINGS.bundle},
        {key: 'ordinal', value: NO_VALUE, meaning: FIELD_MEANINGS.ordinal},
        {key: 'dur', value: NO_VALUE, meaning: FIELD_MEANINGS.instructionDur},
      ],
    };
  }
  const {args, lane, name, start, length} = block;
  const section = sectionOf(lane);
  const bounds: BlockField[] = [
    {
      key: 'start_bundle',
      value: formatCount(args['start_bundle'] ?? start),
      meaning: FIELD_MEANINGS.startBundle,
    },
    {
      key: 'limit_bundle',
      value: formatCount(args['limit_bundle'] ?? start + length),
      meaning: FIELD_MEANINGS.limitBundle,
    },
  ];
  if (section === 'regions') {
    return {
      kind: 'region',
      name,
      lane,
      color: DEFAULT_ACCENT_COLOR,
      fields: [
        {key: 'name', value: name, meaning: FIELD_MEANINGS.regionName},
        ...bounds,
        {
          key: 'dur',
          value: formatCount(length),
          meaning: FIELD_MEANINGS.regionDur,
        },
        {
          key: 'ordinal',
          value: formatCount(args['ordinal']),
          meaning: FIELD_MEANINGS.regionOrdinal,
        },
      ],
    };
  }
  if (section === 'jax') {
    const fields: BlockField[] = [
      {key: 'name', value: name, meaning: FIELD_MEANINGS.scopeName},
      {
        key: 'depth',
        value: formatCount(args['depth']),
        meaning: FIELD_MEANINGS.depth,
      },
      ...bounds,
    ];
    const source = args['source'];
    if (source) {
      fields.push({
        key: 'source',
        value: shortSource(source),
        meaning: FIELD_MEANINGS.source,
      });
    }
    return {kind: 'scope', name, lane, color: SCOPE_COLOR, fields};
  }
  const unit = args['unit'] || lane;
  return {
    kind: 'instruction',
    name,
    lane: unit,
    color: unitColor(unit),
    fields: [
      {
        key: 'opcode',
        value: args['opcode'] || name,
        meaning: FIELD_MEANINGS.opcode,
      },
      {
        key: 'unit',
        value: unit || NO_VALUE,
        meaning: UNIT_DESCRIPTIONS.get(unit) ?? FIELD_MEANINGS.unit,
      },
      {
        key: 'bundle',
        value: formatCount(args['bundle'] ?? start),
        meaning: FIELD_MEANINGS.bundle,
      },
      {
        key: 'ordinal',
        value: formatCount(args['ordinal']),
        meaning: FIELD_MEANINGS.ordinal,
      },
      {
        key: 'dur',
        value: formatCount(length),
        meaning: FIELD_MEANINGS.instructionDur,
      },
    ],
  };
}

/** What the legend shows about the block under the pointer. */
export interface PointerReadout {
  readonly lane: string;
  readonly color: string;
  readonly name: string;
  /** Bundles that the block covers, e.g. `bundle 12,345`. */
  readonly bundles: string;
}

/** Describes the hovered `block` for the timeline legend. */
export function buildPointerReadout(block: TimelineBlock): PointerReadout {
  const section = sectionOf(block.lane);
  const row = TIMELINE_ROWS.find((candidate) => candidate.section === section);
  return {
    lane: block.lane || 'Timeline',
    color:
      section === 'units'
        ? unitColor(block.lane)
        : (row?.color ?? DEFAULT_ACCENT_COLOR),
    name: block.name,
    bundles: bundlesOf(block.start, block.length),
  };
}

/** A particle of the celebration on the last step. */
interface Particle {
  /** Direction the particle flies in, e.g. `30deg`. */
  readonly angle: string;
  /** How far the particle flies, e.g. `44px`. */
  readonly distance: string;
  readonly color: string;
}

const PARTICLE_COLORS = [
  '#7c4dff',
  '#1a73e8',
  '#1e8e3e',
  '#f29900',
  '#d01884',
  '#12a4af',
];

const PARTICLES: readonly Particle[] = Array.from({length: 12}, (_, i) => ({
  angle: `${i * 30 + 15}deg`,
  distance: `${i % 2 === 0 ? 46 : 36}px`,
  color: PARTICLE_COLORS[i % PARTICLE_COLORS.length],
}));

/** A row of the shortcut summary on the last step. */
interface CheatsheetItem {
  readonly keys: readonly string[];
  readonly label: string;
}

const CHEATSHEET: readonly CheatsheetItem[] = [
  {keys: ['K'], label: 'Find HLO ops'},
  {keys: ['[', ']'], label: 'Switch tabs'},
  {keys: ['W', 'A', 'S', 'D'], label: 'Zoom & pan'},
  {keys: ['←', '→'], label: 'Step along a lane'},
  {keys: ['/'], label: 'Search events'},
  {keys: ['?'], label: 'All shortcuts'},
];

let nextTourId = 0;

function plural(count: number, noun: string): string {
  return `${NUMBER_FORMAT.format(count)} ${noun}${count === 1 ? '' : 's'}`;
}

/** Builds the steps of the tour of a profile. */
export function buildKernelViewerTourSteps(
  summary: KernelViewerTourSummary,
): TourStep[] {
  let profile = '';
  if (summary.kernelCount > 0) {
    const kernels = plural(summary.kernelCount, 'HLO op');
    const modules = plural(summary.moduleCount, 'HLO module');
    profile =
      summary.moduleCount > 0
        ? `This profile has ${kernels} across ${modules}. `
        : `This profile has ${kernels}. `;
  }
  return [
    {
      id: 'welcome',
      eyebrow: 'Kernel Viewer',
      title: 'See how your HLO ops are scheduled',
      body:
        `${profile}Take a quick tour to find the HLO ops worth optimizing, ` +
        'read their schedules bundle by bundle, and share what you find.',
      visual: 'welcome',
      spotlight: [],
      placements: [],
    },
    {
      id: 'rail',
      eyebrow: 'HLO Ops',
      title: 'The biggest HLO ops come first',
      body:
        'HLO ops are grouped by HLO module and sorted by schedule length. ' +
        'Each row shows its busiest unit and its length in bundles. Click ' +
        'one to open it.',
      visual: 'rail',
      spotlight: ['.kernel-rail'],
      placements: ['right', 'bottom'],
      padding: 0,
    },
    {
      id: 'search',
      eyebrow: 'Search',
      title: 'Jump to any HLO op',
      body:
        'Press K anywhere to search HLO ops and modules. Enter opens the ' +
        'top match.',
      visual: 'search',
      spotlight: ['.kernel-rail'],
      highlight: ['.rail-search .search-field'],
      placements: ['right', 'bottom'],
      padding: 0,
      task: {
        prompt: 'Try it: press',
        keys: ['k'],
        keyLabels: ['K'],
        success: 'Search is open. Type to filter the list.',
      },
    },
    {
      id: 'tabs',
      eyebrow: 'Tabs',
      title: 'Keep HLO ops side by side',
      body:
        `${summary.backgroundClickKey}-click or middle-click an HLO op to ` +
        'open it in a background tab, then press [ and ] to flip between ' +
        'tabs.',
      visual: 'tabs',
      spotlight: ['.kernel-bar .tab-strip'],
      placements: ['bottom', 'top'],
      padding: 6,
    },
    summary.canCopyLink
      ? {
          id: 'share',
          eyebrow: 'Step & share',
          title: 'Walk the ranking, then share',
          body:
            'The arrows step through HLO ops in ranked order. The link ' +
            'button copies a URL that reopens this exact HLO op for a ' +
            'teammate.',
          visual: 'share',
          spotlight: [
            '.kernel-bar .kernel-stepper',
            '.kernel-bar .copy-link-button',
          ],
          placements: ['bottom', 'left'],
          padding: 6,
        }
      : {
          id: 'share',
          eyebrow: 'Step through',
          title: 'Walk the ranking',
          body:
            'The arrows step through HLO ops in ranked order, from the ' +
            'biggest schedule to the smallest.',
          visual: 'share',
          spotlight: ['.kernel-bar .kernel-stepper'],
          placements: ['bottom', 'left'],
          padding: 6,
        },
    {
      id: 'timeline',
      eyebrow: 'Timeline',
      title: 'Read the timeline, row by row',
      body:
        'The ruler counts bundles, not microseconds. Scopes start collapsed ' +
        'above regions and unit lanes, with Memory and Utilization expanded ' +
        'below. Click a point on those to compare it with the HLO op’s ' +
        'mean and peak.',
      visual: 'legend',
      spotlight: [CANVAS_SELECTOR],
      fallback: ['trace-viewer-container .canvas-container'],
      placements: ['inside', 'left'],
      padding: 0,
      task: {
        prompt: 'Try it: point at a block in the timeline',
        gesture: 'hover',
        success: 'The legend follows your pointer.',
      },
    },
    {
      id: 'navigate',
      eyebrow: 'Explore',
      title: 'Zoom, pan, and measure',
      body:
        'W and S zoom, A and D pan. Zoom in until instructions separate ' +
        // A non-breaking hyphen keeps the gesture on one line.
        'into blocks. Shift\u2011drag to measure a span of bundles: its ' +
        'metrics open below.',
      visual: 'keys',
      spotlight: ['trace-viewer-container .search-container', CANVAS_SELECTOR],
      highlight: [
        'trace-viewer-container .mouse-mode-container',
        'trace-viewer-container .navigation-actions-container',
      ],
      placements: ['inside', 'left'],
      padding: 0,
      task: {
        prompt: 'Try it: press',
        keys: ['w', 'a', 's', 'd'],
        keyLabels: ['W', 'A', 'S', 'D'],
        success: 'Nice. The timeline follows your keys.',
      },
    },
    {
      id: 'inspect',
      eyebrow: 'Inspect',
      title: 'Click a block to inspect it',
      body:
        'Every block records what the compiler did with it. Click one to ' +
        'open its details below the timeline, and see what each field ' +
        'means.',
      visual: 'fields',
      spotlight: [CANVAS_SELECTOR, PANEL_SELECTOR],
      fallback: ['trace-viewer-container .canvas-container'],
      keepClear: [PANEL_SELECTOR],
      placements: ['inside', 'left'],
      padding: 0,
      task: {
        prompt: 'Try it: click a block in a unit lane',
        gesture: 'select',
        success: 'Its details are open below. Click more blocks to compare.',
      },
    },
    {
      id: 'panel',
      eyebrow: 'Details',
      title: 'Read an instruction’s details',
      body:
        'Clicking a block opens its details in this panel. Here is how to ' +
        'read them for an instruction.',
      visual: 'panel',
      spotlight: [PANEL_SELECTOR],
      fallback: [CANVAS_SELECTOR],
      placements: ['top', 'left', 'right'],
      padding: 4,
      task: {
        prompt: 'Try it: follow a link in the panel',
        gesture: 'select',
        success: 'The panel follows along. Its back button returns.',
      },
    },
    {
      id: 'region',
      eyebrow: 'Details',
      title: 'Zoom out to its region',
      body:
        'Regions are loops, branches, and other parts of the schedule. Their ' +
        'details sum up the instructions that start inside them.',
      visual: 'region',
      spotlight: [PANEL_SELECTOR],
      fallback: [CANVAS_SELECTOR],
      placements: ['top', 'left', 'right'],
      padding: 4,
      task: {
        prompt: 'Try it: filter by a unit, or open a child region',
        gesture: 'select',
        success: 'The panel and the timeline follow along.',
      },
    },
    {
      id: 'minimap',
      eyebrow: 'Minimap',
      title: 'See the whole schedule at once',
      body:
        'The minimap shows the whole HLO op: how busy each stretch is, its ' +
        'regions, and a lane per unit. The lens frames what the timeline ' +
        'shows. Drag it to pan, drag its edges to zoom, and double-click to ' +
        'see it all.',
      visual: 'minimap',
      spotlight: ['kernel-minimap .minimap-bar'],
      // The minimap renders once the trace has loaded.
      fallback: ['trace-viewer-container .canvas-container'],
      highlight: ['kernel-minimap .minimap-lens'],
      placements: ['bottom', 'top'],
      padding: 4,
      task: {
        prompt: 'Try it: drag the lens',
        gesture: 'minimap',
        success: 'The timeline follows the lens.',
      },
    },
    {
      id: 'done',
      eyebrow: 'You’re all set',
      title: 'Go find your next speedup',
      body:
        'Press / to find instructions and ? for every shortcut. Replay the ' +
        'tour from the tour button.',
      visual: 'done',
      spotlight: ['.kernel-bar .tour-button'],
      placements: ['bottom', 'left'],
      padding: 4,
    },
  ];
}

/**
 * Returns the rows of the rail illustration: the largest kernels of the
 * profile, or examples if fewer than two kernels have statistics.
 */
export function buildRailRows(kernels: readonly TourKernel[]): RailRow[] {
  const ranked = kernels
    .filter((kernel) => kernel.bundles > 0)
    .sort((a, b) => b.bundles - a.bundles);
  const rows = (ranked.length >= 2 ? ranked : EXAMPLE_KERNELS).slice(0, 4);
  const maxBundles = Math.max(1, ...rows.map((kernel) => kernel.bundles));
  const shuffle = SHUFFLES[rows.length] ?? [];
  return rows.map((kernel, i) => ({
    ...kernel,
    fraction: kernel.bundles / maxBundles,
    from: (shuffle[i] ?? i) - i,
  }));
}

/** Returns what the search illustration types to find `kernel`, e.g. `gmm`. */
export function searchQueryFor(kernel: string): string {
  const word = /[A-Za-z]+/.exec(kernel)?.[0] ?? '';
  return word.slice(0, 8) || kernel.slice(0, 4);
}

/** Splits `text` around the first match of `query`, ignoring case. */
export function splitAtQuery(
  text: string,
  query: string,
): readonly [string, string, string] {
  const start = query ? text.toLowerCase().indexOf(query.toLowerCase()) : -1;
  if (start < 0) {
    return [text, '', ''];
  }
  const end = start + query.length;
  return [text.slice(0, start), text.slice(start, end), text.slice(end)];
}

/** Returns the smallest box that contains `boxes`, if there are any. */
export function unionBox(boxes: readonly Box[]): Box | undefined {
  if (boxes.length === 0) {
    return undefined;
  }
  const left = Math.min(...boxes.map((box) => box.x));
  const top = Math.min(...boxes.map((box) => box.y));
  const right = Math.max(...boxes.map((box) => box.x + box.w));
  const bottom = Math.max(...boxes.map((box) => box.y + box.h));
  return {x: left, y: top, w: right - left, h: bottom - top};
}

/** Grows `box` by `padding` on each side, within the viewport. */
export function padBox(box: Box, padding: number, viewport: Size): Box {
  const left = Math.max(0, box.x - padding);
  const top = Math.max(0, box.y - padding);
  const right = Math.min(viewport.w, box.x + box.w + padding);
  const bottom = Math.min(viewport.h, box.y + box.h + padding);
  return {
    x: left,
    y: top,
    w: Math.max(0, right - left),
    h: Math.max(0, bottom - top),
  };
}

/** Whether boxes `a` and `b` overlap. */
export function overlaps(a: Box, b: Box): boolean {
  return (
    a.x < b.x + b.w && b.x < a.x + a.w && a.y < b.y + b.h && b.y < a.y + a.h
  );
}

/** Returns the area that boxes `a` and `b` share. */
function overlapArea(a: Box, b: Box): number {
  const w = Math.min(a.x + a.w, b.x + b.w) - Math.max(a.x, b.x);
  const h = Math.min(a.y + a.h, b.y + b.h) - Math.max(a.y, b.y);
  return w > 0 && h > 0 ? w * h : 0;
}

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(value, max));
}

/**
 * Places a card of size `card` next to `spotlight`, on the first side of
 * `placements` where it fits, and aligns it with the center of `anchor`. Inside
 * the spotlight, the card takes the first corner that keeps `avoid` in view,
 * rising above the spotlight if it must, or else the corner that covers the
 * least of it. A card that goes above the spotlight first, but does not fit
 * anywhere, rises to the top of the viewport to cover as little of it as it
 * can. Otherwise, centers the card in the viewport if there is no spotlight
 * or no room.
 */
export function placeCard(
  spotlight: Box | undefined,
  anchor: Box | undefined,
  card: Size,
  viewport: Size,
  placements: readonly TourPlacement[],
  avoid: readonly Box[] = [],
): CardPlacement {
  const maxX = viewport.w - card.w - VIEWPORT_MARGIN;
  const maxY = viewport.h - card.h - VIEWPORT_MARGIN;
  const center: CardPlacement = {
    x: Math.max(VIEWPORT_MARGIN, (viewport.w - card.w) / 2),
    y: Math.max(VIEWPORT_MARGIN, (viewport.h - card.h) / 2),
    side: 'center',
  };
  if (!spotlight) {
    return center;
  }
  const focus = anchor ?? spotlight;
  const alignedX = clamp(
    focus.x + focus.w / 2 - card.w / 2,
    VIEWPORT_MARGIN,
    maxX,
  );
  const alignedY = clamp(
    focus.y + focus.h / 2 - card.h / 2,
    VIEWPORT_MARGIN,
    maxY,
  );
  // How much of `avoid` the card covers at a point.
  const coverAt = (point: Point) =>
    avoid.reduce(
      (sum, clear) =>
        sum + overlapArea(clear, {...point, w: card.w, h: card.h}),
      0,
    );
  for (const side of placements) {
    if (side === 'right') {
      const x = spotlight.x + spotlight.w + CARD_GAP;
      if (x <= maxX) {
        return {x, y: alignedY, side};
      }
    } else if (side === 'left') {
      const x = spotlight.x - CARD_GAP - card.w;
      if (x >= VIEWPORT_MARGIN) {
        return {x, y: alignedY, side};
      }
    } else if (side === 'bottom') {
      const y = spotlight.y + spotlight.h + CARD_GAP;
      if (y <= maxY) {
        return {x: alignedX, y, side};
      }
    } else if (side === 'top') {
      const y = spotlight.y - CARD_GAP - card.h;
      if (y >= VIEWPORT_MARGIN) {
        return {x: alignedX, y, side};
      }
    } else if (
      card.w + 2 * INSIDE_INSET <= spotlight.w &&
      card.h + 2 * INSIDE_INSET <= spotlight.h
    ) {
      const left = spotlight.x + INSIDE_INSET;
      const right = spotlight.x + spotlight.w - card.w - INSIDE_INSET;
      const top = spotlight.y + INSIDE_INSET;
      const bottom = spotlight.y + spotlight.h - card.h - INSIDE_INSET;
      // Moves a card at a corner up, over the dimmed page, off what it covers.
      const raise = (corner: Point): Point => {
        const covered = avoid.filter(
          (clear) =>
            overlapArea(clear, {...corner, w: card.w, h: card.h}) > 0,
        );
        if (covered.length === 0) {
          return corner;
        }
        const y = Math.min(...covered.map((clear) => clear.y)) - CARD_GAP;
        return {
          x: corner.x,
          y: clamp(y - card.h, VIEWPORT_MARGIN, corner.y),
        };
      };
      // Bottom right first, like a dialog that waits out of the way.
      const corners: Point[] = [
        {x: right, y: bottom},
        {x: right, y: top},
        {x: left, y: bottom},
        {x: left, y: top},
      ];
      corners.push(raise(corners[1]), raise(corners[3]));
      let best = corners[0];
      let bestCover = Number.POSITIVE_INFINITY;
      for (const corner of corners) {
        const cover = coverAt(corner);
        if (cover < bestCover) {
          best = corner;
          bestCover = cover;
        }
        if (cover === 0) {
          break;
        }
      }
      return {...best, side};
    }
  }
  if (placements[0] === 'top' && spotlight.y > VIEWPORT_MARGIN) {
    return {x: alignedX, y: VIEWPORT_MARGIN, side: 'top'};
  }
  return center;
}

function round(value: number): number {
  return Math.round(value * 100) / 100;
}

/**
 * Returns the corner radii of a box around an element with `corners`, which
 * is `padding` larger on each side. The corners stay concentric with those of
 * the element, turn square where the box meets the edges of the viewport, and
 * fit in the box.
 */
export function fitCorners(
  corners: Corners,
  padding: number,
  box: Box,
  viewport: Size,
): Corners {
  const max = Math.max(0, Math.min(box.w, box.h) / 2);
  const atTop = box.y <= 0;
  const atLeft = box.x <= 0;
  const atRight = box.x + box.w >= viewport.w;
  const atBottom = box.y + box.h >= viewport.h;
  const fitRadius = (radius: number, isAtEdge: boolean) =>
    isAtEdge ? 0 : clamp(radius + padding, 0, max);
  return [
    fitRadius(corners[0], atTop || atLeft),
    fitRadius(corners[1], atTop || atRight),
    fitRadius(corners[2], atBottom || atRight),
    fitRadius(corners[3], atBottom || atLeft),
  ];
}

/**
 * Returns the outline of the backdrop, for `clip-path: path(evenodd, ...)`:
 * the viewport, with a hole of `hole` with rounded `corners`.
 */
export function backdropPath(
  viewport: Size,
  hole: Box,
  corners: Corners,
): string {
  const outline = `M0 0H${round(viewport.w)}V${round(viewport.h)}H0Z`;
  if (hole.w < 1 || hole.h < 1) {
    return outline;
  }
  const max = Math.min(hole.w, hole.h) / 2;
  const [tl, tr, br, bl] = corners.map((radius) =>
    round(clamp(radius, 0, max)),
  );
  const left = round(hole.x);
  const top = round(hole.y);
  const right = round(hole.x + hole.w);
  const bottom = round(hole.y + hole.h);
  const arc = (radius: number) => `A${radius} ${radius} 0 0 1`;
  return (
    `${outline}M${round(left + tl)} ${top}H${round(right - tr)}` +
    `${arc(tr)} ${right} ${round(top + tr)}V${round(bottom - br)}` +
    `${arc(br)} ${round(right - br)} ${bottom}H${round(left + bl)}` +
    `${arc(bl)} ${left} ${round(bottom - bl)}V${round(top + tl)}` +
    `${arc(tl)} ${round(left + tl)} ${top}Z`
  );
}

/**
 * Returns the part of the timeline `canvas` that draws blocks: right of the
 * labels of its lanes, which end where the minimap `track` starts, and below
 * its bundle ruler.
 */
export function timelineArea(canvas: Box, track?: Box): Box {
  const left = track
    ? clamp(track.x, canvas.x, canvas.x + canvas.w)
    : canvas.x + Math.min(DEFAULT_LABEL_WIDTH, canvas.w * 0.3);
  const right = track
    ? clamp(track.x + track.w, left, canvas.x + canvas.w)
    : canvas.x + canvas.w;
  const top = canvas.y + Math.min(RULER_HEIGHT, canvas.h * 0.2);
  return {
    x: left,
    y: top,
    w: Math.max(0, right - left),
    h: Math.max(0, canvas.y + canvas.h - top),
  };
}

/**
 * Returns where the details panel opens in the timeline and drawer `content`,
 * like the drawer of Trace Viewer v2 with its default size.
 */
export function drawerPreview(content: Box): Box {
  const height = (content.h * DEFAULT_DRAWER_SIZE_PERCENT) / 100;
  // Inside the padding of the content area, see the trace viewer stylesheet.
  return {
    x: content.x + 8,
    y: content.y + content.h - height + 8,
    w: Math.max(0, content.w - 24),
    h: Math.max(0, height - 16),
  };
}

/** Labels a ghost box: its own label, or the bundles that it spans. */
export function ghostLabel(ghost: GhostBox): string {
  if (ghost.bundlesPerPixel !== undefined) {
    const bundles = Math.max(1, Math.round(ghost.w * ghost.bundlesPerPixel));
    return plural(bundles, 'bundle');
  }
  return ghost.label ?? '';
}

/** Whether the user has seen this version of the tour. */
export function hasSeenKernelViewerTour(): boolean {
  try {
    return (
      window.localStorage.getItem(KERNEL_VIEWER_TOUR_STORAGE_KEY) ===
      TOUR_VERSION
    );
  } catch {
    // Without storage the tour could not remember being dismissed, so it does
    // not start by itself.
    return true;
  }
}

/** Records that the user has seen this version of the tour. */
export function markKernelViewerTourSeen() {
  try {
    window.localStorage.setItem(KERNEL_VIEWER_TOUR_STORAGE_KEY, TOUR_VERSION);
  } catch {
    // Storage can be unavailable, e.g. in sandboxed iframes.
  }
}

function prefersReducedMotion(): boolean {
  return (
    window.matchMedia?.('(prefers-reduced-motion: reduce)').matches ?? false
  );
}

function isEditableTarget(target: EventTarget | null): boolean {
  return (
    target instanceof HTMLElement &&
    (target.isContentEditable ||
      target.tagName === 'INPUT' ||
      target.tagName === 'TEXTAREA' ||
      target.tagName === 'SELECT')
  );
}

function viewportSize(): Size {
  const root = document.documentElement;
  return {
    w: root.clientWidth || window.innerWidth,
    h: root.clientHeight || window.innerHeight,
  };
}

/** Returns the boxes of the rendered `elements`. */
function visibleBoxes(elements: readonly Element[]): Box[] {
  const boxes: Box[] = [];
  for (const element of elements) {
    const rect = element.getBoundingClientRect();
    if (rect.width > 0 || rect.height > 0) {
      boxes.push({x: rect.left, y: rect.top, w: rect.width, h: rect.height});
    }
  }
  return boxes;
}

/** Reads a computed corner radius, e.g. `16px` or `50%` of `size`. */
function readRadius(value: string, size: number): number {
  // Elliptical radii, e.g. `16px 8px`, start with their horizontal radius.
  const match = /^\s*([\d.]+)(px|%)?/.exec(value);
  const radius = Number(match?.[1] ?? 0);
  if (!Number.isFinite(radius)) {
    return 0;
  }
  return match?.[2] === '%' ? (radius / 100) * size : radius;
}

/**
 * Returns the corner radii of `elements`: those of a single element, or the
 * largest radius of a group, all around.
 */
function cornersFor(elements: readonly Element[]): Corners {
  const corners = elements.map((element): Corners => {
    const style = getComputedStyle(element);
    const rect = element.getBoundingClientRect();
    const size = Math.min(rect.width, rect.height);
    return [
      readRadius(style.borderTopLeftRadius, size),
      readRadius(style.borderTopRightRadius, size),
      readRadius(style.borderBottomRightRadius, size),
      readRadius(style.borderBottomLeftRadius, size),
    ];
  });
  if (corners.length === 1) {
    return corners[0];
  }
  const radius = Math.max(0, ...corners.map((radii) => Math.max(...radii)));
  return [radius, radius, radius, radius];
}

function isSameList(a: readonly Element[], b: readonly Element[]): boolean {
  return a.length === b.length && a.every((element, i) => element === b[i]);
}

/** Sets the text of `element`, only if it changes. */
function setText(element: Element, text: string) {
  if (element.textContent !== text) {
    element.textContent = text;
  }
}

/** Returns a transform that moves an element to `point`. */
function translateTo({x, y}: Point): string {
  return `translate3d(${x}px, ${y}px, 0)`;
}

/** Animates one number with a damped spring. */
class Spring {
  value: number;
  target: number;
  private velocity = 0;

  constructor(
    value: number,
    private readonly stiffness: number,
    private readonly damping: number,
  ) {
    this.value = value;
    this.target = value;
  }

  /** Advances the spring by `dt` seconds, and returns whether it is at rest. */
  step(dt: number): boolean {
    const steps = Math.max(1, Math.ceil(dt / MAX_SPRING_STEP_SECONDS));
    const h = dt / steps;
    for (let i = 0; i < steps; i++) {
      const force =
        this.stiffness * (this.target - this.value) -
        this.damping * this.velocity;
      this.velocity += force * h;
      this.value += this.velocity * h;
    }
    if (
      Math.abs(this.target - this.value) < 0.25 &&
      Math.abs(this.velocity) < 4
    ) {
      this.snap();
    }
    return this.isAtRest;
  }

  get isAtRest(): boolean {
    return this.value === this.target && this.velocity === 0;
  }

  /** Jumps to the target. */
  snap() {
    this.value = this.target;
    this.velocity = 0;
  }

  /** Jumps to `value`, and stays there. */
  reset(value: number) {
    this.value = value;
    this.target = value;
    this.velocity = 0;
  }
}

/** Four numbers that animate together, e.g. a box or its corner radii. */
type Quad = readonly [number, number, number, number];

function boxQuad(box: Box): Quad {
  return [box.x, box.y, box.w, box.h];
}

function quadBox([x, y, w, h]: Quad): Box {
  return {x, y, w, h};
}

/** Animates four numbers with damped springs. */
class QuadSpring {
  private readonly springs: readonly [Spring, Spring, Spring, Spring];

  constructor(values: Quad, {stiffness, damping}: typeof SPOTLIGHT_SPRING) {
    this.springs = [
      new Spring(values[0], stiffness, damping),
      new Spring(values[1], stiffness, damping),
      new Spring(values[2], stiffness, damping),
      new Spring(values[3], stiffness, damping),
    ];
  }

  get value(): Quad {
    const [a, b, c, d] = this.springs;
    return [a.value, b.value, c.value, d.value];
  }

  get isAtRest(): boolean {
    return this.springs.every((spring) => spring.isAtRest);
  }

  setTarget(values: Quad) {
    this.springs.forEach((spring, i) => {
      spring.target = values[i];
    });
  }

  /** Advances the springs by `dt` seconds, and returns whether all rest. */
  step(dt: number): boolean {
    let isAtRest = true;
    for (const spring of this.springs) {
      isAtRest = spring.step(dt) && isAtRest;
    }
    return isAtRest;
  }

  snap() {
    for (const spring of this.springs) {
      spring.snap();
    }
  }

  reset(values: Quad) {
    this.springs.forEach((spring, i) => {
      spring.reset(values[i]);
    });
  }
}

/** The elements of the demo layer, which the tour moves frame by frame. */
interface DemoElements {
  readonly layer: HTMLElement;
  readonly pointer: HTMLElement;
  readonly caption: HTMLElement;
  readonly captionText: HTMLElement;
  readonly keys: HTMLElement;
  readonly ripple: HTMLElement;
  readonly beacon: HTMLElement;
  readonly ghost: HTMLElement;
  readonly ghostLabel: HTMLElement;
  readonly hotspots: readonly HTMLElement[];
}

/** Options of `KernelViewerTour.start`. */
export interface TourStartOptions {
  /** Element to focus once the tour closes. Defaults to the focused element. */
  readonly returnFocus?: HTMLElement;
}

/**
 * Guided tour of the Static Kernel Viewer.
 *
 * Dims the page except for a spotlight on the part of the viewer that each
 * step explains, and shows a card with an illustration next to it. The
 * spotlight stays interactive, so users can try what a step describes. Place
 * it inside the viewer, which it finds the spotlighted elements in, and call
 * `start` to show it.
 *
 * While the user reads, a demo pointer shows how to use the spotlighted part
 * of the viewer: it moves over the real elements, clicks, drags and presses
 * keys. It steps aside as soon as the user tries things, comes back once they
 * pause, and stops once they have done the task of the step.
 *
 * The tour renders in the top layer, so that it covers the whole page, and
 * animates the spotlight and the card with springs.
 */
@Component({
  standalone: true,
  selector: 'kernel-viewer-tour',
  templateUrl: './kernel_viewer_tour.ng.html',
  styleUrls: ['./kernel_viewer_tour.scss'],
  changeDetection: ChangeDetectionStrategy.OnPush,
  imports: [MatButtonModule, MatIconModule],
})
export class KernelViewerTour implements OnInit, OnDestroy {
  /** Emits the id of each step that the tour shows. */
  readonly stepChange = output<string>();
  /** Emits how the tour ended, as soon as it starts to close. */
  readonly closed = output<TourCloseReason>();
  /**
   * Emits the index of a Trace Viewer v2 event to select, so that the details
   * panel shows a real block: the block that the user last pointed at.
   */
  readonly selectEvent = output<number>();
  /**
   * Emits what a details step needs the details panel to show, if the tour
   * knows no block to select for it, e.g. if the user never pointed at one.
   * The viewer then selects a typical instruction or region, and reports the
   * selection right away.
   */
  readonly openDetails = output<DetailsKind>();

  @ViewChild('root', {static: true}) rootRef?: ElementRef<HTMLElement>;
  @ViewChild('backdrop', {static: true}) backdropRef?: ElementRef<HTMLElement>;
  @ViewChild('ring', {static: true}) ringRef?: ElementRef<HTMLElement>;
  @ViewChild('demo', {static: true}) demoRef?: ElementRef<HTMLElement>;
  @ViewChild('hotspots', {static: true}) hotspotsRef?: ElementRef<HTMLElement>;
  @ViewChild('card', {static: true}) cardRef?: ElementRef<HTMLElement>;
  @ViewChild('surface', {static: true}) surfaceRef?: ElementRef<HTMLElement>;
  @ViewChild('content', {static: true}) contentRef?: ElementRef<HTMLElement>;

  readonly steps = signal<readonly TourStep[]>([]);
  readonly index = signal(0);
  /** Which way the user last moved through the steps. */
  readonly direction = signal<'forward' | 'backward'>('forward');
  readonly isOpen = signal(false);
  readonly isClosing = signal(false);
  readonly isTaskDone = signal(false);
  readonly summary = signal(EMPTY_SUMMARY);
  /** Text for screen readers, e.g. the title of a new step. */
  readonly announcement = signal('');
  /** The block of the timeline that the user last pointed at, if any. */
  readonly hoveredBlock = signal<TimelineBlock | undefined>(undefined);
  /** Whether the pointer is on `hoveredBlock` right now. */
  readonly isPointing = signal(false);
  /** The block of the timeline that the user last selected, if any. */
  readonly selectedBlock = signal<TimelineBlock | undefined>(undefined);
  /** Who selected `selectedBlock`: the user, or the tour for them. */
  readonly selectionOrigin = signal<SelectionOrigin>('user');
  /**
   * What the details panel shows for the last selection that the tour saw,
   * as the panel builds it, e.g. the metrics of a region.
   */
  readonly selectedContent = signal<KernelSelectionContent | undefined>(
    undefined,
  );

  readonly step = computed<TourStep | undefined>(
    () => this.steps()[this.index()],
  );
  /** The current step in a list, so that its content renders anew per step. */
  readonly stepView = computed(() => {
    const step = this.step();
    return step ? [step] : [];
  });
  readonly isFirst = computed(() => this.index() === 0);
  readonly isLast = computed(() => this.index() === this.steps().length - 1);
  readonly isWelcome = computed(() => this.step()?.visual === 'welcome');
  readonly primaryLabel = computed(() =>
    this.isFirst() ? 'Start tour' : this.isLast() ? 'Finish' : 'Next',
  );
  readonly taskIcon = computed(() => {
    const gesture = this.step()?.task?.gesture;
    if (this.isTaskDone()) {
      return 'check';
    }
    if (gesture === 'select') {
      return 'mouse';
    }
    if (gesture === 'minimap') {
      return 'pan_tool';
    }
    return gesture === 'hover' ? 'near_me' : 'keyboard';
  });

  readonly railRows = computed(() => buildRailRows(this.summary().topKernels));
  /** The kernel that the search and share illustrations feature. */
  readonly featured = computed(() => this.railRows()[0]);
  readonly tabRows = computed(() => this.railRows().slice(0, 3));
  readonly searchQuery = computed(() => searchQueryFor(this.featured().name));
  readonly searchResult = computed(() =>
    splitAtQuery(this.featured().name, this.searchQuery()),
  );
  readonly kernelTotal = computed(() =>
    NUMBER_FORMAT.format(Math.max(this.summary().kernelCount, 2)),
  );
  /** The row of the timeline legend that the user last pointed at, if any. */
  readonly activeSection = computed(() => {
    const block = this.hoveredBlock();
    return block ? sectionOf(block.lane) : undefined;
  });
  readonly pointerReadout = computed(() => {
    const block = this.hoveredBlock();
    return block ? buildPointerReadout(block) : undefined;
  });
  readonly fieldSheet = computed(() => buildFieldSheet(this.selectedBlock()));
  /** The field sheet of the selected block in a list, empty without one. */
  readonly selectedSheet = computed(() => {
    const sheet = this.fieldSheet();
    return sheet.name ? [sheet] : [];
  });
  /** Whose block the field sheet shows. */
  readonly sheetCaption = computed(() => {
    if (!this.selectedBlock()) {
      return 'Nothing selected yet';
    }
    return captionFor(this.selectionOrigin());
  });
  /** What the current details step makes the details panel show. */
  readonly detailsKind = computed<DetailsKind>(() =>
    this.step()?.visual === 'region' ? 'region' : 'instruction',
  );
  /** The metrics that the details panel shows for the selection. */
  readonly panelMetrics = computed(() =>
    buildPanelMetrics(this.selectedContent(), this.detailsKind()),
  );
  /** The buttons of the header of the details panel. */
  readonly panelActions = computed(() =>
    buildPanelActions(this.selectedContent(), this.detailsKind()),
  );
  /** Whose selection the details panel shows. */
  readonly panelCaption = computed(() =>
    this.selectedContent()
      ? captionFor(this.selectionOrigin())
      : 'Nothing selected yet',
  );
  /**
   * The badge of the selection, like the details panel shows it: e.g. the
   * unit of an instruction, or `Region`.
   */
  readonly panelChips = computed(() => {
    const content = this.selectedContent();
    return content
      ? [
          {
            label: content.badgeLabel,
            color: content.unit?.color ?? content.accentColor,
          },
        ]
      : [];
  });

  readonly welcomeLanes = WELCOME_LANES;
  readonly navigationLanes = NAVIGATION_LANES;
  readonly timelineRows = TIMELINE_ROWS;
  readonly minimapRegions = MINIMAP_REGIONS;
  readonly minimapUtilization = MINIMAP_UTILIZATION;
  readonly particles = PARTICLES;
  readonly cheatsheet = CHEATSHEET;

  private readonly id = nextTourId++;
  readonly titleId = `kernel-viewer-tour-title-${this.id}`;
  readonly bodyId = `kernel-viewer-tour-body-${this.id}`;

  private readonly host =
    inject<ElementRef<HTMLElement>>(ElementRef).nativeElement;
  private readonly changeDetectorRef = inject(ChangeDetectorRef);
  private readonly ngZone = inject(NgZone);

  private readonly spotlight = new QuadSpring(SQUARE, SPOTLIGHT_SPRING);
  private readonly spotlightCorners = new QuadSpring(SQUARE, SPOTLIGHT_SPRING);
  private readonly ring = new QuadSpring(SQUARE, RING_SPRING);
  private readonly ringCorners = new QuadSpring(SQUARE, RING_SPRING);
  private readonly card = new QuadSpring(SQUARE, CARD_SPRING);
  private side: CardPlacement['side'] = 'center';
  private isRingShown = false;
  private spotlightElements: Element[] = [];
  private highlightElements: Element[] = [];
  private keepClearElements: Element[] = [];
  /** The spotlight, without padding or animation, if any. */
  private holeBox: Box | undefined = undefined;
  /** Where the card shrinks into while the tour closes, if anywhere. */
  private closeTarget: Box | undefined = undefined;
  private resizeObserver: ResizeObserver | undefined = undefined;
  private frameId = 0;
  private lastFrameTime = 0;
  private keepAliveUntil = 0;
  /** Whether the spotlight and the card may still move. */
  private isLayoutActive = false;
  private closeTimer: ReturnType<typeof setTimeout> | undefined = undefined;
  private returnFocus: HTMLElement | undefined = undefined;
  private reducedMotion = false;
  private isDestroyed = false;
  /**
   * The block that the user last pointed at, preferring instructions, which
   * the tour selects to show the details of a real block.
   */
  private pointedBlock: TimelineBlock | undefined = undefined;
  /**
   * While the tour selects a block, who to credit for it. Trace Viewer v2
   * reports selections right away, so the selections that it reports in the
   * meantime are the tour's.
   */
  private tourSelection: SelectionOrigin | undefined = undefined;
  /**
   * The regions around the instruction that the details panel last showed,
   * from the outermost to the innermost, which the region step zooms out to.
   */
  private regionsAround: readonly RegionRef[] = [];
  /**
   * Whether the details panel step still has to open the panel by itself,
   * e.g. once the trace loads.
   */
  private shouldOpenPanel = false;
  private openPanelTimer: ReturnType<typeof setTimeout> | undefined =
    undefined;
  /** Where the user last pointed at an instruction, and at any block. */
  private unitSighting: BlockSighting | undefined = undefined;
  private lastSighting: BlockSighting | undefined = undefined;
  /** The bundles that the timeline shows, once the minimap reports them. */
  private visibleRange: VisibleRange | undefined = undefined;
  /** When the user last pressed or scrolled on the minimap. */
  private minimapGestureAt = Number.NEGATIVE_INFINITY;
  /** Where the user pressed down on the minimap lens, if dragging it. */
  private minimapLensDownX: number | undefined = undefined;

  /** The demo of the current step, built from the elements of the viewer. */
  private demoScript: DemoScript | undefined = undefined;
  /** When the current loop of the demo started, or 0 while it does not play. */
  private demoStartedAt = 0;
  private demoWakeTimer: ReturnType<typeof setTimeout> | undefined = undefined;
  private demoWakeAt = 0;
  /** Whether the layout changed since the demo was built. */
  private isDemoStale = false;
  /** Whether the user dismissed the still demo of reduced motion. */
  private isStillDemoDismissed = false;
  private stepEnteredAt = 0;
  private lastActivityAt = Number.NEGATIVE_INFINITY;
  private demoElements: DemoElements | undefined = undefined;
  /** The caption that the demo shows, and its size, to place it. */
  private demoCaptionKey = '';
  private demoCaptionSize: Size = {w: 0, h: 0};

  ngOnInit() {
    // Listens in the capture phase from the start, before the WASM timeline
    // adds its own window listeners, so that tour keys reach the tour first.
    this.ngZone.runOutsideAngular(() => {
      window.addEventListener('keydown', this.onKeydown, {capture: true});
    });
  }

  ngOnDestroy() {
    this.isDestroyed = true;
    window.removeEventListener('keydown', this.onKeydown, {capture: true});
    this.teardown();
  }

  /** Shows the tour from its first step. */
  start(summary: KernelViewerTourSummary, options: TourStartOptions = {}) {
    if (this.isClosing()) {
      // Restarting while the tour closes starts it afresh.
      this.teardown();
      this.isOpen.set(false);
      this.isClosing.set(false);
    }
    const wasShown = this.isOpen();
    this.reducedMotion = prefersReducedMotion();
    this.summary.set(summary);
    this.steps.set(buildKernelViewerTourSteps(summary));
    this.index.set(0);
    this.direction.set('forward');
    this.isOpen.set(true);
    this.closeTarget = undefined;
    this.forgetBlocks();
    if (!wasShown) {
      this.returnFocus =
        options.returnFocus ??
        (document.activeElement instanceof HTMLElement
          ? document.activeElement
          : undefined);
    }
    this.changeDetectorRef.detectChanges();
    const root = this.rootRef?.nativeElement;
    if (root && typeof root.showPopover === 'function' && !wasShown) {
      try {
        root.showPopover();
      } catch {
        // The element cannot be shown, e.g. if it is not in the document.
      }
    }
    if (!wasShown) {
      // The spotlight closes in on its first target from the whole viewport,
      // like an iris, while the card scales in at its place.
      const viewport = viewportSize();
      this.spotlight.reset([0, 0, viewport.w, viewport.h]);
      this.spotlightCorners.reset(SQUARE);
      this.isRingShown = false;
      this.addOpenListeners();
    }
    this.enterStep({placeCardNow: !wasShown});
  }

  /** Shows the next step, or completes the tour from its last step. */
  next() {
    if (!this.isOpen() || this.isClosing()) {
      return;
    }
    if (this.isLast()) {
      this.close('completed');
      return;
    }
    this.direction.set('forward');
    this.index.update((index) => index + 1);
    this.enterStep();
  }

  /** Shows the previous step. */
  back() {
    if (!this.isOpen() || this.isClosing() || this.isFirst()) {
      return;
    }
    this.direction.set('backward');
    this.index.update((index) => index - 1);
    this.enterStep();
  }

  /** Closes the tour, and remembers that the user has seen it. */
  close(reason: TourCloseReason) {
    if (!this.isOpen() || this.isClosing()) {
      return;
    }
    markKernelViewerTourSeen();
    this.isClosing.set(true);
    this.announcement.set('');
    this.stopDemo();
    this.renderHotspots(undefined);
    // The card shrinks into the tour button, which is where the tour lives.
    const button = this.findElements(['.tour-button'])[0];
    const [buttonBox] = button ? visibleBoxes([button]) : [];
    this.closeTarget = buttonBox;
    this.changeDetectorRef.detectChanges();
    this.restoreFocus();
    this.closed.emit(reason);
    this.kick(CLOSE_DURATION_MS);
    this.closeTimer = setTimeout(
      () => {
        this.teardown();
        this.isOpen.set(false);
        this.isClosing.set(false);
        this.changeDetectorRef.detectChanges();
      },
      this.reducedMotion ? 0 : CLOSE_DURATION_MS,
    );
  }

  /**
   * Closes the welcome step on a click outside of the card. Later steps nudge
   * the card instead, so that a stray click does not end the tour.
   */
  onBackdropClick() {
    if (this.isWelcome()) {
      this.close('dismissed');
      return;
    }
    if (this.reducedMotion) {
      return;
    }
    this.surfaceRef?.nativeElement.animate?.(
      [
        {transform: 'scale(1)'},
        {transform: 'scale(1.03)'},
        {transform: 'scale(0.995)'},
        {transform: 'scale(1)'},
      ],
      {duration: 320, easing: 'cubic-bezier(0.2, 0, 0, 1)'},
    );
  }

  private enterStep({placeCardNow = false} = {}) {
    const step = this.step();
    if (!step) {
      return;
    }
    this.isTaskDone.set(false);
    clearTimeout(this.openPanelTimer);
    this.openPanelTimer = undefined;
    const stepText =
      `Step ${this.index() + 1} of ${this.steps().length}. ${step.title}. ` +
      step.body;
    this.announcement.set(stepText);
    this.stepChange.emit(step.id);
    this.changeDetectorRef.detectChanges();
    this.stepEnteredAt = Date.now();
    this.stopDemo();
    this.isStillDemoDismissed = false;
    this.resolveElements(step);
    this.demoScript = this.buildDemo(step);
    this.isDemoStale = false;
    if (placeCardNow) {
      this.layout();
      this.card.snap();
      if (!this.holeBox) {
        this.spotlight.snap();
        this.spotlightCorners.snap();
      }
      this.render();
    }
    this.focusPrimaryButton();
    this.shouldOpenPanel = step.visual === 'panel' || step.visual === 'region';
    const opened = this.openPanel();
    if (opened) {
      this.announcement.set(`${stepText} ${opened}`);
      this.changeDetectorRef.detectChanges();
    }
    this.kick();
  }

  /**
   * Makes the details panel show what its step explains, an instruction or a
   * region, unless it does already. Users may never have clicked a block, and
   * would not know how the panel opens.
   *
   * For an instruction, it selects the instruction that the user selected,
   * else the most frequent instruction of what the panel shows, else the
   * instruction that the user pointed at. For a region, it selects the region
   * around the instruction that the panel showed, else the region that the
   * user selected. Failing that, it asks the viewer to pick a typical one.
   * Trace Viewer v2 zooms to the block, and reports the selection right away.
   *
   * Returns what to announce if it changed what the panel shows.
   */
  private openPanel(): string | undefined {
    const visual = this.step()?.visual;
    if (!this.shouldOpenPanel || (visual !== 'panel' && visual !== 'region')) {
      return undefined;
    }
    const kind = this.detailsKind();
    if (this.isShowing(kind)) {
      this.shouldOpenPanel = false;
      return undefined;
    }
    const content = this.selectedContent();
    const selected = this.selectedBlock();
    const attempts: Array<[SelectionOrigin, () => void]> = [];
    const select = (origin: SelectionOrigin, index: number) => {
      attempts.push([origin, () => this.selectEvent.emit(index)]);
    };
    if (kind === 'instruction') {
      if (selected && sectionOf(selected.lane) === 'units') {
        select(this.selectionOrigin(), selected.index);
      }
      const top = content?.topInstructions?.find(
        (item) => item.firstEventIndex >= 0,
      );
      if (top) {
        select('picked', top.firstEventIndex);
      }
      const peer = content?.bundleEvents?.[0];
      if (peer) {
        select('picked', peer.eventIndex);
      }
      const pointed = this.pointedBlock;
      if (pointed && sectionOf(pointed.lane) === 'units') {
        select('pointed', pointed.index);
      }
    } else {
      const around = pickRegionAround(this.regionsAround);
      if (around) {
        select('picked', around.index);
      }
      if (selected && sectionOf(selected.lane) === 'regions') {
        select(this.selectionOrigin(), selected.index);
      }
    }
    attempts.push([
      'picked',
      () => {
        this.openDetails.emit(kind);
      },
    ]);
    this.ngZone.run(() => {
      for (const [origin, attempt] of attempts) {
        this.selectAs(origin, attempt);
        if (this.isShowing(kind)) {
          break;
        }
      }
    });
    if (!this.isShowing(kind)) {
      // Tries again once the trace loads, which the minimap reports.
      return undefined;
    }
    this.shouldOpenPanel = false;
    this.isDemoStale = true;
    const shown = this.selectedBlock();
    return shown
      ? `Opened the details of ${shown.name}, ` +
          `${bundlesOf(shown.start, shown.length)}.`
      : undefined;
  }

  /** Whether the details panel shows a selection of `kind`. */
  private isShowing(kind: DetailsKind): boolean {
    if (!this.isPanelShown()) {
      return false;
    }
    const content = this.selectedContent();
    if (!content) {
      // The panel opened before the tour, which then saw no selection.
      const grid = kind === 'region' ? 'range' : 'single';
      return this.findElements([`${PANEL_SELECTOR} .${grid}-mode-grid`])
        .length > 0;
    }
    return kind === 'region'
      ? isRegionDetails(content)
      : isInstructionDetails(content);
  }

  /** Runs `select`, crediting the selections that it makes to `origin`. */
  private selectAs(origin: SelectionOrigin, select: () => void) {
    this.tourSelection = origin;
    try {
      select();
    } finally {
      this.tourSelection = undefined;
    }
  }

  /** Whether the details panel shows a selection. */
  private isPanelShown(): boolean {
    return this.findElements([PANEL_SELECTOR]).length > 0;
  }

  /** Marks the task of the step done, and announces it after `prefix`. */
  private completeTask(prefix = '') {
    const task = this.step()?.task;
    if (!task || this.isTaskDone()) {
      return;
    }
    this.isTaskDone.set(true);
    this.announcement.set(prefix + task.success);
    this.stopDemo();
    this.changeDetectorRef.detectChanges();
    this.kick();
  }

  /** Finds the elements that the step spotlights, rings, and keeps clear. */
  private resolveElements(step: TourStep) {
    let spotlightElements = this.findElements(step.spotlight);
    if (spotlightElements.length === 0 && step.fallback) {
      spotlightElements = this.findElements(step.fallback);
    }
    const highlightElements = step.highlight
      ? this.findElements(step.highlight)
      : [];
    const keepClearElements = step.keepClear
      ? this.findElements(step.keepClear)
      : [];
    if (
      isSameList(spotlightElements, this.spotlightElements) &&
      isSameList(highlightElements, this.highlightElements) &&
      isSameList(keepClearElements, this.keepClearElements)
    ) {
      return;
    }
    this.spotlightElements = spotlightElements;
    this.highlightElements = highlightElements;
    this.keepClearElements = keepClearElements;
    this.isDemoStale = true;
    const observer = this.resizeObserver;
    if (observer) {
      observer.disconnect();
      const content = this.contentRef?.nativeElement;
      if (content) {
        observer.observe(content);
      }
      for (const element of [...spotlightElements, ...highlightElements]) {
        observer.observe(element);
      }
    }
  }

  /** Finds the first element that matches each of `selectors` in the viewer. */
  private findElements(selectors: readonly string[]): Element[] {
    const scope = this.host.parentElement ?? document.body;
    const elements: Element[] = [];
    for (const selector of selectors) {
      const element = scope.querySelector(selector);
      if (element && !this.host.contains(element)) {
        elements.push(element);
      }
    }
    return elements;
  }

  /** Returns the boxes of the shown elements that match `selector`. */
  private findBoxes(selector: string): Box[] {
    const scope = this.host.parentElement ?? document.body;
    const elements = Array.from(scope.querySelectorAll(selector)).filter(
      (element) => !this.host.contains(element),
    );
    return visibleBoxes(elements);
  }

  private focusPrimaryButton() {
    this.host
      .querySelector<HTMLElement>('.tour-primary')
      ?.focus({preventScroll: true});
  }

  private restoreFocus() {
    const target = this.returnFocus;
    this.returnFocus = undefined;
    if (
      target?.isConnected &&
      target !== document.body &&
      !this.host.contains(target)
    ) {
      target.focus({preventScroll: true});
    }
    // Focus never stays in the closing tour, e.g. if the target was removed.
    const focused = document.activeElement;
    if (focused instanceof HTMLElement && this.host.contains(focused)) {
      focused.blur();
    }
  }

  private addOpenListeners() {
    this.ngZone.runOutsideAngular(() => {
      window.addEventListener('resize', this.onLayoutChange);
      document.addEventListener('transitionrun', this.onTransition, true);
      document.addEventListener('transitionend', this.onTransition, true);
      document.addEventListener('focusin', this.onFocusIn, true);
      document.addEventListener('pointermove', this.onPointerActivity, {
        capture: true,
        passive: true,
      });
      document.addEventListener('pointerdown', this.onPointerActivity, {
        capture: true,
        passive: true,
      });
      document.addEventListener('wheel', this.onPointerActivity, {
        capture: true,
        passive: true,
      });
      document.addEventListener('click', this.onPanelClick, true);
      window.addEventListener(EVENT_HOVERED_EVENT_NAME, this.onBlockHovered);
      window.addEventListener(EVENT_SELECTED_EVENT_NAME, this.onBlockSelected);
      window.addEventListener(
        EVENTS_SELECTED_EVENT_NAME,
        this.onRangeSelected,
      );
      window.addEventListener(
        MINIMAP_UPDATED_EVENT_NAME,
        this.onMinimapUpdated,
      );
      window.addEventListener(
        LOADING_STATUS_UPDATE_EVENT_NAME,
        this.onLoadingStatus,
      );
      if (typeof ResizeObserver !== 'undefined') {
        this.resizeObserver = new ResizeObserver(this.onLayoutChange);
      }
      document.fonts.ready.then(this.onLayoutChange, () => {});
    });
  }

  private teardown() {
    clearTimeout(this.closeTimer);
    this.closeTimer = undefined;
    clearTimeout(this.demoWakeTimer);
    this.demoWakeTimer = undefined;
    clearTimeout(this.openPanelTimer);
    this.openPanelTimer = undefined;
    this.shouldOpenPanel = false;
    cancelAnimationFrame(this.frameId);
    this.frameId = 0;
    this.isLayoutActive = false;
    this.demoStartedAt = 0;
    this.demoScript = undefined;
    window.removeEventListener('resize', this.onLayoutChange);
    document.removeEventListener('transitionrun', this.onTransition, true);
    document.removeEventListener('transitionend', this.onTransition, true);
    document.removeEventListener('focusin', this.onFocusIn, true);
    document.removeEventListener('pointermove', this.onPointerActivity, true);
    document.removeEventListener('pointerdown', this.onPointerActivity, true);
    document.removeEventListener('wheel', this.onPointerActivity, true);
    document.removeEventListener('click', this.onPanelClick, true);
    window.removeEventListener(EVENT_HOVERED_EVENT_NAME, this.onBlockHovered);
    window.removeEventListener(EVENT_SELECTED_EVENT_NAME, this.onBlockSelected);
    window.removeEventListener(
      EVENTS_SELECTED_EVENT_NAME,
      this.onRangeSelected,
    );
    window.removeEventListener(
      MINIMAP_UPDATED_EVENT_NAME,
      this.onMinimapUpdated,
    );
    window.removeEventListener(
      LOADING_STATUS_UPDATE_EVENT_NAME,
      this.onLoadingStatus,
    );
    this.resizeObserver?.disconnect();
    this.resizeObserver = undefined;
    this.spotlightElements = [];
    this.highlightElements = [];
    this.keepClearElements = [];
    this.holeBox = undefined;
    this.renderDemo(undefined);
    this.renderHotspots(undefined);
    const root = this.rootRef?.nativeElement;
    if (root && typeof root.hidePopover === 'function') {
      try {
        root.hidePopover();
      } catch {
        // Not shown.
      }
    }
  }

  private readonly onLayoutChange = () => {
    this.isDemoStale = true;
    this.kick();
  };

  private readonly onTransition = (event: Event) => {
    // Transitions of the tour itself, e.g. of its buttons, move nothing.
    if (!(event.target instanceof Node && this.host.contains(event.target))) {
      this.isDemoStale = true;
      this.kick();
    }
  };

  /**
   * Renders the step now if its illustration follows the timeline, and lets
   * the card follow its size if `relayout`.
   */
  private refreshLiveVisual({relayout = true} = {}) {
    const visual = this.step()?.visual;
    if (
      visual === 'legend' ||
      visual === 'fields' ||
      visual === 'panel' ||
      visual === 'region'
    ) {
      this.changeDetectorRef.detectChanges();
      if (relayout) {
        this.kick();
      }
    }
  }

  /** Forgets the blocks of the timeline, e.g. as it loads another kernel. */
  private forgetBlocks() {
    this.pointedBlock = undefined;
    this.unitSighting = undefined;
    this.lastSighting = undefined;
    this.visibleRange = undefined;
    this.regionsAround = [];
    this.hoveredBlock.set(undefined);
    this.isPointing.set(false);
    this.selectedBlock.set(undefined);
    this.selectionOrigin.set('user');
    this.selectedContent.set(undefined);
  }

  /** Maps the bundles of the timeline to pixels, once the minimap shows. */
  private timelineMapping(): TimelineMapping | undefined {
    const track = this.findBoxes(MINIMAP_TRACK_SELECTOR)[0];
    if (!track) {
      return undefined;
    }
    let range = this.visibleRange;
    if (!range) {
      const lens = this.findElements(['kernel-minimap .minimap-lens'])[0];
      const [lensBox] = lens ? visibleBoxes([lens]) : [];
      range = lens && lensBox ? readLensRange(lens, lensBox, track) : undefined;
    }
    return range ? {track, ...range} : undefined;
  }

  /**
   * Remembers where the user pointed at `block`, so that the demo of the
   * inspect step can click a real block, even after zooming and panning.
   */
  private rememberSighting(block: TimelineBlock, detail: unknown) {
    const offset = readPointerOffset(detail);
    const canvas = this.findBoxes(CANVAS_SELECTOR)[0];
    if (!offset || !canvas) {
      return;
    }
    const mapping = this.timelineMapping();
    const sighting: BlockSighting = {
      lane: block.lane,
      offset,
      bundle: mapping ? bundleAt(canvas.x + offset.x, mapping) : undefined,
    };
    this.lastSighting = sighting;
    if (sectionOf(block.lane) === 'units') {
      this.unitSighting = sighting;
    }
  }

  /**
   * Follows the pointer over the timeline. Trace Viewer v2 reports a new
   * block, or none, whenever the block under the pointer changes.
   */
  private readonly onBlockHovered = (event: Event) => {
    if (!this.isOpen() || this.isClosing() || !(event instanceof CustomEvent)) {
      return;
    }
    const block = readTimelineBlock(event.detail);
    if (
      block &&
      block.index >= 0 &&
      (sectionOf(block.lane) === 'units' ||
        sectionOf(this.pointedBlock?.lane ?? '') !== 'units')
    ) {
      this.pointedBlock = block;
    }
    if (block && block.index >= 0) {
      this.rememberSighting(block, event.detail);
    }
    // The legend keeps the last block once the pointer leaves it, so that it
    // does not flicker as the pointer crosses the gaps between blocks.
    const shown = block ?? this.hoveredBlock();
    const isPointing = block !== undefined;
    const prev = this.hoveredBlock();
    if (
      shown?.index === prev?.index &&
      shown?.lane === prev?.lane &&
      shown?.start === prev?.start &&
      isPointing === this.isPointing()
    ) {
      return;
    }
    this.hoveredBlock.set(shown);
    this.isPointing.set(isPointing);
    if (block && this.step()?.task?.gesture === 'hover') {
      this.ngZone.run(() => {
        this.completeTask();
      });
    }
    this.refreshLiveVisual({relayout: false});
  };

  /**
   * Shows the fields of the block that the user, or the tour, selected, and
   * the metrics that the details panel shows for it.
   */
  private readonly onBlockSelected = (event: Event) => {
    if (!this.isOpen() || this.isClosing() || !(event instanceof CustomEvent)) {
      return;
    }
    const origin = this.tourSelection;
    const isByTour = origin !== undefined;
    // The panel builds what it shows from the same detail.
    const content = buildKernelSelectionFromEvent(event.detail);
    if (content) {
      this.selectedContent.set(content);
      this.selectionOrigin.set(origin ?? 'user');
      if (isInstructionDetails(content)) {
        this.regionsAround = readRegionAncestors(event.detail);
      }
      this.isDemoStale = true;
    }
    const selected = readTimelineBlock(event.detail);
    if (!selected || selected.index < 0) {
      // Clearing the selection keeps the fields of the last block in view,
      // and points of counters are not blocks.
      if (content) {
        this.refreshLiveVisual();
      }
      return;
    }
    // Falls back to the hovered lane if the selection event omits trackName.
    const lane = [this.hoveredBlock(), this.pointedBlock].find(
      (block) => block?.index === selected.index,
    )?.lane;
    const block = lane ? {...selected, lane} : selected;
    this.selectedBlock.set(block);
    this.selectionOrigin.set(origin ?? 'user');
    this.isDemoStale = true;
    const step = this.step();
    if (step && !isByTour) {
      const bundles = bundlesOf(block.start, block.length);
      const selection = `Selected ${block.name}, ${bundles}. `;
      this.ngZone.run(() => {
        if (step.task?.gesture === 'select' && !this.isTaskDone()) {
          this.completeTask(selection);
        } else if (step.visual === 'fields') {
          this.announcement.set(selection);
        }
      });
    }
    this.refreshLiveVisual();
  };

  /**
   * Follows the details panel as it sums up a measured span or a box of
   * blocks, which it shows like a region.
   */
  private readonly onRangeSelected = (event: Event) => {
    if (!this.isOpen() || this.isClosing() || !(event instanceof CustomEvent)) {
      return;
    }
    const content = buildKernelSelectionFromRange(event.detail);
    if (!content) {
      return;
    }
    this.selectedContent.set(content);
    this.selectionOrigin.set('user');
    this.isDemoStale = true;
    this.ngZone.run(() => {
      this.refreshLiveVisual();
    });
  };

  /**
   * Completes the task of a details step when the user clicks an interactive
   * control of the panel, including the unit filters and search buttons that
   * select no block.
   */
  private readonly onPanelClick = (event: Event) => {
    const visual = this.step()?.visual;
    if (
      !this.isOpen() ||
      this.isClosing() ||
      (visual !== 'panel' && visual !== 'region') ||
      this.isTaskDone() ||
      !(event.target instanceof Element)
    ) {
      return;
    }
    const button = event.target.closest(
      'kernel-selection-panel button:not([disabled]):not(.close-btn):not(.is-selected)',
    );
    if (button) {
      this.ngZone.run(() => {
        this.completeTask();
      });
    }
  };

  /**
   * Follows the bundles that the timeline shows, and completes the task of the
   * minimap step once the user moves the lens. Once the trace shows, the
   * details panel step opens the panel if it could not before.
   */
  private readonly onMinimapUpdated = (event: Event) => {
    if (!this.isOpen() || this.isClosing() || !(event instanceof CustomEvent)) {
      return;
    }
    const range = readVisibleRange(event.detail);
    if (!range) {
      return;
    }
    if (this.shouldOpenPanel && this.openPanelTimer === undefined) {
      // Waits for Trace Viewer v2 to finish the frame that reported the
      // update before selecting a block in it. Tries once more only, so that
      // a panel which the user closed stays closed.
      this.openPanelTimer = setTimeout(() => {
        this.openPanelTimer = undefined;
        const opened = this.openPanel();
        this.shouldOpenPanel = false;
        if (opened) {
          this.announcement.set(opened);
          this.changeDetectorRef.detectChanges();
          this.kick();
        }
      });
    }
    const previous = this.visibleRange;
    this.visibleRange = range;
    if (previous?.start === range.start && previous?.end === range.end) {
      return;
    }
    this.isDemoStale = true;
    const isFromMinimap =
      Date.now() - this.minimapGestureAt < MINIMAP_GESTURE_MS;
    if (this.step()?.task?.gesture === 'minimap' && isFromMinimap) {
      this.ngZone.run(() => {
        this.completeTask();
      });
    }
    this.kick();
  };

  /** Forgets the blocks of a kernel as the timeline loads another one. */
  private readonly onLoadingStatus = (event: Event) => {
    const detail: unknown = event instanceof CustomEvent ? event.detail : {};
    const status =
      typeof detail === 'object' && detail !== null
        ? (detail as Record<string, unknown>)['status']
        : undefined;
    if (
      typeof status !== 'string' ||
      status === TraceViewerV2LoadingStatus.IDLE ||
      status === TraceViewerV2LoadingStatus.ERROR
    ) {
      return;
    }
    this.forgetBlocks();
    this.isDemoStale = true;
    this.refreshLiveVisual();
  };

  /**
   * Notices the user trying things in the spotlight, so that the demo steps
   * aside, and gestures on the minimap, for its task.
   */
  private readonly onPointerActivity = (event: Event) => {
    if (
      !this.isOpen() ||
      this.isClosing() ||
      !(event instanceof MouseEvent)
    ) {
      return;
    }
    const target = event.target;
    if (
      target instanceof Node &&
      this.cardRef?.nativeElement.contains(target)
    ) {
      return;
    }
    if (event.type !== 'pointermove') {
      if (target instanceof Element && target.closest('kernel-minimap')) {
        this.minimapGestureAt = Date.now();
        this.minimapLensDownX =
          event.type === 'pointerdown' && target.closest('.minimap-lens')
            ? event.clientX
            : undefined;
      } else {
        this.minimapLensDownX = undefined;
      }
    } else if (event.buttons > 0) {
      if (Date.now() - this.minimapGestureAt < MINIMAP_GESTURE_MS) {
        this.minimapGestureAt = Date.now();
      }
      if (
        this.minimapLensDownX !== undefined &&
        Math.abs(event.clientX - this.minimapLensDownX) >= 8 &&
        this.step()?.task?.gesture === 'minimap' &&
        !this.isTaskDone()
      ) {
        const track = this.findBoxes(MINIMAP_TRACK_SELECTOR)[0];
        const lens = this.findBoxes('kernel-minimap .minimap-lens')[0];
        // When the lens already spans the whole schedule, panning its body is
        // clamped and does not emit minimap_updated.
        if (track && lens && lens.w >= track.w - 2) {
          this.minimapLensDownX = undefined;
          this.ngZone.run(() => {
            this.completeTask();
          });
        }
      }
    } else {
      this.minimapLensDownX = undefined;
    }
    const hole = this.holeBox;
    if (
      hole &&
      event.clientX >= hole.x &&
      event.clientX <= hole.x + hole.w &&
      event.clientY >= hole.y &&
      event.clientY <= hole.y + hole.h
    ) {
      this.noteActivity();
    }
  };

  /** Hides the demo while the user is busy, until they pause. */
  private noteActivity() {
    this.lastActivityAt = Date.now();
    if (this.reducedMotion) {
      if (!this.isStillDemoDismissed) {
        this.isStillDemoDismissed = true;
        this.renderDemo(undefined);
      }
      return;
    }
    if (this.demoStartedAt) {
      this.stopDemo();
      this.requestFrame();
    }
  }

  /**
   * Whether `event` comes from another dialog, e.g. the shortcut help that ?
   * opens over the tour. Such dialogs handle their own keys and focus.
   */
  private isFromOtherDialog(event: Event): boolean {
    const scope = this.host.parentElement ?? document.body;
    const helpDialog = scope.querySelector('trace-viewer-help-dialog') as
      | (Element & {open?: boolean})
      | null;
    if (helpDialog?.open) {
      return true;
    }
    const card = this.cardRef?.nativeElement;
    return event
      .composedPath()
      .some(
        (node) =>
          node !== card &&
          ((node instanceof HTMLDialogElement && node.open) ||
            (node instanceof HTMLElement &&
              node.getAttribute('role') === 'dialog')),
      );
  }

  /** Keeps keyboard focus in the card or the spotlight, not the dim page. */
  private readonly onFocusIn = (event: FocusEvent) => {
    const target = event.target;
    if (!this.isOpen() || this.isClosing() || !(target instanceof Node)) {
      return;
    }
    if (
      this.step()?.id === 'search' &&
      target instanceof Element &&
      target.closest('.rail-search')
    ) {
      this.noteActivity();
      if (!this.isTaskDone()) {
        this.ngZone.run(() => {
          this.completeTask();
        });
      }
      return;
    }
    if (
      this.host.contains(target) ||
      this.spotlightElements.some((element) => element.contains(target)) ||
      (this.step()?.id === 'tabs' &&
        target instanceof Element &&
        Boolean(target.closest('.kernel-rail.expanded'))) ||
      this.isFromOtherDialog(event)
    ) {
      return;
    }
    this.focusPrimaryButton();
  };

  private readonly onKeydown = (event: KeyboardEvent) => {
    if (!this.isOpen() || this.isClosing() || this.isFromOtherDialog(event)) {
      return;
    }
    const target = event.target;
    const isEditable = isEditableTarget(target);
    const hasModifier = event.altKey || event.ctrlKey || event.metaKey;
    if (event.key === 'Escape' && !isEditable) {
      event.preventDefault();
      event.stopImmediatePropagation();
      this.ngZone.run(() => {
        this.close('dismissed');
      });
      return;
    }
    const isInCard =
      target instanceof Node &&
      (this.cardRef?.nativeElement.contains(target) ?? false);
    if (
      isInCard &&
      !hasModifier &&
      !event.shiftKey &&
      (event.key === 'ArrowRight' || event.key === 'ArrowLeft')
    ) {
      event.preventDefault();
      event.stopImmediatePropagation();
      if (!event.repeat) {
        this.ngZone.run(() => {
          if (event.key === 'ArrowRight') {
            this.next();
          } else {
            this.back();
          }
        });
      }
      return;
    }
    // Task keys also reach the viewer, which does what the task describes.
    const task = this.step()?.task;
    const isTaskKey =
      !isEditable &&
      !hasModifier &&
      (task?.keys?.includes(event.key.toLowerCase()) ?? false);
    if (isTaskKey || (!isInCard && event.key !== 'Tab')) {
      this.noteActivity();
    }
    if (task && isTaskKey && !this.isTaskDone()) {
      this.ngZone.run(() => {
        this.completeTask();
      });
    }
  };

  /** Keeps the animation running for at least `durationMs`. */
  private kick(durationMs = KEEP_ALIVE_MS) {
    if (!this.isOpen() || this.isDestroyed) {
      return;
    }
    const now = Date.now();
    this.keepAliveUntil = Math.max(
      this.keepAliveUntil,
      now + (this.reducedMotion ? 0 : durationMs),
    );
    this.isLayoutActive = true;
    this.requestFrame();
  }

  /** Renders a frame soon, unless one is already on its way. */
  private requestFrame() {
    if (this.frameId || !this.isOpen() || this.isDestroyed) {
      return;
    }
    this.lastFrameTime = Date.now();
    this.frameId = this.ngZone.runOutsideAngular(() =>
      requestAnimationFrame(this.onFrame),
    );
  }

  private readonly onFrame = () => {
    this.frameId = 0;
    if (!this.isOpen() || this.isDestroyed) {
      return;
    }
    const now = Date.now();
    const dt = Math.min((now - this.lastFrameTime) / 1000, MAX_FRAME_SECONDS);
    this.lastFrameTime = now;
    let needsFrame = false;
    if (this.isLayoutActive) {
      this.layout();
      let isAtRest = true;
      for (const spring of [
        this.spotlight,
        this.spotlightCorners,
        this.ring,
        this.ringCorners,
        this.card,
      ]) {
        if (this.reducedMotion) {
          spring.snap();
        } else if (dt > 0) {
          isAtRest = spring.step(dt) && isAtRest;
        } else {
          isAtRest = false;
        }
      }
      this.render();
      this.isLayoutActive = !isAtRest || now < this.keepAliveUntil;
      needsFrame = this.isLayoutActive;
    }
    needsFrame = this.updateDemo(now) || needsFrame;
    if (needsFrame) {
      this.frameId = requestAnimationFrame(this.onFrame);
    }
  };

  /** Measures the step's elements and the card, and retargets the springs. */
  private layout() {
    const viewport = viewportSize();
    const content = this.contentRef?.nativeElement;
    const cardSize = {
      w: content?.offsetWidth ?? 0,
      h: content?.offsetHeight ?? 0,
    };
    if (this.isClosing()) {
      // The spotlight opens up again, and the card shrinks away.
      this.spotlight.setTarget([0, 0, viewport.w, viewport.h]);
      this.spotlightCorners.setTarget(SQUARE);
      const target = this.closeTarget;
      if (target) {
        this.card.setTarget(boxQuad(target));
      }
      return;
    }
    const step = this.step();
    if (step) {
      // Elements come and go, e.g. the details panel as a block is selected.
      this.resolveElements(step);
      if (this.isDemoStale) {
        this.demoScript = this.buildDemo(step);
        this.isDemoStale = false;
      }
    }
    const padding = step?.padding ?? DEFAULT_PADDING;
    const spotlightBox = unionBox(visibleBoxes(this.spotlightElements));
    this.holeBox = spotlightBox;
    const hole = spotlightBox
      ? padBox(spotlightBox, padding, viewport)
      : undefined;
    const holeCorners = hole
      ? fitCorners(cornersFor(this.spotlightElements), padding, hole, viewport)
      : SQUARE;
    const highlightBox = unionBox(visibleBoxes(this.highlightElements));
    const ringBox = highlightBox
      ? padBox(highlightBox, RING_PADDING, viewport)
      : hole;
    const ringCorners =
      highlightBox && ringBox
        ? fitCorners(
            cornersFor(this.highlightElements),
            RING_PADDING,
            ringBox,
            viewport,
          )
        : holeCorners;
    const avoid = visibleBoxes(this.keepClearElements);
    const focus = this.demoScript?.focus;
    if (focus) {
      avoid.push(focus);
    }
    const placement = placeCard(
      hole,
      ringBox,
      cardSize,
      viewport,
      step?.placements ?? [],
      avoid,
    );
    this.side = placement.side;
    this.card.setTarget([placement.x, placement.y, cardSize.w, cardSize.h]);
    if (hole) {
      this.spotlight.setTarget(boxQuad(hole));
    } else {
      // Without a spotlight, the iris closes behind the center of the card.
      this.spotlight.setTarget([
        placement.x + cardSize.w / 2,
        placement.y + cardSize.h / 2,
        0,
        0,
      ]);
    }
    this.spotlightCorners.setTarget(holeCorners);
    if (ringBox && !this.isRingShown) {
      // A new ring appears in place rather than flying in.
      this.ring.reset(boxQuad(ringBox));
      this.ringCorners.reset(ringCorners);
    } else if (ringBox) {
      this.ring.setTarget(boxQuad(ringBox));
      this.ringCorners.setTarget(ringCorners);
    }
    this.isRingShown = ringBox !== undefined;
  }

  /** Moves the elements of the tour to the current values of the springs. */
  private render() {
    const backdrop = this.backdropRef?.nativeElement;
    const ring = this.ringRef?.nativeElement;
    const card = this.cardRef?.nativeElement;
    const surface = this.surfaceRef?.nativeElement;
    if (!backdrop || !ring || !card || !surface) {
      return;
    }
    const hole = quadBox(this.spotlight.value);
    const path = backdropPath(
      viewportSize(),
      hole,
      this.spotlightCorners.value,
    );
    backdrop.style.clipPath = `path(evenodd, '${path}')`;

    const ringBox = quadBox(this.ring.value);
    ring.classList.toggle('visible', this.isRingShown && !this.isClosing());
    ring.style.transform = `translate3d(${ringBox.x}px, ${ringBox.y}px, 0)`;
    ring.style.width = `${Math.max(0, ringBox.w)}px`;
    ring.style.height = `${Math.max(0, ringBox.h)}px`;
    ring.style.borderRadius = this.ringCorners.value
      .map((radius) => `${Math.max(0, radius)}px`)
      .join(' ');

    const cardBox = quadBox(this.card.value);
    card.style.transform = `translate3d(${cardBox.x}px, ${cardBox.y}px, 0)`;
    surface.style.width = `${Math.max(0, cardBox.w)}px`;
    surface.style.height = `${Math.max(0, cardBox.h)}px`;
    card.setAttribute('data-side', this.side);
    // The arrow points at the ring, and shows once the card has landed.
    const anchor = this.isRingShown ? ringBox : hole;
    const isVertical = this.side === 'right' || this.side === 'left';
    const arrowOffset = isVertical
      ? clamp(
          anchor.y + anchor.h / 2 - cardBox.y,
          ARROW_INSET,
          cardBox.h - ARROW_INSET,
        )
      : clamp(
          anchor.x + anchor.w / 2 - cardBox.x,
          ARROW_INSET,
          cardBox.w - ARROW_INSET,
        );
    card.style.setProperty('--arrow-offset', `${arrowOffset}px`);
    card.classList.toggle('landed', this.card.isAtRest);
    this.renderHotspots(this.isClosing() ? undefined : this.demoScript);
  }

  /** Builds the demo of `step` from the elements of the viewer, if any. */
  private buildDemo(step: TourStep): DemoScript | undefined {
    if (!hasDemo(step.id)) {
      return undefined;
    }
    return buildDemoScript(step.id, this.buildDemoContext(step));
  }

  /** Measures what the demos need to know about the viewer. */
  private buildDemoContext(step: TourStep): DemoContext {
    const canvas = this.findBoxes(CANVAS_SELECTOR)[0];
    const mapping = this.timelineMapping();
    const panel = this.findBoxes(PANEL_SELECTOR)[0];
    const content = this.findBoxes(CONTENT_SELECTOR)[0];
    let block: DemoContext['block'] = undefined;
    for (const sighting of [this.unitSighting, this.lastSighting]) {
      const point =
        sighting && canvas
          ? locateSighting(sighting, canvas, mapping)
          : undefined;
      if (sighting && canvas && point && point.y >= canvas.y + RULER_HEIGHT) {
        block = {point, lane: sighting.lane};
        break;
      }
    }
    const isDetails = step.visual === 'panel' || step.visual === 'region';
    return {
      find: (selector) => this.findBoxes(selector)[0],
      findAll: (selector) => this.findBoxes(selector),
      viewport: viewportSize(),
      timeline: canvas ? timelineArea(canvas, mapping?.track) : undefined,
      block,
      bundlesPerPixel: mapping
        ? (mapping.end - mapping.start) / Math.max(1, mapping.track.w)
        : undefined,
      isPanelOpen: panel !== undefined,
      drawer: !panel && content ? drawerPreview(content) : undefined,
      metrics: isDetails
        ? this.panelMetrics().map(({selector, caption}) => ({
            selector,
            caption,
          }))
        : undefined,
    };
  }

  /**
   * Plays the demo of the step while the user reads, and steps aside while
   * they try things. Returns whether it needs another frame.
   */
  private updateDemo(now: number): boolean {
    const step = this.step();
    const script = this.demoScript;
    if (
      !step ||
      !script ||
      this.isClosing() ||
      this.isTaskDone() ||
      loopDuration(script) <= 0
    ) {
      this.stopDemo();
      return false;
    }
    if (this.reducedMotion) {
      // A still of the demo shows where to act, without motion.
      this.renderDemo(
        this.isStillDemoDismissed ? undefined : staticDemoPose(script),
      );
      return false;
    }
    const readyAt = Math.max(
      this.stepEnteredAt + DEMO_START_DELAY_MS,
      this.lastActivityAt + DEMO_IDLE_MS,
    );
    if (now < readyAt) {
      if (this.demoStartedAt) {
        this.stopDemo();
      }
      this.wakeDemo(readyAt);
      return false;
    }
    if (!this.demoStartedAt) {
      this.demoStartedAt = now;
    }
    const loop = loopDuration(script);
    let elapsed = now - this.demoStartedAt;
    if (elapsed >= loop) {
      // Each loop follows the viewer as it is now.
      this.demoStartedAt += Math.floor(elapsed / loop) * loop;
      elapsed = now - this.demoStartedAt;
      this.isDemoStale = true;
      this.kick();
    }
    this.renderDemo(sampleDemo(script, elapsed));
    return true;
  }

  /** Checks back on the demo at `time`, unless a check comes sooner. */
  private wakeDemo(time: number) {
    if (this.demoWakeTimer !== undefined && this.demoWakeAt <= time) {
      return;
    }
    clearTimeout(this.demoWakeTimer);
    this.demoWakeAt = time;
    this.demoWakeTimer = this.ngZone.runOutsideAngular(() =>
      setTimeout(
        () => {
          this.demoWakeTimer = undefined;
          this.requestFrame();
        },
        Math.max(0, time - Date.now()),
      ),
    );
  }

  private stopDemo() {
    this.demoStartedAt = 0;
    this.renderDemo(undefined);
  }

  private getDemoElements(): DemoElements | undefined {
    if (this.demoElements) {
      return this.demoElements;
    }
    const layer = this.demoRef?.nativeElement;
    const hotspots = this.hotspotsRef?.nativeElement;
    const find = (selector: string) =>
      layer?.querySelector<HTMLElement>(selector) ?? undefined;
    const pointer = find('.demo-pointer');
    const caption = find('.demo-caption');
    const captionText = find('.demo-caption-text');
    const keys = find('.demo-keys');
    const ripple = find('.demo-ripple');
    const beacon = find('.demo-beacon');
    const ghost = find('.demo-ghost');
    const ghostLabelElement = find('.demo-ghost-label');
    if (
      !layer ||
      !hotspots ||
      !pointer ||
      !caption ||
      !captionText ||
      !keys ||
      !ripple ||
      !beacon ||
      !ghost ||
      !ghostLabelElement
    ) {
      return undefined;
    }
    this.demoElements = {
      layer,
      pointer,
      caption,
      captionText,
      keys,
      ripple,
      beacon,
      ghost,
      ghostLabel: ghostLabelElement,
      hotspots: Array.from(hotspots.children).filter(
        (child): child is HTMLElement => child instanceof HTMLElement,
      ),
    };
    return this.demoElements;
  }

  /** Moves the demo to `pose`, or hides it. */
  private renderDemo(pose: DemoPose | undefined) {
    const elements = this.getDemoElements();
    if (!elements) {
      return;
    }
    const isShown = pose !== undefined && pose.opacity > 0;
    elements.layer.classList.toggle('active', isShown);
    if (!pose || !isShown) {
      // The layer fades out with the demo where it was.
      return;
    }
    const {pointer, caption, captionText, keys, ripple, beacon, ghost} =
      elements;
    const {x, y} = pose.pointer;
    pointer.style.transform = translateTo(pose.pointer);
    pointer.style.opacity = String(pose.showPointer ? pose.opacity : 0);
    pointer.classList.toggle('pressed', pose.pressed);

    const captionKey = `${pose.keys.join(' ')}|${pose.caption}`;
    if (captionKey !== this.demoCaptionKey) {
      this.demoCaptionKey = captionKey;
      setText(captionText, pose.caption);
      captionText.hidden = pose.caption === '';
      Array.from(keys.children).forEach((key, i) => {
        if (key instanceof HTMLElement) {
          setText(key, pose.keys[i] ?? '');
          key.hidden = i >= pose.keys.length;
        }
      });
      keys.hidden = pose.keys.length === 0;
      this.demoCaptionSize = {w: caption.offsetWidth, h: caption.offsetHeight};
    }
    keys.classList.toggle('down', pose.keysDown);
    const hasCaption = pose.caption !== '' || pose.keys.length > 0;
    caption.style.opacity = hasCaption ? String(pose.opacity) : '0';
    // The caption follows the pointer, and flips to stay in the viewport and
    // off the card.
    const viewport = viewportSize();
    const size = this.demoCaptionSize;
    const isFlippedX = x + CAPTION_OFFSET.x + size.w > viewport.w - 8;
    const isFlippedY = y + CAPTION_OFFSET.y + size.h > viewport.h - 8;
    const captionX = isFlippedX
      ? x - CAPTION_OFFSET.x - size.w
      : x + CAPTION_OFFSET.x;
    const below = y + CAPTION_OFFSET.y;
    const above = y - CAPTION_OFFSET.y - size.h;
    let captionY = isFlippedY ? above : below;
    const cardBox = quadBox(this.card.value);
    const otherY = isFlippedY ? below : above;
    if (
      overlaps({x: captionX, y: captionY, ...size}, cardBox) &&
      otherY >= 8 &&
      otherY + size.h <= viewport.h - 8 &&
      !overlaps({x: captionX, y: otherY, ...size}, cardBox)
    ) {
      captionY = otherY;
    }
    caption.style.transform = translateTo({x: captionX, y: captionY});

    if (pose.ripple) {
      const {progress} = pose.ripple;
      ripple.style.transform =
        `translate3d(${pose.ripple.x}px, ${pose.ripple.y}px, 0) ` +
        `scale(${0.25 + 1.35 * progress})`;
      ripple.style.opacity = String((1 - progress) * pose.opacity);
    } else {
      ripple.style.opacity = '0';
    }

    beacon.classList.toggle('shown', pose.beacon !== undefined);
    if (pose.beacon) {
      beacon.style.transform = translateTo(pose.beacon);
    }

    const ghostBox = pose.ghost;
    if (ghostBox && pose.ghostOpacity > 0) {
      ghost.style.transform = translateTo(ghostBox);
      ghost.style.width = `${Math.max(0, ghostBox.w)}px`;
      ghost.style.height = `${Math.max(0, ghostBox.h)}px`;
      ghost.style.opacity = String(pose.ghostOpacity);
      const label = ghostLabel(ghostBox);
      setText(elements.ghostLabel, label);
      elements.ghostLabel.hidden = label === '';
    } else {
      ghost.style.opacity = '0';
    }
  }

  /** Shows the numbered markers of `script` on the viewer, or hides them. */
  private renderHotspots(script: DemoScript | undefined) {
    const elements = this.getDemoElements();
    if (!elements) {
      return;
    }
    const points = script?.hotspots ?? [];
    elements.hotspots.forEach((hotspot, i) => {
      const point = points[i];
      hotspot.classList.toggle('shown', point !== undefined);
      if (point) {
        hotspot.style.transform = translateTo(point);
      }
    });
  }
}
