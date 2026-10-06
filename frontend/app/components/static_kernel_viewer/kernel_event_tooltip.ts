import {
  AfterViewInit,
  ChangeDetectionStrategy,
  ChangeDetectorRef,
  Component,
  ElementRef,
  inject,
  NgZone,
  OnDestroy,
} from '@angular/core';
import {EVENT_HOVERED_EVENT_NAME} from 'org_xprof/frontend/app/components/trace_viewer_container/trace_viewer_container';

/** A functional unit of a TPU core, which has its own lane in the timeline. */
export interface FunctionalUnit {
  /** Name of the unit and of its lane, e.g. `MXU`. */
  readonly label: string;
  /** What the unit does. */
  readonly description: string;
  /** Color of the unit's chip. */
  readonly color: string;
}

/** A cell in the VLIW bundle strip for a functional unit lane. */
export interface BundleLaneCell {
  readonly unit: FunctionalUnit;
  readonly count: number;
  readonly active: boolean;
}

/** A unit's static utilization bar at the hovered bundle or across a region. */
export interface UtilizationBar {
  readonly unit: FunctionalUnit;
  /** Utilization percentage from 0 to 100. */
  readonly percent: number;
  readonly formattedPercent: string;
  /** True for the busiest active unit at this bundle or in this region. */
  readonly highlighted: boolean;
}

/** Share of instructions belonging to a functional unit inside a region. */
export interface InstructionMixItem {
  readonly unit: FunctionalUnit;
  readonly count: number;
  /** Share of total instructions in the region, from 0 to 100. */
  readonly percent: number;
  readonly formattedPercent: string;
}

/** Position of a bundle or range within the static schedule. */
export interface SchedulePosition {
  /** Position percentage from 0 to 100. */
  readonly percent: number;
  readonly formattedPercent: string;
  readonly totalBundles: string;
}

/** Summary of mean unit utilization across a region. */
export interface RegionUtilizationSummary {
  readonly items: readonly UtilizationBar[];
  /** Label of the likely bottleneck unit when one unit is dominant. */
  readonly boundUnit?: string;
}

/** Single-value gauge shown when hovering a counter or process track. */
export interface CounterGauge {
  readonly label: string;
  readonly formattedValue: string;
  readonly percent: number;
  readonly color: string;
}

/** Register and hardware property badges describing an instruction's output. */
export interface InstructionOutput {
  readonly rowLabel: string;
  readonly register?: string;
  readonly registerKind?: 'vreg' | 'sreg' | 'preg' | 'vmreg' | 'other';
  readonly memorySpace?: string;
  readonly metadata?: string;
  readonly throughput?: string;
}

/** An input operand of an instruction, with its 0-based operand index. */
export interface OperandItem {
  readonly index: number;
  readonly text: string;
}

/**
 * An event or track point that the pointer hovers in the timeline. Kernel
 * traces use bundle numbers as timestamps.
 */
export interface HoveredEvent {
  /** Event name, e.g. an instruction mnemonic, region name, or track name. */
  readonly name: string;
  /** First bundle of the event. */
  readonly start: number;
  /** Number of bundles of the event. */
  readonly duration: number;
  /** Name of the lane of the event, e.g. `MXU` or `Regions`, or ''. */
  readonly trackName: string;
  /** Track kind when hovering a non-flamechart track (`counter` or `process`). */
  readonly trackType?: 'counter' | 'process';
  /** Total number of bundles in the static kernel schedule. */
  readonly totalBundles?: number;
  /** Count of instructions per functional unit lane in the hovered bundle(s). */
  readonly bundleCounts?: Readonly<Record<string, number>>;
  /** Static utilization (0.0 to 1.0) per counter track in the hovered bundle(s). */
  readonly utilization?: Readonly<Record<string, number>>;
  /** Enclosing region hierarchy from outermost to innermost. */
  readonly regions?: readonly string[];
  /** Nesting depth (0-indexed) when hovering a region on the Regions lane. */
  readonly regionDepth?: number;
  /** Hovered sample value when hovering a counter or process utilization chart. */
  readonly counterValue?: number;
  /** Raw key-value arguments attached to the trace event. */
  readonly args?: Readonly<Record<string, string>>;
  /** Program-order ordinal of the instruction or region, e.g. 92. */
  readonly ordinal?: number;
  /** Result register produced by the instruction, e.g. `%s18` or `%v12`. */
  readonly produces?: string;
  /** Memory space accessed or produced, e.g. `vmem`, `smem`, `cmem`, `hbm`. */
  readonly memorySpace?: string;
  /** Throughput in cycles, e.g. 1. */
  readonly throughput?: number;
  /** Opcode-specific metadata summary, e.g. `bf16` or `Treat Copy As Opaque`. */
  readonly metadata?: string;
  /** Input operands (allocations, constants, or producer registers). */
  readonly operands?: readonly string[];
  /** Consumer instructions that read this instruction's output. */
  readonly consumers?: readonly string[];
  /** Vector register spills occurring in the hovered bundle or region. */
  readonly vectorSpills?: number;
  /** Vector register fills occurring in the hovered bundle or region. */
  readonly vectorFills?: number;
}

/** What the tooltip shows for a hovered event or track point. */
export interface KernelEventTooltipContent {
  readonly kind: 'instruction' | 'region' | 'counter' | 'process';
  readonly name: string;
  /** Whether the name is code, e.g. an instruction mnemonic. */
  readonly isCode: boolean;
  /** Functional unit that executes the event, if it is an instruction or unit counter. */
  readonly unit?: FunctionalUnit;
  /** Accent color of the top bar of the card. */
  readonly accentColor: string;
  /** `Bundle`, or `Bundles` if the event spans several bundles. */
  readonly bundlesLabel: string;
  /** Bundle or range of bundles of the event, e.g. `1,024–1,056`. */
  readonly bundles: string;
  /** Number of bundles of an event that spans several, e.g. `32 bundles`. */
  readonly length?: string;
  /** Hexadecimal bundle address, e.g. `<0x3039>`, matching compiler/Percale dumps. */
  readonly hexBundle?: string;
  /** Program-order ordinal badge, e.g. `Ord #92`. */
  readonly ordinalLabel?: string;
  /** Summary chips shown in the header subtitle (e.g. region length, % of schedule, depth). */
  readonly subtitleChips?: readonly string[];
  /** Details shown in muted text, e.g. what the functional unit does. */
  readonly description?: string;
  /** Output register and hardware property badges of an instruction. */
  readonly output?: InstructionOutput;
  /** Indexed input operands of an instruction. */
  readonly operands?: readonly OperandItem[];
  /** Consumer instructions of an instruction's result. */
  readonly consumers?: readonly string[];
  /** Formatted vector register spills/fills summary when present, e.g. `2 spills · 1 fill`. */
  readonly spillsLabel?: string;
  /** True when the hovered bundle or region has at least one vector spill. */
  readonly hasSpills?: boolean;
  /** Strip of all 8 core VLIW lanes showing active units and counts in this bundle. */
  readonly bundleStrip?: readonly BundleLaneCell[];
  /** Static utilization bars at the hovered bundle, sorted by utilization. */
  readonly utilizationBars?: readonly UtilizationBar[];
  /** Primary gauge when hovering a utilization counter or parent process track. */
  readonly counterGauge?: CounterGauge;
  /** Enclosing region breadcrumb path. */
  readonly regions?: readonly string[];
  /** Label for the region row (`Region` for bundle hover, `Inside` for region hover). */
  readonly regionRowLabel?: string;
  /** Position of the bundle in the full static schedule. */
  readonly schedulePosition?: SchedulePosition;
  /** Instruction mix breakdown across functional units inside a region. */
  readonly instructionMix?: readonly InstructionMixItem[];
  /** Mean unit utilization and bottleneck hint inside a region. */
  readonly meanUtilization?: RegionUtilizationSummary;
  /** Instruction density inside a region, e.g. `2.4 instr / bundle`. */
  readonly density?: string;
  /** Whether the card has rich body rows below the header. */
  readonly hasBodyRows: boolean;
  /** Whether to show the footer with click/shift+click interaction hints. */
  readonly showInteractiveHint: boolean;
}

/** Name of the lane of code regions, such as loops and branches. */
const REGIONS_TRACK_NAME = 'Regions';

/** Suffix of per-unit static utilization counter tracks. */
const UTILIZATION_SUFFIX = ' utilization';

/** Default accent color for regions and whole-kernel process tracks. */
export const DEFAULT_ACCENT_COLOR = '#1a73e8';

/** Functional units in the Static Kernel Viewer timeline. */
export const FUNCTIONAL_UNITS: readonly FunctionalUnit[] = [
  {label: 'MXU', description: 'Matrix multiply unit', color: '#7c4dff'},
  {
    label: 'XLU',
    description: 'Cross-lane unit — transpose / permute / reduce',
    color: '#f29900',
  },
  {
    label: 'SALU',
    description: 'Scalar ALU — control flow, addressing, DMA issue',
    color: '#1e8e3e',
  },
  {label: 'VPU', description: 'Vector ALU', color: '#1a73e8'},
  {
    label: 'EUP',
    description: 'Extended unary pipeline — exp, rcp, tanh',
    color: '#12a4af',
  },
  {label: 'VLD', description: 'Vector load — VMEM → vregs', color: '#00a3bf'},
  {label: 'VST', description: 'Vector store — vregs → VMEM', color: '#d01884'},
  {label: 'DMA', description: 'DMA engine — HBM ↔ VMEM', color: '#e8710a'},
  {
    label: 'Other',
    description: 'No dedicated functional unit',
    color: '#5f6368',
  },
];

/** The 8 primary VLIW hardware lanes shown in the bundle strip. */
const BUNDLE_STRIP_UNITS: readonly FunctionalUnit[] = FUNCTIONAL_UNITS.filter(
  (unit) => unit.label !== 'Other',
);

/** Functional units, keyed by the name of their lane. */
export const FUNCTIONAL_UNITS_BY_LANE: ReadonlyMap<string, FunctionalUnit> =
  new Map(FUNCTIONAL_UNITS.map((unit) => [unit.label, unit] as const));

/** Integer formatter for bundle indices and instruction counts. */
export const BUNDLE_FORMAT = new Intl.NumberFormat(undefined, {
  maximumFractionDigits: 0,
});

/** Single-decimal formatter for instructions-per-bundle schedule density. */
export const DENSITY_FORMAT = new Intl.NumberFormat(undefined, {
  minimumFractionDigits: 1,
  maximumFractionDigits: 1,
});

/** Distance between the pointer and the tooltip, in pixels. */
const POINTER_OFFSET = 14;

/** Minimum distance between the tooltip and the edges of its area. */
const EDGE_MARGIN = 8;

/** Extracts finite numeric entries from an untyped JS object record. */
export function readNumberRecord(
  value: unknown,
): Readonly<Record<string, number>> | undefined {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) {
    return undefined;
  }
  const result: Record<string, number> = {};
  let hasEntry = false;
  for (const [key, entry] of Object.entries(value as Record<string, unknown>)) {
    if (typeof entry === 'number' && Number.isFinite(entry)) {
      result[key] = entry;
      hasEntry = true;
    }
  }
  return hasEntry ? result : undefined;
}

function readStringRecord(
  value: unknown,
): Readonly<Record<string, string>> | undefined {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) {
    return undefined;
  }
  const result: Record<string, string> = {};
  let hasEntry = false;
  for (const [key, entry] of Object.entries(value as Record<string, unknown>)) {
    if (typeof entry === 'string' && entry.trim() !== '') {
      result[key] = entry.trim();
      hasEntry = true;
    } else if (typeof entry === 'number' && Number.isFinite(entry)) {
      result[key] = String(entry);
      hasEntry = true;
    } else if (typeof entry === 'boolean') {
      result[key] = String(entry);
      hasEntry = true;
    }
  }
  return hasEntry ? result : undefined;
}

function readStringArray(value: unknown): readonly string[] | undefined {
  if (!Array.isArray(value)) {
    return undefined;
  }
  const items = value
    .filter((item): item is string => typeof item === 'string')
    .map((item) => item.trim())
    .filter((item) => item !== '');
  return items.length > 0 ? items : undefined;
}

function readStringListField(
  direct: unknown,
  fromArgs: string | undefined,
): readonly string[] | undefined {
  const arrayItems = readStringArray(direct);
  if (arrayItems) {
    return arrayItems;
  }
  if (!fromArgs) {
    return undefined;
  }
  const parts = fromArgs
    .split(/\s*;\s*|\s*\|\s*|\r?\n/)
    .map((part) => part.trim())
    .filter((part) => part !== '');
  return parts.length > 0 ? parts : undefined;
}

function readNonEmptyString(...candidates: unknown[]): string | undefined {
  for (const candidate of candidates) {
    if (typeof candidate === 'string' && candidate.trim() !== '') {
      return candidate.trim();
    }
  }
  return undefined;
}

function readFiniteInteger(...candidates: unknown[]): number | undefined {
  for (const candidate of candidates) {
    if (typeof candidate === 'number' && Number.isFinite(candidate)) {
      return Math.round(candidate);
    }
    if (typeof candidate === 'string' && /^-?\d+$/.test(candidate.trim())) {
      const parsed = Number(candidate.trim());
      if (Number.isFinite(parsed)) {
        return parsed;
      }
    }
  }
  return undefined;
}

/** Classifies a hardware register identifier into its LLO register file kind. */
export function classifyRegister(
  register: string,
): NonNullable<InstructionOutput['registerKind']> {
  if (/^[%$]vm\d+/i.test(register)) {
    return 'vmreg';
  }
  if (/^[%$]v\d+/i.test(register)) {
    return 'vreg';
  }
  if (/^[%$]s\d+/i.test(register)) {
    return 'sreg';
  }
  if (/^[%$]p\d+/i.test(register)) {
    return 'preg';
  }
  return 'other';
}

function buildInstructionOutput(
  event: HoveredEvent,
): InstructionOutput | undefined {
  const register = event.produces;
  const memorySpace = event.memorySpace;
  const metadata = event.metadata;
  const throughput =
    event.throughput !== undefined && event.throughput > 0
      ? `${BUNDLE_FORMAT.format(event.throughput)} ${event.throughput === 1 ? 'cycle' : 'cycles'}`
      : undefined;
  if (!register && !memorySpace && !metadata && !throughput) {
    return undefined;
  }
  return {
    rowLabel: register ? 'Produces' : 'Properties',
    ...(register ? {register, registerKind: classifyRegister(register)} : {}),
    ...(memorySpace ? {memorySpace} : {}),
    ...(metadata ? {metadata} : {}),
    ...(throughput ? {throughput} : {}),
  };
}

function unitForCounterTrack(trackName: string): FunctionalUnit | undefined {
  const direct = FUNCTIONAL_UNITS_BY_LANE.get(trackName);
  if (direct) {
    return direct;
  }
  if (trackName.endsWith(UTILIZATION_SUFFIX)) {
    const label = trackName.slice(0, -UTILIZATION_SUFFIX.length).trim();
    return FUNCTIONAL_UNITS_BY_LANE.get(label);
  }
  return undefined;
}

function clampPercent(percent: number): number {
  return Math.max(0, Math.min(100, Math.round(percent)));
}

/** Builds the 8-lane VLIW bundle occupancy cells for a single bundle. */
export function buildBundleStrip(
  bundleCounts: Readonly<Record<string, number>> | undefined,
): readonly BundleLaneCell[] | undefined {
  if (!bundleCounts) {
    return undefined;
  }
  const cells: BundleLaneCell[] = BUNDLE_STRIP_UNITS.map((unit) => {
    const count = Math.max(0, Math.round(bundleCounts[unit.label] ?? 0));
    return {unit, count, active: count > 0};
  });
  const otherUnit = FUNCTIONAL_UNITS_BY_LANE.get('Other');
  const otherCount = Math.max(0, Math.round(bundleCounts['Other'] ?? 0));
  if (otherUnit && otherCount > 0) {
    cells.push({unit: otherUnit, count: otherCount, active: true});
  }
  return cells;
}

/** Builds sorted per-unit static utilization bars with the busiest unit highlighted. */
export function buildUtilizationBars(
  utilization: Readonly<Record<string, number>> | undefined,
): readonly UtilizationBar[] | undefined {
  if (!utilization) {
    return undefined;
  }
  const rawBars: Array<{unit: FunctionalUnit; percent: number}> = [];
  for (const [trackName, rawValue] of Object.entries(utilization)) {
    const unit = unitForCounterTrack(trackName);
    if (!unit) {
      continue;
    }
    const percent = clampPercent(rawValue * 100);
    rawBars.push({unit, percent});
  }
  if (rawBars.length === 0) {
    return undefined;
  }
  rawBars.sort((a, b) => b.percent - a.percent);
  const maxPercent = rawBars[0].percent;
  return rawBars.map((bar, index) => ({
    unit: bar.unit,
    percent: bar.percent,
    formattedPercent: `${bar.percent}%`,
    highlighted: index === 0 && maxPercent > 0,
  }));
}

/** Computes the instruction mix breakdown and total instruction count across lanes. */
export function buildInstructionMix(
  bundleCounts: Readonly<Record<string, number>> | undefined,
): {mix?: readonly InstructionMixItem[]; totalInstructions: number} {
  if (!bundleCounts) {
    return {totalInstructions: 0};
  }
  let totalInstructions = 0;
  const entries: Array<{unit: FunctionalUnit; count: number}> = [];
  for (const unit of FUNCTIONAL_UNITS) {
    const count = Math.max(0, Math.round(bundleCounts[unit.label] ?? 0));
    if (count > 0) {
      totalInstructions += count;
      entries.push({unit, count});
    }
  }
  if (totalInstructions === 0) {
    return {totalInstructions: 0};
  }
  entries.sort((a, b) => b.count - a.count);
  const mix: InstructionMixItem[] = entries.map(({unit, count}) => {
    const percent = clampPercent((count / totalInstructions) * 100);
    return {
      unit,
      count,
      percent,
      formattedPercent: `${percent}%`,
    };
  });
  return {mix, totalInstructions};
}

/**
 * Reads the detail of an `eventhovered` event of Trace Viewer v2. Returns
 * undefined if no event is hovered, or if the detail is malformed.
 */
export function readHoveredEvent(detail: unknown): HoveredEvent | undefined {
  if (typeof detail !== 'object' || detail === null) {
    return undefined;
  }
  // The detail comes from WASM, so its fields are read by their exact names.
  const fields = detail as Record<string, unknown>;
  const name = fields['name'];
  const start = fields['startUs'];
  const duration = fields['durationUs'];
  const trackName = fields['trackName'];
  const trackTypeField = fields['trackType'];
  const trackType =
    trackTypeField === 'counter' || trackTypeField === 'process'
      ? trackTypeField
      : undefined;
  if (
    (fields['eventIndex'] === -1 && trackType === undefined) ||
    typeof name !== 'string' ||
    typeof start !== 'number' ||
    typeof duration !== 'number'
  ) {
    return undefined;
  }
  const args = readStringRecord(fields['args']);
  const ordinal = readFiniteInteger(fields['ordinal'], args?.['ordinal']);
  const produces = readNonEmptyString(
    fields['produces'],
    args?.['produces'],
    args?.['result_register'],
  );
  const memorySpace = readNonEmptyString(
    fields['memorySpace'],
    fields['memory_space'],
    args?.['memory_space'],
    args?.['memorySpace'],
  );
  const throughput = readFiniteInteger(
    fields['throughput'],
    args?.['throughput'],
  );
  const metadata = readNonEmptyString(
    fields['metadata'],
    args?.['metadata'],
    args?.['data_format'],
    args?.['pseudo_kind'],
    args?.['treat_copy_as_opaque'] === 'true'
      ? 'Treat Copy As Opaque'
      : undefined,
  );
  const operands = readStringListField(fields['operands'], args?.['operands']);
  const consumers = readStringListField(
    fields['consumers'],
    args?.['consumers'],
  );
  const vectorSpills = readFiniteInteger(
    fields['vectorSpills'],
    fields['vector_spills'],
    args?.['vector_spills'],
    args?.['vectorSpills'],
  );
  const vectorFills = readFiniteInteger(
    fields['vectorFills'],
    fields['vector_fills'],
    args?.['vector_fills'],
    args?.['vectorFills'],
  );

  const hovered: HoveredEvent = {
    name,
    start,
    duration,
    trackName: typeof trackName === 'string' ? trackName : '',
    ...(trackType !== undefined ? {trackType} : {}),
    ...(typeof fields['totalBundles'] === 'number' && fields['totalBundles'] > 0
      ? {totalBundles: fields['totalBundles']}
      : {}),
    ...(typeof fields['regionDepth'] === 'number' && fields['regionDepth'] >= 0
      ? {regionDepth: Math.round(fields['regionDepth'])}
      : {}),
    ...(typeof fields['counterValue'] === 'number' &&
    Number.isFinite(fields['counterValue'])
      ? {counterValue: fields['counterValue']}
      : {}),
    ...(args ? {args} : {}),
    ...(ordinal !== undefined && ordinal >= 0 ? {ordinal} : {}),
    ...(produces ? {produces} : {}),
    ...(memorySpace ? {memorySpace} : {}),
    ...(throughput !== undefined && throughput > 0 ? {throughput} : {}),
    ...(metadata ? {metadata} : {}),
    ...(operands ? {operands} : {}),
    ...(consumers ? {consumers} : {}),
    ...(vectorSpills !== undefined && vectorSpills > 0 ? {vectorSpills} : {}),
    ...(vectorFills !== undefined && vectorFills > 0 ? {vectorFills} : {}),
  };
  const bundleCounts = readNumberRecord(fields['bundleCounts']);
  const utilization = readNumberRecord(fields['utilization']);
  const regions = readStringArray(fields['regions']);
  return {
    ...hovered,
    ...(bundleCounts ? {bundleCounts} : {}),
    ...(utilization ? {utilization} : {}),
    ...(regions ? {regions} : {}),
  };
}

function formatSpillsLabel(
  spills: number | undefined,
  fills: number | undefined,
): string | undefined {
  const parts: string[] = [];
  if (spills !== undefined && spills > 0) {
    parts.push(
      `${BUNDLE_FORMAT.format(spills)} ${spills === 1 ? 'spill' : 'spills'}`,
    );
  }
  if (fills !== undefined && fills > 0) {
    parts.push(
      `${BUNDLE_FORMAT.format(fills)} ${fills === 1 ? 'fill' : 'fills'}`,
    );
  }
  return parts.length > 0 ? parts.join(' · ') : undefined;
}

/** Returns what the tooltip shows for `event`. */
export function buildKernelEventTooltip(
  event: HoveredEvent,
): KernelEventTooltipContent {
  const isRegion = event.trackName === REGIONS_TRACK_NAME;
  const isCounter = event.trackType === 'counter';
  const isProcess = event.trackType === 'process';
  const kind: KernelEventTooltipContent['kind'] = isCounter
    ? 'counter'
    : isProcess
      ? 'process'
      : isRegion
        ? 'region'
        : 'instruction';

  const unit = isCounter
    ? unitForCounterTrack(event.trackName)
    : FUNCTIONAL_UNITS_BY_LANE.get(event.trackName);
  const accentColor = unit?.color ?? DEFAULT_ACCENT_COLOR;

  let description = unit?.description;
  if (isCounter) {
    description = unit
      ? `${unit.description} — static utilization`
      : 'Static unit utilization';
  } else if (isProcess) {
    description = 'Bundle activity across all functional unit lanes';
  } else if (!unit && !isRegion && event.trackName) {
    // Names the lane of an event that is neither an instruction nor a region.
    description = event.trackName;
  }

  const first = BUNDLE_FORMAT.format(event.start);
  const isSingleBundle = event.duration <= 1;
  const bundlesLabel = isSingleBundle ? 'Bundle' : 'Bundles';
  // Like on the timeline axis, a range ends where its last bundle ends.
  const bundles = isSingleBundle
    ? first
    : `${first}–${BUNDLE_FORMAT.format(event.start + event.duration)}`;
  const length = isSingleBundle
    ? undefined
    : `${BUNDLE_FORMAT.format(event.duration)} bundles`;

  const hasRichScheduleOrLloContext =
    event.totalBundles !== undefined ||
    event.ordinal !== undefined ||
    event.args !== undefined ||
    event.produces !== undefined ||
    event.memorySpace !== undefined ||
    event.operands !== undefined ||
    event.consumers !== undefined ||
    event.vectorSpills !== undefined ||
    event.vectorFills !== undefined;
  const hexBundle =
    !isRegion &&
    isSingleBundle &&
    event.start >= 0 &&
    hasRichScheduleOrLloContext
      ? `<0x${Math.floor(event.start).toString(16)}>`
      : undefined;
  const ordinalLabel =
    !isRegion && event.ordinal !== undefined && event.ordinal >= 0
      ? `Ord #${BUNDLE_FORMAT.format(event.ordinal)}`
      : undefined;

  const subtitleChips: string[] = [];
  if (
    isRegion &&
    (event.totalBundles !== undefined || event.regionDepth !== undefined)
  ) {
    if (length) {
      subtitleChips.push(length);
    }
    if (event.totalBundles !== undefined && event.totalBundles > 0) {
      const share = clampPercent((event.duration / event.totalBundles) * 100);
      subtitleChips.push(`${share}% of static schedule`);
    }
    if (event.regionDepth !== undefined) {
      subtitleChips.push(`depth ${event.regionDepth}`);
    }
  }

  const output =
    kind === 'instruction' ? buildInstructionOutput(event) : undefined;
  const operands: readonly OperandItem[] | undefined =
    kind === 'instruction' && event.operands && event.operands.length > 0
      ? event.operands.map((text, index) => ({index, text}))
      : undefined;
  const consumers: readonly string[] | undefined =
    kind === 'instruction' && event.consumers && event.consumers.length > 0
      ? event.consumers
      : undefined;
  const spillsLabel = formatSpillsLabel(event.vectorSpills, event.vectorFills);
  const hasSpills = (event.vectorSpills ?? 0) > 0;

  const bundleStrip = !isRegion
    ? buildBundleStrip(event.bundleCounts)
    : undefined;
  const utilizationBars = buildUtilizationBars(event.utilization);

  let counterGauge: CounterGauge | undefined = undefined;
  if (isCounter && event.counterValue !== undefined) {
    const percent = clampPercent(event.counterValue * 100);
    counterGauge = {
      label: 'Current',
      formattedValue: `${percent}%`,
      percent,
      color: accentColor,
    };
  } else if (isProcess) {
    const activeOps = bundleStrip
      ? bundleStrip.reduce((sum, cell) => sum + cell.count, 0)
      : event.counterValue !== undefined
        ? Math.round(event.counterValue)
        : undefined;
    if (activeOps !== undefined) {
      const activeLanes = bundleStrip
        ? bundleStrip.filter((cell) => cell.active).length
        : 0;
      const percent = bundleStrip
        ? clampPercent((activeLanes / BUNDLE_STRIP_UNITS.length) * 100)
        : clampPercent((activeOps / BUNDLE_STRIP_UNITS.length) * 100);
      counterGauge = {
        label: 'Active ops',
        formattedValue:
          activeLanes > 0
            ? `${activeOps} ${activeOps === 1 ? 'op' : 'ops'} · ${activeLanes}/${BUNDLE_STRIP_UNITS.length} lanes`
            : `${activeOps} ${activeOps === 1 ? 'op' : 'ops'}`,
        percent,
        color: accentColor,
      };
    }
  }

  let schedulePosition: SchedulePosition | undefined = undefined;
  if (!isRegion && event.totalBundles !== undefined && event.totalBundles > 0) {
    const percent = clampPercent((event.start / event.totalBundles) * 100);
    schedulePosition = {
      percent,
      formattedPercent: `${percent}%`,
      totalBundles: BUNDLE_FORMAT.format(event.totalBundles),
    };
  }

  let instructionMix: readonly InstructionMixItem[] | undefined = undefined;
  let meanUtilization: RegionUtilizationSummary | undefined = undefined;
  let density: string | undefined = undefined;
  if (isRegion) {
    const mixResult = buildInstructionMix(event.bundleCounts);
    instructionMix = mixResult.mix;
    if (mixResult.totalInstructions > 0 && event.duration > 0) {
      density = `${DENSITY_FORMAT.format(mixResult.totalInstructions / event.duration)} instr / bundle`;
    }
    if (utilizationBars && utilizationBars.length > 0) {
      const top = utilizationBars[0];
      meanUtilization = {
        items: utilizationBars.slice(0, 3),
        ...(top.percent >= 20 ? {boundUnit: top.unit.label} : {}),
      };
    }
  }

  const regions =
    event.regions && event.regions.length > 0 ? event.regions : undefined;
  const regionRowLabel = regions ? (isRegion ? 'Inside' : 'Region') : undefined;

  const hasBodyRows = Boolean(
    output ||
      operands ||
      consumers ||
      spillsLabel ||
      counterGauge ||
      bundleStrip ||
      (!isRegion && utilizationBars) ||
      instructionMix ||
      meanUtilization ||
      density ||
      regions ||
      schedulePosition,
  );

  return {
    kind,
    name: event.name,
    isCode: kind === 'instruction' && unit !== undefined,
    unit,
    accentColor,
    bundlesLabel,
    bundles,
    ...(length ? {length} : {}),
    ...(hexBundle ? {hexBundle} : {}),
    ...(ordinalLabel ? {ordinalLabel} : {}),
    ...(subtitleChips.length > 0 ? {subtitleChips} : {}),
    ...(description ? {description} : {}),
    ...(output ? {output} : {}),
    ...(operands ? {operands} : {}),
    ...(consumers ? {consumers} : {}),
    ...(spillsLabel ? {spillsLabel, hasSpills} : {}),
    ...(bundleStrip ? {bundleStrip} : {}),
    ...(!isRegion && utilizationBars ? {utilizationBars} : {}),
    ...(counterGauge ? {counterGauge} : {}),
    ...(regions ? {regions, regionRowLabel} : {}),
    ...(schedulePosition ? {schedulePosition} : {}),
    ...(instructionMix ? {instructionMix} : {}),
    ...(meanUtilization ? {meanUtilization} : {}),
    ...(density ? {density} : {}),
    hasBodyRows,
    showInteractiveHint:
      hasBodyRows && (kind === 'instruction' || kind === 'region'),
  };
}

/**
 * Returns the offset of the tooltip along one axis of its area. The tooltip
 * follows the pointer at `pointer`, flips to the other side of the pointer
 * where it would overflow the area, and keeps a margin from the edges of the
 * area if it fits.
 */
export function tooltipOffset(
  pointer: number,
  size: number,
  areaSize: number,
): number {
  let offset = pointer + POINTER_OFFSET;
  if (offset + size > areaSize - EDGE_MARGIN) {
    offset = pointer - POINTER_OFFSET - size;
  }
  return Math.max(EDGE_MARGIN, Math.min(offset, areaSize - EDGE_MARGIN - size));
}

/**
 * Tooltip of the event that the pointer hovers in the timeline of a kernel.
 *
 * Place it in the `canvasOverlay` slot of a Trace Viewer v2 container, and turn
 * off the built-in tooltip of Trace Viewer v2. It shows the hovered events that
 * WASM reports, next to the pointer while the pointer is over the timeline.
 */
@Component({
  standalone: true,
  selector: 'kernel-event-tooltip',
  templateUrl: './kernel_event_tooltip.ng.html',
  styleUrls: ['./kernel_event_tooltip.scss'],
  changeDetection: ChangeDetectionStrategy.OnPush,
  host: {'aria-hidden': 'true'},
})
export class KernelEventTooltip implements AfterViewInit, OnDestroy {
  private readonly host =
    inject<ElementRef<HTMLElement>>(ElementRef).nativeElement;
  private readonly changeDetectorRef = inject(ChangeDetectorRef);
  private readonly ngZone = inject(NgZone);

  /** What the tooltip shows, or undefined while no event is hovered. */
  content: KernelEventTooltipContent | undefined = undefined;

  // The element that the tooltip is positioned in, i.e. its parent.
  private area: HTMLElement | undefined = undefined;
  private resizeObserver: ResizeObserver | undefined = undefined;
  // Position and size of the padding box of the area, in viewport coordinates.
  // Reading them forces a layout, so they are cached until the area may have
  // moved or resized.
  private isAreaStale = true;
  private areaLeft = 0;
  private areaTop = 0;
  private areaWidth = 0;
  private areaHeight = 0;
  // Size of the tooltip, measured when its content changes.
  private width = 0;
  private height = 0;
  // Last position of the pointer, in viewport coordinates.
  private clientX = 0;
  private clientY = 0;
  private isPointerOverCanvas = false;
  private isPointerDown = false;
  private isShown = false;
  private left = Number.NaN;
  private top = Number.NaN;
  private placedBeforeX = false;
  private placedBeforeY = false;
  private cardAnimation: Animation | undefined = undefined;

  ngAfterViewInit() {
    const area = this.host.parentElement;
    if (!area) {
      return;
    }
    this.area = area;
    // Pointer events are frequent, and only move or hide the tooltip, so they
    // are handled without triggering change detection.
    this.ngZone.runOutsideAngular(() => {
      window.addEventListener(EVENT_HOVERED_EVENT_NAME, this.onEventHovered);
      window.addEventListener('scroll', this.onScroll, {
        capture: true,
        passive: true,
      });
      area.addEventListener('mousemove', this.onMouseMove);
      area.addEventListener('mousedown', this.onMouseButton);
      area.addEventListener('mouseup', this.onMouseButton);
      area.addEventListener('mouseleave', this.onMouseLeave);
      this.resizeObserver = new ResizeObserver(this.onAreaResize);
      this.resizeObserver.observe(area);
    });
  }

  ngOnDestroy() {
    window.removeEventListener(EVENT_HOVERED_EVENT_NAME, this.onEventHovered);
    window.removeEventListener('scroll', this.onScroll, {capture: true});
    this.area?.removeEventListener('mousemove', this.onMouseMove);
    this.area?.removeEventListener('mousedown', this.onMouseButton);
    this.area?.removeEventListener('mouseup', this.onMouseButton);
    this.area?.removeEventListener('mouseleave', this.onMouseLeave);
    this.resizeObserver?.disconnect();
    this.cardAnimation?.cancel();
  }

  private readonly onEventHovered = (event: Event) => {
    const wasShown = this.isShown;
    const prevKind = this.content?.kind;
    const hovered =
      event instanceof CustomEvent ? readHoveredEvent(event.detail) : undefined;
    this.content = hovered ? buildKernelEventTooltip(hovered) : undefined;
    this.host.style.setProperty(
      '--accent-color',
      this.content?.accentColor ?? DEFAULT_ACCENT_COLOR,
    );
    this.changeDetectorRef.detectChanges();
    if (this.content) {
      this.width = this.host.offsetWidth;
      this.height = this.host.offsetHeight;
    }
    this.update();
    if (
      this.isShown &&
      (!wasShown || (prevKind && prevKind !== this.content?.kind))
    ) {
      this.animateCard(
        [
          {opacity: 0, transform: 'translate3d(0, 4px, 0) scale(0.97)'},
          {opacity: 1, transform: 'translate3d(0, 0, 0) scale(1)'},
        ],
        150,
      );
    }
  };

  private readonly onMouseMove = (event: MouseEvent) => {
    this.clientX = event.clientX;
    this.clientY = event.clientY;
    // WASM only tracks the pointer over the canvas, not over other overlays
    // in the area, e.g. the timeline player.
    this.isPointerOverCanvas = event.target instanceof HTMLCanvasElement;
    this.isPointerDown = event.buttons !== 0;
    this.update();
  };

  // Hides the tooltip while the user clicks or drags the timeline.
  private readonly onMouseButton = (event: MouseEvent) => {
    this.isPointerDown = event.buttons !== 0;
    this.update();
  };

  // Keeps the content, since WASM reports no hover change if the pointer
  // comes back to the same event.
  private readonly onMouseLeave = () => {
    this.isPointerOverCanvas = false;
    this.isAreaStale = true;
    this.update();
  };

  private readonly onScroll = () => {
    this.isAreaStale = true;
  };

  private readonly onAreaResize = () => {
    this.isAreaStale = true;
    this.update();
  };

  private update() {
    const wasShown = this.isShown;
    const isShown =
      this.content !== undefined &&
      this.isPointerOverCanvas &&
      !this.isPointerDown;
    if (isShown) {
      this.refreshArea();
      const pointerX = this.clientX - this.areaLeft;
      const pointerY = this.clientY - this.areaTop;
      const left = Math.round(
        tooltipOffset(pointerX, this.width, this.areaWidth),
      );
      const top = Math.round(
        tooltipOffset(pointerY, this.height, this.areaHeight),
      );
      const beforeX = left < pointerX;
      const beforeY = top < pointerY;
      if (left !== this.left || top !== this.top) {
        const prevLeft = this.left;
        const prevTop = this.top;
        const flipped =
          wasShown &&
          Number.isFinite(prevLeft) &&
          Number.isFinite(prevTop) &&
          (beforeX !== this.placedBeforeX || beforeY !== this.placedBeforeY);
        this.left = left;
        this.top = top;
        this.placedBeforeX = beforeX;
        this.placedBeforeY = beforeY;
        this.host.style.transform = `translate3d(${left}px, ${top}px, 0)`;
        if (flipped) {
          const dx = Math.max(-48, Math.min(48, prevLeft - left));
          const dy = Math.max(-48, Math.min(48, prevTop - top));
          this.animateCard(
            [
              {transform: `translate3d(${dx}px, ${dy}px, 0)`},
              {transform: 'translate3d(0, 0, 0)'},
            ],
            160,
          );
        }
      }
    }
    if (isShown !== this.isShown) {
      this.isShown = isShown;
      this.host.classList.toggle('visible', isShown);
    }
  }

  private animateCard(keyframes: Keyframe[], duration: number) {
    if (
      typeof window !== 'undefined' &&
      window.matchMedia?.('(prefers-reduced-motion: reduce)').matches
    ) {
      return;
    }
    const card = this.host.querySelector<HTMLElement>('.tooltip-card');
    if (!card || typeof card.animate !== 'function') {
      return;
    }
    this.cardAnimation?.cancel();
    this.cardAnimation = card.animate(keyframes, {
      duration,
      easing: 'cubic-bezier(0.2, 0, 0, 1)',
    });
  }

  private refreshArea() {
    if (!this.isAreaStale || !this.area) {
      return;
    }
    const rect = this.area.getBoundingClientRect();
    this.areaLeft = rect.left + this.area.clientLeft;
    this.areaTop = rect.top + this.area.clientTop;
    this.areaWidth = this.area.clientWidth;
    this.areaHeight = this.area.clientHeight;
    this.isAreaStale = false;
  }
}
