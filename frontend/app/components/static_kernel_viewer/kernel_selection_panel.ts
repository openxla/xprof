import {CommonModule} from '@angular/common';
import {
  AfterViewInit,
  ChangeDetectionStrategy,
  ChangeDetectorRef,
  Component,
  EventEmitter,
  inject,
  Input,
  NgZone,
  OnDestroy,
  Output,
} from '@angular/core';
import {MatButtonModule} from '@angular/material/button';
import {MatDividerModule} from '@angular/material/divider';
import {MatIconModule} from '@angular/material/icon';
import {MatMenuModule} from '@angular/material/menu';
import {MatTooltipModule} from '@angular/material/tooltip';
import {
  accentColorForCounterTrack,
  buildBundleStrip,
  buildInstructionMix,
  buildKernelEventTooltip,
  buildUtilizationBars,
  BUNDLE_FORMAT,
  classifyCounterTrack,
  classifyRegister,
  DEFAULT_ACCENT_COLOR,
  DENSITY_FORMAT,
  formatCounterTrackValue,
  FUNCTIONAL_UNITS,
  FUNCTIONAL_UNITS_BY_LANE,
  isSparseCoreBundleContext,
  readHoveredEvent,
  readNumberRecord,
  SPARSE_CORE_ONLY_LANES,
  unitFamilyForLabel,
  unitForCounterTrack,
  type BundleLaneCell,
  type ConsumerGroup,
  type CounterGauge,
  type CounterMetricKind,
  type FunctionalUnit,
  type InstructionMixItem,
  type InstructionOutput,
  type OperandItem,
  type RegionUtilizationSummary,
  type SchedulePosition,
  type UnitTokenFamily,
  type UtilizationBar,
} from 'org_xprof/frontend/app/components/static_kernel_viewer/kernel_event_tooltip';
import {
  EVENT_SELECTED_EVENT_NAME,
  EVENTS_SELECTED_EVENT_NAME,
} from 'org_xprof/frontend/app/components/trace_viewer_container/trace_viewer_container';

/** Co-scheduled instruction executing in the same VLIW bundle. */
export interface BundleEventItem {
  readonly eventIndex: number;
  readonly name: string;
  readonly trackName: string;
  readonly unit?: FunctionalUnit;
  readonly start: number;
  readonly duration: number;
  readonly selected: boolean;
  readonly produces?: string;
  readonly registerKind?: NonNullable<InstructionOutput['registerKind']>;
  readonly ordinalLabel?: string;
}

/** Navigable region item (parent or child region in the schedule hierarchy). */
export interface ChildRegionItem {
  readonly eventIndex: number;
  readonly name: string;
  readonly bundles: string;
  readonly durationLabel: string;
}

/** Enclosing ancestor region crumb in the header breadcrumb bar. */
export interface RegionBreadcrumbItem {
  readonly name: string;
  readonly eventIndex?: number;
  readonly bundles?: string;
  readonly durationLabel?: string;
}

/** Aggregated instruction mnemonic frequency inside a selected region or range. */
export interface TopInstructionItem {
  readonly name: string;
  readonly trackName: string;
  readonly unit?: FunctionalUnit;
  readonly count: number;
  readonly sharePercent: number;
  readonly formattedShare: string;
  readonly totalBundlesLabel: string;
  readonly firstEventIndex: number;
}

/** Individual event inside a multi-event rectangle selection. */
export interface SelectedRangeEventItem {
  readonly eventIndex: number;
  readonly name: string;
  readonly trackName: string;
  readonly unit?: FunctionalUnit;
  readonly bundleLabel: string;
  readonly durationLabel: string;
}

/** Key-value attribute not already surfaced as a dedicated LLO badge. */
export interface ExtraAttributeItem {
  readonly key: string;
  readonly value: string;
}

/** Parsed source file and line/column location for a source or stack frame. */
export interface SourceFrameItem {
  /** Full raw location string, e.g. `models/transformer/.../megablocks_gmm.py:627:5`. */
  readonly raw: string;
  /** File basename with `:line[:col]` suffix, e.g. `megablocks_gmm.py:627:5`. */
  readonly shortLocation: string;
  /** Parent directory path without trailing slash, e.g. `models/transformer/.../kernels`. */
  readonly directory?: string;
  /** 0-based index in the call stack (0 = outermost caller, last = innermost callee). */
  readonly index: number;
  /** True when this frame is the innermost leaf frame where the op was emitted. */
  readonly isInnermost: boolean;
}

/** Rich counter track statistics for single-click counter inspection. */
export interface CounterSelectionDetails {
  readonly kind: CounterMetricKind;
  readonly formattedCurrent: string;
  readonly stepRangeLabel: string;
  readonly stepDurationLabel: string;
  readonly formattedMean?: string;
  readonly formattedPeak?: string;
  readonly formattedMin?: string;
  readonly peakBundleLabel?: string;
  readonly activeCoverageLabel?: string;
  readonly deltaFromMeanLabel?: string;
  readonly isAtPeak?: boolean;
  readonly currentPercentOfPeak: number;
  readonly meanPercentOfPeak?: number;
  readonly sampleCount?: number;
}

/** Per-unit utilization summary (mean, peak, active coverage) across a selected range. */
export interface RangeUtilizationMetricItem {
  readonly unit: FunctionalUnit;
  readonly meanPercent: number;
  readonly formattedMean: string;
  readonly peakPercent: number;
  readonly formattedPeak: string;
  readonly peakBundleLabel?: string;
  readonly minPercent: number;
  readonly formattedMin: string;
  readonly activeBundles: number;
  readonly activeSharePercent: number;
  readonly formattedActiveShare: string;
  readonly highlighted: boolean;
}

/** Memory or register counter summary across a selected range. */
export interface RangeResourceMetricItem {
  readonly trackName: string;
  readonly shortLabel: string;
  readonly kind: 'bytes' | 'registers';
  readonly accentColor: string;
  readonly formattedMean: string;
  readonly formattedPeak: string;
  readonly formattedMin: string;
  readonly peakBundleLabel?: string;
  readonly meanRatioOfPeak: number;
}

/** View model for the Static Kernel Viewer bottom selection panel. */
export interface KernelSelectionContent {
  readonly mode: 'single' | 'range';
  readonly kind: 'instruction' | 'region' | 'counter' | 'process' | 'range';
  readonly name: string;
  readonly isCode: boolean;
  readonly unit?: FunctionalUnit;
  readonly accentColor: string;
  readonly badgeLabel: string;
  readonly badgeIcon: string;
  readonly spanTrackName?: string;
  readonly spanKindLabel?: string;
  readonly bundlesLabel: string;
  readonly bundles: string;
  readonly length?: string;
  readonly hexBundle?: string;
  readonly ordinalLabel?: string;
  readonly subtitleChips?: readonly string[];
  readonly description?: string;
  readonly eventIndex?: number;
  readonly prevEventIndex?: number;
  readonly nextEventIndex?: number;
  readonly output?: InstructionOutput;
  readonly registerTypeLabel?: string;
  readonly operands?: readonly OperandItem[];
  readonly consumers?: readonly string[];
  readonly consumerGroups?: readonly ConsumerGroup[];
  readonly spillsLabel?: string;
  readonly hasSpills?: boolean;
  readonly sourceInfo?: SourceFrameItem;
  readonly sourceStack?: readonly SourceFrameItem[];
  readonly extraAttributes?: readonly ExtraAttributeItem[];
  readonly bundleStrip?: readonly BundleLaneCell[];
  readonly hasLargeBundleCounts?: boolean;
  readonly activeLanesSummary?: string;
  readonly bundleEvents?: readonly BundleEventItem[];
  readonly utilizationBars?: readonly UtilizationBar[];
  readonly peakUtilizationLabel?: string;
  readonly counterGauge?: CounterGauge;
  readonly counterDetails?: CounterSelectionDetails;
  readonly utilizationMetrics?: readonly RangeUtilizationMetricItem[];
  readonly resourceCounterMetrics?: readonly RangeResourceMetricItem[];
  readonly regions?: readonly string[];
  readonly regionBreadcrumbs?: readonly RegionBreadcrumbItem[];
  readonly regionRowLabel?: string;
  readonly parentRegion?: ChildRegionItem;
  readonly schedulePosition?: SchedulePosition;
  readonly instructionMix?: readonly InstructionMixItem[];
  readonly totalInstructions: number;
  readonly meanUtilization?: RegionUtilizationSummary;
  readonly density?: string;
  readonly childRegions?: readonly ChildRegionItem[];
  readonly topInstructions?: readonly TopInstructionItem[];
  readonly selectedEvents?: readonly SelectedRangeEventItem[];
  readonly isSparseCore?: boolean;
  readonly headerSwatchFamily?: UnitTokenFamily | 'primary' | 'warning';
  readonly rangeScheduleMeta?: string;
}

/** Keys in `args` that are already rendered as dedicated LLO badges or rows. */
const DEDICATED_ARG_KEYS = new Set([
  'ordinal',
  'program_order',
  'instruction_ordinal',
  'bundle',
  'opcode',
  'unit',
  'start_bundle',
  'limit_bundle',
  'produces',
  'dest',
  'result',
  'result_register',
  'memory_space',
  'memorySpace',
  'throughput',
  'metadata',
  'data_format',
  'pseudo_kind',
  'treat_copy_as_opaque',
  'unit_id',
  'unitId',
  'scalar_const',
  'scalarConst',
  'xmr',
  'msr',
  'result_buffer_addr',
  'mrb_addr',
  'source_bus',
  'operands',
  'consumers',
  'vector_spills',
  'vectorSpills',
  'vector_fills',
  'vectorFills',
  'hlo_module',
  'hlo_op',
  'uid',
  'source',
  'source_stack',
]);

const REGISTER_TYPE_LABELS: Readonly<
  Record<NonNullable<InstructionOutput['registerKind']>, string>
> = {
  vreg: 'Vector Reg',
  sreg: 'Scalar Reg',
  preg: 'Predicate Reg',
  vmreg: 'Vector Mask',
  dreg: 'Descriptor Reg',
  other: 'Register',
};

/** Visible breadcrumb item in the collapsible scope trail. */
export interface VisibleScopeCrumb {
  readonly crumb: RegionBreadcrumbItem;
  readonly isLast: boolean;
}

/** Unit pill rendered in the `Mean util` / `Unit instrs` focus row. */
export interface UnitFocusPillItem {
  readonly unit: string;
  readonly active: boolean;
  readonly metric?: string;
  readonly hoverText?: string;
  readonly ariaLabel: string;
}

/** Option item in the `All units` filter menu of Pillar 3. */
export interface UnitMenuOption {
  readonly unit: string;
  readonly description: string;
  readonly count: number;
  readonly dividerBefore: boolean;
}

const SPAN_TRACK_KINDS: ReadonlyMap<string, string | undefined> = new Map([
  ['Regions', undefined],
  ['Pallas Primitives', 'Pallas primitive'],
  ['Named Scope', 'Named scope'],
]);

const TENSOR_CORE_UNIT_LANES: readonly string[] = FUNCTIONAL_UNITS.filter(
  (u) => u.label !== 'Other',
).map((u) => u.label);

const SPARSE_CORE_UNIT_LANES = [
  'TVPU',
  'XPU',
  'EUP',
  'SALU',
  'SGE',
  'VLD',
  'VST',
  'DMA',
] as const;

const SPARSE_CORE_UNIT_LABELS: ReadonlySet<string> = new Set([
  ...SPARSE_CORE_ONLY_LANES,
  'SC',
]);

const UNIT_DESCRIPTIONS: ReadonlyMap<string, string> = new Map([
  ['MXU', 'Matrix'],
  ['VPU', 'Vector'],
  ['XLU', 'Cross-Lane'],
  ['EUP', 'Elem Unary'],
  ['SALU', 'Scalar'],
  ['VLD', 'Vector Load'],
  ['VST', 'Vector Store'],
  ['DMA', 'DMA'],
  ['TVPU', 'Tile Vector'],
  ['XPU', 'Cross-Lane'],
  ['SGE', 'Scatter-Gather'],
  ['SC', 'SparseCore'],
]);

function computeIsSparseCore(c: KernelSelectionContent): boolean {
  if (c.unit && SPARSE_CORE_UNIT_LABELS.has(c.unit.label)) {
    return true;
  }
  if (
    c.bundleStrip?.some((cell) => SPARSE_CORE_UNIT_LABELS.has(cell.unit.label))
  ) {
    return true;
  }
  if (
    c.instructionMix?.some((item) =>
      SPARSE_CORE_UNIT_LABELS.has(item.unit.label),
    )
  ) {
    return true;
  }
  if (
    c.utilizationBars?.some((bar) =>
      SPARSE_CORE_UNIT_LABELS.has(bar.unit.label),
    )
  ) {
    return true;
  }
  if (
    c.utilizationMetrics?.some((m) => SPARSE_CORE_UNIT_LABELS.has(m.unit.label))
  ) {
    return true;
  }
  if (
    c.topInstructions?.some((item) =>
      SPARSE_CORE_UNIT_LABELS.has(item.trackName),
    )
  ) {
    return true;
  }
  if (
    c.resourceCounterMetrics?.some((rm) =>
      /\b(tilespmem|tile\s+spmem|spmem|descriptor\s+reg)\b/i.test(rm.trackName),
    )
  ) {
    return true;
  }
  if (c.kind === 'counter') {
    const counterUnit = unitForCounterTrack(c.name);
    if (
      (counterUnit && SPARSE_CORE_UNIT_LABELS.has(counterUnit.label)) ||
      /\b(tilespmem|tile\s+spmem|spmem|descriptor\s+reg)\b/i.test(c.name)
    ) {
      return true;
    }
  }
  if (c.regions?.[0] === 'TEC' || c.regions?.[0] === 'SCS') {
    return true;
  }
  return (
    c.extraAttributes?.some(
      (attr) => attr.key === 'core_type' && /tec|sc/i.test(attr.value),
    ) ?? false
  );
}

function computeHeaderSwatchFamily(
  c: KernelSelectionContent,
): UnitTokenFamily | 'primary' | 'warning' {
  if (c.unit) {
    return unitFamilyForLabel(c.unit.label);
  }
  if (c.kind === 'region') {
    return 'scalar';
  }
  if (c.kind === 'counter') {
    const counterUnit = unitForCounterTrack(c.name);
    if (counterUnit) {
      return unitFamilyForLabel(counterUnit.label);
    }
    if (c.counterDetails?.kind === 'registers') {
      return 'scalar';
    }
    return /spill/i.test(c.name) ? 'warning' : 'primary';
  }
  if (c.kind === 'range') {
    return 'memory';
  }
  return 'primary';
}

function computeRangeScheduleMeta(
  c: Pick<KernelSelectionContent, 'subtitleChips' | 'length' | 'spanTrackName'>,
): string | undefined {
  const chips = (c.subtitleChips ?? []).filter(
    (chip) =>
      chip !== c.length &&
      chip !== c.spanTrackName &&
      !/\binstructions?$/i.test(chip),
  );
  return chips.length > 0 ? chips.join(' · ') : undefined;
}

function finalizeSelectionContent(
  base: KernelSelectionContent,
): KernelSelectionContent {
  const isSparseCore = computeIsSparseCore(base);
  const headerSwatchFamily = computeHeaderSwatchFamily(base);
  const rangeScheduleMeta = computeRangeScheduleMeta(base);
  return {
    ...base,
    isSparseCore,
    headerSwatchFamily,
    ...(rangeScheduleMeta ? {rangeScheduleMeta} : {}),
  };
}

function asRecord(value: unknown): Record<string, unknown> | undefined {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : undefined;
}

function clampPercent(percent: number): number {
  return Math.max(0, Math.min(100, Math.round(percent)));
}

function formatBundleRange(start: number, duration: number): string {
  const first = BUNDLE_FORMAT.format(Math.floor(start));
  if (duration <= 1) {
    return first;
  }
  return `${first}–${BUNDLE_FORMAT.format(Math.ceil(start + duration))}`;
}

function readBundleEvents(
  raw: unknown,
): readonly BundleEventItem[] | undefined {
  if (!Array.isArray(raw)) {
    return undefined;
  }
  const items: BundleEventItem[] = [];
  for (const rawEntry of raw) {
    const entry = asRecord(rawEntry);
    if (!entry) continue;
    const name = entry['name'];
    const trackName =
      typeof entry['trackName'] === 'string' ? entry['trackName'] : '';
    const start = typeof entry['startUs'] === 'number' ? entry['startUs'] : 0;
    const duration =
      typeof entry['durationUs'] === 'number' ? entry['durationUs'] : 1;
    const eventIndex =
      typeof entry['eventIndex'] === 'number' ? entry['eventIndex'] : -1;
    if (typeof name !== 'string' || !name) continue;
    const unit = FUNCTIONAL_UNITS_BY_LANE.get(trackName);
    const produces =
      typeof entry['produces'] === 'string' && entry['produces'].trim() !== ''
        ? entry['produces'].trim()
        : undefined;
    const ordinalRaw = entry['ordinal'];
    let ordinalLabel: string | undefined = undefined;
    if (typeof ordinalRaw === 'number' && Number.isFinite(ordinalRaw)) {
      ordinalLabel = `Ord #${BUNDLE_FORMAT.format(Math.round(ordinalRaw))}`;
    } else if (
      typeof ordinalRaw === 'string' &&
      /^\d+$/.test(ordinalRaw.trim())
    ) {
      ordinalLabel = `Ord #${BUNDLE_FORMAT.format(Number(ordinalRaw.trim()))}`;
    }
    items.push({
      eventIndex,
      name,
      trackName,
      ...(unit ? {unit} : {}),
      start,
      duration,
      selected: entry['selected'] === true,
      ...(produces ? {produces, registerKind: classifyRegister(produces)} : {}),
      ...(ordinalLabel ? {ordinalLabel} : {}),
    });
  }
  return items.length > 0 ? items : undefined;
}

function readChildRegions(
  raw: unknown,
): readonly ChildRegionItem[] | undefined {
  if (!Array.isArray(raw)) {
    return undefined;
  }
  const items: ChildRegionItem[] = [];
  for (const rawEntry of raw) {
    const entry = asRecord(rawEntry);
    if (!entry) continue;
    const name = entry['name'];
    const start = typeof entry['startUs'] === 'number' ? entry['startUs'] : 0;
    const duration =
      typeof entry['durationUs'] === 'number' ? entry['durationUs'] : 0;
    const eventIndex =
      typeof entry['eventIndex'] === 'number' ? entry['eventIndex'] : -1;
    if (typeof name !== 'string' || !name) continue;
    items.push({
      eventIndex,
      name,
      bundles: formatBundleRange(start, duration),
      durationLabel: `${BUNDLE_FORMAT.format(Math.max(1, Math.round(duration)))} ${Math.round(duration) === 1 ? 'bundle' : 'bundles'}`,
    });
  }
  return items.length > 0 ? items : undefined;
}

function buildRegionBreadcrumbs(
  regions: readonly string[] | undefined,
  ancestors: readonly ChildRegionItem[] | undefined,
): readonly RegionBreadcrumbItem[] | undefined {
  if (ancestors && ancestors.length > 0) {
    const items: RegionBreadcrumbItem[] = ancestors.map((item) => ({
      name: item.name,
      ...(item.eventIndex >= 0 ? {eventIndex: item.eventIndex} : {}),
      bundles: item.bundles,
      durationLabel: item.durationLabel,
    }));
    if (regions && regions.length > items.length) {
      for (let i = items.length; i < regions.length; i++) {
        items.push({name: regions[i]});
      }
    }
    return items;
  }
  if (regions && regions.length > 0) {
    return regions.map((name) => ({name}));
  }
  return undefined;
}

function readTopInstructions(
  raw: unknown,
  totalInstructions: number,
): readonly TopInstructionItem[] | undefined {
  if (!Array.isArray(raw)) {
    return undefined;
  }
  const items: TopInstructionItem[] = [];
  for (const rawEntry of raw) {
    const entry = asRecord(rawEntry);
    if (!entry) continue;
    const name = entry['name'];
    const trackName =
      typeof entry['trackName'] === 'string' ? entry['trackName'] : '';
    const count =
      typeof entry['count'] === 'number' ? Math.round(entry['count']) : 0;
    const totalBundles =
      typeof entry['totalBundles'] === 'number'
        ? entry['totalBundles']
        : typeof entry['wallTimeUs'] === 'number'
          ? entry['wallTimeUs']
          : count;
    const firstEventIndex =
      typeof entry['firstEventIndex'] === 'number'
        ? entry['firstEventIndex']
        : -1;
    if (typeof name !== 'string' || !name || count <= 0) continue;
    const unit = FUNCTIONAL_UNITS_BY_LANE.get(trackName);
    const sharePercent =
      totalInstructions > 0
        ? Math.max(
            0,
            Math.min(100, Math.round((count / totalInstructions) * 100)),
          )
        : 0;
    items.push({
      name,
      trackName,
      ...(unit ? {unit} : {}),
      count,
      sharePercent,
      formattedShare: `${sharePercent}%`,
      totalBundlesLabel: `${BUNDLE_FORMAT.format(Math.round(totalBundles))} ${Math.round(totalBundles) === 1 ? 'bundle' : 'bundles'}`,
      firstEventIndex,
    });
  }
  return items.length > 0 ? items : undefined;
}

function readSelectedRangeEvents(
  raw: unknown,
): readonly SelectedRangeEventItem[] | undefined {
  if (!Array.isArray(raw)) {
    return undefined;
  }
  const items: SelectedRangeEventItem[] = [];
  for (const rawEntry of raw) {
    const entry = asRecord(rawEntry);
    if (!entry) continue;
    const name = entry['name'];
    const trackName =
      typeof entry['trackName'] === 'string' ? entry['trackName'] : '';
    const start = typeof entry['startUs'] === 'number' ? entry['startUs'] : 0;
    const duration =
      typeof entry['durationUs'] === 'number' ? entry['durationUs'] : 1;
    const eventIndex =
      typeof entry['eventIndex'] === 'number' ? entry['eventIndex'] : -1;
    if (typeof name !== 'string' || !name) continue;
    const unit = FUNCTIONAL_UNITS_BY_LANE.get(trackName);
    items.push({
      eventIndex,
      name,
      trackName,
      ...(unit ? {unit} : {}),
      bundleLabel: formatBundleRange(start, duration),
      durationLabel: `${BUNDLE_FORMAT.format(Math.max(1, Math.round(duration)))} ${Math.round(duration) === 1 ? 'bundle' : 'bundles'}`,
    });
  }
  return items.length > 0 ? items : undefined;
}

function parseSourceFrame(
  raw: string,
  index = 0,
  isInnermost = true,
): SourceFrameItem {
  const trimmed = raw.trim();
  const lastSlash = trimmed.lastIndexOf('/');
  const directory = lastSlash > 0 ? trimmed.slice(0, lastSlash) : undefined;
  const shortLocation = lastSlash >= 0 ? trimmed.slice(lastSlash + 1) : trimmed;
  return {
    raw: trimmed,
    shortLocation: shortLocation || trimmed,
    ...(directory ? {directory} : {}),
    index,
    isInnermost,
  };
}

function parseSourceStack(
  rawStack: string | undefined,
): readonly SourceFrameItem[] | undefined {
  if (!rawStack) {
    return undefined;
  }
  const lines = rawStack
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter((line) => line !== '');
  if (lines.length === 0) {
    return undefined;
  }
  return lines.map((line, index) =>
    parseSourceFrame(line, index, index === lines.length - 1),
  );
}

function buildCounterSelectionDetails(
  detail: Record<string, unknown>,
  trackName: string,
  startBundle: number,
  counterValue: number | undefined,
  totalBundles: number | undefined,
): CounterSelectionDetails | undefined {
  const kind = classifyCounterTrack(trackName);
  const currentVal = counterValue ?? 0;
  const formattedCurrent = formatCounterTrackValue(currentVal, kind);

  const stepStart =
    typeof detail['stepStartBundle'] === 'number'
      ? Math.max(0, Math.floor(detail['stepStartBundle']))
      : Math.max(0, Math.floor(startBundle));
  const stepDuration =
    typeof detail['stepDurationBundles'] === 'number' &&
    detail['stepDurationBundles'] > 0
      ? Math.round(detail['stepDurationBundles'])
      : 1;
  const stepRangeLabel =
    stepDuration <= 1
      ? `Bundle ${BUNDLE_FORMAT.format(stepStart)}`
      : `Bundles ${formatBundleRange(stepStart, stepDuration)}`;
  const stepDurationLabel = `${BUNDLE_FORMAT.format(stepDuration)} ${stepDuration === 1 ? 'bundle' : 'bundles'}`;

  const rawStats = asRecord(detail['counterStats']);
  const meanVal =
    typeof rawStats?.['mean'] === 'number' && Number.isFinite(rawStats['mean'])
      ? rawStats['mean']
      : undefined;
  const maxVal =
    typeof rawStats?.['max'] === 'number' && Number.isFinite(rawStats['max'])
      ? rawStats['max']
      : undefined;
  const minVal =
    typeof rawStats?.['min'] === 'number' && Number.isFinite(rawStats['min'])
      ? rawStats['min']
      : undefined;
  const peakBundle =
    typeof rawStats?.['peakBundle'] === 'number' &&
    Number.isFinite(rawStats['peakBundle'])
      ? Math.max(0, Math.floor(rawStats['peakBundle']))
      : undefined;
  const activeBundles =
    typeof rawStats?.['activeBundles'] === 'number' &&
    Number.isFinite(rawStats['activeBundles'])
      ? Math.max(0, Math.round(rawStats['activeBundles']))
      : undefined;
  const sampleCount =
    typeof rawStats?.['sampleCount'] === 'number' &&
    Number.isFinite(rawStats['sampleCount'])
      ? Math.max(0, Math.round(rawStats['sampleCount']))
      : undefined;

  const formattedMean =
    meanVal !== undefined ? formatCounterTrackValue(meanVal, kind) : undefined;
  const formattedPeak =
    maxVal !== undefined ? formatCounterTrackValue(maxVal, kind) : undefined;
  const formattedMin =
    minVal !== undefined ? formatCounterTrackValue(minVal, kind) : undefined;
  const peakBundleLabel =
    peakBundle !== undefined
      ? `@ Bundle ${BUNDLE_FORMAT.format(peakBundle)}`
      : undefined;

  let activeCoverageLabel: string | undefined = undefined;
  if (activeBundles !== undefined) {
    if (totalBundles !== undefined && totalBundles > 0) {
      const pct = clampPercent((activeBundles / totalBundles) * 100);
      activeCoverageLabel = `${BUNDLE_FORMAT.format(activeBundles)} / ${BUNDLE_FORMAT.format(totalBundles)} bundles (${pct}%)`;
    } else {
      activeCoverageLabel = `${BUNDLE_FORMAT.format(activeBundles)} active ${activeBundles === 1 ? 'bundle' : 'bundles'}`;
    }
  }

  const isAtPeak =
    maxVal !== undefined &&
    maxVal > 0 &&
    Math.abs(currentVal - maxVal) <= Math.max(1e-4, maxVal * 0.005);

  let deltaFromMeanLabel: string | undefined = undefined;
  if (isAtPeak) {
    deltaFromMeanLabel = 'At Op Peak';
  } else if (meanVal !== undefined) {
    if (kind === 'utilization') {
      const diffPct = Math.round((currentVal - meanVal) * 100);
      if (diffPct > 0) {
        deltaFromMeanLabel = `+${diffPct}% vs mean`;
      } else if (diffPct < 0) {
        deltaFromMeanLabel = `${diffPct}% vs mean`;
      } else {
        deltaFromMeanLabel = 'At mean';
      }
    } else if (meanVal > 0) {
      const ratio = currentVal / meanVal;
      deltaFromMeanLabel = `${DENSITY_FORMAT.format(ratio)}× mean`;
    }
  }

  const currentPercentOfPeak =
    kind === 'utilization'
      ? clampPercent(currentVal * 100)
      : maxVal !== undefined && maxVal > 0
        ? clampPercent((currentVal / maxVal) * 100)
        : currentVal > 0
          ? 65
          : 0;
  const meanPercentOfPeak =
    meanVal !== undefined
      ? kind === 'utilization'
        ? clampPercent(meanVal * 100)
        : maxVal !== undefined && maxVal > 0
          ? clampPercent((meanVal / maxVal) * 100)
          : undefined
      : undefined;

  return {
    kind,
    formattedCurrent,
    stepRangeLabel,
    stepDurationLabel,
    ...(formattedMean ? {formattedMean} : {}),
    ...(formattedPeak ? {formattedPeak} : {}),
    ...(formattedMin ? {formattedMin} : {}),
    ...(peakBundleLabel ? {peakBundleLabel} : {}),
    ...(activeCoverageLabel ? {activeCoverageLabel} : {}),
    ...(deltaFromMeanLabel ? {deltaFromMeanLabel} : {}),
    ...(isAtPeak ? {isAtPeak} : {}),
    currentPercentOfPeak,
    ...(meanPercentOfPeak !== undefined ? {meanPercentOfPeak} : {}),
    ...(sampleCount !== undefined ? {sampleCount} : {}),
  };
}

function readRangeCounterMetrics(
  raw: unknown,
  rangeExtent: number,
): {
  utilizationMetrics?: readonly RangeUtilizationMetricItem[];
  resourceCounterMetrics?: readonly RangeResourceMetricItem[];
} {
  if (!Array.isArray(raw)) {
    return {};
  }
  const rawUtilItems: Array<Omit<RangeUtilizationMetricItem, 'highlighted'>> =
    [];
  const resourceItems: RangeResourceMetricItem[] = [];

  for (const rawEntry of raw) {
    const entry = asRecord(rawEntry);
    if (!entry) continue;
    const trackName =
      typeof entry['trackName'] === 'string' ? entry['trackName'].trim() : '';
    if (!trackName) continue;
    const mean =
      typeof entry['mean'] === 'number' && Number.isFinite(entry['mean'])
        ? entry['mean']
        : 0;
    const max =
      typeof entry['max'] === 'number' && Number.isFinite(entry['max'])
        ? entry['max']
        : mean;
    const min =
      typeof entry['min'] === 'number' && Number.isFinite(entry['min'])
        ? entry['min']
        : 0;
    const peakBundle =
      typeof entry['peakBundle'] === 'number' &&
      Number.isFinite(entry['peakBundle'])
        ? Math.max(0, Math.floor(entry['peakBundle']))
        : undefined;
    const activeBundles =
      typeof entry['activeBundles'] === 'number' &&
      Number.isFinite(entry['activeBundles'])
        ? Math.max(0, Math.round(entry['activeBundles']))
        : 0;

    const unit = unitForCounterTrack(trackName);
    const kind = classifyCounterTrack(trackName);
    if (unit && kind === 'utilization') {
      const meanPercent = clampPercent(mean * 100);
      const peakPercent = clampPercent(max * 100);
      const minPercent = clampPercent(min * 100);
      const activeSharePercent =
        rangeExtent > 0 ? clampPercent((activeBundles / rangeExtent) * 100) : 0;
      if (meanPercent > 0 || peakPercent > 0 || activeBundles > 0) {
        rawUtilItems.push({
          unit,
          meanPercent,
          formattedMean: `${meanPercent}%`,
          peakPercent,
          formattedPeak: `${peakPercent}%`,
          ...(peakBundle !== undefined
            ? {peakBundleLabel: `@ ${BUNDLE_FORMAT.format(peakBundle)}`}
            : {}),
          minPercent,
          formattedMin: `${minPercent}%`,
          activeBundles,
          activeSharePercent,
          formattedActiveShare: `${activeSharePercent}% active`,
        });
      }
    } else if (kind === 'bytes' || kind === 'registers') {
      if (mean > 0 || max > 0 || activeBundles > 0) {
        resourceItems.push({
          trackName,
          shortLabel: trackName,
          kind,
          accentColor: accentColorForCounterTrack(),
          formattedMean: formatCounterTrackValue(mean, kind),
          formattedPeak: formatCounterTrackValue(max, kind),
          formattedMin: formatCounterTrackValue(min, kind),
          ...(peakBundle !== undefined
            ? {peakBundleLabel: `@ ${BUNDLE_FORMAT.format(peakBundle)}`}
            : {}),
          meanRatioOfPeak: max > 0 ? clampPercent((mean / max) * 100) : 0,
        });
      }
    }
  }

  rawUtilItems.sort(
    (a, b) => b.meanPercent - a.meanPercent || b.peakPercent - a.peakPercent,
  );
  const topMean = rawUtilItems[0]?.meanPercent ?? 0;
  const topPeak = rawUtilItems[0]?.peakPercent ?? 0;
  const utilizationMetrics: RangeUtilizationMetricItem[] | undefined =
    rawUtilItems.length > 0
      ? rawUtilItems.map((item, idx) => ({
          ...item,
          highlighted: idx === 0 && (topMean > 0 || topPeak > 0),
        }))
      : undefined;

  return {
    ...(utilizationMetrics ? {utilizationMetrics} : {}),
    ...(resourceItems.length > 0
      ? {resourceCounterMetrics: resourceItems}
      : {}),
  };
}

/**
 * Builds the bottom selection panel content from an `eventselected` detail
 * payload. Returns `undefined` when deselected (`eventIndex === -1` and not a
 * counter track selection).
 */
export function buildKernelSelectionFromEvent(
  rawDetail: unknown,
): KernelSelectionContent | undefined {
  const detail = asRecord(rawDetail);
  if (!detail) {
    return undefined;
  }
  const hovered = readHoveredEvent(detail);
  if (!hovered) {
    return undefined;
  }
  const tooltip = buildKernelEventTooltip(hovered);
  const eventIndex =
    typeof detail['eventIndex'] === 'number' && detail['eventIndex'] >= 0
      ? detail['eventIndex']
      : undefined;
  const prevEventIndex =
    typeof detail['prevEventIndex'] === 'number' &&
    detail['prevEventIndex'] >= 0
      ? detail['prevEventIndex']
      : undefined;
  const nextEventIndex =
    typeof detail['nextEventIndex'] === 'number' &&
    detail['nextEventIndex'] >= 0
      ? detail['nextEventIndex']
      : undefined;

  const isSpan = SPAN_TRACK_KINDS.has(hovered.trackName);
  const isRegion = tooltip.kind === 'region' || isSpan;
  const mode: KernelSelectionContent['mode'] = isRegion ? 'range' : 'single';
  const kind: KernelSelectionContent['kind'] = isSpan ? 'region' : tooltip.kind;
  const spanTrackName = isSpan ? hovered.trackName : undefined;
  const spanKindLabel = SPAN_TRACK_KINDS.get(hovered.trackName);
  const counterKind =
    tooltip.kind === 'counter'
      ? classifyCounterTrack(hovered.trackName)
      : undefined;
  const badgeLabel = isRegion
    ? (spanKindLabel ?? 'Region')
    : tooltip.kind === 'counter'
      ? counterKind === 'bytes'
        ? 'Memory Counter'
        : counterKind === 'registers'
          ? 'Register Pressure'
          : 'Utilization'
      : (tooltip.unit?.label ?? 'Instruction');
  const badgeIcon = isRegion
    ? 'account_tree'
    : tooltip.kind === 'counter'
      ? counterKind === 'bytes'
        ? 'storage'
        : counterKind === 'registers'
          ? 'developer_board'
          : 'monitoring'
      : 'memory';

  const sourceStack = parseSourceStack(hovered.args?.['source_stack']);
  const rawSource = hovered.args?.['source']?.trim();
  const sourceInfo = rawSource
    ? parseSourceFrame(rawSource, 0, true)
    : sourceStack && sourceStack.length > 0
      ? sourceStack[sourceStack.length - 1]
      : undefined;

  const extraAttributes: ExtraAttributeItem[] = [];
  if (hovered.args) {
    for (const [key, value] of Object.entries(hovered.args)) {
      if (!DEDICATED_ARG_KEYS.has(key) && value.trim() !== '') {
        const normalizedValue = value.replace(/\s*\r?\n\s*/g, ' ↵ ').trim();
        extraAttributes.push({key, value: normalizedValue});
      }
    }
  }

  const bundleEvents = readBundleEvents(detail['bundleEvents']);
  const bundleStrip =
    tooltip.bundleStrip ??
    buildBundleStrip(
      hovered.bundleCounts,
      isSparseCoreBundleContext(hovered.bundleCounts, hovered),
    );
  const activeLanes = bundleStrip
    ? bundleStrip.filter((cell) => cell.active).length
    : 0;
  const hasLargeBundleCounts =
    bundleStrip?.some((cell) => cell.count >= 10) ?? false;
  const totalOpsInBundle = bundleStrip
    ? bundleStrip.reduce((sum, cell) => sum + cell.count, 0)
    : (bundleEvents?.length ?? 0);
  const activeLanesSummary =
    totalOpsInBundle > 0 && bundleStrip
      ? `${BUNDLE_FORMAT.format(totalOpsInBundle)} ${totalOpsInBundle === 1 ? 'op' : 'ops'} · ${activeLanes}/${bundleStrip.length} lanes active`
      : undefined;

  const utilizationBars =
    tooltip.utilizationBars ?? buildUtilizationBars(hovered.utilization);
  const peakBar = utilizationBars?.[0];
  const peakUtilizationLabel =
    peakBar && peakBar.percent > 0
      ? `Peak: ${peakBar.unit.label} ${peakBar.formattedPercent}`
      : undefined;

  const mixResult = buildInstructionMix(hovered.bundleCounts);
  const totalInstructions =
    typeof detail['totalInstructions'] === 'number' &&
    detail['totalInstructions'] > 0
      ? Math.round(detail['totalInstructions'])
      : mixResult.totalInstructions;
  const regions =
    tooltip.regions ??
    (hovered.regions && hovered.regions.length > 0
      ? hovered.regions
      : undefined);
  const regionRowLabel =
    tooltip.regionRowLabel ?? (isRegion ? 'Parent regions' : 'Regions');
  const regionAncestors = readChildRegions(detail['regionAncestors']);
  const regionBreadcrumbs = buildRegionBreadcrumbs(regions, regionAncestors);
  const parentRegion =
    regionAncestors && regionAncestors.length > 0
      ? regionAncestors[regionAncestors.length - 1]
      : undefined;
  const childRegions = readChildRegions(detail['childRegions']);
  const topInstructions = readTopInstructions(
    detail['topInstructions'],
    totalInstructions,
  );
  const schedulePosition: SchedulePosition | undefined =
    !isRegion && hovered.totalBundles !== undefined && hovered.totalBundles > 0
      ? (() => {
          const percent = Math.max(
            0,
            Math.min(
              100,
              Math.round((hovered.start / hovered.totalBundles) * 100),
            ),
          );
          return {
            percent,
            formattedPercent: `${percent}%`,
            totalBundles: BUNDLE_FORMAT.format(
              Math.floor(hovered.totalBundles),
            ),
          };
        })()
      : undefined;
  const counterDetails =
    tooltip.kind === 'counter'
      ? buildCounterSelectionDetails(
          detail,
          hovered.trackName,
          hovered.start,
          hovered.counterValue,
          hovered.totalBundles,
        )
      : undefined;

  const instructionMix = tooltip.instructionMix ?? mixResult.mix;
  const density =
    tooltip.density ??
    (totalInstructions > 0 && hovered.duration > 0
      ? `${DENSITY_FORMAT.format(totalInstructions / hovered.duration)} instr / bundle`
      : undefined);
  const meanUtilization: RegionUtilizationSummary | undefined =
    tooltip.meanUtilization ??
    (utilizationBars && utilizationBars.length > 0
      ? {items: utilizationBars.slice(0, 3)}
      : undefined);
  const subtitleChips: string[] = isSpan
    ? [
        ...(tooltip.length ? [tooltip.length] : []),
        ...(hovered.totalBundles && hovered.totalBundles > 0
          ? [
              `${clampPercent((hovered.duration / hovered.totalBundles) * 100)}% of static schedule`,
            ]
          : []),
        ...(hovered.regionDepth !== undefined
          ? [`depth ${hovered.regionDepth}`]
          : []),
        ...(totalInstructions > 0
          ? [
              `${BUNDLE_FORMAT.format(totalInstructions)} ${totalInstructions === 1 ? 'instruction' : 'instructions'}`,
            ]
          : []),
      ]
    : [...(tooltip.subtitleChips ?? [])];

  return finalizeSelectionContent({
    mode,
    kind,
    name: tooltip.name,
    isCode: tooltip.isCode,
    ...(tooltip.unit ? {unit: tooltip.unit} : {}),
    accentColor: tooltip.accentColor,
    badgeLabel,
    badgeIcon,
    ...(spanTrackName ? {spanTrackName} : {}),
    ...(spanKindLabel ? {spanKindLabel} : {}),
    bundlesLabel: tooltip.bundlesLabel,
    bundles: tooltip.bundles,
    ...(tooltip.length ? {length: tooltip.length} : {}),
    ...(tooltip.hexBundle ? {hexBundle: tooltip.hexBundle} : {}),
    ...(tooltip.ordinalLabel ? {ordinalLabel: tooltip.ordinalLabel} : {}),
    ...(subtitleChips.length > 0 ? {subtitleChips} : {}),
    ...(tooltip.description ? {description: tooltip.description} : {}),
    ...(eventIndex !== undefined ? {eventIndex} : {}),
    ...(prevEventIndex !== undefined ? {prevEventIndex} : {}),
    ...(nextEventIndex !== undefined ? {nextEventIndex} : {}),
    ...(tooltip.output
      ? {
          output: tooltip.output,
          ...(tooltip.output.registerKind
            ? {
                registerTypeLabel:
                  REGISTER_TYPE_LABELS[tooltip.output.registerKind],
              }
            : {}),
        }
      : {}),
    ...(tooltip.operands ? {operands: tooltip.operands} : {}),
    ...(tooltip.consumers ? {consumers: tooltip.consumers} : {}),
    ...(tooltip.consumerGroups ? {consumerGroups: tooltip.consumerGroups} : {}),
    ...(tooltip.spillsLabel
      ? {spillsLabel: tooltip.spillsLabel, hasSpills: tooltip.hasSpills}
      : {}),
    ...(sourceInfo ? {sourceInfo} : {}),
    ...(sourceStack ? {sourceStack} : {}),
    ...(extraAttributes.length > 0 ? {extraAttributes} : {}),
    ...(bundleStrip ? {bundleStrip} : {}),
    ...(hasLargeBundleCounts ? {hasLargeBundleCounts: true} : {}),
    ...(activeLanesSummary ? {activeLanesSummary} : {}),
    ...(bundleEvents ? {bundleEvents} : {}),
    ...(utilizationBars ? {utilizationBars} : {}),
    ...(peakUtilizationLabel ? {peakUtilizationLabel} : {}),
    ...(tooltip.counterGauge ? {counterGauge: tooltip.counterGauge} : {}),
    ...(counterDetails ? {counterDetails} : {}),
    ...(regions ? {regions, regionRowLabel} : {}),
    ...(regionBreadcrumbs ? {regionBreadcrumbs} : {}),
    ...(parentRegion && parentRegion.eventIndex >= 0 ? {parentRegion} : {}),
    ...(schedulePosition ? {schedulePosition} : {}),
    ...(instructionMix ? {instructionMix} : {}),
    totalInstructions,
    ...(meanUtilization ? {meanUtilization} : {}),
    ...(density ? {density} : {}),
    ...(childRegions ? {childRegions} : {}),
    ...(topInstructions ? {topInstructions} : {}),
  });
}

/**
 * Builds the bottom selection panel content from an `events_selected`
 * multi-event rectangle selection payload. Returns `undefined` when cleared.
 */
export function buildKernelSelectionFromRange(
  rawDetail: unknown,
): KernelSelectionContent | undefined {
  const detail = asRecord(rawDetail);
  if (!detail) {
    return undefined;
  }

  let parsedJson: Record<string, unknown> | undefined = undefined;
  if (
    typeof detail['events_selected_data'] === 'string' &&
    detail['events_selected_data'].trim() !== ''
  ) {
    try {
      const parsed: unknown = JSON.parse(detail['events_selected_data']);
      parsedJson = asRecord(parsed);
    } catch {
      // Ignore malformed JSON fallback.
    }
  }

  const start =
    typeof detail['selectionStart'] === 'number'
      ? Math.max(0, Math.floor(detail['selectionStart']))
      : typeof parsedJson?.['selectionStartUs'] === 'number'
        ? Math.max(0, Math.floor(parsedJson['selectionStartUs']))
        : undefined;
  const extent =
    typeof detail['selectionExtent'] === 'number'
      ? Math.max(0, Math.round(detail['selectionExtent']))
      : typeof parsedJson?.['selectionExtentUs'] === 'number'
        ? Math.max(0, Math.round(parsedJson['selectionExtentUs']))
        : undefined;

  const rangeStart = start ?? 0;
  const rangeExtent = extent && extent > 0 ? extent : 1;

  const bundleCounts = readNumberRecord(detail['bundleCounts']);
  const utilization = readNumberRecord(detail['utilization']);
  const mixResult = buildInstructionMix(bundleCounts);
  const utilizationBars = buildUtilizationBars(utilization);
  const {utilizationMetrics, resourceCounterMetrics} = readRangeCounterMetrics(
    detail['counterMetrics'],
    rangeExtent,
  );

  const selectedEvents =
    readSelectedRangeEvents(detail['selectedEvents']) ??
    readSelectedRangeEvents(parsedJson?.['rawEvents']);

  const totalInstructions =
    typeof detail['totalInstructions'] === 'number' &&
    detail['totalInstructions'] > 0
      ? Math.round(detail['totalInstructions'])
      : mixResult.totalInstructions > 0
        ? mixResult.totalInstructions
        : (selectedEvents?.length ?? 0);

  const totalSelectedEvents =
    typeof detail['totalSelectedEvents'] === 'number'
      ? Math.round(detail['totalSelectedEvents'])
      : (selectedEvents?.length ?? totalInstructions);

  if (
    totalSelectedEvents <= 0 &&
    totalInstructions <= 0 &&
    !selectedEvents &&
    !parsedJson?.['metrics'] &&
    !utilizationBars &&
    !utilizationMetrics &&
    !resourceCounterMetrics
  ) {
    return undefined;
  }

  const topInstructions =
    readTopInstructions(detail['topInstructions'], totalInstructions) ??
    readTopInstructions(parsedJson?.['metrics'], totalInstructions);

  const bundles = formatBundleRange(rangeStart, rangeExtent);
  const length = `${BUNDLE_FORMAT.format(rangeExtent)} ${rangeExtent === 1 ? 'bundle' : 'bundles'}`;

  const subtitleChips: string[] = [length];
  const totalBundles =
    typeof detail['totalBundles'] === 'number' && detail['totalBundles'] > 0
      ? detail['totalBundles']
      : undefined;
  if (totalBundles !== undefined) {
    const share = Math.max(
      0,
      Math.min(100, Math.round((rangeExtent / totalBundles) * 100)),
    );
    subtitleChips.push(`${share}% of static schedule`);
  }
  if (totalInstructions > 0) {
    subtitleChips.push(
      `${BUNDLE_FORMAT.format(totalInstructions)} ${totalInstructions === 1 ? 'instruction' : 'instructions'}`,
    );
  }

  let meanUtilization: RegionUtilizationSummary | undefined = undefined;
  if (utilizationBars && utilizationBars.length > 0) {
    meanUtilization = {
      items: utilizationBars,
    };
  } else if (utilizationMetrics && utilizationMetrics.length > 0) {
    const synthesizedBars: UtilizationBar[] = utilizationMetrics.map(
      (item) => ({
        unit: item.unit,
        percent: item.meanPercent,
        formattedPercent: item.formattedMean,
        highlighted: item.highlighted,
      }),
    );
    meanUtilization = {
      items: synthesizedBars,
    };
  }

  const density =
    totalInstructions > 0 && rangeExtent > 0
      ? `${DENSITY_FORMAT.format(totalInstructions / rangeExtent)} instr / bundle`
      : undefined;

  const topUtilizationBar = meanUtilization?.items[0];
  const dominantUnit =
    topUtilizationBar && topUtilizationBar.percent >= 20
      ? FUNCTIONAL_UNITS_BY_LANE.get(topUtilizationBar.unit.label)
      : mixResult.mix?.[0]?.unit;
  const accentColor = dominantUnit?.color ?? DEFAULT_ACCENT_COLOR;

  return finalizeSelectionContent({
    mode: 'range',
    kind: 'range',
    name: `Bundles ${bundles}`,
    isCode: false,
    ...(dominantUnit ? {unit: dominantUnit} : {}),
    accentColor,
    badgeLabel: 'Range Selection',
    badgeIcon: 'select_all',
    bundlesLabel: 'Bundles',
    bundles,
    length,
    subtitleChips,
    ...(mixResult.mix ? {instructionMix: tooltipOrMix(mixResult.mix)} : {}),
    totalInstructions,
    ...(meanUtilization ? {meanUtilization} : {}),
    ...(utilizationBars ? {utilizationBars} : {}),
    ...(utilizationMetrics ? {utilizationMetrics} : {}),
    ...(resourceCounterMetrics ? {resourceCounterMetrics} : {}),
    ...(density ? {density} : {}),
    ...(topInstructions ? {topInstructions} : {}),
    ...(selectedEvents ? {selectedEvents} : {}),
  });
}

function tooltipOrMix(
  mix: readonly InstructionMixItem[],
): readonly InstructionMixItem[] {
  return mix;
}

/**
 * Bottom selection details panel for the Static Kernel Viewer.
 *
 * Displays a 3-column hardware schedule inspection dashboard when the user
 * clicks an instruction, region, or counter point, or drags a selection
 * rectangle across multiple bundles.
 */
@Component({
  standalone: true,
  selector: 'kernel-selection-panel',
  templateUrl: './kernel_selection_panel.ng.html',
  styleUrls: ['./kernel_selection_panel.scss'],
  changeDetection: ChangeDetectionStrategy.OnPush,
  imports: [
    CommonModule,
    MatButtonModule,
    MatDividerModule,
    MatIconModule,
    MatMenuModule,
    MatTooltipModule,
  ],
})
export class KernelSelectionPanel implements AfterViewInit, OnDestroy {
  /** Active HLO op / kernel name, shown in op-wide counter tooltips. */
  @Input() kernelName: string | undefined = undefined;
  /** Emits true when a selection is active, false when closed or deselected. */
  @Output() readonly openChange = new EventEmitter<boolean>();
  /** Emits an `eventIndex` to select and reveal on the timeline. */
  @Output() readonly selectEvent = new EventEmitter<number>();
  /** Emits a search query to highlight matching events on the timeline. */
  @Output() readonly searchQuery = new EventEmitter<string>();
  /** Emits a source location, e.g. `foo.py:12:3`, to show in the source. */
  @Output() readonly openSource = new EventEmitter<string>();

  content: KernelSelectionContent | undefined = undefined;
  /** Optional unit label filter applied to the Top Instructions / Events table. */
  selectedUnitFilter: string | undefined = undefined;
  /** Currently hovered functional unit in the Instruction Mix stacked bar or list. */
  hoveredMixUnit: string | undefined = undefined;
  /** Tab mode for the right-hand table in range selections (`top` or `events`). */
  rangeTableTab: 'top' | 'events' = 'top';
  /** Active mode for the Pillar 2 unit pills row (`Mean util` or `Unit instrs`). */
  unitPillMode: 'util' | 'instrs' = 'util';
  /** Whether the collapsible `Stack (N)` frame picker under Source is open. */
  stackOpen = false;
  /** 0-based index of the call-stack frame picked by the user, if any. */
  pickedStackFrameIndex: number | undefined = undefined;
  /** Stack of previously inspected selections for in-panel back navigation. */
  historyStack: KernelSelectionContent[] = [];
  /** Whether a deep region scope breadcrumb trail is expanded in-place. */
  scopeExpanded = false;
  /** Cached unit focus pills for the active range selection and mode. */
  unitPills: readonly UnitFocusPillItem[] = [];
  /** Cached options for the `All units` filter menu in Pillar 3. */
  unitMenuOpts: readonly UnitMenuOption[] | undefined = undefined;
  /** Cached collapsible breadcrumb view for the active selection's scope. */
  scopeView: {
    readonly head: readonly VisibleScopeCrumb[];
    readonly tail: readonly VisibleScopeCrumb[];
    readonly hiddenCount: number;
  } = {head: [], tail: [], hiddenCount: 0};
  /** Cached filtered top instructions for the active unit filter. */
  filteredTopInstructions: readonly TopInstructionItem[] = [];
  /** Cached filtered selected events for the active unit filter. */
  filteredSelectedEvents: readonly SelectedRangeEventItem[] = [];
  copied = false;

  private copiedTimer: ReturnType<typeof setTimeout> | undefined = undefined;
  private readonly changeDetectorRef = inject(ChangeDetectorRef);
  private readonly ngZone = inject(NgZone);

  private readonly eventSelectedListener = (event: Event) => {
    const detail = event instanceof CustomEvent ? event.detail : undefined;
    const nextContent = buildKernelSelectionFromEvent(detail);
    this.applyContent(nextContent);
  };

  private readonly eventsSelectedListener = (event: Event) => {
    const detail = event instanceof CustomEvent ? event.detail : undefined;
    const nextContent = buildKernelSelectionFromRange(detail);
    this.applyContent(nextContent);
  };

  ngAfterViewInit(): void {
    this.ngZone.runOutsideAngular(() => {
      window.addEventListener(
        EVENT_SELECTED_EVENT_NAME,
        this.eventSelectedListener,
      );
      window.addEventListener(
        EVENTS_SELECTED_EVENT_NAME,
        this.eventsSelectedListener,
      );
    });
  }

  ngOnDestroy(): void {
    clearTimeout(this.copiedTimer);
    window.removeEventListener(
      EVENT_SELECTED_EVENT_NAME,
      this.eventSelectedListener,
    );
    window.removeEventListener(
      EVENTS_SELECTED_EVENT_NAME,
      this.eventsSelectedListener,
    );
  }

  /** Programmatically sets or clears the active selection content. */
  setSelection(nextContent: KernelSelectionContent | undefined): void {
    this.applyContent(nextContent);
  }

  /** Closes the selection panel and notifies the parent container. */
  close(): void {
    this.historyStack = [];
    this.applyContent(undefined);
  }

  /** Toggles filtering the range instruction table by functional unit. */
  toggleUnitFilter(unitLabel: string): void {
    this.selectedUnitFilter =
      this.selectedUnitFilter === unitLabel ? undefined : unitLabel;
    this.recomputeDerivedState();
    this.changeDetectorRef.detectChanges();
  }

  /** Sets the unit filter from the `All units` dropdown menu. */
  selectUnitFilter(unitLabel: string | undefined): void {
    if (this.selectedUnitFilter !== unitLabel) {
      this.selectedUnitFilter = unitLabel;
      this.recomputeDerivedState();
      this.changeDetectorRef.detectChanges();
    }
  }

  clearUnitFilter(): void {
    if (this.selectedUnitFilter !== undefined) {
      this.selectedUnitFilter = undefined;
      this.recomputeDerivedState();
      this.changeDetectorRef.detectChanges();
    }
  }

  setUnitPillMode(mode: 'util' | 'instrs'): void {
    if (this.unitPillMode !== mode) {
      this.unitPillMode = mode;
      this.recomputeDerivedState();
      this.changeDetectorRef.detectChanges();
    }
  }

  toggleStackOpen(): void {
    this.stackOpen = !this.stackOpen;
    this.changeDetectorRef.detectChanges();
  }

  selectStackFrame(index: number): void {
    if (this.pickedStackFrameIndex !== index) {
      this.pickedStackFrameIndex = index;
      this.changeDetectorRef.detectChanges();
    }
  }

  setHoveredMixUnit(unitLabel: string | undefined): void {
    if (this.hoveredMixUnit !== unitLabel) {
      this.hoveredMixUnit = unitLabel;
      this.changeDetectorRef.detectChanges();
    }
  }

  setRangeTableTab(tab: 'top' | 'events'): void {
    if (this.rangeTableTab !== tab) {
      this.rangeTableTab = tab;
      this.recomputeDerivedState();
      this.changeDetectorRef.detectChanges();
    }
  }

  onSelectEventIndex(eventIndex: number | undefined): void {
    if (eventIndex !== undefined && eventIndex >= 0) {
      if (this.content && this.content.eventIndex !== eventIndex) {
        this.historyStack.push(this.content);
        if (this.historyStack.length > 24) {
          this.historyStack.shift();
        }
        this.changeDetectorRef.markForCheck();
      }
      this.selectEvent.emit(eventIndex);
    }
  }

  /** Navigates back to the previously inspected selection in the panel history. */
  goBack(): void {
    const previous = this.historyStack.pop();
    if (!previous) {
      return;
    }
    this.applyContent(previous);
    if (previous.eventIndex !== undefined && previous.eventIndex >= 0) {
      this.selectEvent.emit(previous.eventIndex);
    }
  }

  get canGoBack(): boolean {
    return this.historyStack.length > 0;
  }

  get previousSelectionLabel(): string {
    const last = this.historyStack[this.historyStack.length - 1];
    return last ? `Back to ${last.name}` : 'No previous selection in history';
  }

  onSearchText(query: string | undefined): void {
    if (query && query.trim() !== '') {
      this.searchQuery.emit(query.trim());
    }
  }

  unitFamily(label: string | undefined): UnitTokenFamily {
    return unitFamilyForLabel(label);
  }

  counterGroupCrumb(c: KernelSelectionContent): string {
    const kind = c.counterDetails?.kind ?? classifyCounterTrack(c.name);
    if (kind === 'bytes') {
      return 'Memory';
    }
    if (kind === 'registers') {
      return 'Registers';
    }
    return (c.isSparseCore ?? computeIsSparseCore(c))
      ? 'SparseCore'
      : 'TensorCore';
  }

  counterAtBundleTooltip(
    c: KernelSelectionContent,
    cd: CounterSelectionDetails,
  ): string {
    const delta = cd.deltaFromMeanLabel ? ` (${cd.deltaFromMeanLabel})` : '';
    return `${c.name} at bundle ${c.bundles}: ${cd.formattedCurrent}${delta}`;
  }

  counterOpMeanTooltip(c: KernelSelectionContent): string {
    const span = c.schedulePosition?.totalBundles
      ? ` · ${c.schedulePosition.totalBundles} bundles`
      : '';
    return `Mean across ${this.counterOpLabel()}${span}`;
  }

  counterOpPeakTooltip(
    c: KernelSelectionContent,
    cd: CounterSelectionDetails,
  ): string {
    const loc = cd.peakBundleLabel
      ? ` · ${cd.peakBundleLabel.replace(/^@\s*/, '')}`
      : '';
    return `Peak of ${c.name} across ${this.counterOpLabel()}${loc}`;
  }

  counterOpRangeTooltip(
    c: KernelSelectionContent,
    cd: CounterSelectionDetails,
  ): string {
    const bounds =
      cd.formattedMin && cd.formattedPeak
        ? `: ${cd.formattedMin} – ${cd.formattedPeak}`
        : '';
    const mean = cd.formattedMean ? ` (mean: ${cd.formattedMean})` : '';
    return `Range of ${c.name} across ${this.counterOpLabel()}${bounds}${mean}`;
  }

  counterOpMeanMarkerTooltip(cd: CounterSelectionDetails): string {
    const opSuffix = this.kernelName ? ` (${this.kernelName})` : '';
    return `Op mean${opSuffix}: ${cd.formattedMean ?? ''}`;
  }

  counterStepHoldTooltip(cd: CounterSelectionDetails): string {
    return `Step hold interval: ${cd.stepRangeLabel} · ${cd.stepDurationLabel}`;
  }

  counterCoverageTooltip(c: KernelSelectionContent): string {
    return `Non-zero bundle coverage of ${c.name} across ${this.counterOpLabel()}`;
  }

  private counterOpLabel(): string {
    const trimmed = this.kernelName?.trim();
    return trimmed ? `HLO op ${trimmed}` : 'this HLO op';
  }

  effectiveStackIndex(c: KernelSelectionContent): number {
    const stack = c.sourceStack;
    if (!stack || stack.length === 0) {
      return 0;
    }
    if (
      this.pickedStackFrameIndex !== undefined &&
      this.pickedStackFrameIndex >= 0 &&
      this.pickedStackFrameIndex < stack.length
    ) {
      return this.pickedStackFrameIndex;
    }
    return stack.length - 1;
  }

  activeSourceFrame(c: KernelSelectionContent): SourceFrameItem | undefined {
    const stack = c.sourceStack;
    if (
      stack &&
      this.pickedStackFrameIndex !== undefined &&
      this.pickedStackFrameIndex >= 0 &&
      this.pickedStackFrameIndex < stack.length
    ) {
      return stack[this.pickedStackFrameIndex];
    }
    return c.sourceInfo ?? stack?.[stack.length - 1];
  }

  shouldShowRangeSource(c: KernelSelectionContent): boolean {
    return Boolean(this.activeSourceFrame(c));
  }

  hasRangeUtil(c: KernelSelectionContent): boolean {
    return Boolean(
      (c.utilizationMetrics && c.utilizationMetrics.length > 0) ||
        (c.utilizationBars && c.utilizationBars.length > 0),
    );
  }

  hasRangeInstrCounts(c: KernelSelectionContent): boolean {
    return Boolean(c.instructionMix && c.instructionMix.length > 0);
  }

  hasRangeTableItems(c: KernelSelectionContent): boolean {
    return Boolean(
      (c.topInstructions && c.topInstructions.length > 0) ||
        (c.selectedEvents && c.selectedEvents.length > 0),
    );
  }

  trackByUnitFocusPill(_index: number, item: UnitFocusPillItem): string {
    return item.unit;
  }

  trackByUnitMenuOption(_index: number, opt: UnitMenuOption): string {
    return opt.unit;
  }

  trackByScopeCrumb(_index: number, item: VisibleScopeCrumb): string {
    return `${item.crumb.eventIndex ?? -1}:${item.crumb.name}`;
  }

  canSwitchUnitPillMode(c: KernelSelectionContent): boolean {
    return this.hasRangeUtil(c) && this.hasRangeInstrCounts(c);
  }

  effectiveUnitPillMode(c: KernelSelectionContent): 'util' | 'instrs' {
    if (this.canSwitchUnitPillMode(c)) {
      return this.unitPillMode;
    }
    return this.hasRangeUtil(c) ? 'util' : 'instrs';
  }

  rangeUnitPills(c: KernelSelectionContent): readonly UnitFocusPillItem[] {
    const mode = this.effectiveUnitPillMode(c);
    const baseLanes =
      (c.isSparseCore ?? computeIsSparseCore(c))
        ? SPARSE_CORE_UNIT_LANES
        : TENSOR_CORE_UNIT_LANES;
    const seen = new Set<string>(baseLanes);
    const extraUnits: string[] = [];
    if (mode === 'instrs') {
      for (const item of c.instructionMix ?? []) {
        if (item.count > 0 && !seen.has(item.unit.label)) {
          seen.add(item.unit.label);
          extraUnits.push(item.unit.label);
        }
      }
    } else {
      for (const item of c.utilizationMetrics ?? []) {
        if (
          (item.meanPercent > 0 ||
            item.peakPercent > 0 ||
            item.activeBundles > 0) &&
          !seen.has(item.unit.label)
        ) {
          seen.add(item.unit.label);
          extraUnits.push(item.unit.label);
        }
      }
      for (const item of c.utilizationBars ?? []) {
        if (item.percent > 0 && !seen.has(item.unit.label)) {
          seen.add(item.unit.label);
          extraUnits.push(item.unit.label);
        }
      }
    }
    const lanes =
      extraUnits.length > 0 ? [...baseLanes, ...extraUnits] : baseLanes;
    return lanes.map((unit) => {
      if (mode === 'instrs') {
        const mixItem = c.instructionMix?.find(
          (item) => item.unit.label === unit,
        );
        const count = mixItem?.count ?? 0;
        if (count <= 0) {
          return {unit, active: false, ariaLabel: unit};
        }
        const metric = `×${BUNDLE_FORMAT.format(count)}`;
        const ariaLabel =
          this.selectedUnitFilter === unit
            ? `${unit}, ${metric}. Show all units`
            : `${unit}, ${metric}. Show only ${unit} instructions`;
        return {unit, active: true, metric, ariaLabel};
      }

      const m = c.utilizationMetrics?.find((item) => item.unit.label === unit);
      const bar = c.utilizationBars?.find((item) => item.unit.label === unit);
      const isActive = m
        ? m.meanPercent > 0 || m.peakPercent > 0 || m.activeBundles > 0
        : (bar?.percent ?? 0) > 0;
      if (!isActive) {
        return {unit, active: false, ariaLabel: unit};
      }
      const percent = m ? m.meanPercent : (bar?.percent ?? 0);
      const metric = m
        ? m.formattedMean
        : (bar?.formattedPercent ?? `${percent}%`);
      const hoverText =
        m && m.peakPercent > 0
          ? `Peak ${m.formattedPeak}${
              m.peakBundleLabel
                ? ` at bundle ${m.peakBundleLabel.replace(/^@\s*/, '')}`
                : ''
            }`
          : undefined;
      const summary = [unit, metric, hoverText].filter(Boolean).join(', ');
      const ariaLabel =
        this.selectedUnitFilter === unit
          ? `${summary}. Show all units`
          : `${summary}. Show only ${unit} instructions`;
      return {
        unit,
        active: true,
        metric,
        ...(hoverText ? {hoverText} : {}),
        ariaLabel,
      };
    });
  }

  unitMenuOptions(
    c: KernelSelectionContent,
  ): readonly UnitMenuOption[] | undefined {
    const counts = new Map<string, number>();
    if (this.rangeTableTab === 'events' && c.selectedEvents?.length) {
      for (const ev of c.selectedEvents) {
        if (ev.trackName) {
          counts.set(ev.trackName, (counts.get(ev.trackName) ?? 0) + 1);
        }
      }
    } else if (c.topInstructions?.length) {
      for (const item of c.topInstructions) {
        if (item.trackName) {
          counts.set(item.trackName, (counts.get(item.trackName) ?? 0) + 1);
        }
      }
    }
    if (this.selectedUnitFilter && !counts.has(this.selectedUnitFilter)) {
      counts.set(this.selectedUnitFilter, 0);
    }
    if (counts.size === 0 || (counts.size < 2 && !this.selectedUnitFilter)) {
      return undefined;
    }
    const entries = Array.from(counts.entries());
    const nonScEntries = entries.filter(
      ([unit]) => !SPARSE_CORE_UNIT_LABELS.has(unit),
    );
    const scEntries = entries.filter(([unit]) =>
      SPARSE_CORE_UNIT_LABELS.has(unit),
    );
    const ordered = [...nonScEntries, ...scEntries];
    return ordered.map(([unit, count], index) => ({
      unit,
      description: UNIT_DESCRIPTIONS.get(unit) ?? '',
      count,
      dividerBefore: nonScEntries.length > 0 && index === nonScEntries.length,
    }));
  }

  expandScope(): void {
    if (!this.scopeExpanded) {
      this.scopeExpanded = true;
      this.recomputeDerivedState();
      this.changeDetectorRef.detectChanges();
    }
  }

  getScopeView(crumbs: readonly RegionBreadcrumbItem[] | undefined): {
    head: readonly VisibleScopeCrumb[];
    tail: readonly VisibleScopeCrumb[];
    hiddenCount: number;
  } {
    if (!crumbs || crumbs.length === 0) {
      return {head: [], tail: [], hiddenCount: 0};
    }
    if (crumbs.length <= 4 || this.scopeExpanded) {
      return {
        head: crumbs.map((crumb, idx) => ({
          crumb,
          isLast: idx === crumbs.length - 1,
        })),
        tail: [],
        hiddenCount: 0,
      };
    }
    return {
      head: [{crumb: crumbs[0], isLast: false}],
      tail: [
        {crumb: crumbs[crumbs.length - 2], isLast: false},
        {crumb: crumbs[crumbs.length - 1], isLast: true},
      ],
      hiddenCount: crumbs.length - 3,
    };
  }

  copySummary(): void {
    if (!this.content || !navigator.clipboard) {
      return;
    }
    const c = this.content;
    const lines: string[] = [];
    lines.push(
      `${c.badgeLabel}: ${c.name} (${c.bundlesLabel} ${c.bundles}${c.hexBundle ? ' ' + c.hexBundle : ''})`,
    );
    if (c.ordinalLabel) {
      lines.push(`Ordinal: ${c.ordinalLabel}`);
    }
    if (c.regions && c.regions.length > 0) {
      lines.push(`${c.regionRowLabel ?? 'Region'}: ${c.regions.join(' › ')}`);
    }
    if (c.output?.register) {
      lines.push(`Produces: ${c.output.register}`);
    }
    if (c.operands && c.operands.length > 0) {
      lines.push(
        `Operands: ${c.operands.map((op) => `[${op.index}] ${op.text}`).join(', ')}`,
      );
    }
    if (c.consumers && c.consumers.length > 0) {
      lines.push(`Consumers: ${c.consumers.join(', ')}`);
    }
    if (c.sourceInfo) {
      lines.push(`Source: ${c.sourceInfo.raw}`);
    }
    if (c.counterDetails) {
      lines.push(`Value: ${c.counterDetails.formattedCurrent}`);
      if (c.counterDetails.formattedMean) {
        lines.push(`Op Mean: ${c.counterDetails.formattedMean}`);
      }
      if (c.counterDetails.formattedPeak) {
        lines.push(
          `Op Peak: ${c.counterDetails.formattedPeak}${c.counterDetails.peakBundleLabel ? ' ' + c.counterDetails.peakBundleLabel : ''}`,
        );
      }
    }
    if (c.density) {
      lines.push(`Density: ${c.density}`);
    }
    navigator.clipboard.writeText(lines.join('\n')).then(
      () => {
        this.copied = true;
        clearTimeout(this.copiedTimer);
        this.copiedTimer = setTimeout(() => {
          this.copied = false;
          this.changeDetectorRef.markForCheck();
        }, 1500);
        this.changeDetectorRef.markForCheck();
      },
      () => {},
    );
  }

  private applyContent(nextContent: KernelSelectionContent | undefined): void {
    const normalized =
      nextContent &&
      (nextContent.isSparseCore === undefined ||
        nextContent.headerSwatchFamily === undefined)
        ? finalizeSelectionContent(nextContent)
        : nextContent;
    const wasOpen = this.content !== undefined;
    const isOpen = normalized !== undefined;
    if (!isOpen) {
      this.historyStack = [];
    }
    this.content = normalized;
    this.selectedUnitFilter = undefined;
    this.hoveredMixUnit = undefined;
    this.scopeExpanded = false;
    this.stackOpen = false;
    this.pickedStackFrameIndex = undefined;
    this.unitPillMode = 'util';
    this.rangeTableTab =
      normalized?.topInstructions && normalized.topInstructions.length > 0
        ? 'top'
        : 'events';
    this.copied = false;
    this.recomputeDerivedState();
    if (wasOpen !== isOpen) {
      this.openChange.emit(isOpen);
    }
    this.changeDetectorRef.detectChanges();
  }

  private recomputeDerivedState(): void {
    const c = this.content;
    if (!c) {
      this.unitPills = [];
      this.unitMenuOpts = undefined;
      this.scopeView = {head: [], tail: [], hiddenCount: 0};
      this.filteredTopInstructions = [];
      this.filteredSelectedEvents = [];
      return;
    }
    const topItems = c.topInstructions ?? [];
    const selEvents = c.selectedEvents ?? [];
    this.filteredTopInstructions = this.selectedUnitFilter
      ? topItems.filter((item) => item.trackName === this.selectedUnitFilter)
      : topItems;
    this.filteredSelectedEvents = this.selectedUnitFilter
      ? selEvents.filter((item) => item.trackName === this.selectedUnitFilter)
      : selEvents;
    this.unitPills = c.mode === 'range' ? this.rangeUnitPills(c) : [];
    this.unitMenuOpts =
      c.mode === 'range' ? this.unitMenuOptions(c) : undefined;
    this.scopeView = this.getScopeView(c.regionBreadcrumbs);
  }
}
