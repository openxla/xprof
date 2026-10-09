import {CommonModule} from '@angular/common';
import {
  AfterViewInit,
  ChangeDetectionStrategy,
  ChangeDetectorRef,
  Component,
  EventEmitter,
  inject,
  NgZone,
  OnDestroy,
  Output,
} from '@angular/core';
import {MatButtonModule} from '@angular/material/button';
import {MatIconModule} from '@angular/material/icon';
import {MatTooltipModule} from '@angular/material/tooltip';
import {
  buildInstructionMix,
  buildKernelEventTooltip,
  buildUtilizationBars,
  BUNDLE_FORMAT,
  classifyRegister,
  DEFAULT_ACCENT_COLOR,
  DENSITY_FORMAT,
  FUNCTIONAL_UNITS_BY_LANE,
  readHoveredEvent,
  readNumberRecord,
  type BundleLaneCell,
  type CounterGauge,
  type FunctionalUnit,
  type InstructionMixItem,
  type InstructionOutput,
  type OperandItem,
  type RegionUtilizationSummary,
  type SchedulePosition,
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
  /** Full raw location string, e.g. `learning/gemini/.../megablocks_gmm.py:627:5`. */
  readonly raw: string;
  /** File basename with `:line[:col]` suffix, e.g. `megablocks_gmm.py:627:5`. */
  readonly shortLocation: string;
  /** Parent directory path without trailing slash, e.g. `learning/gemini/.../kernels`. */
  readonly directory?: string;
  /** 0-based index in the call stack (0 = outermost caller, last = innermost callee). */
  readonly index: number;
  /** True when this frame is the innermost leaf frame where the op was emitted. */
  readonly isInnermost: boolean;
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
  readonly spillsLabel?: string;
  readonly hasSpills?: boolean;
  readonly sourceInfo?: SourceFrameItem;
  readonly sourceStack?: readonly SourceFrameItem[];
  readonly extraAttributes?: readonly ExtraAttributeItem[];
  readonly bundleStrip?: readonly BundleLaneCell[];
  readonly activeLanesSummary?: string;
  readonly bundleEvents?: readonly BundleEventItem[];
  readonly utilizationBars?: readonly UtilizationBar[];
  readonly peakUtilizationLabel?: string;
  readonly counterGauge?: CounterGauge;
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
  other: 'Register',
};

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
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
  for (const entry of raw) {
    if (!isRecord(entry)) continue;
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
  for (const entry of raw) {
    if (!isRecord(entry)) continue;
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
  for (const entry of raw) {
    if (!isRecord(entry)) continue;
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
  for (const entry of raw) {
    if (!isRecord(entry)) continue;
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

/**
 * Builds the bottom selection panel content from an `eventselected` detail
 * payload. Returns `undefined` when deselected (`eventIndex === -1` and not a
 * counter track selection).
 */
export function buildKernelSelectionFromEvent(
  detail: unknown,
): KernelSelectionContent | undefined {
  if (!isRecord(detail)) {
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

  const isRegion = tooltip.kind === 'region';
  const mode: KernelSelectionContent['mode'] = isRegion ? 'range' : 'single';
  const badgeLabel = isRegion
    ? 'Region'
    : tooltip.kind === 'counter'
      ? 'Counter'
      : (tooltip.unit?.label ?? 'Instruction');
  const badgeIcon = isRegion
    ? 'account_tree'
    : tooltip.kind === 'counter'
      ? 'monitoring'
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
  const activeLanes = tooltip.bundleStrip
    ? tooltip.bundleStrip.filter((cell) => cell.active).length
    : 0;
  const totalOpsInBundle = tooltip.bundleStrip
    ? tooltip.bundleStrip.reduce((sum, cell) => sum + cell.count, 0)
    : (bundleEvents?.length ?? 0);
  const activeLanesSummary =
    totalOpsInBundle > 0 && tooltip.bundleStrip
      ? `${totalOpsInBundle} ${totalOpsInBundle === 1 ? 'op' : 'ops'} · ${activeLanes}/${tooltip.bundleStrip.length} lanes active`
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
  const regionAncestors = readChildRegions(detail['regionAncestors']);
  const regionBreadcrumbs = buildRegionBreadcrumbs(
    tooltip.regions,
    regionAncestors,
  );
  const parentRegion =
    regionAncestors && regionAncestors.length > 0
      ? regionAncestors[regionAncestors.length - 1]
      : undefined;
  const childRegions = readChildRegions(detail['childRegions']);
  const topInstructions = readTopInstructions(
    detail['topInstructions'],
    totalInstructions,
  );

  const subtitleChips = [...(tooltip.subtitleChips ?? [])];
  if (isRegion && totalInstructions > 0) {
    subtitleChips.push(
      `${BUNDLE_FORMAT.format(totalInstructions)} instructions`,
    );
  }

  return {
    mode,
    kind: tooltip.kind,
    name: tooltip.name,
    isCode: tooltip.isCode,
    ...(tooltip.unit ? {unit: tooltip.unit} : {}),
    accentColor: tooltip.accentColor,
    badgeLabel,
    badgeIcon,
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
    ...(tooltip.spillsLabel
      ? {spillsLabel: tooltip.spillsLabel, hasSpills: tooltip.hasSpills}
      : {}),
    ...(sourceInfo ? {sourceInfo} : {}),
    ...(sourceStack ? {sourceStack} : {}),
    ...(extraAttributes.length > 0 ? {extraAttributes} : {}),
    ...(tooltip.bundleStrip ? {bundleStrip: tooltip.bundleStrip} : {}),
    ...(activeLanesSummary ? {activeLanesSummary} : {}),
    ...(bundleEvents ? {bundleEvents} : {}),
    ...(utilizationBars ? {utilizationBars} : {}),
    ...(peakUtilizationLabel ? {peakUtilizationLabel} : {}),
    ...(tooltip.counterGauge ? {counterGauge: tooltip.counterGauge} : {}),
    ...(tooltip.regions
      ? {regions: tooltip.regions, regionRowLabel: tooltip.regionRowLabel}
      : {}),
    ...(regionBreadcrumbs ? {regionBreadcrumbs} : {}),
    ...(parentRegion && parentRegion.eventIndex >= 0 ? {parentRegion} : {}),
    ...(tooltip.schedulePosition
      ? {schedulePosition: tooltip.schedulePosition}
      : {}),
    ...(tooltip.instructionMix ? {instructionMix: tooltip.instructionMix} : {}),
    totalInstructions,
    ...(tooltip.meanUtilization
      ? {meanUtilization: tooltip.meanUtilization}
      : {}),
    ...(tooltip.density ? {density: tooltip.density} : {}),
    ...(childRegions ? {childRegions} : {}),
    ...(topInstructions ? {topInstructions} : {}),
  };
}

/**
 * Builds the bottom selection panel content from an `events_selected`
 * multi-event rectangle selection payload. Returns `undefined` when cleared.
 */
export function buildKernelSelectionFromRange(
  detail: unknown,
): KernelSelectionContent | undefined {
  if (!isRecord(detail)) {
    return undefined;
  }

  let parsedJson: Record<string, unknown> | undefined = undefined;
  if (
    typeof detail['events_selected_data'] === 'string' &&
    detail['events_selected_data'].trim() !== ''
  ) {
    try {
      const parsed: unknown = JSON.parse(detail['events_selected_data']);
      if (isRecord(parsed)) {
        parsedJson = parsed;
      }
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

  const bundleCounts = readNumberRecord(detail['bundleCounts']);
  const utilization = readNumberRecord(detail['utilization']);
  const mixResult = buildInstructionMix(bundleCounts);
  const utilizationBars = buildUtilizationBars(utilization);

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
    !parsedJson?.['metrics']
  ) {
    return undefined;
  }

  const topInstructions =
    readTopInstructions(detail['topInstructions'], totalInstructions) ??
    readTopInstructions(parsedJson?.['metrics'], totalInstructions);

  const rangeStart = start ?? 0;
  const rangeExtent = extent && extent > 0 ? extent : 1;
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
    const top = utilizationBars[0];
    meanUtilization = {
      items: utilizationBars,
      ...(top.percent >= 20 ? {boundUnit: top.unit.label} : {}),
    };
  }

  const density =
    totalInstructions > 0 && rangeExtent > 0
      ? `${DENSITY_FORMAT.format(totalInstructions / rangeExtent)} instr / bundle`
      : undefined;

  const dominantUnit = meanUtilization?.boundUnit
    ? FUNCTIONAL_UNITS_BY_LANE.get(meanUtilization.boundUnit)
    : mixResult.mix?.[0]?.unit;
  const accentColor = dominantUnit?.color ?? DEFAULT_ACCENT_COLOR;

  return {
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
    ...(density ? {density} : {}),
    ...(topInstructions ? {topInstructions} : {}),
    ...(selectedEvents ? {selectedEvents} : {}),
  };
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
  imports: [CommonModule, MatButtonModule, MatIconModule, MatTooltipModule],
})
export class KernelSelectionPanel implements AfterViewInit, OnDestroy {
  /** Emits true when a selection is active, false when closed or deselected. */
  @Output() readonly openChange = new EventEmitter<boolean>();
  /** Emits an `eventIndex` to select and reveal on the timeline. */
  @Output() readonly selectEvent = new EventEmitter<number>();
  /** Emits a search query to highlight matching events on the timeline. */
  @Output() readonly searchQuery = new EventEmitter<string>();

  content: KernelSelectionContent | undefined = undefined;
  /** Optional unit label filter applied to the Top Instructions / Events table. */
  selectedUnitFilter: string | undefined = undefined;
  /** Currently hovered functional unit in the Instruction Mix stacked bar or list. */
  hoveredMixUnit: string | undefined = undefined;
  /** Tab mode for the right-hand table in range selections (`top` or `events`). */
  rangeTableTab: 'top' | 'events' = 'top';
  /** Stack of previously inspected selections for in-panel back navigation. */
  historyStack: KernelSelectionContent[] = [];
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
    this.changeDetectorRef.detectChanges();
  }

  clearUnitFilter(): void {
    if (this.selectedUnitFilter !== undefined) {
      this.selectedUnitFilter = undefined;
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
    this.rangeTableTab = tab;
    this.changeDetectorRef.detectChanges();
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

  get filteredTopInstructions(): readonly TopInstructionItem[] {
    const items = this.content?.topInstructions ?? [];
    if (!this.selectedUnitFilter) {
      return items;
    }
    return items.filter((item) => item.trackName === this.selectedUnitFilter);
  }

  get filteredSelectedEvents(): readonly SelectedRangeEventItem[] {
    const items = this.content?.selectedEvents ?? [];
    if (!this.selectedUnitFilter) {
      return items;
    }
    return items.filter((item) => item.trackName === this.selectedUnitFilter);
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
    const wasOpen = this.content !== undefined;
    const isOpen = nextContent !== undefined;
    if (!isOpen) {
      this.historyStack = [];
    }
    this.content = nextContent;
    this.selectedUnitFilter = undefined;
    this.hoveredMixUnit = undefined;
    this.rangeTableTab =
      nextContent?.topInstructions && nextContent.topInstructions.length > 0
        ? 'top'
        : 'events';
    this.copied = false;
    if (wasOpen !== isOpen) {
      this.openChange.emit(isOpen);
    }
    this.changeDetectorRef.detectChanges();
  }
}
