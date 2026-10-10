import {CommonModule} from '@angular/common';
import {
  AfterViewInit,
  ChangeDetectionStrategy,
  ChangeDetectorRef,
  Component,
  ElementRef,
  EventEmitter,
  inject,
  NgZone,
  OnDestroy,
  Output,
  ViewChild,
} from '@angular/core';
import {MatButtonModule} from '@angular/material/button';
import {MatIconModule} from '@angular/material/icon';
import {MatTooltipModule} from '@angular/material/tooltip';

/** A functional unit of a TPU core, which has its own lane in the timeline. */
export interface FunctionalUnit {
  readonly label: string;
  readonly description: string;
  readonly color: string;
}

const DEFAULT_ACCENT_COLOR = '#96c1ff';

/**
 * Trace Viewer V2 default canvas event palette (`kCatapultPalette` in
 * `trace_viewer_v2/color/palettes.h`, used as `kDefaultPalette` by the canvas).
 */
export const TRACE_VIEWER_PALETTE: readonly string[] = [
  '#ffa1a1',
  '#96c1ff',
  '#b1d284',
  '#ff80ff',
  '#80ddcc',
  '#e4b886',
  '#cc9eff',
  '#98dc95',
  '#ff92c1',
  '#85d1ff',
  '#c3ca80',
  '#ff82ff',
  '#86dfb3',
  '#f9ab94',
  '#a7b5ff',
  '#a7d789',
  '#ff85ec',
  '#80dade',
  '#d7c081',
  '#e691ff',
  '#91de9f',
  '#ff9bac',
  '#8ec8ff',
];

/** Active canvas theme colors synchronized from Trace Viewer V2 `ColorPalette`. */
export interface MinimapTheme {
  readonly paletteName: string;
  readonly background: string;
  readonly foreground: string;
  readonly midtone: string;
  readonly flameHeader: string;
  readonly collapsedHeader: string;
  readonly expandedHeader: string;
  readonly subtitle: string;
  readonly rulerText: string;
  readonly rulerLine: string;
  readonly selection: string;
  readonly onSurface: string;
  readonly inverseOnSurface: string;
  readonly traceColors: readonly string[];
}

/** Default Catapult canvas theme matching `application.cc` (`kDefaultPalette`). */
export const DEFAULT_MINIMAP_THEME: MinimapTheme = {
  paletteName: 'Catapult',
  background: '#ffffff',
  foreground: '#222222',
  midtone: '#dddddd',
  flameHeader: '#ffa1a1',
  collapsedHeader: '#cccccc',
  expandedHeader: '#aaaaaa',
  subtitle: '#555555',
  rulerText: '#222222',
  rulerLine: '#cccccc',
  selection: '#dddddd',
  onSurface: '#222222',
  inverseOnSurface: '#ffffff',
  traceColors: TRACE_VIEWER_PALETTE,
};

const FUNCTIONAL_UNITS: readonly FunctionalUnit[] = [
  {label: 'MXU', description: 'Matrix multiply unit', color: '#cc9eff'},
  {
    label: 'XLU',
    description: 'Cross-lane unit — transpose / permute / reduce',
    color: '#e4b886',
  },
  {
    label: 'SALU',
    description: 'Scalar ALU — control flow, addressing, DMA issue',
    color: '#b1d284',
  },
  {label: 'VPU', description: 'Vector ALU', color: '#96c1ff'},
  {
    label: 'EUP',
    description: 'Extended unary pipeline — exp, rcp, tanh',
    color: '#80ddcc',
  },
  {label: 'VLD', description: 'Vector load — VMEM → vregs', color: '#85d1ff'},
  {label: 'VST', description: 'Vector store — vregs → VMEM', color: '#ff92c1'},
  {label: 'DMA', description: 'DMA engine — HBM ↔ VMEM', color: '#f9ab94'},
  {
    label: 'Other',
    description: 'No dedicated functional unit',
    color: '#d7c081',
  },
];

const FUNCTIONAL_UNITS_BY_LANE: ReadonlyMap<string, FunctionalUnit> = new Map(
  FUNCTIONAL_UNITS.map((unit) => [unit.label, unit] as const),
);

const FUNCTIONAL_UNIT_INDEX_BY_LANE: ReadonlyMap<string, number> = new Map(
  FUNCTIONAL_UNITS.map((unit, idx) => [unit.label, idx] as const),
);

const BUNDLE_FORMAT = new Intl.NumberFormat(undefined, {
  maximumFractionDigits: 0,
});

/**
 * Custom event name dispatched from Trace Viewer V2 WASM when the schedule
 * minimap summary or visible bundle range changes.
 */
export const MINIMAP_UPDATED_EVENT_NAME = 'minimap_updated';

/** Minimum visible span in bundles when zooming or resizing the lens. */
const MIN_VISIBLE_BUNDLES = 2;

/** Minimum visible span in microseconds in time-axis mode (1 ns). */
const MIN_VISIBLE_TIME_US = 1e-3;

/** Minimum percentage width of the rendered viewport lens so it stays grabbable. */
const MIN_LENS_WIDTH_PERCENT = 1.2;

/** Pixel movement threshold to distinguish a click-to-center from a brush-zoom drag. */
const BRUSH_DRAG_THRESHOLD_PX = 5;

/** Default track label column width in pixels (`kDefaultLabelWidth` in C++). */
const DEFAULT_LABEL_WIDTH_PX = 250;

/** Formats a duration or relative timestamp in microseconds with optional compact units. */
export function formatMinimapTimeUs(us: number, includeUnit = true): string {
  if (!Number.isFinite(us) || Math.abs(us) < 1e-9) {
    return '0';
  }
  const abs = Math.abs(us);
  if (abs >= 1e6) {
    const num = (us / 1e6).toFixed(2).replace(/\.?0+$/, '');
    return includeUnit ? `${num} s` : num;
  }
  if (abs >= 1e3) {
    const num = (us / 1e3).toFixed(2).replace(/\.?0+$/, '');
    return includeUnit ? `${num} ms` : num;
  }
  if (abs >= 1) {
    const num = us.toFixed(2).replace(/\.?0+$/, '');
    return includeUnit ? `${num} us` : num;
  }
  if (abs >= 1e-3) {
    const num = (us * 1e3).toFixed(1).replace(/\.0$/, '');
    return includeUnit ? `${num} ns` : num;
  }
  const num = `${Math.round(us * 1e6)}`;
  return includeUnit ? `${num} ps` : num;
}

/** A single bucket in the static schedule minimap ribbon. */
export interface MinimapBin {
  readonly index: number;
  readonly unit?: FunctionalUnit;
  readonly secondaryUnit?: FunctionalUnit;
  readonly color: string;
  readonly secondaryColor?: string;
  readonly opacity: number;
  readonly density: number;
  readonly region: string;
  readonly pallasPrimitive: string;
  readonly namedScope: string;
  readonly active: boolean;
  readonly topPercent: number;
  readonly heightPercent: number;
  readonly secondaryTopPercent?: number;
  readonly secondaryHeightPercent?: number;
  readonly gapRight: boolean;
  readonly secondaryGapRight: boolean;
}

/** A major region or Pallas primitive flame bar along the schedule minimap. */
export interface MinimapRegionMarker {
  readonly eventIndex: number;
  readonly name: string;
  readonly startUs: number;
  readonly durationUs: number;
  readonly depth: number;
  readonly row: number;
  readonly topPercent: number;
  readonly heightPercent: number;
  readonly leftPercent: number;
  readonly widthPercent: number;
  readonly color: string;
  readonly textColor: string;
  readonly bundlesLabel: string;
  readonly source?: string;
}

/** A tick mark on the full-schedule bundle ruler above the minimap tracks. */
export interface MinimapRulerTick {
  readonly bundle: number;
  readonly label: string;
  readonly leftPercent: number;
  readonly isRightEdge?: boolean;
}

/** Floating hover pill state when inspecting the minimap ribbon. */
export interface MinimapHoverInfo {
  readonly leftPercent: number;
  readonly align: 'left' | 'center' | 'right';
  readonly bundleLabel: string;
  readonly unit?: FunctionalUnit;
  readonly secondaryUnit?: FunctionalUnit;
  readonly pallasPrimitive?: string;
  readonly pallasColor?: string;
  readonly region?: string;
}

/** View model for the Static Kernel Viewer and Trace Viewer schedule minimap. */
export interface KernelMinimapState {
  readonly dataStartUs: number;
  readonly dataEndUs: number;
  readonly visibleStartUs: number;
  readonly visibleEndUs: number;
  readonly labelWidthPx: number;
  readonly isTimeAxis: boolean;
  readonly visibleRangeLabel: string;
  readonly totalBundlesLabel: string;
  readonly zoomLabel: string;
  readonly isZoomed: boolean;
  readonly lensLeftPercent: number;
  readonly lensWidthPercent: number;
  readonly rightCurtainLeftPercent: number;
  readonly rightCurtainWidthPercent: number;
  readonly theme: MinimapTheme;
  readonly bins: readonly MinimapBin[];
  readonly regions: readonly MinimapRegionMarker[];
  readonly pallasPrimitives: readonly MinimapRegionMarker[];
  readonly hasPallasPrimitives: boolean;
  readonly hasBothScopeRows: boolean;
  readonly ticks: readonly MinimapRulerTick[];
  readonly showStandaloneUnitLabel: boolean;
}

type DragMode = 'pan' | 'resize-left' | 'resize-right' | 'scrub';

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max);
}

/** Computes clean 1-2-5 bundle or time tick marks across `[startUs, endUs]`. */
export function buildMinimapRulerTicks(
  startUs: number,
  endUs: number,
  isTimeAxis = false,
): MinimapRulerTick[] {
  const duration = endUs - startUs;
  if (!(duration > 0)) {
    return [];
  }
  const targetTickCount = 5;
  const rawStep = duration / targetTickCount;
  const minRawStep = isTimeAxis ? 1e-6 : 1;
  const magnitude = Math.pow(
    10,
    Math.floor(Math.log10(Math.max(minRawStep, rawStep))),
  );
  const residual = rawStep / magnitude;
  let step = magnitude;
  if (residual > 5) {
    step = 10 * magnitude;
  } else if (residual > 2.5) {
    step = 5 * magnitude;
  } else if (residual > 1.5) {
    step = 2 * magnitude;
  }
  if (!isTimeAxis) {
    step = Math.max(1, Math.round(step));
  }

  if (isTimeAxis) {
    const ticks: MinimapRulerTick[] = [];
    for (let offset = step; offset <= duration; offset += step) {
      const leftPercent = (offset / duration) * 100;
      if (leftPercent < 6 || leftPercent > 99) {
        continue;
      }
      const isRightEdge = leftPercent > 86;
      ticks.push({
        bundle: startUs + offset,
        label: formatMinimapTimeUs(offset),
        leftPercent,
        isRightEdge,
      });
    }
    return ticks;
  }

  const firstTick = Math.ceil(startUs / step) * step;
  const ticks: MinimapRulerTick[] = [];
  for (let bundle = firstTick; bundle <= endUs; bundle += step) {
    const leftPercent = ((bundle - startUs) / duration) * 100;
    if (leftPercent < 6 || leftPercent > 99) {
      continue;
    }
    const isRightEdge = leftPercent > 86;
    const formatted = BUNDLE_FORMAT.format(Math.round(bundle));
    ticks.push({
      bundle,
      label: isRightEdge ? `${formatted} bundles` : formatted,
      leftPercent,
      isRightEdge,
    });
  }
  return ticks;
}

function readThemeString(
  raw: Record<string, unknown>,
  key: string,
  fallback: string,
): string {
  const val = raw[key];
  return typeof val === 'string' && val.trim().length > 0
    ? val.trim()
    : fallback;
}

function linearSrgbChannel(value255: number): number {
  const c = value255 / 255;
  return c <= 0.03928 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
}

function hexRelativeLuminance(hex: string): number {
  const cleaned = hex.replace(/^#/, '');
  if (cleaned.length !== 6) {
    return 0.5;
  }
  const r = Number(`0x${cleaned.slice(0, 2)}`);
  const g = Number(`0x${cleaned.slice(2, 4)}`);
  const b = Number(`0x${cleaned.slice(4, 6)}`);
  if (Number.isNaN(r) || Number.isNaN(g) || Number.isNaN(b)) {
    return 0.5;
  }
  return (
    0.2126 * linearSrgbChannel(r) +
    0.7152 * linearSrgbChannel(g) +
    0.0722 * linearSrgbChannel(b)
  );
}

function getTextColorForContrast(
  bgHex: string,
  onSurfaceHex: string,
  inverseOnSurfaceHex: string,
): string {
  const bgLum = hexRelativeLuminance(bgHex);
  const onSurfaceLum = hexRelativeLuminance(onSurfaceHex);
  const contrastOnSurface =
    (Math.max(bgLum, onSurfaceLum) + 0.05) /
    (Math.min(bgLum, onSurfaceLum) + 0.05);
  return contrastOnSurface >= 4.5 ? onSurfaceHex : inverseOnSurfaceHex;
}

/** Parses optional `theme` object from `minimap_updated` event detail. */
export function parseMinimapTheme(rawTheme: unknown): MinimapTheme {
  if (!isRecord(rawTheme)) {
    return DEFAULT_MINIMAP_THEME;
  }
  const rawTraceColors = rawTheme['traceColors'];
  const traceColors = Array.isArray(rawTraceColors)
    ? rawTraceColors.filter(
        (c): c is string => typeof c === 'string' && c.trim().length > 0,
      )
    : [];
  return {
    paletteName: readThemeString(
      rawTheme,
      'paletteName',
      DEFAULT_MINIMAP_THEME.paletteName,
    ),
    background: readThemeString(
      rawTheme,
      'background',
      DEFAULT_MINIMAP_THEME.background,
    ),
    foreground: readThemeString(
      rawTheme,
      'foreground',
      DEFAULT_MINIMAP_THEME.foreground,
    ),
    midtone: readThemeString(
      rawTheme,
      'midtone',
      DEFAULT_MINIMAP_THEME.midtone,
    ),
    flameHeader: readThemeString(
      rawTheme,
      'flameHeader',
      DEFAULT_MINIMAP_THEME.flameHeader,
    ),
    collapsedHeader: readThemeString(
      rawTheme,
      'collapsedHeader',
      DEFAULT_MINIMAP_THEME.collapsedHeader,
    ),
    expandedHeader: readThemeString(
      rawTheme,
      'expandedHeader',
      DEFAULT_MINIMAP_THEME.expandedHeader,
    ),
    subtitle: readThemeString(
      rawTheme,
      'subtitle',
      DEFAULT_MINIMAP_THEME.subtitle,
    ),
    rulerText: readThemeString(
      rawTheme,
      'rulerText',
      DEFAULT_MINIMAP_THEME.rulerText,
    ),
    rulerLine: readThemeString(
      rawTheme,
      'rulerLine',
      DEFAULT_MINIMAP_THEME.rulerLine,
    ),
    selection: readThemeString(
      rawTheme,
      'selection',
      DEFAULT_MINIMAP_THEME.selection,
    ),
    onSurface: readThemeString(
      rawTheme,
      'onSurface',
      DEFAULT_MINIMAP_THEME.onSurface,
    ),
    inverseOnSurface: readThemeString(
      rawTheme,
      'inverseOnSurface',
      DEFAULT_MINIMAP_THEME.inverseOnSurface,
    ),
    traceColors:
      traceColors.length > 0 ? traceColors : DEFAULT_MINIMAP_THEME.traceColors,
  };
}

function resolveFunctionalUnit(
  unitKey: string,
  explicitColor: string | undefined,
  theme: MinimapTheme,
  fallbackIndex = 0,
): FunctionalUnit | undefined {
  if (!unitKey) {
    return undefined;
  }
  const baseUnit = FUNCTIONAL_UNITS_BY_LANE.get(unitKey);
  if (!baseUnit) {
    const palette =
      theme.traceColors.length > 0 ? theme.traceColors : TRACE_VIEWER_PALETTE;
    return {
      label: unitKey,
      description: unitKey,
      color:
        explicitColor ??
        palette[fallbackIndex % palette.length] ??
        DEFAULT_ACCENT_COLOR,
    };
  }
  if (explicitColor) {
    return {...baseUnit, color: explicitColor};
  }
  if (theme.paletteName !== 'Catapult' && theme.traceColors.length > 0) {
    const unitIdx = FUNCTIONAL_UNIT_INDEX_BY_LANE.get(unitKey) ?? 0;
    return {
      ...baseUnit,
      color: theme.traceColors[unitIdx % theme.traceColors.length],
    };
  }
  return baseUnit;
}

interface ParsedBinRecord {
  readonly unit?: FunctionalUnit;
  readonly secondaryUnit?: FunctionalUnit;
  readonly density: number;
  readonly region: string;
  readonly pallasPrimitive: string;
  readonly namedScope: string;
  readonly active: boolean;
}

const FUNCTIONAL_UNIT_ROW_INDEX: Readonly<Record<string, number>> = {
  'MXU': 0,
  'XLU': 0,
  'VPU': 1,
  'SALU': 1,
  'EUP': 1,
  'VLD': 2,
  'VST': 2,
  'DMA': 3,
  'Other': 3,
};

const FUNCTIONAL_UNIT_LANE_COUNT = 4;

/** Parses raw `bins` array from `minimap_updated` event detail. */
function parseMinimapBins(
  rawBins: unknown,
  theme: MinimapTheme = DEFAULT_MINIMAP_THEME,
): MinimapBin[] {
  if (!Array.isArray(rawBins)) {
    return [];
  }
  const customUnitOrder = new Map<string, number>();
  const getCustomUnitIndex = (key: string): number => {
    if (!key) return 0;
    const existing = customUnitOrder.get(key);
    if (existing !== undefined) return existing;
    const nextIdx = customUnitOrder.size;
    customUnitOrder.set(key, nextIdx);
    return nextIdx;
  };
  const getRowForUnit = (label: string): number => {
    const fixedRow = FUNCTIONAL_UNIT_ROW_INDEX[label];
    if (fixedRow !== undefined) {
      return fixedRow;
    }
    return getCustomUnitIndex(label) % FUNCTIONAL_UNIT_LANE_COUNT;
  };

  const parsed: ParsedBinRecord[] = [];
  for (let i = 0; i < rawBins.length; i++) {
    const item = rawBins[i];
    if (!isRecord(item)) {
      continue;
    }
    const unitKey = typeof item['unit'] === 'string' ? item['unit'] : '';
    const secondaryKey =
      typeof item['secondaryUnit'] === 'string' ? item['secondaryUnit'] : '';
    const unitColor =
      typeof item['unitColor'] === 'string' && item['unitColor']
        ? item['unitColor']
        : undefined;
    const secondaryUnitColor =
      typeof item['secondaryUnitColor'] === 'string' &&
      item['secondaryUnitColor']
        ? item['secondaryUnitColor']
        : undefined;
    const density =
      typeof item['density'] === 'number' && Number.isFinite(item['density'])
        ? clamp(item['density'], 0, 1)
        : 0;
    const region = typeof item['region'] === 'string' ? item['region'] : '';
    const pallasPrimitive =
      typeof item['pallasPrimitive'] === 'string'
        ? item['pallasPrimitive']
        : '';
    const namedScope =
      typeof item['namedScope'] === 'string' ? item['namedScope'] : '';
    const unit = resolveFunctionalUnit(
      unitKey,
      unitColor,
      theme,
      getCustomUnitIndex(unitKey),
    );
    const secondaryUnit = resolveFunctionalUnit(
      secondaryKey,
      secondaryUnitColor,
      theme,
      getCustomUnitIndex(secondaryKey),
    );
    const active = Boolean(unitKey) || density > 0;
    parsed.push({
      unit,
      secondaryUnit,
      density,
      region,
      pallasPrimitive,
      namedScope,
      active,
    });
  }

  const laneHeightPercent = 100 / FUNCTIONAL_UNIT_LANE_COUNT;
  const bins: MinimapBin[] = [];
  for (let i = 0; i < parsed.length; i++) {
    const current = parsed[i];
    const next = i + 1 < parsed.length ? parsed[i + 1] : undefined;
    const {
      unit,
      secondaryUnit,
      density,
      region,
      pallasPrimitive,
      namedScope,
      active,
    } = current;
    const color =
      unit?.color ??
      (active ? (theme.traceColors[0] ?? DEFAULT_ACCENT_COLOR) : 'transparent');
    const opacity = active ? Number((0.9 + 0.1 * density).toFixed(2)) : 0;
    const primaryRow = unit ? getRowForUnit(unit.label) : 0;
    const topPercent = primaryRow * laneHeightPercent;
    const heightPercent = active ? laneHeightPercent : 0;

    let secondaryTopPercent: number | undefined;
    let secondaryHeightPercent: number | undefined;
    if (secondaryUnit) {
      let secondaryRow = getRowForUnit(secondaryUnit.label);
      if (secondaryRow === primaryRow) {
        secondaryRow = (primaryRow + 1) % FUNCTIONAL_UNIT_LANE_COUNT;
      }
      secondaryTopPercent = secondaryRow * laneHeightPercent;
      secondaryHeightPercent = laneHeightPercent;
    }

    const nextHasPrimary =
      Boolean(unit) &&
      Boolean(next) &&
      next?.region === region &&
      next?.unit?.label === unit?.label;
    const nextHasSecondary =
      Boolean(secondaryUnit) &&
      Boolean(next) &&
      next?.region === region &&
      next?.secondaryUnit?.label === secondaryUnit?.label;

    bins.push({
      index: i,
      unit,
      secondaryUnit,
      color,
      secondaryColor: secondaryUnit?.color,
      opacity,
      density,
      region,
      pallasPrimitive,
      namedScope,
      active,
      topPercent,
      heightPercent,
      secondaryTopPercent,
      secondaryHeightPercent,
      gapRight: !nextHasPrimary,
      secondaryGapRight: !nextHasSecondary,
    });
  }
  return bins;
}

/** Parses raw `regions` or `pallasPrimitives` array from `minimap_updated` event detail. */
function parseMinimapRegions(
  rawRegions: unknown,
  dataStartUs: number,
  dataEndUs: number,
  theme: MinimapTheme = DEFAULT_MINIMAP_THEME,
  isTimeAxis = false,
): MinimapRegionMarker[] {
  const duration = dataEndUs - dataStartUs;
  if (!Array.isArray(rawRegions) || !(duration > 0)) {
    return [];
  }
  interface RawRegion {
    readonly eventIndex: number;
    readonly name: string;
    readonly startUs: number;
    readonly durationUs: number;
    readonly depth: number;
    readonly leftPercent: number;
    readonly widthPercent: number;
    readonly bundlesLabel: string;
    readonly color?: string;
    readonly textColor?: string;
    readonly source?: string;
  }
  const candidates: RawRegion[] = [];
  for (const item of rawRegions) {
    if (!isRecord(item)) {
      continue;
    }
    const eventIndex =
      typeof item['eventIndex'] === 'number' ? item['eventIndex'] : -1;
    const name = typeof item['name'] === 'string' ? item['name'] : '';
    const startUs = typeof item['startUs'] === 'number' ? item['startUs'] : NaN;
    const durationUs =
      typeof item['durationUs'] === 'number' ? item['durationUs'] : 0;
    const depth = typeof item['depth'] === 'number' ? item['depth'] : 0;
    const color =
      typeof item['color'] === 'string' && item['color']
        ? item['color']
        : undefined;
    const textColor =
      typeof item['textColor'] === 'string' && item['textColor']
        ? item['textColor']
        : undefined;
    const source =
      typeof item['source'] === 'string' && item['source']
        ? item['source']
        : undefined;
    if (eventIndex < 0 || !name || !Number.isFinite(startUs)) {
      continue;
    }
    const leftPercent = clamp(
      ((startUs - dataStartUs) / duration) * 100,
      0,
      100,
    );
    if (leftPercent >= 99.5) {
      continue;
    }
    const widthPercent = clamp(
      (durationUs / duration) * 100,
      0.8,
      100 - leftPercent,
    );
    const startBundle = Math.round(startUs);
    const endBundle = Math.round(startUs + durationUs);
    const bundlesLabel = isTimeAxis
      ? formatMinimapTimeUs(durationUs)
      : durationUs > 1
        ? `${BUNDLE_FORMAT.format(startBundle)}–${BUNDLE_FORMAT.format(endBundle)}`
        : BUNDLE_FORMAT.format(startBundle);
    candidates.push({
      eventIndex,
      name,
      startUs,
      durationUs,
      depth,
      leftPercent,
      widthPercent,
      bundlesLabel,
      color,
      textColor,
      source,
    });
  }

  // Pack regions into up to 2 flame rows so nested sub-regions sit directly
  // beneath their enclosing region, matching the viewer's flame chart levels.
  const maxRows = 2;
  const rowEndPercent = [-1, -1];
  const assignedRows: number[] = [];
  let usedRows = 1;
  for (const candidate of candidates) {
    let row = 0;
    while (
      row < maxRows - 1 &&
      candidate.leftPercent < rowEndPercent[row] - 0.2
    ) {
      row++;
    }
    rowEndPercent[row] = Math.max(
      rowEndPercent[row],
      candidate.leftPercent + candidate.widthPercent,
    );
    assignedRows.push(row);
    usedRows = Math.max(usedRows, row + 1);
  }

  const rowHeightPercent = Number((100 / usedRows).toFixed(2));
  const palette =
    theme.traceColors.length > 0 ? theme.traceColors : TRACE_VIEWER_PALETTE;
  const regions: MinimapRegionMarker[] = [];
  for (let i = 0; i < candidates.length; i++) {
    const c = candidates[i];
    const row = assignedRows[i];
    const color = c.color ?? palette[i % palette.length];
    const textColor =
      c.textColor ??
      getTextColorForContrast(color, theme.onSurface, theme.inverseOnSurface);
    regions.push({
      eventIndex: c.eventIndex,
      name: c.name,
      startUs: c.startUs,
      durationUs: c.durationUs,
      depth: c.depth,
      row,
      topPercent: Number(((row * 100) / usedRows).toFixed(2)),
      heightPercent: rowHeightPercent,
      leftPercent: c.leftPercent,
      widthPercent: c.widthPercent,
      color,
      textColor,
      bundlesLabel: c.bundlesLabel,
      source: c.source,
    });
  }
  return regions;
}

/**
 * Horizontal schedule minimap for the Static Kernel Viewer and Trace Viewer.
 *
 * Renders a full-schedule VLIW functional unit or track ribbon above the
 * timeline canvas, aligned pixel-for-pixel with the track label column and
 * timeline ruler. Supports dragging the viewport lens to pan, dragging lens
 * handles to resize/zoom, clicking or brushing outside the lens to jump/zoom,
 * mouse-wheel zooming, and double-clicking or clicking Fit to reset to the full
 * schedule.
 */
@Component({
  standalone: true,
  selector: 'kernel-minimap',
  templateUrl: './kernel_minimap.ng.html',
  styleUrls: ['./kernel_minimap.scss'],
  changeDetection: ChangeDetectionStrategy.OnPush,
  imports: [CommonModule, MatButtonModule, MatIconModule, MatTooltipModule],
})
export class KernelMinimap implements AfterViewInit, OnDestroy {
  /** Emits when the user pans, resizes, or zooms the visible bundle range. */
  @Output()
  readonly visibleRangeChange = new EventEmitter<{
    startUs: number;
    endUs: number;
  }>();

  /** Emits when the user clicks a region marker to select that region. */
  @Output() readonly selectEvent = new EventEmitter<number>();

  @ViewChild('trackElement') trackElement?: ElementRef<HTMLElement>;

  state?: KernelMinimapState;
  hoverInfo?: MinimapHoverInfo;
  isDragging = false;
  dragMode?: DragMode;

  private dataStartUs = 0;
  private dataEndUs = 0;
  private visibleStartUs = 0;
  private visibleEndUs = 0;
  private labelWidthPx = DEFAULT_LABEL_WIDTH_PX;
  private isTimeAxis = false;
  private cachedTheme: MinimapTheme = DEFAULT_MINIMAP_THEME;
  private cachedBins: readonly MinimapBin[] = [];
  private cachedRegions: readonly MinimapRegionMarker[] = [];
  private cachedPallasPrimitives: readonly MinimapRegionMarker[] = [];
  private cachedTicks: readonly MinimapRulerTick[] = [];

  private dragStartClientX = 0;
  private dragTrackLeftPx = 0;
  private dragTrackWidthPx = 1;
  private dragInitialStartUs = 0;
  private dragInitialEndUs = 0;
  private scrubAnchorUs = 0;

  private readonly changeDetectorRef = inject(ChangeDetectorRef);
  private readonly ngZone = inject(NgZone);

  private readonly minimapUpdatedListener = (event: Event) => {
    this.onMinimapUpdated(event);
  };
  private readonly windowMouseMoveListener = (event: MouseEvent) => {
    this.onWindowMouseMove(event);
  };
  private readonly windowMouseUpListener = (event: MouseEvent) => {
    this.onWindowMouseUp(event);
  };

  ngAfterViewInit(): void {
    this.ngZone.runOutsideAngular(() => {
      window.addEventListener(
        MINIMAP_UPDATED_EVENT_NAME,
        this.minimapUpdatedListener,
      );
    });
  }

  ngOnDestroy(): void {
    window.removeEventListener(
      MINIMAP_UPDATED_EVENT_NAME,
      this.minimapUpdatedListener,
    );
    this.stopDragging();
  }

  /** Resets the visible range to cover the full static kernel schedule. */
  resetZoom(event?: MouseEvent): void {
    event?.stopPropagation();
    if (!(this.dataEndUs > this.dataStartUs)) {
      return;
    }
    this.applyAndEmitVisibleRange(this.dataStartUs, this.dataEndUs);
  }

  private getMinVisibleSpan(): number {
    return this.isTimeAxis ? MIN_VISIBLE_TIME_US : MIN_VISIBLE_BUNDLES;
  }

  /** Zooms in or out by a step factor centered on the current viewport. */
  stepZoom(factor: number, event?: MouseEvent): void {
    event?.stopPropagation();
    const totalDuration = this.dataEndUs - this.dataStartUs;
    const currentDuration = this.visibleEndUs - this.visibleStartUs;
    if (!(totalDuration > 0) || !(currentDuration > 0)) {
      return;
    }
    const center = (this.visibleStartUs + this.visibleEndUs) * 0.5;
    const newDuration = clamp(
      currentDuration * factor,
      Math.min(this.getMinVisibleSpan(), totalDuration),
      totalDuration,
    );
    let newStart = center - newDuration * 0.5;
    let newEnd = newStart + newDuration;
    if (newStart < this.dataStartUs) {
      newStart = this.dataStartUs;
      newEnd = newStart + newDuration;
    }
    if (newEnd > this.dataEndUs) {
      newEnd = this.dataEndUs;
      newStart = Math.max(this.dataStartUs, newEnd - newDuration);
    }
    this.applyAndEmitVisibleRange(newStart, newEnd);
  }

  onLensMouseDown(event: MouseEvent): void {
    if (event.button !== 0) {
      return;
    }
    event.preventDefault();
    event.stopPropagation();
    this.startDragging('pan', event);
  }

  onLeftHandleMouseDown(event: MouseEvent): void {
    if (event.button !== 0) {
      return;
    }
    event.preventDefault();
    event.stopPropagation();
    this.startDragging('resize-left', event);
  }

  onRightHandleMouseDown(event: MouseEvent): void {
    if (event.button !== 0) {
      return;
    }
    event.preventDefault();
    event.stopPropagation();
    this.startDragging('resize-right', event);
  }

  onTrackMouseDown(event: MouseEvent): void {
    if (event.button !== 0 || !(this.dataEndUs > this.dataStartUs)) {
      return;
    }
    event.preventDefault();
    this.startDragging('scrub', event);
    this.scrubAnchorUs = this.clientXToBundle(event.clientX);
  }

  onTrackDoubleClick(event: MouseEvent): void {
    event.preventDefault();
    event.stopPropagation();
    this.resetZoom();
  }

  onTrackWheel(event: WheelEvent): void {
    const totalDuration = this.dataEndUs - this.dataStartUs;
    const currentDuration = this.visibleEndUs - this.visibleStartUs;
    if (!(totalDuration > 0) || !(currentDuration > 0)) {
      return;
    }
    event.preventDefault();
    event.stopPropagation();

    const rect = this.trackElement?.nativeElement.getBoundingClientRect();
    const trackWidth = Math.max(1, rect?.width ?? 1);
    const pointerRatio = rect
      ? clamp((event.clientX - rect.left) / trackWidth, 0, 1)
      : 0.5;

    // Horizontal wheel or Shift+wheel pans the viewport.
    if (
      Math.abs(event.deltaX) > Math.abs(event.deltaY) ||
      (event.shiftKey && event.deltaY !== 0)
    ) {
      const deltaPx =
        Math.abs(event.deltaX) > Math.abs(event.deltaY)
          ? event.deltaX
          : event.deltaY;
      const deltaBundles = (deltaPx / trackWidth) * currentDuration * 0.5;
      const shift = clamp(
        deltaBundles,
        this.dataStartUs - this.visibleStartUs,
        this.dataEndUs - this.visibleEndUs,
      );
      this.applyAndEmitVisibleRange(
        this.visibleStartUs + shift,
        this.visibleEndUs + shift,
      );
      return;
    }

    if (event.deltaY === 0) {
      return;
    }
    const zoomFactor = event.deltaY < 0 ? 0.8 : 1.25;
    const newDuration = clamp(
      currentDuration * zoomFactor,
      Math.min(this.getMinVisibleSpan(), totalDuration),
      totalDuration,
    );
    const anchorBundle = this.dataStartUs + pointerRatio * totalDuration;
    // Keep the hovered point stationary relative to the new window.
    const ratioWithinLens = clamp(
      (anchorBundle - this.visibleStartUs) / currentDuration,
      0,
      1,
    );
    let newStart = anchorBundle - ratioWithinLens * newDuration;
    let newEnd = newStart + newDuration;
    if (newStart < this.dataStartUs) {
      newStart = this.dataStartUs;
      newEnd = newStart + newDuration;
    }
    if (newEnd > this.dataEndUs) {
      newEnd = this.dataEndUs;
      newStart = Math.max(this.dataStartUs, newEnd - newDuration);
    }
    this.applyAndEmitVisibleRange(newStart, newEnd);
  }

  onTrackMouseMove(event: MouseEvent): void {
    if (this.isDragging || !this.state) {
      return;
    }
    this.updateHoverInfo(event.clientX);
  }

  onTrackMouseLeave(): void {
    if (this.hoverInfo !== undefined) {
      this.hoverInfo = undefined;
      this.changeDetectorRef.markForCheck();
    }
  }

  onRegionMarkerClick(marker: MinimapRegionMarker, event: MouseEvent): void {
    event.preventDefault();
    event.stopPropagation();
    this.selectEvent.emit(marker.eventIndex);
  }

  trackByBinIndex(index: number, bin: MinimapBin): number {
    return bin.index;
  }

  trackByRegionIndex(index: number, marker: MinimapRegionMarker): number {
    return marker.eventIndex;
  }

  trackByTickBundle(index: number, tick: MinimapRulerTick): number {
    return tick.bundle;
  }

  private onMinimapUpdated(event: Event): void {
    if (!(event instanceof CustomEvent) || !isRecord(event.detail)) {
      return;
    }
    const detail = event.detail;
    const dataStartUs =
      typeof detail['dataStartUs'] === 'number' ? detail['dataStartUs'] : NaN;
    const dataEndUs =
      typeof detail['dataEndUs'] === 'number' ? detail['dataEndUs'] : NaN;
    const visibleStartUs =
      typeof detail['visibleStartUs'] === 'number'
        ? detail['visibleStartUs']
        : NaN;
    const visibleEndUs =
      typeof detail['visibleEndUs'] === 'number' ? detail['visibleEndUs'] : NaN;

    if (
      !Number.isFinite(dataStartUs) ||
      !Number.isFinite(dataEndUs) ||
      !(dataEndUs > dataStartUs) ||
      !Number.isFinite(visibleStartUs) ||
      !Number.isFinite(visibleEndUs)
    ) {
      if (this.state !== undefined) {
        this.ngZone.run(() => {
          this.state = undefined;
          this.hoverInfo = undefined;
          this.changeDetectorRef.markForCheck();
        });
      }
      return;
    }

    const nextIsTimeAxis =
      typeof detail['timeAxisUnit'] === 'string'
        ? detail['timeAxisUnit'] === 'time'
        : this.isTimeAxis;
    const unitChanged = nextIsTimeAxis !== this.isTimeAxis;
    this.isTimeAxis = nextIsTimeAxis;

    const boundsChanged =
      dataStartUs !== this.dataStartUs ||
      dataEndUs !== this.dataEndUs ||
      unitChanged;
    this.dataStartUs = dataStartUs;
    this.dataEndUs = dataEndUs;
    this.visibleStartUs = clamp(visibleStartUs, dataStartUs, dataEndUs);
    this.visibleEndUs = clamp(
      Math.max(this.visibleStartUs, visibleEndUs),
      dataStartUs,
      dataEndUs,
    );
    if (
      typeof detail['labelWidthPx'] === 'number' &&
      detail['labelWidthPx'] > 0
    ) {
      this.labelWidthPx = Math.round(detail['labelWidthPx']);
    }

    if (isRecord(detail['theme'])) {
      this.cachedTheme = parseMinimapTheme(detail['theme']);
    }

    if (Array.isArray(detail['bins'])) {
      this.cachedBins = parseMinimapBins(detail['bins'], this.cachedTheme);
    }
    if (Array.isArray(detail['regions'])) {
      this.cachedRegions = parseMinimapRegions(
        detail['regions'],
        dataStartUs,
        dataEndUs,
        this.cachedTheme,
        this.isTimeAxis,
      );
    } else if (boundsChanged && this.cachedRegions.length > 0) {
      this.cachedRegions = parseMinimapRegions(
        this.cachedRegions,
        dataStartUs,
        dataEndUs,
        this.cachedTheme,
        this.isTimeAxis,
      );
    }
    if (Array.isArray(detail['pallasPrimitives'])) {
      this.cachedPallasPrimitives = parseMinimapRegions(
        detail['pallasPrimitives'],
        dataStartUs,
        dataEndUs,
        this.cachedTheme,
        this.isTimeAxis,
      );
    } else if (boundsChanged && this.cachedPallasPrimitives.length > 0) {
      this.cachedPallasPrimitives = parseMinimapRegions(
        this.cachedPallasPrimitives,
        dataStartUs,
        dataEndUs,
        this.cachedTheme,
        this.isTimeAxis,
      );
    }
    if (boundsChanged || this.cachedTicks.length === 0) {
      this.cachedTicks = buildMinimapRulerTicks(
        dataStartUs,
        dataEndUs,
        this.isTimeAxis,
      );
    }

    this.ngZone.run(() => {
      this.rebuildViewModel();
      this.changeDetectorRef.markForCheck();
    });
  }

  private startDragging(mode: DragMode, event: MouseEvent): void {
    const rect = this.trackElement?.nativeElement.getBoundingClientRect();
    this.dragTrackLeftPx = rect?.left ?? 0;
    this.dragTrackWidthPx = Math.max(1, rect?.width ?? 1);
    this.dragStartClientX = event.clientX;
    this.dragInitialStartUs = this.visibleStartUs;
    this.dragInitialEndUs = this.visibleEndUs;
    this.isDragging = true;
    this.dragMode = mode;
    this.hoverInfo = undefined;

    this.ngZone.runOutsideAngular(() => {
      window.addEventListener('mousemove', this.windowMouseMoveListener);
      window.addEventListener('mouseup', this.windowMouseUpListener);
    });
    this.changeDetectorRef.markForCheck();
  }

  private onWindowMouseMove(event: MouseEvent): void {
    if (!this.isDragging || !this.dragMode) {
      return;
    }
    const totalDuration = this.dataEndUs - this.dataStartUs;
    if (!(totalDuration > 0)) {
      return;
    }
    const deltaPx = event.clientX - this.dragStartClientX;
    const deltaBundles = (deltaPx / this.dragTrackWidthPx) * totalDuration;
    const minSpan = Math.min(this.getMinVisibleSpan(), totalDuration);

    if (this.dragMode === 'pan') {
      const span = this.dragInitialEndUs - this.dragInitialStartUs;
      const clampedShift = clamp(
        deltaBundles,
        this.dataStartUs - this.dragInitialStartUs,
        this.dataEndUs - this.dragInitialEndUs,
      );
      this.ngZone.run(() => {
        this.applyAndEmitVisibleRange(
          this.dragInitialStartUs + clampedShift,
          this.dragInitialStartUs + clampedShift + span,
        );
      });
    } else if (this.dragMode === 'resize-left') {
      const newStart = clamp(
        this.dragInitialStartUs + deltaBundles,
        this.dataStartUs,
        this.dragInitialEndUs - minSpan,
      );
      this.ngZone.run(() => {
        this.applyAndEmitVisibleRange(newStart, this.dragInitialEndUs);
      });
    } else if (this.dragMode === 'resize-right') {
      const newEnd = clamp(
        this.dragInitialEndUs + deltaBundles,
        this.dragInitialStartUs + minSpan,
        this.dataEndUs,
      );
      this.ngZone.run(() => {
        this.applyAndEmitVisibleRange(this.dragInitialStartUs, newEnd);
      });
    } else if (this.dragMode === 'scrub') {
      if (Math.abs(deltaPx) >= BRUSH_DRAG_THRESHOLD_PX) {
        const currentUs = this.clientXToBundle(event.clientX);
        const startUs = clamp(
          Math.min(this.scrubAnchorUs, currentUs),
          this.dataStartUs,
          this.dataEndUs - minSpan,
        );
        const endUs = clamp(
          Math.max(this.scrubAnchorUs, currentUs),
          startUs + minSpan,
          this.dataEndUs,
        );
        this.ngZone.run(() => {
          this.applyAndEmitVisibleRange(startUs, endUs);
        });
      }
    }
  }

  private onWindowMouseUp(event: MouseEvent): void {
    if (!this.isDragging) {
      return;
    }
    const mode = this.dragMode;
    const deltaPx = Math.abs(event.clientX - this.dragStartClientX);
    this.stopDragging();

    this.ngZone.run(() => {
      if (mode === 'scrub' && deltaPx < BRUSH_DRAG_THRESHOLD_PX) {
        // Clicking outside the lens centers the current viewport window at the
        // clicked bundle.
        const clickedBundle = this.clientXToBundle(event.clientX);
        const span = this.dragInitialEndUs - this.dragInitialStartUs;
        const totalDuration = this.dataEndUs - this.dataStartUs;
        if (span > 0 && totalDuration > 0) {
          const newStart = clamp(
            clickedBundle - span * 0.5,
            this.dataStartUs,
            Math.max(this.dataStartUs, this.dataEndUs - span),
          );
          this.applyAndEmitVisibleRange(
            newStart,
            Math.min(this.dataEndUs, newStart + span),
          );
        }
      }
      this.changeDetectorRef.markForCheck();
    });
  }

  private stopDragging(): void {
    this.isDragging = false;
    this.dragMode = undefined;
    window.removeEventListener('mousemove', this.windowMouseMoveListener);
    window.removeEventListener('mouseup', this.windowMouseUpListener);
  }

  private clientXToBundle(clientX: number): number {
    const ratio = clamp(
      (clientX - this.dragTrackLeftPx) / this.dragTrackWidthPx,
      0,
      1,
    );
    return this.dataStartUs + ratio * (this.dataEndUs - this.dataStartUs);
  }

  private updateHoverInfo(clientX: number): void {
    const rect = this.trackElement?.nativeElement.getBoundingClientRect();
    const totalDuration = this.dataEndUs - this.dataStartUs;
    if (!rect || !(rect.width > 0) || !(totalDuration > 0)) {
      return;
    }
    const ratio = clamp((clientX - rect.left) / rect.width, 0, 1);
    const leftPercent = ratio * 100;
    const hoveredUs = this.dataStartUs + ratio * totalDuration;
    const bundle = Math.round(hoveredUs);
    const binIndex =
      this.cachedBins.length > 0
        ? clamp(
            Math.floor(ratio * this.cachedBins.length),
            0,
            this.cachedBins.length - 1,
          )
        : -1;
    const bin = binIndex >= 0 ? this.cachedBins[binIndex] : undefined;
    const align: MinimapHoverInfo['align'] =
      leftPercent < 14 ? 'left' : leftPercent > 86 ? 'right' : 'center';
    const pallasPrimitive = bin?.pallasPrimitive || undefined;
    const pallasColor = pallasPrimitive
      ? (this.cachedPallasPrimitives.find(
          (m) =>
            m.name === pallasPrimitive &&
            hoveredUs >= m.startUs &&
            hoveredUs <= m.startUs + m.durationUs,
        )?.color ??
        this.cachedPallasPrimitives.find((m) => m.name === pallasPrimitive)
          ?.color ??
        this.cachedTheme.traceColors[2] ??
        '#b1d284')
      : undefined;

    this.hoverInfo = {
      leftPercent,
      align,
      bundleLabel: this.isTimeAxis
        ? formatMinimapTimeUs(hoveredUs - this.dataStartUs)
        : BUNDLE_FORMAT.format(bundle),
      unit: bin?.unit,
      secondaryUnit: bin?.secondaryUnit,
      region: bin?.region || undefined,
      pallasPrimitive,
      pallasColor,
    };
    this.changeDetectorRef.markForCheck();
  }

  private applyAndEmitVisibleRange(startUs: number, endUs: number): void {
    const clampedStart = clamp(startUs, this.dataStartUs, this.dataEndUs);
    const clampedEnd = clamp(
      Math.max(clampedStart, endUs),
      this.dataStartUs,
      this.dataEndUs,
    );
    this.visibleStartUs = clampedStart;
    this.visibleEndUs = clampedEnd;
    this.rebuildViewModel();
    this.visibleRangeChange.emit({startUs: clampedStart, endUs: clampedEnd});
    this.changeDetectorRef.markForCheck();
  }

  private rebuildViewModel(): void {
    const totalDuration = this.dataEndUs - this.dataStartUs;
    if (!(totalDuration > 0)) {
      this.state = undefined;
      return;
    }
    const visibleDuration = Math.max(
      0,
      this.visibleEndUs - this.visibleStartUs,
    );
    const rawLeftPercent = clamp(
      ((this.visibleStartUs - this.dataStartUs) / totalDuration) * 100,
      0,
      100,
    );
    const rawWidthPercent = clamp(
      (visibleDuration / totalDuration) * 100,
      0,
      100,
    );
    const lensWidthPercent = Math.max(MIN_LENS_WIDTH_PERCENT, rawWidthPercent);
    const lensLeftPercent = clamp(
      rawLeftPercent,
      0,
      Math.max(0, 100 - lensWidthPercent),
    );
    const rightCurtainLeftPercent = clamp(
      lensLeftPercent + lensWidthPercent,
      0,
      100,
    );
    const rightCurtainWidthPercent = Math.max(0, 100 - rightCurtainLeftPercent);

    const zoomFactor =
      visibleDuration > 0 ? totalDuration / visibleDuration : 1;
    const isZoomed = zoomFactor > 1.02;
    const zoomLabel =
      zoomFactor >= 10
        ? `${Math.round(zoomFactor)}×`
        : `${zoomFactor.toFixed(1).replace(/\.0$/, '')}×`;

    const visibleRangeLabel = this.isTimeAxis
      ? `${formatMinimapTimeUs(
          this.visibleStartUs - this.dataStartUs,
          false,
        )}–${formatMinimapTimeUs(this.visibleEndUs - this.dataStartUs, false)}`
      : `${BUNDLE_FORMAT.format(
          Math.round(this.visibleStartUs),
        )}–${BUNDLE_FORMAT.format(Math.round(this.visibleEndUs))}`;
    const totalBundlesLabel = this.isTimeAxis
      ? formatMinimapTimeUs(totalDuration, false)
      : BUNDLE_FORMAT.format(Math.round(totalDuration));

    const showStandaloneUnitLabel =
      !this.isTimeAxis && !this.cachedTicks.some((tick) => tick.isRightEdge);

    this.state = {
      dataStartUs: this.dataStartUs,
      dataEndUs: this.dataEndUs,
      visibleStartUs: this.visibleStartUs,
      visibleEndUs: this.visibleEndUs,
      labelWidthPx: this.labelWidthPx,
      isTimeAxis: this.isTimeAxis,
      visibleRangeLabel,
      totalBundlesLabel,
      zoomLabel,
      isZoomed,
      lensLeftPercent,
      lensWidthPercent,
      rightCurtainLeftPercent,
      rightCurtainWidthPercent,
      theme: this.cachedTheme,
      bins: this.cachedBins,
      regions: this.cachedRegions,
      pallasPrimitives: this.cachedPallasPrimitives,
      hasPallasPrimitives: this.cachedPallasPrimitives.length > 0,
      hasBothScopeRows:
        this.cachedPallasPrimitives.length > 0 && this.cachedRegions.length > 0,
      ticks: this.cachedTicks,
      showStandaloneUnitLabel,
    };
  }
}
