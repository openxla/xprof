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

/** Height of the bundle/time ruler above the minimap track. */
const RULER_HEIGHT_PX = 16;

/** Track height with at most one scope section (Regions or Pallas). */
const TRACK_HEIGHT_PX = 51;

/** Track height when both scope sections are shown. */
const TRACK_HEIGHT_TWO_SCOPES_PX = 65;

/** Height of the activity density strip at the top of the track. */
const STRIP_HEIGHT_PX = 5;

/** Vertical gap between the strip, scope sections, and activity lanes. */
const BAND_GAP_PX = 2;

/**
 * Height of one mirrored flame level: a 4px bar plus a 1px gap, echoing the
 * level padding of the canvas.
 */
const SCOPE_ROW_HEIGHT_PX = 5;

/** Height of a single-level scope section, tall enough for inline labels. */
const SCOPE_SINGLE_ROW_HEIGHT_PX = 12;

/** Maximum mirrored flame levels per scope section (`kMaxMinimapScopeRows`). */
const MAX_SCOPE_ROWS = 3;

/** Scope markers at least this wide (in % of the track) join the tab order. */
const MIN_FOCUSABLE_MARKER_WIDTH_PERCENT = 1.5;

/** Hover tolerance (in % of the track) for hitting very narrow markers. */
const MARKER_HOVER_TOLERANCE_PERCENT = 0.3;

/** WCAG 2.x AA minimum contrast for small text. */
const TEXT_CONTRAST_RATIO = 4.5;

/** WCAG 2.x AA minimum contrast for UI components and graphical objects. */
const UI_CONTRAST_RATIO = 3;

const HEX_COLOR_PATTERN = /^#(?:[0-9a-f]{3}|[0-9a-f]{6}|[0-9a-f]{8})$/i;

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
  readonly density: number;
  readonly region: string;
  readonly pallasPrimitive: string;
  readonly namedScope: string;
  readonly active: boolean;
}

/** A major region or Pallas primitive flame bar along the schedule minimap. */
export interface MinimapRegionMarker {
  /** Unique key across scope sections, e.g. `regions:12`. */
  readonly key: string;
  readonly eventIndex: number;
  readonly name: string;
  readonly startUs: number;
  readonly durationUs: number;
  /** Flame level within the scope track; also the minimap row. */
  readonly depth: number;
  readonly row: number;
  readonly topPercent: number;
  readonly heightPercent: number;
  readonly leftPercent: number;
  readonly widthPercent: number;
  readonly color: string;
  readonly textColor: string;
  readonly bundlesLabel: string;
  /** Number of sub-pixel events merged into this marker (1 for real events). */
  readonly count: number;
  readonly ariaLabel: string;
  readonly focusable: boolean;
  readonly source?: string;
}

/** One mirrored scope track (Regions or Pallas Primitives) of the canvas. */
export interface MinimapScopeSection {
  readonly key: 'pallas' | 'regions';
  readonly trackName: string;
  readonly rows: number;
  readonly topPx: number;
  readonly heightPx: number;
  readonly showLabels: boolean;
  readonly markers: readonly MinimapRegionMarker[];
}

/** An activity lane mirroring one (or several merged) canvas unit tracks. */
export interface MinimapLane {
  readonly name: string;
  /** CSS `#rrggbb` / `#rrggbbaa` colors per cell; empty for idle cells. */
  readonly cells: readonly string[];
  /** Hard-stop CSS gradient rendering `cells` across the lane. */
  readonly background: string;
  readonly heightPx: number;
}

/** Vertical layout of the minimap track in CSS pixels. */
export interface MinimapLayout {
  readonly barHeightPx: number;
  readonly trackHeightPx: number;
  readonly stripHeightPx: number;
  readonly lanesTopPx: number;
  readonly lanesHeightPx: number;
}

/**
 * Colors derived from the canvas theme that are guaranteed to stay legible on
 * every palette (WCAG AA: 4.5:1 for text, 3:1 for UI affordances).
 */
export interface MinimapColorTokens {
  /** Primary text on the minimap background. */
  readonly ink: string;
  /** Secondary text on the minimap background. */
  readonly inkMuted: string;
  /** Ruler tick labels on the ruler (midtone) background. */
  readonly rulerInk: string;
  /** Viewport lens frame and resize handle fill. */
  readonly lens: string;
  /** Grip marks drawn on the resize handles. */
  readonly onLens: string;
  /** Ring separating the lens and handles from the colorful tracks. */
  readonly halo: string;
  /** Translucent veil dimming the schedule outside the viewport lens. */
  readonly curtain: string;
  /** Elevated surface of the hover pill. */
  readonly surfaceRaised: string;
  readonly pillInk: string;
  readonly pillInkMuted: string;
  readonly pillBorder: string;
}

/** A tick mark on the full-schedule bundle ruler above the minimap tracks. */
export interface MinimapRulerTick {
  readonly bundle: number;
  readonly label: string;
  readonly leftPercent: number;
  readonly isRightEdge?: boolean;
}

/** A colored chip in the hover pill with a guaranteed-legible text color. */
export interface MinimapHoverChip {
  readonly label: string;
  readonly color: string;
  readonly textColor: string;
}

/** Floating hover pill state when inspecting the minimap ribbon. */
export interface MinimapHoverInfo {
  readonly leftPercent: number;
  readonly align: 'left' | 'center' | 'right';
  readonly bundleLabel: string;
  /** Track name of the scope row or lane under the pointer. */
  readonly trackName?: string;
  /** Scope marker or lane activity under the pointer. */
  readonly target?: MinimapHoverChip;
  /** Extra context for `target`, e.g. bundle range and source location. */
  readonly targetDetail?: string;
  /** Key of the hovered scope marker, used to highlight it. */
  readonly markerKey?: string;
  readonly unit?: MinimapHoverChip;
  readonly secondaryUnit?: MinimapHoverChip;
  readonly pallas?: MinimapHoverChip;
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
  readonly tokens: MinimapColorTokens;
  readonly layout: MinimapLayout;
  readonly bins: readonly MinimapBin[];
  readonly lanes: readonly MinimapLane[];
  readonly scopeSections: readonly MinimapScopeSection[];
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

interface Rgba {
  readonly r: number;
  readonly g: number;
  readonly b: number;
  readonly a: number;
}

const BLACK: Rgba = {r: 0, g: 0, b: 0, a: 1};
const WHITE: Rgba = {r: 255, g: 255, b: 255, a: 1};

/** Parses a CSS `#rgb`, `#rrggbb`, or `#rrggbbaa` color. */
function parseHexColor(hex: string): Rgba | undefined {
  const value = hex.trim();
  if (!HEX_COLOR_PATTERN.test(value)) {
    return undefined;
  }
  let digits = value.slice(1);
  if (digits.length === 3) {
    digits = digits
      .split('')
      .map((c) => c + c)
      .join('');
  }
  // The pattern above guarantees valid hex digits for these conversions.
  const channel = (start: number) =>
    Number(`0x${digits.slice(start, start + 2)}`);
  return {
    r: channel(0),
    g: channel(2),
    b: channel(4),
    a: digits.length === 8 ? channel(6) / 255 : 1,
  };
}

function toHexColor({r, g, b}: Rgba): string {
  const channel = (v: number) =>
    Math.round(clamp(v, 0, 255))
      .toString(16)
      .padStart(2, '0');
  return `#${channel(r)}${channel(g)}${channel(b)}`;
}

/** Linearly interpolates from `from` (t = 0) to `to` (t = 1). */
function mixRgb(from: Rgba, to: Rgba, t: number): Rgba {
  return {
    r: from.r + (to.r - from.r) * t,
    g: from.g + (to.g - from.g) * t,
    b: from.b + (to.b - from.b) * t,
    a: 1,
  };
}

function linearSrgbChannel(value255: number): number {
  const c = value255 / 255;
  return c <= 0.03928 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
}

function relativeLuminance({r, g, b}: Rgba): number {
  return (
    0.2126 * linearSrgbChannel(r) +
    0.7152 * linearSrgbChannel(g) +
    0.0722 * linearSrgbChannel(b)
  );
}

function rgbContrast(a: Rgba, b: Rgba): number {
  const la = relativeLuminance(a);
  const lb = relativeLuminance(b);
  return (Math.max(la, lb) + 0.05) / (Math.min(la, lb) + 0.05);
}

/** WCAG 2.x contrast ratio between two opaque CSS hex colors (1 to 21). */
export function contrastRatio(foreground: string, background: string): number {
  const fg = parseHexColor(foreground);
  const bg = parseHexColor(background);
  return fg && bg ? rgbContrast(fg, bg) : 1;
}

/**
 * Returns `color` as `#rrggbb`, nudged toward black or white (whichever moves
 * it away from `background`) just enough to reach `minRatio`. Keeps the hue of
 * theme colors that already pass so the minimap stays on-palette.
 */
export function ensureContrast(
  color: string,
  background: string,
  minRatio: number,
): string {
  const fg = parseHexColor(color);
  const bg = parseHexColor(background);
  if (!fg || !bg) {
    return color;
  }
  if (rgbContrast(fg, bg) >= minRatio) {
    return toHexColor(fg);
  }
  const target =
    rgbContrast(BLACK, bg) >= rgbContrast(WHITE, bg) ? BLACK : WHITE;
  if (rgbContrast(target, bg) < minRatio) {
    return toHexColor(target);
  }
  let lo = 0;
  let hi = 1;
  for (let i = 0; i < 12; i++) {
    const mid = (lo + hi) / 2;
    if (rgbContrast(mixRgb(fg, target, mid), bg) >= minRatio) {
      hi = mid;
    } else {
      lo = mid;
    }
  }
  // Rounding channels to integers can land just under the threshold; step a
  // little further toward the target until the emitted color passes.
  let result = toHexColor(mixRgb(fg, target, hi));
  while (hi < 1 && contrastRatio(result, background) < minRatio) {
    hi = Math.min(1, hi + 0.01);
    result = toHexColor(mixRgb(fg, target, hi));
  }
  return result;
}

/** Composites a possibly translucent CSS hex color over an opaque backdrop. */
export function compositeOver(color: string, backdrop: string): string {
  const fg = parseHexColor(color);
  const bg = parseHexColor(backdrop);
  if (!fg) {
    return backdrop;
  }
  return bg ? toHexColor(mixRgb(bg, fg, fg.a)) : toHexColor(fg);
}

/**
 * Picks the most legible text color for a chip or bar filled with
 * `background`: the best of `preferred` (theme inks) when it reaches WCAG AA
 * (4.5:1) for small text, otherwise whichever of black or white contrasts
 * more, since nudging a theme ink only gets mid-tone fills to the bare
 * minimum.
 */
function readableTextColor(
  background: string,
  preferred: readonly string[],
): string {
  const bg = parseHexColor(background);
  if (!bg) {
    return preferred[0] ?? toHexColor(BLACK);
  }
  let best: Rgba | undefined;
  let bestRatio = -1;
  for (const candidate of preferred) {
    const rgba = parseHexColor(candidate);
    const ratio = rgba ? rgbContrast(rgba, bg) : -1;
    if (rgba && ratio > bestRatio) {
      best = rgba;
      bestRatio = ratio;
    }
  }
  if (best && bestRatio >= TEXT_CONTRAST_RATIO) {
    return toHexColor(best);
  }
  return toHexColor(
    rgbContrast(BLACK, bg) >= rgbContrast(WHITE, bg) ? BLACK : WHITE,
  );
}

/**
 * Picks the most legible text color for a chip or bar filled with
 * `background`, preferring the canvas theme's own inks and guaranteeing WCAG
 * AA contrast (4.5:1) wherever any color can reach it.
 */
export function pickReadableTextColor(
  background: string,
  theme: MinimapTheme,
): string {
  return readableTextColor(background, [
    theme.onSurface,
    theme.inverseOnSurface,
    theme.foreground,
    theme.background,
  ]);
}

/** Derives contrast-safe minimap chrome colors from the canvas theme. */
export function buildMinimapColorTokens(
  theme: MinimapTheme,
): MinimapColorTokens {
  const background = parseHexColor(theme.background)
    ? theme.background
    : DEFAULT_MINIMAP_THEME.background;
  const ink = ensureContrast(theme.foreground, background, TEXT_CONTRAST_RATIO);
  const backgroundRgb = parseHexColor(background) ?? WHITE;
  const inkRgb = parseHexColor(ink) ?? BLACK;
  // A slightly elevated surface keeps the hover pill distinct from the canvas
  // it floats over, in both light and dark themes.
  const surfaceRaised = toHexColor(mixRgb(backgroundRgb, inkRgb, 0.07));
  const lens = ensureContrast(theme.foreground, background, UI_CONTRAST_RATIO);
  // Pastel event colors already sit close to light backgrounds, so light
  // palettes get a lighter veil to keep the out-of-view schedule readable.
  const curtainAlpha = relativeLuminance(backgroundRgb) > 0.5 ? 0.4 : 0.5;
  return {
    ink,
    inkMuted: ensureContrast(theme.subtitle, background, TEXT_CONTRAST_RATIO),
    rulerInk: ensureContrast(
      theme.rulerText,
      parseHexColor(theme.midtone) ? theme.midtone : background,
      TEXT_CONTRAST_RATIO,
    ),
    lens,
    onLens: ensureContrast(toHexColor(backgroundRgb), lens, UI_CONTRAST_RATIO),
    halo: toHexColor(backgroundRgb),
    curtain: `rgba(${backgroundRgb.r}, ${backgroundRgb.g}, ${backgroundRgb.b}, ${curtainAlpha})`,
    surfaceRaised,
    pillInk: ensureContrast(
      theme.foreground,
      surfaceRaised,
      TEXT_CONTRAST_RATIO,
    ),
    pillInkMuted: ensureContrast(
      theme.subtitle,
      surfaceRaised,
      TEXT_CONTRAST_RATIO,
    ),
    pillBorder: toHexColor(mixRgb(backgroundRgb, inkRgb, 0.24)),
  };
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

  const bins: MinimapBin[] = [];
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
    bins.push({
      index: bins.length,
      unit,
      secondaryUnit,
      color:
        unit?.color ??
        (active
          ? (theme.traceColors[0] ?? DEFAULT_ACCENT_COLOR)
          : 'transparent'),
      secondaryColor: secondaryUnit?.color,
      density,
      region,
      pallasPrimitive,
      namedScope,
      active,
    });
  }
  return bins;
}

function formatGradientPercent(fraction: number): string {
  return `${Number((fraction * 100).toFixed(3))}%`;
}

/**
 * Builds a hard-stop horizontal CSS gradient painting each lane cell with its
 * color (or leaving it transparent when idle). Adjacent equal cells share one
 * stop pair so long runs stay cheap to render.
 */
export function buildLaneGradient(cells: readonly string[]): string {
  const n = cells.length;
  const stops: string[] = [];
  let hasColor = false;
  let runStart = 0;
  for (let i = 1; i <= n; i++) {
    if (i < n && cells[i] === cells[runStart]) {
      continue;
    }
    const color = cells[runStart] || 'transparent';
    hasColor = hasColor || Boolean(cells[runStart]);
    stops.push(
      `${color} ${formatGradientPercent(runStart / n)} ${formatGradientPercent(
        i / n,
      )}`,
    );
    runStart = i;
  }
  if (!hasColor) {
    return 'none';
  }
  if (stops.length === 1) {
    return `linear-gradient(to right, ${cells[0]} 0%, ${cells[0]} 100%)`;
  }
  return `linear-gradient(to right, ${stops.join(', ')})`;
}

/** Splits `totalPx` into `count` integer heights so lanes render crisply. */
function distributeHeights(totalPx: number, count: number): number[] {
  if (count <= 0) {
    return [];
  }
  const base = Math.floor(totalPx / count);
  const extra = totalPx - base * count;
  return Array.from({length: count}, (_, i) =>
    Math.max(1, base + (i < extra ? 1 : 0)),
  );
}

/** Parses raw `lanes` from `minimap_updated` (one lane per canvas track). */
function parseMinimapLanes(rawLanes: unknown): Array<{
  name: string;
  cells: string[];
}> {
  if (!Array.isArray(rawLanes)) {
    return [];
  }
  const lanes: Array<{name: string; cells: string[]}> = [];
  for (const item of rawLanes) {
    if (!isRecord(item)) {
      continue;
    }
    const rawCells: unknown = item['cells'];
    if (!Array.isArray(rawCells)) {
      continue;
    }
    const name = typeof item['name'] === 'string' ? item['name'] : '';
    const cells = rawCells.map((cell: unknown) =>
      typeof cell === 'string' && HEX_COLOR_PATTERN.test(cell) ? cell : '',
    );
    if (cells.some(Boolean)) {
      lanes.push({name, cells});
    }
  }
  return lanes;
}

/**
 * Derives activity lanes from bins for payloads without `lanes`: one lane per
 * functional unit in canonical canvas order, painted where the unit is the
 * primary or secondary unit of a bin.
 */
function buildLanesFromBins(
  bins: readonly MinimapBin[],
): Array<{name: string; cells: string[]}> {
  const order = new Map<string, number>();
  for (const bin of bins) {
    for (const unit of [bin.unit, bin.secondaryUnit]) {
      if (unit && !order.has(unit.label)) {
        order.set(
          unit.label,
          FUNCTIONAL_UNIT_INDEX_BY_LANE.get(unit.label) ??
            FUNCTIONAL_UNITS.length + order.size,
        );
      }
    }
  }
  return [...order.entries()]
    .sort((a, b) => a[1] - b[1])
    .map(([label]) => ({
      name: label,
      cells: bins.map((bin) => {
        if (bin.unit?.label === label) return bin.unit.color;
        if (bin.secondaryUnit?.label === label) return bin.secondaryUnit.color;
        return '';
      }),
    }));
}

/**
 * Parses raw `regions` or `pallasPrimitives` from `minimap_updated` event
 * detail. Each marker is placed on the row of its canvas flame level (`depth`)
 * so nested scopes keep the exact shapes the timeline draws.
 */
function parseMinimapRegions(
  rawRegions: unknown,
  dataStartUs: number,
  dataEndUs: number,
  theme: MinimapTheme = DEFAULT_MINIMAP_THEME,
  isTimeAxis = false,
  sectionKey: MinimapScopeSection['key'] = 'regions',
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
    readonly count: number;
    readonly color?: string;
    readonly textColor?: string;
    readonly source?: string;
  }
  const candidates: RawRegion[] = [];
  let maxDepth = 0;
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
    const depth =
      typeof item['depth'] === 'number' && Number.isFinite(item['depth'])
        ? Math.max(0, Math.floor(item['depth']))
        : 0;
    const count =
      typeof item['count'] === 'number' && item['count'] > 1
        ? Math.floor(item['count'])
        : 1;
    const color =
      typeof item['color'] === 'string' && parseHexColor(item['color'])
        ? item['color']
        : undefined;
    const textColor =
      typeof item['textColor'] === 'string' && parseHexColor(item['textColor'])
        ? item['textColor']
        : undefined;
    const source =
      typeof item['source'] === 'string' && item['source']
        ? item['source']
        : undefined;
    if (
      eventIndex < 0 ||
      !name ||
      !Number.isFinite(startUs) ||
      depth >= MAX_SCOPE_ROWS
    ) {
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
    // True-to-scale width; CSS keeps sub-pixel markers visible with a minimum
    // width instead of inflating them here.
    const widthPercent = clamp(
      (durationUs / duration) * 100,
      0,
      100 - leftPercent,
    );
    const startBundle = Math.round(startUs);
    const endBundle = Math.round(startUs + durationUs);
    const bundlesLabel = isTimeAxis
      ? formatMinimapTimeUs(durationUs)
      : durationUs > 1
        ? `${BUNDLE_FORMAT.format(startBundle)}–${BUNDLE_FORMAT.format(endBundle)}`
        : BUNDLE_FORMAT.format(startBundle);
    maxDepth = Math.max(maxDepth, depth);
    candidates.push({
      eventIndex,
      name,
      startUs,
      durationUs,
      depth,
      leftPercent,
      widthPercent,
      bundlesLabel,
      count,
      color,
      textColor,
      source,
    });
  }

  const rows = maxDepth + 1;
  const rowHeightPercent = 100 / rows;
  const palette =
    theme.traceColors.length > 0 ? theme.traceColors : TRACE_VIEWER_PALETTE;
  const kind = sectionKey === 'pallas' ? 'Pallas primitive' : 'region';
  const regions: MinimapRegionMarker[] = [];
  for (let i = 0; i < candidates.length; i++) {
    const c = candidates[i];
    const color = c.color ?? palette[i % palette.length];
    // Re-validate the canvas-provided label color so it always reaches AA.
    const textColor = c.textColor
      ? readableTextColor(color, [
          c.textColor,
          theme.onSurface,
          theme.inverseOnSurface,
        ])
      : pickReadableTextColor(color, theme);
    const label = c.count > 1 ? `${c.name} +${c.count - 1} more` : c.name;
    regions.push({
      key: `${sectionKey}:${c.eventIndex}`,
      eventIndex: c.eventIndex,
      name: c.name,
      startUs: c.startUs,
      durationUs: c.durationUs,
      depth: c.depth,
      row: c.depth,
      topPercent: c.depth * rowHeightPercent,
      heightPercent: rowHeightPercent,
      leftPercent: c.leftPercent,
      widthPercent: c.widthPercent,
      color,
      textColor,
      bundlesLabel: c.bundlesLabel,
      count: c.count,
      ariaLabel: `Select ${kind} ${label} (${c.bundlesLabel}${
        isTimeAxis ? '' : ' bundles'
      })`,
      focusable:
        c.count === 1 && c.widthPercent >= MIN_FOCUSABLE_MARKER_WIDTH_PERCENT,
      source: c.source,
    });
  }
  return regions;
}

/** Number of mirrored flame levels used by a list of scope markers. */
function scopeRowCount(markers: readonly MinimapRegionMarker[]): number {
  let maxDepth = 0;
  for (const marker of markers) {
    maxDepth = Math.max(maxDepth, marker.depth);
  }
  return markers.length > 0 ? maxDepth + 1 : 0;
}

interface ScopeSectionInput {
  readonly key: MinimapScopeSection['key'];
  readonly trackName: string;
  readonly markers: readonly MinimapRegionMarker[];
}

/**
 * Stacks the strip, scope sections, and activity lanes inside the track using
 * whole pixels so thin rows stay crisp.
 */
function buildMinimapLayout(
  sections: readonly ScopeSectionInput[],
  laneCount: number,
): {layout: MinimapLayout; sections: MinimapScopeSection[]} {
  const trackHeightPx =
    sections.length >= 2 ? TRACK_HEIGHT_TWO_SCOPES_PX : TRACK_HEIGHT_PX;
  let y = STRIP_HEIGHT_PX + BAND_GAP_PX;
  const laidOut: MinimapScopeSection[] = [];
  for (const section of sections) {
    const rows = Math.max(1, scopeRowCount(section.markers));
    const heightPx =
      rows === 1 ? SCOPE_SINGLE_ROW_HEIGHT_PX : rows * SCOPE_ROW_HEIGHT_PX;
    laidOut.push({
      ...section,
      rows,
      topPx: y,
      heightPx,
      showLabels: rows === 1,
    });
    y += heightPx + BAND_GAP_PX;
  }
  // Leave a 1px gutter above the bar's bottom border.
  const lanesHeightPx = laneCount > 0 ? Math.max(0, trackHeightPx - 1 - y) : 0;
  return {
    layout: {
      // The bar includes a 1px bottom border.
      barHeightPx: RULER_HEIGHT_PX + trackHeightPx + 1,
      trackHeightPx,
      stripHeightPx: STRIP_HEIGHT_PX,
      lanesTopPx: y,
      lanesHeightPx,
    },
    sections: laidOut,
  };
}

/**
 * Horizontal schedule minimap for the Static Kernel Viewer and Trace Viewer.
 *
 * Renders a full-schedule miniature of the timeline canvas above it, aligned
 * pixel-for-pixel with the track label column and timeline ruler: the top
 * flame levels of the scope tracks (Regions, Pallas Primitives, Steps, ...)
 * row by row, followed by one activity lane per unit track in canvas order,
 * all in the canvas's own colors. Supports dragging the viewport lens to pan,
 * dragging lens handles to resize/zoom, clicking or brushing outside the lens
 * to jump/zoom, mouse-wheel zooming, and double-clicking or clicking Fit to
 * reset to the full schedule.
 */
@Component({
  standalone: true,
  selector: 'kernel-minimap',
  templateUrl: './kernel_minimap.ng.html',
  styleUrls: ['./kernel_minimap.scss'],
  changeDetection: ChangeDetectionStrategy.OnPush,
  imports: [CommonModule, MatIconModule, MatTooltipModule],
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
  private cachedTokens: MinimapColorTokens = buildMinimapColorTokens(
    DEFAULT_MINIMAP_THEME,
  );
  private cachedBins: readonly MinimapBin[] = [];
  private cachedLaneData: ReadonlyArray<{
    readonly name: string;
    readonly cells: readonly string[];
  }> = [];
  private hasExplicitLanes = false;
  private cachedRegions: readonly MinimapRegionMarker[] = [];
  private cachedPallasPrimitives: readonly MinimapRegionMarker[] = [];
  private cachedRegionsFirst = false;
  private cachedRegionsTrackName = 'Regions';
  private cachedPallasTrackName = 'Pallas Primitives';
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
    this.updateHoverInfo(event.clientX, event.clientY);
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

  trackByRegionIndex(index: number, marker: MinimapRegionMarker): string {
    return marker.key;
  }

  trackBySectionKey(index: number, section: MinimapScopeSection): string {
    return section.key;
  }

  trackByLaneIndex(index: number): number {
    return index;
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
      this.cachedTokens = buildMinimapColorTokens(this.cachedTheme);
    }

    if (Array.isArray(detail['bins'])) {
      this.cachedBins = parseMinimapBins(detail['bins'], this.cachedTheme);
    }
    if (Array.isArray(detail['lanes'])) {
      this.cachedLaneData = parseMinimapLanes(detail['lanes']);
      this.hasExplicitLanes = true;
    } else if (Array.isArray(detail['bins']) && !this.hasExplicitLanes) {
      this.cachedLaneData = buildLanesFromBins(this.cachedBins);
    }
    if (typeof detail['regionsFirst'] === 'boolean') {
      this.cachedRegionsFirst = detail['regionsFirst'];
    }
    if (typeof detail['regionsTrackName'] === 'string') {
      this.cachedRegionsTrackName = detail['regionsTrackName'];
    }
    if (typeof detail['pallasTrackName'] === 'string') {
      this.cachedPallasTrackName = detail['pallasTrackName'];
    }
    if (Array.isArray(detail['regions'])) {
      this.cachedRegions = parseMinimapRegions(
        detail['regions'],
        dataStartUs,
        dataEndUs,
        this.cachedTheme,
        this.isTimeAxis,
        'regions',
      );
    } else if (boundsChanged && this.cachedRegions.length > 0) {
      this.cachedRegions = parseMinimapRegions(
        this.cachedRegions,
        dataStartUs,
        dataEndUs,
        this.cachedTheme,
        this.isTimeAxis,
        'regions',
      );
    }
    if (Array.isArray(detail['pallasPrimitives'])) {
      this.cachedPallasPrimitives = parseMinimapRegions(
        detail['pallasPrimitives'],
        dataStartUs,
        dataEndUs,
        this.cachedTheme,
        this.isTimeAxis,
        'pallas',
      );
    } else if (boundsChanged && this.cachedPallasPrimitives.length > 0) {
      this.cachedPallasPrimitives = parseMinimapRegions(
        this.cachedPallasPrimitives,
        dataStartUs,
        dataEndUs,
        this.cachedTheme,
        this.isTimeAxis,
        'pallas',
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

  private updateHoverInfo(clientX: number, clientY: number): void {
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
    const theme = this.cachedTheme;
    const chip = (label: string, color: string): MinimapHoverChip => ({
      label,
      color,
      textColor: pickReadableTextColor(color, theme),
    });
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
        theme.traceColors[2] ??
        '#b1d284')
      : undefined;

    this.hoverInfo = {
      leftPercent,
      align,
      bundleLabel: this.isTimeAxis
        ? formatMinimapTimeUs(hoveredUs - this.dataStartUs)
        : BUNDLE_FORMAT.format(bundle),
      ...this.findHoverTarget(leftPercent, clientY - rect.top),
      unit: bin?.unit ? chip(bin.unit.label, bin.unit.color) : undefined,
      secondaryUnit: bin?.secondaryUnit
        ? chip(`+${bin.secondaryUnit.label}`, bin.secondaryUnit.color)
        : undefined,
      pallas:
        pallasPrimitive && pallasColor
          ? chip(pallasPrimitive, pallasColor)
          : undefined,
      region: bin?.region || undefined,
    };
    this.changeDetectorRef.markForCheck();
  }

  /**
   * Resolves the scope marker or activity lane under the pointer, using the
   * same geometry the template renders so it also works under the lens.
   */
  private findHoverTarget(
    leftPercent: number,
    offsetY: number,
  ): Pick<
    MinimapHoverInfo,
    'trackName' | 'target' | 'targetDetail' | 'markerKey'
  > {
    const state = this.state;
    if (!state) {
      return {};
    }
    for (const section of state.scopeSections) {
      if (
        offsetY < section.topPx ||
        offsetY >= section.topPx + section.heightPx
      ) {
        continue;
      }
      const row = Math.floor(
        ((offsetY - section.topPx) / section.heightPx) * section.rows,
      );
      let best: MinimapRegionMarker | undefined;
      let bestDistance = Infinity;
      for (const marker of section.markers) {
        if (marker.row !== row) {
          continue;
        }
        const start = marker.leftPercent;
        const end = marker.leftPercent + marker.widthPercent;
        const distance =
          leftPercent < start
            ? start - leftPercent
            : leftPercent > end
              ? leftPercent - end
              : 0;
        if (
          distance <= MARKER_HOVER_TOLERANCE_PERCENT &&
          distance < bestDistance
        ) {
          best = marker;
          bestDistance = distance;
        }
      }
      if (!best) {
        return {};
      }
      const unitSuffix = state.isTimeAxis ? '' : ' bundles';
      return {
        trackName: section.trackName,
        markerKey: best.key,
        target: {
          label: best.count > 1 ? `${best.name} +${best.count - 1}` : best.name,
          color: best.color,
          textColor: best.textColor,
        },
        targetDetail: `${best.bundlesLabel}${unitSuffix}${
          best.source ? ` · ${best.source}` : ''
        }`,
      };
    }
    const {lanesTopPx, lanesHeightPx} = state.layout;
    if (
      state.lanes.length > 0 &&
      offsetY >= lanesTopPx &&
      offsetY < lanesTopPx + lanesHeightPx
    ) {
      let laneTop = lanesTopPx;
      for (const lane of state.lanes) {
        if (offsetY < laneTop + lane.heightPx) {
          const cell =
            lane.cells[
              clamp(
                Math.floor((leftPercent / 100) * lane.cells.length),
                0,
                lane.cells.length - 1,
              )
            ];
          if (!cell) {
            return {trackName: lane.name, targetDetail: 'Idle'};
          }
          // Chip shows the lane cell exactly as painted over the background.
          const color = compositeOver(cell, state.theme.background);
          return {
            target: {
              label: lane.name,
              color,
              textColor: pickReadableTextColor(color, state.theme),
            },
          };
        }
        laneTop += lane.heightPx;
      }
    }
    return {};
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

    // Scope sections stack in the same order as their tracks on the canvas.
    const scopeInputs: ScopeSectionInput[] = [];
    if (this.cachedPallasPrimitives.length > 0) {
      scopeInputs.push({
        key: 'pallas',
        trackName: this.cachedPallasTrackName,
        markers: this.cachedPallasPrimitives,
      });
    }
    if (this.cachedRegions.length > 0) {
      const regionsSection: ScopeSectionInput = {
        key: 'regions',
        trackName: this.cachedRegionsTrackName,
        markers: this.cachedRegions,
      };
      if (this.cachedRegionsFirst) {
        scopeInputs.unshift(regionsSection);
      } else {
        scopeInputs.push(regionsSection);
      }
    }
    const {layout, sections} = buildMinimapLayout(
      scopeInputs,
      this.cachedLaneData.length,
    );
    const laneHeights = distributeHeights(
      layout.lanesHeightPx,
      Math.min(this.cachedLaneData.length, layout.lanesHeightPx),
    );
    const previousLanes = this.state?.lanes ?? [];
    const lanes: MinimapLane[] = laneHeights.map((heightPx, i) => {
      const {name, cells} = this.cachedLaneData[i];
      // Reuse the previous gradient while only the viewport moves.
      const previous = previousLanes[i];
      const background =
        previous?.cells === cells
          ? previous.background
          : buildLaneGradient(cells);
      return {name, cells, background, heightPx};
    });

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
      tokens: this.cachedTokens,
      layout,
      bins: this.cachedBins,
      lanes,
      scopeSections: sections,
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
