/**
 * Choreography of the demo pointer of the Kernel Viewer tour.
 *
 * Each step of the tour can show how to use the part of the viewer that it is
 * about: a pointer glides over the real viewer, clicks, drags and presses keys,
 * next to a caption of what it does. The scripts are built from the positions
 * of the real elements of the viewer, so that the demo happens right where the
 * user then tries it, and are sampled frame by frame.
 */

import {FUNCTIONAL_UNITS_BY_LANE} from 'org_xprof/frontend/app/components/static_kernel_viewer/kernel_event_tooltip';

/** A position, in viewport pixels. */
export interface Point {
  readonly x: number;
  readonly y: number;
}

/** A position and size, in viewport pixels. */
export interface Box {
  readonly x: number;
  readonly y: number;
  readonly w: number;
  readonly h: number;
}

/** A size, in pixels. */
export interface Size {
  readonly w: number;
  readonly h: number;
}

/**
 * A rectangle that previews the effect of a gesture, e.g. where the minimap
 * lens goes, or the span of bundles that a drag measures.
 */
export interface GhostBox extends Box {
  /** Text above the rectangle. */
  readonly label?: string;
  /** Bundles per pixel, to label the rectangle with the bundles it spans. */
  readonly bundlesPerPixel?: number;
}

interface BeatBase {
  /** How long the beat lasts, in milliseconds. */
  readonly duration: number;
  /** What the pointer does, shown next to it until a later beat changes it. */
  readonly caption?: string;
  /** Keys shown with the caption, e.g. `['Shift']`, during the beat only. */
  readonly keys?: readonly string[];
}

/** The pointer glides to `to`. */
export interface MoveBeat extends BeatBase {
  readonly kind: 'move';
  readonly to: Point;
  /** Whether a beacon marks `to` while the pointer is on its way. */
  readonly beacon?: boolean;
}

/** The pointer clicks where it is, once or twice. */
export interface ClickBeat extends BeatBase {
  readonly kind: 'click';
  readonly count?: 1 | 2;
  /** Where the ghost box goes once clicked, e.g. where the lens jumps. */
  readonly ghost?: GhostBox;
  /** Whether the ghost box fades away. */
  readonly clearGhost?: boolean;
}

/** The pointer presses, moves to `to`, and lets go. */
export interface DragBeat extends BeatBase {
  readonly kind: 'drag';
  readonly to: Point;
  /** Where the ghost box starts, if there is none yet. */
  readonly ghostFrom?: GhostBox;
  /** Where the ghost box ends up. It moves along with the pointer. */
  readonly ghost?: GhostBox;
}

/** Keys are pressed. The pointer hides, and the keys show at `at`. */
export interface KeysBeat extends BeatBase {
  readonly kind: 'keys';
  readonly keys: readonly string[];
  readonly at: Point;
}

/** The pointer stays, while the ghost box may move or fade. */
export interface HoldBeat extends BeatBase {
  readonly kind: 'hold';
  /** Where the ghost box starts, if there is none yet. */
  readonly ghostFrom?: GhostBox;
  /** Where the ghost box goes. */
  readonly ghost?: GhostBox;
  /** Whether the ghost box fades away. */
  readonly clearGhost?: boolean;
}

/** A beat of a demo. */
export type DemoBeat = MoveBeat | ClickBeat | DragBeat | KeysBeat | HoldBeat;

/** A demo of how to use a part of the viewer, which plays in a loop. */
export interface DemoScript {
  /** Where the pointer appears. */
  readonly start: Point;
  readonly beats: readonly DemoBeat[];
  /** Pause after the last beat, before the demo plays again. */
  readonly rest: number;
  /** What the tour card should not cover, e.g. the block to click. */
  readonly focus?: Box;
  /**
   * Numbered markers on parts of the viewer that the card lists, in its
   * order. A part that is not shown has no marker.
   */
  readonly hotspots?: ReadonlyArray<Point | undefined>;
}

/** A click ripple, which spreads as its `progress` goes from 0 to 1. */
export interface Ripple extends Point {
  readonly progress: number;
}

/** What the demo shows at a moment. */
export interface DemoPose {
  /** Opacity of the pointer and its caption, 0 while the demo rests. */
  readonly opacity: number;
  readonly pointer: Point;
  /** Whether the pointer shows, rather than only its caption and keys. */
  readonly showPointer: boolean;
  /** Whether the mouse button is down. */
  readonly pressed: boolean;
  readonly caption: string;
  readonly keys: readonly string[];
  /** Whether the keys are down. */
  readonly keysDown: boolean;
  /** The ripple of the last click, while it spreads. */
  readonly ripple?: Ripple;
  /** Where the pointer is going, marked while it is on its way. */
  readonly beacon?: Point;
  readonly ghost?: GhostBox;
  /** Opacity of the ghost box. */
  readonly ghostOpacity: number;
}

/** How long the pointer takes to appear as a demo starts. */
const FADE_IN_MS = 280;

/** How long the pointer takes to disappear at the end of a demo. */
const FADE_OUT_MS = 360;

/** How long a click ripple spreads. */
export const RIPPLE_MS = 640;

/** When the button is down during clicks, as fractions of their beats. */
const CLICK_PRESSES: Readonly<Record<1 | 2, ReadonlyArray<[number, number]>>> =
  {
    1: [[0.14, 0.3]],
    2: [
      [0.1, 0.2],
      [0.3, 0.4],
    ],
  };

/** When the button goes down and up during drags, as fractions of drags. */
const DRAG_PRESS: readonly [number, number] = [0.1, 0.9];

/** When the pointer moves during drags, as fractions of drags. */
const DRAG_MOTION: readonly [number, number] = [0.18, 0.82];

/** When the keys are down during key beats, as fractions of the beats. */
const KEYS_DOWN: readonly [number, number] = [0.22, 0.5];

/** When the ghost box moves during clicks, as fractions of clicks. */
const CLICK_GHOST_MOTION: readonly [number, number] = [0.32, 0.86];

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(value, max));
}

function lerp(from: number, to: number, t: number): number {
  return from + (to - from) * t;
}

function lerpPoint(from: Point, to: Point, t: number): Point {
  return {x: lerp(from.x, to.x, t), y: lerp(from.y, to.y, t)};
}

function lerpGhost(from: GhostBox, to: GhostBox, t: number): GhostBox {
  return {
    x: lerp(from.x, to.x, t),
    y: lerp(from.y, to.y, t),
    w: lerp(from.w, to.w, t),
    h: lerp(from.h, to.h, t),
    label: to.label,
    bundlesPerPixel: to.bundlesPerPixel,
  };
}

/** Eases in and out, like a hand that speeds up and slows down. */
export function easeInOut(t: number): number {
  const x = clamp(t, 0, 1);
  return x < 0.5 ? 4 * x * x * x : 1 - Math.pow(-2 * x + 2, 3) / 2;
}

/** Returns how far `t` is through `[start, end]`, between 0 and 1. */
function progressIn(t: number, [start, end]: readonly [number, number]) {
  return end > start ? clamp((t - start) / (end - start), 0, 1) : 1;
}

function isWithin(t: number, [start, end]: readonly [number, number]) {
  return t >= start && t < end;
}

/** How long the beats of `script` take, without the rest at its end. */
export function demoDuration(script: DemoScript): number {
  return script.beats.reduce((total, beat) => total + beat.duration, 0);
}

/** How long a loop of `script` takes, with the rest at its end. */
export function loopDuration(script: DemoScript): number {
  return demoDuration(script) + script.rest;
}

/** Where the pointer is at the end of `beat`, if it starts at `from`. */
function endPoint(beat: DemoBeat, from: Point): Point {
  return beat.kind === 'move' || beat.kind === 'drag' ? beat.to : from;
}

/** The ghost box at the end of `beat`, after `previous`. */
function endGhost(
  beat: DemoBeat,
  previous: GhostBox | undefined,
): GhostBox | undefined {
  if (beat.kind === 'move' || beat.kind === 'keys') {
    return previous;
  }
  if (beat.kind !== 'drag' && beat.clearGhost) {
    return undefined;
  }
  return beat.ghost ?? previous;
}

/** The state of the demo as a beat starts. */
interface BeatStart {
  readonly beat: DemoBeat;
  /** When the beat starts, in milliseconds since the start of the loop. */
  readonly start: number;
  readonly pointer: Point;
  readonly caption: string;
  readonly ghost?: GhostBox;
}

function beatStarts(script: DemoScript): BeatStart[] {
  const starts: BeatStart[] = [];
  let start = 0;
  let pointer = script.start;
  let caption = '';
  let ghost: GhostBox | undefined = undefined;
  for (const beat of script.beats) {
    starts.push({beat, start, pointer, caption, ghost});
    start += beat.duration;
    pointer = endPoint(beat, pointer);
    caption = beat.caption ?? caption;
    ghost = endGhost(beat, ghost);
  }
  return starts;
}

/** Returns the ripple of the last click before `time`, while it spreads. */
function rippleAt(starts: readonly BeatStart[], time: number) {
  let ripple: Ripple | undefined = undefined;
  for (const {beat, start, pointer} of starts) {
    if (beat.kind !== 'click') {
      continue;
    }
    for (const [pressStart] of CLICK_PRESSES[beat.count ?? 1]) {
      const at = start + pressStart * beat.duration;
      if (time >= at && time < at + RIPPLE_MS) {
        ripple = {...pointer, progress: (time - at) / RIPPLE_MS};
      }
    }
  }
  return ripple;
}

/** The ghost box and its opacity during `beat`, `t` of the way through. */
function ghostDuring(
  beat: DemoBeat,
  t: number,
  previous: GhostBox | undefined,
): {ghost?: GhostBox; opacity: number} {
  if (beat.kind === 'move' || beat.kind === 'keys') {
    return {ghost: previous, opacity: previous ? 1 : 0};
  }
  const motion =
    beat.kind === 'drag'
      ? DRAG_MOTION
      : beat.kind === 'click'
        ? CLICK_GHOST_MOTION
        : ([0, 1] as const);
  const progress = easeInOut(progressIn(t, motion));
  if (beat.kind !== 'drag' && beat.clearGhost) {
    return {ghost: previous, opacity: previous ? 1 - progress : 0};
  }
  const target = beat.ghost;
  if (!target) {
    return {ghost: previous, opacity: previous ? 1 : 0};
  }
  const from = previous ?? (beat.kind === 'click' ? undefined : beat.ghostFrom);
  if (!from) {
    // A new ghost box without a start fades in where it ends up.
    return {ghost: target, opacity: progress};
  }
  return {ghost: lerpGhost(from, target, progress), opacity: 1};
}

/** Returns what `script` shows `time` milliseconds into a loop. */
export function sampleDemo(script: DemoScript, time: number): DemoPose {
  const starts = beatStarts(script);
  const total = demoDuration(script);
  const hidden: DemoPose = {
    opacity: 0,
    pointer: script.start,
    showPointer: false,
    pressed: false,
    caption: '',
    keys: [],
    keysDown: false,
    ghostOpacity: 0,
  };
  if (starts.length === 0 || time < 0 || time >= total) {
    return hidden;
  }
  const current =
    [...starts].reverse().find(({start}) => start <= time) ?? starts[0];
  const {beat, start, pointer: from, ghost: previousGhost} = current;
  const t = clamp((time - start) / beat.duration, 0, 1);
  const opacity = Math.min(
    1,
    time / FADE_IN_MS,
    Math.max(0, (total - time) / FADE_OUT_MS),
  );
  const caption = beat.caption ?? current.caption;
  const keys = beat.keys ?? [];
  const {ghost, opacity: ghostOpacity} = ghostDuring(beat, t, previousGhost);
  const pose = {
    opacity,
    caption,
    keys,
    ripple: rippleAt(starts, time),
    ghost,
    ghostOpacity: ghost ? ghostOpacity * Math.min(1, opacity * 2) : 0,
  };
  switch (beat.kind) {
    case 'move':
      return {
        ...pose,
        pointer: lerpPoint(from, beat.to, easeInOut(t)),
        showPointer: true,
        pressed: false,
        keysDown: false,
        beacon: beat.beacon && t < 0.94 ? beat.to : undefined,
      };
    case 'click': {
      const isPressed = CLICK_PRESSES[beat.count ?? 1].some((press) =>
        isWithin(t, press),
      );
      return {
        ...pose,
        pointer: from,
        showPointer: true,
        pressed: isPressed,
        keysDown: isPressed,
      };
    }
    case 'drag': {
      const isPressed = isWithin(t, DRAG_PRESS);
      const motion = easeInOut(progressIn(t, DRAG_MOTION));
      return {
        ...pose,
        pointer: lerpPoint(from, beat.to, motion),
        showPointer: true,
        pressed: isPressed,
        keysDown: isPressed,
      };
    }
    case 'keys':
      return {
        ...pose,
        keys: beat.keys,
        pointer: beat.at,
        showPointer: false,
        pressed: false,
        keysDown: isWithin(t, KEYS_DOWN),
      };
    case 'hold':
      return {
        ...pose,
        pointer: from,
        showPointer: true,
        pressed: false,
        keysDown: false,
      };
    default:
      return hidden;
  }
}

/**
 * Returns a still of `script` for users who prefer reduced motion: the moment
 * of its first click, drag or key press, without fading, or else where its
 * first move ends.
 */
export function staticDemoPose(script: DemoScript): DemoPose | undefined {
  let start = 0;
  let firstMove: MoveBeat | undefined = undefined;
  let firstMoveEnd = 0;
  for (const beat of script.beats) {
    let at: number | undefined = undefined;
    if (beat.kind === 'click') {
      at = start + CLICK_PRESSES[beat.count ?? 1][0][0] * beat.duration + 1;
    } else if (beat.kind === 'drag') {
      at = start + DRAG_MOTION[1] * beat.duration;
    } else if (beat.kind === 'keys') {
      at = start + KEYS_DOWN[0] * beat.duration + 1;
    } else if (beat.kind === 'move' && !firstMove) {
      firstMove = beat;
      firstMoveEnd = start + beat.duration;
    }
    if (at !== undefined) {
      const pose = sampleDemo(script, at);
      // A still ripple marks the click, as if it had just started to spread.
      const ripple = pose.ripple ? {...pose.ripple, progress: 0.35} : undefined;
      return {...pose, opacity: 1, ripple};
    }
    start += beat.duration;
  }
  if (!firstMove) {
    return undefined;
  }
  // The pointer has arrived, and its beacon, if any, stays on the target.
  return {
    ...sampleDemo(script, firstMoveEnd - 1),
    pointer: firstMove.to,
    opacity: 1,
    beacon: firstMove.beacon ? firstMove.to : undefined,
  };
}

/** A block of the timeline that the user pointed at, and where. */
export interface BlockSighting {
  /** Lane of the block, e.g. `MXU`, or ''. */
  readonly lane: string;
  /** Where the pointer was, relative to the top left corner of the canvas. */
  readonly offset: Point;
  /** The bundle under the pointer, if the visible bundles were known. */
  readonly bundle?: number;
}

/**
 * Where the timeline shows bundles: the horizontal extent of the timeline, and
 * the bundles that it spans.
 */
export interface TimelineMapping {
  /** The box of the timeline, right of the labels of its lanes. */
  readonly track: Box;
  /** The first visible bundle. */
  readonly start: number;
  /** The bundle after the last visible one. */
  readonly end: number;
}

/** Returns the bundle at `x`, in viewport pixels. */
export function bundleAt(x: number, mapping: TimelineMapping): number {
  const fraction = (x - mapping.track.x) / Math.max(1, mapping.track.w);
  return mapping.start + fraction * (mapping.end - mapping.start);
}

/**
 * Returns where a block that the user pointed at is now, following the
 * timeline as it zooms and pans, or undefined if it is out of view.
 */
export function locateSighting(
  sighting: BlockSighting,
  canvas: Box,
  mapping?: TimelineMapping,
): Point | undefined {
  let x = canvas.x + sighting.offset.x;
  if (sighting.bundle !== undefined && mapping && mapping.end > mapping.start) {
    const fraction =
      (sighting.bundle - mapping.start) / (mapping.end - mapping.start);
    x = mapping.track.x + fraction * mapping.track.w;
  }
  const y = canvas.y + sighting.offset.y;
  const left = mapping?.track.x ?? canvas.x;
  const right = mapping
    ? mapping.track.x + mapping.track.w
    : canvas.x + canvas.w;
  if (
    x < left + 2 ||
    x > right - 2 ||
    y < canvas.y ||
    y > canvas.y + canvas.h
  ) {
    return undefined;
  }
  return {x, y};
}

/** What the scripts know about the viewer, as measured on the page. */
export interface DemoContext {
  /** Returns the box of the first shown element that matches `selector`. */
  readonly find: (selector: string) => Box | undefined;
  /** Returns the boxes of the shown elements that match `selector`. */
  readonly findAll: (selector: string) => readonly Box[];
  readonly viewport: Size;
  /** The part of the timeline canvas that draws blocks, right of its labels. */
  readonly timeline?: Box;
  /** Where a block that the user pointed at is now, if it is in view. */
  readonly block?: {readonly point: Point; readonly lane: string};
  /** Bundles per pixel of the timeline, if known, to label measurements. */
  readonly bundlesPerPixel?: number;
  /** Whether the details panel of a selection is open. */
  readonly isPanelOpen: boolean;
  /** Where the details panel opens, while it is closed. */
  readonly drawer?: Box;
  /**
   * The metrics of the details panel that the card explains, in its order:
   * where the panel shows each, and what to say about it.
   */
  readonly metrics?: ReadonlyArray<{
    readonly selector: string;
    readonly caption: string;
  }>;
}

/** Returns the point `fx` and `fy` of the way across `box`. */
export function pointIn(box: Box, fx = 0.5, fy = 0.5): Point {
  return {x: box.x + box.w * fx, y: box.y + box.h * fy};
}

/** Returns `point`, moved by `dx` and `dy` and kept in the viewport. */
function offsetPoint(point: Point, dx: number, dy: number, viewport: Size) {
  return {
    x: clamp(point.x + dx, 8, viewport.w - 8),
    y: clamp(point.y + dy, 8, viewport.h - 8),
  };
}

/** Returns a box around `point`, `radius` away from it on each side. */
function boxAround(point: Point, radius: number): Box {
  return {
    x: point.x - radius,
    y: point.y - radius,
    w: 2 * radius,
    h: 2 * radius,
  };
}

/** What the demo of the inspect step asks the user to click. */
export function clickCaptionFor(lane: string | undefined): string {
  if (lane === undefined) {
    return 'Click a block in a unit lane';
  }
  if (FUNCTIONAL_UNITS_BY_LANE.has(lane)) {
    return `Click this ${lane} instruction`;
  }
  if (lane === 'Regions') {
    return 'Click this region';
  }
  return 'Click this block';
}

function railDemo(context: DemoContext): DemoScript | undefined {
  const rows = context.findAll('.kernel-rail .kernel-row');
  const first = rows[0];
  if (!first) {
    return undefined;
  }
  const leader = pointIn(first, 0.38);
  const beats: DemoBeat[] = [
    {
      kind: 'move',
      to: leader,
      duration: 950,
      caption: 'The biggest schedule comes first',
      beacon: true,
    },
    {kind: 'hold', duration: 650},
  ];
  // The statistics of the biggest kernel, at the end of its row.
  const stats = context
    .findAll('.kernel-rail .kernel-row .kernel-meta')
    .find((meta) => meta.y >= first.y - 1 && meta.y < first.y + first.h);
  if (stats) {
    beats.push(
      {
        kind: 'move',
        to: pointIn(stats, 0.45),
        duration: 700,
        caption: 'Busiest unit · bundles',
      },
      {kind: 'hold', duration: 900},
    );
  }
  const target = rows[1] ?? first;
  beats.push(
    {
      kind: 'move',
      to: pointIn(target, 0.38),
      duration: 700,
      caption: 'Click an op to open it',
    },
    {kind: 'click', duration: 950},
    {kind: 'hold', duration: 450},
  );
  return {
    start: offsetPoint(pointIn(first, 1, 2.5), 48, 24, context.viewport),
    beats,
    rest: 1800,
  };
}

function searchDemo(context: DemoContext): DemoScript | undefined {
  const field = context.find('.kernel-rail .rail-search .search-field');
  if (!field) {
    return undefined;
  }
  const keysAt = {x: field.x + 4, y: field.y + field.h - 14};
  const target = pointIn(field, 0.32);
  return {
    start: offsetPoint(pointIn(field, 0.7, 1), 40, 70, context.viewport),
    beats: [
      {
        kind: 'keys',
        keys: ['K'],
        at: keysAt,
        duration: 1600,
        caption: 'Search from anywhere',
      },
      {
        kind: 'move',
        to: target,
        duration: 850,
        caption: 'or click here, then type',
        beacon: true,
      },
      {kind: 'click', duration: 950},
      {kind: 'hold', duration: 600},
    ],
    rest: 1600,
    focus: field,
  };
}

function tabsDemo(context: DemoContext): DemoScript | undefined {
  const strip = context.find('.kernel-bar .tab-strip');
  if (!strip) {
    return undefined;
  }
  const tabs = context.findAll('.kernel-bar .kernel-tab');
  const newTab = context.find('.kernel-bar .new-tab-button');
  const beats: DemoBeat[] = [];
  let first = pointIn(strip, 0.3);
  if (tabs[1]) {
    first = pointIn(tabs[1], 0.4);
    beats.push(
      {
        kind: 'move',
        to: first,
        duration: 950,
        caption: 'Click a tab to switch to it',
        beacon: true,
      },
      {kind: 'click', duration: 950},
    );
  } else if (newTab) {
    first = pointIn(newTab);
    beats.push(
      {
        kind: 'move',
        to: first,
        duration: 950,
        caption: 'Open another op',
        beacon: true,
      },
      {kind: 'click', duration: 950},
    );
  }
  beats.push({
    kind: 'keys',
    keys: ['[', ']'],
    at: {x: strip.x + 12, y: strip.y + strip.h - 10},
    duration: 1700,
    caption: 'Flip between tabs',
  });
  return {
    start: offsetPoint(first, 60, 80, context.viewport),
    beats,
    rest: 1600,
  };
}

function shareDemo(context: DemoContext): DemoScript | undefined {
  const steppers = context.findAll('.kernel-bar .kernel-stepper .step-button');
  const next = steppers[steppers.length - 1];
  if (!next) {
    return undefined;
  }
  const link = context.find('.kernel-bar .copy-link-button');
  const beats: DemoBeat[] = [
    {
      kind: 'move',
      to: pointIn(next),
      duration: 950,
      caption: 'Next op in the ranking',
      beacon: true,
    },
    {kind: 'click', duration: 950},
  ];
  if (link) {
    beats.push(
      {
        kind: 'move',
        to: pointIn(link),
        duration: 750,
        caption: 'Copy a link to this op',
        beacon: true,
      },
      {kind: 'click', duration: 950},
    );
  }
  beats.push({kind: 'hold', duration: 700});
  return {
    start: offsetPoint(pointIn(next), -40, 90, context.viewport),
    beats,
    rest: 1700,
  };
}

function timelineDemo(context: DemoContext): DemoScript | undefined {
  const area = context.timeline;
  if (!area) {
    return undefined;
  }
  return {
    start: pointIn(area, 0.06, 0.3),
    beats: [
      {
        kind: 'move',
        to: pointIn(area, 0.16, 0.38),
        duration: 650,
        caption: 'Point at any block',
      },
      {kind: 'move', to: pointIn(area, 0.36, 0.62), duration: 1200},
      {kind: 'move', to: pointIn(area, 0.58, 0.48), duration: 1200},
      {kind: 'move', to: pointIn(area, 0.72, 0.7), duration: 1100},
      {kind: 'hold', duration: 600},
    ],
    rest: 1400,
  };
}

function navigateDemo(context: DemoContext): DemoScript | undefined {
  const area = context.timeline;
  if (!area) {
    return undefined;
  }
  const keysAt = pointIn(area, 0.4, 0.42);
  const from = pointIn(area, 0.3, 0.6);
  const to = pointIn(area, 0.58, 0.6);
  const band = (x: number): GhostBox => ({
    x: Math.min(from.x, x),
    y: area.y,
    w: Math.abs(x - from.x),
    h: area.h,
    bundlesPerPixel: context.bundlesPerPixel,
    label: context.bundlesPerPixel === undefined ? 'Measured span' : undefined,
  });
  return {
    start: pointIn(area, 0.2, 0.8),
    beats: [
      {
        kind: 'keys',
        keys: ['W'],
        at: keysAt,
        duration: 1000,
        caption: 'Zoom in',
      },
      {
        kind: 'keys',
        keys: ['S'],
        at: keysAt,
        duration: 1000,
        caption: 'Zoom out',
      },
      {
        kind: 'keys',
        keys: ['A'],
        at: keysAt,
        duration: 1000,
        caption: 'Pan left',
      },
      {
        kind: 'keys',
        keys: ['D'],
        at: keysAt,
        duration: 1000,
        caption: 'Pan right',
      },
      {
        kind: 'move',
        to: from,
        duration: 800,
        caption: 'Drag to measure',
        keys: ['Shift'],
      },
      {
        kind: 'drag',
        to,
        duration: 1600,
        keys: ['Shift'],
        ghostFrom: band(from.x),
        ghost: band(to.x),
      },
      {kind: 'hold', duration: 900},
      {kind: 'hold', duration: 450, clearGhost: true},
    ],
    rest: 1500,
    focus: {x: from.x, y: from.y - 30, w: to.x - from.x, h: 60},
  };
}

function inspectDemo(context: DemoContext): DemoScript | undefined {
  const area = context.timeline;
  if (!area) {
    return undefined;
  }
  // A block that the user pointed at, or else where unit lanes usually are,
  // below the scope and region rows.
  const target = context.block?.point ?? pointIn(area, 0.42, 0.55);
  const beats: DemoBeat[] = [
    {
      kind: 'move',
      to: target,
      duration: 1150,
      caption: clickCaptionFor(context.block?.lane),
      beacon: true,
    },
    {kind: 'hold', duration: 250},
    {kind: 'click', duration: 950},
  ];
  const drawer = context.drawer;
  if (!context.isPanelOpen && drawer) {
    // Previews where its details open, like the panel sliding up.
    beats.push(
      {
        kind: 'hold',
        duration: 1400,
        caption: 'Its details open below',
        ghostFrom: {...drawer, y: drawer.y + drawer.h, h: 0},
        ghost: {...drawer, label: 'Details panel'},
      },
      {kind: 'hold', duration: 500, clearGhost: true},
    );
  } else {
    beats.push({
      kind: 'hold',
      duration: 1100,
      caption: 'Its details show below',
    });
  }
  return {
    start: offsetPoint(target, 110, -80, context.viewport),
    beats,
    rest: 1500,
    focus: boxAround(target, 48),
  };
}

const PANEL = 'kernel-selection-panel .selection-panel';

/**
 * Links of the details panel that the details steps follow, in order of
 * preference. Each one selects another block, or filters what the panel
 * lists, and the panel follows along.
 */
const PANEL_LINKS: ReadonlyArray<readonly [string, string]> = [
  [
    `${PANEL} .bundle-event-row:not(.is-selected)`,
    'Jump to an op in this bundle',
  ],
  [`${PANEL} .stacked-segment`, 'Filter by unit'],
  [
    `${PANEL} .child-region-row:not(.parent-region-row)`,
    'Open a child region',
  ],
  [`${PANEL} .next-lane-btn:not([disabled])`, 'Next block on this lane'],
  [`${PANEL} .prev-lane-btn:not([disabled])`, 'Previous block on this lane'],
  [`${PANEL} .top-instruction-row`, 'Jump to this instruction'],
];

/** The most metrics that the demo of a details step points at. */
const MAX_DEMO_METRICS = 4;

/** Where the numbered marker of a part of the viewer goes: left of it. */
function hotspotFor(box: Box): Point {
  return {x: box.x - 8, y: box.y + Math.min(box.h / 2, 11)};
}

/**
 * The demo of the details steps: numbers the metrics that the card explains
 * right on the panel, points at the first few, then follows a link.
 */
function panelDemo(context: DemoContext): DemoScript | undefined {
  const panel = context.find(PANEL);
  if (!panel) {
    return undefined;
  }
  const metrics = (context.metrics ?? []).map(({selector, caption}) => ({
    box: context.find(selector),
    caption,
  }));
  const hotspots = metrics.map(({box}) => (box ? hotspotFor(box) : undefined));
  const beats: DemoBeat[] = [];
  let first: Point | undefined = undefined;
  for (const {box, caption} of metrics.filter(({box}) => box)) {
    if (!box || beats.length >= 2 * MAX_DEMO_METRICS) {
      break;
    }
    const to = pointIn(box, Math.min(0.5, 28 / Math.max(box.w, 1)));
    first ??= to;
    beats.push(
      {kind: 'move', to, duration: 850, caption},
      {kind: 'hold', duration: 650},
    );
  }
  const link = PANEL_LINKS.map(
    ([selector, caption]) => [context.find(selector), caption] as const,
  ).find(([box]) => box);
  if (link?.[0]) {
    const to = pointIn(link[0], Math.min(0.5, 24 / Math.max(link[0].w, 1)));
    first ??= to;
    beats.push(
      {kind: 'move', to, duration: 900, caption: link[1], beacon: true},
      {kind: 'click', duration: 900},
      {kind: 'hold', duration: 500},
    );
  }
  if (!first) {
    return {start: pointIn(panel), beats: [], rest: 0, hotspots};
  }
  return {
    start: offsetPoint(first, 80, -90, context.viewport),
    beats,
    rest: 1800,
    hotspots,
  };
}

function minimapDemo(context: DemoContext): DemoScript | undefined {
  const track = context.find('kernel-minimap .minimap-track');
  const lens = context.find('kernel-minimap .minimap-lens');
  if (!track || !lens || track.w < 60) {
    return undefined;
  }
  // The pointer works over the unit lanes of the minimap, below its regions.
  const y = track.y + track.h * 0.66;
  const lensAt = (x: number, w: number): GhostBox => {
    const width = clamp(w, 8, track.w);
    return {
      x: clamp(x, track.x, track.x + track.w - width),
      y: track.y,
      w: width,
      h: track.h,
    };
  };
  const real = lensAt(lens.x, lens.w);
  const beats: DemoBeat[] = [];
  let ghost: GhostBox;
  if (real.w > track.w * 0.7) {
    // Zooms in first, by dragging the right edge of the lens.
    const zoomed = lensAt(real.x, track.w * 0.42);
    const panned = lensAt(zoomed.x + track.w * 0.3, zoomed.w);
    beats.push(
      {
        kind: 'move',
        to: {x: real.x + real.w - 3, y},
        duration: 950,
        caption: 'Drag an edge to zoom',
        beacon: true,
      },
      {
        kind: 'drag',
        to: {x: zoomed.x + zoomed.w - 3, y},
        duration: 1300,
        ghostFrom: real,
        ghost: zoomed,
      },
      {
        kind: 'move',
        to: {x: zoomed.x + zoomed.w / 2, y},
        duration: 650,
        caption: 'Drag the lens to pan',
      },
      {
        kind: 'drag',
        to: {x: panned.x + panned.w / 2, y},
        duration: 1300,
        ghost: panned,
      },
    );
    ghost = panned;
  } else {
    // Pans first, towards the side with more room, then widens the lens.
    const direction =
      real.x + real.w / 2 < track.x + track.w / 2 ? 1 : -1;
    const panned = lensAt(real.x + direction * track.w * 0.22, real.w);
    const widened = lensAt(
      panned.x,
      Math.min(panned.w + track.w * 0.14, track.x + track.w - panned.x),
    );
    beats.push(
      {
        kind: 'move',
        to: {x: real.x + real.w / 2, y},
        duration: 950,
        caption: 'Drag the lens to pan',
        beacon: true,
      },
      {
        kind: 'drag',
        to: {x: panned.x + panned.w / 2, y},
        duration: 1300,
        ghostFrom: real,
        ghost: panned,
      },
      {
        kind: 'move',
        to: {x: panned.x + panned.w - 3, y},
        duration: 650,
        caption: 'Drag an edge to zoom',
      },
      {
        kind: 'drag',
        to: {x: widened.x + widened.w - 3, y},
        duration: 1100,
        ghost: widened,
      },
    );
    ghost = widened;
  }
  // Jumps to the far side of the schedule.
  const isLeft = ghost.x + ghost.w / 2 < track.x + track.w / 2;
  const jumpX = track.x + track.w * (isLeft ? 0.86 : 0.14);
  beats.push(
    {
      kind: 'move',
      to: {x: jumpX, y},
      duration: 800,
      caption: 'Click to jump there',
    },
    {kind: 'click', duration: 950, ghost: lensAt(jumpX - ghost.w / 2, ghost.w)},
    {kind: 'hold', duration: 250, caption: 'Double-click to see it all'},
    {kind: 'click', count: 2, duration: 1150, ghost: {...track}},
    {kind: 'hold', duration: 700, clearGhost: true},
  );
  return {
    start: offsetPoint(pointIn(track, 0.5, 1), 30, 70, context.viewport),
    beats,
    rest: 1800,
    focus: track,
  };
}

function doneDemo(context: DemoContext): DemoScript | undefined {
  const button = context.find('.kernel-bar .tour-button');
  if (!button) {
    return undefined;
  }
  return {
    start: offsetPoint(pointIn(button), -70, 90, context.viewport),
    beats: [
      {
        kind: 'move',
        to: pointIn(button),
        duration: 1000,
        caption: 'Replay the tour here',
        beacon: true,
      },
      {kind: 'hold', duration: 1800},
    ],
    rest: 2600,
  };
}

const DEMOS: ReadonlyMap<
  string,
  (context: DemoContext) => DemoScript | undefined
> = new Map([
  ['rail', railDemo],
  ['search', searchDemo],
  ['tabs', tabsDemo],
  ['share', shareDemo],
  ['timeline', timelineDemo],
  ['navigate', navigateDemo],
  ['inspect', inspectDemo],
  ['panel', panelDemo],
  ['region', panelDemo],
  ['minimap', minimapDemo],
  ['done', doneDemo],
]);

/** Whether the step `stepId` has a demo. */
export function hasDemo(stepId: string): boolean {
  return DEMOS.has(stepId);
}

/**
 * Builds the demo of the step `stepId` from the elements of the viewer, or
 * returns undefined if the step has no demo, or its elements are not shown.
 */
export function buildDemoScript(
  stepId: string,
  context: DemoContext,
): DemoScript | undefined {
  return DEMOS.get(stepId)?.(context);
}
