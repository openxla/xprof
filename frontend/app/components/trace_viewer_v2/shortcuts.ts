/**
 * Mouse modes for trace viewer interaction.
 * Must match the values in C++ MouseMode enum.
 */
export enum MouseMode {
  SELECT = 1,
  PAN = 2,
  ZOOM = 3,
  TIMING = 4,
}

/**
 * Standard keyboard and mouse action key tokens.
 */
export const ShortcutKey = {
  // Mouse
  CLICK: 'Click',
  DRAG: 'Drag',
  CLICK_OR_DRAG: 'Click/Drag',
  SCROLL_WHEEL: 'Scroll Wheel',
  // Modifiers
  SHIFT: 'Shift',
  // Navigation & Control
  W: 'W',
  S: 'S',
  A: 'A',
  D: 'D',
  F: 'F',
  Z: 'Z',
  ZERO: '0',
  ARROW_LEFT: '←',
  ARROW_RIGHT: '→',
  // General Shortcuts
  SPACE: 'Space',
  SLASH: '/',
  ENTER: 'Enter',
  SHIFT_ENTER: 'Shift+Enter',
  M: 'M',
  SEMICOLON: ';',
  QUESTION: '?',
} as const;

/**
 * Visual key separators.
 */
export enum KeySeparator {
  OR = ' / ',
  COMBO = ' + ',
}

/**
 * Defines a single keyboard shortcut mapping.
 */
export interface ShortcutItem {
  description: string;
  iconName?: string;
  keys: string[];
  separator?: string;
  context?: string;
}

/**
 * Groups multiple shortcuts together into a thematic section.
 */
export interface ShortcutSection {
  title: string;
  items: ShortcutItem[];
}

/**
 * Configuration for status bar hints associated with a specific mouse mode.
 */
export interface MouseModeStatusConfig {
  icon: string;
  hints: ShortcutItem[];
}

/**
 * Single modular definition connecting a mouse mode's dialog display, hotkey, and HUD hints.
 */
export interface MouseModeDefinition {
  mode: MouseMode;
  name: string;
  hotkey: string;
  matIcon: string;
  svgIcon: string;
  hudHints: ShortcutItem[];
}

/**
 * Common navigation shortcuts displayed across all mouse modes in the status bar HUD.
 */
export const HUD_COMMON_NAVIGATION_HINTS: readonly ShortcutItem[] = Object.freeze([
  {
    description: 'Zoom',
    keys: [ShortcutKey.W, ShortcutKey.S],
    separator: KeySeparator.OR,
    context: 'GLOBAL',
  },
  {
    description: 'Pan',
    keys: [ShortcutKey.A, ShortcutKey.D],
    separator: KeySeparator.OR,
    context: 'GLOBAL',
  },
]);

/**
 * Reusable selection modifier shortcut.
 */
export const HUD_ADD_SELECTION_HINT: ShortcutItem = Object.freeze({
  description: 'Add Selection',
  keys: [ShortcutKey.SHIFT, ShortcutKey.CLICK_OR_DRAG],
  separator: KeySeparator.COMBO,
  context: 'SELECTION',
});

/**
 * The modular registry of all supported mouse modes and their shortcut behaviors.
 */
export const MOUSE_MODE_DEFINITIONS: readonly MouseModeDefinition[] = Object.freeze([
  {
    mode: MouseMode.SELECT,
    name: 'Select Mode',
    hotkey: '1',
    matIcon: 'mouse',
    svgIcon: 'select',
    hudHints: [
      {
        description: 'Box Select',
        keys: [ShortcutKey.CLICK, ShortcutKey.DRAG],
        separator: KeySeparator.OR,
        context: 'SELECT',
      },
      HUD_ADD_SELECTION_HINT,
      ...HUD_COMMON_NAVIGATION_HINTS,
    ],
  },
  {
    mode: MouseMode.PAN,
    name: 'Pan Mode',
    hotkey: '2',
    matIcon: 'pan_tool',
    svgIcon: 'pan',
    hudHints: [
      {
        description: 'Pan Left/Right',
        keys: [ShortcutKey.DRAG],
        context: 'PAN',
      },
      HUD_ADD_SELECTION_HINT,
      ...HUD_COMMON_NAVIGATION_HINTS,
    ],
  },
  {
    mode: MouseMode.ZOOM,
    name: 'Zoom Mode',
    hotkey: '3',
    matIcon: 'search',
    svgIcon: 'zoom',
    hudHints: [
      {
        description: 'Vertical Zoom',
        keys: [ShortcutKey.DRAG],
        context: 'ZOOM',
      },
      ...HUD_COMMON_NAVIGATION_HINTS,
    ],
  },
  {
    mode: MouseMode.TIMING,
    name: 'Measure Mode',
    hotkey: '4',
    matIcon: 'straighten',
    svgIcon: 'measure',
    hudHints: [
      {
        description: 'Measure Time',
        keys: [ShortcutKey.DRAG],
        context: 'TIMING',
      },
      {
        description: 'Add Measurement',
        keys: [ShortcutKey.SHIFT, ShortcutKey.CLICK_OR_DRAG],
        separator: KeySeparator.COMBO,
        context: 'TIMING',
      },
      ...HUD_COMMON_NAVIGATION_HINTS,
    ],
  },
]);

/**
 * Status bar configurations for each mouse mode.
 * Keyed by MouseMode integer for O(1) lookup.
 */
export const MOUSE_MODE_STATUS_CONFIGS: Readonly<
  Record<number, MouseModeStatusConfig>
> = Object.freeze(
  Object.fromEntries(
    MOUSE_MODE_DEFINITIONS.map((def) => [
      def.mode,
      {
        icon: def.matIcon,
        hints: def.hudHints,
      },
    ]),
  ),
);

/**
 * Returns the status bar configuration for the given mouse mode.
 */
export function getMouseModeStatusConfig(
  mode: number | null | undefined,
): MouseModeStatusConfig | undefined {
  if (mode === null || mode === undefined) return undefined;
  return MOUSE_MODE_STATUS_CONFIGS[mode];
}

/**
 * The definitive list of all XProf trace viewer keyboard shortcuts and mouse controls.
 */
export const TRACE_VIEWER_SHORTCUTS: ShortcutSection[] = [
  {
    title: 'Navigation',
    items: [
      {
        description: 'Zoom in / out',
        keys: [ShortcutKey.W, ShortcutKey.S],
        separator: KeySeparator.OR,
      },
      {
        description: 'Pan left / right',
        keys: [ShortcutKey.A, ShortcutKey.D],
        separator: KeySeparator.OR,
      },
      {
        description: 'Select prev / next event',
        keys: [ShortcutKey.ARROW_LEFT, ShortcutKey.ARROW_RIGHT],
        separator: KeySeparator.OR,
      },
      {
        description: 'Zoom to fit selection',
        keys: [ShortcutKey.F],
      },
      {
        description: 'Reset zoom and pan',
        keys: [ShortcutKey.Z, ShortcutKey.ZERO],
        separator: KeySeparator.OR,
      },
    ],
  },
  {
    title: 'Mouse Modes',
    items: MOUSE_MODE_DEFINITIONS.map((def) => ({
      description: def.name,
      iconName: def.svgIcon,
      keys: [def.hotkey],
    })),
  },
  {
    title: 'Mouse Controls',
    items: [
      {
        description: 'Select event',
        keys: [ShortcutKey.CLICK],
      },
      {
        description: 'Zoom in / out',
        keys: [ShortcutKey.SCROLL_WHEEL],
      },
      {
        description: 'Box select events',
        keys: [`${ShortcutKey.DRAG} (Mode 1)`],
      },
      {
        description: 'Pan timeline',
        keys: [`${ShortcutKey.DRAG} (Mode 2)`],
      },
      {
        description: 'Vertical zoom',
        keys: [`${ShortcutKey.DRAG} (Mode 3)`],
      },
      {
        description: 'Measure time range',
        keys: [`${ShortcutKey.DRAG} (Mode 4)`],
      },
      {
        description: 'Add selection / measure',
        keys: [ShortcutKey.SHIFT, ShortcutKey.CLICK_OR_DRAG],
        separator: KeySeparator.COMBO,
      },
    ],
  },
  {
    title: 'General',
    items: [
      {
        description: 'Search events',
        keys: [ShortcutKey.SLASH],
      },
      {
        description: 'Next / prev search result',
        keys: [ShortcutKey.ENTER, ShortcutKey.SHIFT_ENTER],
        separator: KeySeparator.OR,
      },
      {
        description: 'Bookmark selection',
        keys: [ShortcutKey.M],
      },
      {
        description: 'Open Settings',
        keys: [ShortcutKey.SEMICOLON],
      },
      {
        description: 'Play / Pause timeline',
        keys: [ShortcutKey.SPACE],
      },
      {
        description: 'Open Help menu',
        keys: [ShortcutKey.QUESTION],
      },
    ],
  },
];

