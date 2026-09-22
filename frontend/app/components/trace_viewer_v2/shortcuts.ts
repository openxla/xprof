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
  OR = '/',
  COMBO = '+',
  NONE = '',
}

/**
 * Configuration for a mouse control action.
 */
export interface MouseControlItem {
  label: string;
  icon?: string;
}

/**
 * Defines a single keyboard shortcut mapping.
 */
export interface ShortcutItem {
  description: string;
  iconName?: string;
  keys: string[];
  separator?: string;
  mouseControl?: MouseControlItem;
  context?: string;
}

/**
 * Groups multiple shortcuts together into a thematic section.
 */
export interface ShortcutSection {
  title: string;
  iconName?: string;
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
export const HUD_COMMON_NAVIGATION_HINTS: readonly ShortcutItem[] =
  Object.freeze([
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
export const HUD_ADD_MEASUREMENT_HINT: ShortcutItem = Object.freeze({
  description: 'Add Measurement',
  keys: [ShortcutKey.SHIFT, ShortcutKey.CLICK_OR_DRAG],
  separator: KeySeparator.COMBO,
  context: 'TIMING',
});

/**
 * The modular registry of all supported mouse modes and their shortcut behaviors.
 */
export const MOUSE_MODE_DEFINITIONS: readonly MouseModeDefinition[] =
  Object.freeze([
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
        HUD_ADD_MEASUREMENT_HINT,
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
        HUD_ADD_MEASUREMENT_HINT,
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
        HUD_ADD_MEASUREMENT_HINT,
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
        HUD_ADD_MEASUREMENT_HINT,
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
    title: 'Navigation & Zooming',
    iconName: 'navigation',
    items: [
      {
        description: 'Zoom in / out',
        keys: [ShortcutKey.W, ShortcutKey.S],
        separator: KeySeparator.OR,
        mouseControl: {
          label: 'Scroll Wheel',
          icon: 'mouse',
        },
      },
      {
        description: 'Pan left / right',
        keys: [ShortcutKey.A, ShortcutKey.D],
        separator: KeySeparator.OR,
        mouseControl: {
          label: 'Drag (Mode 2)',
          icon: 'pan',
        },
      },
      {
        description: 'Select prev / next event',
        keys: [ShortcutKey.ARROW_LEFT, ShortcutKey.ARROW_RIGHT],
        separator: KeySeparator.OR,
        mouseControl: {
          label: 'Click event',
          icon: 'click',
        },
      },
      {
        description: 'Zoom to fit selection',
        keys: [ShortcutKey.F],
        mouseControl: {
          label: 'Double Click',
          icon: 'double_click',
        },
      },
      {
        description: 'Reset zoom and pan',
        keys: [ShortcutKey.Z, ShortcutKey.ZERO],
        separator: KeySeparator.OR,
        mouseControl: {
          label: 'Double Click Header',
          icon: 'refresh',
        },
      },
    ],
  },
  {
    title: 'Mouse & Tool Modes',
    iconName: 'gesture',
    items: [
      {
        description: 'Select Mode',
        iconName: 'select',
        keys: ['1'],
        mouseControl: {
          label: 'Drag (Mode 1)',
          icon: 'timer',
        },
      },
      {
        description: 'Pan Mode',
        iconName: 'pan',
        keys: ['2'],
        mouseControl: {
          label: 'Drag (Mode 2)',
          icon: 'pan',
        },
      },
      {
        description: 'Zoom Mode',
        iconName: 'zoom',
        keys: ['3'],
        mouseControl: {
          label: 'Drag (Mode 3)',
          icon: 'zoom',
        },
      },
      {
        description: 'Measure Mode',
        iconName: 'measure',
        keys: ['4'],
        mouseControl: {
          label: 'Drag (Mode 4)',
          icon: 'measure',
        },
      },
    ],
  },
  {
    title: 'Selection & Canvas Controls',
    iconName: 'select_all',
    items: [
      {
        description: 'Select event',
        keys: [],
        mouseControl: {
          label: 'Click event',
          icon: 'click',
        },
      },
      {
        description: 'Box select events',
        keys: [],
        mouseControl: {
          label: 'Drag (Mode 1)',
          icon: 'timer',
        },
      },
      {
        description: 'Pan timeline',
        keys: [],
        mouseControl: {
          label: 'Drag (Mode 2)',
          icon: 'pan',
        },
      },
      {
        description: 'Vertical zoom',
        keys: [],
        mouseControl: {
          label: 'Drag (Mode 3)',
          icon: 'zoom',
        },
      },
      {
        description: 'Measure time range',
        keys: [],
        mouseControl: {
          label: 'Drag (Mode 4)',
          icon: 'measure',
        },
      },
      {
        description: 'Add selection / measure',
        keys: [ShortcutKey.SHIFT],
        mouseControl: {
          label: 'Shift + Click/Drag',
          icon: 'click',
        },
      },
      {
        description: 'Clear selection',
        keys: ['Esc'],
        mouseControl: {
          label: 'Click background',
          icon: 'click',
        },
      },
    ],
  },
  {
    title: 'General & Workflow',
    iconName: 'general',
    items: [
      {
        description: 'Search events',
        keys: [ShortcutKey.SLASH],
        mouseControl: {
          label: 'Click search bar',
          icon: 'search',
        },
      },
      {
        description: 'Next search result',
        keys: [ShortcutKey.ENTER],
        mouseControl: {
          label: 'Click next button',
          icon: 'click',
        },
      },
      {
        description: 'Previous search result',
        keys: [ShortcutKey.SHIFT, ShortcutKey.ENTER],
        separator: KeySeparator.COMBO,
        mouseControl: {
          label: 'Click prev button',
          icon: 'click',
        },
      },
      {
        description: 'Bookmark selection',
        keys: [ShortcutKey.M],
        mouseControl: {
          label: 'Click bookmark icon',
          icon: 'click',
        },
      },
      {
        description: 'Open Settings',
        keys: [ShortcutKey.SEMICOLON],
        mouseControl: {
          label: 'Click settings gear',
          icon: 'click',
        },
      },
      {
        description: 'Play / Pause timeline',
        keys: [ShortcutKey.SPACE],
        mouseControl: {
          label: 'Click play button',
          icon: 'click',
        },
      },
      {
        description: 'Open Help menu',
        keys: [ShortcutKey.QUESTION],
        mouseControl: {
          label: 'Click help icon',
          icon: 'click',
        },
      },
    ],
  },
];
