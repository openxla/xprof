/**
 * Defines a single keyboard shortcut mapping.
 */
export interface ShortcutItem {
  description: string;
  // A template string or raw HTML string if we want to include SVGs,
  // but let's keep it simple: just an optional icon name or boolean if it's a mouse mode.
  iconName?: string;
  keys: string[]; // e.g., ['W', 'S'] for W / S, or ['Shift', 'Click/Drag'] for Shift + Click/Drag
  // How the keys are separated visually. Default is ' / ' if not specified, except for combo keys which might be ' + '.
  separator?: string;
  // Context to which this shortcut belongs, e.g., "GLOBAL" or "SELECT_MODE". Useful for status bar filtering.
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
 * The definitive list of all XProf trace viewer keyboard shortcuts and mouse controls.
 */
export const TRACE_VIEWER_SHORTCUTS: ShortcutSection[] = [
  {
    title: 'Navigation',
    items: [
      {description: 'Zoom in / out', keys: ['W', 'S'], separator: ' / '},
      {description: 'Pan left / right', keys: ['A', 'D'], separator: ' / '},
      {
        description: 'Select prev / next event',
        keys: ['←', '→'],
        separator: ' / ',
      },
      {description: 'Zoom to fit selection', keys: ['F']},
      {description: 'Reset zoom and pan', keys: ['Z', '0'], separator: ' / '},
    ],
  },
  {
    title: 'Mouse Modes',
    items: [
      {description: 'Select Mode', iconName: 'select', keys: ['1']},
      {description: 'Pan Mode', iconName: 'pan', keys: ['2']},
      {description: 'Zoom Mode', iconName: 'zoom', keys: ['3']},
      {description: 'Measure Mode', iconName: 'measure', keys: ['4']},
    ],
  },
  {
    title: 'Mouse Controls',
    items: [
      {description: 'Select event', keys: ['Click']},
      {description: 'Zoom in / out', keys: ['Scroll Wheel']},
      {description: 'Box select events', keys: ['Drag (Mode 1)']},
      {description: 'Pan timeline', keys: ['Drag (Mode 2)']},
      {description: 'Vertical zoom', keys: ['Drag (Mode 3)']},
      {description: 'Measure time range', keys: ['Drag (Mode 4)']},
      {
        description: 'Add selection / measure',
        keys: ['Shift', 'Click/Drag'],
        separator: ' + ',
      },
    ],
  },
  {
    title: 'General',
    items: [
      {description: 'Search events', keys: ['/']},
      {
        description: 'Next / prev search result',
        keys: ['Enter', 'Shift+Enter'],
        separator: ' / ',
      },
      {description: 'Bookmark selection', keys: ['M']},
      {description: 'Open Settings', keys: [';']},
      {description: 'Play / Pause timeline', keys: ['Space']},
      {description: 'Open Help menu', keys: ['?']},
    ],
  },
];
