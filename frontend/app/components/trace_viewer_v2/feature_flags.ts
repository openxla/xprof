/**
 * Represents a feature flag configuration for the trace viewer.
 */
export declare interface FeatureFlag {
  /** The unique identifier for the feature flag. */
  id: string;
  /** The human-readable name of the feature flag. */
  name: string;
  /** A detailed description of what the feature flag does. */
  description: string;
  /** The default value of the feature flag. */
  default: boolean;
}

const FEATURE_FLAGS = {
  'use_pb': {
    name: 'Use Protobuf Pipeline in Trace Viewer',
    description:
      'Use the high-performance protobuf data pipeline in Trace Viewer. Uncheck to fallback to JSON.',
    default: true,
  },
  'bookmarks': {
    name: 'Enable Bookmarks',
    description: 'Enable adding bookmarks with Ctrl/Meta + Click.',
    default: false,
  },
  'enable_track_management': {
    name: 'Enable Track Management',
    description: 'Enable the track management feature in Trace Viewer.',
    default: false,
  },
  'enable_customization': {
    name: 'Enable User Customization',
    description:
      'Enable user customization of timeline settings, including zooming/panning speed, color palettes, and shortcuts, etc.',
    default: false,
  },
} as const;

/**
 * Represents the union of all valid feature flag identifiers.
 */
export type FeatureFlagId = keyof typeof FEATURE_FLAGS;

/** Pre-computed array of feature flags to avoid GC allocation on every call. */
const FEATURE_FLAGS_ARRAY: FeatureFlag[] = Object.entries(FEATURE_FLAGS).map(
  ([id, flag]) => ({
    id,
    name: flag.name,
    description: flag.description,
    default: flag.default,
  }),
);

const FEATURE_FLAGS_MAP = new Map<string, FeatureFlag>(
  FEATURE_FLAGS_ARRAY.map((f) => [f.id, f]),
);

/**
 * Wrapper around feature flags to allow spying in tests.
 */
export const featureFlagsInternal = {
  getFeatureFlags: (): FeatureFlag[] => {
    return FEATURE_FLAGS_ARRAY;
  },
};

/**
 * Returns the list of all available feature flags.
 */
export function getFeatureFlags(): FeatureFlag[] {
  return featureFlagsInternal.getFeatureFlags();
}

/**
 * Gets the default value for a feature flag.
 */
export function getDefaultFeatureFlag(id: string): boolean {
  const flags = featureFlagsInternal.getFeatureFlags();
  // Fast path for production: avoid iterating or allocating when using static flags.
  if (flags === FEATURE_FLAGS_ARRAY) {
    return FEATURE_FLAGS_MAP.get(id)?.default ?? false;
  }
  // Fallback path for unit tests to support spies/mocks.
  const flag = flags.find((f) => f.id === id);
  return flag?.default ?? false;
}

/**
 * Gets all feature flags as an array.
 * Useful for UI components that need to list flags.
 */
export function getAllFeatureFlags(): FeatureFlag[] {
  return FEATURE_FLAGS_ARRAY;
}

/**
 * Storage key prefix for feature flags in localStorage.
 */
export const FEATURE_FLAG_STORAGE_PREFIX = 'xprof_ff_';

/**
 * Reads a feature flag's persisted value from localStorage.
 */
export function getStoredFeatureFlag(id: string): boolean {
  try {
    const storedValue = window.localStorage.getItem(
      FEATURE_FLAG_STORAGE_PREFIX + id,
    );
    if (storedValue === 'true') return true;
    if (storedValue === 'false') return false;
  } catch {
    // Ignore localStorage access failures (e.g. sandboxed iframe).
  }
  return getDefaultFeatureFlag(id);
}

/**
 * Saves a feature flag's value to localStorage, purging default values and legacy keys.
 */
export function saveFeatureFlag(
  id: string,
  value: boolean,
  defaultValue: boolean,
): void {
  const key = FEATURE_FLAG_STORAGE_PREFIX + id;
  try {
    if (value === defaultValue) {
      window.localStorage.removeItem(key);
    } else {
      window.localStorage.setItem(key, value ? 'true' : 'false');
    }
    if (id === 'use_pb') {
      window.localStorage.removeItem('use_pb');
      window.localStorage.removeItem('use_pb_format');
    }
  } catch {
    // Ignore localStorage write failures (e.g. sandboxed iframe).
  }
}
