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
  /** Optional badge tag displayed next to flag name. */
  badge?: string;
}

/** Pre-computed configuration of feature flags. */
const FEATURE_FLAG_CONFIGS = [
  {
    id: 'use_pb',
    name: 'Use Protobuf Pipeline in Trace Viewer',
    description: 'Enable the new protobuf-based data pipeline in Trace Viewer.',
    default: false,
  },
  {
    id: 'bookmarks',
    name: 'Enable Bookmarks',
    description: 'Enable adding bookmarks with Ctrl/Meta + Click.',
    default: false,
  },
  {
    id: 'enable_track_management',
    name: 'Enable Track Management',
    description: 'Enable the track management feature in Trace Viewer.',
    default: false,
  },
  {
    id: 'enable_customization',
    name: 'Enable User Customization',
    description:
      'Enable user customization of timeline settings, including zooming/panning speed, color palettes, and shortcuts, etc.',
    default: false,
  },
  {
    id: 'enable_gm3',
    name: 'Enable Google Material 3 (GM3) Update',
    description:
      'Enable Google Material 3 (GM3) styling and component modernization.',
    default: false,
  },
  {
    id: 'enable_timeline_player',
    name: 'Enable Timeline Player',
    description: 'Enable the timeline player component for trace playback.',
    default: false,
  },
] as const;

/**
 * Represents the union of all valid feature flag identifiers.
 */
export type FeatureFlagId = (typeof FEATURE_FLAG_CONFIGS)[number]['id'];

/** Pre-computed array of feature flags to avoid GC allocation on every call. */
const FEATURE_FLAGS_ARRAY: FeatureFlag[] = FEATURE_FLAG_CONFIGS.map((flag) => ({
  ...flag,
}));

const FEATURE_FLAGS_MAP = new Map<string, FeatureFlag>(
  FEATURE_FLAGS_ARRAY.map((f) => [f.id, f]),
);

/**
 * Returns the list of all available feature flags.
 */
export function getFeatureFlags(): FeatureFlag[] {
  return featureFlagsInternal.getFeatureFlags();
}

/**
 * Internal helper to allow test overrides while avoiding circular imports.
 * In production, always returns the static FEATURE_FLAGS_ARRAY.
 */
export const featureFlagsInternal = {
  getFeatureFlags: (): FeatureFlag[] => FEATURE_FLAGS_ARRAY,
};

/**
 * Returns the default value for a feature flag.
 * Optimized for O(1) direct dictionary lookup without array iteration.
 *
 * @param id The id of the feature flag.
 * @return The default value of the feature flag, or false if not found.
 */
export function getDefaultFeatureFlag(
  id: FeatureFlagId | (string & {}),
): boolean {
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
 * Returns all available feature flags.
 * Returns the cached array to prevent GC allocations in production.
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
export function getStoredFeatureFlag(
  id: FeatureFlagId | (string & {}),
): boolean {
  let storedValue: string | null = null;
  try {
    storedValue = window.localStorage.getItem(
      `${FEATURE_FLAG_STORAGE_PREFIX}${id}`,
    );
  } catch {
    // Ignore localStorage access failures (e.g. sandboxed iframe).
  }
  if (storedValue === 'true') {
    return true;
  }
  if (storedValue === 'false') {
    return false;
  }
  return getDefaultFeatureFlag(id);
}

/**
 * Saves a feature flag's value to localStorage, purging default values and legacy keys.
 */
export function saveFeatureFlag(
  id: FeatureFlagId | (string & {}),
  value: boolean,
  defaultValue: boolean,
): void {
  const key = `${FEATURE_FLAG_STORAGE_PREFIX}${id}`;
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
