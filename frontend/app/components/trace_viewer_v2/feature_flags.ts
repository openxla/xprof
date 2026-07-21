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

/**
 * The list of all available feature flags.
 * Enforced to be unique by design using object keys.
 */
const FEATURE_FLAGS = {
  'use_pb': {
    name: 'Use Protobuf Pipeline in Trace Viewer',
    description:
      'Use the high-performance protobuf data pipeline in Trace Viewer. Uncheck to fallback to JSON.',
    default: true,
  },
  'snap_to_time_range': {
    name: 'Snap to Time Range',
    description: 'Enable snapping to event boundaries and selected ranges.',
    default: false,
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
  'fullscreen': {
    name: 'Fullscreen Mode',
    description: 'Enable the fullscreen button in the Trace Viewer.',
    default: false,
  },
} as const;

/**
 * Represents the union of all valid feature flag identifiers.
 * Derived dynamically from the keys of the static FEATURE_FLAGS configuration.
 */
export type FeatureFlagId = keyof typeof FEATURE_FLAGS;

/** Pre-computed array of feature flags to avoid GC allocation on every call. */
const FEATURE_FLAGS_ARRAY: FeatureFlag[] = Object.entries(FEATURE_FLAGS).map(
  ([id, flag]) => ({
    id,
    ...flag,
  }),
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
    const flag = FEATURE_FLAGS[id as FeatureFlagId];
    return flag?.default ?? false;
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
