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
 */
const FEATURE_FLAGS: FeatureFlag[] = [
  {
    id: 'use_pb',
    name: 'Use Protobuf Pipeline in Trace Viewer',
    description: 'Enable the new protobuf-based data pipeline in Trace Viewer.',
    default: false,
  },
  {
    id: 'snap_to_time_range',
    name: 'Snap to Time Range',
    description: 'Enable snapping to event boundaries and selected ranges.',
    default: false,
  },
];

/**
 * Wrapper around feature flags to allow spying in tests.
 */
export const featureFlagsInternal = {
  getFeatureFlags: (): FeatureFlag[] => {
    return FEATURE_FLAGS;
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
  const flag = featureFlagsInternal.getFeatureFlags().find((f) => f.id === id);
  return flag?.default ?? false;
}
