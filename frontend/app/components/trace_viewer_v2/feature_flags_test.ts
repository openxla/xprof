import {
  FeatureFlag,
  getAllFeatureFlags,
  getDefaultFeatureFlag,
  getFeatureFlags,
} from './feature_flags';

/**
 * Runtime guards for FEATURE_FLAGS.
 *
 * Object keys already enforce unique ids at the TypeScript level. This suite
 * additionally asserts unique display labels (names) and non-empty metadata so
 * CI fails if two flags share a UI label or a flag is registered without a
 * description.
 */
describe('FEATURE_FLAGS', () => {
  let flags: FeatureFlag[];

  beforeEach(() => {
    flags = getFeatureFlags();
  });

  it('exposes a non-empty list via getFeatureFlags and getAllFeatureFlags', () => {
    expect(flags.length).toBeGreaterThan(0);
    expect(getAllFeatureFlags()).toEqual(flags);
  });

  it('has unique ids', () => {
    const ids = flags.map((f) => f.id);
    expect(new Set(ids).size).toBe(ids.length);
  });

  it('has unique display names (labels)', () => {
    const names = flags.map((f) => f.name);
    const duplicates = names.filter(
      (name, index) => names.indexOf(name) !== index,
    );
    expect(duplicates).toEqual([]);
    expect(new Set(names).size).toBe(names.length);
  });

  it('has non-empty ids, names, and descriptions', () => {
    for (const flag of flags) {
      expect(flag.id.trim().length > 0).toBe(true);
      expect(flag.name.trim().length > 0).toBe(true);
      expect(flag.description.trim().length > 0).toBe(true);
    }
  });

  it('returns the registered default for known ids', () => {
    for (const flag of flags) {
      expect(getDefaultFeatureFlag(flag.id)).toBe(flag.default);
    }
  });

  it('defaults unknown flag ids to false', () => {
    expect(getDefaultFeatureFlag('__not_a_real_feature_flag__')).toBe(false);
  });
});
