import {
  buildAdjacentNodesCacheKey,
  buildEventArgsCacheKey,
  buildNetworkCacheScope,
  shouldInvalidateNetworkCache,
} from './utils';

/**
 * Unit tests for Trace Viewer network-dedup key composition and scope
 * invalidation (FIX-015 / PR #2814 follow-up).
 *
 * Caches store entries under short event keys (no run/host). Callers clear the
 * maps when {@link buildNetworkCacheScope} changes so a prior profile cannot
 * serve stale event args or adjacent nodes.
 */
describe('trace_viewer network dedup cache keys', () => {
  describe('buildNetworkCacheScope', () => {
    it('joins run, tag, and host with | separators', () => {
      expect(buildNetworkCacheScope('runA', 'trace_viewer', 'host1')).toBe(
        'runA|trace_viewer|host1',
      );
    });

    it('preserves empty fields so missing host still changes scope', () => {
      expect(buildNetworkCacheScope('runA', 'tag', '')).toBe('runA|tag|');
      expect(buildNetworkCacheScope('', '', '')).toBe('||');
    });

    it('differs when only run changes', () => {
      const a = buildNetworkCacheScope('runA', 'tag', 'host');
      const b = buildNetworkCacheScope('runB', 'tag', 'host');
      expect(a).not.toBe(b);
    });

    it('differs when only host changes', () => {
      const a = buildNetworkCacheScope('run', 'tag', 'hostA');
      const b = buildNetworkCacheScope('run', 'tag', 'hostB');
      expect(a).not.toBe(b);
    });

    it('differs when only tag/tool changes', () => {
      const a = buildNetworkCacheScope('run', 'trace_viewer', 'host');
      const b = buildNetworkCacheScope('run', 'trace_viewer@', 'host');
      expect(a).not.toBe(b);
    });
  });

  describe('buildEventArgsCacheKey', () => {
    it('composes name:startUs', () => {
      expect(buildEventArgsCacheKey('op.fusion', 12345000)).toBe(
        'op.fusion:12345000',
      );
    });

    it('does not embed run/host (invalidation is scope-based)', () => {
      const key = buildEventArgsCacheKey('same', 1);
      expect(key.includes('|')).toBe(false);
      expect(key).toBe('same:1');
    });
  });

  describe('buildAdjacentNodesCacheKey', () => {
    it('composes nodeName-moduleName', () => {
      expect(buildAdjacentNodesCacheKey('%fusion.1', 'module_main')).toBe(
        '%fusion.1-module_main',
      );
    });
  });

  describe('shouldInvalidateNetworkCache', () => {
    it('does not invalidate on the first scope observation', () => {
      expect(
        shouldInvalidateNetworkCache(null, 'runA|tag|host'),
      ).toBe(false);
    });

    it('does not invalidate when scope is unchanged', () => {
      const scope = buildNetworkCacheScope('run', 'tag', 'host');
      expect(shouldInvalidateNetworkCache(scope, scope)).toBe(false);
    });

    it('invalidates when run changes (acceptance: changing run clears cache)', () => {
      const prev = buildNetworkCacheScope('runA', 'tag', 'host');
      const next = buildNetworkCacheScope('runB', 'tag', 'host');
      expect(shouldInvalidateNetworkCache(prev, next)).toBe(true);
    });

    it('invalidates when host changes', () => {
      const prev = buildNetworkCacheScope('run', 'tag', 'hostA');
      const next = buildNetworkCacheScope('run', 'tag', 'hostB');
      expect(shouldInvalidateNetworkCache(prev, next)).toBe(true);
    });
  });
});
