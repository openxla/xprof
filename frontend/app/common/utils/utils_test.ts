import {parseFrameworkOpType} from './utils';

/**
 * Unit coverage for framework op-type parsing (extracted in #2893).
 *
 * Covers the known-type path, empty-after-colon fallback from the op_name
 * path, and empty/undefined sentinels so UI regressions are caught next to
 * the pure helper rather than only in op_table_entry component tests.
 */
describe('parseFrameworkOpType', () => {
  describe('known op type after colon', () => {
    it('returns the type segment after the last colon', () => {
      expect(parseFrameworkOpType('gradients/layer/Einsum_1:Einsum'))
          .toBe('Einsum');
    });

    it('strips path prefixes inside the type segment', () => {
      // After colon: "path/to/MatMul" -> last path component only.
      expect(parseFrameworkOpType('scope:path/to/MatMul')).toBe('MatMul');
    });

    it('handles a bare type with no path prefix', () => {
      expect(parseFrameworkOpType('MatMul')).toBe('MatMul');
    });

    it('uses the last path component when there is no colon', () => {
      expect(parseFrameworkOpType('model/layer/Conv2D')).toBe('Conv2D');
    });
  });

  describe('fallback when type after colon is empty', () => {
    it('derives type from the last path component of the op name', () => {
      expect(parseFrameworkOpType('gradients/pointwise/dot_general:'))
          .toBe('dot_general');
    });

    it('preserves numeric suffixes on fallback names', () => {
      expect(parseFrameworkOpType('model/layer/MatMul_1:')).toBe('MatMul_1');
    });

    it('falls back to a single-segment name before the colon', () => {
      expect(parseFrameworkOpType('dot_general:')).toBe('dot_general');
    });
  });

  describe('empty / unknown / missing inputs', () => {
    it('returns "-" for undefined', () => {
      expect(parseFrameworkOpType(undefined)).toBe('-');
    });

    it('returns "-" for an empty string', () => {
      expect(parseFrameworkOpType('')).toBe('-');
    });

    it('returns "-" for a colon-only string', () => {
      expect(parseFrameworkOpType(':')).toBe('-');
    });
  });
});
