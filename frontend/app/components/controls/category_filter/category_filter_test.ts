import {categoryFilterOptionTitle} from './category_filter';

describe('categoryFilterOptionTitle', () => {
  it('returns the full string for long category names', () => {
    const longName =
        'namespace::VeryLongKernelNameThatDoesNotFitInOneHundredTwentyPixels';
    expect(categoryFilterOptionTitle(longName)).toBe(longName);
  });

  it('stringifies numbers and booleans used as filter values', () => {
    expect(categoryFilterOptionTitle(42)).toBe('42');
    expect(categoryFilterOptionTitle(true)).toBe('true');
    expect(categoryFilterOptionTitle(false)).toBe('false');
  });

  it('maps null and undefined to empty title attributes', () => {
    expect(categoryFilterOptionTitle(null)).toBe('');
    expect(categoryFilterOptionTitle(undefined)).toBe('');
  });
});
