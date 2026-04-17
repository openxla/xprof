import {GVIZ_FOR_TESTING} from 'org_xprof/frontend/app/common/constants/testing';

/** Interface for injecting mock libraries into the window built-in object. */
declare interface WindowForTesting {
  google?: typeof GVIZ_FOR_TESTING;
  gtag?: Function;
}

/** Injects GVIZ_FOR_TESTING into the global window. */
export function injectMockGviz() {
  if ((window as unknown as Record<string, unknown>)['__WANT_REAL_GVIZ__']) {
    console.log(
      'Test: Skipping injectMockGviz because __WANT_REAL_GVIZ__ is true',
    );
    return;
  }
  // Need to cast through unknown because the types do not overlap.
  const windowForTesting = window as unknown as WindowForTesting;
  windowForTesting.google = GVIZ_FOR_TESTING;
}

/** Injects GA tag into the global window. */
export function injectGtag() {
  // Need to cast through unknown because the types do not overlap.
  const windowForTesting = window as unknown as WindowForTesting;
  windowForTesting.gtag = () => {};
}
