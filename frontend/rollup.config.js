const { nodeResolve } = require('@rollup/plugin-node-resolve');
const commonjs = require('@rollup/plugin-commonjs');

const path = require('path');

/**
 * A Rollup plugin to resolve imports starting with 'org_xprof/' by aliasing them
 * to paths relative to the current working directory (workspace root).
 * @return {!Object} A Rollup plugin object.
 */
function workspaceAlias() {
  return {
    name: 'workspace-alias',
    async resolveId(source, importer) {
      if (source.startsWith('org_xprof/')) {
        const target = source.replace('org_xprof/', '');
        const absoluteTarget = path.resolve(process.cwd(), target);
        const resolution = await this.resolve(absoluteTarget, importer, { skipSelf: true, custom: { 'workspace-alias': true } });
        if (resolution) return resolution;
        return absoluteTarget;
      }
      return null;
    }
  };
}

module.exports = {
  plugins: [
    workspaceAlias(),
    nodeResolve({
      mainFields: ['browser', 'es2015', 'module', 'jsnext:main', 'main'],
    }),
    commonjs(),
  ],
  output: {
    // Name field is required for iife|umd file format
    name: 'tbp',
    strict: false,
    format: 'iife',
    sourcemap: false,
  },
  onwarn: (warning, warn) => {
    // Typescript decorator transpiled code checks `this` in case there are
    // global decorator. Rollup warns `this` is undefined.
    // This is safe to ignore.
    if (warning.code === 'THIS_IS_UNDEFINED') {
      return;
    }
    warn(warning);
  }
};
