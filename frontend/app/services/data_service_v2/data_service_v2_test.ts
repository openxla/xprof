import {
  TOOL_URL_SCHEMA_VERSION,
  TOOL_URL_VERSION_PARAM,
  buildToolUrlSearchParams,
  parseToolUrl,
  parseToolUrlSearchParams,
} from './data_service_v2';

/**
 * Unit tests for createToolUrl deep-link schema versioning (FIX-013 / #2899).
 *
 * New links emit `v=1`. Parsers accept missing `v` as schema version 1 so
 * pre-versioned bookmarks keep working.
 */
describe('createToolUrl deep-link schema version', () => {
  describe('buildToolUrlSearchParams', () => {
    it('always sets v to the current schema version', () => {
      const params = buildToolUrlSearchParams({
        toolName: 'graph_viewer',
        sessionId: 'run_abc',
        params: {node_name: '%fusion.1', module_name: 'main'},
      });
      expect(params.get(TOOL_URL_VERSION_PARAM)).toBe(
        String(TOOL_URL_SCHEMA_VERSION),
      );
      expect(TOOL_URL_SCHEMA_VERSION).toBe(1);
    });

    it('sets tool, run, tool-specific params, and optional paths', () => {
      const params = buildToolUrlSearchParams({
        toolName: 'op_profile',
        sessionId: 'session-1',
        params: {host: 'worker0'},
        sessionPath: '/tmp/sess',
        runPath: '/tmp/run',
      });
      expect(params.get('tool')).toBe('op_profile');
      expect(params.get('run')).toBe('session-1');
      expect(params.get('host')).toBe('worker0');
      expect(params.get('session_path')).toBe('/tmp/sess');
      expect(params.get('run_path')).toBe('/tmp/run');
    });

    it('omits empty tool-specific param values', () => {
      const params = buildToolUrlSearchParams({
        toolName: 'trace_viewer',
        sessionId: 'r1',
        params: {keep: 'yes', drop: ''},
      });
      expect(params.get('keep')).toBe('yes');
      expect(params.has('drop')).toBe(false);
    });
  });

  describe('parseToolUrlSearchParams (compat)', () => {
    it('treats missing v as schema version 1 (old bookmarks)', () => {
      // Pre-versioned createToolUrl shape from #2899 (no `v` query param).
      const oldUrl =
        'tool=graph_viewer&run=run_old&module_name=main&node_name=%25fusion.1';
      const parsed = parseToolUrlSearchParams(oldUrl);
      expect(parsed.schemaVersion).toBe(1);
      expect(parsed.toolName).toBe('graph_viewer');
      expect(parsed.sessionId).toBe('run_old');
      expect(parsed.params['module_name']).toBe('main');
      expect(parsed.params['node_name']).toBe('%fusion.1');
    });

    it('parses explicit v=1 as schema version 1', () => {
      const params = buildToolUrlSearchParams({
        toolName: 'graph_viewer',
        sessionId: 'run_new',
        params: {module_name: 'main', node_name: '%add.2'},
      });
      const parsed = parseToolUrlSearchParams(params);
      expect(parsed.schemaVersion).toBe(1);
      expect(parsed.toolName).toBe('graph_viewer');
      expect(parsed.sessionId).toBe('run_new');
      expect(parsed.params['node_name']).toBe('%add.2');
      // Version key is reserved, not a tool param.
      expect(parsed.params[TOOL_URL_VERSION_PARAM]).toBeUndefined();
    });

    it('parses both old and new URLs to the same logical fields', () => {
      const oldSearch =
        'tool=trace_viewer&run=r1&session_path=%2Fsess&run_path=%2Frun';
      const newSearch = buildToolUrlSearchParams({
        toolName: 'trace_viewer',
        sessionId: 'r1',
        params: {},
        sessionPath: '/sess',
        runPath: '/run',
      }).toString();

      const oldParsed = parseToolUrlSearchParams(oldSearch);
      const newParsed = parseToolUrlSearchParams(newSearch);

      expect(oldParsed.schemaVersion).toBe(1);
      expect(newParsed.schemaVersion).toBe(1);
      expect(oldParsed.toolName).toBe(newParsed.toolName);
      expect(oldParsed.sessionId).toBe(newParsed.sessionId);
      expect(oldParsed.sessionPath).toBe(newParsed.sessionPath);
      expect(oldParsed.runPath).toBe(newParsed.runPath);
    });

    it('accepts a leading ? on the query string', () => {
      const parsed = parseToolUrlSearchParams('?tool=op_profile&run=r2');
      expect(parsed.schemaVersion).toBe(1);
      expect(parsed.toolName).toBe('op_profile');
      expect(parsed.sessionId).toBe('r2');
    });

    it('surfaces a future explicit version number when present', () => {
      const parsed = parseToolUrlSearchParams('v=2&tool=x&run=y');
      expect(parsed.schemaVersion).toBe(2);
      expect(parsed.toolName).toBe('x');
      expect(parsed.sessionId).toBe('y');
    });
  });

  describe('parseToolUrl (full href)', () => {
    it('parses createToolUrl-style hrefs with hash fragment', () => {
      const href =
        'https://example.test/?v=1&tool=graph_viewer&run=sess1&module_name=m#profile';
      const parsed = parseToolUrl(href);
      expect(parsed.schemaVersion).toBe(1);
      expect(parsed.toolName).toBe('graph_viewer');
      expect(parsed.sessionId).toBe('sess1');
      expect(parsed.params['module_name']).toBe('m');
    });

    it('parses old full hrefs without v as version 1', () => {
      const href =
        'https://example.test/?tool=graph_viewer&run=sess1&node_name=op#profile';
      const parsed = parseToolUrl(href);
      expect(parsed.schemaVersion).toBe(1);
      expect(parsed.toolName).toBe('graph_viewer');
      expect(parsed.params['node_name']).toBe('op');
    });
  });
});
