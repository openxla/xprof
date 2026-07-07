import {
  FilterField,
  FilterFieldCategory,
  FilterOperator,
  FilterOperatorId,
  FilterOperatorType,
} from './trace_viewer_typings';

// LINT.IfChange
/** Filter field name for event duration, in ms. */
export const FILTER_FIELD_EVENT_DURATION = 'duration';

/** Constants of filter fields. */
// TODO(jonahweaver) strip start_time_ms and end_time_ms from the filter category
export const FILTER_FIELDS: FilterField[] = [
  {
    value: FilterFieldCategory.HOST,
    info: {category: FilterFieldCategory.HOST},
    displayName: 'host',
    operatorTypes: [FilterOperatorType.EXACT, FilterOperatorType.REGEX],
    hasMultiSelectOptions: true,
  },
  {
    value: FilterFieldCategory.PROCESS,
    info: {category: FilterFieldCategory.PROCESS},
    displayName: 'process',
    operatorTypes: [FilterOperatorType.EXACT, FilterOperatorType.REGEX],
    hasMultiSelectOptions: true,
  },
  {
    value: FilterFieldCategory.THREAD,
    info: {category: FilterFieldCategory.THREAD},
    displayName: 'thread',
    operatorTypes: [FilterOperatorType.REGEX],
  },
  {
    value: FilterFieldCategory.EVENT + '_name',
    info: {category: FilterFieldCategory.EVENT, name: 'name'},
    displayName: 'event',
    operatorTypes: [FilterOperatorType.REGEX],
  },
  {
    value: FilterFieldCategory.START_TIME_MS,
    info: {category: FilterFieldCategory.START_TIME_MS},
    displayName: 'start(ms)',
    operatorTypes: [FilterOperatorType.EXACT],
  },
  {
    value: FilterFieldCategory.END_TIME_MS,
    info: {category: FilterFieldCategory.END_TIME_MS},
    displayName: 'end(ms)',
    operatorTypes: [FilterOperatorType.EXACT],
  },
  {
    value: FilterFieldCategory.EVENT + '_duration',
    displayName: 'event_duration(ms)',
    info: {
      category: FilterFieldCategory.EVENT,
      name: FILTER_FIELD_EVENT_DURATION,
    },
    operatorTypes: [
      FilterOperatorType.EXACT,
      FilterOperatorType.GREATER_THAN,
      FilterOperatorType.GREATER_THAN_EQUAL,
      FilterOperatorType.LESS_THAN,
      FilterOperatorType.LESS_THAN_EQUAL,
    ],
  },
];

/** Array of filter operators.
 * Operators that are of length 2 will always end with an '='
 * If this changes, this logic will need to be updated.
 */
export const FILTER_OPERATORS: FilterOperator[] = [
  {
    displayName: 'Exact (=)',
    value: FilterOperatorType.EXACT,
    opId: FilterOperatorId.EQUAL,
  },
  {
    displayName: 'Less than (<)',
    value: FilterOperatorType.LESS_THAN,
    opId: FilterOperatorId.LESS_THAN,
  },
  {
    displayName: 'Greater than (>)',
    value: FilterOperatorType.GREATER_THAN,
    opId: FilterOperatorId.GREATER_THAN,
  },
  {
    displayName: 'Less than or equal to (<=)',
    value: FilterOperatorType.LESS_THAN_EQUAL,
    opId: FilterOperatorId.LESS_THAN_EQUAL,
  },
  {
    displayName: 'Greater than or equal to (>=)',
    value: FilterOperatorType.GREATER_THAN_EQUAL,
    opId: FilterOperatorId.GREATER_THAN_EQUAL,
  },
  {
    displayName: 'Regex (~)',
    value: FilterOperatorType.REGEX,
    opId: FilterOperatorId.REGEX,
  },
];

/** Query param name for trace filter config */
export const FILTER_CONFIG = 'trace_filter_config';

/** Query param name for session id */
export const SESSION_ID_QUERY_PARAM = 'session_id';

/** Host filter value for all hosts. */
export enum HostFilter {
  ALL = 'all',
  NONE = 'none',
}

/** Key for mpmd pipeline view query param. */
export const MPMD_PIPELINE_VIEW_PARAM = 'mpmd_pipeline_view';

/** Key for use_trace_viewer_v2 query param and local storage. */
export const USE_TRACE_VIEWER_V2_KEY = 'use_trace_viewer_v2';

/**
 * Tool name and label for stack trace page for cross-links.
 */
export const STACK_TRACE_TOOL_NAME = [
  'stack_trace_page',
  'Source Code Snippet with IR Text',
];
/**
 * Tool name and label for roofline model for cross-links.
 */
export const ROOFLINE_MODEL_TOOL_NAME = ['roofline_model', 'Roofline Model'];

/**
 * Separates event filter properties in the string representation of the filter
 * config
 */
export const FILTER_SEPARATOR = '%';

/**
 * Separates event filters, represented as a collection of properties, in the
 * string representation of the filter config
 * String format:
 * <value1>@<operator1>@<field1>%<value2>@<operator2>@<field2>...
 *
 * Example for filtering event name by regex match of "memcpy":
 * memcpy@5@name
 *
 * Example for filtering event name by exact match of "memcpy" and duration longer
 * than 2 ms:
 * memcpy@0@name%2@2@duration
 */
export const FILTER_PROPERTY_SEPARATOR = '@';

/**
 * The id of the flow category that represents no flow category.
 */
export const NONE_FLOW_CATEGORY_ID = -2;

/**
 * The id of the flow category that represents all flow categories.
 */
export const ALL_FLOW_CATEGORY_ID = -1;

/**
 * The name of the viewport changed custom event, dispatched from WASM in Trace
 * Viewer v2.
 */
export const VIEWPORT_CHANGED_EVENT_NAME = 'viewport-changed';

/**
 * The name of the palette chosen for Trace Viewer v2.
 */
export const COLOR_PALETTE_STORAGE_KEY = 'trace_viewer_palette';

/**
 * The key to store whether the user has been prompted about the new color palette.
 */
export const COLOR_PALETTE_PROMPTED_STORAGE_KEY =
  'trace_viewer_palette_prompted';

/**
 * Local storage prefix for feature flag keys.
 */
export const FEATURE_FLAG_STORAGE_PREFIX = 'xprof_ff_';

/**
 * Available palettes for Trace Viewer v2.
 */
export const COLOR_PALETTES = [
  'Default',
  'Material',
  'Dracula',
  'Monokai',
  'Solarized Dark',
  'Solarized Light',
  'Catapult',
];
