/** Enum for a filter operator type. */
export enum FilterOperatorType {
  EXACT = '=',
  REGEX = '~',
  GREATER_THAN = '>',
  GREATER_THAN_EQUAL = '>=',
  LESS_THAN = '<',
  LESS_THAN_EQUAL = '<=',
}

/** Enum for a filter field value. */
export enum FilterFieldCategory {
  HOST = 'host',
  PROCESS = 'process',
  THREAD = 'thread',
  EVENT = 'event',
  START_TIME_MS = 'start_time_ms',
  END_TIME_MS = 'end_time_ms',
}

/** Enum for a filter operator id. */
export enum FilterOperatorId {
  EQUAL = 0,
  LESS_THAN = 1,
  GREATER_THAN = 2,
  LESS_THAN_EQUAL = 3,
  GREATER_THAN_EQUAL = 4,
  REGEX = 5,
}

/** Typing for a filter option, eg. field, operator.. */
// FilterOption can be FilterField, FilterOperator, FilterValue.
// For other future interfaces that may be added, such as FilterField,
// we will need to add discrimination logic here.
export type FilterOption = FilterField | FilterOperator | FilterValue;

/** Typing for a filter value. */
export interface FilterValue {
  value: string;
  displayName?: string;
  checked?: boolean;
}

/** Typing for a filter field. */
export interface FilterField {
  value: string;
  info: FilterFieldInfo;
  displayName: string;
  operatorTypes?: string[];
  // If the filter field has prefetched option list to support multi-select (eg. host, process)
  hasMultiSelectOptions?: boolean;
}

/**
 * Stores the category (event, process, thread, host, etc.) and the name of
 * the field to filter by (name, duration, etc.).
 * The combination of category and name defines the unique key of a filter field (eg. event_duration)
 */
export interface FilterFieldInfo {
  category: FilterFieldCategory;
  name?: string;
}

/** Typing for a filter operator. */
export interface FilterOperator {
  displayName: string;
  value: string;
  opId: number;
}

/**
 * Filter state is represented via an array of FilterEntries.
 */
export interface FilterEntry {
  field: FilterField;
  value: string;
  operator: FilterOperator;
}

/**
 * Payload for filter changes event
 */
export interface FilterChangeEvent {
  value: string; // current value after filter change
  index: number; // index of changed filter in the selected filters list
}

/**
 * Payload for filter remove event
 */
export interface FilterRemoveEvent {
  index: number; // index of the filter to be removed in the selected filters list
}

/**
 * Concept & Var naming clarification:
 * (1) TraceFilters: corresponds to the trace_filter_config query param
 * (2) TraceViewConfig: the config saved/used for session trace filtering,
 * should include TraceFilters + other filters (eg. time range filter)
 * (3) the trace filter form: All trace relative filters, in our case, including
 * TraceFilters (thread/process/event filter) + host filter + time range filter
 **/
export declare interface TraceFilters {
  device_regexes: string[];
  resource_regexes: string[];
  trace_event_filters: TraceEventFilter[];
}

/**
 * Represents a category for grouping flow events within the Trace Viewer.
 * Each category has a unique numeric `id` and a human-readable `name`.
 * These categories are used to filter and visualize different types of flows
 * in the trace view.
 */
export declare interface FlowCategory {
  id: number;
  name: string;
}

/**
 * Interface for a filter instruction of event level filtering
 * Now only supports name field regex matching, following definition at
 * http://org_xprof/convert/trace_viewer/trace_filter_config.proto;l=8;rcl=395789773
 */
// TODO(jonahweaver) Add support for values for other than string to allow for stricter user defined typing
// when adding filters
export declare interface TraceEventFilter {
  field_name: string;
  op_id: FilterOperatorId;
  str_value?: string;
  int_value?: string;
  double_value?: string;
  regex_value?: string;
}

/**
 * The configuration of a trace view, including filter config and time range.
 */
export declare interface TraceViewConfig {
  trace_filter_config: TraceFilters;
  start_time_ms?: string;
  end_time_ms?: string;
}

/**
 * A map of trace view configurations saved in local storage.
 */
export declare interface TraceFilterLocalStorageMap {
  [configName: string]: TraceViewConfig;
}

/**
 * Data about the viewport.
 */
export declare interface ViewportData {
  range?: {minMs: number; maxMs: number};
}

/**
 * The detail of a 'ViewportChanged' custom event.
 */
export declare interface ViewportChangedEventDetail {
  range?: {min?: number; max?: number; min_ms?: number; max_ms?: number};
}

/**
 * The full detail of a selected event, including arguments and process name.
 */
export declare interface SelectedEvent {
  name: string;
  ph: string;
  pid: number;
  tid: number;
  ts: number;
  dur: number;
  startUs?: number;
  durationUs?: number;
  startUsFormatted?: string;
  durationUsFormatted?: string;
  uid?: string;
  processName?: string;
  args?: {[key: string]: string};
  stackTraceLinkHtml?: string;
  rooflineModelLinkHtml?: string;
  hloModule?: string;
  hloOpName?: string;
  graphViewerLinkHtml?: string;
}

/**
 * Data about trace events.
 */
export declare interface TraceEventsData {
  traceEvents: SelectedEvent[];
}
