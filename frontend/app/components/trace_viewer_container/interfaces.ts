/**
 * The interface for selected event property.
 */
export declare interface SelectedEventProperty {
  property?: string;
  value?: string | number;
  [key: string]: string | number | undefined;
}

/** Represents an item in the metrics array of events selection data. */
export declare interface MetricsItem {
  [key: string]: string | number;
  name: string;
  count: number;
  wallTimeUs: number;
  selfTimeUs: number;
  avgWallDurationUs: number;
}

/** Represents an item in the counters array of events selection data. */
export declare interface CounterSelectionItem {
  [key: string]: string | number;
  counter: string;
  series: string;
  time: number;
  value: number;
}

/** Represents the parsed JSON data structure for events selection. */
export declare interface EventsSelectedData {
  [key: string]: number | MetricsItem[] | CounterSelectionItem[] | undefined;
  selectionStartUs?: number;
  selectionExtentUs?: number;
  metrics?: MetricsItem[];
  counters?: CounterSelectionItem[];
}

/** Represents an aggregated event property with specific metrics. */
export interface AggregatedEventProperty extends SelectedEventProperty {
  name?: string;
  occurrences?: number;
  wallDuration?: number;
  selfTime?: number;
  avgWallDuration?: number;
}

/**
 * The name of the event selected custom event, dispatched from WASM in Trace
 * Viewer v2.
 */
export const EVENT_SELECTED_EVENT_NAME = 'eventselected';

/**
 * The name of the events selected custom event, dispatched from WASM in Trace
 * Viewer v2.
 */
export const EVENTS_SELECTED_EVENT_NAME = 'events_selected';

/**
 * The detail of an 'EventsSelected' custom event. The properties are quoted to
 * prevent renaming during minification.
 */
export declare interface EventsSelectedEventDetail {
  /** The JSON string containing events selected data. */
  // tslint:disable-next-line:enforce-name-casing
  events_selected_data: string;
}

/**
 * Type guard for the 'EventsSelected' custom event.
 * @param event The event to check.
 * @return True if the event is a CustomEvent with valid EventsSelectedEventDetail.
 */
export function isEventsSelectedEvent(
  event: Event,
): event is CustomEvent<EventsSelectedEventDetail> {
  if (!(event instanceof CustomEvent)) return false;
  const detail = event.detail as unknown;
  return (
    typeof detail === 'object' &&
    detail !== null &&
    'events_selected_data' in detail &&
    typeof (detail as EventsSelectedEventDetail).events_selected_data ===
      'string'
  );
}

/**
 * The detail of an 'EntrySelected' custom event. The properties are quoted to
 * prevent renaming during minification.
 */
export declare interface EntrySelectedEventDetail {
  /** The index of the selected event. */
  eventIndex: number;
  /** The name of the selected event. */
  name: string;
  /** The start time of the selected event in microseconds. */
  startUs: number;
  /** The duration of the selected event in microseconds. */
  durationUs: number;
  /** The formatted start time of the selected event. */
  startUsFormatted: string;
  /** The formatted duration of the selected event. */
  durationUsFormatted: string;
  /** The process ID of the selected event. */
  pid?: number;
  /** The unique ID of the selected event. */
  uid?: string;
  /** The HLO module name of the selected event. */
  hloModuleName?: string;
  /** The HLO op name of the selected event. */
  hloOpName?: string;
}

/**
 * Type guard for the 'EntrySelected' custom event.
 * @param event The event to check.
 * @return True if the event is a CustomEvent with valid EntrySelectedEventDetail.
 */
export function isEntrySelectedEvent(
  event: Event,
): event is CustomEvent<EntrySelectedEventDetail> {
  if (!(event instanceof CustomEvent)) return false;
  const detail = event.detail as unknown;
  return (
    typeof detail === 'object' &&
    detail !== null &&
    'eventIndex' in detail &&
    (detail as {eventIndex: unknown}).eventIndex !== undefined
  );
}

/**
 * The interface for a selected event.
 */
export declare interface SelectedEvent {
  /** The name of the event. */
  name: string;
  /** The formatted start time of the event. */
  startUsFormatted?: string;
  /** The formatted duration of the event. */
  durationUsFormatted?: string;
  /** The HTML link to the stack trace. */
  stackTraceLinkHtml?: string;
  /** The HTML link to the roofline model. */
  rooflineModelLinkHtml?: string;
  /** The HTML link to the graph viewer. */
  graphViewerLinkHtml?: string;
  /** The HLO module name. */
  hloModule?: string;
  /** The HLO op name. */
  hloOpName?: string;
  /** The unique ID of the event. */
  uid?: string;
  /** The process ID of the event. */
  pid?: number;
  /** The arguments associated with the event. */
  args?: {[key: string]: string};
  /** The phase of the event. */
  ph?: string;
  /** The thread ID of the event. */
  tid?: number;
  /** The timestamp of the event. */
  ts?: number;
  /** The duration of the event. */
  dur?: number;
  /** The start time in microseconds. */
  startUs?: number;
  /** The duration in microseconds. */
  durationUs?: number;
  /** The name of the process. */
  processName?: string;
}

/**
 * Mouse modes for trace viewer interaction.
 * Must match the values in C++ MouseMode enum.
 */
export enum MouseMode {
  /** Select mode. */
  SELECT = 1,
  /** Pan mode. */
  PAN = 2,
  /** Zoom mode. */
  ZOOM = 3,
  /** Timing mode. */
  TIMING = 4,
}

/** Event name for mouse mode changes. */
export const MOUSE_MODE_CHANGED_EVENT_NAME = 'mouse_mode_changed';

/** Detail for mouse mode changed event. */
export declare interface MouseModeChangedEventDetail {
  /** The new mouse mode. */
  mouseMode: number;
}

/**
 * Type guard for MouseModeChangedEvent.
 * @param event The event to check.
 * @return True if the event is a CustomEvent with valid MouseModeChangedEventDetail.
 */
export function isMouseModeChangedEvent(
  event: Event,
): event is CustomEvent<MouseModeChangedEventDetail> {
  return !!(
    event instanceof CustomEvent &&
    event.detail &&
    typeof event.detail.mouseMode === 'number'
  );
}

/**
 * Response type for graph_viewer.json?type=adj_nodes.
 */
export declare interface AdjNodesResponse {
  /** Names of operand nodes. */
  operand_names: string[];
  /** Names of consumer nodes. */
  consumer_names: string[];
  /** Maps to string arrays. */
  [key: string]: string[];
}
