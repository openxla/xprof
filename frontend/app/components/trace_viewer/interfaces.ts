import {
  RawEventItem,
  SelectedEventProperty,
} from 'org_xprof/frontend/app/components/trace_viewer_container/trace_viewer_container';

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

/** Structure of the payload emitted from WASM when multiple events are selected. */
export declare interface EventsSelectedData {
  [key: string]:
    | number
    | MetricsItem[]
    | CounterSelectionItem[]
    | RawEventItem[]
    | undefined;
  selectionStartUs?: number;
  selectionExtentUs?: number;
  metrics?: MetricsItem[];
  counters?: CounterSelectionItem[];
  rawEvents?: RawEventItem[];
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
 * Response type for graph_viewer.json?type=adj_nodes.
 */
export declare interface AdjacentNodesResponse {
  operand_names: string[];
  consumer_names: string[];
  [key: string]: string[];
}
