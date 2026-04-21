import {SelectedEventProperty} from 'org_xprof/frontend/app/components/trace_viewer_container/trace_viewer_container';

/** Represents an item in the metrics array of events selection data. */
export declare interface MetricsItem {
  [key: string]: string | number;
  name: string;
  count: number;
  wallTimeUs: number;
  selfTimeUs: number;
  avgWallDurationUs: number;
}

/** Represents the parsed JSON data structure for events selection. */
export declare interface EventsSelectedData {
  [key: string]: number | MetricsItem[] | undefined;
  selectionStartUs?: number;
  selectionExtentUs?: number;
  metrics?: MetricsItem[];
}

/** Represents an aggregated event property with specific metrics. */
export interface AggregatedEventProperty extends SelectedEventProperty {
  name: string;
  occurrences: number;
  wallDuration?: number;
  selfTime?: number;
  avgWallDuration?: number;
}
