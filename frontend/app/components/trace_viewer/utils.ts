import {TraceViewerV2Module} from 'org_xprof/frontend/app/components/trace_viewer_v2/main';
import {FILTER_OPERATORS} from './constants';
import {
  AggregatedEventProperty,
  CounterSelectionItem,
  EventsSelectedData,
  MetricsItem,
} from './interfaces';
import {FilterField, FilterOperator} from './trace_viewer_typings';

function isMetricsItem(item: unknown): item is MetricsItem {
  if (typeof item !== 'object' || item === null) return false;
  const record = item as Record<string, unknown>;
  return (
    typeof record['name'] === 'string' &&
    typeof record['count'] === 'number' &&
    typeof record['wallTimeUs'] === 'number' &&
    typeof record['selfTimeUs'] === 'number' &&
    typeof record['avgWallDurationUs'] === 'number'
  );
}

function isCounterSelectionItem(item: unknown): item is CounterSelectionItem {
  if (typeof item !== 'object' || item === null) return false;
  const record = item as Record<string, unknown>;
  return (
    typeof record['counter'] === 'string' &&
    typeof record['series'] === 'string' &&
    typeof record['time'] === 'number' &&
    typeof record['value'] === 'number'
  );
}

function isEventsSelectedData(data: unknown): data is EventsSelectedData {
  if (typeof data !== 'object' || data === null) return false;
  const record = data as Record<string, unknown>;

  const metrics = record['metrics'];
  if (Array.isArray(metrics) && !metrics.every(isMetricsItem)) return false;

  const counters = record['counters'];
  if (Array.isArray(counters) && !counters.every(isCounterSelectionItem)) {
    return false;
  }

  return (
    (record['selectionStartUs'] === undefined ||
      typeof record['selectionStartUs'] === 'number') &&
    (record['selectionExtentUs'] === undefined ||
      typeof record['selectionExtentUs'] === 'number')
  );
}

function isMetricsItemArray(data: unknown): data is MetricsItem[] {
  return Array.isArray(data) && data.every(isMetricsItem);
}

/**
 * Parses events selected data JSON string and returns aggregated properties and formatted time ranges.
 * @param dataString The JSON string containing events selected data.
 * @return An object containing aggregated properties and formatted time ranges.
 */
export function parseEventsSelectedData(dataString: string): {
  properties: AggregatedEventProperty[];
  selectionStartFormat?: string;
  selectionExtentFormat?: string;
  isCounter?: boolean;
} {
  const properties: AggregatedEventProperty[] = [];
  let selectionStartFormat: string | undefined;
  let selectionExtentFormat: string | undefined;
  let isCounter = false;

  try {
    const data = JSON.parse(dataString) as unknown;

    let metricsData: MetricsItem[] = [];
    let countersData: CounterSelectionItem[] = [];
    let selectionStartUs: number | undefined;
    let selectionExtentUs: number | undefined;

    if (isMetricsItemArray(data)) {
      metricsData = data;
    } else if (isEventsSelectedData(data)) {
      metricsData = (data['metrics'] as MetricsItem[]) ?? [];
      countersData = (data['counters'] as CounterSelectionItem[]) ?? [];
      selectionStartUs = data['selectionStartUs'] as number | undefined;
      selectionExtentUs = data['selectionExtentUs'] as number | undefined;
    } else {
      throw new Error('Invalid events selected data format');
    }

    isCounter = countersData.length > 0;

    for (const item of countersData) {
      properties.push({
        'property': item['counter'],
        'counter': item['counter'],
        'series': item['series'],
        'time': item['time'],
        'value': item['value'],
      });
    }

    for (const item of metricsData) {
      const name = item['name'] as string;
      properties.push({
        'property': name,
        'value': '',
        'name': name,
        'occurrences': item['count'] as number,
        'wallDuration': item['wallTimeUs'] as number,
        'selfTime': item['selfTimeUs'] as number,
        'avgWallDuration': item['avgWallDurationUs'] as number,
      });
    }

    selectionStartFormat =
      selectionStartUs !== undefined
        ? `${(selectionStartUs * 1000).toFixed(0)} ns`
        : undefined;
    selectionExtentFormat =
      selectionExtentUs !== undefined
        ? `${(selectionExtentUs * 1000).toFixed(0)} ns`
        : undefined;
  } catch (e) {
    console.error('Failed to parse events_selected_data:', e);
    throw e;
  }

  return {properties, selectionStartFormat, selectionExtentFormat, isCounter};
}

/**
 * Extracts and parses process mappings from the WASM module.
 */
export function getProcessMappingsFromWasm(
  traceViewerModule: TraceViewerV2Module | null,
): Map<number, string> {
  const result = new Map<number, string>();
  if (!traceViewerModule || !traceViewerModule.application) {
    return result;
  }
  try {
    const dict = traceViewerModule.application
      .instance()
      .dataProvider()
      .getProcessMappings();

    if (dict) {
      const keys = Object.keys(dict);
      for (const pidStr of keys) {
        const host = (dict as Record<string, string>)[pidStr];
        result.set(Number(pidStr), host);
      }
    }
  } catch (e) {
    console.warn('Failed to get process mappings from WASM:', e);
  }
  return result;
}

/**
 * Lookup FilterOperator by operator value.
 * By default set regex operator to match string input.
 */
export function lookupFilterOperator(operatorValue: string): FilterOperator {
  return (
    FILTER_OPERATORS.find((operator) => operator.value === operatorValue) ||
    FILTER_OPERATORS[1]
  );
}

/**
 * Generate unique tracking key for filter field.
 */
export function filterFieldKey(field: FilterField): string {
  if (field.info?.name === undefined) {
    return field.info?.category || '';
  }
  return `${field.info.category}_${field.info.name}`;
}

/**
 * Extracts and parses process names from the WASM module.
 */
export function getProcessNamesFromWasm(
  traceViewerModule: TraceViewerV2Module | null,
): Map<number, string> {
  const result = new Map<number, string>();
  if (!traceViewerModule || !traceViewerModule.application) {
    return result;
  }
  try {
    const dataProvider = traceViewerModule.application
      .instance()
      .dataProvider();
    if (dataProvider && dataProvider.getProcessNames) {
      const dict = dataProvider.getProcessNames();
      if (dict) {
        const keys = Object.keys(dict);
        for (const pidStr of keys) {
          const processName = (dict as Record<string, string>)[pidStr];
          result.set(Number(pidStr), processName);
        }
      }
    }
  } catch (e) {
    console.warn('Failed to get process names from WASM:', e);
  }
  return result;
}
