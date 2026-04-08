import {
  AggregatedEventProperty,
  EventsSelectedData,
  MetricsItem,
} from './interfaces';

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

function isEventsSelectedData(data: unknown): data is EventsSelectedData {
  if (typeof data !== 'object' || data === null) return false;
  const record = data as Record<string, unknown>;

  const metrics = record['metrics'];
  if (metrics !== undefined && !Array.isArray(metrics)) return false;
  if (Array.isArray(metrics) && !metrics.every(isMetricsItem)) return false;

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
} {
  const properties: AggregatedEventProperty[] = [];
  let selectionStartFormat: string | undefined;
  let selectionExtentFormat: string | undefined;

  try {
    const data = JSON.parse(dataString) as unknown;

    let metricsData: MetricsItem[] = [];
    let selectionStartUs: number | undefined;
    let selectionExtentUs: number | undefined;

    if (isMetricsItemArray(data)) {
      metricsData = data;
    } else if (isEventsSelectedData(data)) {
      metricsData = data.metrics ?? [];
      selectionStartUs = data.selectionStartUs;
      selectionExtentUs = data.selectionExtentUs;
    } else {
      throw new Error('Invalid events selected data format');
    }

    for (const item of metricsData) {
      const name = item.name;
      properties.push({
        'property': name,
        'value': '',
        'name': name,
        'occurrences': item.count,
        'wallDuration': item.wallTimeUs,
        'selfTime': item.selfTimeUs,
        'avgWallDuration': item.avgWallDurationUs,
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

  return {properties, selectionStartFormat, selectionExtentFormat};
}
