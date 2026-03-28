import {AggregatedEventProperty} from './interfaces';

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
    const data = JSON.parse(dataString) as
      | Record<string, unknown>
      | Array<Record<string, unknown>>;
    const metricsData = Array.isArray(data)
      ? data
      : (data['metrics'] as Array<Record<string, unknown>> ?? []);

    for (const item of metricsData) {
      const name = item['name'] as string;
      properties.push({
        'property': name,
        'value': '',
        'name': name,
        'occurrences': (item['count'] as number) ?? 0,
        'wallDuration':
          item['wallTimeUs'] !== undefined
            ? Number(item['wallTimeUs'])
            : undefined,
        'selfTime':
          item['selfTimeUs'] !== undefined
            ? Number(item['selfTimeUs'])
            : undefined,
        'avgWallDuration':
          item['avgWallDurationUs'] !== undefined
            ? Number(item['avgWallDurationUs'])
            : undefined,
      });
    }

    selectionStartFormat =
      !Array.isArray(data) && data['selectionStartUs'] !== undefined
        ? `${(Number(data['selectionStartUs']) * 1000).toFixed(0)} ns`
        : undefined;
    selectionExtentFormat =
      !Array.isArray(data) && data['selectionExtentUs'] !== undefined
        ? `${(Number(data['selectionExtentUs']) * 1000).toFixed(0)} ns`
        : undefined;
  } catch (e) {
    console.error('Failed to parse events_selected_data:', e);
    throw e;
  }

  return {properties, selectionStartFormat, selectionExtentFormat};
}
