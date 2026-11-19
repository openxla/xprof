import {
  ChartDataInfo,
  ChartOptions,
} from 'org_xprof/frontend/app/common/interfaces/chart';
import {SimpleDataTable} from 'org_xprof/frontend/app/common/interfaces/data_table';
import {DefaultDataProvider} from 'org_xprof/frontend/app/components/chart/default_data_provider';
import {
  COMMON_CHART_OPTIONS,
  LATENCY_BREAKDOWN_OPTIONS,
} from './grpc_metrics_charts';
import {
  DataTableColumn,
  GrpcMetricsDatabase,
  GrpcStraggler,
  HloCollectiveStats,
  HostNetworkInterfacesGroup,
  InsightLayer,
  InsightSeverity,
  InsightSummary,
  InsightType,
  NetworkInterfaceInfo,
  PeerNodeLatencyInfo,
  SystemConfig,
  TcpStatsSnapshot,
  TimelineInfo,
  TimelineMarker,
  TimeSeriesPoint,
  TpuVmAnalytics,
  TpuVmLatencyInfo,
  TpuVmStepLatencyInfo,
  TraceEvent,
  XorTimestamps,
} from './types';

/**
 * Normalizes an ID (string or number) to a canonical string for map keying.
 * The id is compared as an exact digit string and is never routed through
 * Number(): 64-bit flow ids exceed Number.MAX_SAFE_INTEGER, so a numeric
 * round-trip silently rounds the low digits and collapses distinct flows into a
 * single bucket, mismatching timeline markers to the wrong flow.
 * Callers must supply the id as a string (see quoteLargeIntIds) to preserve full
 * precision; a numeric input can only be stringified as-is (already lossy).
 */
export function normalizeIdForMatching(
  id: string | number | undefined | null,
): string {
  if (id === undefined || id === null) return '';
  const clean = String(id)
    .replace(/^["']+|["']+$/g, '')
    .trim();
  // Canonicalize non-negative integer strings by stripping leading zeros, so a
  // string source and a numeric source for the same id compare equal.
  if (/^\d+$/.test(clean)) {
    return clean.replace(/^0+(?=\d)/, '');
  }
  return clean;
}

/**
 * Wraps the integer values of identity fields (flow_id, chunk_id) in quotes
 * within a raw trace-event JSON string, so that a subsequent JSON.parse keeps
 * them as exact strings. 64-bit ids exceed Number.MAX_SAFE_INTEGER and would
 * otherwise be silently rounded by JSON.parse, colliding distinct flows and
 * mismatching their timeline markers. Values that are already
 * quoted are left untouched, so this transform is idempotent.
 */
export function quoteLargeIntIds(jsonStr: string): string {
  return jsonStr.replace(/"(flow_id|chunk_id)":\s*(-?\d+)/g, '"$1":"$2"');
}

/**
 * Parses an ISO 8601 string timestamp (possibly with nanosecond precision)
 * into absolute nanoseconds since epoch, preserving nanosecond precision.
 */
export function parseFathomTimestampToNs(
  tsStr: string | number | undefined | null,
): number {
  if (tsStr === undefined || tsStr === null) return 0;
  if (typeof tsStr === 'number') return tsStr;
  const cleanStr = String(tsStr)
    .replace(/^"+|"+$/g, '')
    .trim();
  if (!isNaN(Number(cleanStr))) return Number(cleanStr);

  try {
    const baseMs = Date.parse(cleanStr.replace(/\.\d+/, '.000'));
    if (isNaN(baseMs)) return 0;

    const match = cleanStr.match(/\.(\d+)/);
    const nanoStr = match ? match[1].slice(0, 9).padEnd(9, '0') : '0';
    const nano = Number(nanoStr);

    return baseMs * 1e6 + nano;
  } catch (e) {
    console.error('Failed to parse fathom timestamp:', tsStr, e);
    return 0;
  }
}

/** Magic numbers for insight detection thresholds. */

const P50_PERCENTILE_RANK = 0.5;
const P90_PERCENTILE_RANK = 0.9;
const P99_PERCENTILE_RANK = 0.99;
const TCP_LATENCY_FRACTION_THRESHOLD = 0.4;

interface InsightConfig {
  type: InsightType;
  description: string;
  severity: InsightSeverity;
  layer: InsightLayer;
  condition: (
    straggler: GrpcStraggler,
    p90Latency: number,
    tcpEventsMap?: Map<string, TcpEventsInfo>,
    p90ClientLatency?: number,
  ) => boolean;
}

/** Parses a metric value from proto args into a number or undefined. */
export function parseMetricVal(val: unknown): number | undefined {
  if (val === undefined || val === null) return undefined;
  if (typeof val === 'number') return isNaN(val) ? undefined : val;
  if (typeof val === 'string') {
    const clean = val.replace(/^["']+|["']+$/g, '').trim();
    if (clean === '' || clean === 'true') {
      return clean === 'true' ? 1 : undefined;
    }
    if (clean === 'false') return 0;
    const n = Number(clean);
    return isNaN(n) ? undefined : n;
  }
  return undefined;
}

/**
 * Returns the change of `metricNames` from the first SENDMSG to the last ACK,
 * as reported on those two events, or undefined when either event lacks a
 * metric (Fathom omits unchanged metrics) or a counter went down.
 */
export function getMetricDeltas(
  tcpEvents: TcpEventsInfo | undefined,
  metricNames: ReadonlyArray<keyof TcpStatsSnapshot & string>,
): Record<string, number> | undefined {
  const start = tcpEvents?.sendmsg?.args;
  const end = tcpEvents?.ack?.args;
  if (!start || !end) return undefined;
  const deltas: Record<string, number> = {};
  for (const name of metricNames) {
    const startVal = parseMetricVal(start[name]);
    const endVal = parseMetricVal(end[name]);
    if (startVal === undefined || endVal === undefined || endVal < startVal) {
      return undefined;
    }
    deltas[name] = endVal - startVal;
  }
  return deltas;
}

/**
 * Returns the share of the chunk latency spent in TCP (TCP Latency / Latency),
 * or undefined when the TCP latency is unknown or incomplete.
 */
export function getTcpLatencyFraction(s: GrpcStraggler): number | undefined {
  const tcpMs = s.ack_to_sendmsg_ms;
  if (s.tcp_latency_incomplete || tcpMs == null || isNaN(tcpMs)) {
    return undefined;
  }
  return s.latency_ms > 0 ? tcpMs / s.latency_ms : undefined;
}

function isTcpLatencyDominant(s: GrpcStraggler): boolean {
  const fraction = getTcpLatencyFraction(s);
  return fraction !== undefined && fraction > TCP_LATENCY_FRACTION_THRESHOLD;
}

/** Appended to the descriptions of the TCP Network Recommendations. */
const TCP_INSIGHT_SCOPE =
  ' Only chunks whose TCP Latency (ACK-SENDMSG) is complete and > 40% of ' +
  'Latency are evaluated.';

const INSIGHT_DEFINITIONS: InsightConfig[] = [
  {
    type: InsightType.HIGH_TCP_QUEUING_DELAY,
    description:
      'High TCP Queuing Delays observed. Evaluated when data_not_sent / delivery_rate > 20ms during SENDMSG.' +
      TCP_INSIGHT_SCOPE,
    severity: InsightSeverity.WARNING,
    layer: InsightLayer.TCP,
    condition: (s, p90, tcpEventsMap) => {
      if (!tcpEventsMap) return false;
      const tcpEvents = tcpEventsMap.get(normalizeIdForMatching(s.chunk_id));
      if (!tcpEvents) return false;

      const eventsToSearch = [tcpEvents.sendmsg];
      let dataNotSent: number | undefined;
      let deliveryRate: number | undefined;

      for (const ev of eventsToSearch) {
        if (ev?.args) {
          const dns = parseMetricVal(ev.args['data_notsent']);
          const dr = parseMetricVal(ev.args['delivery_rate']);
          if (dns !== undefined && dr !== undefined && dr > 0) {
            dataNotSent = dns;
            deliveryRate = dr;
            break;
          }
        }
      }

      if (
        dataNotSent === undefined ||
        deliveryRate === undefined ||
        deliveryRate === 0
      ) {
        return false;
      }
      const queuingDelayMs = (dataNotSent / deliveryRate) * 1000;
      return queuingDelayMs > 20 && isTcpLatencyDominant(s);
    },
  },
  {
    type: InsightType.NETWORK_CONGESTION_LOSS,
    description:
      'High retransmissions occur alongside high latency, indicating potential reachability problems. Evaluated when packet retransmission rate > 1% and Latency > P90 Latency.' +
      TCP_INSIGHT_SCOPE,
    severity: InsightSeverity.WARNING,
    layer: InsightLayer.TCP,
    condition: (s, p90, tcpEventsMap) => {
      if (!tcpEventsMap) return false;
      const tcpEvents = tcpEventsMap.get(normalizeIdForMatching(s.chunk_id));
      const deltas = getMetricDeltas(tcpEvents, [
        'packet_retx',
        'packet_delivered',
      ]);
      if (!deltas || !isTcpLatencyDominant(s)) return false;
      const deltaRetx = deltas['packet_retx'];
      const deltaDelivered = deltas['packet_delivered'];

      if (deltaDelivered === 0) {
        return deltaRetx > 0 && (s.latency_ms || 0) > p90;
      }
      return deltaRetx / deltaDelivered > 0.01 && (s.latency_ms || 0) > p90;
    },
  },
  {
    type: InsightType.NETWORK_CONGESTION_ECN,
    description:
      'High proportion of packets were delivered with Congestion Experienced (CE) marks. Evaluated when CE packets > 50% of delivered packets.' +
      TCP_INSIGHT_SCOPE,
    severity: InsightSeverity.WARNING,
    layer: InsightLayer.TCP,
    condition: (s, p90, tcpEventsMap) => {
      if (!tcpEventsMap) return false;
      const tcpEvents = tcpEventsMap.get(normalizeIdForMatching(s.chunk_id));
      const deltas = getMetricDeltas(tcpEvents, [
        'packet_delivered_ce',
        'packet_delivered',
      ]);
      if (!deltas || !isTcpLatencyDominant(s)) return false;
      const deltaCe = deltas['packet_delivered_ce'];
      const deltaDelivered = deltas['packet_delivered'];

      if (deltaDelivered === 0) return false;
      return deltaCe / deltaDelivered > 0.5;
    },
  },
  {
    type: InsightType.HIGH_APPLICATION_LATENCY,
    description:
      'MXLA chunk transfer latency and gRPC client latency are both high, but TCP-level transfer time does not account for it, pointing to delays above the TCP layer (e.g. gRPC/MXLA host-side queuing or scheduling). Evaluated when Latency > P90 Latency, gRPC Client Latency >= P90 gRPC Client Latency, and TCP Latency (ACK-SENDMSG) <= 40% of Latency. Chunks without a complete TCP Latency are not evaluated.',
    severity: InsightSeverity.WARNING,
    layer: InsightLayer.RPC,
    condition: (s, p90, tcpEventsMap, p90ClientLatency) => {
      const clientLatencyMs = s.grpc_client_latency_ms || 0;
      if (clientLatencyMs <= 0 || (s.latency_ms || 0) <= p90) return false;
      if (clientLatencyMs < (p90ClientLatency ?? 0)) return false;

      // Exclude chunks whose latency is explained by the TCP phase; those are
      // covered by the TCP Network Recommendations. Without a complete TCP
      // latency (no TCP events, or incomplete ones) the TCP phase cannot be
      // ruled out, so such chunks are not blamed on the application.
      const tcpEvents = tcpEventsMap?.get(normalizeIdForMatching(s.chunk_id));
      if (hasIncompleteTcpEvents(s, tcpEvents)) return false;
      const fraction = getTcpLatencyFraction(s);
      return (
        fraction !== undefined && fraction <= TCP_LATENCY_FRACTION_THRESHOLD
      );
    },
  },
  {
    type: InsightType.APPLICATION_BOUND,
    description:
      "The network has bandwidth, but the application isn't writing data fast enough. Evaluated when is_delivery_rate_app_limited = 1 at the ACK. Chunks without a complete TCP Latency (ACK-SENDMSG) are not evaluated.",
    severity: InsightSeverity.INFO,
    layer: InsightLayer.RPC,
    condition: (s, p90, tcpEventsMap) => {
      if (!tcpEventsMap || getTcpLatencyFraction(s) === undefined) return false;
      const tcpEvents = tcpEventsMap.get(normalizeIdForMatching(s.chunk_id));
      if (!tcpEvents || !tcpEvents.ack || !tcpEvents.ack.args) {
        return false;
      }
      const isAppLimited = parseMetricVal(
        tcpEvents.ack.args['is_delivery_rate_app_limited'],
      );
      return isAppLimited === 1;
    },
  },
  {
    type: InsightType.RECEIVER_WINDOW_LIMITED,
    description:
      "The receiver's TCP buffer is full, forcing the sender to pause. Evaluated when Receiver Window Limited time accounts for > 50% of total busy time." +
      TCP_INSIGHT_SCOPE,
    severity: InsightSeverity.WARNING,
    layer: InsightLayer.TCP,
    condition: (s, p90, tcpEventsMap) => {
      if (!tcpEventsMap) return false;
      const tcpEvents = tcpEventsMap.get(normalizeIdForMatching(s.chunk_id));
      const deltas = getMetricDeltas(tcpEvents, [
        'busy_usec',
        'rwnd_limited_usec',
      ]);
      if (!deltas || !isTcpLatencyDominant(s)) return false;
      const deltaBusy = deltas['busy_usec'];
      const deltaRwnd = deltas['rwnd_limited_usec'];

      if (deltaBusy === 0) return false;
      return deltaRwnd / deltaBusy > 0.5;
    },
  },
  {
    type: InsightType.SENDER_BUFFER_LIMITED,
    description:
      "The sender's TCP socket buffer is full, stalling application writes. Evaluated when Send Buffer Limited time accounts for > 50% of total busy time." +
      TCP_INSIGHT_SCOPE,
    severity: InsightSeverity.WARNING,
    layer: InsightLayer.TCP,
    condition: (s, p90, tcpEventsMap) => {
      if (!tcpEventsMap) return false;
      const tcpEvents = tcpEventsMap.get(normalizeIdForMatching(s.chunk_id));
      const deltas = getMetricDeltas(tcpEvents, [
        'busy_usec',
        'sndbuf_limited_usec',
      ]);
      if (!deltas || !isTcpLatencyDominant(s)) return false;
      const deltaBusy = deltas['busy_usec'];
      const deltaSndBuf = deltas['sndbuf_limited_usec'];

      if (deltaBusy === 0) return false;
      return deltaSndBuf / deltaBusy > 0.5;
    },
  },
];

/**
 * Pure data transformation logic for gRPC metrics.
 * Moves heavy lifting out of the Angular component to keep it lean.
 */

interface AggregatedPoint {
  sum: number;
  count: number;
  min: number;
}

/**
 * How a chart aggregates a series per time bucket.
 *
 * - 'gauge': a level (e.g. CWND), averaged.
 * - 'min': a level where the smallest value matters (e.g. min RTT).
 * - 'counter': a cumulative per-connection counter (e.g. bytes sent). Its
 *   increments are computed per connection and summed in the bucket of the
 *   sample that reported them; "/s" charts divide by the bucket width.
 */
export type ChartMetricKind = 'gauge' | 'min' | 'counter';

/** Number of time buckets across the time span of a chart. */
const CHART_TIME_BUCKETS = 200;

/**
 * Builds a multi-line chart data table from time series points, grouped by
 * host or connection, in fixed time buckets (see ChartMetricKind).
 */
export function buildMultiLineChartData(
  series: readonly TimeSeriesPoint[] | undefined,
  yLabel: string,
  selectedHosts: readonly string[],
  forceGroupMode?: 'tpu_vm' | 'endpoint_id',
  preFilteredMap?: Map<
    readonly TimeSeriesPoint[],
    Map<string, TimeSeriesPoint[]>
  >,
  endpointToPeerMap?: Map<string, string>,
  kind: ChartMetricKind = 'gauge',
): SimpleDataTable | null {
  if (!series || series.length === 0) return null;

  const activeGroupMode = forceGroupMode ?? 'tpu_vm';
  let filteredPoints: TimeSeriesPoint[] = [];

  if (preFilteredMap && preFilteredMap.has(series)) {
    const hostMap = preFilteredMap.get(series);
    if (hostMap) {
      for (const host of selectedHosts) {
        const pts = hostMap.get(host);
        if (pts) {
          filteredPoints = filteredPoints.concat(pts);
        }
      }
    }
  } else {
    const selectedHostsSet = new Set(selectedHosts);
    filteredPoints =
      selectedHosts.length === 0
        ? []
        : series.filter((pt) => selectedHostsSet.has(pt.tpu_vm));
  }

  if (filteredPoints.length === 0) return null;

  // The JSON omits zero times and values.
  const timeOf = (pt: TimeSeriesPoint) => pt.timestamp_rel_sec ?? 0;
  const valueOf = (pt: TimeSeriesPoint) => pt.value ?? 0;

  const connections = new Map<string, TimeSeriesPoint[]>();
  const hosts = new Set<string>();
  let minTime = Infinity;
  let maxTime = -Infinity;
  for (const pt of filteredPoints) {
    const connectionKey = `${pt.tpu_vm}|${pt.endpoint_id}`;
    let points = connections.get(connectionKey);
    if (!points) {
      points = [];
      connections.set(connectionKey, points);
    }
    points.push(pt);
    hosts.add(pt.tpu_vm);
    minTime = Math.min(minTime, timeOf(pt));
    maxTime = Math.max(maxTime, timeOf(pt));
  }
  const bucketSec =
    maxTime > minTime ? (maxTime - minTime) / CHART_TIME_BUCKETS : 1;
  const bucketOf = (timeSec: number) =>
    Math.min(
      CHART_TIME_BUCKETS - 1,
      Math.floor((timeSec - minTime) / bucketSec),
    );

  // With several hosts, connection numbers are only unique per host.
  const seriesKeyOf = (pt: TimeSeriesPoint): string => {
    if (activeGroupMode === 'tpu_vm') return pt.tpu_vm;
    const connection = `Connection ${parseEndpointId(pt.endpoint_id).connectionNum}`;
    return hosts.size > 1 ? `${pt.tpu_vm} ${connection}` : connection;
  };

  const cells = new Map<number, Map<string, AggregatedPoint>>();
  const addToBucket = (bucket: number, key: string, value: number) => {
    let bucketCells = cells.get(bucket);
    if (!bucketCells) {
      bucketCells = new Map<string, AggregatedPoint>();
      cells.set(bucket, bucketCells);
    }
    const cell: AggregatedPoint = bucketCells.get(key) || {
      sum: 0,
      count: 0,
      min: value,
    };
    cell.sum += value;
    cell.count++;
    cell.min = Math.min(cell.min, value);
    bucketCells.set(key, cell);
  };

  for (const points of connections.values()) {
    points.sort((a, b) => timeOf(a) - timeOf(b));
    const key = seriesKeyOf(points[0]);
    if (kind === 'counter') {
      // A sample below the running max (out of order) adds nothing.
      let runningMax = valueOf(points[0]);
      addToBucket(bucketOf(timeOf(points[0])), key, 0);
      for (let i = 1; i < points.length; i++) {
        const pt = points[i];
        const increment = Math.max(0, valueOf(pt) - runningMax);
        addToBucket(bucketOf(timeOf(pt)), key, increment);
        runningMax = Math.max(runningMax, valueOf(pt));
      }
    } else {
      for (const pt of points) {
        addToBucket(bucketOf(timeOf(pt)), key, valueOf(pt));
      }
    }
  }

  const seriesNames = Array.from(
    new Set(
      Array.from(connections.values(), (points) => seriesKeyOf(points[0])),
    ),
  ).sort();
  const cols: DataTableColumn[] = [
    {id: 'time', label: 'Time (s)', type: 'number'},
  ];
  for (const name of seriesNames) {
    cols.push({id: name, label: name, type: 'number'});
  }

  const isRate = yLabel.includes('/s');
  const buckets = Array.from(cells.keys()).sort((a, b) => a - b);
  const values: Array<Array<number | null>> = buckets.map((bucket) => {
    const bucketCells = cells.get(bucket) ?? new Map<string, AggregatedPoint>();
    return seriesNames.map((name) => {
      const cell = bucketCells.get(name);
      if (kind === 'counter') {
        const sum = cell ? cell.sum : 0;
        return isRate ? sum / bucketSec : sum;
      }
      if (!cell) return null;
      return kind === 'min' ? cell.min : cell.sum / cell.count;
    });
  });

  // Binary unit scaling for bit charts, based on the plotted values.
  let divisor = 1;
  let unitSuffix = '';
  if (yLabel.includes('(Bits)') || yLabel.includes('(Bits/s)')) {
    let maxVal = 0;
    for (const row of values) {
      for (const val of row) {
        if (val !== null && val > maxVal) maxVal = val;
      }
    }
    const KB = 1024;
    const MB = KB * 1024;
    const GB = MB * 1024;
    const TB = GB * 1024;
    const PB = TB * 1024;
    if (maxVal >= PB) {
      divisor = PB;
      unitSuffix = 'Pib';
    } else if (maxVal >= TB) {
      divisor = TB;
      unitSuffix = 'Tib';
    } else if (maxVal >= GB) {
      divisor = GB;
      unitSuffix = 'Gib';
    } else if (maxVal >= MB) {
      divisor = MB;
      unitSuffix = 'Mib';
    } else if (maxVal >= KB) {
      divisor = KB;
      unitSuffix = 'Kib';
    } else {
      unitSuffix = 'bits';
    }
  }

  const rows: Array<{c: Array<{v: string | number | boolean | null}>}> =
    buckets.map((bucket, i) => ({
      c: [
        {v: minTime + (bucket + 0.5) * bucketSec},
        ...values[i].map((val) => ({v: val === null ? null : val / divisor})),
      ],
    }));

  return {cols, rows, p: {unitSuffix, divisor}};
}

/** Prepares chart data for a latency breakdown pie chart. */
export function prepareLatencyBreakdown(element: GrpcStraggler): ChartDataInfo {
  const cols: DataTableColumn[] = [
    {id: 'phase', label: 'Phase', type: 'string'},
    {id: 'duration', label: 'Duration (ms)', type: 'number'},
  ];
  const rows: Array<{c: Array<{v: string | number | boolean | null}>}> = [];
  let accountedMs = 0;
  if (element.grpc_server_latency_ms > 0) {
    rows.push({c: [{v: 'gRPC Server'}, {v: element.grpc_server_latency_ms}]});
    accountedMs += element.grpc_server_latency_ms;
  }

  const overhead = element.latency_ms - accountedMs;
  if (overhead > 0.1) {
    rows.push({c: [{v: 'Other/Overhead'}, {v: overhead}]});
  }
  return {
    data: {cols, rows},
    dataProvider: new DefaultDataProvider(),
    options: LATENCY_BREAKDOWN_OPTIONS,
  };
}

/** Prepares positioning info for the latency timeline visualization. */
export function prepareLatencyTimelineInfo(
  element: GrpcStraggler,
  xorTimestamps?: XorTimestamps,
  tcpEvents?: TcpEventsInfo,
): TimelineInfo {
  if (!element || element.latency_ms <= 0 || isNaN(element.latency_ms)) {
    return {
      clientLeft: 0,
      clientWidth: 0,
      serverLeft: 0,
      serverWidth: 0,
      tcpLeft: 0,
      tcpWidth: 0,
      clientMarkers: [],
      serverMarkers: [],
      tcpMarkers: [],
      markers: [],
      hasError: true,
    };
  }

  const totalMs = element.latency_ms;
  const toPercent = (valMs: number) => (valMs / totalMs) * 100;
  let hasError = false;

  // Row 2: Client Latency (rely on ground-truth element.grpc_client_latency_ms from backend)
  const clientLatencyMs =
    element.grpc_client_latency_ms && element.grpc_client_latency_ms > 0
      ? element.grpc_client_latency_ms
      : xorTimestamps?.client_latency_ns
        ? xorTimestamps.client_latency_ns / 1e6
        : 0;

  const clientWidthClamped = Math.min(
    100,
    Math.max(0, toPercent(clientLatencyMs)),
  );
  const clientLeftClamped = Math.min(
    100,
    Math.max(0, 100 - clientWidthClamped),
  );

  const clientMarkers: TimelineMarker[] = [];
  let clientTrailingMetadataPct = clientWidthClamped;
  let payloadSentOffsetNs: number | undefined;
  let serverTrailingMetadataOffsetNs: number | undefined;

  if (xorTimestamps) {
    const maxValidNs = Math.max(totalMs * 10, 60000) * 1e6;
    const addClientMarker = (rawNs: number | undefined, label: string) => {
      if (rawNs !== undefined && !isNaN(rawNs) && rawNs >= 0) {
        let relativeNs = rawNs;
        if (
          rawNs > 1e15 &&
          xorTimestamps.client_start_time_ns &&
          xorTimestamps.client_start_time_ns > 1e15
        ) {
          relativeNs = rawNs - xorTimestamps.client_start_time_ns;
        }
        if (relativeNs < -60000 * 1e6 || relativeNs > maxValidNs) {
          hasError = true;
          return relativeNs;
        }
        const left = clientLeftClamped + toPercent(relativeNs / 1e6);
        clientMarkers.push({
          positionLeft: Math.min(100, Math.max(0, left)),
          label,
        });
        return relativeNs;
      }
      return undefined;
    };

    addClientMarker(
      xorTimestamps.client_on_client_initial_metadata_time_ns,
      'Initial Metadata Sent',
    );
    payloadSentOffsetNs = addClientMarker(
      xorTimestamps.client_on_client_to_server_first_message_time_ns,
      'Payload Sent',
    );
    addClientMarker(
      xorTimestamps.client_on_server_initial_metadata_time_ns,
      'Server Initial Metadata',
    );
    addClientMarker(
      xorTimestamps.client_on_server_to_client_first_message_time_ns,
      'Server to Client Message',
    );
    serverTrailingMetadataOffsetNs = addClientMarker(
      xorTimestamps.client_on_server_trailing_metadata_time_ns,
      'Server Trailing Metadata',
    );

    if (
      xorTimestamps.client_on_server_trailing_metadata_time_ns !== undefined &&
      xorTimestamps.client_on_server_trailing_metadata_time_ns >= 0
    ) {
      let relativeNs = xorTimestamps.client_on_server_trailing_metadata_time_ns;
      if (
        relativeNs > 1e15 &&
        xorTimestamps.client_start_time_ns &&
        xorTimestamps.client_start_time_ns > 1e15
      ) {
        relativeNs -= xorTimestamps.client_start_time_ns;
      }
      clientTrailingMetadataPct = toPercent(relativeNs / 1e6);
    }
  }

  // Row 3: Server Latency (rely on ground-truth element.grpc_server_latency_ms from backend)
  const serverLatencyMs =
    element.grpc_server_latency_ms && element.grpc_server_latency_ms > 0
      ? element.grpc_server_latency_ms
      : xorTimestamps?.server_latency_ns
        ? xorTimestamps.server_latency_ns / 1e6
        : 0;

  const serverWidthClamped = Math.min(
    100,
    Math.max(0, toPercent(serverLatencyMs)),
  );
  const serverEndLeft = clientLeftClamped + clientTrailingMetadataPct;
  const serverLeftClamped = Math.min(
    100,
    Math.max(0, serverEndLeft - serverWidthClamped),
  );

  const serverMarkers: TimelineMarker[] = [];
  if (xorTimestamps) {
    const maxValidNs = Math.max(totalMs * 10, 60000) * 1e6;
    const addServerMarker = (rawNs: number | undefined, label: string) => {
      if (rawNs !== undefined && !isNaN(rawNs) && rawNs >= 0) {
        let relativeNs = rawNs;
        const baseNs =
          xorTimestamps.server_start_time_ns ||
          xorTimestamps.client_start_time_ns;
        if (rawNs > 1e15 && baseNs && baseNs > 1e15) {
          relativeNs = rawNs - baseNs;
        }
        if (relativeNs < -60000 * 1e6 || relativeNs > maxValidNs) {
          hasError = true;
          return;
        }
        const left = serverLeftClamped + toPercent(relativeNs / 1e6);
        serverMarkers.push({
          positionLeft: Math.min(100, Math.max(0, left)),
          label,
        });
      }
    };

    addServerMarker(
      xorTimestamps.server_on_client_initial_metadata_time_ns,
      'Initial Metadata Received',
    );
    addServerMarker(
      xorTimestamps.server_on_server_initial_metadata_time_ns,
      'Server Initial Metadata',
    );
    addServerMarker(
      xorTimestamps.server_on_client_to_server_first_message_time_ns,
      'Payload Received',
    );
    addServerMarker(
      xorTimestamps.server_on_server_to_client_first_message_time_ns,
      'Server to Client Message',
    );
  }

  // Row 4: TCP Latency
  const tcpMarkers: TimelineMarker[] = [];
  let sendmsgDeltaNs: number | undefined;
  let ackDeltaNs: number | undefined;

  if (
    tcpEvents &&
    xorTimestamps &&
    xorTimestamps.client_start_time_ns !== undefined
  ) {
    const clientStartNs = xorTimestamps.client_start_time_ns;
    const maxValidNs = Math.max(totalMs * 10, 60000) * 1e6;

    const addTcpMarker = (event: TraceEvent | undefined, label: string) => {
      if (event && event.args && event.args.fathom_timestamp) {
        const fathomNs = parseFathomTimestampToNs(event.args.fathom_timestamp);
        if (!isNaN(fathomNs)) {
          const tcpAbslNs = fathomNs;
          if (clientStartNs < 1e15 && tcpAbslNs > 1e15) {
            // Cannot align absolute TCP events with relative client start
            return undefined;
          }
          const deltaNs = tcpAbslNs - clientStartNs;
          // Safeguard against wildly incorrect deltas
          if (deltaNs < -60000 * 1e6 || deltaNs > maxValidNs) {
            hasError = true;
            return undefined;
          }
          const left = clientLeftClamped + toPercent(deltaNs / 1e6);
          tcpMarkers.push({
            positionLeft: Math.min(100, Math.max(0, left)),
            label,
          });
          return deltaNs;
        }
      }
      return undefined;
    };

    sendmsgDeltaNs = addTcpMarker(tcpEvents.sendmsg, 'SENDMSG');
    addTcpMarker(tcpEvents.scheduled, 'SCHEDULED');
    addTcpMarker(tcpEvents.sent, 'SENT');
    ackDeltaNs = addTcpMarker(tcpEvents.ack, 'ACK');
  }

  // Hard Invariant Checks for TPU MXLA workload over gRPC Chaotic Good v2:
  // Invariant 1: Client Marker Chronological Order in gRPC Chaotic Good v2:
  // Initial Metadata Sent <= Payload Sent <= Server Trailing Metadata
  if (
    payloadSentOffsetNs !== undefined &&
    serverTrailingMetadataOffsetNs !== undefined &&
    payloadSentOffsetNs > serverTrailingMetadataOffsetNs + 0.5 * 1e6
  ) {
    // Physical impossibility: Payload Sent cannot happen after Server Trailing Metadata
    hasError = true;
  }

  // Invariant 2: Chaotic Good Data Channel TCP Invariant:
  // All TCP events (SENDMSG, SCHEDULED, SENT, ACK) transmit the payload data buffer
  // on the data channel, and therefore must occur at or after Payload Sent:
  //   Payload Sent <= TCP SENDMSG
  if (
    payloadSentOffsetNs !== undefined &&
    sendmsgDeltaNs !== undefined &&
    sendmsgDeltaNs < payloadSentOffsetNs - 2.0 * 1e6 // 2ms tolerance for scheduling/timer drift
  ) {
    hasError = true;
  }

  // Invariant 3: TCP ACK must occur before or around Server Trailing Metadata
  if (
    ackDeltaNs !== undefined &&
    serverTrailingMetadataOffsetNs !== undefined &&
    ackDeltaNs > serverTrailingMetadataOffsetNs + 10.0 * 1e6 // 10ms tolerance
  ) {
    hasError = true;
  }

  // Invariant 4: SENDMSG happens significantly before client latency start (> 50ms before)
  if (sendmsgDeltaNs !== undefined && sendmsgDeltaNs < -50 * 1e6) {
    hasError = true;
  }

  // Invariant 5: TPU MXLA Latency Hierarchy:
  //   Total MXLA Latency >= gRPC Client Latency >= gRPC Server Latency
  const tolerance = (val: number) => val * 1.05 + 0.5; // 5% + 0.5ms tolerance for rounding/clock drift
  if (
    clientLatencyMs > tolerance(totalMs) ||
    serverLatencyMs > tolerance(totalMs) ||
    (clientLatencyMs > 0 && serverLatencyMs > tolerance(clientLatencyMs))
  ) {
    hasError = true;
  }

  // Invariant 6: Substantial DB client latency (> 1ms) but resolved client latency collapsed to near zero (< 0.001ms)
  if ((element.grpc_client_latency_ms || 0) > 1 && clientLatencyMs < 0.001) {
    hasError = true;
  }

  const result: TimelineInfo = {
    clientLeft: clientLeftClamped,
    clientWidth: clientWidthClamped,
    serverLeft: serverLeftClamped,
    serverWidth: serverWidthClamped,
    tcpLeft: 0,
    tcpWidth: 0,
    clientMarkers,
    serverMarkers,
    tcpMarkers,
    markers: [...clientMarkers, ...serverMarkers, ...tcpMarkers],
    hasError,
  };

  return result;
}

/** Returns the lower-case text that the general search matches for a row. */
function buildStragglerSearchString(straggler: GrpcStraggler): string {
  return `${straggler.tpu_vm} ${straggler.chunk_id} ${straggler.hlo_collective_name || ''} ${(straggler.endpoint_ids || []).join(' ')} ${straggler.peer_address || ''}`.toLowerCase();
}

/**
 * Sets the general search text of each straggler. computeInsights does this
 * too; use this when insights are not computed.
 */
export function setStragglerSearchStrings(stragglers: GrpcStraggler[]) {
  for (const straggler of stragglers) {
    straggler.searchStr = buildStragglerSearchString(straggler);
  }
}

/** Computes insights for all stragglers and groups them by layer. */
export function computeInsights(
  stragglers: GrpcStraggler[],
  p90Latency: number,
  tcpEventsMap?: Map<string, TcpEventsInfo>,
): {
  mlInsights: InsightSummary[];
  rpcInsights: InsightSummary[];
  tcpInsights: InsightSummary[];
  insightSummaries: InsightSummary[];
} {
  const insightMap = new Map<InsightType, InsightSummary>();

  // Pre-populate insights from static definitions.
  for (const def of INSIGHT_DEFINITIONS) {
    insightMap.set(def.type, {
      type: def.type,
      description: def.description,
      severity: def.severity,
      layer: def.layer,
      flows: [],
    });
  }

  // P90 gRPC client latency across stragglers that reported client latency.
  const clientLatencies = stragglers
    .map((s) => s.grpc_client_latency_ms || 0)
    .filter((latencyMs) => latencyMs > 0)
    .sort((a, b) => a - b);
  const p90ClientLatency =
    clientLatencies.length > 0
      ? clientLatencies[
          Math.floor(clientLatencies.length * P90_PERCENTILE_RANK)
        ]
      : 0;

  for (const straggler of stragglers) {
    straggler.insights = [];
    for (const def of INSIGHT_DEFINITIONS) {
      if (
        def.condition(straggler, p90Latency, tcpEventsMap, p90ClientLatency)
      ) {
        straggler.insights.push({
          text: def.type,
          severity: def.severity,
          description: def.description,
        });
        const summary = insightMap.get(def.type);
        if (summary) {
          summary.flows.push(straggler);
        }
      }
    }

    straggler.searchStr = buildStragglerSearchString(straggler);
  }

  const mlInsights: InsightSummary[] = [];
  const rpcInsights: InsightSummary[] = [];
  const tcpInsights: InsightSummary[] = [];
  const allInsights = Array.from(insightMap.values());

  for (const insight of allInsights) {
    switch (insight.layer) {
      case InsightLayer.ML:
        mlInsights.push(insight);
        break;
      case InsightLayer.RPC:
        rpcInsights.push(insight);
        break;
      case InsightLayer.TCP:
        tcpInsights.push(insight);
        break;
      default:
        break;
    }
  }

  return {
    mlInsights,
    rpcInsights,
    tcpInsights,
    insightSummaries: allInsights,
  };
}

/** Computes background data stats for hosts and collectives. */
/** Calculates performance analytics for a specific host. */
function calculateTpuVmAnalytics(
  hostName: string,
  stragglers: GrpcStraggler[],
): TpuVmAnalytics {
  if (stragglers.length === 0) {
    return {
      tpu_vm: hostName,
      total_transfers: 0,
      p50_latency_ms: 0,
      p99_latency_ms: 0,
      p100_latency_ms: 0,
      high_latency_count: 0,
    };
  }

  // Use the 'nearest rank' method to determine the percentiles.
  // The array is sorted by latency first.
  stragglers.sort((a, b) => (a.latency_ms || 0) - (b.latency_ms || 0));

  const p50Idx = Math.min(
    Math.floor(stragglers.length * P50_PERCENTILE_RANK),
    stragglers.length - 1,
  );
  const p50Latency = stragglers[p50Idx].latency_ms || 0;

  const p99Idx = Math.min(
    Math.floor(stragglers.length * P99_PERCENTILE_RANK),
    stragglers.length - 1,
  );
  const p99Latency = stragglers[p99Idx].latency_ms || 0;

  const p100Latency = stragglers[stragglers.length - 1].latency_ms || 0;

  const highLatencyFlows = stragglers.filter(
    (s) => (s.latency_ms || 0) > p99Latency,
  );

  return {
    tpu_vm: hostName,
    total_transfers: stragglers.length,
    p50_latency_ms: p50Latency,
    p99_latency_ms: p99Latency,
    p100_latency_ms: p100Latency,
    high_latency_count: highLatencyFlows.length,
  };
}

/** Computes background data stats for hosts and collectives. */
export function computeBackgroundData(stragglers: GrpcStraggler[]) {
  const tpuVmMap = new Map<string, GrpcStraggler[]>();
  const collectiveMap = new Map<string, HloCollectiveStats>();

  for (const straggler of stragglers) {
    if (straggler.tpu_vm) {
      let tpuList = tpuVmMap.get(straggler.tpu_vm);
      if (!tpuList) {
        tpuList = [];
        tpuVmMap.set(straggler.tpu_vm, tpuList);
      }
      tpuList.push(straggler);
    }

    const collectiveName =
      straggler.hlo_collective_name || 'Unknown HLO Collective';
    let stats = collectiveMap.get(collectiveName);
    if (!stats) {
      stats = {
        hlo_collective_name: collectiveName,
        total_flows: 0,
        total_latency_ms: 0,
        total_tpu_stall_ms: 0,
        total_slack_ms: 0,
      };
      collectiveMap.set(collectiveName, stats);
    }

    stats.total_flows++;
    stats.total_latency_ms += straggler.latency_ms || 0;
    stats.total_tpu_stall_ms += straggler.tpu_stall_time_ms || 0;
    stats.total_slack_ms += straggler.slack_time_ms || 0;
  }

  const availableTpuVms = Array.from(tpuVmMap.keys()).sort();

  const tpuVmAnalyticsData = Array.from(
    tpuVmMap.entries(),
    ([hostName, items]) => calculateTpuVmAnalytics(hostName, items),
  );

  return {
    tpuVmMap,
    collectiveStats: Array.from(collectiveMap.values()),
    availableTpuVms,
    tpuVmAnalyticsData,
  };
}

/** Prepares host-specific step latency chart configurations. */
export function prepareTpuVmStepLatencies(
  db: GrpcMetricsDatabase,
  chartOptions?: ChartOptions,
) {
  const tpuVmStepLatencies: TpuVmStepLatencyInfo[] = [];
  if (!db.chunk_latency_series) return tpuVmStepLatencies;

  const options = chartOptions || COMMON_CHART_OPTIONS;

  const tpuVmMap = new Map<string, TimeSeriesPoint[]>();
  for (const pt of db.chunk_latency_series) {
    const h = pt.tpu_vm || 'Unknown Host';
    let pts = tpuVmMap.get(h);
    if (!pts) {
      pts = [];
      tpuVmMap.set(h, pts);
    }
    pts.push(pt);
  }

  // Hoist column definitions outside of the loop to avoid re-creation
  const stepLatencyCols: readonly DataTableColumn[] = [
    {id: 't', label: 'Time', type: 'number'},
    {id: 'v', label: 'Val', type: 'number'},
  ];
  const histogramCols: readonly DataTableColumn[] = [
    {id: 'l', label: 'Latency', type: 'number'},
  ];

  for (const [hostname, points] of tpuVmMap.entries()) {
    const active = points.filter((pt) => (pt.value ?? 0) !== 0);
    if (active.length > 0) {
      tpuVmStepLatencies.push({
        tpu_vm: hostname,
        stepLatencyChartDataInfo: {
          data: {
            cols: [...stepLatencyCols],
            rows: active.map((pt) => ({
              c: [{v: pt.timestamp_rel_sec ?? 0}, {v: pt.value ?? 0}],
            })),
          },
          dataProvider: new DefaultDataProvider(),
          options: {
            ...options,
            vAxis: {
              ...((options as google.visualization.LineChartOptions).vAxis ||
                {}),
              title: 'Latency (ms)',
              viewWindow: {min: 0},
            },
          },
        },
        stepLatencyHistogramDataInfo: {
          data: {
            cols: [...histogramCols],
            rows: active.map((pt) => ({c: [{v: pt.value ?? 0}]})),
          },
          dataProvider: new DefaultDataProvider(),
          options: {
            ...options,
            vAxis: {
              ...((options as google.visualization.LineChartOptions).vAxis ||
                {}),
              title: 'Count',
              viewWindow: {min: 0},
            },
          },
        },
      });
    }
  }
  return tpuVmStepLatencies.sort((a, b) => a.tpu_vm.localeCompare(b.tpu_vm));
}

/**
 * Parses a qualified endpoint_id formatted as host_name_socketId
 * into its individual structural parts.
 */
export function parseEndpointId(endpointId: string): {
  host: string;
  connectionNum: string;
  endpointNum: string;
  connectionKey: string;
} {
  if (!endpointId) {
    return {
      host: 'Unknown',
      connectionNum: 'Unknown',
      endpointNum: 'Unknown',
      connectionKey: 'Unknown',
    };
  }
  const parts = endpointId.split('_');
  if (parts.length >= 2) {
    const host = parts.slice(0, parts.length - 1).join('_');
    const socketId = parts[parts.length - 1];
    return {
      host,
      connectionNum: socketId,
      endpointNum: socketId,
      connectionKey: endpointId,
    };
  }
  return {
    host: 'Unknown',
    connectionNum: 'Unknown',
    endpointNum: endpointId,
    connectionKey: endpointId,
  };
}

/** Parses SystemConfig from trace events. */
export function parseSystemConfigs(traceEvents?: TraceEvent[]): SystemConfig[] {
  if (!traceEvents) return [];
  const configs: SystemConfig[] = [];
  for (const event of traceEvents) {
    if (event.name === 'GrpcSystemConfigTrace' && event.args) {
      configs.push(event.args as SystemConfig);
    }
  }
  return configs;
}

/** Parses NetworkInterfaceInfo items from SystemConfig array. */
export function parseNetworkInterfaces(
  configs?: SystemConfig[],
): NetworkInterfaceInfo[] {
  if (!configs) return [];
  const interfaces: NetworkInterfaceInfo[] = [];

  for (const config of configs) {
    const raw = config['network_interfaces'];
    if (!raw) continue;

    let items: unknown;
    if (typeof raw === 'string') {
      try {
        items = JSON.parse(raw);
      } catch (e) {
        console.error('Failed to parse network_interfaces JSON:', raw, e);
        continue;
      }
    } else if (Array.isArray(raw)) {
      items = raw;
    }

    if (Array.isArray(items)) {
      const hostName = (config.tpu_vm || config.Host || '') as string;
      for (const item of items) {
        if (typeof item === 'object' && item !== null && 'iface' in item) {
          const ifaceObj = item as Record<string, unknown>;
          interfaces.push({
            tpu_vm: hostName,
            iface: String(ifaceObj['iface'] || ''),
            is_physical: Boolean(ifaceObj['is_physical']),
            numa_node:
              ifaceObj['numa_node'] !== undefined
                ? Number(ifaceObj['numa_node'])
                : undefined,
            speed_mbps:
              ifaceObj['speed_mbps'] !== undefined
                ? Number(ifaceObj['speed_mbps'])
                : undefined,
            mtu:
              ifaceObj['mtu'] !== undefined
                ? Number(ifaceObj['mtu'])
                : undefined,
            operstate: String(ifaceObj['operstate'] || 'unknown'),
            duplex: String(ifaceObj['duplex'] || 'unknown'),
            carrier:
              ifaceObj['carrier'] !== undefined
                ? Number(ifaceObj['carrier'])
                : undefined,
            delta_rx_bytes:
              ifaceObj['delta_rx_bytes'] !== undefined
                ? Number(ifaceObj['delta_rx_bytes'])
                : 0,
            delta_tx_bytes:
              ifaceObj['delta_tx_bytes'] !== undefined
                ? Number(ifaceObj['delta_tx_bytes'])
                : 0,
            delta_rx_packets:
              ifaceObj['delta_rx_packets'] !== undefined
                ? Number(ifaceObj['delta_rx_packets'])
                : 0,
            delta_tx_packets:
              ifaceObj['delta_tx_packets'] !== undefined
                ? Number(ifaceObj['delta_tx_packets'])
                : 0,
            delta_rx_dropped:
              ifaceObj['delta_rx_dropped'] !== undefined
                ? Number(ifaceObj['delta_rx_dropped'])
                : 0,
            delta_tx_dropped:
              ifaceObj['delta_tx_dropped'] !== undefined
                ? Number(ifaceObj['delta_tx_dropped'])
                : 0,
            delta_rx_errors:
              ifaceObj['delta_rx_errors'] !== undefined
                ? Number(ifaceObj['delta_rx_errors'])
                : 0,
            delta_tx_errors:
              ifaceObj['delta_tx_errors'] !== undefined
                ? Number(ifaceObj['delta_tx_errors'])
                : 0,
            start_rx_bytes:
              ifaceObj['start_rx_bytes'] !== undefined
                ? Number(ifaceObj['start_rx_bytes'])
                : 0,
            stop_rx_bytes:
              ifaceObj['stop_rx_bytes'] !== undefined
                ? Number(ifaceObj['stop_rx_bytes'])
                : 0,
            start_tx_bytes:
              ifaceObj['start_tx_bytes'] !== undefined
                ? Number(ifaceObj['start_tx_bytes'])
                : 0,
            stop_tx_bytes:
              ifaceObj['stop_tx_bytes'] !== undefined
                ? Number(ifaceObj['stop_tx_bytes'])
                : 0,
          });
        }
      }
    }
  }

  return interfaces;
}

/**
 * Formats a byte count into a human-readable string with units (e.g. 1.83 GB, 500 KB, 120 B).
 */
export function formatBytes(bytes?: number): string {
  if (bytes === undefined || bytes === null || isNaN(bytes)) {
    return '0 B';
  }
  if (bytes === 0) {
    return '0 B';
  }
  const units = ['B', 'KB', 'MB', 'GB', 'TB', 'PB'];
  const k = 1024;
  const i = Math.floor(Math.log(Math.abs(bytes)) / Math.log(k));
  const unitIndex = Math.min(Math.max(0, i), units.length - 1);
  if (unitIndex === 0) {
    return `${bytes.toLocaleString()} B`;
  }
  const value = bytes / Math.pow(k, unitIndex);
  return `${value.toLocaleString(undefined, {
    minimumFractionDigits: 0,
    maximumFractionDigits: 2,
  })} ${units[unitIndex]}`;
}

/**
 * Groups network interfaces by TPU VM host, filtering for physical interfaces
 * and ordering interfaces naturally by name.
 */
export function groupNetworkInterfacesByHost(
  interfaces: NetworkInterfaceInfo[],
): HostNetworkInterfacesGroup[] {
  const groupsMap = new Map<string, NetworkInterfaceInfo[]>();
  for (const nic of interfaces) {
    if (!nic.is_physical) {
      continue;
    }
    const host = nic.tpu_vm || 'Unknown Host';
    if (!groupsMap.has(host)) {
      groupsMap.set(host, []);
    }
    groupsMap.get(host)!.push(nic);
  }

  const groups: HostNetworkInterfacesGroup[] = [];
  for (const [tpuVm, nics] of groupsMap.entries()) {
    nics.sort((a, b) =>
      a.iface.localeCompare(b.iface, undefined, {numeric: true}),
    );
    groups.push({
      tpu_vm: tpuVm,
      interfaces: nics,
      isExpanded: true,
    });
  }

  groups.sort((a, b) =>
    a.tpu_vm.localeCompare(b.tpu_vm, undefined, {numeric: true}),
  );
  return groups;
}

/** Parses XOR timestamps from trace events and calculates clock skew. */
export function parseXorTimestamps(
  traceEvents?: TraceEvent[],
): Map<string, XorTimestamps> {
  const map = new Map<string, XorTimestamps>();
  if (!traceEvents) return map;

  for (const event of traceEvents) {
    if (!event.args) continue;
    const rawId = event.args.chunk_id || event.args.flow_id;
    if (rawId === undefined || rawId === null) continue;
    const chunkId = normalizeIdForMatching(rawId);
    let info = map.get(chunkId);
    if (!info) {
      info = {};
      map.set(chunkId, info);
    }

    const isClient =
      event.name === 'grpc_core::XorClientFilter::ClientCallMetrics';
    const isServer =
      event.name === 'grpc_core::XorServerFilter::ServerCallMetrics';

    if (event.args.start_time_ns !== undefined) {
      let startNs = Number(event.args.start_time_ns);
      if (!isNaN(startNs)) {
        if (startNs > 1e11 && startNs < 1e15) {
          startNs = startNs * 1e6; // epoch ms -> epoch ns
        }
        if (isClient) {
          info.client_start_time_ns = startNs;
        }
        if (isServer) {
          info.server_start_time_ns = startNs;
        }
      }
    }
    if (event.args.client_latency_ns !== undefined) {
      if (isClient) {
        info.client_latency_ns = Number(event.args.client_latency_ns);
      }
    }
    if (event.args.server_latency_ns !== undefined) {
      if (isServer) {
        info.server_latency_ns = Number(event.args.server_latency_ns);
      }
    }

    if (isClient) {
      if (event.args.request_size !== undefined) {
        info.client_request_size = Number(event.args.request_size);
      }
      if (event.args.response_size !== undefined) {
        info.client_response_size = Number(event.args.response_size);
      }
      if (event.args.on_client_initial_metadata_time_ns !== undefined) {
        info.client_on_client_initial_metadata_time_ns = Number(
          event.args.on_client_initial_metadata_time_ns,
        );
      }
      if (event.args.on_client_to_server_first_message_time_ns !== undefined) {
        info.client_on_client_to_server_first_message_time_ns = Number(
          event.args.on_client_to_server_first_message_time_ns,
        );
      }
      if (event.args.on_server_initial_metadata_time_ns !== undefined) {
        info.client_on_server_initial_metadata_time_ns = Number(
          event.args.on_server_initial_metadata_time_ns,
        );
      }
      if (event.args.on_server_to_client_first_message_time_ns !== undefined) {
        info.client_on_server_to_client_first_message_time_ns = Number(
          event.args.on_server_to_client_first_message_time_ns,
        );
      }
      if (event.args.on_server_trailing_metadata_time_ns !== undefined) {
        info.client_on_server_trailing_metadata_time_ns = Number(
          event.args.on_server_trailing_metadata_time_ns,
        );
      }
    } else if (isServer) {
      if (event.args.on_client_initial_metadata_time_ns !== undefined) {
        info.server_on_client_initial_metadata_time_ns = Number(
          event.args.on_client_initial_metadata_time_ns,
        );
      }
      if (event.args.on_client_to_server_first_message_time_ns !== undefined) {
        info.server_on_client_to_server_first_message_time_ns = Number(
          event.args.on_client_to_server_first_message_time_ns,
        );
      }
      if (event.args.on_server_initial_metadata_time_ns !== undefined) {
        info.server_on_server_initial_metadata_time_ns = Number(
          event.args.on_server_initial_metadata_time_ns,
        );
      }
      if (event.args.on_server_to_client_first_message_time_ns !== undefined) {
        info.server_on_server_to_client_first_message_time_ns = Number(
          event.args.on_server_to_client_first_message_time_ns,
        );
      }
    }
  }

  return map;
}

/**
 * TCP events of a flow: the first SENDMSG, SCHEDULED and SENT, and the last
 * ACK (a write can need several sendmsg() calls).
 */
export declare interface TcpEventsInfo {
  sendmsg?: TraceEvent;
  scheduled?: TraceEvent;
  sent?: TraceEvent;
  ack?: TraceEvent;
}

/** Parsed Fathom times of TCP events (NaN: no usable time). */
const tcpEventTimeNsCache = new WeakMap<TraceEvent, number>();

/** Returns the Fathom timestamp of a TCP event in ns, or undefined. */
export function getTcpEventTimeNs(event?: TraceEvent): number | undefined {
  if (!event) return undefined;
  let ns = tcpEventTimeNsCache.get(event);
  if (ns === undefined) {
    const raw = event.args?.fathom_timestamp;
    const parsed =
      raw === undefined || raw === null || raw === ''
        ? NaN
        : parseFathomTimestampToNs(raw);
    ns = parsed > 0 ? parsed : NaN;
    tcpEventTimeNsCache.set(event, ns);
  }
  return isNaN(ns) ? undefined : ns;
}

/** Returns the Fathom `byte_offset` of a TCP event, or undefined. */
export function getTcpEventByteOffset(event?: TraceEvent): number | undefined {
  return parseMetricVal(event?.args?.['byte_offset']);
}

/** Earlier by Fathom time; events with a Fathom time win. */
function isEarlierTcpEvent(
  candidate: TraceEvent,
  current?: TraceEvent,
): boolean {
  if (!current) return true;
  const candidateNs = getTcpEventTimeNs(candidate);
  const currentNs = getTcpEventTimeNs(current);
  if (candidateNs !== undefined && currentNs !== undefined) {
    return candidateNs < currentNs;
  }
  return candidateNs !== undefined;
}

/**
 * Later by Fathom time, then by byte offset; events with a Fathom time win.
 * Without times the last seen event is kept.
 */
function isLaterTcpEvent(candidate: TraceEvent, current?: TraceEvent): boolean {
  if (!current) return true;
  const candidateNs = getTcpEventTimeNs(candidate);
  const currentNs = getTcpEventTimeNs(current);
  if (candidateNs === undefined || currentNs === undefined) {
    return currentNs === undefined;
  }
  if (candidateNs !== currentNs) return candidateNs > currentNs;
  return (
    (getTcpEventByteOffset(candidate) ?? -1) >
    (getTcpEventByteOffset(current) ?? -1)
  );
}

/** Parses TCP events per flow (see TcpEventsInfo). */
export function parseTcpEvents(
  traceEvents?: TraceEvent[],
): Map<string, TcpEventsInfo> {
  const map = new Map<string, TcpEventsInfo>();
  if (!traceEvents) return map;

  for (const event of traceEvents) {
    if (!event.args) continue;
    const rawId = event.args.chunk_id || event.args.flow_id;
    if (rawId === undefined || rawId === null) continue;
    const chunkId = normalizeIdForMatching(rawId);
    let info = map.get(chunkId);
    if (!info) {
      info = {};
      map.set(chunkId, info);
    }

    const metadataType = String(event.args.metadata_type || event.name || '')
      .replace(/^["']+|["']+$/g, '')
      .toUpperCase();
    const isSendMsg = metadataType.includes('SENDMSG');
    const isScheduled = metadataType.includes('SCHEDULED');
    const isSent = metadataType.includes('SENT');
    const isAck = metadataType.includes('ACK');

    if (isSendMsg) {
      if (isEarlierTcpEvent(event, info.sendmsg)) info.sendmsg = event;
    } else if (isScheduled) {
      if (isEarlierTcpEvent(event, info.scheduled)) info.scheduled = event;
    } else if (isSent) {
      if (isEarlierTcpEvent(event, info.sent)) info.sent = event;
    } else if (isAck) {
      if (isLaterTcpEvent(event, info.ack)) info.ack = event;
    }
  }

  // An ACK at or before the first SENDMSG cannot acknowledge this chunk.
  for (const info of map.values()) {
    const sendmsgNs = getTcpEventTimeNs(info.sendmsg);
    const ackNs = getTcpEventTimeNs(info.ack);
    if (sendmsgNs !== undefined && ackNs !== undefined && ackNs <= sendmsgNs) {
      info.ack = undefined;
    }
  }

  return map;
}

/**
 * Shown next to a TCP latency that ends before the last byte of the write was
 * acknowledged.
 */
export const INCOMPLETE_TCP_LATENCY_NOTE =
  'This write needed more than one sendmsg() call and the ACK of its last ' +
  'byte was not recorded, so the TCP latency shown may be lower than it ' +
  'actually was.';

/**
 * True when the backend flagged the TCP latency as incomplete, or the chunk has
 * TCP events but no TCP latency.
 */
export function hasIncompleteTcpEvents(
  straggler: GrpcStraggler,
  tcpEvents?: TcpEventsInfo,
): boolean {
  if (straggler.tcp_latency_incomplete) return true;
  const hasTcpEvents = Boolean(
    tcpEvents &&
      (tcpEvents.sendmsg ||
        tcpEvents.scheduled ||
        tcpEvents.sent ||
        tcpEvents.ack),
  );
  const latencyMs = straggler.ack_to_sendmsg_ms;
  const hasLatency =
    latencyMs !== undefined && latencyMs !== null && !isNaN(latencyMs);
  return hasTcpEvents && !hasLatency;
}

/** Shown next to the TCP latency of an RPC that shared its TCP write. */
export const SHARED_TCP_WRITE_NOTE =
  'This RPC was sent in one TCP write together with other RPCs, so they ' +
  'share the same TCP timestamps. gRPC combines queued RPCs into one write ' +
  'under heavy concurrent traffic, so the TCP latency shown may be higher ' +
  'than this RPC actually took.';

/**
 * Returns the warning shown next to the TCP latency of a chunk, or undefined
 * when the latency needs no warning.
 */
export function getTcpLatencyNote(
  straggler: GrpcStraggler,
): string | undefined {
  const notes: string[] = [];
  if (straggler.tcp_latency_incomplete) notes.push(INCOMPLETE_TCP_LATENCY_NOTE);
  if (straggler.tcp_write_shared) notes.push(SHARED_TCP_WRITE_NOTE);
  return notes.length > 0 ? notes.join(' ') : undefined;
}

/** Bit-rate units in ascending order of decimal (SI) multiples of 1000. */
const BIT_RATE_UNITS = ['bps', 'Kbps', 'Mbps', 'Gbps', 'Tbps'];

/**
 * Formats a bit rate (bits/s) into a human-readable string scaled to the
 * largest fitting unit, e.g. 24,073,294,600 bits/s is rendered as
 * `24.07 Gbps`. Network rates follow the decimal (SI) convention, so
 * 1 Gbps = 10^9 bits/s.
 */
export function formatBitRate(bitsPerSec: number): string {
  if (!Number.isFinite(bitsPerSec) || bitsPerSec === 0) {
    return '0 bps';
  }
  const exponent = Math.floor(Math.log10(Math.abs(bitsPerSec)) / 3);
  const unitIndex = Math.min(Math.max(0, exponent), BIT_RATE_UNITS.length - 1);
  const scaledVal = bitsPerSec / Math.pow(1000, unitIndex);
  return `${scaledVal.toLocaleString(undefined, {
    minimumFractionDigits: 0,
    maximumFractionDigits: 2,
  })} ${BIT_RATE_UNITS[unitIndex]}`;
}

/**
 * Formats a metric value based on its key name.
 * If the value is a number, it will be formatted as a string with appropriate units.
 * Otherwise, the value will be returned unchanged.
 */
export function formatMetricValue(key: string, val: unknown): unknown {
  if (typeof val !== 'number') {
    return val;
  }

  if (key.includes('app_limited')) {
    return val.toLocaleString(undefined, {
      minimumFractionDigits: 0,
      maximumFractionDigits: 2,
    });
  } else if (key.includes('rate')) {
    // Raw values in args are Bytes/s. Convert to bits/s and scale to
    // Gbps/Mbps/etc. for readability.
    return formatBitRate(val * 8);
  } else if (key.includes('data') || key.includes('byte')) {
    const formattedNum = val.toLocaleString(undefined, {
      minimumFractionDigits: 0,
      maximumFractionDigits: 2,
    });
    return `${formattedNum} bytes`;
  } else if (
    key.includes('packet') ||
    key.includes('window') ||
    key.includes('cwnd') ||
    key.includes('ssthresh') ||
    key === 'congestion_window'
  ) {
    const formattedNum = val.toLocaleString(undefined, {
      minimumFractionDigits: 0,
      maximumFractionDigits: 2,
    });
    return `${formattedNum} packets`;
  } else if (key.includes('usec') || key.includes('rtt')) {
    const formattedNum = val.toLocaleString(undefined, {
      minimumFractionDigits: 0,
      maximumFractionDigits: 2,
    });
    return `${formattedNum} us`;
  } else {
    return val.toLocaleString(undefined, {
      minimumFractionDigits: 0,
      maximumFractionDigits: 2,
    });
  }
}

/**
 * Matches megascale transfer-key labels that end in a collective op tag plus a
 * shard range, e.g. `all-reduce.169_0_rs_0_sr_0_1`. Group 1 is the XLA
 * rendezvous name (`all-reduce.169_0`).
 */
const DCN_OP_SUFFIX_PATTERN =
  /^(.*?)_(?:ag|rs|ALL_GATHER|REDUCE_SCATTER|ALL_REDUCE|ALL_TO_ALL|ONE_TO_ONE|RAGGED_ALL_TO_ALL|send|recv|recv-done)(?:_[0-9]+)*_sr_[0-9]+_[0-9]+$/;

/**
 * Matches megascale transfer-key labels that end only in a shard range, e.g.
 * `all-reduce_1_sr_0_1`. Group 1 is the XLA rendezvous name (`all-reduce_1`).
 */
const DCN_SHARD_RANGE_SUFFIX_PATTERN = /^(.*?)_sr_[0-9]+_[0-9]+$/;

/**
 * Converts a DCN collective label into the XLA rendezvous name so that it
 * matches the names shown in Megascale Stats and Perfetto.
 *
 * Only the suffixes appended by megascale to the transfer-key label (collective
 * op tag and `_sr_<a>_<b>` shard range) are removed. The rendezvous name itself,
 * including any trailing `_<N>` generated by XLA (e.g. `psum.197_0`), is kept
 * intact.
 */
export function cleanDcnLabel(label: string): string {
  const opSuffixMatch = label.match(DCN_OP_SUFFIX_PATTERN);
  if (opSuffixMatch?.[1]) {
    return opSuffixMatch[1];
  }
  const shardRangeMatch = label.match(DCN_SHARD_RANGE_SUFFIX_PATTERN);
  if (shardRangeMatch?.[1]) {
    return shardRangeMatch[1];
  }
  return label;
}

/** Extracts the full IPv6 address from a raw peer address string. */
export function extractFullIpv6(peerAddress: string): string {
  if (!peerAddress) return '';
  const parts = peerAddress.split('-');
  const remotePart = parts[parts.length - 1];

  const bracketMatch = remotePart.match(/\[(.*?)\]/);
  return bracketMatch ? bracketMatch[1] : remotePart;
}

/** Groups stragglers by TPU VM and computes latency distribution profiles per Peer. */
export function computeTpuVmLatencyGroups(
  stragglers: GrpcStraggler[],
): TpuVmLatencyInfo[] {
  const hostGroups = new Map<string, Map<string, number[]>>();

  for (const s of stragglers) {
    if (!s.tpu_vm || !s.peer_address) continue;

    const peerIpv6 = extractFullIpv6(s.peer_address);
    if (!peerIpv6) continue;

    let peerMap = hostGroups.get(s.tpu_vm);
    if (!peerMap) {
      peerMap = new Map<string, number[]>();
      hostGroups.set(s.tpu_vm, peerMap);
    }

    let latencies = peerMap.get(peerIpv6);
    if (!latencies) {
      latencies = [];
      peerMap.set(peerIpv6, latencies);
    }
    latencies.push(s.latency_ms || 0);
  }

  const result: TpuVmLatencyInfo[] = [];

  for (const [tpuVm, peerMap] of hostGroups.entries()) {
    let hostMax = 0;
    const peersList: PeerNodeLatencyInfo[] = [];

    for (const [peerAddress, latencies] of peerMap.entries()) {
      latencies.sort((a, b) => a - b);
      const count = latencies.length;
      if (count === 0) continue;

      const p50 = latencies[Math.floor(count * 0.5)];
      const p99 = latencies[Math.floor(count * 0.99)];
      const p100 = latencies[count - 1];

      if (p100 > hostMax) {
        hostMax = p100;
      }

      peersList.push({
        peer_address: peerAddress,
        p50_latency_ms: p50,
        p99_latency_ms: p99,
        p100_latency_ms: p100,
      });
    }

    result.push({
      tpu_vm: tpuVm,
      max_latency_ms: hostMax,
      peers: peersList,
      isExpanded: false,
    });
  }

  // Sort top-level by max latency descending
  result.sort((a, b) => b.max_latency_ms - a.max_latency_ms);

  return result;
}
