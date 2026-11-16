import {ChartDataInfo} from 'org_xprof/frontend/app/common/interfaces/chart';

// Disable name casing warnings since interface properties map directly to
// snake_case C++ Protobuf fields serialized with preserve_proto_field_names.
/* tslint:disable:enforce-name-casing */

/** Insight severity levels. */
export enum InsightSeverity {
  INFO = 'info',
  WARNING = 'warning',
  CRITICAL = 'critical',
}

/** Insight layers for categorization. */
export enum InsightLayer {
  ML = 'ML',
  RPC = 'RPC',
  TCP = 'TCP',
}

/** Specific types of insights identified in the data. */
export enum InsightType {
  NETWORK_CONGESTION_LOSS = 'Network Congestion (Packet Loss)',
  NETWORK_CONGESTION_ECN = 'Network Congestion (ECN)',

  APPLICATION_BOUND = 'Application Bound (Not Network)',
  HIGH_APPLICATION_LATENCY = 'High application (MXLA/gRPC) latency',
  HIDDEN_NETWORK_STRAGGLER = 'Hidden Network Straggler',
  RECEIVER_WINDOW_LIMITED = 'Receiver Window Limited',
  HIGH_TCP_QUEUING_DELAY = 'High TCP Queuing Delay',
  SENDER_BUFFER_LIMITED = 'Sender Buffer Limited',
}

/** Latency info for each peer address grouped per TPU VM/Host. */
export declare interface PeerNodeLatencyInfo {
  readonly peer_address: string;
  readonly p50_latency_ms: number;
  readonly p99_latency_ms: number;
  readonly p100_latency_ms: number;
}

/** Grouped latency details per TPU VM. */
export declare interface TpuVmLatencyInfo {
  readonly tpu_vm: string;
  readonly max_latency_ms: number;
  readonly peers: PeerNodeLatencyInfo[];
  isExpanded?: boolean;
}

/** Insight information for a straggler. */
export declare interface Insight {
  readonly text: string;
  readonly severity: string;
  readonly description?: string;
}

/** A column specification for SimpleDataTable. */
export declare interface DataTableColumn {
  readonly id?: string;
  readonly label?: string;
  readonly type: string;
}

/** A single point marker on the timeline. */
export declare interface TimelineMarker {
  readonly positionLeft: number;
  readonly label: string;
}

/** TCP segments details for timeline visualization. */
export declare interface TcpSegmentsInfo {
  readonly sendQueuePct: number;
  readonly sendQueueMs: number;
  readonly deliveryPct: number;
  readonly deliveryMs: number;
  readonly inflightPct: number;
  readonly inflightMs: number;
}

/** Timeline positioning for visualization. */
export declare interface TimelineInfo {
  readonly clientLeft: number;
  readonly clientWidth: number;
  readonly serverLeft: number;
  readonly serverWidth: number;
  readonly tcpLeft?: number;
  readonly tcpWidth?: number;
  readonly clientMarkers?: TimelineMarker[];
  readonly serverMarkers?: TimelineMarker[];
  readonly tcpMarkers?: TimelineMarker[];
  readonly markers?: TimelineMarker[];
  readonly tcpSegments?: TcpSegmentsInfo;
  readonly hasError?: boolean;
}

/** Details about a gRPC straggler. */
export declare interface GrpcStraggler {
  readonly chunk_id: string;
  readonly latency_ms: number;
  readonly grpc_client_latency_ms: number;
  readonly grpc_server_latency_ms: number;
  readonly grpc_status: number;
  readonly tpu_vm: string;
  readonly endpoint_ids?: string[];
  readonly hlo_collective_name?: string;
  readonly slack_time_ms?: number;
  readonly tpu_stall_time_ms?: number;
  readonly communication_overlap_ms?: number;
  readonly tcp_sample_count?: number;
  readonly tcp_closed_before_acked?: boolean;
  readonly peer_address?: string;
  readonly ack_to_sendmsg_ms?: number;
  /** The RPC shared its TCP write (and TCP events) with other RPCs. */
  readonly tcp_write_shared?: boolean;
  /** `ack_to_sendmsg_ms` ends before the ACK of the write's last byte. */
  readonly tcp_latency_incomplete?: boolean;
  readonly dcn_source_device_id?: string;
  readonly dcn_destination_device_id?: string;
  readonly dcn_launch_id?: string;
  readonly dcn_loop_index?: string;
  readonly dcn_chunk_index?: string;
  readonly dcn_shard_range?: string;
  readonly dcn_collective_label?: string;
  readonly raw_transfer_key?: string;
  readonly client_request_size?: number;
  readonly client_response_size?: number;

  insights?: Insight[];
  latencyBreakdownDataInfo?: ChartDataInfo;
  timelineInfo?: TimelineInfo;
  searchStr?: string;
}

/** XOR filter relative timestamps in nanoseconds. */
export declare interface XorTimestamps {
  client_start_time_ns?: number;
  client_latency_ns?: number;
  client_on_client_initial_metadata_time_ns?: number;
  client_on_client_to_server_first_message_time_ns?: number;
  client_on_server_initial_metadata_time_ns?: number;
  client_on_server_to_client_first_message_time_ns?: number;
  client_on_server_trailing_metadata_time_ns?: number;

  server_start_time_ns?: number;
  server_latency_ns?: number;
  server_on_client_initial_metadata_time_ns?: number;
  server_on_client_to_server_first_message_time_ns?: number;
  server_on_server_initial_metadata_time_ns?: number;
  server_on_server_to_client_first_message_time_ns?: number;
  clock_skew_ns?: number;
  client_request_size?: number;
  client_response_size?: number;
}

/** Expected argument fields in a trace event. */
export declare interface TraceEventArgs {
  chunk_id?: string | number;
  flow_id?: string | number;
  fathom_timestamp?: string | number;
  local_addr?: string;
  peer_addr?: string;
  start_time_ns?: string | number;
  client_latency_ns?: string | number;
  server_latency_ns?: string | number;
  on_client_initial_metadata_time_ns?: string | number;
  on_client_to_server_first_message_time_ns?: string | number;
  on_server_initial_metadata_time_ns?: string | number;
  on_server_to_client_first_message_time_ns?: string | number;
  on_server_trailing_metadata_time_ns?: string | number;
  metadata_type?: string;
  request_size?: string | number;
  response_size?: string | number;
  packet_retx?: number;
  packet_delivered?: number;
  busy_usec?: number;
  sndbuf_limited_usec?: number;
  [key: string]: unknown;
}

/** A raw trace event from the XSpace JSON. */
export declare interface TraceEvent {
  name?: string;
  timestamp_ns?: string | number;
  args?: TraceEventArgs;
  [key: string]: unknown;
}

/** Host system configurations. */
export declare interface SystemConfig {
  Host?: string;
  tpu_vm?: string;
  cpuidle_driver?: string;
  tcp_rmem?: string;
  network_interfaces?: string | NetworkInterfaceInfo[];
  [key: string]: unknown;
}

/** Network interface information parsed from GrpcSystemConfigTrace. */
export declare interface NetworkInterfaceInfo {
  tpu_vm?: string;
  iface: string;
  is_physical?: boolean;
  numa_node?: number;
  speed_mbps?: number;
  mtu?: number;
  operstate?: string;
  duplex?: string;
  carrier?: number;
  delta_rx_bytes?: number;
  delta_tx_bytes?: number;
  delta_rx_packets?: number;
  delta_tx_packets?: number;
  delta_rx_dropped?: number;
  delta_tx_dropped?: number;
  delta_rx_errors?: number;
  delta_tx_errors?: number;
  start_rx_bytes?: number;
  stop_rx_bytes?: number;
  start_tx_bytes?: number;
  stop_tx_bytes?: number;
}

/** Grouped network interfaces per TPU VM host. */
export declare interface HostNetworkInterfacesGroup {
  readonly tpu_vm: string;
  readonly interfaces: NetworkInterfaceInfo[];
  isExpanded?: boolean;
}

/** Snapshot of TCP metrics at a specific point in time. */
export declare interface TcpStatsSnapshot {
  readonly min_rtt?: number;
  readonly congestion_window?: number;
  readonly pacing_rate?: number;
  readonly delivery_rate?: number;
  readonly packet_sent?: number;
  readonly packet_delivered?: number;
  readonly data_sent?: number;
  readonly data_retx?: number;
  readonly packet_retx?: number;
  readonly busy_usec?: number;
  readonly rwnd_limited_usec?: number;
  readonly sndbuf_limited_usec?: number;
  readonly data_not_sent?: number;
  readonly is_delivery_rate_app_limited?: number;
  readonly srtt?: number;
  readonly net_rtt_usec?: number;
  readonly data_notsent?: number;
  readonly byte_offset?: number;
  readonly snd_ssthresh?: number;
  readonly ssthresh?: number;
  readonly packet_spurious_retx?: number;
  readonly packet_delivered_ce?: number;
  readonly reordering?: number;
  readonly recurring_retrans?: number;
}

/** Grouping of insights by type. */
export declare interface InsightSummary {
  readonly type: InsightType;
  readonly description: string;
  readonly severity: InsightSeverity;
  readonly layer: InsightLayer;
  readonly flows: GrpcStraggler[];
}

/** Per-TPU VM performance analytics. */
export declare interface TpuVmAnalytics {
  readonly tpu_vm: string;
  readonly total_transfers: number;
  readonly p50_latency_ms: number;
  readonly p99_latency_ms: number;
  readonly p100_latency_ms: number;
  readonly high_latency_count: number;
}

/** Stats for HLO collectives. */
export declare interface HloCollectiveStats {
  hlo_collective_name: string;
  total_flows: number;
  total_latency_ms: number;
  total_tpu_stall_ms: number;
  total_slack_ms: number;
}

/**
 * A single point in a time series. The JSON omits `timestamp_rel_sec` and
 * `value` when they are 0.
 */
export declare interface TimeSeriesPoint {
  readonly timestamp_rel_sec?: number;
  readonly value?: number;
  readonly tpu_vm: string;
  readonly endpoint_id: string;
}

/** Information about downsampled series. */
export declare interface DownsampledInfo {
  readonly tpu_vm: string;
  readonly series_name: string;
}

/** Container for gRPC metrics database. */
export declare interface GrpcMetricsDatabase {
  readonly top_stragglers: GrpcStraggler[];
  readonly chunk_latency_series: TimeSeriesPoint[];
  readonly tcp_min_rtt_series: TimeSeriesPoint[];
  readonly grpc_client_latency_series: TimeSeriesPoint[];
  readonly grpc_server_latency_series: TimeSeriesPoint[];
  readonly tcp_packet_retx_series: TimeSeriesPoint[];
  readonly tcp_data_sent_series: TimeSeriesPoint[];
  readonly tcp_pacing_rate_series: TimeSeriesPoint[];
  readonly tcp_delivery_rate_series: TimeSeriesPoint[];
  readonly tcp_ssthresh_series: TimeSeriesPoint[];
  readonly tcp_is_delivery_rate_app_limited_series: TimeSeriesPoint[];
  readonly tcp_byte_offset_series: TimeSeriesPoint[];
  readonly tcp_data_notsent_series: TimeSeriesPoint[];
  readonly tcp_packet_spurious_retx_series: TimeSeriesPoint[];
  readonly tcp_packet_delivered_ce_series: TimeSeriesPoint[];
  readonly tcp_reordering_series: TimeSeriesPoint[];
  readonly tcp_recurring_retrans_series: TimeSeriesPoint[];
  readonly tcp_busy_usec_series: TimeSeriesPoint[];
  readonly tcp_rwnd_limited_usec_series: TimeSeriesPoint[];
  readonly tcp_sndbuf_limited_usec_series: TimeSeriesPoint[];
  readonly tcp_congestion_window_series: TimeSeriesPoint[];
  readonly tcp_packet_sent_series: TimeSeriesPoint[];
  readonly tcp_packet_delivered_series: TimeSeriesPoint[];
  readonly total_network_stall_ms: number;
  readonly total_compute_overlap_ms: number;
  readonly downsampled_info: DownsampledInfo[];
  readonly trace_events_json?: string[];
}

/** Information about step latencies for a TPU VM. */
export interface TpuVmStepLatencyInfo {
  readonly tpu_vm: string;
  readonly stepLatencyChartDataInfo: ChartDataInfo;
  readonly stepLatencyHistogramDataInfo: ChartDataInfo;
}
