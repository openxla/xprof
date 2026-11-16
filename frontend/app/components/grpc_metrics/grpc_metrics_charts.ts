import {
  ChartDataInfo,
  ChartOptions,
} from 'org_xprof/frontend/app/common/interfaces/chart';
import {DefaultDataProvider} from 'org_xprof/frontend/app/components/chart/default_data_provider';

/** Base chart options used across all gRPC metric charts. */
export const BASE_CANVAS_OPTIONS: google.visualization.LineChartOptions = {
  colors: [
    '#4285F4',
    '#DB4437',
    '#F4B400',
    '#0F9D58',
    '#AB47BC',
    '#00ACC1',
    '#FF7043',
    '#9E9D24',
    '#5C6BC0',
    '#F06292',
    '#4DB6AC',
    '#C2185B',
    '#E67E22',
    '#2ECC71',
    '#3498DB',
    '#9B59B6',
    '#34495E',
    '#16A085',
    '#27AE60',
    '#2980B9',
    '#8E44AD',
    '#2C3E50',
    '#F1C40F',
    '#D35400',
    '#C0392B',
  ],
  height: 400,
  pointSize: 2,
  dataOpacity: 0.8,
  animation: {duration: 0},
  explorer: {actions: ['dragToZoom', 'rightClickToReset']},
  tooltip: {trigger: 'focus'},
  crosshair: {
    trigger: 'both',
    orientation: 'vertical',
    color: '#888',
  },
};

/** Options for the latency breakdown pie chart. */
export const LATENCY_BREAKDOWN_OPTIONS: google.visualization.PieChartOptions = {
  title: 'Latency Breakdown (ms)',
  height: 300,
  pieHole: 0.4,
  chartArea: {width: '90%', height: '80%'},
  legend: {position: 'right'},
  colors: ['#1a73e8', '#f9ab00', '#34a853', '#7b1fa2', '#5f6368'],
};

/** Common chart options for line and scatter charts. */
export const COMMON_CHART_OPTIONS: google.visualization.LineChartOptions = {
  ...BASE_CANVAS_OPTIONS,
  legend: {position: 'top', maxLines: 3},
  chartArea: {
    left: 80,
    bottom: 50,
    right: 30,
    top: 50,
    width: '85%',
    height: '75%',
  },
  hAxis: {
    title: 'Time (s)',
    textPosition: 'out',
  },
  vAxis: {
    format: 'short',
    title: 'Value',
    textPosition: 'out',
    viewWindow: {min: 0},
  },
  interpolateNulls: true,
};

/**
 * Factory class to initialize all the boilerplate chart configurations.
 * Helps keep the main component class clean and focused on UI logic.
 */
export class GrpcChartConfigFactory {
  readonly baseCanvasOptions = BASE_CANVAS_OPTIONS;
  readonly chartOptions = COMMON_CHART_OPTIONS;

  readonly stepLatencyChartDataInfo = this.createDefault(
    'Chunk Transfer Latency (ms)',
  );
  readonly tcpRttDataInfo = this.createDefault('Min RTT (us)');
  readonly grpcClientLatencyDataInfo = this.createDefault(
    'Client Latency (ms)',
  );
  readonly grpcServerLatencyDataInfo = this.createDefault(
    'Server Latency (ms)',
  );
  readonly tcpPacketRetxDataInfo = this.createDefault('Retx Rate (Pkts/s)');
  readonly tcpDataSentDataInfo = this.createDefault(
    'Sent Rate (Bits/s)',
    'short',
  );
  readonly tcpPacingRateDataInfo = this.createDefault(
    'Pacing Rate (Bits/s)',
    'short',
  );
  readonly tcpDeliveryRateDataInfo = this.createDefault(
    'Delivery Rate (Bits/s)',
    'short',
  );
  readonly tcpSsthreshDataInfo = this.createDefault('Ssthresh (Packets)');
  readonly tcpAppLimitedDataInfo = this.createDefault('App Limited (%)');
  readonly tcpByteOffsetDataInfo = this.createDefault(
    'Bit Offset (Bits)',
    'short',
  );
  readonly tcpDataNotsentDataInfo = this.createDefault(
    'Data Not Sent (Bits)',
    'short',
  );
  readonly tcpPktSpurRetxDataInfo = this.createDefault('Spurious Retx (Count)');
  readonly tcpPktDeliveredCeDataInfo = this.createDefault('CE Rate (Pkts/s)');
  readonly tcpReorderingDataInfo = this.createDefault(
    'Reordering Rate (Pkts/s)',
  );
  readonly tcpRecurRetransDataInfo = this.createDefault(
    'Recurring Retrans Rate (Pkts/s)',
  );
  readonly tcpBusyUsecDataInfo = this.createDefault('Busy Time (us)');
  readonly tcpRwndLimitedUsecDataInfo = this.createDefault('RWND Limited (us)');
  readonly tcpSndbufLimitedUsecDataInfo = this.createDefault(
    'SndBuf Limited (us)',
  );
  readonly tcpCwndDataInfo = this.createDefault('CWND (Packets)');
  readonly tcpPacketSentDataInfo = this.createDefault('Packet Sent (Pkts/s)');
  readonly tcpPacketDeliveredDataInfo = this.createDefault(
    'Delivered Rate (Pkts/s)',
  );

  private createDefault(
    vAxisTitle: string,
    vAxisFormat = 'short',
  ): ChartDataInfo {
    const options: ChartOptions = {
      ...COMMON_CHART_OPTIONS,
      vAxis: {
        ...COMMON_CHART_OPTIONS.vAxis,
        title: vAxisTitle,
        format: vAxisFormat,
      },
    };
    return {
      data: [],
      dataProvider: new DefaultDataProvider(),
      options,
    };
  }

}
