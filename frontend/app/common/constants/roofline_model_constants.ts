/**
 * color palette for roofline model pie chart
 */
export const PIE_CHART_PALETTE = [
  '#3366CC', '#FF9900', '#109618', '#990099', '#3B3EAC', '#0099C6', '#DD4477',
  '#66AA00', '#B82E2E', '#316395', '#994499', '#22AA99', '#AAAA11', '#6633CC',
  '#E67300', '#8B0707', '#329262', '#5574A6', '#3B3EAC',
];

/** axis boundary for roofline model scatter chart */
export const SCATTER_CHART_AXIS = {
  minX: 0.00001,
  maxX: 100000,
  minY: 0.00001,
  maxY: 10000000,
};

/** scatter base options for roofline chart */
export const SCATTER_CHART_OPTIONS = {
  title: 'Roofline Model',
  width: 720,
  height: 400,
  hAxis: {
    title: 'FLOP/Byte (log scale)',
    scaleType: 'log',
    viewWindow: {
      min: SCATTER_CHART_AXIS.minX,
      max: SCATTER_CHART_AXIS.maxX,
    },
    // Ticks have to be explicitly defined for scaling axis evenly.
    ticks: [
      0,
      0.00001,
      0.0001,
      0.001,
      0.01,
      0.1,
      1,
      10,
      100,
      1000,
      10000,
      100000,
    ],
  },
  vAxis: {
    title: 'GFLOP/s (log scale)',
    scaleType: 'log',
    viewWindow: {
      min: SCATTER_CHART_AXIS.minY,
      max: SCATTER_CHART_AXIS.maxY,
    },
    // Ticks have to be explicitly defined for scaling axis evenly.
    ticks: [0, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000, 1000000],
  },
  legend: {position: 'right' as google.visualization.ChartLegendPosition},
  tooltip: {isHtml: true},
  // Be mindful that series is not specified here, otherwise the shallow copy in
  // components of this object could result in overwritten series styles
};

/** roofline plot line styles */
export const ROOFLINE_STYLES = {
  read: {
    lineWidth: 1,
    color: 'red',
    pointsVisible: false,
  },
  write: {
    lineWidth: 1,
    color: 'green',
    pointsVisible: false,
  },
  hbm: {
    lineWidth: 1,
    color: 'black',
    pointsVisible: false,
  },
};

/** Roofline names. */
export const ROOFLINE_NAMES = {
  HBM: 'HBM',
  SHARED_MEM_L1: 'Shared Mem / L1',
  VMEM_READ: 'VMEM Read',
  VMEM_WRITE: 'VMEM Write',
  CMEM_READ: 'CMEM Read',
  CMEM_WRITE: 'CMEM Write',
};

/** Roofline series names. */
export const ROOFLINE_SERIES_NAMES = {
  HBM: 'HBM Roofline',
  SHARED_MEM_L1: 'Shared Mem / L1 Roofline',
  VMEM_READ: 'VMEM Read Roofline',
  VMEM_WRITE: 'VMEM Write Roofline',
  CMEM_READ: 'CMEM Read Roofline',
  CMEM_WRITE: 'CMEM Write Roofline',
};

/** roofline model properties configuration */
export const DEVICE_INFO = [
  {
    id: 'device_type',
    label: 'Device Type',
    type: 'string',
    display: true,
  },
  {
    id: 'megacore',
    label: 'Megacore',
    type: 'string',
    context: '',
    display: true,
  },
  {
    id: 'peak_flop_rate',
    label: 'Peak FLOP Rate per TensorCore',
    type: 'number',
    unit: 'GFLOP/s',
    display: true,
  },
  {
    id: 'peak_hbm_bw',
    label: 'Peak HBM Bandwidth per TensorCore',
    type: 'number',
    unit: 'GiB/s',
    context: '',
    display: true,
  },
  {
    id: 'peak_vmem_read_bw',
    label: 'Peak VMEM Read Bandwidth per TensorCore',
    type: 'number',
    unit: 'GiB/s',
    display: true,
  },
  {
    id: 'peak_vmem_write_bw',
    label: 'Peak VMEM Write Bandwidth per TensorCore',
    type: 'number',
    unit: 'GiB/s',
    display: true,
  },
  {
    id: 'peak_cmem_read_bw',
    label: 'Peak CMEM Read Bandwidth per TensorCore',
    type: 'number',
    unit: 'GiB/s',
    display: true,
  },
  {
    id: 'peak_cmem_write_bw',
    label: 'Peak CMEM Write Bandwidth per TensorCore',
    type: 'number',
    unit: 'GiB/s',
    display: true,
  },
  {
    id: 'cmem_write_ridge_point',
    label: 'CMEM Write Ridge Point',
    type: 'number',
    unit: 'Flop/byte',
    display: false,
  },
  {
    id: 'cmem_read_ridge_point',
    label: 'CMEM Read Ridge Point',
    type: 'number',
    unit: 'Flop/byte',
    display: false,
  },
  {
    id: 'vmem_write_ridge_point',
    label: 'VMEM Write Ridge Point',
    type: 'number',
    unit: 'Flop/byte',
    display: false,
  },
  {
    id: 'vmem_read_ridge_point',
    label: 'VMEM Read Ridge Point',
    type: 'number',
    unit: 'Flop/byte',
    display: false,
  },
  {
    id: 'hbm_ridge_point',
    label: 'HBM Ridge Point',
    type: 'number',
    unit: 'Flop/byte',
    display: false,
  },
  {
    id: 'time_scale_multiplier',
    label: 'Time Scale Multiplier',
    type: 'number',
    unit: '',
    display: true,
  },
];

/**
 * numeric data display formatting config
 *  might be feasible to be passed through api response data
 */
export const NUMERIC_DATA_FORMAT:
    {[key: string]: {type: string; digit?: number};} = {
      'total_time_per_core': {
        type: 'decimal',
        digit: 0,
      },
      'total_time': {
        type: 'decimal',
        digit: 2,
      },
      'avg_time': {
        type: 'decimal',
        digit: 2,
      },
      'total_self_time': {
        type: 'decimal',
        digit: 2,
      },
      'avg_self_time': {
        type: 'decimal',
        digit: 2,
      },
      'measured_flop_rate': {
        type: 'decimal',
        digit: 2,
      },
      'measured_memory_bw': {
        type: 'decimal',
        digit: 2,
      },
      'hbm_bw': {
        type: 'decimal',
        digit: 2,
      },
      'cmem_read_bw': {
        type: 'decimal',
        digit: 2,
      },
      'cmem_write_bw': {
        type: 'decimal',
        digit: 2,
      },
      'operational_intensity': {
        type: 'decimal',
        digit: 2,
      },
      'total_self_time_percent': {
        type: 'percent',
        digit: 1,
      },
      'cumulative_total_self_time_percent': {
        type: 'percent',
        digit: 1,
      },
      'dma_stall_percent': {
        type: 'percent',
        digit: 1,
      },
      'roofline_efficiency': {
        type: 'percent',
        digit: 1,
      },
      'compute_efficiency': {
        type: 'percent',
        digit: 1,
      },
      'max_mem_bw_utilization': {
        type: 'percent',
        digit: 1,
      },
    };
