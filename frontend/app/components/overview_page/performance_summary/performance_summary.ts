import {Component, Input, OnChanges, OnInit, SimpleChanges} from '@angular/core';
import {type GeneralAnalysis, type InputPipelineAnalysis} from 'org_xprof/frontend/app/common/interfaces/data_table';
import {GeneralProps, SummaryInfo, SummaryInfoConfig} from 'org_xprof/frontend/app/common/interfaces/summary_info';

/**
 * Configuration Assumptions:
 * 1. by default, values/sdv values/nested children values all reads from
 * a k-v pair property with field key specified in the config object.
 * 2. It's required that each record in the same config object (eg.
 * NON_TPU_SUMMARY_INFO) should consume the same property (eg.
 * inputPipelineAnalysisProperty).
 * 3. The summary info data is in nested structure, currently consumes 2 levels
 * of metrics (Top layer level 1 + its children layer)
 */

/**
 * Configuration Instructions
 * 1. In this file we chain multiple config objects to display metrics in order:
 * GENERIC_SUMMARY_INFO_BEFORE + TPU_SUMMARY_INFO/NON_TPU_SUMMARY_INFO +
 * GENERIC_SUMMARY_INFO_AFTER
 * 2. Values read from the k-v pair property could be undefined for oss
 * (dependent on backend version). Be sure to add a fallback logic (eg. || '')
 * when reading it in the parseDataFromConfig function call.
 * 3. Values are by default read from k-v pairs, we can also use a custom
 * callback function (eg. getChildValues) to read data into the propertyValues
 * as a string list.
 * 4. If `customInput` is specified when calling `parseDataFromConfig`, it will
 * be used as input to callback function `getValue` or `getChildValues`.
 */

/** Generic summary info, display on top of the list */
const GENERIC_SUMMARY_INFO_BEFORE: SummaryInfoConfig[] = [
  {
    title: 'Average Step Time',
    goodMetric: false,
    valueKey: 'steptime_ms_average',
    sdvKey: 'steptime_ms_standard_deviation',
    unit: 'ms',
    valueColor: 'red',
    trainingOnly: true,
    childrenInfoConfig: [
      {title: 'Idle', valueKey: 'idle_ms_average', unit: 'ms'},
      {title: 'Input', valueKey: 'input_ms_average', unit: 'ms'},
      {title: 'Compute', valueKey: 'compute_ms_average', unit: 'ms'},
    ],
  },
];


/** Generic summary info, display on bottom of the list */
const GENERIC_SUMMARY_INFO_AFTER: SummaryInfoConfig[] = [
  {
    title: 'Framework Op Placement',
    tooltip: 'Ratio of Framework Ops executed on the host and device.',
    description: 'generally desired to have more ops on device',
    childrenInfoConfig: [
      {title: 'Host', valueKey: 'host_tf_op_percent'},
      {title: 'Device', valueKey: 'device_tf_op_percent'},
    ]
  },
  {
    title: 'Op Time Spent on Eager Execution',
    tooltip:
        'Out of the total op execution time on host (device), excluding idle time, the percentage of which used eager execution.',
    goodMetric: false,
    childrenInfoConfig: [
      {title: 'Host', valueKey: 'host_op_time_eager_percent'},
      {title: 'Device', valueKey: 'device_op_time_eager_percent'},
    ],
  },
  {
    title: 'Device Compute Precisions',
    description: 'out of Total Device Time',
    childrenInfoConfig: [
      {title: '16-bit', valueKey: 'device_compute_16bit_percent'},
      {title: '32-bit', valueKey: 'device_compute_32bit_percent'},
    ],
  },
];

const NON_TPU_SUMMARY_INFO: SummaryInfoConfig[] = [
  {
    title: 'All Others Time',
    unit: 'ms',
    valueKey: 'other_time_ms_avg',
    sdvKey: 'other_time_ms_sdv',
  },
  {
    title: 'Compilation Time',
    unit: 'ms',
    valueKey: 'compile_time_ms_avg',
    sdvKey: 'compile_time_ms_sdv',
  },
  {
    title: 'Output Time',
    unit: 'ms',
    valueKey: 'outfeed_time_ms_avg',
    sdvKey: 'outfeed_time_ms_sdv',
  },
  {
    title: 'Input Time',
    unit: 'ms',
    valueKey: 'infeed_time_ms_avg',
    sdvKey: 'infeed_time_ms_sdv',
  },
  {
    title: 'Kernel Launch Time',
    unit: 'ms',
    valueKey: 'kernel_launch_time_ms_avg',
    sdvKey: 'kernel_launch_time_ms_sdv',
  },
  {
    title: 'Host Compute Time',
    unit: 'ms',
    valueKey: 'host_compute_time_ms_avg',
    sdvKey: 'host_compute_time_ms_sdv',
  },
  {
    title: 'Device Collective Communication Time',
    unit: 'ms',
    valueKey: 'device_collectives_time_ms_avg',
    sdvKey: 'device_collectives_time_ms_sdv',
  },
  {
    title: 'Device to Device Time',
    unit: 'ms',
    valueKey: 'device_to_device_time_ms_avg',
    sdvKey: 'device_to_device_time_ms_sdv',
  },
  {
    title: 'Device Compute Time',
    unit: 'ms',
    valueKey: 'device_compute_time_ms_avg',
    sdvKey: 'device_compute_time_ms_sdv',
  },
];

const TPU_SUMMARY_INFO: SummaryInfoConfig[] = [
  {
    title: 'FLOPS Utilization',
    goodMetric: true,
    tooltip:
        'Why two numbers: The first number shows the hardware utilization based on the hardware performance counter. The second one shows the performance compared to the program\'s optimal performance considering the instruction mix (i.e., the ratio of floating-point operations and memory operations).',
    childrenInfoConfig: [
      {
        title: 'Utilization of TPU Matrix Units',
        valueKey: 'mxu_utilization_percent',
      },
      {
        title: 'Compared to Program\'s Optimal FLOPS',
        valueKey: 'flop_rate_utilization_relative_to_roofline',
        description: `see <a href="/roofline_model/${
            window.location.pathname.split('/')[2]}">roofline_model</a>`,
      },
    ]
  },
  {
    title: 'TPU Duty Cycle',
    tooltip: 'Percentage of the device time that is busy.',
    goodMetric: true,
    valueKey: 'device_duty_cycle_percent',
  },
  {
    title: 'Memory Bandwidth Utilization',
    tooltip: 'Percentage of the peak device memory bandwidth that is used.',
    goodMetric: true,
    valueKey: 'memory_bw_utilization_relative_to_hw_limit',
  },
  {
    title: 'Firmware Power Metrics (Power/Timescale)',
    tooltip:
        'Max power consumption of different components/rails with different timescale.',
    childrenInfoConfig: [
      {
        title: 'Max VDD Core FW PowerMeter PL1',
        valueKey: 'fw_max_vdd_core_pl1_power_watts',
      },
      {
        title: 'Max VDD Core FW PowerMeter PL2',
        valueKey: 'fw_max_vdd_core_pl2_power_watts',
      },
      {
        title: 'Max VDD Core FW PowerMeter PL3',
        valueKey: 'fw_max_vdd_core_pl3_power_watts',
      },
      {
        title: 'Max VDD Core FW PowerMeter PL4',
        valueKey: 'fw_max_vdd_core_pl4_power_watts',
      },
      {
        title: 'Max HBM FW PowerMeter PL1',
        valueKey: 'fw_max_hbm_pl1_power_watts',
      },
      {
        title: 'Max HBM FW PowerMeter PL2',
        valueKey: 'fw_max_hbm_pl2_power_watts',
      },
      {
        title: 'Max HBM FW PowerMeter PL3',
        valueKey: 'fw_max_hbm_pl3_power_watts',
      },
      {
        title: 'Max HBM FW PowerMeter PL4',
        valueKey: 'fw_max_hbm_pl4_power_watts',
      },
    ]
  },
  {
    title: 'SPI Sampler Power Metrics',
    tooltip:
        'Avg/Max power consumption of different components/rails, including max of moving average of window size of 100us/1ms/10ms/1s.',
    description:
        'HBM: HBM SPI Sampler PowerMeter; Core: VDD Core SPI Sampler PowerMeter',
    getChildValues: (props) =>
        (((props as GeneralProps)['power_metrics']))?.split('##') || [],
  },
];

/** A performance summary view component. */
@Component({
  standalone: false,
  selector: 'performance-summary',
  templateUrl: './performance_summary.ng.html',
  styleUrls: ['./performance_summary.scss']
})
export class PerformanceSummary implements OnChanges, OnInit {
  /** Identify if this is an inference or training session */
  @Input() isInference?: boolean;

  /** The general analysis data. */
  @Input() generalAnalysis?: GeneralAnalysis;

  /** The input pipeline analyis data. */
  @Input() inputPipelineAnalysis?: InputPipelineAnalysis;

  /** Inference latency analysis data */
  @Input() inferenceLatencyData?: GeneralAnalysis;

  title = 'Performance Summary';
  summaryInfoCombined: SummaryInfo[] = [];

  get isTpu() {
    return this.inputPipelineProps['hardware_type'] === 'TPU';
  }

  get generalProps() {
    return (this.generalAnalysis || {}).p as GeneralProps || {};
  }

  get inputPipelineProps() {
    return (this.inputPipelineAnalysis || {}).p as GeneralProps || {};
  }

  get inferenceLatencyProps() {
    return (this.inferenceLatencyData || {}).p as GeneralProps || {};
  }

  get remarkText() {
    return this.generalProps['remark_text'] || '';
  }

  get remarkColor() {
    return this.generalProps['remark_color'] || '';
  }

  ngOnInit() {
    this.parseSummaryData();
  }

  ngOnChanges(changes: SimpleChanges) {
    this.parseSummaryData();
  }

  parseSummaryData() {
    this.summaryInfoCombined = [];
    this.parseDataFromConfig(
        GENERIC_SUMMARY_INFO_BEFORE, this.inputPipelineProps,
        this.summaryInfoCombined, 1);
    if (this.isTpu) {
      this.parseDataFromConfig(
          TPU_SUMMARY_INFO, this.generalProps, this.summaryInfoCombined, 1);
    } else {
      this.parseDataFromConfig(
          NON_TPU_SUMMARY_INFO, this.inputPipelineProps,
          this.summaryInfoCombined, 1);
    }
    this.parseDataFromConfig(
        GENERIC_SUMMARY_INFO_AFTER, this.generalProps, this.summaryInfoCombined,
        1);
  }

  readSummaryInfoFromConfig(
      config: SummaryInfoConfig, props: GeneralProps,
      customInput?: google.visualization.DataObjectCell[]): SummaryInfo|null {
    if (config.trainingOnly && this.isInference) return null;
    if (config.inferenceOnly && !this.isInference) return null;

    const descriptions = [];
    // We've seen 'nan' in the sdv value
    if (config.sdvKey &&
        (props[config.sdvKey] && props[config.sdvKey] !== 'nan')) {
      descriptions.push(
          `(σ = ${props[config.sdvKey] || ''} ${config.unit || ''})`);
    }
    if (config.goodMetric !== undefined) {
      descriptions.push(`${config.goodMetric ? 'higher' : 'lower'} is better.`);
    }
    if (config.description) {
      descriptions.push(config.description);
    }
    let value = '';
    let valueStr = '';
    // valueKey and getValue are mutual exclusive
    if (config.valueKey) {
      value = props[config.valueKey];
      valueStr = `${value} ${config.unit || ''}`;
    } else if (config.getValue && customInput) {
      value = config.getValue(customInput);
      valueStr = `${value} ${config.unit || ''}`;
    }
    const propertyValues = config.getChildValues ?
        config.getChildValues(customInput || props) :
        [];
    const childrenInfoCombined: SummaryInfo[] = [];
    if (config.childrenInfoConfig) {
      this.parseDataFromConfig(
          config.childrenInfoConfig, props, childrenInfoCombined);
    }
    if (value || childrenInfoCombined.length > 0 || propertyValues.length > 0) {
      return {
        title: config.title,
        tooltip: config.tooltip || '',
        value: valueStr,
        propertyValues,
        descriptions,
        valueColor: config.valueColor || '',
        childrenInfo: childrenInfoCombined,
      };
    }
    return null;
  }

  // TODO: Remove customInput argument, read all metrics from property
  parseDataFromConfig(
      summaryConfigs: SummaryInfoConfig[]|undefined, props: GeneralProps,
      summaryInfoCombined: SummaryInfo[], level = 0,
      customInput?: google.visualization.DataObjectCell[]) {
    if (!summaryConfigs) return;
    summaryConfigs.forEach((config: SummaryInfoConfig) => {
      const summaryInfo =
          this.readSummaryInfoFromConfig(config, props, customInput);
      if (summaryInfo) {
        summaryInfoCombined.push({...summaryInfo, level});
      }
    });
  }
}
