import {
  DEFAULT_SIMPLE_DATA_TABLE,
  HostOpTable,
  InputPipelineDataTable,
  InputPipelineDeviceAnalysis,
  InputPipelineHostAnalysis,
  MetaHostOpTable,
  SimpleDataTable,
} from 'org_xprof/frontend/app/common/interfaces/data_table';
import {Diagnostics} from 'org_xprof/frontend/app/common/interfaces/diagnostics';
import {parseDiagnosticsDataTable} from 'org_xprof/frontend/app/common/utils/utils';

const COLUMN_ID_DEVICE_ANALYSIS = 'stepnum';
const COLUMN_ID_HOST_ANALYSIS = 'opName';
const COLUMN_ID_RECOMMENDATION = 'link';
const COLUMN_ID_DIAGNOSTICS = 'severity';
const PROPERTIES_DEVICE_ANALYSIS = [
  'infeed_percent_average',
  'infeed_percent_maximum',
  'infeed_percent_minimum',
  'infeed_percent_standard_deviation',
  'steptime_ms_average',
  'steptime_ms_maximum',
  'steptime_ms_minimum',
  'steptime_ms_standard_deviation',
  'input_conclusion',
  'summary_nextstep',
];
const PROPERTIES_HOST_ANALYSIS = [
  'advanced_file_read_us',
  'demanded_file_read_us',
  'enqueue_us',
  'preprocessing_us',
  'unclassified_nonequeue_us',
];

/** A common class for the input-pipeline component. */
export class InputPipelineCommon {
  deviceAnalysis: InputPipelineDeviceAnalysis | null = null;
  hostAnalysis: InputPipelineHostAnalysis | null = null;
  recommendation: SimpleDataTable | null = null;
  hasDiviceAanlysisRows = true;
  diagnostics: Diagnostics = {info: [], warnings: [], errors: []};
  hasHostOpTables = false;
  metaHostOpTable: MetaHostOpTable | null = null;
  hostOpTables: HostOpTable[] = [];
  maxInfeedCoreTable: SimpleDataTable | null = null;
  isTpu = false;

  parseHostOpTables(data: InputPipelineDataTable[]) {
    if (!data) return;
    for (let i = 0; i < data.length; i++) {
      const analysis = data[i];
      if (!analysis || !analysis.p) continue;
      const foundMetaHostOpTable = Object.keys(analysis.p).find(
        (property) => property === 'num_host_op_tables',
      );
      if (!foundMetaHostOpTable) continue;
      this.metaHostOpTable = analysis as MetaHostOpTable;
      const numHostOpTables = Number(
        this.metaHostOpTable.p?.num_host_op_tables || 0,
      );
      if (numHostOpTables <= 0) continue;
      this.hasHostOpTables = true;
      const hostOpTables: HostOpTable[] = [];
      for (let k = 0; k < numHostOpTables; k++) {
        hostOpTables.push(data[i + 1 + k] as HostOpTable);
      }
      this.hostOpTables = hostOpTables;
      return;
    }
  }

  findAnalysisData(
    data: InputPipelineDataTable[],
    columnId: string,
    properties: string[] = [],
  ):
    | InputPipelineDeviceAnalysis
    | InputPipelineHostAnalysis
    | SimpleDataTable
    | null {
    if (!data) {
      return DEFAULT_SIMPLE_DATA_TABLE;
    }
    for (let i = 0; i < data.length; i++) {
      const analysis = data[i];
      if (!analysis) {
        continue;
      }
      if (analysis.cols) {
        const foundCols = analysis.cols.find(
          (column) => column.id === columnId,
        );
        if (!!foundCols) {
          return analysis;
        }
      }
      if (analysis['p']) {
        const foundProperties = Object.keys(analysis['p']).find((property) =>
          properties.includes(property),
        );
        if (!!foundProperties) {
          return analysis;
        }
      }
    }
    return DEFAULT_SIMPLE_DATA_TABLE;
  }

  updateHasDeviceAanlysisRows() {
    const analysis = this.deviceAnalysis || {};
    analysis.p = analysis.p || {};
    this.hasDiviceAanlysisRows = !!analysis.rows && analysis.rows.length > 0;
  }

  parseCommonInputData(data: InputPipelineDataTable[]) {
    this.deviceAnalysis = this.findAnalysisData(
      data,
      COLUMN_ID_DEVICE_ANALYSIS,
      PROPERTIES_DEVICE_ANALYSIS,
    ) as InputPipelineDeviceAnalysis | null;
    this.hostAnalysis = this.findAnalysisData(
      data,
      COLUMN_ID_HOST_ANALYSIS,
      PROPERTIES_HOST_ANALYSIS,
    ) as InputPipelineHostAnalysis;
    this.recommendation = this.findAnalysisData(
      data,
      COLUMN_ID_RECOMMENDATION,
    ) as SimpleDataTable | null;
    this.diagnostics = parseDiagnosticsDataTable(
      this.findAnalysisData(data, COLUMN_ID_DIAGNOSTICS),
    );
    this.updateHasDeviceAanlysisRows();
  }
}
