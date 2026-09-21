import {
  ChangeDetectionStrategy,
  Component,
  OnInit,
  effect,
  input,
} from '@angular/core';
import {ChartDataInfo} from 'org_xprof/frontend/app/common/interfaces/chart';
import {
  DEFAULT_SIMPLE_DATA_TABLE,
  type InputPipelineHostAnalysis,
  type SimpleDataTable,
} from 'org_xprof/frontend/app/common/interfaces/data_table';
import {TABLE_OPTIONS} from 'org_xprof/frontend/app/components/chart/chart_options';
import {ArrayDataProvider} from 'org_xprof/frontend/app/components/chart/default_data_provider';

import {
  MatExpansionPanel,
  MatExpansionPanelHeader,
  MatExpansionPanelTitle,
} from '@angular/material/expansion';
import {Chart} from '../../chart/chart';
import {HostSideAnalysisDetailTableDataProvider} from './host_side_analysis_detail_table_data_provider';

/** A host-side analysis detail view component. */
@Component({
  changeDetection: ChangeDetectionStrategy.OnPush,
  selector: 'host-side-analysis-detail',
  templateUrl: './host_side_analysis_detail.ng.html',
  styleUrls: ['./host_side_analysis_detail.scss'],
  imports: [
    Chart,
    MatExpansionPanel,
    MatExpansionPanelHeader,
    MatExpansionPanelTitle,
  ],
})
export class HostSideAnalysisDetail implements OnInit {
  /** The input pipeline host anaysis data. */
  readonly hostAnalysis = input<InputPipelineHostAnalysis | null>(null);

  /** The recommendation data. */
  readonly recommendation = input<SimpleDataTable | null>(null);

  hasHostOps = false;
  recommendations: string[] = [];
  dataInfoForColumnChart: ChartDataInfo = {
    data: null,
    dataProvider: new ArrayDataProvider(),
    options: {
      bar: {groupWidth: '45%'},
      chartArea: {
        left: 40,
        width: '50%',
        height: '90%',
      },
      width: 800,
      height: 300,
      colors: ['red', 'blue', 'orange', 'green', 'purple'],
      backgroundColor: {'fill': 'transparent'},
      isStacked: 'percent',
    },
  };
  dataProviderForTable = new HostSideAnalysisDetailTableDataProvider();
  dataInfoForTable: ChartDataInfo = {
    data: null,
    dataProvider: this.dataProviderForTable,
    options: {
      ...TABLE_OPTIONS,
      alternatingRowStyle: false,
    },
  };

  constructor() {
    effect(() => {
      const rec = this.recommendation() || DEFAULT_SIMPLE_DATA_TABLE;
      const recs: string[] = [];
      const rows = rec.rows || [];
      rows.forEach((row) => {
        if (row.c && row.c[0] && row.c[0].v) {
          recs.push(String(row.c[0].v));
        }
      });
      this.recommendations = recs;
    });

    effect(() => {
      const hostAnalysis = this.hostAnalysis();
      this.parseColumnChartData(hostAnalysis);
      this.dataInfoForTable = {
        ...this.dataInfoForTable,
        data: hostAnalysis,
      };
    });
  }

  ngOnInit() {
    this.dataProviderForTable.setHasHostOpsChangedEventListener(
      (hasHostOps: boolean) => {
        Promise.resolve().then(() => {
          this.hasHostOps = hasHostOps;
        });
      },
    );
  }

  parseColumnChartData(hostAnalysis: InputPipelineHostAnalysis | null) {
    if (!hostAnalysis) {
      this.hasHostOps = false;
      return;
    }
    this.hasHostOps = true;

    const kUsPerMs = 1000.0;
    const p = hostAnalysis.p || {};
    const unclassifiedNonEnqueueMs =
      Number(p['unclassified_nonequeue_us']) / kUsPerMs;
    const demandedFileReadMs = Number(p['demanded_file_read_us']) / kUsPerMs;
    const advancedFileReadMs = Number(p['advanced_file_read_us']) / kUsPerMs;
    const preprocessingMs = Number(p['preprocessing_us']) / kUsPerMs;
    const enqueueMs = Number(p['enqueue_us']) / kUsPerMs;

    const data = [
      [
        'Input time breakdown',
        'Other data reading or processing (in ms)',
        'Reading data from files on demand (in ms)',
        'Reading data from files in advance [including caching, prefetching, interleaving] (in ms)',
        'Data preprocessing (in ms)',
        'Enqueuing data to be transferred to device (in ms)',
      ],
      [
        '',
        unclassifiedNonEnqueueMs,
        demandedFileReadMs,
        advancedFileReadMs,
        preprocessingMs,
        enqueueMs,
      ],
    ];

    this.dataInfoForColumnChart = {
      ...this.dataInfoForColumnChart,
      data,
    };
  }
}
