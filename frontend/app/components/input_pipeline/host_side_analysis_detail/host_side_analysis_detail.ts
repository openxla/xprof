import {CommonModule} from '@angular/common';
import {Component, Input, OnChanges, OnInit, SimpleChanges} from '@angular/core';
import {MatExpansionModule} from '@angular/material/expansion';
import {ChartDataInfo} from 'org_xprof/frontend/app/common/interfaces/chart';
import {DEFAULT_SIMPLE_DATA_TABLE, type InputPipelineHostAnalysis, type SimpleDataTable} from 'org_xprof/frontend/app/common/interfaces/data_table';
import {ChartModule} from 'org_xprof/frontend/app/components/chart/chart';
import {TABLE_OPTIONS} from 'org_xprof/frontend/app/components/chart/chart_options';
import {ArrayDataProvider} from 'org_xprof/frontend/app/components/chart/default_data_provider';

import {HostSideAnalysisDetailTableDataProvider} from './host_side_analysis_detail_table_data_provider';

/** A host-side analysis detail view component. */
@Component({
  standalone: true,
  selector: 'host-side-analysis-detail',
  templateUrl: 'host_side_analysis_detail.ng.html',
  styleUrls: ['host_side_analysis_detail.scss'],
  imports: [CommonModule, MatExpansionModule, ChartModule],
})
export class HostSideAnalysisDetail implements OnInit, OnChanges {
  /** The input pipeline host anaysis data. */
  @Input() hostAnalysis: InputPipelineHostAnalysis|null = null;

  /** The recommendation data. */
  @Input()
  set recommendation(data: SimpleDataTable|null) {
    data = data || DEFAULT_SIMPLE_DATA_TABLE;
    data.rows = data.rows || [];
    data.rows.forEach(row => {
      if (row.c && row.c[0] && row.c[0].v) {
        this.recommendations.push(String(row.c[0].v));
      }
    });
  }

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

  ngOnInit() {
    this.dataProviderForTable.setHasHostOpsChangedEventListener(
        (hasHostOps: boolean) => {
          Promise.resolve().then(() => {
            this.hasHostOps = hasHostOps;
          });
        });
  }

  ngOnChanges(changes: SimpleChanges) {
    this.parseColumnChartData(this.hostAnalysis);
    this.dataInfoForTable = {
      ...this.dataInfoForTable,
      data: this.hostAnalysis,
    };
  }

  parseColumnChartData(hostAnalysis: InputPipelineHostAnalysis|null) {
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
