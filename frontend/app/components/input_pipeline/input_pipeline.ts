import {Component, inject, OnDestroy} from '@angular/core';
import {ActivatedRoute, Params} from '@angular/router';
import {Store} from '@ngrx/store';
import {Throbber} from 'org_xprof/frontend/app/common/classes/throbber';
import {STACK_CHART_FILL_COLORS} from 'org_xprof/frontend/app/common/constants/constants';
import {InputPipelineDataTable, SimpleDataTable, } from 'org_xprof/frontend/app/common/interfaces/data_table';
import {setLoadingState} from 'org_xprof/frontend/app/common/utils/utils';
import {DATA_SERVICE_INTERFACE_TOKEN, type DataServiceV2Interface} from 'org_xprof/frontend/app/services/data_service_v2/data_service_v2_interface';
import {setCurrentToolStateAction} from 'org_xprof/frontend/app/store/actions';
import {combineLatest, ReplaySubject} from 'rxjs';
import {takeUntil} from 'rxjs/operators';

import {InputPipelineCommon} from './input_pipeline_common';

const COLUMN_ID_MAX_INFEED_CORE = 'index';
// TODO(muditgokhale) : Pass this as a property in Input Pipeline Data Table
// once the DCU is complete.
const STEPTIME_COLUMN_IDS_FOR_TPU_INTERNAL = [
  'stepnum',
  'tcComputeTimeMs',
  'scv0ComputeTimeMs',
  'scv0InfeedTimeMs,',
  'tcInfeedTimeMs',
  'tcOutfeedTimeMs',
  'tcIdleTimeMs',
  'tooltip',
];

/** An input pipeline component. */
@Component({
  standalone: false,
  selector: 'input-pipeline',
  templateUrl: './input_pipeline.ng.html',
  styleUrls: ['./input_pipeline.css']
})
export class InputPipeline extends InputPipelineCommon implements OnDestroy {
  private readonly dataService: DataServiceV2Interface =
      inject(DATA_SERVICE_INTERFACE_TOKEN);

  tool = 'input_pipeline';
  sessionId = '';
  host = '';
  readonly columnIds = STEPTIME_COLUMN_IDS_FOR_TPU_INTERNAL;
  readonly columnColors = STACK_CHART_FILL_COLORS;
  /** Handles on-destroy Subject, used to unsubscribe. */
  private readonly destroyed = new ReplaySubject<void>(1);
  private readonly throbber = new Throbber(this.tool);

  constructor(route: ActivatedRoute, private readonly store: Store<{}>) {
    super();
    combineLatest([route.params, route.queryParams])
        .pipe(takeUntil(this.destroyed))
        .subscribe(([params, queryParams]) => {
          this.sessionId = params['sessionId'] || this.sessionId;
          this.processQueryParams(queryParams);
          this.update();
        });
    this.store.dispatch(setCurrentToolStateAction({currentTool: this.tool}));
  }

  processQueryParams(params: Params) {
    this.tool = params['tag'] || this.tool;
    this.sessionId = params['run'] || params['sessionId'] || this.sessionId;
    this.host = params['host'] || this.host;
  }

  update() {
    setLoadingState(true, this.store, 'Loading input pipeline data');
    this.throbber.start();

    this.dataService.getData(this.sessionId, this.tool, this.host)
        .pipe(takeUntil(this.destroyed))
        .subscribe((data) => {
          this.throbber.stop();
          setLoadingState(false, this.store);
          const inputPipelineData = (data || []) as InputPipelineDataTable[];
          this.parseCommonInputData(inputPipelineData);
          this.isTpu = !!this.deviceAnalysis && !!this.deviceAnalysis.p &&
              this.deviceAnalysis.p['hardware_type'] === 'TPU';
          this.maxInfeedCoreTable = this.findAnalysisData(
                                        inputPipelineData,
                                        COLUMN_ID_MAX_INFEED_CORE,
                                        ) as SimpleDataTable |
              null;
          if (this.isTpu) {
            this.parseHostOpTables(inputPipelineData);
          }
        });
  }

  ngOnDestroy() {
    setLoadingState(false, this.store);
    // Unsubscribes all pending subscriptions.
    this.destroyed.next();
    this.destroyed.complete();
  }
}
