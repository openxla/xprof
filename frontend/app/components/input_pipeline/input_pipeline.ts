import {Component, inject, OnDestroy} from '@angular/core';
import {ActivatedRoute, Params} from '@angular/router';
import {Store} from '@ngrx/store';
import {STACK_CHART_FILL_COLORS} from 'org_xprof/frontend/app/common/constants/constants';
import {InputPipelineDataTable, SimpleDataTable, } from 'org_xprof/frontend/app/common/interfaces/data_table';
import {setLoadingState} from 'org_xprof/frontend/app/common/utils/utils';
import {DATA_SERVICE_INTERFACE_TOKEN, type DataServiceV2Interface} from 'org_xprof/frontend/app/services/data_service_v2/data_service_v2_interface';
import {setCurrentToolStateAction} from 'org_xprof/frontend/app/store/actions';
import {ReplaySubject} from 'rxjs';
import {takeUntil} from 'rxjs/operators';

import {InputPipelineCommon} from './input_pipeline_common';

const COLUMN_ID_MAX_INFEED_CORE = 'index';
const STEPTIME_COLUMN_IDS_FOR_TPU_INTERNAL = [
  'stepnum',
  'tcComputeTimeMs',
  'scv0ComputeTimeMs',
  'bcInfeedTimeMs',
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

  constructor(route: ActivatedRoute, private readonly store: Store<{}>) {
    super();
    route.params.pipe(takeUntil(this.destroyed)).subscribe((params) => {
      this.processQuery(params);
      this.update();
    });
    this.store.dispatch(setCurrentToolStateAction({currentTool: this.tool}));
  }

  processQuery(params: Params) {
    this.tool = params['tag'] || this.tool;
    this.sessionId = params['run'] || params['sessionId'] || this.sessionId;
    this.host = params['host'] || this.host;
  }

  update() {
    setLoadingState(true, this.store, 'Loading input pipeline data');

    this.dataService.getData(this.sessionId, this.tool, this.host)
        .pipe(takeUntil(this.destroyed))
        .subscribe((data) => {
          setLoadingState(false, this.store);
          data = (data || []) as InputPipelineDataTable[];
          this.parseCommonInputData(data);
          this.isTpu = !!this.deviceAnalysis && !!this.deviceAnalysis.p &&
              this.deviceAnalysis.p['hardware_type'] === 'TPU';
          this.maxInfeedCoreTable = this.findAnalysisData(
                                        data,
                                        COLUMN_ID_MAX_INFEED_CORE,
                                        ) as SimpleDataTable |
              null;
          if (this.isTpu) {
            this.parseHostOpTables(data);
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
