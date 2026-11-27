import {
  ChangeDetectionStrategy,
  Component,
  inject,
  OnDestroy,
} from '@angular/core';
import {ActivatedRoute, Params} from '@angular/router';
import {Store} from '@ngrx/store';
import {Throbber} from 'org_xprof/frontend/app/common/classes/throbber';
import {STACK_CHART_FILL_COLORS} from 'org_xprof/frontend/app/common/constants/constants';
import {
  InputPipelineDataTable,
  SimpleDataTable,
} from 'org_xprof/frontend/app/common/interfaces/data_table';
import {
  createSessionIdResolver,
  getSessionIdFromParams,
  resolveHostFromParams,
  setLoadingState,
} from 'org_xprof/frontend/app/common/utils/utils';
import {
  DATA_SERVICE_INTERFACE_TOKEN,
  type DataServiceV2Interface,
} from 'org_xprof/frontend/app/services/data_service_v2/data_service_v2_interface';
import {setCurrentToolStateAction} from 'org_xprof/frontend/app/store/actions';
import {combineLatest, ReplaySubject, Subscription} from 'rxjs';
import {takeUntil} from 'rxjs/operators';

import {InputPipelineCommon} from './input_pipeline_common';

const COLUMN_ID_MAX_INFEED_CORE = 'index';
// TODO(muditgokhale) : Pass this as a property in Input Pipeline Data Table
// once the DCU is complete.
const STEPTIME_COLUMN_IDS_FOR_TPU_INTERNAL = [
  'stepnum',
  'tcComputeTimeMs',
  'scv0ComputeTimeMs',
  'scv0InfeedTimeMs',
  'tcInfeedTimeMs',
  'tcOutfeedTimeMs',
  'tcIdleTimeMs',
  'hostTransferTimeMs',
  'tooltip',
];

/** An input pipeline component. */
@Component({
  changeDetection: ChangeDetectionStrategy.Default,standalone: false,
  selector: 'input-pipeline',
  templateUrl: './input_pipeline.ng.html',
  styleUrls: ['./input_pipeline.scss']
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
  private readonly throbber: Throbber;
  private dataSubscription?: Subscription;

  constructor(route: ActivatedRoute, private readonly store: Store<{}>) {
    super();
    const routeTool = route.routeConfig?.path?.split('/')[0];
    if (
      routeTool === 'input_pipeline_analyzer' ||
      routeTool === 'input_pipeline'
    ) {
      this.tool = routeTool;
    } else if (window.location.pathname.includes('/data/plugin/profile')) {
      this.tool = 'input_pipeline_analyzer';
    }
    this.throbber = new Throbber(this.tool);
    const resolveSessionId = createSessionIdResolver();
    combineLatest([route.params, route.queryParams])
        .pipe(takeUntil(this.destroyed))
        .subscribe(([params, queryParams]) => {
          const oldSessionId = this.sessionId;
          const oldTool = this.tool;
          const oldHost = this.host;

        this.sessionId = resolveSessionId(params, queryParams, this.sessionId);
        this.processQueryParams(queryParams, oldSessionId);
        // Trigger update only if the parameters actually changed.
        const hasChanged =
          this.sessionId !== oldSessionId ||
          this.tool !== oldTool ||
          this.host !== oldHost;
        if (hasChanged) {
          this.update();
        }
      });
    this.store.dispatch(setCurrentToolStateAction({currentTool: this.tool}));
  }

  processQueryParams(params: Params, prevSessionId = this.sessionId) {
    this.tool = params['tag'] || this.tool;
    this.sessionId = getSessionIdFromParams(undefined, params, this.sessionId);
    this.host = resolveHostFromParams(
      params,
      prevSessionId,
      this.sessionId,
      this.host,
    );
  }

  update() {
    this.dataSubscription?.unsubscribe();
    setLoadingState(true, this.store, 'Loading input pipeline data');
    this.throbber.start();

    this.dataSubscription = this.dataService
      .getData(this.sessionId, this.tool, this.host)
      .pipe(takeUntil(this.destroyed))
      .subscribe({
        next: (data) => {
          this.throbber.stop();
          setLoadingState(false, this.store);
          const inputPipelineData = (data || []) as InputPipelineDataTable[];
          this.parseCommonInputData(inputPipelineData);
          this.maxInfeedCoreTable = this.findAnalysisData(
            inputPipelineData,
            COLUMN_ID_MAX_INFEED_CORE,
          ) as SimpleDataTable | null;
        },
        error: () => {
          this.throbber.stop();
          setLoadingState(false, this.store);
          this.parseCommonInputData([]);
          this.maxInfeedCoreTable = null;
        },
      });
  }

  ngOnDestroy() {
    this.dataSubscription?.unsubscribe();
    setLoadingState(false, this.store);
    // Unsubscribes all pending subscriptions.
    this.destroyed.next();
    this.destroyed.complete();
  }
}
