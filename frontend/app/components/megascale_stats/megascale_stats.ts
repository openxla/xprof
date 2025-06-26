import {Component, inject, OnDestroy} from '@angular/core';
import {ActivatedRoute, Params} from '@angular/router';
import {Store} from '@ngrx/store';
import {ChartDataInfo} from 'org_xprof/frontend/app/common/interfaces/chart';
import {SimpleDataTable} from 'org_xprof/frontend/app/common/interfaces/data_table';
import {Diagnostics} from 'org_xprof/frontend/app/common/interfaces/diagnostics';
import {parseDiagnosticsDataTable} from 'org_xprof/frontend/app/common/utils/utils';
import {TABLE_OPTIONS} from 'org_xprof/frontend/app/components/chart/chart_options';
import {Dashboard} from 'org_xprof/frontend/app/components/chart/dashboard/dashboard';
import {DefaultDataProvider} from 'org_xprof/frontend/app/components/chart/default_data_provider';
import {DATA_SERVICE_INTERFACE_TOKEN, DataServiceV2Interface} from 'org_xprof/frontend/app/services/data_service_v2/data_service_v2_interface';
import {setCurrentToolStateAction, setLoadingStateAction} from 'org_xprof/frontend/app/store/actions';
import {ReplaySubject} from 'rxjs';
import {takeUntil} from 'rxjs/operators';

const MEGASCALE_STATS_INDEX = 0;
const DIAGNOSTICS_INDEX = 1;

/** A Megascale Stats page component. */
@Component({
  standalone: false,
  selector: 'megascale-stats',
  templateUrl: './megascale_stats.ng.html',
  styleUrls: ['./megascale_stats.scss']
})
export class MegascaleStats extends Dashboard implements OnDestroy {
  tool = 'megascale_stats';
  /** Handles on-destroy Subject, used to unsubscribe. */
  private readonly destroyed = new ReplaySubject<void>(1);
  private readonly dataService: DataServiceV2Interface =
      inject(DATA_SERVICE_INTERFACE_TOKEN);

  sessionId = '';
  host = '';

  diagnostics: Diagnostics = {info: [], warnings: [], errors: []};

  dataInfo: ChartDataInfo = {
    data: null,
    dataProvider: new DefaultDataProvider(),
    filters: [],
    options: {
      ...TABLE_OPTIONS,
      showRowNumber: false,
    },
  };

  constructor(route: ActivatedRoute, private readonly store: Store<{}>) {
    super();
    route.params.pipe(takeUntil(this.destroyed)).subscribe((params) => {
      this.processQuery(params);
      this.update();
    });
    this.store.dispatch(setCurrentToolStateAction({currentTool: this.tool}));
  }

  processQuery(params: Params) {
    this.sessionId = params['run'] || params['sessionId'] || this.sessionId;
    this.tool = params['tag'] || this.tool;
    this.host = params['host'] || this.host;
  }

  update() {
    this.store.dispatch(setLoadingStateAction({
      loadingState: {
        loading: true,
        message: 'Loading Megascale Stats data',
      }
    }));

    this.dataService.getData(this.sessionId, this.tool, this.host)
        .pipe(takeUntil(this.destroyed))
        .subscribe((data) => {
          this.store.dispatch(setLoadingStateAction({
            loadingState: {
              loading: false,
              message: '',
            }
          }));

          if (data) {
            const d = data as SimpleDataTable[] | null;
            if (d) {
              if (d.hasOwnProperty(DIAGNOSTICS_INDEX)) {
                this.diagnostics = parseDiagnosticsDataTable(
                    d[DIAGNOSTICS_INDEX],
                );
              }
              if (d.hasOwnProperty(MEGASCALE_STATS_INDEX)) {
                this.parseData(d[MEGASCALE_STATS_INDEX]);
                this.dataInfo = {
                  ...this.dataInfo,
                  data: d[MEGASCALE_STATS_INDEX],
                };
              }
            }
          }
        });
  }

  override updateView() {
    this.dataInfo = {
      ...this.dataInfo,
      filters: this.getFilters(),
    };
  }

  ngOnDestroy() {
    // Unsubscribes all pending subscriptions.
    this.destroyed.next();
    this.destroyed.complete();
  }
}
