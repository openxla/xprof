import {Component, inject, OnDestroy} from '@angular/core';
import {ActivatedRoute, Params} from '@angular/router';
import {Store} from '@ngrx/store';
import {Throbber} from 'org_xprof/frontend/app/common/classes/throbber';
import {OpProfileProto} from 'org_xprof/frontend/app/common/interfaces/data_table';
import {setLoadingState} from 'org_xprof/frontend/app/common/utils/utils';
import {DATA_SERVICE_INTERFACE_TOKEN, DataServiceV2Interface} from 'org_xprof/frontend/app/services/data_service_v2/data_service_v2_interface';
import {setProfilingDeviceTypeAction} from 'org_xprof/frontend/app/store/actions';
import {combineLatest, Observable, of, ReplaySubject} from 'rxjs';
import {combineLatestWith, map, takeUntil} from 'rxjs/operators';

const GROUP_BY_RULES = ['program', 'category', 'provenance'];

/** An op profile component. */
@Component({
  standalone: false,
  selector: 'op-profile',
  templateUrl: './op_profile.ng.html',
  styleUrls: ['./op_profile_common.scss']
})
export class OpProfile implements OnDestroy {
  private tool = 'hlo_op_profile';
  /** Handles on-destroy Subject, used to unsubscribe. */
  private readonly destroyed = new ReplaySubject<void>(1);
  private readonly throbber = new Throbber(this.tool);
  private readonly dataService: DataServiceV2Interface =
      inject(DATA_SERVICE_INTERFACE_TOKEN);
  private readonly opProfileDataCache = new Map<string, OpProfileProto>();

  sessionId = '';
  host = '';
  moduleList: string[] = [];
  opProfileData: OpProfileProto|null = null;
  groupBy = GROUP_BY_RULES[0]; // Default value

  constructor(
      route: ActivatedRoute,
      private readonly store: Store<{}>,
  ) {
    combineLatest([route.params, route.queryParams])
        .pipe(takeUntil(this.destroyed))
        .subscribe(([params, queryParams]) => {
          this.sessionId = params['sessionId'] || this.sessionId;
          this.processQueryParams(queryParams);
          this.update();
        });
  }

  processQueryParams(params: Params) {
    this.sessionId = params['run'] || params['sessionId'] || this.sessionId;
    this.tool = params['tag'] || this.tool;
    this.host = params['host'] || this.host;
  }

  private fetchData(groupBy: string): Observable<OpProfileProto|null> {
    const cachedData = this.opProfileDataCache.get(groupBy);
    if (cachedData) {
      return of(cachedData);
    }

    setLoadingState(true, this.store, 'Loading op profile data');
    this.throbber.start();

    const params = new Map<string, string>();
    params.set('group_by', groupBy);
    return this.dataService
        .getData(this.sessionId, this.tool, this.host, params)
        .pipe(
            map((data) => {
              this.throbber.stop();
              setLoadingState(false, this.store);
              if (data) {
                const opProfileData = data as OpProfileProto;
                this.opProfileDataCache.set(groupBy, opProfileData);
                return opProfileData;
              }
              return null;
            }),
        );
  }

  update() {
    if (!this.sessionId || !this.tool) {
      return;
    }
    const $data = this.fetchData(this.groupBy);
    const $moduleList = this.dataService.getModuleList(
        this.sessionId,
    );
    $data.pipe(combineLatestWith($moduleList), takeUntil(this.destroyed))
        .subscribe(([data, moduleList]) => {
          if (data) {
            this.opProfileData = data;
            this.store.dispatch(
                setProfilingDeviceTypeAction({
                  deviceType: this.opProfileData.deviceType,
                }),
            );
          }
          if (moduleList) {
            this.moduleList = moduleList.split(',');
          }
        });
  }

  updateTable() {
    this.fetchData(this.groupBy)
        .pipe(takeUntil(this.destroyed))
        .subscribe((data) => {
          if (data) {
            this.opProfileData = data;
          }
        });
  }

  onGroupByChange(newGroupBy: string) {
    this.groupBy = newGroupBy;
    this.updateTable();
  }

  ngOnDestroy() {
    // Unsubscribes all pending subscriptions.
    setLoadingState(false, this.store);
    this.destroyed.next();
    this.destroyed.complete();
  }
}
