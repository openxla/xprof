import {Component, inject, OnDestroy} from '@angular/core';
import {Store} from '@ngrx/store';
import {Diagnostics} from 'org_xprof/frontend/app/common/interfaces/diagnostics';
import {NavigationEvent} from 'org_xprof/frontend/app/common/interfaces/navigation_event';
import {CommunicationService} from 'org_xprof/frontend/app/services/communication_service/communication_service';
import {DATA_SERVICE_INTERFACE_TOKEN, DataServiceV2Interface} from 'org_xprof/frontend/app/services/data_service_v2/data_service_v2_interface';
import {getErrorMessage, getLoadingState, } from 'org_xprof/frontend/app/store/selectors';
import {LoadingState} from 'org_xprof/frontend/app/store/state';
import {ReplaySubject} from 'rxjs';
import {takeUntil} from 'rxjs/operators';

/** A main page component. */
@Component({
  standalone: false,
  selector: 'main-page',
  templateUrl: './main_page.ng.html',
  styleUrls: ['./main_page.scss']
})
export class MainPage implements OnDestroy {
  /** Handles on-destroy Subject, used to unsubscribe. */
  private readonly destroyed = new ReplaySubject<void>(1);
  private readonly dataService: DataServiceV2Interface =
      inject(DATA_SERVICE_INTERFACE_TOKEN);

  loading = true;
  loadingMessage = '';
  isSideNavOpen = true;
  navigationReady = false;
  errorMessage = '';
  /** The version string of the XProf plugin. */
  pluginVersion = '';

  constructor(
      store: Store<{}>,
      private readonly communicationService: CommunicationService,
  ) {
    window.sessionStorage.setItem(
        'searchParams',
        new URLSearchParams(window.location.search).toString(),
    );
    store.select(getLoadingState)
        .pipe(takeUntil(this.destroyed))
        .subscribe((loadingState: LoadingState) => {
          this.loading = loadingState.loading;
          this.loadingMessage = loadingState.message;
        });
    store.select(getErrorMessage)
        .pipe(takeUntil(this.destroyed))
        .subscribe((errorMessage: string) => {
          this.errorMessage = errorMessage;
        });
    this.communicationService.navigationReady.subscribe(
        (navigationEvent: NavigationEvent) => {
          this.navigationReady = true;
          // TODO(fe-unification): Remove this constraint once the sidepanel
          // content of the 3 tools are moved out from sidenav with consolidated
          // templates.
          const toolsWithSideNav =
              ['op_profile', 'memory_viewer', 'pod_viewer'];
          this.isSideNavOpen =
              (navigationEvent.firstLoad ||
               toolsWithSideNav
                       .filter(tool => navigationEvent?.tag?.startsWith(tool))
                       .length > 0);
        });
    this.dataService.getPluginVersion()
        .pipe(takeUntil(this.destroyed))
        .subscribe((version: string | null) => {
          this.pluginVersion = version || '';
        });
  }

  get diagnostics(): Diagnostics {
    return {
      errors: this.errorMessage ? [this.errorMessage] : [],
      info: [],
      warnings: [],
    };
  }

  ngOnDestroy() {
    // Unsubscribes all pending subscriptions.
    this.destroyed.next();
    this.destroyed.complete();
  }
}
