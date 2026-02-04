import {CommonModule, PlatformLocation} from '@angular/common';
import {Component, inject, Injector, OnDestroy} from '@angular/core';
import {SafePipe} from 'org_xprof/frontend/app/pipes/safe_pipe';
import {ActivatedRoute} from '@angular/router';
import {Store} from '@ngrx/store';
import {API_PREFIX, DATA_API, PLUGIN_NAME} from 'org_xprof/frontend/app/common/constants/constants';
import {HostMetadata} from 'org_xprof/frontend/app/common/interfaces/hosts';
import {NavigationEvent} from 'org_xprof/frontend/app/common/interfaces/navigation_event';
import {SOURCE_CODE_SERVICE_INTERFACE_TOKEN} from 'org_xprof/frontend/app/services/source_code_service/source_code_service_interface';
import {getHostsState} from 'org_xprof/frontend/app/store/selectors';
import {combineLatest, ReplaySubject} from 'rxjs';
import {takeUntil} from 'rxjs/operators';

/** A trace viewer component. */
@Component({
  standalone: true,
  selector: 'trace-viewer',
  templateUrl: 'trace_viewer.ng.html',
  styleUrls: ['trace_viewer.css'],
  imports: [CommonModule, SafePipe]
})
export class TraceViewer implements OnDestroy {
  /** Handles on-destroy Subject, used to unsubscribe. */
  private readonly destroyed = new ReplaySubject<void>(1);
  private readonly injector = inject(Injector);
  private readonly store = inject(Store<{}>);

  url = '';
  pathPrefix = '';
  sourceCodeServiceIsAvailable = false;
  hostList: string[] = [];

  constructor(
      platformLocation: PlatformLocation,
      route: ActivatedRoute,
  ) {
    if (String(platformLocation.pathname).includes(API_PREFIX + PLUGIN_NAME)) {
      this.pathPrefix =
          String(platformLocation.pathname).split(API_PREFIX + PLUGIN_NAME)[0];
    }
    combineLatest(
        [route.params, route.queryParams, this.store.select(getHostsState)])
        .pipe(takeUntil(this.destroyed))
        .subscribe(([params, queryParams, hostsMetadata]) => {
          if (hostsMetadata && hostsMetadata.length > 0) {
            this.hostList =
                hostsMetadata.map((host: HostMetadata) => host.hostname);
          }
          this.update(params as NavigationEvent);
        });

    // We don't need the source code service to be persistently available.
    // We temporarily use the service to check if it is available and show
    // UI accordingly.
    const sourceCodeService =
        this.injector.get(SOURCE_CODE_SERVICE_INTERFACE_TOKEN, null);
    sourceCodeService?.isAvailable()
        .pipe(takeUntil(this.destroyed))
        .subscribe((isAvailable) => {
          this.sourceCodeServiceIsAvailable = isAvailable;
        });
  }

  update(event: NavigationEvent) {
    const isStreaming = (event.tag === 'trace_viewer@');
    const run = event.run || '';
    const tag = event.tag || '';
    const runPath = event.run_path || '';
    const sessionPath = event.session_path || '';
    let queryString = `run=${run}&tag=${tag}`;

    if (sessionPath) {
      queryString += `&session_path=${sessionPath}`;
    } else if (runPath) {
      queryString += `&run_path=${runPath}`;
    }

    if (event.hosts && typeof event.hosts === 'string') {
      // Since event.hosts is a comma-separated string, we can use it directly.
      queryString += `&hosts=${event.hosts}`;
    } else if (event.host) {
      queryString += `&host=${event.host}`;
    } else {
      queryString +=
          `&host=${this.hostList.length > 0 ? this.hostList[0] : ''}`;
    }

    const traceDataUrl = `${this.pathPrefix}${DATA_API}?${queryString}`;
    this.url = `${this.pathPrefix}${API_PREFIX}${
        PLUGIN_NAME}/trace_viewer_index.html?is_streaming=${
        isStreaming}&is_oss=true&trace_data_url=${
        encodeURIComponent(traceDataUrl)}&source_code_service=${
        this.sourceCodeServiceIsAvailable}`;
  }

  ngOnDestroy() {
    // Unsubscribes all pending subscriptions.
    this.destroyed.next();
    this.destroyed.complete();
  }
}
