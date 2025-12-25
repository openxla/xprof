import {PlatformLocation} from '@angular/common';
import {Component, inject, Injector, OnDestroy} from '@angular/core';
import {ActivatedRoute, Params} from '@angular/router';
import {API_PREFIX, DATA_API, PLUGIN_NAME} from 'org_xprof/frontend/app/common/constants/constants';
import {SOURCE_CODE_SERVICE_INTERFACE_TOKEN} from 'org_xprof/frontend/app/services/source_code_service/source_code_service_interface';
import {combineLatest, ReplaySubject} from 'rxjs';
import {takeUntil} from 'rxjs/operators';

/** A trace viewer component. */
@Component({
  standalone: false,
  selector: 'trace-viewer',
  templateUrl: './trace_viewer.ng.html',
  styleUrls: ['./trace_viewer.css']
})
export class TraceViewer implements OnDestroy {
  /** Handles on-destroy Subject, used to unsubscribe. */
  private readonly destroyed = new ReplaySubject<void>(1);
  private readonly injector = inject(Injector);

  url = '';
  pathPrefix = '';
  sessionId = '';
  host = '';
  hosts = '';  // Comma-separated list of hosts, to support multi-host sessions.
  tool = 'trace_viewer';
  runPath = '';
  sessionPath = '';

  constructor(
      platformLocation: PlatformLocation,
      route: ActivatedRoute,
  ) {
    if (String(platformLocation.pathname).includes(API_PREFIX + PLUGIN_NAME)) {
      this.pathPrefix =
          String(platformLocation.pathname).split(API_PREFIX + PLUGIN_NAME)[0];
    }
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
    this.hosts = params['hosts'] || this.hosts;
    this.runPath = params['run_path'] || this.runPath;
    this.sessionPath = params['session_path'] || this.sessionPath;
  }

  update() {
    const isStreaming = (this.tool === 'trace_viewer@');
    let queryString = `run=${this.sessionId}&tag=${this.tool}`;

    if (this.sessionPath) {
      queryString += `&session_path=${this.sessionPath}`;
    } else if (this.runPath) {
      queryString += `&run_path=${this.runPath}`;
    }

    if (this.hosts && typeof this.hosts === 'string') {
      // Since event.hosts is a comma-separated string, we can use it directly.
      queryString += `&hosts=${this.hosts}`;
    } else if (this.host) {
      queryString += `&host=${this.host}`;
    }

    const traceDataUrl = `${this.pathPrefix}${DATA_API}?${queryString}`;
    this.url = `${this.pathPrefix}${API_PREFIX}${
        PLUGIN_NAME}/trace_viewer_index.html?is_streaming=${
        isStreaming}&is_oss=true&trace_data_url=${
        encodeURIComponent(traceDataUrl)}&source_code_service=${
        this.isSourceCodeServiceAvailable()}`;
  }

  private isSourceCodeServiceAvailable() {
    // We don't need the source code service to be persistently available.
    // We temporarily use the service to check if it is available and show
    // UI accordingly.
    const sourceCodeService = this.injector.get(
        SOURCE_CODE_SERVICE_INTERFACE_TOKEN,
        null,
    );
    return sourceCodeService?.isAvailable() === true;
  }

  ngOnDestroy() {
    // Unsubscribes all pending subscriptions.
    this.destroyed.next();
    this.destroyed.complete();
  }
}
