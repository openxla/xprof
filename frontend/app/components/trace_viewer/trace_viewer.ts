import {PlatformLocation} from '@angular/common';
import {Component, inject, Injector, OnDestroy} from '@angular/core';
import {ActivatedRoute} from '@angular/router';
import {API_PREFIX, DATA_API, PLUGIN_NAME} from 'org_xprof/frontend/app/common/constants/constants';
import {NavigationEvent} from 'org_xprof/frontend/app/common/interfaces/navigation_event';
import {SOURCE_CODE_SERVICE_INTERFACE_TOKEN} from 'org_xprof/frontend/app/services/source_code_service/source_code_service_interface';
import {ReplaySubject} from 'rxjs';
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

  constructor(
      platformLocation: PlatformLocation,
      route: ActivatedRoute,
  ) {
    if (String(platformLocation.pathname).includes(API_PREFIX + PLUGIN_NAME)) {
      this.pathPrefix =
          String(platformLocation.pathname).split(API_PREFIX + PLUGIN_NAME)[0];
    }
    route.params.pipe(takeUntil(this.destroyed)).subscribe((params) => {
      this.update(params as NavigationEvent);
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
