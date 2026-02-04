import {PlatformLocation} from '@angular/common';
import {
  AfterViewInit,
  Component,
  inject,
  Injector,
  OnDestroy,
  OnInit,
} from '@angular/core';
import {ActivatedRoute} from '@angular/router';
import {Store} from '@ngrx/store';
import {
  API_PREFIX,
  DATA_API,
  PLUGIN_NAME,
} from 'org_xprof/frontend/app/common/constants/constants';
import {HostMetadata} from 'org_xprof/frontend/app/common/interfaces/hosts';
import {NavigationEvent} from 'org_xprof/frontend/app/common/interfaces/navigation_event';
import {
  LOADING_STATUS_UPDATE_EVENT_NAME,
  TraceViewerV2LoadingStatus,
  traceViewerV2Main,
  TraceViewerV2Module,
} from 'org_xprof/frontend/app/components/trace_viewer_v2/main';
import {SOURCE_CODE_SERVICE_INTERFACE_TOKEN} from 'org_xprof/frontend/app/services/source_code_service/source_code_service_interface';
import {getHostsState} from 'org_xprof/frontend/app/store/selectors';
import {combineLatest, ReplaySubject} from 'rxjs';
import {takeUntil} from 'rxjs/operators';

/** A trace viewer component. */
@Component({
  standalone: false,
  selector: 'trace-viewer',
  templateUrl: './trace_viewer.ng.html',
  styleUrls: ['./trace_viewer.css'],
})
export class TraceViewer implements OnDestroy, OnInit, AfterViewInit {
  /** Handles on-destroy Subject, used to unsubscribe. */
  private readonly destroyed = new ReplaySubject<void>(1);
  private readonly injector = inject(Injector);
  private readonly store = inject(Store<{}>);
  private lastNavigationEvent?: NavigationEvent;

  url = '';
  pathPrefix = '';
  useTraceViewerV2 = false;

  traceViewerModule: TraceViewerV2Module|null = null;
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
    route.queryParams.pipe(takeUntil(this.destroyed)).subscribe(queryParams => {
      if (queryParams['use_trace_viewer_v2'] === 'false') {
        this.useTraceViewerV2 = false;
      }
    });
    combineLatest(
        [route.params, route.queryParams, this.store.select(getHostsState)])
        .pipe(takeUntil(this.destroyed))
        .subscribe(([params, queryParams, hostsMetadata]) => {
          this.lastNavigationEvent = params as NavigationEvent;
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

  async update(event: NavigationEvent) {
    this.lastNavigationEvent = event;
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

    if (this.useTraceViewerV2) {
      if (!this.traceViewerModule) {
        // If module is not ready, skip update. It will be called again from
        // initializeWasmApp after module is loaded.
        return;
      }

      if (!this.traceViewerModule.loadJsonData) {
        const message = 'Trace Viewer v2 loadJsonData is not available.';
        console.error(message);
        window.dispatchEvent(
            new CustomEvent(LOADING_STATUS_UPDATE_EVENT_NAME, {
              detail: {
                status: TraceViewerV2LoadingStatus.ERROR,
                message,
              },
            }),
        );
        return;
      }
      await this.traceViewerModule.loadJsonData(traceDataUrl);
    } else {
      this.url = `${this.pathPrefix}${API_PREFIX}${
          PLUGIN_NAME}/trace_viewer_index.html?use_trace_viewer_v2=false&is_streaming=${
          isStreaming}&is_oss=true&trace_data_url=${
          encodeURIComponent(traceDataUrl)}&source_code_service=${
          this.sourceCodeServiceIsAvailable}`;
    }
  }

  ngOnInit() {}

  ngAfterViewInit() {
    // TODO: b/433979009 - Update to use Trace Viewer v2 if the browser supports
    // WebGPU.
    if (this.useTraceViewerV2) {
      // Use setTimeout to defer initialization to the next tick to avoid
      // ExpressionChangedAfterItHasBeenCheckedError when setting 'loading'
      // state.
      setTimeout(() => {
        this.initializeWasmApp();
      });
    }
  }

  async initializeWasmApp() {
    try {
      window.dispatchEvent(
          new CustomEvent(LOADING_STATUS_UPDATE_EVENT_NAME, {
            detail: {status: TraceViewerV2LoadingStatus.INITIALIZING},
          }),
      );
      this.traceViewerModule = await traceViewerV2Main();
      if (!this.traceViewerModule) {
        if (!document.querySelector('#canvas')) {
          // Log elements to console.
          const elements = document.querySelectorAll('*');
          console.log('Elements:');
          for (let i = 0; i < elements.length; i++) {
            console.log(elements[i].outerHTML);
          }
          throw new Error(
              'Trace Viewer v2 initialization failed: canvas not found.',
          );
        }
        throw new Error('Trace Viewer v2 initialization failed');
      }
      if (this.lastNavigationEvent) {
        await this.update(this.lastNavigationEvent);
      }
    } catch (e) {
      console.error(
          'Failed to initialize Trace Viewer v2, falling back to the old version:',
          e,
      );
      this.switchToOldFrontend();
      window.dispatchEvent(
          new CustomEvent(LOADING_STATUS_UPDATE_EVENT_NAME, {
            detail: {status: TraceViewerV2LoadingStatus.IDLE},
          }),
      );
    }
  }

  // TODO(b/433979009): Remove this method when the old frontend is deprecated.
  switchToOldFrontend() {
    const queryParams = new URLSearchParams(window.location.search);
    queryParams.set('use_trace_viewer_v2', 'false');
    window.location.search = queryParams.toString();
  }

  ngOnDestroy() {
    // Unsubscribes all pending subscriptions.
    this.destroyed.next();
    this.destroyed.complete();
  }
}
