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
  TraceViewerV2Module,
  traceViewerV2Main,
} from 'org_xprof/frontend/app/components/trace_viewer_v2/main';
import {SOURCE_CODE_SERVICE_INTERFACE_TOKEN} from 'org_xprof/frontend/app/services/source_code_service/source_code_service_interface';
import {getHostsState} from 'org_xprof/frontend/app/store/selectors';
import {combineLatest, interval, ReplaySubject, Subscription} from 'rxjs';
import {takeUntil} from 'rxjs/operators';

// The tutorials to display while the trace viewer is loading.
const TUTORIALS = Object.freeze([
  'Pan: A/D or Shift+Scroll or Drag',
  'Zoom: W/S or Ctrl+Scroll',
  'Scroll: Up/Down Arrow or Scroll',
]);

// The interval at which to rotate the tutorials.
const TUTORIAL_ROTATION_INTERVAL_MS = 3_000;

/**
 * The detail of a 'LoadingStatusUpdate' custom event.
 */
declare interface LoadingStatusUpdateEventDetail {
  status: TraceViewerV2LoadingStatus;
  message?: string;
}

// Type guard for the 'LoadingStatusUpdate' custom event.
function isLoadingStatusUpdateEvent(
    event: Event,
    ): event is CustomEvent<LoadingStatusUpdateEventDetail> {
  return (
      event instanceof CustomEvent && event.detail && event.detail.status &&
      Object.values(TraceViewerV2LoadingStatus).includes(event.detail.status));
}

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
  protected readonly TraceViewerV2LoadingStatus = TraceViewerV2LoadingStatus;
  traceViewerV2LoadingStatus: TraceViewerV2LoadingStatus =
      TraceViewerV2LoadingStatus.IDLE;
  protected traceViewerV2ErrorMessage?: string;

  protected readonly tutorials = TUTORIALS;
  protected currentTutorialIndex = 0;
  private tutorialSubscription?: Subscription;
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
        console.error('Trace Viewer v2 loadJsonData is not available.');
        this.updateLoadingStatus(TraceViewerV2LoadingStatus.ERROR);
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

  ngOnInit() {
    window.addEventListener(
        LOADING_STATUS_UPDATE_EVENT_NAME,
        this.loadingStatusUpdateEventListener,
    );
  }

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
      this.updateLoadingStatus(TraceViewerV2LoadingStatus.INITIALIZING);
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
      this.updateLoadingStatus(TraceViewerV2LoadingStatus.IDLE);
    }
  }

  private readonly loadingStatusUpdateEventListener = (event: Event) => {
    if (!isLoadingStatusUpdateEvent(event)) {
      return;
    }

    this.updateLoadingStatus(event.detail.status);

    if (event.detail.status !== TraceViewerV2LoadingStatus.ERROR) {
      this.traceViewerV2ErrorMessage = undefined;
    } else {
      this.traceViewerV2ErrorMessage = event.detail.message;
    }
  };

  /**
   * Updates the loading status and starts/stops the tutorial rotation
   * accordingly.
   *
   * If the status changes to IDLE or ERROR, the tutorial rotation is stopped.
   * Otherwise (e.g., INITIALIZING, LOADING_DATA), the tutorial rotation is
   * started to provide user feedback.
   */
  private updateLoadingStatus(status: TraceViewerV2LoadingStatus) {
    if (this.traceViewerV2LoadingStatus === status) {
      return;
    }
    this.traceViewerV2LoadingStatus = status;

    if (this.traceViewerV2LoadingStatus === TraceViewerV2LoadingStatus.IDLE ||
        this.traceViewerV2LoadingStatus === TraceViewerV2LoadingStatus.ERROR) {
      // Stop the tutorial rotation when loading is finished or failed.
      this.stopTutorialRotation();
    } else {
      // Start the tutorial rotation when loading is in progress.
      this.startTutorialRotation();
    }
  }

  /**
   * Starts the tutorial rotation.
   *
   * This method initializes the `tutorialSubscription` to rotate through
   * tutorials at a set interval. It ensures only one subscription is active at
   * a time. The subscription lifecycle is managed here and will be terminated
   * when `stopTutorialRotation` is called or when the component is destroyed.
   */
  private startTutorialRotation() {
    if (this.tutorialSubscription) return;

    this.tutorialSubscription =
        interval(TUTORIAL_ROTATION_INTERVAL_MS)
            .pipe(takeUntil(this.destroyed))
            .subscribe(() => {
              this.currentTutorialIndex =
                  (this.currentTutorialIndex + 1) % this.tutorials.length;
            });
  }

  /**
   * Stops the tutorial rotation.
   *
   * This method unsubscribes from the `tutorialSubscription` and clears the
   * reference, stopping the interval timer.
   */
  private stopTutorialRotation() {
    if (this.tutorialSubscription) {
      this.tutorialSubscription.unsubscribe();
      this.tutorialSubscription = undefined;
    }
  }

  // TODO(b/433979009): Remove this method when the old frontend is deprecated.
  switchToOldFrontend() {
    const queryParams = new URLSearchParams(window.location.search);
    queryParams.set('use_trace_viewer_v2', 'false');
    window.location.search = queryParams.toString();
  }

  ngOnDestroy() {
    window.removeEventListener(
        LOADING_STATUS_UPDATE_EVENT_NAME,
        this.loadingStatusUpdateEventListener,
    );
    // Unsubscribes all pending subscriptions.
    this.destroyed.next();
    this.destroyed.complete();
    this.stopTutorialRotation();
  }
}
