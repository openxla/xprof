import 'org_xprof/frontend/app/common/interfaces/window';

import {PlatformLocation} from '@angular/common';
import {
  AfterViewInit,
  ChangeDetectionStrategy,
  Component,
  inject,
  Injector,
  OnDestroy,
  OnInit,
  ViewChild,
} from '@angular/core';
import {ActivatedRoute, Router} from '@angular/router';
import {Store} from '@ngrx/store';
import {
  API_PREFIX,
  PLUGIN_NAME,
} from 'org_xprof/frontend/app/common/constants/constants';
import {HostMetadata} from 'org_xprof/frontend/app/common/interfaces/hosts';
import {NavigationEvent} from 'org_xprof/frontend/app/common/interfaces/navigation_event';
import {
  EntrySelectedEventDetail,
  EventsSelectedEventDetail,
  SelectedEvent,
  SelectedEventProperty,
  TraceViewerContainer,
} from 'org_xprof/frontend/app/components/trace_viewer_container/trace_viewer_container';
import {
  DETAILS_RECEIVED_EVENT_NAME,
  isDetailsReceivedEvent,
  SearchEventsEventDetail,
  TraceDetailKey,
  TraceDetails,
  traceViewerV2Main,
  TraceViewerV2Module,
} from 'org_xprof/frontend/app/components/trace_viewer_v2/main';
import {DataServiceV2} from 'org_xprof/frontend/app/services/data_service_v2/data_service_v2';
import {SOURCE_CODE_SERVICE_INTERFACE_TOKEN} from 'org_xprof/frontend/app/services/source_code_service/source_code_service_interface';
import {getHostsState} from 'org_xprof/frontend/app/store/selectors';
import {combineLatest, ReplaySubject} from 'rxjs';
import {takeUntil} from 'rxjs/operators';
import {parseEventsSelectedData} from './utils';

interface TraceData {
  traceEvents?: Array<{[key: string]: unknown}>;
  [key: string]: unknown;
}

/**
 * The name of the event selected custom event, dispatched from WASM in Trace
 * Viewer v2.
 */
export const EVENT_SELECTED_EVENT_NAME = 'eventselected';

const DEFAULT_EVENT_DETAIL_COLUMNS = Object.freeze(['property', 'value']);

function parseHostsList(hosts: unknown): string[] {
  if (typeof hosts === 'string') {
    return hosts
      .split(',')
      .map((h) => h.trim())
      .filter(Boolean);
  }
  return Array.isArray(hosts) ? hosts : [];
}

/** A trace viewer component. */
@Component({
  changeDetection: ChangeDetectionStrategy.Default,
  standalone: false,
  selector: 'trace-viewer',
  templateUrl: './trace_viewer.ng.html',
  styleUrls: ['./trace_viewer.css'],
})
export class TraceViewer implements OnInit, AfterViewInit, OnDestroy {
  /** Handles on-destroy Subject, used to unsubscribe. */
  private readonly destroyed = new ReplaySubject<void>(1);
  private navigationEvent: NavigationEvent = {};
  private readonly injector = inject(Injector);
  private readonly store = inject(Store<{}>);

  url = '';
  pathPrefix = '';
  sourceCodeServiceIsAvailable = false;
  hostList: string[] = [];
  useTraceViewerV2 = (() => {
    try {
      return (
        new URLSearchParams(window.location.search).get(
          'use_trace_viewer_v2',
        ) === 'true' ||
        window.localStorage.getItem('use_trace_viewer_v2') === 'true'
      );
    } catch {
      return (
        new URLSearchParams(window.location.search).get(
          'use_trace_viewer_v2',
        ) === 'true'
      );
    }
  })();
  traceViewerModule: TraceViewerV2Module | null = null;
  selectedEvent: SelectedEvent | null = null;
  selectedEventProperties: SelectedEventProperty[] = [];
  eventDetailColumns = [...DEFAULT_EVENT_DETAIL_COLUMNS];

  selectionStartFormat?: string;
  selectionExtentFormat?: string;
  private readonly eventArgsCache = new Map<string, {[key: string]: string}>();
  private queryString = '';
  searching = false;
  readonly availableDetails: Array<{key: TraceDetailKey; label: string}> = [
    {key: 'full_dma', label: 'Full DMA'},
  ];
  traceDetails: TraceDetails = new Map();

  @ViewChild(TraceViewerContainer, {static: false})
  container?: TraceViewerContainer;

  private readonly router = inject(Router);

  constructor(
    private readonly dataService: DataServiceV2,
    platformLocation: PlatformLocation,
    route: ActivatedRoute,
  ) {
    if (String(platformLocation.pathname).includes(API_PREFIX + PLUGIN_NAME)) {
      this.pathPrefix = String(platformLocation.pathname).split(
        API_PREFIX + PLUGIN_NAME,
      )[0];
    }
    combineLatest([
      route.params,
      route.queryParams,
      this.store.select(getHostsState),
    ])
      .pipe(takeUntil(this.destroyed))
      .subscribe(([params, queryParams, hostsMetadata]) => {
        if (hostsMetadata && hostsMetadata.length > 0) {
          this.hostList = hostsMetadata.map(
            (host: HostMetadata) => host.hostname,
          );
        }
        try {
          this.useTraceViewerV2 =
            queryParams['use_trace_viewer_v2'] === 'true' ||
            window.localStorage.getItem('use_trace_viewer_v2') === 'true';
        } catch {
          this.useTraceViewerV2 = queryParams['use_trace_viewer_v2'] === 'true';
        }
        this.navigationEvent = {...params, ...queryParams};
        this.update(this.navigationEvent);
      });

    window.addEventListener(
      DETAILS_RECEIVED_EVENT_NAME,
      this.detailsReceivedEventListener,
    );

    // We don't need the source code service to be persistently available.
    // We temporarily use the service to check if it is available and show
    // UI accordingly.
    const sourceCodeService = this.injector.get(
      SOURCE_CODE_SERVICE_INTERFACE_TOKEN,
      null,
    );
    sourceCodeService
      ?.isAvailable()
      .pipe(takeUntil(this.destroyed))
      .subscribe((isAvailable) => {
        this.sourceCodeServiceIsAvailable = isAvailable;
      });
  }

  ngOnInit() {
    // Event listeners are handled by TraceViewerContainer.
  }

  ngAfterViewInit() {
    if (this.useTraceViewerV2) {
      // Use setTimeout to defer initialization to the next tick to avoid
      // ExpressionChangedAfterItHasBeenCheckedError when setting 'loading'
      // state.
      this.onInitializeWasm();
    }
  }

  async initializeWasmApp() {
    this.traceViewerModule = await traceViewerV2Main();
    this.update(this.navigationEvent);
  }

  update(event: NavigationEvent) {
    const isStreaming = event.tag === 'trace_viewer@';
    const run = event.run || '';
    const tag = event.tag || '';
    const runPath = event.run_path || '';
    const sessionPath = event.session_path || '';
    this.queryString = `run=${run}&tag=${tag}`;

    if (sessionPath) {
      this.queryString += `&session_path=${sessionPath}`;
    } else if (runPath) {
      this.queryString += `&run_path=${runPath}`;
    }

    let hostsString = '';
    if (event.hosts) {
      const hostsList = parseHostsList(event.hosts);

      // Sort hosts to ensure stable query string
      hostsString = hostsList.sort().slice(0, 10).join(',');
      this.queryString += `&hosts=${hostsString}`;
    } else if (event.host) {
      this.queryString += `&host=${event.host}`;
    } else {
      this.queryString += `&host=${this.hostList.length > 0 ? this.hostList[0] : ''}`;
    }

    const additionalParams = new Map<string, string>();
    if (sessionPath) {
      additionalParams.set('session_path', sessionPath);
    } else if (runPath) {
      additionalParams.set('run_path', runPath);
    }

    if (hostsString) {
      additionalParams.set('hosts', hostsString);
    }

    // Sort keys to ensure stable query string regardless of insertion order
    const entries = Array.from(this.traceDetails.entries()).sort((a, b) =>
      a[0].localeCompare(b[0]),
    );
    entries.forEach(([key, value]) => {
      if (value) {
        additionalParams.set(key, 'true');
      }
    });

    const traceDataUrl = this.dataService.getDataUrl(
      run,
      tag,
      this.getCurrentHost(event),
      additionalParams,
    );

    if (this.useTraceViewerV2) {
      if (this.traceViewerModule && this.traceViewerModule.loadTraceData) {
        this.traceViewerModule.loadTraceData(traceDataUrl);
      }
    } else {
      this.url = `${this.pathPrefix}${API_PREFIX}${
        PLUGIN_NAME
      }/trace_viewer_index.html?is_streaming=${
        isStreaming
      }&is_oss=true&trace_data_url=${encodeURIComponent(
        traceDataUrl,
      )}&source_code_service=${this.sourceCodeServiceIsAvailable}`;
    }
  }

  ngOnDestroy() {
    // Unsubscribes all pending subscriptions.
    this.destroyed.next();
    this.destroyed.complete();
    window.removeEventListener(
      DETAILS_RECEIVED_EVENT_NAME,
      this.detailsReceivedEventListener,
    );
  }

  private readonly detailsReceivedEventListener = (event: Event) => {
    if (!isDetailsReceivedEvent(event)) {
      return;
    }
    const eventDetails = event.detail.details;

    if (this.areDetailsChanged(eventDetails)) {
      this.traceDetails = new Map(eventDetails);
      void this.update(this.navigationEvent);
    }
  };

  private areDetailsChanged(newDetails: TraceDetails): boolean {
    if (this.traceDetails.size !== newDetails.size) return true;
    for (const [key, value] of newDetails.entries()) {
      if (this.traceDetails.get(key) !== value) return true;
    }
    return false;
  }

  toggleDetail(name: TraceDetailKey, checked: boolean) {
    this.traceDetails.set(name, checked);
    void this.update(this.navigationEvent);
  }

  getCurrentHost(event: NavigationEvent = this.navigationEvent): string {
    if (event.hosts) {
      const hostsList = parseHostsList(event.hosts);
      if (hostsList.length > 0) {
        return hostsList[0];
      }
    }
    return event.host || (this.hostList && this.hostList[0]) || '';
  }

  // START Trace Viewer V2 WASM App Methods

  onEventSelected(event: EntrySelectedEventDetail | null) {
    if (!event) {
      this.selectedEvent = null;
      this.selectedEventProperties = [];
      this.selectionStartFormat = undefined;
      this.selectionExtentFormat = undefined;
      return;
    }
    this.selectionStartFormat = undefined;
    this.selectionExtentFormat = undefined;
    this.eventDetailColumns = [...DEFAULT_EVENT_DETAIL_COLUMNS];

    const {
      name,
      startUsFormatted,
      durationUsFormatted,
      hloModuleName,
      hloOpName,
      uid,
      startUs,
      durationUs,
    } = event;

    this.selectedEvent = {
      name,
      startUsFormatted,
      durationUsFormatted,
    };

    const properties: SelectedEventProperty[] = [];
    properties.push({property: 'Name', value: name});
    properties.push({property: 'Start Time', value: startUsFormatted});
    properties.push({property: 'Duration', value: durationUsFormatted});
    if (hloModuleName) {
      properties.push({property: 'HLO Module', value: hloModuleName});
    }
    if (hloOpName) {
      properties.push({property: 'HLO Op', value: hloOpName});
    }
    this.selectedEventProperties = properties;

    if (uid) {
      this.maybeFetchEventArgs(name, startUs, durationUs, uid);
    }
  }

  onEventsSelected(event: EventsSelectedEventDetail | null) {
    if (!event) {
      this.selectedEvent = null;
      this.selectedEventProperties = [];
      this.selectionStartFormat = undefined;
      this.selectionExtentFormat = undefined;
      return;
    }

    this.selectedEvent = {
      name: 'Multiple Events Selected',
    };

    try {
      const result = parseEventsSelectedData(event.events_selected_data);
      this.selectedEventProperties = result.properties;
      this.selectionStartFormat = result.selectionStartFormat;
      this.selectionExtentFormat = result.selectionExtentFormat;

      this.eventDetailColumns = [
        'name',
        'occurrences',
        'wallDuration',
        'selfTime',
        'avgWallDuration',
      ];
    } catch (e) {
      console.error('Failed to parse events selected data:', e);
      this.selectedEventProperties = [];
      this.eventDetailColumns = [...DEFAULT_EVENT_DETAIL_COLUMNS];
    }
  }

  onSearchEvents(detail: SearchEventsEventDetail) {
    const query = detail.events_query;
    if (!this.traceViewerModule) return;

    const app = this.traceViewerModule.application.instance();
    app.setSearchQuery(query || '');
    this.container?.updateSearchResultCountText();
  }

  onInitializeWasm() {
    setTimeout(() => {
      this.initializeWasmApp();
    });
  }

  private maybeFetchEventArgs(
    name: string,
    startUs: number,
    durationUs: number,
    uid: string,
  ) {
    const key = `${name}-${startUs}-${durationUs}`;
    if (this.eventArgsCache.has(key)) {
      if (this.selectedEvent) {
        this.addArgsToSelectedEvent(this.eventArgsCache.get(key)!);
      }
      return;
    }

    const params = new Map<string, string>();
    params.set('event_name', name);
    params.set('start_time_ms', (startUs / 1000).toString());
    params.set('duration_ms', (durationUs / 1000).toString());
    params.set('unique_id', Math.floor(Number(uid)).toString());

    let tool = this.navigationEvent.tag || '';
    if (this.useTraceViewerV2 && tool === 'trace_viewer') {
      tool = 'trace_viewer.pb';
    }

    this.dataService
      .getData(
        this.navigationEvent.run || '',
        tool,
        this.getCurrentHost(),
        params,
      )
      .pipe(takeUntil(this.destroyed))
      .subscribe((data) => {
        const traceData = data as TraceData;
        if (
          !traceData ||
          !traceData.traceEvents ||
          traceData.traceEvents.length === 0
        ) {
          return;
        }
        const lastEvent =
          traceData.traceEvents[traceData.traceEvents.length - 1];
        if (
          lastEvent['ph'] === 'X' &&
          this.selectedEvent &&
          lastEvent['args']
        ) {
          const args = lastEvent['args'] as {[key: string]: string};
          this.eventArgsCache.set(key, args);
          this.addArgsToSelectedEvent(args);
        }
      });
  }

  private addArgsToSelectedEvent(args: {[key: string]: string}) {
    if (!this.selectedEvent) return;
    const properties = [...this.selectedEventProperties];
    for (const key of Object.keys(args)) {
      properties.push({property: key, value: args[key]});
    }
    this.selectedEventProperties = properties;
  }

  // END Trace Viewer V2 WASM App Methods

  switchToOldFrontend(showSurvey = true) {
    this.useTraceViewerV2 = false;
    window.gtag &&
      window.gtag('event', 'switch-frontend', {
        'event_category': 'user_interaction',
        'event_label': 'switch_to_old',
        'screen_name': 'trace viewer',
        'tool_name': 'trace viewer',
      });
    const queryParams = this.dataService.getSearchParams();
    queryParams.set('use_trace_viewer_v2', 'false');
    // Store preference to stay on v1
    window.localStorage.removeItem('use_trace_viewer_v2');

    // Add a flag to tell the Angular to show the HaTS survey.
    // Set to true when user switches from v2 to v1 by clicking the "Switch to
    // old frontend" button.
    // Set to false when there is some error in v2 frontend and we force to
    // switch to v1.
    if (showSurvey) {
      queryParams.set('show_hats_survey', 'true');
    }
    this.dataService.setSearchParams(queryParams);
    this.router.navigate([], {
      queryParams: (() => {
        const params: Record<string, string> = {};
        queryParams.forEach((value, key) => {
          params[key] = value;
        });
        return params;
      })(),
      replaceUrl: true,
    });
  }

  switchToV2Frontend() {
    this.useTraceViewerV2 = true;
    window.gtag &&
      window.gtag('event', 'switch-frontend', {
        'event_category': 'user_interaction',
        'event_label': 'switch_to_v2',
        'screen_name': 'trace viewer',
        'tool_name': 'trace viewer',
      });
    const queryParams = this.dataService.getSearchParams();
    queryParams.set('use_trace_viewer_v2', 'true');
    window.localStorage.setItem('use_trace_viewer_v2', 'true');

    // Delete the survey flag in v2 to keep the url clean.
    queryParams.delete('show_hats_survey');
    this.dataService.setSearchParams(queryParams);
    this.router.navigate([], {
      queryParams: (() => {
        const params: Record<string, string> = {};
        queryParams.forEach((value, key) => {
          params[key] = value;
        });
        return params;
      })(),
      replaceUrl: true,
    });
    if (!this.traceViewerModule) {
      this.onInitializeWasm();
    }
  }

  switchVersion() {
    if (this.useTraceViewerV2) {
      this.switchToOldFrontend();
    } else {
      this.switchToV2Frontend();
    }
  }
}
