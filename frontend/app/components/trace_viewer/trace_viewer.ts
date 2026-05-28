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
  TemplateRef,
  ViewChild,
} from '@angular/core';
import {MatDialog, MatDialogConfig} from '@angular/material/dialog';
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
  TraceData as MainTraceData,
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
import {
  COLOR_PALETTE_STORAGE_KEY,
  COLOR_PALETTES,
  FILTER_CONFIG,
  FILTER_FIELD_EVENT_DURATION,
  FILTER_FIELDS,
  FILTER_OPERATORS,
  FILTER_PROPERTY_SEPARATOR,
  FILTER_SEPARATOR,
} from './constants';
import {
  FilterChangeEvent,
  FilterEntry,
  FilterFieldCategory,
  FilterOperatorType,
  FilterRemoveEvent,
  FlowCategory,
  TraceEventFilter,
  TraceFilters,
} from './trace_viewer_typings';
import {
  getProcessMappingsFromWasm,
  getProcessNamesFromWasm,
  parseEventsSelectedData,
} from './utils';

interface TraceData {
  traceEvents?: Array<{[key: string]: unknown}>;
  [key: string]: unknown;
}

/** The name of the event triggered when a trace event is selected. */
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
  private readonly destroyed = new ReplaySubject<void>(1);
  private navigationEvent: NavigationEvent = {};
  private readonly injector = inject(Injector);
  private readonly store = inject(Store<{}>);
  private readonly dialog = inject(MatDialog);

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

  @ViewChild('paletteDialog', {static: true})
  paletteDialog!: TemplateRef<{}>;

  private readonly router = inject(Router);

  selectedFilters: FilterEntry[] = [];
  validFilterFields = FILTER_FIELDS;
  processes: {[host: string]: string[]} = {};
  processesListFromJson: string[] = [];
  isUploadMode = false;
  fileUploaded = false;
  selectedPalette = 'Default';
  COLOR_PALETTES = COLOR_PALETTES;
  flowCategories: FlowCategory[] = [];
  allFlowCategories: FlowCategory[] = [];
  selectedFlowCategoryIds = new Set<number>();

  get filterSelectedHosts(): string[] {
    const hostFilterEntry: FilterEntry | undefined = this.selectedFilters.find(
      (filterEntry) =>
        filterEntry.field?.info.category === FilterFieldCategory.HOST,
    );
    if (hostFilterEntry?.operator.value === FilterOperatorType.EXACT) {
      return hostFilterEntry?.value
        .split(',')
        .map((h) => h.trim())
        .filter(Boolean);
    }
    return [];
  }

  get filterSelectedTimeRange(): [string | undefined, string | undefined] {
    const startTimeFilterEntry: FilterEntry | undefined =
      this.selectedFilters.find(
        (filterEntry) =>
          filterEntry.field?.info.category ===
          FilterFieldCategory.START_TIME_MS,
      );
    const endTimeFilterEntry: FilterEntry | undefined =
      this.selectedFilters.find(
        (filterEntry) =>
          filterEntry.field?.info.category === FilterFieldCategory.END_TIME_MS,
      );
    return [startTimeFilterEntry?.value, endTimeFilterEntry?.value];
  }

  get processesFilterValue() {
    const processFilterEntry: FilterEntry | undefined =
      this.selectedFilters.find(
        (filterEntry) =>
          filterEntry.field?.info.category === FilterFieldCategory.PROCESS,
      );

    if (processFilterEntry?.operator.value === FilterOperatorType.EXACT) {
      return processFilterEntry?.value
        .split(',')
        .map(
          (deviceString) =>
            deviceString.match(/^([^ ]+)\s+(.*)\s+\(pid\s+(\d+)\)$/)?.[2],
        )
        .filter(Boolean)
        .join(',');
    }
    return processFilterEntry?.value || '';
  }

  get threadsFilterValue() {
    const threadFilterEntry: FilterEntry | undefined =
      this.selectedFilters.find(
        (filterEntry) =>
          filterEntry.field?.info.category === FilterFieldCategory.THREAD,
      );
    return threadFilterEntry?.value || '';
  }

  get eventsFilterValue() {
    const eventFilterEntries: FilterEntry[] | undefined =
      this.selectedFilters.filter(
        (filterEntry) =>
          filterEntry.field?.info.category === FilterFieldCategory.EVENT,
      );

    let eventFilterValue = '';
    for (const eventFilterEntry of eventFilterEntries) {
      eventFilterValue +=
        eventFilterEntry.value +
        FILTER_PROPERTY_SEPARATOR +
        eventFilterEntry.operator.opId.toString() +
        FILTER_PROPERTY_SEPARATOR +
        eventFilterEntry.field.info.name! +
        FILTER_SEPARATOR;
    }

    return eventFilterValue.slice(0, -1);
  }

  get processList() {
    const processes: string[] = [];
    Object.values(this.processes).forEach((arr) => {
      processes.push(...arr);
    });
    return this.processesListFromJson.length > 0
      ? this.processesListFromJson
      : processes;
  }

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

    // Event listeners are handled by TraceViewerContainer.
    // WASM initialization is triggered via the child component's
    // (initializeWasm) output event once the canvas element is fully
    // mounted in the DOM.
    window.addEventListener(
      DETAILS_RECEIVED_EVENT_NAME,
      this.detailsReceivedEventListener,
    );

    const sourceCodeService = this.injector.get(
      SOURCE_CODE_SERVICE_INTERFACE_TOKEN,
      null,
    );
    // We don't need the source code service to be persistently available.
    // We temporarily use the service to check if it is available and show
    // UI accordingly.
    sourceCodeService
      ?.isAvailable()
      .pipe(takeUntil(this.destroyed))
      .subscribe((isAvailable) => {
        this.sourceCodeServiceIsAvailable = isAvailable;
      });
  }

  ngOnInit() {}

  ngAfterViewInit() {}

  async initializeWasmApp() {
    this.traceViewerModule = await traceViewerV2Main();

    const savedPalette = window.localStorage.getItem(COLOR_PALETTE_STORAGE_KEY);
    if (savedPalette && this.traceViewerModule) {
      this.selectedPalette = savedPalette;
      this.traceViewerModule.SetPalette(savedPalette);
    }

    if (this.traceViewerModule && this.traceViewerModule.getAllFlowCategories) {
      this.allFlowCategories = this.traceViewerModule.getAllFlowCategories();
      this.selectAllFlowCategories();
    }

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

    if (this.hasValidTraceFilters()) {
      additionalParams.set(
        FILTER_CONFIG,
        JSON.stringify(this.getTraceFilters()),
      );
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
        this.traceViewerModule.loadTraceData(traceDataUrl).then(() => {
          this.updateFlowCategories();
          this.updateWasmFlowCategories();
          this.updateWasmProcessMappings();
        });
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

  isDetailChecked(key: TraceDetailKey): boolean {
    return this.traceDetails.get(key) || false;
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
      pid,
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
      this.maybeFetchEventArgs(name, startUs, durationUs, uid, pid);
    }
  }

  updateWasmProcessMappings() {
    const mappings = getProcessMappingsFromWasm(this.traceViewerModule);
    for (const [pid, host] of mappings.entries()) {
      this.pidToHostMap.set(pid, host);
    }

    const processNames = getProcessNamesFromWasm(this.traceViewerModule);
    const uniqueHosts = new Set<string>(this.hostList);
    const hostToProcessList: {[host: string]: Set<string>} = {};

    for (const [pid, processName] of processNames.entries()) {
      if (processName) {
        const host = processName.split(' ')[0];
        if (host) {
          uniqueHosts.add(host);
          if (!hostToProcessList[host]) {
            hostToProcessList[host] = new Set<string>();
          }
          hostToProcessList[host].add(`${host} ${processName} (pid ${pid})`);
        }
      }
    }

    if (uniqueHosts.size > 0) {
      this.hostList = Array.from(uniqueHosts).sort();
    }

    for (const host of Object.keys(hostToProcessList)) {
      this.processes[host] = Array.from(hostToProcessList[host]).sort();
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

      if (result.isCounter) {
        this.eventDetailColumns = ['counter', 'series', 'time', 'value'];
      } else {
        this.eventDetailColumns = [
          'name',
          'occurrences',
          'wallDuration',
          'selfTime',
          'avgWallDuration',
        ];
      }
    } catch (e) {
      console.error('Failed to parse events selected data:', e);
      this.selectedEventProperties = [];
      this.eventDetailColumns = [...DEFAULT_EVENT_DETAIL_COLUMNS];
    }
  }

  onSearchEvents(detail: SearchEventsEventDetail) {
    const query = detail.events_query;
    if (!query) {
      this.traceViewerModule?.setSearchResultsInWasm({traceEvents: []});
      this.container?.updateSearchResultCountText();
      return;
    }
    this.searching = true;
    this.dataService
      .getData(
        this.navigationEvent.run || '',
        this.navigationEvent.tag || '',
        this.getCurrentHost(),
        new Map([['search_prefix', query]]),
      )
      .pipe(takeUntil(this.destroyed))
      .subscribe((data) => {
        this.searching = false;
        if (this.traceViewerModule && data) {
          const traceData = data as TraceData;
          this.traceViewerModule.setSearchResultsInWasm({
            ...traceData,
            traceEvents: traceData.traceEvents || [],
          } as MainTraceData);
          this.container?.updateSearchResultCountText();
        }
      });
  }

  onInitializeWasm() {
    setTimeout(() => {
      this.initializeWasmApp();
    });
  }

  private readonly pidToHostMap = new Map<number, string>();

  private maybeFetchEventArgs(
    name: string,
    startUs: number,
    durationUs: number,
    uid: string,
    pid?: number,
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
    const sanitizedUid = uid.includes('.')
      ? Math.floor(Number(uid)).toString()
      : uid;
    params.set('unique_id', sanitizedUid);

    let host = '';
    // Use precise host from pidToHostMap if available
    if (pid !== undefined && this.pidToHostMap.has(pid)) {
      host = this.pidToHostMap.get(pid)!;
    }
    if (!host) {
      host = this.getCurrentHost();
    }

    this.dataService
      .getData(
        this.navigationEvent.run || '',
        this.navigationEvent.tag || '',
        host,
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

  // END Trace Viewer V2 WASM App Methods

  // START Support of trace event filtering

  hasValidTraceFilters() {
    return (
      this.processesFilterValue.length > 0 ||
      this.threadsFilterValue.length > 0 ||
      this.eventsFilterValue.length > 0
    );
  }

  getTraceFilters(): TraceFilters {
    const eventFilters = [];
    const event = this.eventsFilterValue;

    const traceEventFilterStringList = event.split(FILTER_SEPARATOR);
    for (const tef of traceEventFilterStringList) {
      if (tef.length === 0) continue;
      const tefProps = tef.split(FILTER_PROPERTY_SEPARATOR);
      const opIdNumber = Number(tefProps[1]);
      if (tefProps?.[0]?.length > 0) {
        const filter: TraceEventFilter = {
          field_name: tefProps[2],
          op_id: opIdNumber,
        };
        if (opIdNumber === FILTER_OPERATORS[5].opId) {
          filter.regex_value = tefProps[0];
        } else if (tefProps[2] === FILTER_FIELD_EVENT_DURATION) {
          filter.double_value = String(Number(tefProps[0]));
        } else {
          filter.str_value = tefProps[0];
        }
        eventFilters.push(filter);
      }
    }

    return {
      device_regexes: this.processesFilterValue
        ? this.processesFilterValue.split(',')
        : [],
      resource_regexes: this.threadsFilterValue
        ? this.threadsFilterValue.split(',')
        : [],
      trace_event_filters: eventFilters,
    };
  }

  onFilterAdd(filter: FilterEntry) {
    this.selectedFilters.push(filter);
    this.refreshDataAfterFilterChange();
  }

  onFilterRemove(event: FilterRemoveEvent) {
    const {index} = event;
    this.selectedFilters.splice(index, 1);
    this.refreshDataAfterFilterChange();
  }

  onFiltersReset() {
    this.selectedFilters = [];
    this.refreshDataAfterFilterChange();
  }

  onFilterEdit(event: FilterChangeEvent) {
    const {value, index} = event;
    if (!value.length || value === this.selectedFilters[index].value) {
      return;
    }
    this.selectedFilters[index].value = value;
    this.refreshDataAfterFilterChange();
  }

  refreshDataAfterFilterChange() {
    void this.update(this.navigationEvent);
  }

  // END Support of trace event filtering

  // START Support of flow categories view

  selectAllFlowCategories() {
    this.selectedFlowCategoryIds = new Set(
      this.flowCategories.map((c) => c.id),
    );
    this.updateWasmFlowCategories();
  }

  selectNoneFlowCategories() {
    this.selectedFlowCategoryIds.clear();
    this.updateWasmFlowCategories();
  }

  toggleFlowCategory(category: FlowCategory) {
    if (this.selectedFlowCategoryIds.has(category.id)) {
      this.selectedFlowCategoryIds.delete(category.id);
    } else {
      this.selectedFlowCategoryIds.add(category.id);
    }
    this.updateWasmFlowCategories();
  }

  isSelectedFlowCategory(category: FlowCategory) {
    return this.selectedFlowCategoryIds.has(category.id);
  }

  updateWasmFlowCategories() {
    if (!this.traceViewerModule || !this.traceViewerModule.application) {
      return;
    }
    const instance = this.traceViewerModule.application.instance?.();
    if (instance && instance.setVisibleFlowCategories) {
      const idsArray = Array.from(this.selectedFlowCategoryIds).map(Number);
      instance.setVisibleFlowCategories(idsArray);
    }
  }

  private updateFlowCategories() {
    if (!this.traceViewerModule || !this.traceViewerModule.application) return;
    const instance = this.traceViewerModule.application.instance?.();
    if (!instance || !instance.dataProvider) return;
    const dataProvider = instance.dataProvider();
    if (!dataProvider || !dataProvider.getFlowCategories) return;
    const presentCategories = dataProvider.getFlowCategories();
    if (!presentCategories) return;
    const presentCategoriesSet = new Set();
    for (let i = 0; i < presentCategories.size(); i++) {
      presentCategoriesSet.add(presentCategories.get(i));
    }
    this.flowCategories = this.allFlowCategories.filter(
      (category: FlowCategory) => presentCategoriesSet.has(category.id),
    );
  }

  // END Support of flow categories view

  // START Support of color palettes selection

  openColorPaletteSettings() {
    const config: MatDialogConfig = {maxWidth: 350};
    const dialogRef = this.dialog.open(this.paletteDialog, config);

    dialogRef.afterClosed().subscribe((result: string | undefined) => {
      if (result && this.traceViewerModule) {
        this.selectedPalette = result;
        this.traceViewerModule.SetPalette(result);
        window.localStorage.setItem(COLOR_PALETTE_STORAGE_KEY, result);
      }
    });
  }

  onPaletteChange(palette: string) {
    this.selectedPalette = palette;
  }

  // END Support of color palettes selection
}
