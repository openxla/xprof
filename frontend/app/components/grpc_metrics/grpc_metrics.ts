import {CommonModule} from '@angular/common';
import {
  ChangeDetectorRef,
  Component,
  DestroyRef,
  HostListener,
  OnInit,
  effect,
  inject,
  signal,
  viewChild,
} from '@angular/core';
import {takeUntilDestroyed} from '@angular/core/rxjs-interop';
import {FormsModule} from '@angular/forms';
import {MatButtonModule} from '@angular/material/button';
import {MatCheckboxModule} from '@angular/material/checkbox';
import {MatChipsModule} from '@angular/material/chips';
import {MatExpansionModule} from '@angular/material/expansion';
import {MatFormFieldModule} from '@angular/material/form-field';
import {MatIconModule} from '@angular/material/icon';
import {MatInputModule} from '@angular/material/input';
import {MatPaginator, MatPaginatorModule} from '@angular/material/paginator';
import {MatProgressSpinnerModule} from '@angular/material/progress-spinner';
import {MatSelectModule} from '@angular/material/select';
import {MatSort, MatSortModule} from '@angular/material/sort';
import {MatTableDataSource, MatTableModule} from '@angular/material/table';
import {MatTabsModule} from '@angular/material/tabs';
import {MatTooltipModule} from '@angular/material/tooltip';
import {ActivatedRoute, Params} from '@angular/router';

import {
  ChartDataInfo,
  ChartType,
} from 'org_xprof/frontend/app/common/interfaces/chart';
import {SimpleDataTable} from 'org_xprof/frontend/app/common/interfaces/data_table';
import {ChartModule} from 'org_xprof/frontend/app/components/chart/chart';
import {combineLatest} from 'rxjs';
import {distinctUntilChanged, filter, map} from 'rxjs/operators';

import {GrpcChartConfigFactory} from './grpc_metrics_charts';
import {
  GRPC_METRICS_DATA_SERVICE,
  GRPC_METRICS_OPTIONS,
  RECOMMENDATIONS_QUERY_PARAM,
  resolveEnableRecommendations,
} from './grpc_metrics_options';
import {
  ChartMetricKind,
  TcpEventsInfo,
  buildMultiLineChartData,
  cleanDcnLabel,
  computeBackgroundData,
  computeInsights,
  computeTpuVmLatencyGroups,
  formatBytes,
  formatMetricValue,
  getTcpLatencyNote,
  groupNetworkInterfacesByHost,
  normalizeIdForMatching,
  parseEndpointId,
  parseNetworkInterfaces,
  parseSystemConfigs,
  parseTcpEvents,
  parseXorTimestamps,
  prepareLatencyBreakdown,
  prepareLatencyTimelineInfo,
  prepareTpuVmStepLatencies,
  quoteLargeIntIds,
  setStragglerSearchStrings,
} from './grpc_metrics_utils';
import {
  GrpcMetricsDatabase,
  GrpcStraggler,
  HloCollectiveStats,
  HostNetworkInterfacesGroup,
  InsightSummary,
  NetworkInterfaceInfo,
  SystemConfig,
  TcpStatsSnapshot,
  TimeSeriesPoint,
  TpuVmAnalytics,
  TpuVmLatencyInfo,
  TpuVmStepLatencyInfo,
  TraceEvent,
  XorTimestamps,
} from './types';

/** Group of DCN transfer chunks aggregated by source/destination endpoints and TPU VM. */
export declare interface DcnLinkGroup {
  srcDst: string;
  tpuVm: string;
  chunksCount: number;
  totalLatency: number;
  maxLatency: number;
  chunks: GrpcStraggler[];
}

/** DCN collective operation details aggregated by HLO instruction and collective label. */
export declare interface DcnOperation {
  hloInstruction: string;
  collectiveName: string;
  chunksCount: number;
  totalLatency: number;
  maxLatency: number;
  linkGroups: DcnLinkGroup[];
  chunks: GrpcStraggler[];
}

interface ScaledDataTable extends SimpleDataTable {
  p?: {
    unitSuffix?: string;
    divisor?: number;
  };
}

interface ChartOptionsWithVAxis {
  vAxis?: {
    title?: string;
  };
}

interface WindowWithGoogle extends Window {
  google?: {
    charts?: unknown;
  };
}

declare interface FilterObject {
  general?: string;
  hloCollective?: string;
  srcDevice?: string;
  dstDevice?: string;
  tpuVm?: string;
  launchId?: string;
  loopIndex?: string;
  chunkIndex?: string;
}

/** Session and host of a data request. */
declare interface DataRequest {
  sessionId: string;
  host?: string;
}

/**
 * Returns the session and host to load from the route. The `sessionId` path
 * parameter wins and is sent without a host. Otherwise the session is the
 * `run` query parameter and the host is the `host` query parameter, if any.
 * An empty session means there is nothing to load.
 */
function getDataRequest(params: Params, queryParams: Params): DataRequest {
  const pathSessionId: unknown = params['sessionId'];
  if (pathSessionId) {
    return {sessionId: String(pathSessionId)};
  }
  const runSessionId: unknown = queryParams['run'];
  if (!runSessionId) {
    return {sessionId: ''};
  }
  const host: unknown = queryParams['host'];
  return {
    sessionId: String(runSessionId),
    host: host ? String(host) : undefined,
  };
}

/**
 * Main component for displaying gRPC metrics.
 * Provides insights into network performance, TPU stalls, and TCP statistics.
 */
@Component({
  standalone: true,
  selector: 'grpc-metrics',
  templateUrl: './grpc_metrics.ng.html',
  styleUrls: ['./grpc_metrics.scss'],
  imports: [
    CommonModule,
    MatTableModule,
    MatPaginatorModule,
    ChartModule,
    MatSortModule,
    MatTabsModule,
    MatTooltipModule,
    MatIconModule,
    MatInputModule,
    MatFormFieldModule,
    MatChipsModule,
    MatSelectModule,
    MatProgressSpinnerModule,
    FormsModule,
    MatExpansionModule,
    MatButtonModule,
    MatCheckboxModule,
  ],
})
export class GrpcMetrics implements OnInit {
  private readonly route = inject(ActivatedRoute);
  private readonly dataService = inject(GRPC_METRICS_DATA_SERVICE);
  private readonly options = inject(GRPC_METRICS_OPTIONS);
  private readonly cdr = inject(ChangeDetectorRef);
  private readonly destroyRef = inject(DestroyRef);
  private isDestroyed = false;

  /**
   * Whether the Network Recommendations tab, the insight chips and the
   * `insight:` search are shown. ngOnInit applies the URL override.
   */
  enableRecommendations = this.options.enableRecommendations;
  isLoading = true;
  readonly showHloCollectives = false;
  sessionId = '';
  /**
   * Host sent with the data request. Only set when the session comes from the
   * `run` query parameter; undefined when it comes from the route path.
   */
  host?: string;
  selectedTabIndex = signal(0);
  filterInputValue = '';
  p90Latency = 0;

  // Combined filter fields
  filterHloCollective = '';
  filterSrcDevice = '';
  filterDstDevice = '';
  filterTpuVm = '';
  filterLaunchId = '';
  filterLoopIndex = '';
  filterChunkIndex = '';
  filterGeneralSearch = '';
  expandedElement: GrpcStraggler | null = null;
  chartGroupingMode: 'tpu_vm' | 'endpoint_id' = 'tpu_vm';
  topPanelHeight = 350;
  private isResizing = false;

  startResizing(event: MouseEvent) {
    event.preventDefault();
    this.isResizing = true;
    document.addEventListener('mousemove', this.onMouseMove);
    document.addEventListener('mouseup', this.onMouseUp);
  }

  private readonly onMouseMove = (event: MouseEvent) => {
    if (!this.isResizing) return;
    const container = document.querySelector('.table-container');
    if (container) {
      const containerRect = container.getBoundingClientRect();
      const newHeight = event.clientY - containerRect.top;
      const minHeight = 200;
      const maxHeight = Math.max(minHeight, containerRect.height - 200);
      this.topPanelHeight = Math.max(minHeight, Math.min(newHeight, maxHeight));
      this.cdr.detectChanges();
    }
  };

  private readonly onMouseUp = () => {
    this.isResizing = false;
    document.removeEventListener('mousemove', this.onMouseMove);
    document.removeEventListener('mouseup', this.onMouseUp);
  };

  isChartsLoading = false;
  isChartsPrepLoading = false;
  isGoogleChartsLoaded = false;
  endpointToPeerMap = new Map<string, string>();

  availableTpuVms: string[] = [];
  selectedTpuVms: string[] = [];

  readonly MAX_CONCURRENT_SECTIONS = 2;
  openSections: string[] = [];
  sectionsLoading = new Map<string, boolean>();

  toggleSection(sectionId: string, isOpen: boolean) {
    if (isOpen) {
      if (!this.openSections.includes(sectionId)) {
        this.openSections.push(sectionId);
      }
      if (this.openSections.length > this.MAX_CONCURRENT_SECTIONS) {
        this.openSections.shift();
      }
      // Lazy load data for this specific section with a deferred loader
      this.sectionsLoading.set(sectionId, true);
      this.cdr.detectChanges();

      setTimeout(() => {
        this.sectionsLoading.set(sectionId, false);
        this.cdr.detectChanges();
        this.updateSectionCharts(sectionId);
      }, 100);
    } else {
      this.openSections = this.openSections.filter((id) => id !== sectionId);
      this.sectionsLoading.delete(sectionId);
    }
  }

  isSectionOpen(sectionId: string): boolean {
    return this.openSections.includes(sectionId);
  }

  rawStragglers: GrpcStraggler[] = [];
  hotCache: GrpcStraggler[] = [];

  readonly downsampledGraphs = new Set<string>();

  updateTpuVmSelection(host: string, checked: boolean) {
    if (checked) {
      if (!this.selectedTpuVms.includes(host)) {
        this.selectedTpuVms = [...this.selectedTpuVms, host];
      }
    } else {
      this.selectedTpuVms = this.selectedTpuVms.filter((h) => h !== host);
    }
    this.onTpuVmSelectionChange();
  }

  selectAllTpuVms() {
    this.selectedTpuVms = [...this.availableTpuVms];
    this.onTpuVmSelectionChange();
  }

  clearTpuVmSelection() {
    this.selectedTpuVms = [];
    this.onTpuVmSelectionChange();
  }

  toggleTpuVmExpand(group: TpuVmLatencyInfo) {
    group.isExpanded = !group.isExpanded;
    this.cdr.detectChanges();
  }
  readonly stragglers = new MatTableDataSource<GrpcStraggler>([]);
  readonly tpuVmAnalytics = new MatTableDataSource<TpuVmAnalytics>([]);
  readonly hloCollectives = new MatTableDataSource<HloCollectiveStats>([]);
  readonly hostSystemConfigs = new MatTableDataSource<SystemConfig>([]);
  rawNetworkInterfaces: NetworkInterfaceInfo[] = [];
  readonly networkInterfaces = new MatTableDataSource<NetworkInterfaceInfo>([]);
  hostNetworkInterfaceGroups: HostNetworkInterfacesGroup[] = [];
  readonly formatBytes = formatBytes;
  readonly networkInterfacesColumns = [
    'tpu_vm',
    'iface',
    'numa_node',
    'speed_mbps',
    'mtu',
    'operstate',
    'duplex',
    'delta_rx_bytes',
    'delta_tx_bytes',
    'delta_rx_dropped',
    'delta_tx_dropped',
    'delta_rx_errors',
    'delta_tx_errors',
  ];
  tpuVmLatencyGroups: TpuVmLatencyInfo[] = [];
  readonly dcnOperationsDataSource = new MatTableDataSource<DcnOperation>([]);
  readonly dcnLinkGroupsDataSource = new MatTableDataSource<DcnLinkGroup>([]);
  readonly selectedLinkGroupChunksDataSource =
    new MatTableDataSource<GrpcStraggler>([]);
  selectedDcnOperation: DcnOperation | null = null;
  selectedLinkGroup: DcnLinkGroup | null = null;
  dcnOperationsColumns = ['hloInstruction', 'collectiveName', 'chunksCount'];
  dcnLinkGroupsColumns = ['src_dst', 'tpu_vm', 'chunksCount', 'maxLatency'];
  dcnChunksColumns = [
    'chunk_id',
    'latency_ms',
    'iteration_id',
    'shard_range',
    'dcn_tag',
    'raw_transfer_key',
  ];

  updateNetworkInterfacesTable() {
    this.networkInterfaces.data = this.rawNetworkInterfaces.filter(
      (nic) => nic.is_physical,
    );
    this.hostNetworkInterfaceGroups = groupNetworkInterfacesByHost(
      this.rawNetworkInterfaces,
    );
    this.cdr.detectChanges();
  }

  expandAllHostNics(expanded: boolean) {
    for (const group of this.hostNetworkInterfaceGroups) {
      group.isExpanded = expanded;
    }
    this.cdr.detectChanges();
  }

  systemConfigColumns: string[] = [];
  dynamicSystemConfigColumns: string[] = [];
  xorTimestampsMap = new Map<string, XorTimestamps>();
  tcpEventsMap = new Map<string, TcpEventsInfo>();

  insightSummaries: InsightSummary[] = [];
  mlInsights: InsightSummary[] = [];
  rpcInsights: InsightSummary[] = [];
  tcpInsights: InsightSummary[] = [];
  diagnostics?: {
    stallMs: number;
    overlapMs: number;
  };

  readonly charts = new GrpcChartConfigFactory();
  readonly chartType = ChartType.SCATTER_CHART;
  readonly histogramChartType = ChartType.HISTOGRAM;
  readonly pieChartType = ChartType.PIE_CHART;
  hostStepLatencies: TpuVmStepLatencyInfo[] = [];
  readonly displayedColumns = [
    'chunk_id',
    'tpu_vm',
    'latency_ms',
    'grpc_status',
  ];
  readonly analyticsDisplayedColumns = [
    'tpu_vm',
    'total_transfers',
    'p50_latency_ms',
    'p99_latency_ms',
    'p100_latency_ms',
    'high_latency_count',
  ];
  readonly hloCollectivesColumns = [
    'hlo_collective_name',
    'total_flows',
    'total_latency_ms',
    'total_tpu_stall_ms',
    'total_slack_ms',
  ];
  readonly matSort = viewChild(MatSort);
  readonly stragglersPaginator = viewChild<MatPaginator>('stragglersPaginator');
  readonly dcnOpsPaginator = viewChild<MatPaginator>('dcnOpsPaginator');

  constructor() {
    effect(() => {
      const s = this.matSort();
      const index = this.selectedTabIndex();
      if (s) {
        // We use custom "Hot Cache" sorting for index 0 (stragglers) to prevent UI freeze.
        if (index === 1) this.tpuVmAnalytics.sort = s;
        else if (index === 2) {
          this.hostSystemConfigs.sort = s;
          this.networkInterfaces.sort = s;
        } else if (index === 4) this.hloCollectives.sort = s; // was 3 before, but we add a new tab at 2
        this.cdr.detectChanges();
      }
    });

    effect(() => {
      const p = this.stragglersPaginator();
      if (p) {
        this.stragglers.paginator = p;
        this.stragglers.data = this.hotCache;
        this.cdr.detectChanges();
      }
    });

    effect(() => {
      const p = this.dcnOpsPaginator();
      if (p) {
        this.dcnOperationsDataSource.paginator = p;
        this.cdr.detectChanges();
      }
    });
  }

  chartsToDisplay = 10;
  filteredHostStepLatenciesList: TpuVmStepLatencyInfo[] = [];
  totalFilteredHostStepLatenciesCount = 0;

  updateFilteredHostStepLatencies() {
    if (this.selectedTpuVms.length === 0) {
      this.filteredHostStepLatenciesList = [];
      this.totalFilteredHostStepLatenciesCount = this.hostStepLatencies.length;
      return;
    }
    const selectedSet = new Set(this.selectedTpuVms);
    const list = this.hostStepLatencies.filter((h) =>
      selectedSet.has(h.tpu_vm),
    );
    this.totalFilteredHostStepLatenciesCount = list.length;
    this.filteredHostStepLatenciesList = list.slice(0, this.chartsToDisplay);
  }

  loadMoreCharts() {
    this.chartsToDisplay += 10;
    this.updateFilteredHostStepLatencies();
    this.cdr.detectChanges();
  }

  @HostListener('window:resize')
  onResize() {
    // Redraw charts on window resize
    this.cdr.detectChanges();
    if (this.isGoogleChartsLoaded) {
      // Force chart layout recalculation
      const tmp = this.chartsToDisplay;
      this.chartsToDisplay = tmp;
    }
  }
  private dbCache?: GrpcMetricsDatabase;

  ngOnInit() {
    // Testers can turn recommendations on with a URL query parameter where
    // they are off by default.
    this.enableRecommendations = resolveEnableRecommendations(
      this.options.enableRecommendations,
      this.route.snapshot?.queryParamMap?.get(RECOMMENDATIONS_QUERY_PARAM),
    );
    this.checkGoogleChartsLoaded();
    this.destroyRef.onDestroy(() => {
      this.isDestroyed = true;
      document.removeEventListener('mousemove', this.onMouseMove);
      document.removeEventListener('mouseup', this.onMouseUp);
    });
    this.stragglers.filterPredicate = (
      data: GrpcStraggler,
      filterJson: string,
    ) => {
      // Evaluates targeted search prefixes for structured queries.
      const matchSearchPrefixes = (val: string): boolean | null => {
        val = val.trim().toLowerCase();
        if (!val) return true;
        const supportedPrefixes = [
          'tpu_vm:',
          // Searching by insight is part of the recommendations.
          ...(this.enableRecommendations ? ['insight:'] : []),
          'chunk:',
          'endpoint:',
          'flow:',
          'hlo_collective:',
          'dcn_link:',
        ];
        const matchedPrefix = supportedPrefixes.find((p) => val.startsWith(p));
        if (!matchedPrefix) {
          return null;
        }

        const query = val.substring(matchedPrefix.length).trim();
        if (!query) return true;

        if (matchedPrefix === 'tpu_vm:') {
          return (data.tpu_vm || '').toLowerCase().includes(query);
        }
        if (matchedPrefix === 'flow:' || matchedPrefix === 'chunk:') {
          return (data.chunk_id || '').toLowerCase().includes(query);
        }
        if (matchedPrefix === 'endpoint:') {
          const endpointsStr =
            `${(data.endpoint_ids || []).join(' ')} ${data.peer_address || ''} ${data.dcn_source_device_id || ''} -> ${data.dcn_destination_device_id || ''}`.toLowerCase();
          const tokens = query.split(/\s+/).filter(Boolean);
          return tokens.every((token) => endpointsStr.includes(token));
        }
        if (matchedPrefix === 'insight:') {
          const insightsStr = (data.insights || [])
            .map((i) => `${i.text} ${i.description || ''}`)
            .join(' ')
            .toLowerCase();
          return insightsStr.includes(query);
        }
        if (matchedPrefix === 'hlo_collective:') {
          const hloStr =
            `${this.formatHloInstructionName(data)} ${data.hlo_collective_name || ''} ${data.dcn_collective_label || ''}`.toLowerCase();
          return hloStr.includes(query);
        }
        if (matchedPrefix === 'dcn_link:') {
          if (data.dcn_source_device_id == null) return false;
          const linkStr =
            `${data.dcn_source_device_id} -> ${data.dcn_destination_device_id}`.toLowerCase();
          return linkStr.includes(query);
        }
        return false;
      };

      let filterObj: FilterObject;
      try {
        filterObj = JSON.parse(filterJson) as FilterObject;
      } catch (e) {
        const val = filterJson.trim().toLowerCase();
        if (!val) return true;
        const prefixMatch = matchSearchPrefixes(val);
        if (prefixMatch !== null) {
          return prefixMatch;
        }
        return (
          data.chunk_id.toLowerCase().includes(val) ||
          this.formatHloInstructionName(data).toLowerCase().includes(val) ||
          (data.hlo_collective_name || '').toLowerCase().includes(val) ||
          (data.dcn_collective_label || '').toLowerCase().includes(val) ||
          (data.tpu_vm || '').toLowerCase().includes(val) ||
          (data.peer_address || '').toLowerCase().includes(val) ||
          (data.endpoint_ids || []).some((ep) =>
            ep.toLowerCase().includes(val),
          ) ||
          (data.searchStr || '').includes(val)
        );
      }

      if (filterObj.general) {
        const val = filterObj.general.trim().toLowerCase();
        const prefixMatch = matchSearchPrefixes(val);
        if (prefixMatch !== null) {
          if (!prefixMatch) return false;
        } else {
          const isMatched =
            data.chunk_id.toLowerCase().includes(val) ||
            this.formatHloInstructionName(data).toLowerCase().includes(val) ||
            (data.hlo_collective_name || '').toLowerCase().includes(val) ||
            (data.dcn_collective_label || '').toLowerCase().includes(val) ||
            (data.tpu_vm || '').toLowerCase().includes(val) ||
            (data.peer_address || '').toLowerCase().includes(val) ||
            (data.endpoint_ids || []).some((ep) =>
              ep.toLowerCase().includes(val),
            ) ||
            (data.searchStr || '').includes(val);
          if (!isMatched) return false;
        }
      }

      if (filterObj.hloCollective) {
        const val = filterObj.hloCollective.trim().toLowerCase();
        const formatted = this.formatHloInstructionName(data).toLowerCase();
        const rawHlo = (data.hlo_collective_name || '').toLowerCase();
        const collLabel = (data.dcn_collective_label || '').toLowerCase();

        if (
          !formatted.includes(val) &&
          !rawHlo.includes(val) &&
          !collLabel.includes(val)
        ) {
          return false;
        }
      }
      if (filterObj.srcDevice) {
        const val = filterObj.srcDevice.trim().toLowerCase();
        if (!(data.dcn_source_device_id || '').toLowerCase().includes(val)) {
          return false;
        }
      }
      if (filterObj.dstDevice) {
        const val = filterObj.dstDevice.trim().toLowerCase();
        if (
          !(data.dcn_destination_device_id || '').toLowerCase().includes(val)
        ) {
          return false;
        }
      }
      if (filterObj.tpuVm) {
        const val = filterObj.tpuVm.trim().toLowerCase();
        if (!(data.tpu_vm || '').toLowerCase().includes(val)) {
          return false;
        }
      }
      if (filterObj.launchId) {
        const val = filterObj.launchId.trim().toLowerCase();
        if (!(data.dcn_launch_id || '').toLowerCase().includes(val)) {
          return false;
        }
      }
      if (filterObj.loopIndex) {
        const val = filterObj.loopIndex.trim().toLowerCase();
        if (!(data.dcn_loop_index || '').toLowerCase().includes(val)) {
          return false;
        }
      }
      if (filterObj.chunkIndex) {
        const val = filterObj.chunkIndex.trim().toLowerCase();
        if (!(data.dcn_chunk_index || '').toLowerCase().includes(val)) {
          return false;
        }
      }

      return true;
    };

    this.dcnOperationsDataSource.filterPredicate = (
      data: DcnOperation,
      filter: string,
    ) => {
      const matchStr = (
        data.hloInstruction +
        ' ' +
        (data.collectiveName || '')
      ).toLowerCase();
      return matchStr.includes(filter);
    };

    // The session comes from the route path (`grpc_metrics/:sessionId`) or,
    // without one, from the `run` query parameter. Data is loaded once per
    // session and host, so other query parameter changes do not reload it.
    combineLatest([this.route.params, this.route.queryParams])
      .pipe(
        map(([params, queryParams]) => getDataRequest(params, queryParams)),
        filter((request) => request.sessionId !== ''),
        distinctUntilChanged(
          (previous, current) =>
            previous.sessionId === current.sessionId &&
            previous.host === current.host,
        ),
        takeUntilDestroyed(this.destroyRef),
      )
      .subscribe((request) => {
        this.sessionId = request.sessionId;
        this.host = request.host;
        this.loadData();
      });
  }

  loadData() {
    this.dataService
      .getData(this.sessionId, 'grpc_metrics', this.host, undefined, {
        ignoreError: true,
      })
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe({
        next: (data: unknown) => {
          const db = data as GrpcMetricsDatabase;
          if (!db) {
            console.warn('Received null database in loadData');
            this.isLoading = false;
            this.cdr.detectChanges();
            return;
          }

          // Fast Path: parse core tables & insights, render immediately
          this.parseCoreData(db);
          this.isLoading = false;
          this.cdr.detectChanges();

          // Slow Path: defer heavy data & chart mapping calculations
          this.isChartsPrepLoading = true;
          this.cdr.detectChanges();

          setTimeout(() => {
            this.parseHeavyData(db);
            this.isChartsPrepLoading = false;
            for (const sectionId of this.openSections) {
              this.updateSectionCharts(sectionId);
            }
            this.cdr.detectChanges();
          }, 50);
        },
        error: (err: unknown) => {
          console.error('Failed to load gRPC metrics:', err);
          this.isLoading = false;
          this.cdr.detectChanges();
        },
      });
  }

  private parseCoreData(db: GrpcMetricsDatabase) {
    this.dbCache = db;
    this.downsampledGraphs.clear();
    for (const info of db.downsampled_info ?? []) {
      this.downsampledGraphs.add(`${info.tpu_vm}:${info.series_name}`);
    }

    try {
      const list = db.top_stragglers || [];
      this.rawStragglers = list;
      this.endpointToPeerMap.clear();
      for (const s of list) {
        if (s.dcn_collective_label) {
          // tslint:disable-next-line:enforce-name-casing
          const mutableStraggler = s as {dcn_collective_label?: string};
          mutableStraggler.dcn_collective_label = cleanDcnLabel(
            s.dcn_collective_label,
          );
        }
        if (s.peer_address && s.endpoint_ids) {
          for (const ep of s.endpoint_ids) {
            this.endpointToPeerMap.set(ep, s.peer_address);
          }
        }
      }
      this.applySortAndCache();
      this.computeDcnOperations(list);
      const latencies = list.map((s) => s.latency_ms).sort((a, b) => a - b);
      this.p90Latency =
        latencies.length > 0
          ? latencies[Math.floor(latencies.length * 0.9)]
          : 0;

      if (this.enableRecommendations) {
        const insights = computeInsights(list, this.p90Latency);
        this.mlInsights = insights.mlInsights;
        this.rpcInsights = insights.rpcInsights;
        this.tcpInsights = insights.tcpInsights;
        this.insightSummaries = insights.insightSummaries;
      } else {
        // computeInsights also sets the general search text of each row.
        setStragglerSearchStrings(list);
      }

      const bgData = computeBackgroundData(list);
      this.tpuVmAnalytics.data = bgData.tpuVmAnalyticsData;
      this.hloCollectives.data = bgData.collectiveStats;
      this.availableTpuVms = bgData.availableTpuVms;
      this.selectedTpuVms = [...this.availableTpuVms];
      this.tpuVmLatencyGroups = computeTpuVmLatencyGroups(list);

      this.diagnostics = {
        stallMs: db.total_network_stall_ms || 0,
        overlapMs: db.total_compute_overlap_ms || 0,
      };

      setTimeout(() => {
        const container = document.querySelector('.table-container');
        if (container && container.clientHeight > 0) {
          this.topPanelHeight = container.clientHeight / 2;
          this.cdr.detectChanges();
        }
      }, 0);
    } catch (e) {
      console.error('Error in parseCoreData:', e);
    }
  }

  formatHloInstructionName(s: GrpcStraggler): string {
    if (s.hlo_collective_name) {
      return s.hlo_collective_name;
    }
    if (s.dcn_collective_label) {
      return s.dcn_collective_label;
    }
    if (s.raw_transfer_key) {
      return s.raw_transfer_key;
    }
    return 'recv-done';
  }

  getDcnOpNames(s: GrpcStraggler): {
    hloInstruction: string;
    collectiveName: string;
  } {
    const hloInst = this.formatHloInstructionName(s);
    const collName =
      s.dcn_collective_label ||
      (s.hlo_collective_name !== hloInst ? s.hlo_collective_name : '') ||
      s.raw_transfer_key ||
      'Unknown';

    return {hloInstruction: hloInst, collectiveName: collName};
  }

  computeDcnOperations(stragglers: GrpcStraggler[]) {
    const dcnMap = new Map<string, DcnOperation>();
    for (const s of stragglers) {
      const names = this.getDcnOpNames(s);
      const key = `${names.hloInstruction} | ${names.collectiveName}`;
      if (!dcnMap.has(key)) {
        dcnMap.set(key, {
          hloInstruction: names.hloInstruction,
          collectiveName: names.collectiveName,
          chunksCount: 0,
          totalLatency: 0,
          maxLatency: 0,
          linkGroups: [],
          chunks: [],
        });
      }
      const op = dcnMap.get(key);
      if (op) {
        op.chunksCount++;
        op.totalLatency += s.latency_ms || 0;
        if ((s.latency_ms || 0) > op.maxLatency) {
          op.maxLatency = s.latency_ms || 0;
        }
      }
    }
    let operations = Array.from(dcnMap.values());
    operations.sort((a, b) => b.totalLatency - a.totalLatency);

    // Keep only top 100 operations
    operations = operations.slice(0, 100);

    // Load chunks initially only up to 20,000 total
    let totalChunksLoaded = 0;
    for (const op of operations) {
      if (totalChunksLoaded >= 20000) {
        break;
      }
      const opStragglers = stragglers.filter((s) => {
        const names = this.getDcnOpNames(s);
        return (
          names.hloInstruction === op.hloInstruction &&
          names.collectiveName === op.collectiveName
        );
      });
      const limit = Math.min(opStragglers.length, 20000 - totalChunksLoaded);
      const loadedStragglers = opStragglers.slice(0, limit);

      op.chunks = loadedStragglers;
      op.linkGroups = this.groupChunksByLink(loadedStragglers);
      totalChunksLoaded += loadedStragglers.length;
    }

    this.dcnOperationsDataSource.data = operations;
  }

  groupChunksByLink(chunks: GrpcStraggler[]): DcnLinkGroup[] {
    const linkMap = new Map<string, DcnLinkGroup>();
    for (const c of chunks) {
      const srcDst = this.getSrcDst(c);
      const key = `${srcDst}_${c.tpu_vm || 'Unknown'}`;
      if (!linkMap.has(key)) {
        linkMap.set(key, {
          srcDst,
          tpuVm: c.tpu_vm || 'Unknown',
          chunksCount: 0,
          totalLatency: 0,
          maxLatency: 0,
          chunks: [],
        });
      }
      const group = linkMap.get(key);
      if (group) {
        group.chunksCount++;
        group.totalLatency += c.latency_ms || 0;
        group.maxLatency = Math.max(group.maxLatency, c.latency_ms || 0);
        group.chunks.push(c);
      }
    }
    return Array.from(linkMap.values()).sort(
      (a, b) => b.totalLatency - a.totalLatency,
    );
  }

  selectDcnOperation(op: DcnOperation) {
    this.selectedDcnOperation = this.selectedDcnOperation === op ? null : op;
    this.selectedLinkGroup = null;
    this.selectedLinkGroupChunksDataSource.data = [];

    const selectedOp = this.selectedDcnOperation;
    if (selectedOp) {
      if (selectedOp.chunks.length < selectedOp.chunksCount) {
        // Load all chunks on-demand
        const allOpStragglers = this.rawStragglers.filter((s) => {
          const names = this.getDcnOpNames(s);
          return (
            names.hloInstruction === selectedOp.hloInstruction &&
            names.collectiveName === selectedOp.collectiveName
          );
        });
        selectedOp.chunks = allOpStragglers;
        selectedOp.linkGroups = this.groupChunksByLink(allOpStragglers);
      }
      this.dcnLinkGroupsDataSource.data = selectedOp.linkGroups;
    } else {
      this.dcnLinkGroupsDataSource.data = [];
    }
    this.cdr.detectChanges();
  }

  selectLinkGroup(group: DcnLinkGroup) {
    this.selectedLinkGroup = this.selectedLinkGroup === group ? null : group;
    if (this.selectedLinkGroup) {
      this.selectedLinkGroupChunksDataSource.data =
        this.selectedLinkGroup.chunks;
    } else {
      this.selectedLinkGroupChunksDataSource.data = [];
    }
    this.cdr.detectChanges();
  }

  getIterationId(s: GrpcStraggler): string {
    const launch = s.dcn_launch_id || 'N/A';
    const loop = s.dcn_loop_index || 'N/A';
    return `${launch} / ${loop}`;
  }

  getSrcDst(s: GrpcStraggler): string {
    const src = s.dcn_source_device_id || '?';
    const dst = s.dcn_destination_device_id || '?';
    return `${src} -> ${dst}`;
  }

  private parseHeavyData(db: GrpcMetricsDatabase) {
    try {
      const rawTraceEvents = db.trace_events_json || [];
      const traceEvents = rawTraceEvents
        .map((s: string) => {
          try {
            return JSON.parse(quoteLargeIntIds(s)) as TraceEvent;
          } catch (e) {
            console.error('Failed to parse trace event JSON:', s, e);
            return null;
          }
        })
        .filter((e: TraceEvent | null): e is TraceEvent => e !== null);

      const configs = parseSystemConfigs(traceEvents);
      this.hostSystemConfigs.data = configs;
      if (configs.length > 0) {
        const columnsSet = new Set(configs.flatMap((c) => Object.keys(c)));
        const columns = Array.from(columnsSet).filter(
          (col) =>
            col !== 'wmem_max' &&
            col !== 'rmem_max' &&
            col !== 'net.core.wmem_max' &&
            col !== 'net.core.rmem_max' &&
            col !== 'network_interfaces' &&
            col !== 'tpu_vm',
        );
        if (columnsSet.has('tpu_vm')) {
          this.dynamicSystemConfigColumns = ['tpu_vm', ...columns];
        } else {
          this.dynamicSystemConfigColumns = columns;
        }
        this.systemConfigColumns = [...this.dynamicSystemConfigColumns];
      }
      this.rawNetworkInterfaces = parseNetworkInterfaces(configs);
      this.updateNetworkInterfacesTable();
      this.xorTimestampsMap = parseXorTimestamps(traceEvents);
      for (const s of this.rawStragglers) {
        const xorTimestamps = this.xorTimestampsMap.get(
          normalizeIdForMatching(s.chunk_id),
        );
        if (xorTimestamps) {
          const mutableStraggler = s as {
            // tslint:disable-next-line:enforce-name-casing
            grpc_server_latency_ms?: number;
            // tslint:disable-next-line:enforce-name-casing
            client_request_size?: number;
            // tslint:disable-next-line:enforce-name-casing
            client_response_size?: number;
          };
          if (xorTimestamps.server_latency_ns !== undefined) {
            mutableStraggler.grpc_server_latency_ms =
              xorTimestamps.server_latency_ns / 1e6;
          }
          if (xorTimestamps.client_request_size !== undefined) {
            mutableStraggler.client_request_size =
              xorTimestamps.client_request_size;
          }
          if (xorTimestamps.client_response_size !== undefined) {
            mutableStraggler.client_response_size =
              xorTimestamps.client_response_size;
          }
        }
      }
      this.tcpEventsMap = parseTcpEvents(traceEvents);
      if (this.enableRecommendations) {
        const insights = computeInsights(
          this.rawStragglers,
          this.p90Latency,
          this.tcpEventsMap,
        );
        this.mlInsights = insights.mlInsights;
        this.rpcInsights = insights.rpcInsights;
        this.tcpInsights = insights.tcpInsights;
        this.insightSummaries = insights.insightSummaries;
      } else {
        setStragglerSearchStrings(this.rawStragglers);
      }

      this.hostStepLatencies = prepareTpuVmStepLatencies(
        db,
        this.charts.chartOptions,
      );
      this.chartsToDisplay = 10;
      this.selectedTpuVms = [];
      this.updateFilteredHostStepLatencies();
    } catch (e) {
      console.error('Error in parseHeavyData:', e);
    }
  }

  applySortAndCache() {
    const sort = this.matSort();
    const active = sort?.active || 'latency_ms';
    const direction = sort?.direction || 'desc';

    this.rawStragglers.sort((a, b) => {
      const dictA = a as unknown as Record<string, string | number | undefined>;
      const dictB = b as unknown as Record<string, string | number | undefined>;
      const valA = dictA[active];
      const valB = dictB[active];
      if (typeof valA === 'string' && typeof valB === 'string') {
        return direction === 'asc'
          ? valA.localeCompare(valB)
          : valB.localeCompare(valA);
      }
      const numA = (valA as number) || 0;
      const numB = (valB as number) || 0;
      const comparator = numA < numB ? -1 : numA > numB ? 1 : 0;
      return direction === 'asc' ? comparator : -comparator;
    });

    if (this.rawStragglers.length <= 16000) {
      this.hotCache = this.rawStragglers;
    } else {
      this.hotCache = [
        ...this.rawStragglers.slice(0, 8000),
        ...this.rawStragglers.slice(-8000),
      ];
    }
    if (this.hasActiveFilters()) {
      this.applyCombinedFilter();
    } else {
      const p = this.stragglersPaginator();
      if (!p) {
        this.stragglers.data = this.hotCache.slice(0, 50);
      } else {
        this.stragglers.data = this.hotCache;
      }
    }
    this.cdr.detectChanges();
  }

  private updateChart(
    dataInfo: ChartDataInfo,
    series: readonly TimeSeriesPoint[] | undefined,
    label: string,
    kind: ChartMetricKind,
  ) {
    const data = buildMultiLineChartData(
      series,
      label,
      this.selectedTpuVms,
      this.chartGroupingMode,
      undefined,
      this.endpointToPeerMap,
      kind,
    );
    dataInfo.data = data;
    if (dataInfo.dataProvider) {
      dataInfo.dataProvider.parseData(data);
      dataInfo.dataProvider.notifyCharts();
    }

    // Update axis title with binary unit if scaled
    const scaledData = data as ScaledDataTable;
    if (scaledData && scaledData.p && scaledData.p.unitSuffix) {
      const suffix = scaledData.p.unitSuffix;
      const options = (dataInfo.options as ChartOptionsWithVAxis) || {};
      if (options.vAxis) {
        const baseTitle = label.split(' (')[0];
        if (label.includes('/s')) {
          options.vAxis.title = `${baseTitle} (${suffix}/s)`;
        } else {
          options.vAxis.title = `${baseTitle} (${suffix})`;
        }
      }
    }
  }

  private updateSectionCharts(sectionId: string) {
    if (!this.dbCache) return;
    const c = this.charts;
    const db = this.dbCache;

    const MAP: Record<
      string,
      Array<{
        key: ChartDataInfo;
        series?: readonly TimeSeriesPoint[];
        label: string;
        kind: ChartMetricKind;
      }>
    > = {
      'grpc_latency': [
        {
          key: c.grpcClientLatencyDataInfo,
          series: db.grpc_client_latency_series,
          label: 'Client Latency (ms)',
          kind: 'gauge',
        },
        {
          key: c.grpcServerLatencyDataInfo,
          series: db.grpc_server_latency_series,
          label: 'Server Latency (ms)',
          kind: 'gauge',
        },
      ],
      'tcp_sec_1': [
        {
          key: c.tcpRttDataInfo,
          series: db.tcp_min_rtt_series,
          label: 'Min RTT (us)',
          kind: 'min',
        },
      ],
      'tcp_sec_2': [
        {
          key: c.tcpDeliveryRateDataInfo,
          series: db.tcp_delivery_rate_series,
          label: 'Delivery Rate (Bits/s)',
          kind: 'gauge',
        },
        {
          key: c.tcpPacingRateDataInfo,
          series: db.tcp_pacing_rate_series,
          label: 'Pacing Rate (Bits/s)',
          kind: 'gauge',
        },
        {
          key: c.tcpDataSentDataInfo,
          series: db.tcp_data_sent_series,
          label: 'Sent Rate (Bits/s)',
          kind: 'counter',
        },
        {
          key: c.tcpPacketSentDataInfo,
          series: db.tcp_packet_sent_series,
          label: 'Packet Sent (Pkts/s)',
          kind: 'counter',
        },
        {
          key: c.tcpPacketDeliveredDataInfo,
          series: db.tcp_packet_delivered_series,
          label: 'Delivered Rate (Pkts/s)',
          kind: 'counter',
        },
        {
          key: c.tcpAppLimitedDataInfo,
          series: db.tcp_is_delivery_rate_app_limited_series,
          label: 'App Limited (share of samples)',
          kind: 'gauge',
        },
      ],
      'tcp_sec_3': [
        {
          key: c.tcpCwndDataInfo,
          series: db.tcp_congestion_window_series,
          label: 'CWND (Packets)',
          kind: 'gauge',
        },
        {
          key: c.tcpSsthreshDataInfo,
          series: db.tcp_ssthresh_series,
          label: 'Ssthresh (Packets)',
          kind: 'gauge',
        },
      ],
      'tcp_sec_4': [
        {
          key: c.tcpByteOffsetDataInfo,
          series: db.tcp_byte_offset_series,
          label: 'Bit Offset (Bits)',
          kind: 'gauge',
        },
        {
          key: c.tcpDataNotsentDataInfo,
          series: db.tcp_data_notsent_series,
          label: 'Data Not Sent (Bits)',
          kind: 'gauge',
        },
      ],
      'tcp_sec_5': [
        {
          key: c.tcpPacketRetxDataInfo,
          series: db.tcp_packet_retx_series,
          label: 'Retx Rate (Pkts/s)',
          kind: 'counter',
        },
        {
          key: c.tcpPktSpurRetxDataInfo,
          series: db.tcp_packet_spurious_retx_series,
          label: 'Spurious Retx (Pkts/s)',
          kind: 'counter',
        },
        {
          key: c.tcpPktDeliveredCeDataInfo,
          series: db.tcp_packet_delivered_ce_series,
          label: 'CE Rate (Pkts/s)',
          kind: 'counter',
        },
        {
          key: c.tcpReorderingDataInfo,
          series: db.tcp_reordering_series,
          label: 'Reordering (Pkts)',
          kind: 'gauge',
        },
        {
          key: c.tcpRecurRetransDataInfo,
          series: db.tcp_recurring_retrans_series,
          label: 'Recurring Retrans (count)',
          kind: 'gauge',
        },
      ],
      'tcp_sec_6': [
        {
          key: c.tcpBusyUsecDataInfo,
          series: db.tcp_busy_usec_series,
          label: 'Busy Time (us/s)',
          kind: 'counter',
        },
        {
          key: c.tcpRwndLimitedUsecDataInfo,
          series: db.tcp_rwnd_limited_usec_series,
          label: 'RWND Limited (us/s)',
          kind: 'counter',
        },
        {
          key: c.tcpSndbufLimitedUsecDataInfo,
          series: db.tcp_sndbuf_limited_usec_series,
          label: 'SndBuf Limited (us/s)',
          kind: 'counter',
        },
      ],
    };

    if (MAP[sectionId]) {
      for (const m of MAP[sectionId]) {
        this.updateChart(m.key, m.series, m.label, m.kind);
      }
    }
  }

  onTpuVmSelectionChange() {
    this.isChartsLoading = true;
    this.updateFilteredHostStepLatencies();
    this.cdr.detectChanges();

    // Defer chart update to allow loader to show
    setTimeout(() => {
      // Re-trigger updates for all CURRENTLY OPEN sections
      for (const sectionId of this.openSections) {
        this.updateSectionCharts(sectionId);
      }
      this.isChartsLoading = false;
      this.cdr.detectChanges();
    }, 10);
  }

  onGroupingModeChange(mode: 'tpu_vm' | 'endpoint_id') {
    this.chartGroupingMode = mode;
    this.onTpuVmSelectionChange();
  }

  hasActiveFilters(): boolean {
    return !!(
      this.filterHloCollective ||
      this.filterSrcDevice ||
      this.filterDstDevice ||
      this.filterTpuVm ||
      this.filterLaunchId ||
      this.filterLoopIndex ||
      this.filterChunkIndex ||
      this.filterGeneralSearch
    );
  }

  applyCombinedFilter() {
    const filterObj = {
      hloCollective: this.filterHloCollective,
      srcDevice: this.filterSrcDevice,
      dstDevice: this.filterDstDevice,
      tpuVm: this.filterTpuVm,
      launchId: this.filterLaunchId,
      loopIndex: this.filterLoopIndex,
      chunkIndex: this.filterChunkIndex,
      general: this.filterGeneralSearch,
    };
    const filterStr = JSON.stringify(filterObj);
    const predicate = this.stragglers.filterPredicate;
    const filtered = this.rawStragglers.filter((item) =>
      predicate(item, filterStr),
    );

    // Cap search results to 1000 items to prevent rendering lags
    this.stragglers.data = filtered.slice(0, 1000);
    this.stragglers.filter = filterStr;
  }

  clearAllFilters() {
    this.filterHloCollective = '';
    this.filterSrcDevice = '';
    this.filterDstDevice = '';
    this.filterTpuVm = '';
    this.filterLaunchId = '';
    this.filterLoopIndex = '';
    this.filterChunkIndex = '';
    this.filterGeneralSearch = '';
    this.filterInputValue = '';
    this.applyCombinedFilter();
  }

  private filterStragglers(filterVal: string) {
    this.filterGeneralSearch = filterVal;
    this.filterInputValue = filterVal;
    this.applyCombinedFilter();
  }

  applyFilter(event: Event) {
    const val = (event.target as HTMLInputElement).value;
    this.filterGeneralSearch = val;
    this.filterInputValue = val;
    this.applyCombinedFilter();
  }

  applyDcnOperationsFilter(event: Event) {
    const val = (event.target as HTMLInputElement).value;
    this.dcnOperationsDataSource.filter = val.trim().toLowerCase();
  }

  filterByInsight(insight: string) {
    this.filterStragglers(`insight:${insight}`);
    this.selectedTabIndex.set(0);
  }

  filterByFlow(flowId: string) {
    this.filterStragglers(`flow:${flowId}`);
    this.selectedTabIndex.set(0);
  }

  filterByHloCollective(hlo: string) {
    this.filterHloCollective = hlo;
    this.applyCombinedFilter();
    this.selectedTabIndex.set(0); // Move to Step Overview tab
  }

  updateFilterForDcnRow(op: DcnOperation) {
    if (this.selectedDcnOperation === op) {
      this.filterHloCollective = op.hloInstruction;
    } else {
      this.filterHloCollective = '';
    }
    this.applyCombinedFilter();
  }

  updateFilterForLinkGroup(group: DcnLinkGroup) {
    if (this.selectedLinkGroup === group) {
      this.filterHloCollective = this.selectedDcnOperation
        ? this.selectedDcnOperation.hloInstruction
        : '';
      const parts = group.srcDst.split('->');
      if (parts.length === 2) {
        this.filterSrcDevice = parts[0].trim();
        this.filterDstDevice = parts[1].trim();
      }
      this.filterTpuVm = group.tpuVm !== 'Unknown' ? group.tpuVm : '';
      this.selectedTabIndex.set(0);
    } else {
      this.filterSrcDevice = '';
      this.filterDstDevice = '';
      this.filterTpuVm = '';
    }
    this.applyCombinedFilter();
  }

  copyToClipboard(text: string) {
    navigator.clipboard.writeText(text);
  }

  selectedTcpEventSnapshot: TcpStatsSnapshot | null = null;
  selectedTcpEventName: string | null = null;

  onTcpEventSelect(eventName: string, snapshot: TcpStatsSnapshot) {
    this.selectedTcpEventName = eventName;
    this.selectedTcpEventSnapshot = snapshot;
  }

  getTcpEventsForFlow(element: GrpcStraggler) {
    const events = [];
    const tcpEvents = this.tcpEventsMap.get(
      normalizeIdForMatching(element.chunk_id),
    );

    if (tcpEvents) {
      if (tcpEvents.sendmsg && tcpEvents.sendmsg.args) {
        events.push({
          name: 'SENDMSG',
          snapshot: tcpEvents.sendmsg.args as TcpStatsSnapshot,
        });
      }
      if (tcpEvents.scheduled && tcpEvents.scheduled.args) {
        events.push({
          name: 'SCHEDULED',
          snapshot: tcpEvents.scheduled.args as TcpStatsSnapshot,
        });
      }
      if (tcpEvents.sent && tcpEvents.sent.args) {
        events.push({
          name: 'SENT',
          snapshot: tcpEvents.sent.args as TcpStatsSnapshot,
        });
      }
      if (tcpEvents.ack && tcpEvents.ack.args) {
        events.push({
          name: 'ACK',
          snapshot: tcpEvents.ack.args as TcpStatsSnapshot,
        });
      }
    }
    return events;
  }

  toggleElement(element: GrpcStraggler) {
    this.expandedElement = this.expandedElement === element ? null : element;
    if (this.expandedElement === element) {
      if (!element.latencyBreakdownDataInfo) {
        element.latencyBreakdownDataInfo = prepareLatencyBreakdown(element);
      }
      if (!element.timelineInfo) {
        element.timelineInfo = prepareLatencyTimelineInfo(
          element,
          this.xorTimestampsMap.get(normalizeIdForMatching(element.chunk_id)),
          this.tcpEventsMap.get(normalizeIdForMatching(element.chunk_id)),
        );
      }
      const tcpEvents = this.getTcpEventsForFlow(element);
      if (tcpEvents.length > 0) {
        this.selectedTcpEventName = tcpEvents[0].name;
        this.selectedTcpEventSnapshot = tcpEvents[0].snapshot;
      } else {
        this.selectedTcpEventName = null;
        this.selectedTcpEventSnapshot = null;
      }
    }
    this.cdr.detectChanges();
  }

  rebindTableTools() {
    this.cdr.detectChanges();
  }

  getConnectionAndEndpoint(endpointId?: string): string {
    if (!endpointId) return 'Unknown Connection';
    const parsed = parseEndpointId(endpointId);
    return `connection_id:${parsed.connectionNum}, endpoint_id:${parsed.endpointNum}`;
  }

  private checkGoogleChartsLoaded() {
    const g = (window as WindowWithGoogle).google;
    if (g && g.charts) {
      this.isGoogleChartsLoaded = true;
      this.cdr.detectChanges();
    } else {
      setTimeout(() => {
        if (this.isDestroyed) return;
        this.checkGoogleChartsLoaded();
      }, 50);
    }
  }

  /**
   * Returns the warning shown next to the TCP latency of the chunk, or
   * undefined when the latency needs no warning.
   */
  protected tcpLatencyNote(element: GrpcStraggler): string | undefined {
    return getTcpLatencyNote(element);
  }

  getConnectionAddresses(
    element: GrpcStraggler,
  ): {local: string; peer: string} | null {
    const tcpEvents = this.tcpEventsMap.get(
      normalizeIdForMatching(element.chunk_id),
    );
    if (tcpEvents && tcpEvents.sendmsg && tcpEvents.sendmsg.args) {
      return {
        local: tcpEvents.sendmsg.args.local_addr || 'Unknown',
        peer: tcpEvents.sendmsg.args.peer_addr || 'Unknown',
      };
    }
    return null;
  }

  getTransferMetrics(
    transfer: TcpStatsSnapshot | null | undefined,
  ): Array<{key: string; value: unknown}> {
    if (!transfer) return [];
    const record = transfer as unknown as Record<string, unknown>;
    const excludedKeys = new Set([
      'local_addr',
      'peer_addr',
      'flow_id',
      'fathom_timestamp',
      'metadata_type',
    ]);
    return Object.keys(record)
      .filter((key) => !excludedKeys.has(key))
      .map((key) => {
        const formattedKey = key
          .replace(/_/g, ' ')
          .replace(/\b\w/g, (l) => l.toUpperCase());
        const val = record[key];
        const formattedValue = formatMetricValue(key, val);

        return {
          key: formattedKey,
          value: formattedValue,
        };
      });
  }

  isGraphDownsampled(host: string, series: string): boolean {
    return this.downsampledGraphs.has(`${host}:${series}`);
  }

  getOperationTag(element: GrpcStraggler): string {
    const hlo = element.hlo_collective_name || '';
    if (hlo.startsWith('recv') || hlo.startsWith('send')) {
      return hlo;
    }
    return 'recv-done';
  }

  getAddressDisplay(element: GrpcStraggler): string {
    if (element.peer_address) {
      return element.peer_address;
    }
    const addrs = this.getConnectionAddresses(element);
    if (addrs) {
      return `${addrs.local} -> ${addrs.peer}`;
    }
    return 'N/A';
  }
}
