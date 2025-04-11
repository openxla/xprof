import {Component, EventEmitter, inject, Input, Output} from '@angular/core';
import {Store} from '@ngrx/store';
import {Node} from 'org_xprof/frontend/app/common/interfaces/op_profile.jsonpb_decls';
import {NavigationEvent} from 'org_xprof/frontend/app/common/interfaces/navigation_event';
import * as utils from 'org_xprof/frontend/app/common/utils/utils';
import {DATA_SERVICE_INTERFACE_TOKEN} from 'org_xprof/frontend/app/services/data_service_v2/data_service_v2_interface';
import {getActiveOpProfileNodeState, getCurrentRun, getOpProfileRootNode, getProfilingGeneralState, getSelectedOpNodeChainState} from 'org_xprof/frontend/app/store/selectors';
import {ProfilingGeneralState} from 'org_xprof/frontend/app/store/state';
import {Observable, ReplaySubject} from 'rxjs';
import {takeUntil} from 'rxjs/operators';

/** An op details view component. */
@Component({
  standalone: false,
  selector: 'op-details',
  templateUrl: './op_details.ng.html',
  styleUrls: ['./op_details.scss']
})
export class OpDetails {
  /** Handles on-destroy Subject, used to unsubscribe. */
  private readonly destroyed = new ReplaySubject<void>(1);
  private readonly dataService = inject(DATA_SERVICE_INTERFACE_TOKEN);

  /** When updating app route to other tools through crosslink */
  @Output() readonly updateRoute = new EventEmitter<NavigationEvent>();

  /** the session id */
  @Input() sessionId = '';
  /** the list of modules */
  @Input() moduleList: string[] = [];

  currentRun$: Observable<string>;

  rootNode?: Node;
  node?: Node;
  color: string = '';
  name: string = '';
  subheader: string = '';
  flopsRate: string = '';
  bf16FlopsRate = '';
  flopsUtilization: string = '';
  flopsColor: string = '';
  bandwidths: string[] =
      Array.from<string>({length: utils.MemBwType.MEM_BW_TYPE_MAX + 1})
          .fill('');
  bandwidthUtilizations: string[] =
      Array.from<string>({length: utils.MemBwType.MEM_BW_TYPE_MAX + 1})
          .fill('');
  bwColors: string[] =
      Array.from<string>({length: utils.MemBwType.MEM_BW_TYPE_MAX + 1})
          .fill('');
  programId = '';
  expression: string = '';
  provenance: string = '';
  rawTimeMs = '';
  occurrences = 0;
  avgTime = '';
  fused: boolean = false;
  hasCategory: boolean = false;
  hasLayout: boolean = false;
  dimensions: Node.XLAInstruction.LayoutAnalysis.Dimension[] = [];
  computationPrimitiveSize: string = '';
  selectedOpNodeChain: string[] = [];
  memBwType = utils.MemBwType;
  currentRun = '';
  showUtilizationWarning = false;
  deviceType = 'TPU';

  constructor(
      private readonly store: Store<{}>,
  ) {
    this.currentRun$ =
        this.store.select(getCurrentRun).pipe(takeUntil(this.destroyed));
    this.store.select(getActiveOpProfileNodeState)
        .pipe(takeUntil(this.destroyed))
        .subscribe((node: Node|null) => {
          this.update(node);
        });
    this.store.select(getSelectedOpNodeChainState)
        .pipe(takeUntil(this.destroyed))
        .subscribe((nodeChain: string[]) => {
          this.selectedOpNodeChain = nodeChain;
        });
    this.store.select(getOpProfileRootNode)
        .pipe(takeUntil(this.destroyed))
        .subscribe((node: Node|null) => {
          this.rootNode = node || undefined;
        });
    this.store.select(getProfilingGeneralState)
        .pipe(takeUntil(this.destroyed))
        .subscribe((generalState: ProfilingGeneralState|null) => {
          this.deviceType = (generalState && generalState.deviceType) ?
              generalState.deviceType :
              'TPU';
        });

    this.currentRun$.subscribe(run => {
      if (run) {
        this.currentRun = run;
      }
    });
  }

  getTitleByDeviceType(titlePrefix: string, titleSuffix: string) {
    if (this.deviceType === 'GPU') {
      return `${titlePrefix} (per gpu)${titleSuffix}`;
    } else if (this.deviceType === 'TPU') {
      return `${titlePrefix} (per core)${titleSuffix}`;
    } else {
      return `${titlePrefix}${titleSuffix}`;
    }
  }

  // expression format assumption: '%<op_name> = ...'
  get selectedOpName() {
    return this.expression.split('=')[0].trim().slice(1);
  }

  get selectedModuleName() {
    const aggregatedBy = this.selectedOpNodeChain[0];
    // 'by_program' or 'by_category'
    return aggregatedBy === 'by_program' ? this.selectedOpNodeChain[1] : '';
  }

  get graphViewerLink() {
    if (this.selectedModuleName) {
      return this.dataService.getGraphViewerLink(
          this.sessionId, this.selectedModuleName, this.selectedOpName, '');
    }
    if (this.programId) {
      return this.dataService.getGraphViewerLink(
          this.sessionId, '', this.selectedOpName, this.programId);
    }
    return '';
  }

  get isValidCustomCall() {
    return this.getCustomCallTextLink() !== '';
  }

  getCustomCallTextLink() {
    return this.dataService.getCustomCallTextLink(
        this.sessionId, this.selectedModuleName, this.selectedOpName);
  }

  dimensionColor(dimension?: Node.XLAInstruction.LayoutAnalysis.Dimension):
      string {
    if (!dimension || !dimension.alignment) {
      return '';
    }
    const ratio = (dimension.size || 0) / dimension.alignment;
    // Colors should grade harshly: 50% in a dimension is already very bad.
    const harshCurve = (x: number) => 1 - Math.sqrt(1 - x);
    return utils.flameColor(ratio / Math.ceil(ratio), 1, 0.25, harshCurve);
  }

  dimensionHint(dimension?: Node.XLAInstruction.LayoutAnalysis.Dimension):
      string {
    if (!dimension || !dimension.alignment) {
      return '';
    }
    const size = dimension.size || 0;
    const mul = Math.ceil(size / dimension.alignment);
    const mulSuffix = (mul === 1) ?
        '' :
        ': ' + mul.toString() + ' x ' + dimension.alignment.toString();
    if (size % dimension.alignment === 0) {
      return 'Exact fit' + mulSuffix;
    }
    return 'Pad to ' + (mul * dimension.alignment).toString() + mulSuffix;
  }

  private getSubheader(): string {
    if (!this.node) {
      return '';
    }
    if (this.node.xla && this.node.xla.category) {
      return this.node.xla.category + ' operation';
    }
    if (this.node.category) {
      return 'Operation category';
    }
    return 'Unknown';
  }

  update(node: Node|null) {
    this.node = node || undefined;
    if (!this.node || !this.rootNode) {
      return;
    }
    this.showUtilizationWarning = false;
    this.color = utils.flameColor(
        utils.flopsUtilization(this.node, this.rootNode), 0.7, 1, Math.sqrt);
    this.name = this.node.name || '';
    this.subheader = this.getSubheader();

    if (utils.hasFlopsUtilization(this.node)) {
      const flopsUtilization = utils.flopsUtilization(this.node, this.rootNode);
      if (flopsUtilization === 1) {
        this.showUtilizationWarning = true;
      }
      this.flopsUtilization = utils.percent(flopsUtilization, '');
      this.flopsColor = utils.flopsColor(flopsUtilization);
    } else {
      this.flopsUtilization = '';
    }

    const flopsRate = utils.flopsRate(this.node);
    const bf16FlopsRate = utils.normalizeToBf16FlopsRate(this.node);
    if (isNaN(flopsRate)) {
      this.flopsRate = '';
      this.bf16FlopsRate = '';
    } else {
      this.flopsRate = utils.humanReadableText(
          flopsRate, {si: true, dp: 2, suffix: 'FLOP/s'});
      this.bf16FlopsRate = utils.humanReadableText(
          bf16FlopsRate, {si: true, dp: 2, suffix: 'FLOP/s'});
    }

    for (let i = utils.MemBwType.MEM_BW_TYPE_FIRST;
         i <= utils.MemBwType.MEM_BW_TYPE_MAX; i++) {
      if (utils.hasBandwidthUtilization(this.node, i)) {
        const utilization = utils.memoryBandwidthUtilization(this.node, i);
        if (utilization === 1) {
          this.showUtilizationWarning = true;
        }
        this.bandwidthUtilizations[i] = utils.percent(utilization, '');
        this.bwColors[i] = utils.bwColor(utilization);
      } else {
        this.bandwidthUtilizations[i] = '';
      }
      const memoryBW = utils.memoryBandwidth(this.node, i);
      if (isNaN(memoryBW)) {
        this.bandwidths[i] = '';
      } else {
        this.bandwidths[i] =
            utils.humanReadableText(memoryBW, {si: true, dp: 2, suffix: 'B/s'});
      }
    }

    this.programId = this.node.xla?.programId || '';
    this.expression = this.node.xla?.expression || '';
    this.provenance = this.node.xla?.provenance || '';

    if (this.node.metrics && this.node.metrics.rawTime) {
      this.rawTimeMs = utils.humanReadableText(
          this.node.metrics.rawTime / 1e9, {si: true, dp: 2, suffix: ' ms'});
    } else {
      this.rawTimeMs = '';
    }
    this.occurrences = this.node.metrics?.occurrences || 0;

    if (this.node.metrics && this.node.metrics.avgTimePs) {
      this.avgTime = utils.formatDurationPs(this.node.metrics.avgTimePs);
    } else {
      this.avgTime = '';
    }

    this.fused = !!this.node.xla && !this.node.metrics;
    this.hasCategory = !!this.node.category;
    this.hasLayout = !!this.node.xla && !!this.node.xla.layout &&
        !!this.node.xla.layout.dimensions &&
        this.node.xla.layout.dimensions.length > 0;
    if (this.node.xla && this.node.xla.layout) {
      this.dimensions = this.node.xla.layout.dimensions || [];
    }

    this.computationPrimitiveSize =
        ((this.node?.xla?.computationPrimitiveSize) ?
             `${this.node.xla.computationPrimitiveSize} bits` :
             '');
  }

  ngOnDestroy() {
    // Unsubscribes all pending subscriptions.
    this.destroyed.next();
    this.destroyed.complete();
  }
}
