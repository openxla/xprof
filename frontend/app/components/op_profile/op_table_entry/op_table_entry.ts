import {
  ChangeDetectionStrategy,
  Component,
  EventEmitter,
  inject,
  Input,
  OnChanges,
  OnDestroy,
  OnInit,
  Output,
  SimpleChanges,
} from '@angular/core';
import {Store} from '@ngrx/store';
import {type DiffNode} from 'org_xprof/frontend/app/common/interfaces/op_profile_diff';
import {
  formatServingBadge,
  formatServingTooltip,
  getServingMetadata,
  normalizeServingMetadata,
} from 'org_xprof/frontend/app/common/utils/serving_metadata_utils';
import * as utils from 'org_xprof/frontend/app/common/utils/utils';
import {DATA_SERVICE_INTERFACE_TOKEN} from 'org_xprof/frontend/app/services/data_service_v2/data_service_v2_interface';
import {updateSelectedOpNodeChainAction} from 'org_xprof/frontend/app/store/actions';
import {getOpAnalysisState} from 'org_xprof/frontend/app/store/selectors';
import {OpAnalysisState} from 'org_xprof/frontend/app/store/state';
import {type Node} from 'org_xprof/frontend/app/common/interfaces/op_profile.jsonpb_decls';
import {ReplaySubject} from 'rxjs';
import {takeUntil} from 'rxjs/operators';

function getHbmFraction(node?: Node, root?: Node): number {
  const hbmType = utils.MemBwType.MEM_BW_TYPE_HBM_RW;
  const nodeBytes = node?.metrics?.rawBytesAccessedArray;
  const rootBytes = root?.metrics?.rawBytesAccessedArray;
  if (
    nodeBytes &&
    rootBytes &&
    nodeBytes.length > hbmType &&
    rootBytes.length > hbmType &&
    rootBytes[hbmType] !== 0
  ) {
    return nodeBytes[hbmType] / rootBytes[hbmType];
  }
  return NaN;
}

/**
 * An op table entry view component.
 *
 * @deprecated Use MatTable in OpTable instead.
 */
@Component({
  changeDetection: ChangeDetectionStrategy.Default,
  standalone: false,
  selector: 'op-table-entry',
  templateUrl: './op_table_entry.ng.html',
  styleUrls: ['./op_table_entry.scss'],
})
export class OpTableEntry implements OnChanges, OnDestroy, OnInit {
  /** Handles on-destroy Subject, used to unsubscribe. */
  private readonly destroyed = new ReplaySubject<void>(1);
  private readonly dataService = inject(DATA_SERVICE_INTERFACE_TOKEN);
  private readonly store = inject(Store);

  /** The depth of node. */
  @Input() level = 0;

  /** The main node. */
  @Input() node?: Node;

  /** The root node. */
  @Input() rootNode?: Node;

  /** The session id. */
  @Input() sessionId = '';

  /** The list of modules. */
  @Input() moduleList: string[] = [];

  /** Session tags inherited from the parent table. */
  @Input() tags: string[] = [];

  /** The selected node. */
  @Input() selectedNode?: Node;

  /** The property to sort by waste time. */
  @Input() byWasted = false;

  /** The property to show top 90%. */
  @Input() showP90 = false;

  /** The number of children nodes to be shown. */
  @Input() childrenCount = 10;

  /**
   * The internal property used to react to changes of applyScalingFactor in the
   * nested op table entry.
   */
  @Input() applyScalingFactorInternal = false;

  /** The event when the mouse enter or leave. */
  @Output() readonly hover = new EventEmitter<Node | null>();

  /** The event when the selection is changed. */
  @Output() readonly selected = new EventEmitter<Node>();

  // TODO(xprof): rename the variable to be more self-explanatory or add
  // docstring
  children: Node[] = [];
  expanded = false;
  barWidth = '';
  flameColor = '';
  name = '';
  offset = '';
  percent = '';
  provenance = '-';
  timeWasted = '';
  hbmFraction = '';
  flopsUtilization = '';
  hbmUtilization = '';
  hbmFlameColor = '';
  numLeftOut = 0;
  applyScalingFactor = false;
  kernelViewerLink = utils.NO_KERNEL_VIEWER_LINK;
  programServingBadgeText = '';
  programServingTooltipText = '';
  programServingAccessibleText = '';

  constructor() {
    this.store
      .select(getOpAnalysisState)
      .pipe(takeUntil(this.destroyed))
      .subscribe((opAnalysisState: OpAnalysisState) => {
        this.applyScalingFactor = opAnalysisState.applyScalingFactor;
      });
  }

  asDiffNode(node?: Node): DiffNode | undefined {
    if (!node) return undefined;
    return node as DiffNode;
  }

  ngOnInit() {
    this.updateProperties();
  }

  ngOnChanges(changes: SimpleChanges) {
    this.updateProperties();
  }

  ngOnDestroy() {
    this.destroyed.next();
    this.destroyed.complete();
  }

  private updateKernelViewerLink() {
    this.kernelViewerLink = utils.getOpProfileKernelViewerLink({
      node: this.node,
      tags: this.tags,
      sessionId: this.sessionId,
      dataService: this.dataService,
      rootNode: this.rootNode,
      moduleList: this.moduleList,
    });
  }

  private updateProperties() {
    this.updateProgramServingMetadata();
    if (!this.node || !this.rootNode) {
      this.children = [];
      this.kernelViewerLink = utils.NO_KERNEL_VIEWER_LINK;
      return;
    }

    this.updateKernelViewerLink();

    if (this.level === 0) {
      this.expanded = true;
    }
    this.children = this.getChildren();
    this.numLeftOut = this.getLeftOut();

    const diffNode = this.node as DiffNode | undefined;
    const diffRoot = this.rootNode as DiffNode | undefined;
    const hasBaseline =
      !!diffNode?.baseline ||
      diffNode?.activeOnly ||
      diffNode?.baselineOnly ||
      !!diffRoot?.baseline;
    const baseRoot = diffRoot?.baseline || this.rootNode;

    const activeFraction =
      this.node && this.rootNode && this.node.metrics
        ? utils.timeFraction(this.node, this.rootNode)
        : 0;
    const baseFraction =
      hasBaseline && diffNode?.baseline && diffRoot?.baseline
        ? utils.timeFraction(diffNode.baseline, diffRoot.baseline)
        : 0;

    if (
      this.node &&
      this.rootNode &&
      (this.node.metrics || diffNode?.baselineOnly)
    ) {
      this.percent = this.formatMetricDiff(
        activeFraction,
        baseFraction,
        hasBaseline,
        diffNode,
      );
      this.barWidth = diffNode?.baselineOnly
        ? '0'
        : utils.percent(activeFraction);
    } else {
      this.barWidth = '0';
      this.percent = '';
    }

    const utilization = utils.flopsUtilization(
      this.node,
      this.rootNode,
      this.applyScalingFactor,
    );
    const baseUtilization =
      hasBaseline && diffNode?.baseline
        ? utils.flopsUtilization(
            diffNode.baseline,
            baseRoot,
            this.applyScalingFactor,
          )
        : NaN;
    this.flopsUtilization = this.formatMetricDiff(
      utilization,
      baseUtilization,
      hasBaseline,
      diffNode,
    );
    const colorUtilization = diffNode?.baselineOnly
      ? baseUtilization
      : utilization;
    this.flameColor = utils.flameColor(colorUtilization, 0.7, 1, Math.sqrt);

    this.name = this.node && this.node.name ? this.node.name : '';
    this.offset = this.level.toString() + 'em';
    this.provenance = utils.parseFrameworkOpType(this.node?.xla?.provenance);

    const activeWasted = utils.timeWasted(this.node, this.rootNode);
    const baseWasted =
      hasBaseline && diffNode?.baseline
        ? utils.timeWasted(diffNode.baseline, baseRoot)
        : NaN;
    this.timeWasted = this.formatMetricDiff(
      activeWasted,
      baseWasted,
      hasBaseline,
      diffNode,
    );

    const activeHbmFrac = getHbmFraction(this.node, this.rootNode);
    const baseHbmFrac =
      hasBaseline && diffNode?.baseline
        ? getHbmFraction(diffNode.baseline, baseRoot)
        : NaN;
    this.hbmFraction = this.formatMetricDiff(
      activeHbmFrac,
      baseHbmFrac,
      hasBaseline,
      diffNode,
      /* defaultValueIfNull= */ '',
    );

    const hbmUtilization = utils.memoryBandwidthUtilization(
      this.node,
      utils.MemBwType.MEM_BW_TYPE_HBM_RW,
    );
    const baseHbmUtil =
      hasBaseline && diffNode?.baseline
        ? utils.memoryBandwidthUtilization(
            diffNode.baseline,
            utils.MemBwType.MEM_BW_TYPE_HBM_RW,
          )
        : NaN;
    this.hbmUtilization = this.formatMetricDiff(
      hbmUtilization,
      baseHbmUtil,
      hasBaseline,
      diffNode,
    );
    const colorHbmUtil = diffNode?.baselineOnly ? baseHbmUtil : hbmUtilization;
    this.hbmFlameColor = utils.bwColor(colorHbmUtil);
  }

  /** Builds the serving label, its hover text, and its screen reader text. */
  private updateProgramServingMetadata() {
    this.programServingBadgeText = '';
    this.programServingTooltipText = '';
    this.programServingAccessibleText = '';
    if (!this.node || !this.rootNode) {
      return;
    }

    const normalized = normalizeServingMetadata(getServingMetadata(this.node));
    if (!normalized) {
      return;
    }

    this.programServingBadgeText = formatServingBadge(normalized);
    this.programServingTooltipText = formatServingTooltip(normalized);
    this.programServingAccessibleText = ` (${this.programServingTooltipText})`;
  }

  /**
   * Formats a percentage diff string for active, baseline, added, or removed ops.
   */
  private formatMetricDiff(
    activeVal: number,
    baseVal: number,
    hasBaseline: boolean,
    diffNode?: DiffNode,
    defaultValueIfNull = '-',
  ): string {
    if (hasBaseline) {
      if (diffNode?.activeOnly) {
        return !isNaN(activeVal)
          ? `${(activeVal * 100).toFixed(2)}% (Added)`
          : '-';
      } else if (diffNode?.baselineOnly) {
        return !isNaN(baseVal)
          ? `0.00% (base: ${(baseVal * 100).toFixed(2)}%)`
          : '-';
      } else if (!isNaN(activeVal) && !isNaN(baseVal)) {
        const activePct = activeVal * 100;
        const basePct = baseVal * 100;
        const diffPct = activePct - basePct;
        const deltaStr =
          diffPct >= 0 ? `+${diffPct.toFixed(2)}%` : `${diffPct.toFixed(2)}%`;
        return `${activePct.toFixed(2)}% (${deltaStr})`;
      }
    }
    return utils.percent(activeVal, defaultValueIfNull);
  }

  private get90ChildrenIndex() {
    if (
      !this.showP90 ||
      !this.node ||
      !this.rootNode ||
      !this.node.children ||
      this.node.children.length === 0 ||
      !this.node.metrics ||
      !this.node.metrics.rawTime
    ) {
      return this.childrenCount;
    }

    let tot = 0;
    const targetP90NodeRawTimePs = this.node.metrics.rawTime * 0.9;
    const targetCount = Math.min(this.childrenCount, this.node.children.length);
    for (let i = 0; i < targetCount; i++) {
      if (tot >= targetP90NodeRawTimePs) {
        return i;
      }
      const child = this.node.children[i];
      if (child && child.metrics && child.metrics.rawTime) {
        tot += child.metrics.rawTime;
      }
    }
    return this.childrenCount;
  }

  private getChildren(): Node[] {
    if (!this.node || !this.node.children || !this.rootNode) {
      return [];
    }
    let children: Node[] = this.node.children.slice();
    if (this.byWasted && this.rootNode) {
      children.sort((a, b) => {
        const timeWastedA = utils.timeWasted(a, this.rootNode!);
        const timeWastedB = utils.timeWasted(b, this.rootNode!);
        if (isNaN(timeWastedA)) {
          return 1;
        } else if (isNaN(timeWastedB)) {
          return -1;
        }
        return (
          utils.timeWasted(b, this.rootNode!) -
          utils.timeWasted(a, this.rootNode!)
        );
      });
    }
    const k = this.get90ChildrenIndex();

    children = this.level ? children.slice(0, k) : children;

    return children;
  }

  private getLeftOut(): number {
    if (!this.level || !this.node || !this.node.numChildren) return 0;
    return (
      this.node.numChildren - Math.min(this.childrenCount, this.children.length)
    );
  }

  onSelect($event: Node) {
    this.selected.emit($event);
    this.store.dispatch(
      updateSelectedOpNodeChainAction({
        selectedOpNodeName: this.node?.name,
      }),
    );
  }

  toggleExpanded() {
    this.expanded = !this.expanded;
    this.selected.emit(this.node);
  }
}
