import {
  ChangeDetectionStrategy,
  Component,
  OnInit,
  effect,
  forwardRef,
  inject,
  input,
  output,
} from '@angular/core';
import {Store} from '@ngrx/store';
import {type DiffNode} from 'org_xprof/frontend/app/common/interfaces/op_profile_diff';
import * as utils from 'org_xprof/frontend/app/common/utils/utils';
import {updateSelectedOpNodeChainAction} from 'org_xprof/frontend/app/store/actions';
import {getOpAnalysisState} from 'org_xprof/frontend/app/store/selectors';
import {OpAnalysisState} from 'org_xprof/frontend/app/store/state';
import {type Node} from 'org_xprof/frontend/app/common/interfaces/op_profile.jsonpb_decls';
import {ReplaySubject} from 'rxjs';
import {takeUntil} from 'rxjs/operators';

/** An op table entry view component. */
@Component({
  changeDetection: ChangeDetectionStrategy.Default,
  selector: 'op-table-entry',
  templateUrl: './op_table_entry.ng.html',
  styleUrls: ['./op_table_entry.scss'],
  imports: [forwardRef(() => OpTableEntry)],
})
export class OpTableEntry implements OnInit {
  private readonly store = inject<Store<{}>>(Store);

  /** Handles on-destroy Subject, used to unsubscribe. */
  private readonly destroyed = new ReplaySubject<void>(1);

  /** The depth of node. */
  readonly level = input(0);

  /** The main node. */
  readonly node = input<Node>();

  /** The root node. */
  readonly rootNode = input<Node>();

  /** The selected node. */
  readonly selectedNode = input<Node>();

  /** The property to sort by waste time. */
  readonly byWasted = input(false);

  /** The property to show top 90%. */
  readonly showP90 = input(false);

  /** The number of children nodes to be shown. */
  readonly childrenCount = input(10);

  /**
   * The internal property used to react to changes of applyScalingFactor in the
   * nested op table entry.
   */
  readonly applyScalingFactorInternal = input(false);

  /** The event when the mouse enter or leave. */
  readonly hover = output<Node | undefined>();

  /** The event when the selection is changed. */
  readonly selected = output<Node>();

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

  constructor() {
    this.store
      .select(getOpAnalysisState)
      .pipe(takeUntil(this.destroyed))
      .subscribe((opAnalysisState: OpAnalysisState) => {
        this.applyScalingFactor = opAnalysisState.applyScalingFactor;
      });

    effect(() => {
      this.level();
      this.node();
      this.rootNode();
      this.selectedNode();
      this.byWasted();
      this.showP90();
      this.childrenCount();
      this.applyScalingFactorInternal();
      this.updateProperties();
    });
  }

  asDiffNode(node?: Node): DiffNode | undefined {
    if (!node) return undefined;
    return node as DiffNode;
  }

  ngOnInit() {
    this.updateProperties();
  }

  private updateProperties() {
    const node = this.node();
    const rootNode = this.rootNode();
    const level = this.level();
    if (!node || !rootNode) {
      this.children = [];
      return;
    }

    if (level === 0) {
      this.expanded = true;
    }
    this.children = this.getChildren();
    this.numLeftOut = this.getLeftOut();

    const diffNode = node as DiffNode | undefined;
    const diffRoot = rootNode as DiffNode | undefined;
    const hasBaseline =
      !!diffNode?.baseline ||
      diffNode?.activeOnly ||
      diffNode?.baselineOnly ||
      !!diffRoot?.baseline;

    const activeFraction =
      node && rootNode && node.metrics ? utils.timeFraction(node, rootNode) : 0;
    const baseFraction =
      hasBaseline && diffNode?.baseline && diffRoot?.baseline
        ? utils.timeFraction(diffNode.baseline, diffRoot.baseline)
        : 0;

    if (node && rootNode && (node.metrics || diffNode?.baselineOnly)) {
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
      node,
      rootNode,
      this.applyScalingFactor,
    );
    let baseUtilization = NaN;
    if (hasBaseline && diffNode?.baseline) {
      baseUtilization = utils.flopsUtilization(
        diffNode.baseline,
        diffRoot?.baseline || rootNode,
        this.applyScalingFactor,
      );
    }
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

    this.name = node && node.name ? node.name : '';
    this.offset = level.toString() + 'em';
    this.provenance = utils.parseFrameworkOpType(node?.xla?.provenance);

    const activeWasted = utils.timeWasted(node, rootNode);
    let baseWasted = NaN;
    if (hasBaseline && diffNode?.baseline) {
      baseWasted = utils.timeWasted(
        diffNode.baseline,
        diffRoot?.baseline || rootNode,
      );
    }
    this.timeWasted = this.formatMetricDiff(
      activeWasted,
      baseWasted,
      hasBaseline,
      diffNode,
    );

    const hbmType = utils.MemBwType.MEM_BW_TYPE_HBM_RW;
    let activeHbmFrac = NaN;
    if (
      node?.metrics?.rawBytesAccessedArray &&
      rootNode?.metrics?.rawBytesAccessedArray &&
      node.metrics.rawBytesAccessedArray.length > hbmType &&
      rootNode.metrics.rawBytesAccessedArray.length > hbmType &&
      rootNode.metrics.rawBytesAccessedArray[hbmType] !== 0
    ) {
      activeHbmFrac =
        node.metrics.rawBytesAccessedArray[hbmType] /
        rootNode.metrics.rawBytesAccessedArray[hbmType];
    }

    let baseHbmFrac = NaN;
    if (hasBaseline && diffNode?.baseline) {
      const baseRoot = diffRoot?.baseline || rootNode;
      if (
        diffNode.baseline.metrics?.rawBytesAccessedArray &&
        baseRoot?.metrics?.rawBytesAccessedArray &&
        diffNode.baseline.metrics.rawBytesAccessedArray.length > hbmType &&
        baseRoot.metrics.rawBytesAccessedArray.length > hbmType &&
        baseRoot.metrics.rawBytesAccessedArray[hbmType] !== 0
      ) {
        baseHbmFrac =
          diffNode.baseline.metrics.rawBytesAccessedArray[hbmType] /
          baseRoot.metrics.rawBytesAccessedArray[hbmType];
      }
    }
    this.hbmFraction = this.formatMetricDiff(
      activeHbmFrac,
      baseHbmFrac,
      hasBaseline,
      diffNode,
      /* defaultValueIfNull= */ '',
    );

    const hbmUtilization = utils.memoryBandwidthUtilization(
      node,
      utils.MemBwType.MEM_BW_TYPE_HBM_RW,
    );
    let baseHbmUtil = NaN;
    if (hasBaseline && diffNode?.baseline) {
      baseHbmUtil = utils.memoryBandwidthUtilization(
        diffNode.baseline,
        utils.MemBwType.MEM_BW_TYPE_HBM_RW,
      );
    }
    this.hbmUtilization = this.formatMetricDiff(
      hbmUtilization,
      baseHbmUtil,
      hasBaseline,
      diffNode,
    );
    const colorHbmUtil = diffNode?.baselineOnly ? baseHbmUtil : hbmUtilization;
    this.hbmFlameColor = utils.bwColor(colorHbmUtil);
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
    const node = this.node();
    const rootNode = this.rootNode();
    if (
      !this.showP90() ||
      !node ||
      !rootNode ||
      !node.children ||
      node.children.length === 0 ||
      !node.metrics ||
      !node.metrics.rawTime
    ) {
      return this.childrenCount();
    }

    let tot = 0;
    const targetP90NodeRawTimePs = node.metrics.rawTime * 0.9;
    const targetCount = Math.min(this.childrenCount(), node.children.length);
    for (let i = 0; i < targetCount; i++) {
      if (tot >= targetP90NodeRawTimePs) {
        return i;
      }
      const child = node.children[i];
      if (child && child.metrics && child.metrics.rawTime) {
        tot += child.metrics.rawTime;
      }
    }
    return this.childrenCount();
  }

  private getChildren(): Node[] {
    const node = this.node();
    const rootNode = this.rootNode();
    if (!node || !node.children || !rootNode) {
      return [];
    }
    let children: Node[] = node.children.slice();
    if (this.byWasted() && rootNode) {
      children.sort((a, b) => {
        const timeWastedA = utils.timeWasted(a, rootNode);
        const timeWastedB = utils.timeWasted(b, rootNode);
        if (isNaN(timeWastedA)) {
          return 1;
        } else if (isNaN(timeWastedB)) {
          return -1;
        }
        return utils.timeWasted(b, rootNode) - utils.timeWasted(a, rootNode);
      });
    }
    const k = this.get90ChildrenIndex();

    children = this.level() ? children.slice(0, k) : children;

    return children;
  }

  private getLeftOut(): number {
    const node = this.node();
    if (!this.level() || !node || !node.numChildren) return 0;
    return (
      node.numChildren - Math.min(this.childrenCount(), this.children.length)
    );
  }

  onSelect($event: Node) {
    this.selected.emit($event);
    this.store.dispatch(
      updateSelectedOpNodeChainAction({
        selectedOpNodeName: this.node()?.name,
      }),
    );
  }

  toggleExpanded() {
    this.expanded = !this.expanded;
    const node = this.node();
    if (node) {
      this.selected.emit(node);
    }
  }
}
