import {Component, EventEmitter, Input, OnChanges, OnInit, Output, SimpleChanges, ChangeDetectionStrategy} from '@angular/core';
import {Store} from '@ngrx/store';
import * as utils from 'org_xprof/frontend/app/common/utils/utils';
import {updateSelectedOpNodeChainAction} from 'org_xprof/frontend/app/store/actions';
import {getOpAnalysisState} from 'org_xprof/frontend/app/store/selectors';
import {OpAnalysisState} from 'org_xprof/frontend/app/store/state';
import {type Node} from 'org_xprof/frontend/app/common/interfaces/op_profile.jsonpb_decls';
import {ReplaySubject} from 'rxjs';
import {takeUntil} from 'rxjs/operators';
import {type DiffNode} from 'org_xprof/frontend/app/common/interfaces/op_profile_diff';

/** An op table entry view component. */
@Component({
  changeDetection: ChangeDetectionStrategy.Default,standalone: false,
  selector: 'op-table-entry',
  templateUrl: './op_table_entry.ng.html',
  styleUrls: ['./op_table_entry.scss']
})
export class OpTableEntry implements OnChanges, OnInit {
  /** Handles on-destroy Subject, used to unsubscribe. */
  private readonly destroyed = new ReplaySubject<void>(1);

  /** The depth of node. */
  @Input() level = 0;

  /** The main node. */
  @Input() node?: Node;

  /** The root node. */
  @Input() rootNode?: Node;

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
  @Output() readonly hover = new EventEmitter<Node|null>();

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

  constructor(private readonly store: Store<{}>) {
    this.store.select(getOpAnalysisState)
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

  private updateProperties() {
    if (!this.node || !this.rootNode) {
      this.children = [];
      return;
    }

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

    const activeFraction =
      this.node && this.rootNode && this.node.metrics
        ? utils.timeFraction(this.node, this.rootNode)
        : 0;
    const activePct = activeFraction * 100;

    if (
      this.node &&
      this.rootNode &&
      (this.node.metrics || diffNode?.baselineOnly)
    ) {
      if (hasBaseline) {
        const baseFraction =
          diffNode?.baseline && diffRoot?.baseline
            ? utils.timeFraction(diffNode.baseline, diffRoot.baseline)
            : 0;
        const basePct = baseFraction * 100;
        const diffPct = activePct - basePct;
        const deltaStr =
          diffPct >= 0 ? `+${diffPct.toFixed(2)}%` : `${diffPct.toFixed(2)}%`;

        if (diffNode?.activeOnly) {
          this.percent = `${activePct.toFixed(2)}% (Added)`;
          this.barWidth = utils.percent(activeFraction);
        } else if (diffNode?.baselineOnly) {
          this.percent = `0.00% (base: ${basePct.toFixed(2)}%)`;
          this.barWidth = '0';
        } else {
          this.percent = `${activePct.toFixed(2)}% (${deltaStr})`;
          this.barWidth = utils.percent(activeFraction);
        }
      } else {
        this.percent = utils.percent(activeFraction);
        this.barWidth = utils.percent(activeFraction);
      }
    } else {
      this.barWidth = '0';
      this.percent = '';
    }

    const utilization = utils.flopsUtilization(
        this.node, this.rootNode, this.applyScalingFactor);

    let baseUtilization = NaN;
    if (hasBaseline && diffNode?.baseline) {
      baseUtilization = utils.flopsUtilization(
        diffNode.baseline,
        diffRoot?.baseline || this.rootNode,
        this.applyScalingFactor,
      );
    }

    if (hasBaseline) {
      if (diffNode?.activeOnly) {
        this.flopsUtilization = !isNaN(utilization)
          ? `${(utilization * 100).toFixed(2)}% (Added)`
          : '-';
        this.flameColor = utils.flameColor(utilization, 0.7, 1, Math.sqrt);
      } else if (diffNode?.baselineOnly) {
        this.flopsUtilization = !isNaN(baseUtilization)
          ? `0.00% (base: ${(baseUtilization * 100).toFixed(2)}%)`
          : '-';
        this.flameColor = utils.flameColor(baseUtilization, 0.7, 1, Math.sqrt);
      } else if (!isNaN(utilization) && !isNaN(baseUtilization)) {
        const activeFlopsPct = utilization * 100;
        const baseFlopsPct = baseUtilization * 100;
        const diffFlopsPct = activeFlopsPct - baseFlopsPct;
        const deltaStr =
          diffFlopsPct >= 0
            ? `+${diffFlopsPct.toFixed(2)}%`
            : `${diffFlopsPct.toFixed(2)}%`;
        this.flopsUtilization = `${activeFlopsPct.toFixed(2)}% (${deltaStr})`;
        this.flameColor = utils.flameColor(utilization, 0.7, 1, Math.sqrt);
      } else {
        this.flopsUtilization = utils.percent(utilization);
        this.flameColor = utils.flameColor(utilization, 0.7, 1, Math.sqrt);
      }
    } else {
      this.flopsUtilization = utils.percent(utilization);
      this.flameColor = utils.flameColor(utilization, 0.7, 1, Math.sqrt);
    }

    this.name = (this.node && this.node.name) ? this.node.name : '';
    this.offset = this.level.toString() + 'em';
    this.provenance = utils.parseFrameworkOpType(this.node?.xla?.provenance);

    const activeWasted = utils.timeWasted(this.node, this.rootNode);
    let baseWasted = NaN;
    if (hasBaseline && diffNode?.baseline) {
      baseWasted = utils.timeWasted(
        diffNode.baseline,
        diffRoot?.baseline || this.rootNode,
      );
    }

    if (hasBaseline) {
      if (diffNode?.activeOnly) {
        this.timeWasted = !isNaN(activeWasted)
          ? `${(activeWasted * 100).toFixed(2)}% (Added)`
          : '-';
      } else if (diffNode?.baselineOnly) {
        this.timeWasted = !isNaN(baseWasted)
          ? `0.00% (base: ${(baseWasted * 100).toFixed(2)}%)`
          : '-';
      } else if (!isNaN(activeWasted) && !isNaN(baseWasted)) {
        const activeWastedPct = activeWasted * 100;
        const baseWastedPct = baseWasted * 100;
        const diffWastedPct = activeWastedPct - baseWastedPct;
        const deltaStr =
          diffWastedPct >= 0
            ? `+${diffWastedPct.toFixed(2)}%`
            : `${diffWastedPct.toFixed(2)}%`;
        this.timeWasted = `${activeWastedPct.toFixed(2)}% (${deltaStr})`;
      } else {
        this.timeWasted = utils.percent(activeWasted);
      }
    } else {
      this.timeWasted = utils.percent(activeWasted);
    }

    const hbmType = utils.MemBwType.MEM_BW_TYPE_HBM_RW;
    let activeHbmFrac = NaN;
    if (
      this.node?.metrics?.rawBytesAccessedArray &&
      this.rootNode?.metrics?.rawBytesAccessedArray &&
      this.node.metrics.rawBytesAccessedArray.length > hbmType &&
      this.rootNode.metrics.rawBytesAccessedArray.length > hbmType &&
      this.rootNode.metrics.rawBytesAccessedArray[hbmType] !== 0
    ) {
      activeHbmFrac =
        this.node.metrics.rawBytesAccessedArray[hbmType] /
        this.rootNode.metrics.rawBytesAccessedArray[hbmType];
    }

    let baseHbmFrac = NaN;
    if (hasBaseline && diffNode?.baseline) {
      const baseRoot = diffRoot?.baseline || this.rootNode;
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

    if (hasBaseline) {
      if (diffNode?.activeOnly) {
        this.hbmFraction = !isNaN(activeHbmFrac)
          ? `${(activeHbmFrac * 100).toFixed(2)}% (Added)`
          : '-';
      } else if (diffNode?.baselineOnly) {
        this.hbmFraction = !isNaN(baseHbmFrac)
          ? `0.00% (base: ${(baseHbmFrac * 100).toFixed(2)}%)`
          : '-';
      } else if (!isNaN(activeHbmFrac) && !isNaN(baseHbmFrac)) {
        const activeHbmFracPct = activeHbmFrac * 100;
        const baseHbmFracPct = baseHbmFrac * 100;
        const diffHbmFracPct = activeHbmFracPct - baseHbmFracPct;
        const deltaStr =
          diffHbmFracPct >= 0
            ? `+${diffHbmFracPct.toFixed(2)}%`
            : `${diffHbmFracPct.toFixed(2)}%`;
        this.hbmFraction = `${activeHbmFracPct.toFixed(2)}% (${deltaStr})`;
      } else {
        this.hbmFraction = isNaN(activeHbmFrac)
          ? ''
          : utils.percent(activeHbmFrac);
      }
    } else {
      this.hbmFraction = isNaN(activeHbmFrac)
        ? ''
        : utils.percent(activeHbmFrac);
    }

    const hbmUtilization = utils.memoryBandwidthUtilization(
      this.node,
      utils.MemBwType.MEM_BW_TYPE_HBM_RW,
    );
    let baseHbmUtil = NaN;
    if (hasBaseline && diffNode?.baseline) {
      baseHbmUtil = utils.memoryBandwidthUtilization(
        diffNode.baseline,
        utils.MemBwType.MEM_BW_TYPE_HBM_RW,
      );
    }

    if (hasBaseline) {
      if (diffNode?.activeOnly) {
        this.hbmUtilization = !isNaN(hbmUtilization)
          ? `${(hbmUtilization * 100).toFixed(2)}% (Added)`
          : '-';
        this.hbmFlameColor = utils.bwColor(hbmUtilization);
      } else if (diffNode?.baselineOnly) {
        this.hbmUtilization = !isNaN(baseHbmUtil)
          ? `0.00% (base: ${(baseHbmUtil * 100).toFixed(2)}%)`
          : '-';
        this.hbmFlameColor = utils.bwColor(baseHbmUtil);
      } else if (!isNaN(hbmUtilization) && !isNaN(baseHbmUtil)) {
        const activeHbmPct = hbmUtilization * 100;
        const baseHbmPct = baseHbmUtil * 100;
        const diffHbmPct = activeHbmPct - baseHbmPct;
        const deltaStr =
          diffHbmPct >= 0
            ? `+${diffHbmPct.toFixed(2)}%`
            : `${diffHbmPct.toFixed(2)}%`;
        this.hbmUtilization = `${activeHbmPct.toFixed(2)}% (${deltaStr})`;
        this.hbmFlameColor = utils.bwColor(hbmUtilization);
      } else {
        this.hbmUtilization = utils.percent(hbmUtilization);
        this.hbmFlameColor = utils.bwColor(hbmUtilization);
      }
    } else {
      this.hbmUtilization = utils.percent(hbmUtilization);
      this.hbmFlameColor = utils.bwColor(hbmUtilization);
    }
  }

  private get90ChildrenIndex() {
    if (!this.showP90 || !this.node || !this.rootNode || !this.node.children ||
        this.node.children.length === 0 || !this.node.metrics ||
        !this.node.metrics.rawTime) {
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
    let children: Node[]  = this.node.children.slice();
    if (this.byWasted && this.rootNode) {
      children.sort(
          (a, b) => {
        const timeWastedA = utils.timeWasted(a, this.rootNode!);
        const timeWastedB = utils.timeWasted(b, this.rootNode!);
        if (isNaN(timeWastedA)) {
          return 1;
        } else if (isNaN(timeWastedB)) {
          return -1;
        }
            return utils.timeWasted(b, this.rootNode!) -
              utils.timeWasted(a, this.rootNode!);
      });
    }
    const k = this.get90ChildrenIndex();

    children = this.level ? children.slice(0, k) : children;

    return children;
  }

  private getLeftOut(): number {
    if (!this.level || !this.node || !this.node.numChildren) return 0;
    return this.node.numChildren -
        Math.min(this.childrenCount, this.children.length);
  }

  onSelect($event: Node) {
    this.selected.emit($event);
    this.store.dispatch(updateSelectedOpNodeChainAction({
        selectedOpNodeName: this.node?.name,
    }));
  }

  toggleExpanded() {
    this.expanded = !this.expanded;
    this.selected.emit(this.node);
  }
}
