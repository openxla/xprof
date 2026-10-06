import {ChangeDetectionStrategy, Component, inject, Input, OnChanges, OnDestroy, OnInit, SimpleChanges} from '@angular/core';
import {MatTableDataSource} from '@angular/material/table';
import {Store} from '@ngrx/store';
import {ReplaySubject} from 'rxjs';
import {takeUntil} from 'rxjs/operators';
import {type DiffNode} from 'org_xprof/frontend/app/common/interfaces/op_profile_diff';
import {formatServingBadge, formatServingTooltip, getServingMetadata, normalizeServingMetadata} from 'org_xprof/frontend/app/common/utils/serving_metadata_utils';
import * as utils from 'org_xprof/frontend/app/common/utils/utils';
import {DATA_SERVICE_INTERFACE_TOKEN} from 'org_xprof/frontend/app/services/data_service_v2/data_service_v2_interface';
import {setActiveOpProfileNodeAction, updateSelectedOpNodeChainAction} from 'org_xprof/frontend/app/store/actions';
import {getOpAnalysisState, getTagsState} from 'org_xprof/frontend/app/store/selectors';
import {type OpAnalysisState} from 'org_xprof/frontend/app/store/state';
import {type Node} from 'org_xprof/frontend/app/common/interfaces/op_profile.jsonpb_decls';

/** Interface representing a row in the flattened Op Profile tree table. */
export interface FlatOpNode {
  node: Node;
  level: number;
  expanded: boolean;
  hasChildren: boolean;
  barWidth: string;
  percent: string;
  timeWasted: string;
  hbmFraction: string;
  name: string;
  offset: string;
  provenance: string;
  flopsUtilization: string;
  flameColor: string;
  hbmUtilization: string;
  hbmFlameColor: string;
  diffNode?: DiffNode;
  isLeftout?: boolean;
  numLeftOut?: number;
  kernelViewerLink?: utils.KernelViewerLink;
  programServingBadgeText?: string;
  programServingTooltipText?: string;
  programServingAccessibleText?: string;
}

function fmtDiff(activeVal: number, baseVal: number, hasBase: boolean, diffNode?: DiffNode, defVal = '-'): string {
  if (hasBase) {
    if (diffNode?.activeOnly) {
      return !isNaN(activeVal) ? `${(activeVal * 100).toFixed(2)}% (Added)` : '-';
    }
    if (diffNode?.baselineOnly) {
      return !isNaN(baseVal) ? `0.00% (base: ${(baseVal * 100).toFixed(2)}%)` : '-';
    }
    if (!isNaN(activeVal) && !isNaN(baseVal)) {
      const aPct = activeVal * 100;
      const dPct = aPct - baseVal * 100;
      return `${aPct.toFixed(2)}% (${dPct >= 0 ? '+' : ''}${dPct.toFixed(2)}%)`;
    }
  }
  return utils.percent(activeVal, defVal);
}

function getHbmFrac(node?: Node, root?: Node): number {
  const hbm = utils.MemBwType.MEM_BW_TYPE_HBM_RW;
  const nb = node?.metrics?.rawBytesAccessedArray;
  const rb = root?.metrics?.rawBytesAccessedArray;
  return nb && rb && nb.length > hbm && rb.length > hbm && rb[hbm] ? nb[hbm] / rb[hbm] : NaN;
}

/** An op table view component using MatTable for high-density tree representation. */
@Component({
  changeDetection: ChangeDetectionStrategy.Default,
  standalone: false,
  selector: 'op-table',
  templateUrl: './op_table.ng.html',
  styleUrls: ['./op_table.scss'],
  host: {'role': 'treegrid', '[attr.aria-label]': '"Op Profile Table"'},
})
export class OpTable implements OnInit, OnChanges, OnDestroy {
  @Input() rootNode?: Node;
  @Input() sessionId = '';
  @Input() moduleList: string[] = [];
  @Input() byWasted = false;
  @Input() showP90 = false;
  @Input() childrenCount = 10;

  selectedNode?: Node;
  tags: string[] = [];
  readonly expandedNodes = new Set<Node>();
  readonly dataSource = new MatTableDataSource<FlatOpNode>([]);
  applyScalingFactor = false;

  private readonly destroyed = new ReplaySubject<void>(1);
  readonly displayedColumns = [
    'time', 'wasted', 'hbmFraction', 'name', 'provenance',
    'utilization', 'flameColorCircle', 'hbmUtilization', 'hbmFlameColorCircle',
  ];
  readonly isDataRow = (_index: number, row: FlatOpNode) => !row.isLeftout;
  readonly isLeftoutRow = (_index: number, row: FlatOpNode) => !!row.isLeftout;
  private readonly store: Store<{}> = inject(Store);
  private readonly dataService = inject(DATA_SERVICE_INTERFACE_TOKEN, {optional: true});

  constructor() {
    this.store.select(getOpAnalysisState).pipe(takeUntil(this.destroyed)).subscribe((state: OpAnalysisState) => {
      if (this.applyScalingFactor !== state.applyScalingFactor) {
        this.applyScalingFactor = state.applyScalingFactor;
        this.refreshTable();
      }
    });
    this.store.select(getTagsState).pipe(takeUntil(this.destroyed)).subscribe((tags: string[]) => {
      this.tags = tags ?? [];
      this.refreshTable();
    });
  }

  ngOnInit() {
    this.refreshTable();
  }

  ngOnChanges(changes: SimpleChanges) {
    if (changes['rootNode']) {
      this.expandedNodes.clear();
    }
    this.refreshTable();
  }

  ngOnDestroy() {
    this.destroyed.next();
    this.destroyed.complete();
    this.store.dispatch(setActiveOpProfileNodeAction({activeOpProfileNode: null}));
  }

  updateActive(node: Node | null) {
    this.store.dispatch(setActiveOpProfileNodeAction({activeOpProfileNode: node || this.selectedNode || null}));
  }

  toggleExpanded(row: FlatOpNode) {
    if (row.isLeftout) return;
    if (!this.expandedNodes.delete(row.node)) {
      this.expandedNodes.add(row.node);
    }
    this.selectedNode = row.node;
    this.store.dispatch(updateSelectedOpNodeChainAction({selectedOpNodeName: row.node?.name}));
    this.updateActive(row.node);
    this.refreshTable();
  }

  trackByNode = (index: number, row: FlatOpNode) =>
    `${row.isLeftout ? 'leftout-' : ''}${row.node?.name || ''}-${row.level}-${index}`;

  private getChildren(node: Node, root: Node, level: number): Node[] {
    if (!node?.children || !root) return [];
    const ch = node.children.slice();
    if (this.byWasted) {
      ch.sort((a, b) => {
        const wa = utils.timeWasted(a, root);
        const wb = utils.timeWasted(b, root);
        return isNaN(wa) ? 1 : isNaN(wb) ? -1 : wb - wa;
      });
    }
    if (!level) return ch;
    let limit = Math.min(this.childrenCount, ch.length);
    if (this.showP90 && node.metrics?.rawTime) {
      let tot = 0;
      for (let i = 0; i < limit; i++) {
        tot += ch[i]?.metrics?.rawTime || 0;
        if (tot >= node.metrics.rawTime * 0.9) {
          limit = i + 1;
          break;
        }
      }
    }
    return ch.slice(0, limit);
  }

  private toRow(node: Node, root: Node, level: number, hasChildren: boolean, expanded: boolean): FlatOpNode {
    const dn = node as DiffNode | undefined;
    const dr = root as DiffNode | undefined;
    const hb = !!dn?.baseline || !!dn?.activeOnly || !!dn?.baselineOnly || !!dr?.baseline;
    const br = dr?.baseline || root;
    const bNode = dn?.baseline;
    const af = node.metrics ? utils.timeFraction(node, root) : 0;
    const bf = hb && bNode && dr?.baseline ? utils.timeFraction(bNode, dr.baseline) : 0;
    const hasTime = !!(node.metrics || dn?.baselineOnly);
    const util = utils.flopsUtilization(node, root, this.applyScalingFactor);
    const bUtil = hb && bNode ? utils.flopsUtilization(bNode, br, this.applyScalingFactor) : NaN;
    const bw = utils.MemBwType.MEM_BW_TYPE_HBM_RW;
    const hbmU = utils.memoryBandwidthUtilization(node, bw);
    const bHbmU = hb && bNode ? utils.memoryBandwidthUtilization(bNode, bw) : NaN;
    const normalizedServing = normalizeServingMetadata(getServingMetadata(node));
    const servingTooltip = normalizedServing ? formatServingTooltip(normalizedServing) : '';
    const kernelViewerLink = this.dataService
      ? utils.getOpProfileKernelViewerLink({
          node,
          tags: this.tags,
          sessionId: this.sessionId,
          dataService: this.dataService,
          rootNode: root,
          moduleList: this.moduleList,
        })
      : utils.NO_KERNEL_VIEWER_LINK;
    return {
      node, level, expanded, hasChildren, diffNode: dn,
      barWidth: hasTime && !dn?.baselineOnly ? utils.percent(af) : '0',
      percent: hasTime ? fmtDiff(af, bf, hb, dn) : '',
      timeWasted: fmtDiff(utils.timeWasted(node, root), hb && bNode ? utils.timeWasted(bNode, br) : NaN, hb, dn),
      hbmFraction: fmtDiff(getHbmFrac(node, root), hb && bNode ? getHbmFrac(bNode, br) : NaN, hb, dn, ''),
      name: node.name || '',
      offset: `${level}em`,
      provenance: utils.parseFrameworkOpType(node.xla?.provenance),
      flopsUtilization: fmtDiff(util, bUtil, hb, dn),
      flameColor: utils.flameColor(dn?.baselineOnly ? bUtil : util, 0.7, 1, Math.sqrt),
      hbmUtilization: fmtDiff(hbmU, bHbmU, hb, dn),
      hbmFlameColor: utils.bwColor(dn?.baselineOnly ? bHbmU : hbmU),
      kernelViewerLink,
      programServingBadgeText: normalizedServing ? formatServingBadge(normalizedServing) : '',
      programServingTooltipText: servingTooltip,
      programServingAccessibleText: servingTooltip ? ` (${servingTooltip})` : '',
    };
  }

  private refreshTable() {
    const root = this.rootNode;
    if (!root) {
      this.dataSource.data = [];
      return;
    }
    const rows: FlatOpNode[] = [];
    const walk = (children: Node[], level: number) => {
      for (const ch of children) {
        const sub = this.getChildren(ch, root, level);
        const exp = this.expandedNodes.has(ch);
        rows.push(this.toRow(ch, root, level, sub.length > 0, exp));
        if (exp && sub.length > 0) {
          walk(sub, level + 1);
          const leftOut = (ch.numChildren || 0) - Math.min(this.childrenCount, sub.length);
          if (leftOut > 0) {
            rows.push({...this.toRow(ch, root, level + 1, false, false), barWidth: '0', isLeftout: true, numLeftOut: leftOut});
          }
        }
      }
    };
    walk(this.getChildren(root, root, 0), 1);
    this.dataSource.data = rows;
  }
}
