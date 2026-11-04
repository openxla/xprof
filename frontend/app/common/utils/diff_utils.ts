/**
 * @fileoverview Utilities for diffing and aligning data tables.
 */

import * as utils from './utils';

/** Represents the presence of a row across active and baseline sessions. */
export type DiffStatus = 'added' | 'removed' | 'common';

/**
 * Generic row comparison wrapper.
 */
export interface ComparisonRow<T> {
  active: T | null;
  baseline: T | null;
  delta: number | null;
  status: DiffStatus;
}

/** Generic diff flags for comparing an active node against a baseline node. */
export interface DiffComparisonFlags<T = TreeMetricNode> {
  baseline?: T;
  activeOnly?: boolean;
  baselineOnly?: boolean;
}

/** Generic hierarchical metric node for tree table flattening and diffing. */
export interface TreeMetricNode extends DiffComparisonFlags<TreeMetricNode> {
  name?: string;
  numChildren?: number;
  children?: TreeMetricNode[];
  metrics?: {
    rawTime?: number;
    flops?: number;
    uncappedFlops?: number;
    rawFlops?: number;
    bf16Flops?: number;
    normalizedTimePs?: number;
    bandwidthUtils?: number[];
    rawBytesAccessedArray?: number[];
    occurrences?: number;
    avgTimePs?: number;
  };
  xla?: {
    provenance?: string;
  };
}

/** Interface representing a row in a flattened metric tree table. */
export interface FlatOpNode {
  node: TreeMetricNode;
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
  diffNode?: TreeMetricNode;
  isLeftout?: boolean;
  numLeftOut?: number;
}

/** Configuration options for flattening an Op Profile tree. */
export interface FlatOpTreeOptions {
  byWasted: boolean;
  showP90: boolean;
  childrenCount: number;
  applyScalingFactor: boolean;
}

/**
 * Aligns items from active and baseline sessions into a unified map of
 * comparison rows indexed by a unique key.
 */
export function alignTables<T>(
  active: T[] | null | undefined,
  baseline: T[] | null | undefined,
  keySelector: (item: T) => string,
  deltaCalculator?: (active: T, baseline: T) => number,
  baselineKeySelector?: (item: T) => string,
): Map<string, ComparisonRow<T>> {
  const result = new Map<string, ComparisonRow<T>>();

  if (active) {
    for (const item of active) {
      const key = keySelector(item);
      result.set(key, {
        active: item,
        baseline: null,
        delta: null,
        status: 'added',
      });
    }
  }

  if (baseline) {
    const selector = baselineKeySelector || keySelector;
    for (const item of baseline) {
      const key = selector(item);
      const existing = result.get(key);

      if (existing) {
        existing.baseline = item;
        if (existing.active !== null) {
          existing.status = 'common';
          existing.delta = deltaCalculator
            ? deltaCalculator(existing.active, item)
            : 0;
        } else {
          existing.status = 'removed';
          existing.delta = null;
        }
      } else {
        result.set(key, {
          active: null,
          baseline: item,
          delta: null,
          status: 'removed',
        });
      }
    }
  }

  return result;
}

/** Formats a metric value with optional baseline diff percentage. */
export function formatOpMetricDiff(
  activeVal: number,
  baseVal: number,
  hasBaseline: boolean,
  diffNode?: DiffComparisonFlags,
  defaultValueIfNull = '-',
): string {
  if (hasBaseline) {
    if (diffNode?.activeOnly) {
      return !isNaN(activeVal)
        ? `${(activeVal * 100).toFixed(2)}% (Added)`
        : defaultValueIfNull;
    } else if (diffNode?.baselineOnly) {
      return !isNaN(baseVal)
        ? `0.00% (base: ${(baseVal * 100).toFixed(2)}%)`
        : defaultValueIfNull;
    } else if (!isNaN(activeVal) && !isNaN(baseVal)) {
      const activePct = activeVal * 100;
      const diffPct = activePct - baseVal * 100;
      const deltaStr =
        diffPct >= 0 ? `+${diffPct.toFixed(2)}%` : `${diffPct.toFixed(2)}%`;
      return `${activePct.toFixed(2)}% (${deltaStr})`;
    }
  }
  return utils.percent(activeVal, defaultValueIfNull);
}

function getHbmFraction(node?: TreeMetricNode, root?: TreeMetricNode): number {
  const hbm = utils.MemBwType.MEM_BW_TYPE_HBM_RW;
  const nodeBytes = node?.metrics?.rawBytesAccessedArray;
  const rootBytes = root?.metrics?.rawBytesAccessedArray;
  if (
    nodeBytes &&
    rootBytes &&
    nodeBytes.length > hbm &&
    rootBytes.length > hbm &&
    rootBytes[hbm] !== 0
  ) {
    return nodeBytes[hbm] / rootBytes[hbm];
  }
  return NaN;
}

/** Computes display properties and diff metrics for a single metric tree node. */
export function computeOpNodeProps(
  node: TreeMetricNode,
  rootNode: TreeMetricNode,
  level: number,
  hasChildren: boolean,
  expanded: boolean,
  applyScalingFactor: boolean,
): FlatOpNode {
  const hasBaseline =
    !!node?.baseline ||
    !!node?.activeOnly ||
    !!node?.baselineOnly ||
    !!rootNode?.baseline;
  const baseRoot = rootNode?.baseline || rootNode;

  const activeFraction = utils.timeFraction(node, rootNode);
  const baseFraction =
    node?.baseline && rootNode?.baseline
      ? utils.timeFraction(node.baseline, rootNode.baseline)
      : 0;

  let percent = '';
  let barWidth = '0';
  if (node.metrics || node?.baselineOnly) {
    percent = formatOpMetricDiff(
      activeFraction,
      baseFraction,
      hasBaseline,
      node,
    );
    barWidth = node?.baselineOnly ? '0' : utils.percent(activeFraction);
  }

  const utilization = utils.flopsUtilization(
    node,
    rootNode,
    applyScalingFactor,
  );
  const baseUtilization = node?.baseline
    ? utils.flopsUtilization(node.baseline, baseRoot, applyScalingFactor)
    : NaN;
  const flopsUtilization = formatOpMetricDiff(
    utilization,
    baseUtilization,
    hasBaseline,
    node,
  );
  const flameColor = utils.flopsColor(
    node?.baselineOnly ? baseUtilization : utilization,
  );

  const activeWasted = utils.timeWasted(node, rootNode);
  const baseWasted = node?.baseline
    ? utils.timeWasted(node.baseline, baseRoot)
    : NaN;
  const timeWasted = formatOpMetricDiff(
    activeWasted,
    baseWasted,
    hasBaseline,
    node,
  );

  const activeHbmFrac = getHbmFraction(node, rootNode);
  const baseHbmFrac = node?.baseline
    ? getHbmFraction(node.baseline, baseRoot)
    : NaN;
  const hbmFraction = formatOpMetricDiff(
    activeHbmFrac,
    baseHbmFrac,
    hasBaseline,
    node,
    '',
  );

  const hbmUtil = utils.memoryBandwidthUtilization(
    node,
    utils.MemBwType.MEM_BW_TYPE_HBM_RW,
  );
  const baseHbmUtil = node?.baseline
    ? utils.memoryBandwidthUtilization(
        node.baseline,
        utils.MemBwType.MEM_BW_TYPE_HBM_RW,
      )
    : NaN;
  const hbmUtilization = formatOpMetricDiff(
    hbmUtil,
    baseHbmUtil,
    hasBaseline,
    node,
  );
  const hbmFlameColor = utils.bwColor(
    node?.baselineOnly ? baseHbmUtil : hbmUtil,
  );

  return {
    node,
    level,
    expanded,
    hasChildren,
    barWidth,
    percent,
    timeWasted,
    hbmFraction,
    name: node?.name || '',
    offset: `${level}em`,
    provenance: utils.parseFrameworkOpType(node?.xla?.provenance),
    flopsUtilization,
    flameColor,
    hbmUtilization,
    hbmFlameColor,
    diffNode: node,
  };
}

function get90ChildrenIndex(
  node: TreeMetricNode,
  children: TreeMetricNode[],
  options: FlatOpTreeOptions,
): number {
  if (!options.showP90 || !children.length || !node.metrics?.rawTime) {
    return options.childrenCount;
  }
  let tot = 0;
  const targetP90 = node.metrics.rawTime * 0.9;
  const targetCount = Math.min(options.childrenCount, children.length);
  for (let i = 0; i < targetCount; i++) {
    tot += children[i]?.metrics?.rawTime || 0;
    if (tot >= targetP90) return i + 1;
  }
  return targetCount;
}

function getVisibleChildren(
  node: TreeMetricNode,
  rootNode: TreeMetricNode,
  level: number,
  options: FlatOpTreeOptions,
): TreeMetricNode[] {
  if (!node?.children || !rootNode) return [];
  const children = options.byWasted
    ? node.children
        .map((child) => ({
          child,
          wasted: utils.timeWasted(child, rootNode),
        }))
        .sort((a, b) => {
          if (isNaN(a.wasted) && isNaN(b.wasted)) return 0;
          if (isNaN(a.wasted)) return 1;
          if (isNaN(b.wasted)) return -1;
          return b.wasted - a.wasted;
        })
        .map((entry) => entry.child)
    : node.children.slice();
  return level
    ? children.slice(0, get90ChildrenIndex(node, children, options))
    : children;
}

/** Builds the flattened list of visible rows for a metric tree table. */
export function buildFlatOpNodes(
  rootNode: TreeMetricNode | undefined,
  expandedNodes: ReadonlySet<TreeMetricNode>,
  options: FlatOpTreeOptions,
): FlatOpNode[] {
  if (!rootNode) return [];
  const flatList: FlatOpNode[] = [];

  const processChildren = (children: TreeMetricNode[], level: number) => {
    for (const child of children) {
      const hasChildren = Boolean(
        child.children?.length && options.childrenCount > 0,
      );
      const expanded = expandedNodes.has(child);
      flatList.push(
        computeOpNodeProps(
          child,
          rootNode,
          level,
          hasChildren,
          expanded,
          options.applyScalingFactor,
        ),
      );
      if (expanded && hasChildren) {
        const grandChildren = getVisibleChildren(
          child,
          rootNode,
          level,
          options,
        );
        processChildren(grandChildren, level + 1);
        const leftOut = child.numChildren
          ? child.numChildren -
            Math.min(options.childrenCount, grandChildren.length)
          : 0;
        if (leftOut > 0) {
          flatList.push({
            node: child,
            level: level + 1,
            expanded: false,
            hasChildren: false,
            barWidth: '0',
            percent: '',
            timeWasted: '',
            hbmFraction: '',
            name: '',
            offset: `${level + 1}em`,
            provenance: '',
            flopsUtilization: '',
            flameColor: '',
            hbmUtilization: '',
            hbmFlameColor: '',
            isLeftout: true,
            numLeftOut: leftOut,
          });
        }
      }
    }
  };

  processChildren(getVisibleChildren(rootNode, rootNode, 0, options), 1);
  return flatList;
}
