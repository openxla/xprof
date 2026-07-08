import {
  ChangeDetectionStrategy,
  Component,
  computed,
  effect,
  input,
  output,
  signal,
  untracked,
} from '@angular/core';
import type {MemoryAnalysisBuffer} from 'org_xprof/frontend/app/common/interfaces/memory_analysis';
import type {
  ColorMode,
  TreeNode,
} from 'org_xprof/frontend/app/common/interfaces/memory_analysis_view';
import {humanReadableText} from 'org_xprof/frontend/app/common/utils/utils';

/** Represents a bounding box in the treemap layout. */
interface Bounds {
  readonly x: number;
  readonly y: number;
  readonly width: number;
  readonly height: number;
}

/** Represents a single rectangle in the treemap visualization. */
interface TreemapRect {
  readonly node: TreeNode;
  readonly x: number;
  readonly y: number;
  readonly width: number;
  readonly height: number;
  readonly color: string;
  readonly label: string;
  readonly paddingWidth: number;
  readonly paddingHeight: number;
  readonly dimmed: boolean;
  readonly tooltip: string;
}

function getTruncatedLabel(options: {
  readonly name: string;
  readonly width: number;
  readonly height: number;
}): string {
  const {name, width, height} = options;
  if (width < 60 || height < 22) return '';
  const approxCharWidth = 9;
  const maxChars = Math.floor(width / approxCharWidth);
  if (maxChars < 3) return '';
  if (name.length <= maxChars) return name;
  return name.substring(0, maxChars - 2) + '..';
}

function formatNumber(options: {
  readonly value: number;
  readonly minFraction: number;
  readonly maxFraction: number;
}): string {
  const {value, minFraction, maxFraction} = options;
  return value.toLocaleString('en-US', {
    minimumFractionDigits: minFraction,
    maximumFractionDigits: maxFraction,
  });
}

function worstRatio(options: {
  readonly row: readonly TreeNode[];
  readonly shorterSide: number;
  readonly bounds: Bounds;
  readonly totalValue: number;
}): number {
  const {row, shorterSide, bounds, totalValue} = options;
  if (totalValue <= 0) return Infinity;

  const scale = (bounds.width * bounds.height) / totalValue;
  const rowValue = row.reduce((sum, node) => sum + node.value, 0);
  const rowArea = rowValue * scale;
  const shorterSideSq = shorterSide * shorterSide;

  if (rowArea === 0 || shorterSideSq === 0) return Infinity;

  let minArea = Infinity;
  let maxArea = -Infinity;
  for (const node of row) {
    const area = node.value * scale;
    if (area < minArea) minArea = area;
    if (area > maxArea) maxArea = area;
  }

  if (minArea <= 0) return Infinity;

  const worst1 = (shorterSideSq * maxArea) / (rowArea * rowArea);
  const worst2 = (rowArea * rowArea) / (shorterSideSq * minArea);

  return Math.max(worst1, worst2);
}

function improvesRatio(options: {
  readonly row: readonly TreeNode[];
  readonly node: Readonly<TreeNode>;
  readonly shorterSide: number;
  readonly bounds: Bounds;
  readonly totalValue: number;
}): boolean {
  const {row, node, shorterSide, bounds, totalValue} = options;
  if (row.length === 0) return true;

  const currentWorst = worstRatio({row, shorterSide, bounds, totalValue});
  const nextWorst = worstRatio({
    row: [...row, node],
    shorterSide,
    bounds,
    totalValue,
  });

  return nextWorst <= currentWorst;
}

function getColor(options: {
  readonly node: Readonly<TreeNode>;
  readonly isHovered: boolean;
  readonly colorMode: ColorMode;
}): string {
  const {node, isHovered, colorMode} = options;
  if (isHovered) return '#1a73e8';

  if (node.isLeaf) {
    if (colorMode === 'padding') {
      const ratio =
        node.totalValue > 0 ? node.paddingValue / node.totalValue : 0;
      if (ratio < 0.05) {
        return 'hsl(138, 35%, 91%)';
      }
      const hue = 138 - ratio * 138;
      const saturation = 35 + ratio * 35;
      const lightness = 91 - ratio * 5;
      return `hsl(${hue}, ${saturation}%, ${lightness}%)`;
    }

    if (colorMode === 'category') {
      const category = node.buffer?.category ?? 'Other';
      switch (category) {
        case 'Weight':
          return '#d3e4ff';
        case 'Input':
          return '#c2eed0';
        case 'Optimizer State':
          return '#ffe0b2';
        case 'SparseCore':
          return '#e8def8';
        case 'Temporary/Activation':
        case 'Temporary':
          return '#ffdad6';
        case 'Gradient':
          return '#ffdbb5';
        case 'Output':
          return '#c2f0fc';
        default:
          return '#f3f0f4';
      }
    }

    let hash = 0;
    for (let i = 0; i < node.name.length; i++) {
      hash = node.name.charCodeAt(i) + ((hash << 5) - hash);
    }
    const h = Math.abs(hash);
    const warmHue = (h % 45) + 10;
    const s = 80 + (h % 12);
    const l = 60 + (h % 12);
    return `hsl(${warmHue}, ${s}%, ${l}%)`;
  }

  return 'hsl(240, 6%, 95%)';
}

/**
 * Visualizes hierarchical memory traces using a squarified treemap layout,
 * allowing users to inspect memory allocation proportions and padding overhead.
 */
@Component({
  changeDetection: ChangeDetectionStrategy.OnPush,
  selector: 'memory-treemap',
  templateUrl: './treemap.ng.html',
  styleUrls: ['./treemap.scss'],
  standalone: false,
})
export class MemoryTreemap {
  // Modern Signal-based input APIs
  readonly rootNode = input.required<TreeNode>();
  readonly sizeMetric = input<'total' | 'padding'>('total');
  readonly hoveredBuffer = input<MemoryAnalysisBuffer | null>(null);
  readonly hoveredPath = input('');
  readonly colorMode = input<ColorMode>('category');
  readonly selectedNode = input<TreeNode | null>(null);
  readonly searchTerm = input('');

  // Modern Signal-based output APIs
  readonly nodeSelected = output<TreeNode>();
  readonly nodeHovered = output<TreeNode | null>();

  readonly currentRoot = signal<TreeNode | null>(null);
  readonly zoomStack = signal<TreeNode[]>([]);

  readonly viewWidth = 1200;
  readonly viewHeight = 750;

  private lastRootNode?: TreeNode;
  private lastSelectedNode?: TreeNode | null;

  readonly humanReadableTotalSize = computed(() => {
    const currentRoot = this.currentRoot();
    if (!currentRoot || currentRoot.value <= 0) {
      return '0.00 B';
    }
    return humanReadableText(currentRoot.value * 1024 * 1024);
  });

  readonly rects = computed(() => {
    const currentRoot = this.currentRoot();
    if (!currentRoot || currentRoot.value <= 0) {
      return [];
    }
    const rects: TreemapRect[] = [];
    const search = this.searchTerm().toLowerCase();
    const hoveredBuffer = this.hoveredBuffer();
    const hoveredPath = this.hoveredPath();
    const colorMode = this.colorMode();

    const activeChildren = currentRoot.children.filter(
      (childNode) => childNode.value > 0,
    );
    if (activeChildren.length > 0) {
      this.squarify({
        children: activeChildren,
        row: [],
        bounds: {x: 0, y: 0, width: this.viewWidth, height: this.viewHeight},
        totalValue: currentRoot.value,
        rects,
        hoveredBuffer,
        hoveredPath,
        search,
        colorMode,
      });
    }
    return rects;
  });

  constructor() {
    effect(
      () => {
        const rootNode = this.rootNode();
        const selectedNode = this.selectedNode();

        untracked(() => {
          if (rootNode !== this.lastRootNode) {
            this.lastRootNode = rootNode;
            this.lastSelectedNode = selectedNode;
            const activePath =
              this.currentRoot()?.path ?? selectedNode?.path ?? '';
            if (activePath) {
              this.reconstructZoomStackForPath(activePath);
            } else {
              this.currentRoot.set(rootNode);
              this.zoomStack.set([rootNode]);
            }
          } else if (selectedNode !== this.lastSelectedNode) {
            this.lastSelectedNode = selectedNode;
            if (selectedNode) {
              this.reconstructZoomStackForPath(selectedNode.path);
            } else {
              this.currentRoot.set(rootNode);
              this.zoomStack.set([rootNode]);
            }
          }
        });
      },
      {allowSignalWrites: true},
    );
  }

  private reconstructZoomStackForPath(path: string): void {
    const root = this.rootNode();
    if (!root) return;
    const pathSegments = path.split('/').filter(Boolean);
    let current = root;
    const newStack: TreeNode[] = [current];
    for (const segment of pathSegments) {
      if (current === root && segment === root.name) continue;
      const childNode = current.children.find(
        (childNode) => childNode.name === segment,
      );
      if (childNode) {
        if (childNode.isLeaf) break;
        newStack.push(childNode);
        current = childNode;
      } else {
        break;
      }
    }
    this.zoomStack.set(newStack);
    this.currentRoot.set(current);
  }

  private squarify(options: {
    readonly children: readonly TreeNode[];
    readonly row: readonly TreeNode[];
    readonly bounds: Bounds;
    readonly totalValue: number;
    readonly rects: TreemapRect[];
    readonly hoveredBuffer: MemoryAnalysisBuffer | null;
    readonly hoveredPath: string;
    readonly search: string;
    readonly colorMode: ColorMode;
  }): void {
    const {
      children,
      row,
      bounds,
      totalValue,
      rects,
      hoveredBuffer,
      hoveredPath,
      search,
      colorMode,
    } = options;
    if (bounds.width <= 0 || bounds.height <= 0) return;

    if (children.length === 0) {
      if (row.length > 0) {
        this.layoutRow({
          row,
          bounds,
          totalValue,
          rects,
          hoveredBuffer,
          hoveredPath,
          search,
          colorMode,
        });
      }
      return;
    }

    const shorterSide = Math.min(bounds.width, bounds.height);
    const nextNode = children[0];

    if (
      improvesRatio({
        row,
        node: nextNode,
        shorterSide,
        bounds,
        totalValue,
      })
    ) {
      this.squarify({
        children: children.slice(1),
        row: [...row, nextNode],
        bounds,
        totalValue,
        rects,
        hoveredBuffer,
        hoveredPath,
        search,
        colorMode,
      });
    } else {
      const newBounds = this.layoutRow({
        row,
        bounds,
        totalValue,
        rects,
        hoveredBuffer,
        hoveredPath,
        search,
        colorMode,
      });
      this.squarify({
        children,
        row: [],
        bounds: newBounds,
        totalValue,
        rects,
        hoveredBuffer,
        hoveredPath,
        search,
        colorMode,
      });
    }
  }

  private layoutRow(options: {
    readonly row: readonly TreeNode[];
    readonly bounds: Bounds;
    readonly totalValue: number;
    readonly rects: TreemapRect[];
    readonly hoveredBuffer: MemoryAnalysisBuffer | null;
    readonly hoveredPath: string;
    readonly search: string;
    readonly colorMode: ColorMode;
  }): Bounds {
    const {
      row,
      bounds,
      totalValue,
      rects,
      hoveredBuffer,
      hoveredPath,
      search,
      colorMode,
    } = options;
    if (totalValue <= 0) return bounds;

    const scale = (bounds.width * bounds.height) / totalValue;
    const rowValue = row.reduce((sum, node) => sum + node.value, 0);
    const rowArea = rowValue * scale;

    const isVertical = bounds.width >= bounds.height;
    const rowWidth = isVertical
      ? bounds.height > 0
        ? rowArea / bounds.height
        : 0
      : bounds.width;
    const rowHeight = isVertical
      ? bounds.height
      : bounds.width > 0
        ? rowArea / bounds.width
        : 0;

    let coord = isVertical ? bounds.y : bounds.x;

    for (const node of row) {
      const area = node.value * scale;
      const width = isVertical
        ? rowWidth
        : rowHeight > 0
          ? area / rowHeight
          : 0;
      const height = isVertical
        ? rowWidth > 0
          ? area / rowWidth
          : 0
        : rowHeight;

      const rectX = isVertical ? bounds.x : coord;
      const rectY = isVertical ? coord : bounds.y;

      this.addRect({
        node,
        x: rectX,
        y: rectY,
        width,
        height,
        rects,
        hoveredBuffer,
        hoveredPath,
        search,
        colorMode,
      });

      coord += isVertical ? height : width;
    }

    return {
      x: isVertical ? bounds.x + rowWidth : bounds.x,
      y: isVertical ? bounds.y : bounds.y + rowHeight,
      width: isVertical ? Math.max(0, bounds.width - rowWidth) : bounds.width,
      height: isVertical
        ? bounds.height
        : Math.max(0, bounds.height - rowHeight),
    };
  }

  private addRect(options: {
    readonly node: Readonly<TreeNode>;
    readonly x: number;
    readonly y: number;
    readonly width: number;
    readonly height: number;
    readonly rects: TreemapRect[];
    readonly hoveredBuffer: MemoryAnalysisBuffer | null;
    readonly hoveredPath: string;
    readonly search: string;
    readonly colorMode: ColorMode;
  }): void {
    const {
      node,
      x,
      y,
      width,
      height,
      rects,
      hoveredBuffer,
      hoveredPath,
      search,
      colorMode,
    } = options;
    if (width < 1 || height < 1) return;

    const buffer = node.buffer;

    const matchesSearch =
      !search ||
      node.name.toLowerCase().includes(search) ||
      node.path.toLowerCase().includes(search) ||
      !!(buffer && buffer.name.toLowerCase().includes(search));

    const isHovered = !!(hoveredBuffer && buffer && buffer === hoveredBuffer);

    const containsHovered = !!(
      hoveredPath &&
      (node.path === '' ||
        hoveredPath === node.path ||
        hoveredPath.startsWith(node.path + '/'))
    );

    const paddingRatio =
      node.totalValue > 0 ? node.paddingValue / node.totalValue : 0;

    const isVerticalSplit = width >= height;
    const paddingWidth = isVerticalSplit ? width * paddingRatio : width;
    const paddingHeight = isVerticalSplit ? height : height * paddingRatio;

    const totalValueString = formatNumber({
      value: node.totalValue,
      minFraction: 1,
      maxFraction: 2,
    });
    const paddingValueString = formatNumber({
      value: node.paddingValue,
      minFraction: 1,
      maxFraction: 2,
    });
    const percent =
      node.totalValue > 0 ? (node.paddingValue / node.totalValue) * 100 : 0;
    const percentStr = formatNumber({
      value: percent,
      minFraction: 1,
      maxFraction: 1,
    });
    const tooltip = `${node.path}\nSize: ${totalValueString} MiB\nPadding: ${paddingValueString} MiB (${percentStr}%)`;

    rects.push({
      node,
      x,
      y,
      width,
      height,
      color: getColor({node, isHovered, colorMode}),
      label: getTruncatedLabel({name: node.name, width, height}),
      paddingWidth,
      paddingHeight,
      dimmed:
        (!!search && !matchesSearch) ||
        (!!hoveredPath && !isHovered && !containsHovered),
      tooltip,
    });

    if (!node.children || node.children.length === 0) return;

    const padding = 2;
    if (width <= padding * 2 + 10 || height <= padding * 2 + 16 + 10) return;

    const activeChildren = node.children.filter(
      (childNode) => childNode.value > 0,
    );
    if (activeChildren.length === 0) return;

    this.squarify({
      children: activeChildren,
      row: [],
      bounds: {
        x: x + padding,
        y: y + padding + 16,
        width: width - padding * 2,
        height: height - padding * 2 - 16,
      },
      totalValue: node.value,
      rects,
      hoveredBuffer,
      hoveredPath,
      search,
      colorMode,
    });
  }

  selectNode(node: TreeNode): void {
    if (node.isLeaf || node === this.currentRoot()) return;
    this.lastSelectedNode = node;
    this.currentRoot.set(node);
    this.zoomStack.update((stack) => [...stack, node]);
    this.nodeSelected.emit(node);
  }

  zoomOut(): void {
    const stack = this.zoomStack();
    if (stack.length <= 1) return;
    const newStack = stack.slice(0, stack.length - 1);
    this.zoomStack.set(newStack);
    const newRoot = newStack[newStack.length - 1];
    this.lastSelectedNode = newRoot;
    this.currentRoot.set(newRoot);
    this.nodeSelected.emit(newRoot);
  }

  resetZoom(): void {
    const root = this.rootNode();
    this.lastSelectedNode = root;
    this.currentRoot.set(root);
    this.zoomStack.set([root]);
    this.nodeSelected.emit(root);
  }

  onHover(node?: TreeNode | null): void {
    this.nodeHovered.emit(node ?? null);
  }
}
