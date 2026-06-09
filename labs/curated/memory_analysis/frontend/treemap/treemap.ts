import {
  ChangeDetectionStrategy,
  ChangeDetectorRef,
  Component,
  EventEmitter,
  inject,
  Input,
  OnChanges,
  Output,
  SimpleChanges,
} from '@angular/core';
import type {MemoryAnalysisBuffer} from 'org_xprof/frontend/app/common/interfaces/memory_analysis';
import type {
  ColorMode,
  TreeNode,
} from 'org_xprof/frontend/app/common/interfaces/memory_analysis_view';
import {humanReadableText} from 'org_xprof/frontend/app/common/utils/utils';

interface TreemapRect {
  node: TreeNode;
  x: number;
  y: number;
  width: number;
  height: number;
  color: string;
  label: string;
  paddingWidth: number;
  paddingHeight: number;
  dimmed: boolean;
}

/**
 * An interactive, zero-dependency SVG-based Treemap component.
 * Visualizes hierarchical memory buffers as space-filling nested squares,
 * optimizing bounds aspect ratios close to 1:1 using squarified algorithms,
 * featuring custom hatched padding overlays, hovers, and zoom states.
 */
@Component({
  changeDetection: ChangeDetectionStrategy.OnPush,
  selector: 'memory-treemap',
  templateUrl: './treemap.ng.html',
  styleUrls: ['./treemap.scss'],
  standalone: false,
})
export class MemoryTreemap implements OnChanges {
  private readonly cdr = inject(ChangeDetectorRef);

  @Input({required: true}) rootNode!: TreeNode;
  @Input() sizeMetric: 'total' | 'padding' = 'total';
  @Input() hoveredBuffer: MemoryAnalysisBuffer | null = null;
  @Input() hoveredPath = '';
  @Input() colorMode: ColorMode = 'category';
  @Input() selectedNode: TreeNode | null = null;
  @Input() searchTerm = '';

  @Output() readonly nodeSelected = new EventEmitter<TreeNode>();
  @Output() readonly nodeHovered = new EventEmitter<TreeNode | null>();

  rects: TreemapRect[] = [];
  zoomStack: TreeNode[] = [];
  currentRoot!: TreeNode;
  humanReadableTotalSize = '';

  viewWidth = 1200;
  viewHeight = 750;

  ngOnChanges(changes: SimpleChanges) {
    if (changes['rootNode'] && this.rootNode) {
      const activePath = this.currentRoot ? this.currentRoot.path : '';
      if (activePath) {
        this.reconstructZoomStackForPath(activePath);
      } else {
        this.currentRoot = this.rootNode;
        this.zoomStack = [this.rootNode];
      }
      this.render();
    } else if (changes['selectedNode']) {
      if (this.selectedNode && this.rootNode) {
        this.reconstructZoomStackForPath(this.selectedNode.path);
      } else if (this.rootNode) {
        this.currentRoot = this.rootNode;
        this.zoomStack = [this.rootNode];
      }
      this.render();
    } else if (
      changes['sizeMetric'] ||
      changes['hoveredBuffer'] ||
      changes['hoveredPath'] ||
      changes['searchTerm']
    ) {
      this.render();
    }
  }

  private reconstructZoomStackForPath(path: string) {
    if (!this.rootNode) return;
    const pathSegments = path.split('/').filter(Boolean);
    let curr = this.rootNode;
    const newStack: TreeNode[] = [curr];
    for (const segment of pathSegments) {
      const child = curr.children.find((c: TreeNode) => c.name === segment);
      if (child) {
        newStack.push(child);
        curr = child;
      } else {
        break;
      }
    }
    this.zoomStack = newStack;
    this.currentRoot = curr;
  }

  render() {
    if (!this.currentRoot || this.currentRoot.value <= 0) {
      this.rects = [];
      this.humanReadableTotalSize = '0.00 B';
      this.cdr.markForCheck();
      return; // Gracefully abort layout on zero total bytes to prevent division-by-zero
    }
    this.humanReadableTotalSize = humanReadableText(
      this.currentRoot.value * 1024 * 1024,
    );
    this.rects = [];
    // Filter out any children with 0 value (e.g. 0 padding) as they occupy zero area
    const activeChildren = this.currentRoot.children.filter(
      (c: TreeNode) => c.value > 0,
    );
    if (activeChildren.length > 0) {
      this.squarify(
        activeChildren,
        [],
        {x: 0, y: 0, w: this.viewWidth, h: this.viewHeight},
        this.currentRoot.value,
      );
    }
    this.cdr.markForCheck();
  }

  /**
   * Classic Squarified Treemap Algorithm.
   * Divides bounds into squares optimizing aspect ratios.
   */
  private squarify(
    children: TreeNode[],
    row: TreeNode[],
    bounds: {x: number; y: number; w: number; h: number},
    totalValue: number,
  ) {
    if (bounds.w <= 0 || bounds.h <= 0) return; // Safeguard against zero/negative bounds

    if (children.length === 0) {
      if (row.length > 0) {
        this.layoutRow(row, bounds, totalValue);
      }
      return;
    }

    const shorterSide = Math.min(bounds.w, bounds.h);
    const nextNode = children[0];

    if (this.improvesRatio(row, nextNode, shorterSide, bounds, totalValue)) {
      this.squarify(children.slice(1), [...row, nextNode], bounds, totalValue);
    } else {
      const newBounds = this.layoutRow(row, bounds, totalValue);
      this.squarify(children, [], newBounds, totalValue);
    }
  }

  private improvesRatio(
    row: TreeNode[],
    node: TreeNode,
    shorterSide: number,
    bounds: {x: number; y: number; w: number; h: number},
    totalValue: number,
  ): boolean {
    if (row.length === 0) return true;

    const currentWorst = this.worstRatio(row, shorterSide, bounds, totalValue);
    const nextWorst = this.worstRatio(
      [...row, node],
      shorterSide,
      bounds,
      totalValue,
    );

    return nextWorst <= currentWorst;
  }

  private worstRatio(
    row: TreeNode[],
    shorterSide: number,
    bounds: {x: number; y: number; w: number; h: number},
    totalValue: number,
  ): number {
    if (totalValue <= 0) return Infinity;

    const scale = (bounds.w * bounds.h) / totalValue;
    const rowValue = row.reduce((sum, n) => sum + n.value, 0);
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

    // Defensive checks against division by zero under 0 padding metrics
    if (minArea <= 0) return Infinity;

    const worst1 = (shorterSideSq * maxArea) / (rowArea * rowArea);
    const worst2 = (rowArea * rowArea) / (shorterSideSq * minArea);

    return Math.max(worst1, worst2);
  }

  private layoutRow(
    row: TreeNode[],
    bounds: {x: number; y: number; w: number; h: number},
    totalValue: number,
  ): {x: number; y: number; w: number; h: number} {
    if (totalValue <= 0) return bounds;

    const scale = (bounds.w * bounds.h) / totalValue;
    const rowValue = row.reduce((sum, n) => sum + n.value, 0);
    const rowArea = rowValue * scale;

    const isVertical = bounds.w >= bounds.h;
    const rowWidth = isVertical
      ? bounds.h > 0
        ? rowArea / bounds.h
        : 0
      : bounds.w;
    const rowHeight = isVertical
      ? bounds.h
      : bounds.w > 0
        ? rowArea / bounds.w
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

      this.addRect(node, rectX, rectY, width, height);

      coord += isVertical ? height : width;
    }

    // Enforce Math.max(0, ...) to prevent negative boundaries from float arithmetic inaccuracies
    return {
      x: isVertical ? bounds.x + rowWidth : bounds.x,
      y: isVertical ? bounds.y : bounds.y + rowHeight,
      w: isVertical ? Math.max(0, bounds.w - rowWidth) : bounds.w,
      h: isVertical ? bounds.h : Math.max(0, bounds.h - rowHeight),
    };
  }

  private addRect(node: TreeNode, x: number, y: number, w: number, h: number) {
    if (w < 1 || h < 1) return;

    const matchesSearch =
      !this.searchTerm ||
      node.name.toLowerCase().includes(this.searchTerm.toLowerCase()) ||
      node.path.toLowerCase().includes(this.searchTerm.toLowerCase()) ||
      !!(
        node.buffer &&
        node.buffer.name.toLowerCase().includes(this.searchTerm.toLowerCase())
      );

    // Precise object reference match rather than string name to prevent collision highlights
    const isHovered = !!(
      this.hoveredBuffer &&
      node.buffer &&
      node.buffer === this.hoveredBuffer
    );

    // O(1) path-based parent highlighted subtree check instead of O(N^2) recursion
    const containsHovered = !!(
      this.hoveredPath &&
      (this.hoveredPath === node.path ||
        this.hoveredPath.startsWith(node.path + '/'))
    );

    const paddingRatio =
      node.totalValue > 0 ? node.paddingValue / node.totalValue : 0;

    const isVerticalSplit = w >= h;
    const paddingWidth = isVerticalSplit ? w * paddingRatio : w;
    const paddingHeight = isVerticalSplit ? h : h * paddingRatio;

    this.rects.push({
      node,
      x,
      y,
      width: w,
      height: h,
      color: this.getColor(node, isHovered),
      label: this.getTruncatedLabel(node.name, w, h),
      paddingWidth,
      paddingHeight,
      dimmed:
        (!!this.searchTerm && !matchesSearch) ||
        (!!this.hoveredPath && !isHovered && !containsHovered),
    });

    // Lay out active children inside this node's rectangle (nested active treemaps)
    if (node.children && node.children.length > 0) {
      const padding = 2;
      if (w > padding * 2 + 10 && h > padding * 2 + 16 + 10) {
        // Filter zero value nodes inside recursive children treemaps
        const activeChildren = node.children.filter(
          (c: TreeNode) => c.value > 0,
        );
        if (activeChildren.length > 0) {
          this.squarify(
            activeChildren,
            [],
            {
              x: x + padding,
              y: y + padding + 16,
              w: w - padding * 2,
              h: h - padding * 2 - 16,
            },
            node.value,
          );
        }
      }
    }
  }

  private getColor(node: TreeNode, isHovered: boolean): string {
    if (isHovered) return '#1a73e8'; // Active GM3 Blue highlight

    // 1. Parent scopes: Cool Slate Gray
    if (!node.isLeaf) {
      return 'hsl(240, 6%, 95%)'; // Surface Container Low Cool Slate
    }

    // 2. Padding (Memory Waste Heatmap) Mode
    if (this.colorMode === 'padding') {
      const ratio =
        node.totalValue > 0 ? node.paddingValue / node.totalValue : 0;
      if (ratio < 0.05) {
        return 'hsl(138, 35%, 91%)'; // Calm Pastel Sage Green (0% waste)
      }
      const hue = 138 - ratio * 138; // Sage Green (138) -> Soft Gold (48) -> Rose Red (0/360)
      const saturation = 35 + ratio * 35; // 35% -> 70%
      const lightness = 91 - ratio * 5; // 91% -> 86%
      return `hsl(${hue}, ${saturation}%, ${lightness}%)`;
    }

    // 3. Category Mode: Modern GM3 Tonal Accent Containers (Tone 90/95)
    if (this.colorMode === 'category') {
      const category = node.buffer?.category || 'Other';
      switch (category) {
        case 'Weight':
          return '#d3e4ff'; // M3 Blue Container (Tone 90)
        case 'Input':
          return '#c2eed0'; // M3 Green Container (Tone 90)
        case 'Optimizer State':
          return '#ffe0b2'; // M3 Amber Container (Tone 90)
        case 'SparseCore':
          return '#e8def8'; // M3 Purple Container (Tone 90)
        case 'Temporary/Activation':
        case 'Temporary':
          return '#ffdad6'; // M3 Red Container (Tone 90)
        case 'Gradient':
          return '#ffdbb5'; // M3 Orange Container (Tone 90)
        case 'Output':
          return '#c2f0fc'; // M3 Teal Container (Tone 90)
        default:
          return '#f3f0f4'; // M3 Neutral Gray Container (Tone 95)
      }
    }

    // 4. Classic Flame Mode: High-Vibrancy GM3 Fire Hashing Spectrum
    let hash = 0;
    for (let i = 0; i < node.name.length; i++) {
      hash = node.name.charCodeAt(i) + ((hash << 5) - hash);
    }
    const h = Math.abs(hash);
    const warmHue = (h % 45) + 10; // Warm fire: 10 (Crimson) -> 55 (Amber)
    const s = 80 + (h % 12); // 80% -> 92% (Rich, clean saturation)
    const l = 60 + (h % 12); // 60% -> 72% (Highly legible cards)
    return `hsl(${warmHue}, ${s}%, ${l}%)`;
  }

  private getTruncatedLabel(name: string, w: number, h: number): string {
    if (w < 60 || h < 22) return '';
    const approxCharWidth = 9;
    const maxChars = Math.floor(w / approxCharWidth);
    if (maxChars < 3) return '';
    if (name.length <= maxChars) return name;
    return name.substring(0, maxChars - 2) + '..';
  }

  selectNode(node: TreeNode) {
    if (node.isLeaf) return;
    this.currentRoot = node;
    this.zoomStack.push(node);
    this.nodeSelected.emit(node);
    this.render();
  }

  zoomOut() {
    if (this.zoomStack.length <= 1) return;
    this.zoomStack.pop();
    this.currentRoot = this.zoomStack[this.zoomStack.length - 1];
    this.nodeSelected.emit(this.currentRoot);
    this.render();
  }

  resetZoom() {
    this.currentRoot = this.rootNode;
    this.zoomStack = [this.rootNode];
    this.nodeSelected.emit(this.rootNode);
    this.render();
  }

  onHover(node: TreeNode | null) {
    this.nodeHovered.emit(node);
  }
}
