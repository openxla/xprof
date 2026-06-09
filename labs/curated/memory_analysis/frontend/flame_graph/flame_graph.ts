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

interface FlameBar {
  node: TreeNode;
  x: number;
  width: number;
  y: number;
  height: number;
  color: string;
  label: string;
  paddingWidth: number;
  dimmed: boolean;
}

interface RulerTick {
  x: number;
  label: string;
  anchor: 'start' | 'middle' | 'end';
}

/**
 * An interactive, zero-dependency SVG-based Flame Graph component.
 * Visualizes hierarchical memory traces with responsive coordinate scaling,
 * visual hatched padding overlays, breadcrumb zooming, and synced highlights.
 */
@Component({
  changeDetection: ChangeDetectionStrategy.OnPush,
  selector: 'memory-flame-graph',
  templateUrl: './flame_graph.ng.html',
  styleUrls: ['./flame_graph.scss'],
  standalone: false,
})
export class MemoryFlameGraph implements OnChanges {
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

  bars: FlameBar[] = [];
  zoomStack: TreeNode[] = [];
  currentRoot!: TreeNode;
  rulerTicks: RulerTick[] = [];

  // Layout configurations
  viewWidth = 1200;
  barHeight = 60;
  rowSpacing = 2;
  maxDepth = 0;
  svgHeight = 400;

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
    if (!this.currentRoot) return;
    this.bars = [];
    this.maxDepth = 0;
    this.calculateLayout(
      this.currentRoot,
      0,
      this.viewWidth,
      0,
      this.currentRoot.value,
    );
    this.svgHeight =
      (this.maxDepth + 1) * (this.barHeight + this.rowSpacing) + 20;

    const numTicks = 5;
    this.rulerTicks = [];
    const maxValue = this.currentRoot.value;
    for (let i = 0; i < numTicks; i++) {
      const ratio = i / (numTicks - 1);
      const x = ratio * this.viewWidth;
      const valBytes = ratio * maxValue * 1024 * 1024;
      const label = humanReadableText(valBytes);
      let anchor: 'start' | 'middle' | 'end' = 'middle';
      if (i === 0) {
        anchor = 'start';
      } else if (i === numTicks - 1) {
        anchor = 'end';
      }
      this.rulerTicks.push({x, label, anchor});
    }
    this.cdr.markForCheck();
  }

  private calculateLayout(
    node: TreeNode,
    x: number,
    width: number,
    y: number,
    rootValue: number,
  ) {
    if (!width || isNaN(width) || width < 0.5) return; // Defensive check to prevent NaN loops

    if (y > this.maxDepth) {
      this.maxDepth = y;
    }

    const paddingRatio =
      node.totalValue > 0 ? node.paddingValue / node.totalValue : 0;
    const paddingWidth = width * paddingRatio;

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

    const bar: FlameBar = {
      node,
      x,
      width,
      y: y * (this.barHeight + this.rowSpacing),
      height: this.barHeight,
      color: this.getColor(node, isHovered),
      label: this.getTruncatedLabel(node.name, width),
      paddingWidth,
      dimmed:
        (!!this.searchTerm && !matchesSearch) ||
        (!!this.hoveredPath && !isHovered && !containsHovered),
    };

    this.bars.push(bar);

    if (node.children && node.children.length > 0) {
      let childX = x;
      const totalChildValue = node.children.reduce(
        (sum: number, c: TreeNode) => sum + c.value,
        0,
      );
      for (const child of node.children) {
        // Safe divide-by-zero check using aggregate total child value
        const childWidth =
          totalChildValue > 0 ? (child.value / totalChildValue) * width : 0;
        this.calculateLayout(child, childX, childWidth, y + 1, rootValue);
        childX += childWidth;
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

  private getTruncatedLabel(name: string, width: number): string {
    const approxCharWidth = 10;
    const maxChars = Math.floor(width / approxCharWidth);
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
