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

/** Represents a single visual bar in the flame graph. */
interface FlameBar {
  readonly node: TreeNode;
  readonly x: number;
  readonly width: number;
  readonly y: number;
  readonly height: number;
  readonly color: string;
  readonly label: string;
  readonly labelX: number;
  readonly labelY: number;
  readonly paddingWidth: number;
  readonly dimmed: boolean;
  readonly tooltip: string;
}

/** Represents a tick mark on the flame graph ruler. */
interface RulerTick {
  readonly x: number;
  readonly label: string;
  readonly anchor: 'start' | 'middle' | 'end';
}

function getTruncatedLabel(options: {
  readonly name: string;
  readonly width: number;
}): string {
  const {name, width} = options;
  const approxCharWidth = 10;
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
export class MemoryFlameGraph {
  // Signal-based inputs
  readonly rootNode = input.required<TreeNode>();
  readonly sizeMetric = input<'total' | 'padding'>('total');
  readonly hoveredBuffer = input<MemoryAnalysisBuffer | null>(null);
  readonly hoveredPath = input('');
  readonly colorMode = input<ColorMode>('category');
  readonly selectedNode = input<TreeNode | null>(null);
  readonly searchTerm = input('');

  // Signal-based outputs
  readonly nodeSelected = output<TreeNode>();
  readonly nodeHovered = output<TreeNode | null>();

  readonly zoomStack = signal<TreeNode[]>([]);
  readonly currentRoot = signal<TreeNode | null>(null);

  private readonly layoutData = computed(() => this.render());
  readonly bars = computed(() => this.layoutData().bars);
  readonly rulerTicks = computed(() => this.layoutData().rulerTicks);
  readonly svgHeight = computed(() => this.layoutData().svgHeight);

  readonly viewWidth = 1200;

  private readonly barHeight = 60;
  private readonly rowSpacing = 2;

  private lastRootNode?: TreeNode;
  private lastSelectedNode?: TreeNode | null;

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

  private render(): {
    bars: FlameBar[];
    rulerTicks: RulerTick[];
    svgHeight: number;
  } {
    const currentRoot = this.currentRoot();
    if (!currentRoot) return {bars: [], rulerTicks: [], svgHeight: 400};
    const bars: FlameBar[] = [];
    const context = {maxDepth: 0};
    const search = this.searchTerm().toLowerCase();
    const hoveredBuffer = this.hoveredBuffer();
    const hoveredPath = this.hoveredPath();
    const colorMode = this.colorMode();

    this.calculateLayout({
      node: currentRoot,
      x: 0,
      width: this.viewWidth,
      y: 0,
      rootValue: currentRoot.value,
      bars,
      context,
      hoveredBuffer,
      hoveredPath,
      search,
      colorMode,
    });

    const svgHeight =
      (context.maxDepth + 1) * (this.barHeight + this.rowSpacing) + 20;

    const numTicks = 5;
    const rulerTicks: RulerTick[] = [];
    const maxValue = currentRoot.value;
    for (let i = 0; i < numTicks; i++) {
      const ratio = i / (numTicks - 1);
      const x = ratio * this.viewWidth;
      const valueBytes = ratio * maxValue * 1024 * 1024;
      const label = humanReadableText(valueBytes);
      const anchor: 'start' | 'middle' | 'end' =
        i === 0 ? 'start' : i === numTicks - 1 ? 'end' : 'middle';
      rulerTicks.push({x, label, anchor});
    }
    return {bars, rulerTicks, svgHeight};
  }

  private calculateLayout(options: {
    readonly node: TreeNode;
    readonly x: number;
    readonly width: number;
    readonly y: number;
    readonly rootValue: number;
    readonly bars: FlameBar[];
    readonly context: {maxDepth: number};
    readonly hoveredBuffer: MemoryAnalysisBuffer | null;
    readonly hoveredPath: string;
    readonly search: string;
    readonly colorMode: ColorMode;
  }): void {
    const {
      node,
      x,
      width,
      y,
      rootValue,
      bars,
      context,
      hoveredBuffer,
      hoveredPath,
      search,
      colorMode,
    } = options;
    if (!width || isNaN(width) || width < 0.5) return;

    if (y > context.maxDepth) {
      context.maxDepth = y;
    }

    const paddingRatio =
      node.totalValue > 0 ? node.paddingValue / node.totalValue : 0;
    const paddingWidth = width * paddingRatio;

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

    const bar: FlameBar = {
      node,
      x,
      width,
      y: y * (this.barHeight + this.rowSpacing),
      height: this.barHeight,
      color: getColor({node, isHovered, colorMode}),
      label: getTruncatedLabel({name: node.name, width}),
      labelX: width / 2,
      labelY: this.barHeight / 2,
      paddingWidth,
      dimmed:
        (!!search && !matchesSearch) ||
        (!!hoveredPath && !isHovered && !containsHovered),
      tooltip,
    };

    bars.push(bar);

    if (!node.children || node.children.length === 0) {
      return;
    }

    let childX = x;
    const totalChildValue = node.children.reduce(
      (sum, childNode) => sum + childNode.value,
      0,
    );
    const widthPerValue = totalChildValue > 0 ? width / totalChildValue : 0;
    for (const childNode of node.children) {
      const childWidth = childNode.value * widthPerValue;
      this.calculateLayout({
        node: childNode,
        x: childX,
        width: childWidth,
        y: y + 1,
        rootValue,
        bars,
        context,
        hoveredBuffer,
        hoveredPath,
        search,
        colorMode,
      });
      childX += childWidth;
    }
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
