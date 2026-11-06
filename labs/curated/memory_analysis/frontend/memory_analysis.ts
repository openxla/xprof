import {KeyValue} from '@angular/common';
import {
  ChangeDetectionStrategy,
  ChangeDetectorRef,
  Component,
  DestroyRef,
  inject,
  signal,
} from '@angular/core';
import {takeUntilDestroyed} from '@angular/core/rxjs-interop';
import {ActivatedRoute} from '@angular/router';
import {Store} from '@ngrx/store';
import {combineLatest, Observable, of} from 'rxjs';
import {catchError, map, switchMap, tap} from 'rxjs/operators';

import {
  setCurrentToolStateAction,
  setLoadingStateAction,
} from 'org_xprof/frontend/app/store/actions';

import type {
  MemoryAnalysisBuffer,
  MemoryAnalysisResult,
  MemorySpaceBreakdown,
  RawMemoryAnalysisResult,
} from 'org_xprof/frontend/app/common/interfaces/memory_analysis';
import type {
  ColorMode,
  TreeNode,
} from 'org_xprof/frontend/app/common/interfaces/memory_analysis_view';
import {DATA_SERVICE_INTERFACE_TOKEN} from 'org_xprof/frontend/app/services/data_service_v2/data_service_v2_interface';
import {buildTree, getPathSegments} from './utils/tree_builder';

/** Represents a summary of memory usage for a specific category. */
interface CategorySummary {
  readonly name: string;
  readonly value: number;
  readonly percentage: number;
}

declare interface RawMemorySpaceBreakdown {
  readonly [space: string]: number | undefined;
}

declare interface RawBuffer {
  readonly label?: string;
  readonly size?: number;
  readonly unpaddedSize?: number;
  readonly category?: string;
  readonly subCategory?: string;
  readonly tfOp?: string;
  readonly shape?: string;
  readonly jaxPath?: string;
  readonly group?: string;
  readonly memorySpace?: string;
}

const DEFAULT_MEMORY_SPACE = 'HBM';

function preprocessResult(raw: RawMemoryAnalysisResult): MemoryAnalysisResult {
  const memorySpaceBreakdown: MemorySpaceBreakdown = {};
  // Cast to declare interface for minification safety
  const rawBreakdown =
    (raw['memorySpaceBreakdown'] as unknown as RawMemorySpaceBreakdown) ?? {};
  for (const [space, bytes] of Object.entries(rawBreakdown)) {
    if (bytes !== undefined) {
      memorySpaceBreakdown[space] = bytes / 1048576.0;
    }
  }

  const buffers: MemoryAnalysisBuffer[] = [];
  const bufferTotals: MemorySpaceBreakdown = {};
  // Cast to declare interface for minification safety
  const rawBuffers = (raw['peakHeapBuffers'] as unknown as RawBuffer[]) ?? [];
  for (const rawBuffer of rawBuffers) {
    const shapeStr = rawBuffer.shape ?? '';
    const dtypeMatch = shapeStr.match(/^([a-z0-9]+)\[/);
    const dtype = dtypeMatch ? dtypeMatch[1] : '';
    let shape = shapeStr.includes('[')
      ? shapeStr.substring(shapeStr.indexOf('['))
      : shapeStr;
    if (shape.includes(',') && !shape.includes(', ')) {
      shape = shape.replace(/,/g, ', ');
    }

    const sizeBytes = rawBuffer.size ?? 0;
    const sizeMib = sizeBytes / 1048576.0;
    const unpaddedSize = rawBuffer.unpaddedSize ?? sizeBytes;
    const paddingBytes = sizeBytes - unpaddedSize;
    const paddingMib = paddingBytes / 1048576.0;
    const paddingPercentage =
      sizeBytes > 0 ? (paddingBytes / sizeBytes) * 100.0 : 0.0;
    const memorySpace = rawBuffer.memorySpace || DEFAULT_MEMORY_SPACE;

    buffers.push({
      name: rawBuffer.label ?? '',
      sizeMib,
      category: rawBuffer.category ?? '',
      subCategory: rawBuffer.subCategory ?? '',
      group: rawBuffer.group ?? '',
      dtype,
      tfOpName: rawBuffer.tfOp ?? '',
      shape,
      jaxVariablePath: rawBuffer.jaxPath ?? '',
      paddingMib,
      paddingPercentage,
      memorySpace,
      'path': '',
    });
    bufferTotals[memorySpace] = (bufferTotals[memorySpace] ?? 0) + sizeMib;
  }

  for (const [space, totalMib] of Object.entries(bufferTotals)) {
    if (memorySpaceBreakdown[space] === undefined) {
      memorySpaceBreakdown[space] = totalMib;
    }
  }

  return {
    moduleName: raw['moduleName'] ?? '',
    memorySpaceBreakdown,
    buffers,
  };
}

function findNodeInTree(options: {
  readonly node: Readonly<TreeNode>;
  readonly name: string;
  readonly targetDepth: number;
}): TreeNode | null {
  const {node, name, targetDepth} = options;
  if (node.depth === targetDepth && node.name === name) {
    // Cast Readonly<TreeNode> back to TreeNode for return type
    return node as TreeNode;
  }
  for (const childNode of node.children) {
    const result = findNodeInTree({node: childNode, name, targetDepth});
    if (result) return result;
  }
  return null;
}

/**
 * Returns leaf buffers under `node`; `allBuffers` must be scoped to the tree's
 * memory space.
 */
function getLeafBuffers(options: {
  readonly node: Readonly<TreeNode>;
  readonly allBuffers: Readonly<MemoryAnalysisBuffer[]>;
}): MemoryAnalysisBuffer[] {
  const {node, allBuffers} = options;
  if (node.isLeaf) {
    return allBuffers.filter((b) => b['path'] === node.path);
  }
  let leaves: MemoryAnalysisBuffer[] = [];
  for (const childNode of node.children) {
    leaves = [...leaves, ...getLeafBuffers({node: childNode, allBuffers})];
  }
  return leaves;
}

function findTreeNodeForBuffer(options: {
  readonly node: Readonly<TreeNode>;
  readonly buffer: MemoryAnalysisBuffer;
}): TreeNode | null {
  const {node, buffer} = options;
  if (node.isLeaf && node.path === buffer['path']) {
    // Cast Readonly<TreeNode> back to TreeNode for return type
    return node as TreeNode;
  }
  for (const childNode of node.children) {
    const result = findTreeNodeForBuffer({node: childNode, buffer});
    if (result) return result;
  }
  return null;
}

/**
 * Populates `set` with buffers under `node`; `allBuffers` must be scoped to the
 * tree's memory space.
 */
function populateHoveredBuffersSet(options: {
  readonly node: Readonly<TreeNode>;
  readonly set: Set<MemoryAnalysisBuffer>;
  readonly allBuffers: Readonly<MemoryAnalysisBuffer[]>;
}): void {
  const {node, set, allBuffers} = options;
  if (node.isLeaf) {
    for (const b of allBuffers.filter((b) => b['path'] === node.path)) {
      set.add(b);
    }
    return;
  }
  for (const childNode of node.children) {
    populateHoveredBuffersSet({node: childNode, set, allBuffers});
  }
}

/**
 * Selects HBM when present in populated spaces, otherwise the largest space.
 */
function pickDefaultSpace(
  breakdown: Readonly<MemorySpaceBreakdown>,
  spaceList: readonly string[],
): string {
  let candidates = [...spaceList];
  if (candidates.length === 0) {
    const breakdownKeys = Object.keys(breakdown);
    const nonZeroSpaces = breakdownKeys.filter((s) => breakdown[s] > 0);
    candidates = nonZeroSpaces.length > 0 ? nonZeroSpaces : breakdownKeys;
  }
  if (candidates.includes(DEFAULT_MEMORY_SPACE)) {
    return DEFAULT_MEMORY_SPACE;
  }
  return (
    candidates.sort((a, b) => breakdown[b] - breakdown[a])[0] ??
    DEFAULT_MEMORY_SPACE
  );
}

/**
 * Displays a peak heap memory breakdown, providing interactive visual tools
 * (Flame Graph and Treemap) to analyze category, sub-category, and data-type
 * allocation across memory spaces.
 */
@Component({
  changeDetection: ChangeDetectionStrategy.OnPush,
  selector: 'memory-analysis',
  templateUrl: './memory_analysis.ng.html',
  styleUrls: ['./memory_analysis.scss'],
  standalone: false,
})
export class MemoryAnalysis {
  private readonly dataService = inject(DATA_SERVICE_INTERFACE_TOKEN);
  private readonly changeDetectorRef = inject(ChangeDetectorRef);
  private readonly route = inject(ActivatedRoute);
  private readonly destroyRef = inject(DestroyRef);
  private readonly store = inject(Store);

  /** Total memory in MiB (exposed for testing). */
  totalMemoryMib = 0;
  selectedModule = '';
  moduleList: string[] = [];
  readonly loading = signal(false);
  errorMessage = '';

  result: MemoryAnalysisResult | null = null;
  readonly filteredBuffers = signal<MemoryAnalysisBuffer[]>([]);
  selectedCategories = new Set<string>();
  readonly activeCategorySummaries = signal<CategorySummary[]>([]);

  // Visualizer Toggles & Variables
  activeTab = 0;
  sizeMetric: 'total' | 'padding' = 'total';
  colorMode: ColorMode = 'classic';
  readonly sidebarExpanded = signal(true);

  rootNode: TreeNode | null = null;
  selectedVisualNode: TreeNode | null = null;
  hoveredVisualNode: TreeNode | null = null;
  readonly hoveredBuffersSet = signal(new Set<MemoryAnalysisBuffer>());

  searchTerm = '';

  /**
   * Compares two memory space key-value pairs to sort memory space cards in
   * descending order of memory size.
   */
  readonly compareSpacesDesc = (
    a: KeyValue<string, number>,
    b: KeyValue<string, number>,
  ): number => {
    return b.value - a.value;
  };

  selectedDtype = 'all';
  selectedSpace = DEFAULT_MEMORY_SPACE;

  dtypeList: string[] = [];
  spaceList: string[] = [];

  private sessionId = '';
  private host = '';

  // Displayed table columns.
  displayedColumns: string[] = [
    'name',
    'memorySpace',
    'sizeMib',
    'category',
    'subCategory',
    'group',
    'paddingOverhead',
    'dtype',
    'shape',
    'tfOpName',
    'jaxVariablePath',
  ];

  // Cached full buffers list to avoid re-preprocessing
  private fullBuffers: MemoryAnalysisBuffer[] = [];
  // Cached buffers of the currently selected memory space.
  private spaceBuffers: MemoryAnalysisBuffer[] = [];

  constructor() {
    this.store.dispatch(
      setCurrentToolStateAction({currentTool: 'memory_analysis'}),
    );
    combineLatest([this.route.params, this.route.queryParams])
      .pipe(
        tap(([params, queryParams]) => {
          this.sessionId = params['sessionId'] ?? this.sessionId;
          this.host = queryParams['host'] ?? '';
          this.loading.set(true);
          this.store.dispatch(
            setLoadingStateAction({
              loadingState: {loading: true, message: 'Loading modules...'},
            }),
          );
        }),
        switchMap(([, queryParams]) => {
          const initialModule =
            queryParams['moduleName'] ?? queryParams['module_name'];
          const moduleList$ =
            this.moduleList.length === 0
              ? this.dataService.getMemoryAnalysisModuleList(
                  this.sessionId,
                  this.host,
                )
              : of(this.moduleList.join(','));

          return moduleList$.pipe(
            switchMap((moduleListStr) => {
              if (moduleListStr) {
                const moduleList = moduleListStr.split(',').filter(Boolean);
                // Logical OR (||) is required here because initialModule can be an empty string ('') from URL query parameters, which should fall back to the first module in the list.
                const selectedModule =
                  initialModule || (moduleList.length > 0 ? moduleList[0] : '');
                if (!selectedModule) {
                  return of({moduleList, selectedModule, data: null});
                }
                return this.dataService
                  .getDataByModuleNameAndMemorySpace(
                    'memory_analysis',
                    this.sessionId,
                    this.host,
                    selectedModule,
                    0,
                  )
                  .pipe(map((data) => ({moduleList, selectedModule, data})));
              } else {
                return of({moduleList: [], selectedModule: '', data: null});
              }
            }),
            catchError((error) => {
              this.errorMessage = error.message;
              return of({moduleList: [], selectedModule: '', data: null});
            }),
          );
        }),
        takeUntilDestroyed(this.destroyRef),
      )
      .subscribe(({moduleList, selectedModule, data}) => {
        this.moduleList = moduleList;
        this.selectedModule = selectedModule;
        this.handleModuleData(data);
      });
  }

  private loadModuleDataObservable(): Observable<unknown> {
    this.loading.set(true);
    this.store.dispatch(
      setLoadingStateAction({
        loadingState: {loading: true, message: 'Loading module data...'},
      }),
    );
    return this.dataService.getDataByModuleNameAndMemorySpace(
      'memory_analysis',
      this.sessionId,
      this.host,
      this.selectedModule,
      0,
    );
  }

  private loadModuleData(): void {
    this.loadModuleDataObservable()
      .pipe(
        catchError((error) => {
          this.errorMessage = error.message;
          return of(null);
        }),
        takeUntilDestroyed(this.destroyRef),
      )
      .subscribe((data) => {
        this.handleModuleData(data);
      });
  }

  private handleModuleData(data: unknown): void {
    this.loading.set(false);
    this.store.dispatch(
      setLoadingStateAction({
        loadingState: {loading: false, message: ''},
      }),
    );
    if (data) {
      // Cast unknown data to RawMemoryAnalysisResult
      this.result = preprocessResult(data as RawMemoryAnalysisResult);
      this.fullBuffers = this.result.buffers;
      for (const b of this.fullBuffers) {
        // Object.assign is required because MemoryAnalysisBuffer has a readonly index signature.
        Object.assign(b, {path: this.getBufferPath(b)});
      }

      this.spaceList = Array.from(
        new Set(
          this.fullBuffers
            .map((b) => b.memorySpace)
            // Cast filtered array to string[]
            .filter(Boolean) as string[],
        ),
      );
      this.selectedSpace = pickDefaultSpace(
        this.result.memorySpaceBreakdown,
        this.spaceList,
      );

      this.rebuildTree();
    } else {
      this.result = null;
      this.fullBuffers = [];
      this.spaceList = [];
      this.selectedSpace = DEFAULT_MEMORY_SPACE;
      this.rebuildTree();
    }
    this.applyFilters();
    this.changeDetectorRef.markForCheck();
  }

  private rebuildTree(): void {
    this.spaceBuffers = this.fullBuffers.filter(
      (b) => b.memorySpace === this.selectedSpace,
    );
    this.selectedDtype = 'all';
    this.hoveredVisualNode = null;
    this.hoveredBuffersSet.set(new Set());
    if (this.spaceBuffers.length === 0) {
      this.totalMemoryMib = 0;
      this.rootNode = null;
      this.selectedVisualNode = null;
      this.selectedCategories = new Set();
      this.activeCategorySummaries.set([]);
      this.dtypeList = [];
      return;
    }

    this.dtypeList = [
      'all',
      ...new Set(this.spaceBuffers.map((b) => b.dtype).filter(Boolean)),
    ];

    const summaryMap: Record<string, number> = {};
    for (const b of this.spaceBuffers) {
      summaryMap[b.category] = (summaryMap[b.category] ?? 0) + b.sizeMib;
    }

    const total = Object.values(summaryMap).reduce<number>(
      (sum, v) => sum + v,
      0,
    );
    this.totalMemoryMib = total;

    const summaries = Object.entries(summaryMap)
      .map(([name, value]) => ({
        name,
        value,
        percentage: total > 0 ? (value / total) * 100 : 0,
      }))
      .sort((a, b) => b.value - a.value);
    this.activeCategorySummaries.set(summaries);
    this.selectedCategories = new Set(summaries.map((s) => s.name));

    this.rootNode = buildTree({
      buffers: this.spaceBuffers,
      metric: this.sizeMetric,
    });
    this.selectedVisualNode = this.rootNode;
  }

  onSizeMetricChange(value: 'total' | 'padding'): void {
    this.sizeMetric = value;
    this.rebuildTree();
    this.applyFilters();
  }

  onColorModeChange(value: ColorMode): void {
    this.colorMode = value;
    this.changeDetectorRef.markForCheck();
  }

  onVisualNodeSelected(node: TreeNode): void {
    this.selectedVisualNode = node;

    if (node.depth === 0) {
      this.selectedCategories = new Set(
        this.activeCategorySummaries().map((s) => s.name),
      );
    } else {
      const activeCategoryNames = this.activeCategorySummaries().map(
        (s) => s.name,
      );
      const matchedCategory = activeCategoryNames.find(
        (catName) =>
          node.path === catName || node.path.startsWith(`${catName}/`),
      );
      if (matchedCategory) {
        this.selectedCategories = new Set([matchedCategory]);
      }
    }

    this.applyFilters();
  }

  onVisualNodeHovered(node: TreeNode | null): void {
    this.hoveredVisualNode = node;
    const newSet = new Set<MemoryAnalysisBuffer>();
    if (node) {
      populateHoveredBuffersSet({
        node,
        set: newSet,
        allBuffers: this.spaceBuffers,
      });
    }
    this.hoveredBuffersSet.set(newSet);
    this.changeDetectorRef.markForCheck();
  }

  onSearchChanged(event: Event): void {
    // Cast event target to HTMLInputElement to access search value
    const inputElement = event.target as HTMLInputElement;
    this.searchTerm = inputElement.value;
    this.applyFilters();
  }

  onModuleSelected(moduleName: string): void {
    if (this.loading() || this.selectedModule === moduleName) {
      return;
    }
    this.selectedModule = moduleName;
    const searchParams = this.dataService.getSearchParams();
    searchParams.set('module_name', moduleName);
    this.dataService.setSearchParams(searchParams);

    this.loadModuleData();
  }

  toggleSidebar(): void {
    this.sidebarExpanded.set(!this.sidebarExpanded());
    this.changeDetectorRef.markForCheck();
  }

  toggleCategory(category: string): void {
    if (this.selectedCategories.has(category)) {
      this.selectedCategories.delete(category);
    } else {
      this.selectedCategories.add(category);
    }

    if (this.selectedCategories.size === 1) {
      const activeCategory = Array.from(this.selectedCategories)[0];
      if (this.rootNode) {
        const catNode = findNodeInTree({
          node: this.rootNode,
          name: activeCategory,
          targetDepth: 1,
        });
        if (catNode) {
          this.selectedVisualNode = catNode;
        } else {
          this.selectedVisualNode = this.rootNode;
          this.selectedDtype = 'all';
        }
      }
    } else {
      this.selectedVisualNode = this.rootNode;
      this.selectedDtype = 'all';
    }

    this.applyFilters();
  }

  onDtypeSelected(value: string): void {
    this.selectedDtype = value;
    this.applyFilters();
  }

  onSpaceSelected(spaceName: string): void {
    if (this.selectedSpace === spaceName) return;
    this.selectedSpace = spaceName;
    this.rebuildTree();
    this.applyFilters();
    this.changeDetectorRef.markForCheck();
  }

  getBufferPath(buffer: MemoryAnalysisBuffer): string {
    return getPathSegments({buffer}).join('/');
  }

  private applyFilters(): void {
    if (!this.result) {
      this.filteredBuffers.set([]);
      return;
    }

    let buffers = this.spaceBuffers;

    if (this.selectedVisualNode && this.selectedVisualNode !== this.rootNode) {
      buffers = getLeafBuffers({
        node: this.selectedVisualNode,
        allBuffers: this.spaceBuffers,
      });
    }

    const isDtypeAll = this.selectedDtype === 'all';
    const isCategoryEmpty = this.selectedCategories.size === 0;
    const lowercasedSearch = this.searchTerm.toLowerCase();
    const hasSearch = !!lowercasedSearch;

    this.filteredBuffers.set(
      buffers.filter((b) => {
        const dtypeMatch = isDtypeAll || b.dtype === this.selectedDtype;
        const cardMatch =
          isCategoryEmpty || this.selectedCategories.has(b.category);
        const searchMatch =
          !hasSearch ||
          b.name.toLowerCase().includes(lowercasedSearch) ||
          b.tfOpName.toLowerCase().includes(lowercasedSearch) ||
          b.jaxVariablePath.toLowerCase().includes(lowercasedSearch);

        return dtypeMatch && cardMatch && searchMatch;
      }),
    );
    this.changeDetectorRef.markForCheck();
  }

  onRowHover(row: MemoryAnalysisBuffer | null): void {
    if (!row) {
      this.hoveredVisualNode = null;
      this.hoveredBuffersSet.set(new Set());
      return;
    }

    if (!this.rootNode) return;

    const matchNode = findTreeNodeForBuffer({node: this.rootNode, buffer: row});
    if (matchNode) {
      this.hoveredVisualNode = matchNode;
      const allMatches = this.spaceBuffers.filter(
        (b) => b['path'] === matchNode.path,
      );
      this.hoveredBuffersSet.set(new Set(allMatches));
    }
    this.changeDetectorRef.markForCheck();
  }
}
