/**
 * @fileoverview Angular component to display peak heap memory breakdown.
 */

import {KeyValue} from '@angular/common';
import {
  ChangeDetectionStrategy,
  ChangeDetectorRef,
  Component,
  inject,
  OnDestroy,
  OnInit,
} from '@angular/core';
import {ActivatedRoute} from '@angular/router';
import {combineLatest, ReplaySubject} from 'rxjs';
import {takeUntil} from 'rxjs/operators';

import type {
  CategorySummaries,
  ColorMode,
  MemoryAnalysisBuffer,
  MemoryAnalysisResult,
  MemorySpaceBreakdown,
  RawMemoryAnalysisResult,
  TreeNode,
} from 'org_xprof/frontend/app/common/interfaces/memory_analysis';
import {DATA_SERVICE_INTERFACE_TOKEN} from 'org_xprof/frontend/app/services/data_service_v2/data_service_v2_interface';
import {buildTree} from './utils/tree_builder';

/** A component for memory analysis visualization. */
@Component({
  changeDetection: ChangeDetectionStrategy.OnPush,
  selector: 'memory-analysis',
  templateUrl: './memory_analysis.ng.html',
  styleUrls: ['./memory_analysis.scss'],
  standalone: false,
})
export class MemoryAnalysis implements OnInit, OnDestroy {
  private readonly dataService = inject(DATA_SERVICE_INTERFACE_TOKEN);
  private readonly changeDetectorRef = inject(ChangeDetectorRef);
  private readonly destroyed = new ReplaySubject<void>(1);

  totalMemoryMib = 0;
  sessionId = '';
  host = '';
  selectedModule = '';
  moduleList: string[] = [];
  loading = false;

  result: MemoryAnalysisResult | null = null;
  filteredBuffers: MemoryAnalysisBuffer[] = [];
  selectedCategories = new Set<string>();
  activeCategorySummaries: CategorySummaries = {};

  // Visualizer Toggles & Variables
  activeTab = 0; // 0 = Flame Graph, 1 = Treemap
  hierarchyType: 'jax' | 'category' = 'jax';
  sizeMetric: 'total' | 'padding' = 'total';
  colorMode: ColorMode = 'category';
  sidebarExpanded = true;

  rootNode: TreeNode | null = null;
  selectedVisualNode: TreeNode | null = null;
  hoveredVisualNode: TreeNode | null = null;
  readonly hoveredBuffersSet = new Set<MemoryAnalysisBuffer>();

  searchTerm = '';

  /**
   * Custom comparator for keyvalue pipe to sort category cards in descending order of memory size.
   */
  readonly compareCategoriesDesc = (
    a: KeyValue<string, number>,
    b: KeyValue<string, number>,
  ): number => {
    return b.value - a.value; // Largest memory usage first!
  };

  /**
   * Custom comparator for keyvalue pipe to sort memory space cards in descending order of memory size.
   */
  readonly compareSpacesDesc = this.compareCategoriesDesc;

  // Dropdown Filter states
  selectedSubcat = 'all';
  selectedType = 'all';
  selectedDtype = 'all';
  selectedSpace = 'all';

  // Dynamic Filter option lists
  subcatList: string[] = [];
  typeList: string[] = [];
  dtypeList: string[] = [];
  spaceList: string[] = [];

  // Displayed table columns.
  displayedColumns: string[] = [
    'name',
    'memorySpace',
    'category',
    'subCategory',
    'group',
    'shape',
    'dtype',
    'sizeMib',
    'paddingOverhead',
    'tfOpName',
    'jaxVariablePath',
  ];

  // Cached full buffers list to avoid re-preprocessing
  private fullBuffers: MemoryAnalysisBuffer[] = [];

  constructor(route: ActivatedRoute) {
    combineLatest([route.params, route.queryParams])
      .pipe(takeUntil(this.destroyed))
      .subscribe(([params, queryParams]) => {
        this.sessionId = params['sessionId'] || this.sessionId;
        this.selectedModule =
          queryParams['moduleName'] || queryParams['module_name'] || '';
        this.host = queryParams['host'] || '';
        this.load();
      });
  }

  ngOnInit() {}

  /**
   * Loads peak memory analysis data from the data service.
   */
  load() {
    if (!this.selectedModule) {
      this.loading = true;
      this.dataService
        .getMemoryAnalysisModuleList(this.sessionId, this.host)
        .pipe(takeUntil(this.destroyed))
        .subscribe((moduleListStr) => {
          this.loading = false;
          if (!moduleListStr) {
            this.moduleList = [];
            this.selectedModule = '';
            this.result = null;
            this.rootNode = null;
            this.filteredBuffers = [];
            this.fullBuffers = [];
            this.changeDetectorRef.markForCheck();
            return;
          }
          this.moduleList = moduleListStr.split(',').filter(Boolean);
          if (this.moduleList.length > 0) {
            this.selectedModule = this.moduleList[0];
            this.loadModuleData();
          } else {
            this.selectedModule = '';
            this.result = null;
            this.rootNode = null;
            this.filteredBuffers = [];
            this.fullBuffers = [];
            this.changeDetectorRef.markForCheck();
          }
        });
    } else {
      if (this.moduleList.length === 0) {
        this.dataService
          .getMemoryAnalysisModuleList(this.sessionId, this.host)
          .pipe(takeUntil(this.destroyed))
          .subscribe((moduleListStr) => {
            if (moduleListStr) {
              this.moduleList = moduleListStr.split(',').filter(Boolean);
              this.changeDetectorRef.markForCheck();
            }
          });
      }
      this.loadModuleData();
    }
  }

  private loadModuleData() {
    this.loading = true;
    this.dataService
      .getDataByModuleNameAndMemorySpace(
        'memory_analysis',
        this.sessionId,
        this.host,
        this.selectedModule,
        0, // Default memory space (HBM).
      )
      .pipe(takeUntil(this.destroyed))
      .subscribe((data) => {
        this.loading = false;
        if (data) {
          this.selectedSpace = 'all';
          this.result = this.preprocessResult(data as RawMemoryAnalysisResult);
          this.fullBuffers = this.result.buffers;

          this.spaceList = Array.from(
            new Set(
              this.fullBuffers
                .map((b) => b.memorySpace)
                .filter(Boolean) as string[],
            ),
          );

          this.rebuildTree();

          this.subcatList = [
            'all',
            ...new Set(
              this.fullBuffers.map((b) => b.subCategory).filter(Boolean),
            ),
          ];
          this.typeList = [
            'all',
            ...new Set(this.fullBuffers.map((b) => b.group).filter(Boolean)),
          ];
          this.dtypeList = [
            'all',
            ...new Set(this.fullBuffers.map((b) => b.dtype).filter(Boolean)),
          ];

          this.selectedCategories = new Set(
            Object.keys(this.activeCategorySummaries),
          );
        } else {
          this.result = null;
          this.totalMemoryMib = 0;
          this.rootNode = null;
          this.fullBuffers = [];
          this.selectedCategories = new Set();
          this.activeCategorySummaries = {};
          this.subcatList = [];
          this.typeList = [];
          this.dtypeList = [];
          this.spaceList = [];
          this.selectedSpace = 'all';
        }
        this.applyFilters();
        this.changeDetectorRef.markForCheck();
      });
  }

  private preprocessResult(raw: RawMemoryAnalysisResult): MemoryAnalysisResult {
    const categorySummaries: CategorySummaries = {};
    const rawSummary = raw['summary'] || {};
    for (const [cat, bytes] of Object.entries(rawSummary)) {
      categorySummaries[cat] = (bytes as number) / 1048576.0; // Bytes to MiB
    }

    const memorySpaceBreakdown: MemorySpaceBreakdown = {};
    const rawBreakdown = raw['memorySpaceBreakdown'] || {};
    for (const [space, bytes] of Object.entries(rawBreakdown)) {
      memorySpaceBreakdown[space] = (bytes as number) / 1048576.0;
    }

    const buffers: MemoryAnalysisBuffer[] = [];
    const rawBuffers = raw['peakHeapBuffers'] || [];
    for (const rawBuf of rawBuffers) {
      const shapeStr = rawBuf['shape'] || '';
      const dtypeMatch = shapeStr.match(/^([a-z0-9]+)\[/);
      const dtype = dtypeMatch ? dtypeMatch[1] : '';
      let shape = shapeStr.includes('[')
        ? shapeStr.substring(shapeStr.indexOf('['))
        : shapeStr;
      if (shape.includes(',') && !shape.includes(', ')) {
        shape = shape.replace(/,/g, ', ');
      }

      const sizeBytes = (rawBuf['size'] as number) || 0;
      const unpaddedSize =
        rawBuf['unpaddedSize'] !== undefined
          ? (rawBuf['unpaddedSize'] as number)
          : sizeBytes;
      const paddingBytes = sizeBytes - unpaddedSize;
      const paddingMib = paddingBytes / 1048576.0;
      const paddingPercentage =
        sizeBytes > 0 ? (paddingBytes / sizeBytes) * 100.0 : 0.0;

      buffers.push({
        name: rawBuf['label'] || '',
        sizeMib: sizeBytes / 1048576.0, // Bytes to MiB
        category: rawBuf['category'] || '',
        subCategory: rawBuf['subCategory'] || '',
        group: rawBuf['group'] || '',
        dtype,
        tfOpName: rawBuf['tfOp'] || '',
        shape,
        jaxVariablePath: rawBuf['jaxPath'] || '',
        paddingMib,
        paddingPercentage,
        memorySpace: rawBuf['memorySpace'] || 'HBM',
      });
    }

    if (Object.keys(memorySpaceBreakdown).length === 0) {
      const totalBytes = Object.values(rawSummary).reduce(
        (sum, v) => sum + (v as number),
        0,
      );
      memorySpaceBreakdown['HBM'] = totalBytes / 1048576.0;
    }

    return {
      moduleName: raw['moduleName'] || '',
      categorySummaries,
      memorySpaceBreakdown,
      buffers,
    };
  }

  rebuildTree() {
    if (!this.fullBuffers || this.fullBuffers.length === 0) return;
    const buffersToBuild =
      this.selectedSpace === 'all'
        ? this.fullBuffers
        : this.fullBuffers.filter((b) => b.memorySpace === this.selectedSpace);

    if (this.selectedSpace === 'all') {
      this.activeCategorySummaries = this.result
        ? this.result.categorySummaries
        : {};
      this.totalMemoryMib = Object.values(this.activeCategorySummaries).reduce(
        (sum, v) => sum + v,
        0,
      );
    } else {
      const activeSummaries: CategorySummaries = {};
      for (const b of buffersToBuild) {
        activeSummaries[b.category] =
          (activeSummaries[b.category] || 0) + b.sizeMib;
      }
      this.activeCategorySummaries = activeSummaries;
      this.totalMemoryMib = Object.values(activeSummaries).reduce(
        (sum, v) => sum + v,
        0,
      );
    }

    this.rootNode = buildTree(
      buffersToBuild,
      this.hierarchyType,
      this.sizeMetric,
    );
    this.selectedVisualNode = this.rootNode;
  }

  onHierarchyTypeChange(val: 'jax' | 'category') {
    this.hierarchyType = val;
    this.rebuildTree();
    this.applyFilters();
  }

  onSizeMetricChange(val: 'total' | 'padding') {
    this.sizeMetric = val;
    this.rebuildTree();
    this.applyFilters();
  }

  onColorModeChange(val: ColorMode) {
    this.colorMode = val;
    this.changeDetectorRef.markForCheck();
  }

  onVisualNodeSelected(node: TreeNode) {
    this.selectedVisualNode = node;

    if (node.depth === 0 || node.name === 'root') {
      if (this.result) {
        this.selectedCategories = new Set(
          Object.keys(this.result.categorySummaries),
        );
      }
      this.selectedSubcat = 'all';
      this.selectedType = 'all';
    } else if (this.hierarchyType === 'category') {
      const segments = node.path.split('/').filter(Boolean);
      if (segments.length > 0) {
        this.selectedCategories = new Set([segments[0]]);
        this.selectedSubcat = segments[1] ? segments[1] : 'all';
        this.selectedType = segments[2] ? segments[2] : 'all';
      }
    }

    this.applyFilters();
  }

  onVisualNodeHovered(node: TreeNode | null) {
    this.hoveredVisualNode = node;
    this.hoveredBuffersSet.clear();
    if (node) {
      this.populateHoveredBuffersSet(node);
    }
    this.changeDetectorRef.markForCheck();
  }

  private populateHoveredBuffersSet(node: TreeNode) {
    if (node.isLeaf && node.buffer) {
      this.hoveredBuffersSet.add(node.buffer);
      return;
    }
    for (const child of node.children) {
      this.populateHoveredBuffersSet(child);
    }
  }

  onSearchChanged(event: Event) {
    const input = event.target as HTMLInputElement;
    this.searchTerm = input.value;
    this.applyFilters();
  }

  onModuleSelected(moduleName: string) {
    if (this.selectedModule === moduleName) {
      return;
    }
    this.selectedModule = moduleName;
    const searchParams = this.dataService.getSearchParams();
    searchParams.set('module_name', moduleName);
    this.dataService.setSearchParams(searchParams);

    this.loadModuleData();
  }

  toggleSidebar() {
    this.sidebarExpanded = !this.sidebarExpanded;
    this.changeDetectorRef.markForCheck();
  }

  /**
   * Toggles category filter state.
   * @param category The name of the category to toggle.
   */
  toggleCategory(category: string) {
    if (this.selectedCategories.has(category)) {
      this.selectedCategories.delete(category);
    } else {
      this.selectedCategories.add(category);
    }

    if (this.selectedCategories.size === 1) {
      const activeCategory = Array.from(this.selectedCategories)[0];
      if (this.rootNode) {
        const catNode = this.findNodeInTree(this.rootNode, activeCategory, 1);
        if (catNode) {
          this.selectedVisualNode = catNode;
          this.selectedSubcat = 'all';
          this.selectedType = 'all';
        }
      }
    } else {
      this.selectedVisualNode = this.rootNode;
      this.selectedSubcat = 'all';
      this.selectedType = 'all';
    }

    this.applyFilters();
  }

  onSubcatSelected(val: string) {
    this.selectedSubcat = val;
    if (this.hierarchyType === 'category' && val !== 'all' && this.rootNode) {
      // Search active tree for matching Sub-category node within active category (depth 2) (Issue 4)
      const match = this.findNodeInActiveCategories(val, 2);
      if (match) {
        this.selectedVisualNode = match;
      }
    }
    this.applyFilters();
  }

  onTypeSelected(val: string) {
    this.selectedType = val;
    if (this.hierarchyType === 'category' && val !== 'all' && this.rootNode) {
      // Search active tree for matching Group/Type node within active category (depth 3) (Issue 4)
      const match = this.findNodeInActiveCategories(val, 3);
      if (match) {
        this.selectedVisualNode = match;
      }
    }
    this.applyFilters();
  }

  private findNodeInActiveCategories(
    name: string,
    targetDepth: number,
  ): TreeNode | null {
    if (!this.rootNode) return null;

    // Search strictly within selected category branches first to prevent sibling name collisions (Issue 4)
    if (this.selectedCategories.size > 0) {
      for (const catName of this.selectedCategories) {
        const catNode = this.rootNode.children.find((c) => c.name === catName);
        if (catNode) {
          const match = this.findNodeInTree(catNode, name, targetDepth);
          if (match) return match;
        }
      }
    }

    // Fallback to general tree search if no category is selected
    return this.findNodeInTree(this.rootNode, name, targetDepth);
  }

  private findNodeInTree(
    node: TreeNode,
    name: string,
    targetDepth: number,
  ): TreeNode | null {
    if (node.depth === targetDepth && node.name === name) {
      return node;
    }
    for (const child of node.children) {
      const res = this.findNodeInTree(child, name, targetDepth);
      if (res) return res;
    }
    return null;
  }

  onDtypeSelected(val: string) {
    this.selectedDtype = val;
    this.applyFilters();
  }

  onSpaceSelected(spaceName: string) {
    if (this.selectedSpace === spaceName) return;
    this.selectedSpace = spaceName;
    this.rebuildTree();
    this.applyFilters();
    this.changeDetectorRef.markForCheck();
  }

  getBufferPath(buffer: MemoryAnalysisBuffer): string {
    if (this.hierarchyType === 'jax') {
      const jaxPath = buffer.jaxVariablePath || '';
      if (!jaxPath) {
        return `Others/${buffer.category || 'Uncategorized'}/${buffer.subCategory || 'Others'}/${buffer.name}`;
      }
      const cleanPath = jaxPath.replace(/^\/|\/$/g, '');
      return `${cleanPath}/${buffer.name}`; // Consistent segment append (Issue 2.2)
    } else {
      return [
        buffer.category || 'Uncategorized',
        buffer.subCategory || 'Others',
        buffer.group || 'General',
        buffer.name,
      ]
        .filter(Boolean)
        .join('/');
    }
  }

  /**
   * Filters buffers based on selected category sets and dropdown filters.
   */
  applyFilters() {
    if (!this.result) {
      this.filteredBuffers = [];
      return;
    }

    let buffers = this.fullBuffers;
    if (this.selectedSpace !== 'all') {
      buffers = buffers.filter((b) => b.memorySpace === this.selectedSpace);
    }

    // 1. Apply visual zoom/subtree node selection filter
    if (this.selectedVisualNode && this.selectedVisualNode.name !== 'root') {
      buffers = this.getLeafBuffers(this.selectedVisualNode);
    }

    // 2. Apply standard dropdown, category cards, and search queries
    this.filteredBuffers = buffers.filter((b) => {
      const subcatMatch =
        this.selectedSubcat === 'all' || b.subCategory === this.selectedSubcat;
      const typeMatch =
        this.selectedType === 'all' || b.group === this.selectedType;
      const dtypeMatch =
        this.selectedDtype === 'all' || b.dtype === this.selectedDtype;
      const cardMatch =
        this.selectedCategories.size === 0 ||
        this.selectedCategories.has(b.category);
      const searchMatch =
        !this.searchTerm ||
        b.name.toLowerCase().includes(this.searchTerm.toLowerCase()) ||
        b.tfOpName.toLowerCase().includes(this.searchTerm.toLowerCase()) ||
        b.jaxVariablePath.toLowerCase().includes(this.searchTerm.toLowerCase());

      return subcatMatch && typeMatch && dtypeMatch && cardMatch && searchMatch;
    });
    this.changeDetectorRef.markForCheck();
  }

  private getLeafBuffers(node: TreeNode): MemoryAnalysisBuffer[] {
    if (node.isLeaf && node.buffer) {
      return [node.buffer];
    }
    let leaves: MemoryAnalysisBuffer[] = [];
    for (const child of node.children) {
      leaves = [...leaves, ...this.getLeafBuffers(child)];
    }
    return leaves;
  }

  // Fast O(1) Set membership check instead of O(N * M) recursive traversals in digest loops (Issue 3)
  isRowHovered(row: MemoryAnalysisBuffer): boolean {
    return this.hoveredBuffersSet.has(row);
  }

  onRowHover(row: MemoryAnalysisBuffer | null) {
    if (!row) {
      this.hoveredVisualNode = null;
      this.hoveredBuffersSet.clear();
      return;
    }

    // Find the TreeNode corresponding to this row buffer in the active tree recursively
    if (this.rootNode) {
      const matchNode = this.findTreeNodeForBuffer(this.rootNode, row);
      if (matchNode) {
        this.hoveredVisualNode = matchNode;
        this.hoveredBuffersSet.clear();
        this.hoveredBuffersSet.add(row); // Single row leaf buffer cache
      }
    }
    this.changeDetectorRef.markForCheck();
  }

  private findTreeNodeForBuffer(
    node: TreeNode,
    buffer: MemoryAnalysisBuffer,
  ): TreeNode | null {
    if (node.isLeaf && node.buffer === buffer) {
      return node;
    }
    for (const child of node.children) {
      const res = this.findTreeNodeForBuffer(child, buffer);
      if (res) return res;
    }
    return null;
  }

  ngOnDestroy() {
    this.destroyed.next();
    this.destroyed.complete();
  }
}
