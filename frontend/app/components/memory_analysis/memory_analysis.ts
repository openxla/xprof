/**
 * @fileoverview Angular component to display peak heap memory breakdown.
 */

import {
  ChangeDetectionStrategy,
  Component,
  inject,
  OnDestroy,
  OnInit,
} from '@angular/core';
import {ActivatedRoute} from '@angular/router';
import {combineLatest, ReplaySubject} from 'rxjs';
import {takeUntil} from 'rxjs/operators';

import {
  MemoryAnalysisBuffer,
  MemoryAnalysisResult,
} from 'org_xprof/frontend/app/common/interfaces/memory_analysis';
import {DATA_SERVICE_INTERFACE_TOKEN} from 'org_xprof/frontend/app/services/data_service_v2/data_service_v2_interface';

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
  private readonly destroyed = new ReplaySubject<void>(1);

  sessionId = '';
  host = '';
  selectedModule = '';
  loading = false;

  result: MemoryAnalysisResult | null = null;
  filteredBuffers: MemoryAnalysisBuffer[] = [];
  selectedCategories = new Set<string>();

  // Displayed table columns.
  displayedColumns: string[] = [
    'name',
    'sizeMib',
    'category',
    'dtype',
    'shape',
    'tfOpName',
    'jaxVariablePath',
  ];

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
        this.result = data as unknown as MemoryAnalysisResult;
        if (this.result) {
          const summaries = this.result['categorySummaries'];
          this.selectedCategories = new Set(Object.keys(summaries));
        }
        this.applyFilters();
      });
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
    this.applyFilters();
  }

  /**
   * Filters buffers based on selected category sets.
   */
  applyFilters() {
    if (!this.result) return;
    const buffers = this.result['buffers'];
    this.filteredBuffers = buffers.filter((b) =>
      this.selectedCategories.has(b['category']),
    );
  }

  ngOnDestroy() {
    this.destroyed.next();
    this.destroyed.complete();
  }
}
