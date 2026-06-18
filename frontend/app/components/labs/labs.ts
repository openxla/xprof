import {
  ChangeDetectionStrategy,
  ChangeDetectorRef,
  Component,
  inject,
  OnInit,
} from '@angular/core';
import {MatSnackBar} from '@angular/material/snack-bar';
import {ActivatedRoute, Router} from '@angular/router';
import {Store} from '@ngrx/store';
import {
  setCurrentToolStateAction,
  setLoadingStateAction,
} from 'org_xprof/frontend/app/store/actions';
import {trySanitizeUrl} from 'safevalues';
import {windowOpen} from 'safevalues/dom';

interface CuratedTool {
  key: string;
  label: string;
  status: 'Alpha' | 'Beta';
  description: string;
  icon: string;
  url?: string;
  route?: string;
  tags: string[];
  isFeatured: boolean;
}

/**
 * Component representing the XProf Labs view, featuring experimental tools
 * and data playground exploration.
 */
@Component({
  selector: 'xprof-labs',
  templateUrl: './labs.ng.html',
  styleUrls: ['./labs.scss'],
  changeDetection: ChangeDetectionStrategy.OnPush,
  standalone: false,
})
export class LabsComponent implements OnInit {
  private readonly route = inject(ActivatedRoute);
  private readonly router = inject(Router);
  private readonly snackBar = inject(MatSnackBar);
  private readonly cdr = inject(ChangeDetectorRef);
  private readonly store = inject(Store);

  sessionId = '';
  searchQuery = '';
  selectedCategory = 'All';
  isLabsEnabled = false;
  activeView: 'experiments' | 'playground' = 'experiments';

  featuredTools: CuratedTool[] = [];
  regularTools: CuratedTool[] = [];

  setActiveView(view: 'experiments' | 'playground') {
    this.activeView = view;
    this.cdr.markForCheck();
  }

  readonly categories = ['All', 'Featured', 'Memory', 'Compiler', 'Network'];

  readonly curatedTools: CuratedTool[] = [
    {
      key: 'memory_analysis',
      label: 'Memory Analysis Tool',
      status: 'Beta',
      description:
        'Parse HLO memory profiles, model parameter weights, and analyze memory consumption breakdowns.',
      icon: 'overview_key',
      route: '/memory_analysis',
      tags: ['Memory'],
      isFeatured: true,
    },
    {
      key: 'tpu_viz',
      label: 'TPU-viz (Pod Topology)',
      status: 'Alpha',
      description:
        'Visualize TPU network nodes, link activities, and network routing pathways in interactive 3D.',
      icon: 'hive',
      tags: ['Network'],
      isFeatured: true,
    },
    {
      key: 'lunc_explorer',
      label: 'Lunc Compiler Explorer',
      status: 'Alpha',
      description:
        'Interactive visualization showing compiler pass transformations and instruction-level TPU scheduling.',
      icon: 'graph_2',
      tags: ['Compiler'],
      isFeatured: false,
    },
    {
      key: 'megascale_plugin',
      label: 'Megascale GRPC Plugin',
      status: 'Alpha',
      description:
        'Diagnose inter-slice network bottlenecks and RPC latencies across multi-node TPU slices.',
      icon: 'network_check',
      tags: ['Network'],
      isFeatured: false,
    },
  ];

  ngOnInit() {
    this.sessionId = this.route.snapshot.paramMap.get('sessionId') || '';

    // Check URL query param first, then fallback to localStorage
    const labsQueryParam = this.route.snapshot.queryParamMap.get('labs');
    if (labsQueryParam === 'true' || labsQueryParam === '1') {
      window.localStorage.setItem('xprof_labs_enabled', 'true');
    } else if (labsQueryParam === 'false' || labsQueryParam === '0') {
      window.localStorage.removeItem('xprof_labs_enabled');
    }

    // Route redirection guard: check if Labs feature flag is persistent in localStorage
    this.isLabsEnabled =
      window.localStorage.getItem('xprof_labs_enabled') === 'true';
    if (!this.isLabsEnabled) {
      // Redirect manual URL direct bypasses back to Overview Page
      const queryParams = this.route.snapshot.queryParams;
      // Wrap in setTimeout to schedule navigation after current routing activation completes
      setTimeout(() => {
        this.router.navigate(['/overview_page', this.sessionId], {
          queryParams,
          replaceUrl: true,
        });
      }, 0);
      return;
    }

    this.store.dispatch(setCurrentToolStateAction({currentTool: 'labs'}));
    this.store.dispatch(
      setLoadingStateAction({
        loadingState: {loading: false, message: ''},
      }),
    );

    this.updateFilteredTools();
  }

  onSearchChange(event: Event) {
    const input = event.target as HTMLInputElement;
    this.searchQuery = input.value;
    this.updateFilteredTools();
  }

  onCategoryChange(category: string) {
    this.selectedCategory = category;
    this.updateFilteredTools();
  }

  private updateFilteredTools() {
    const query = this.searchQuery.trim().toLowerCase();

    this.featuredTools = this.curatedTools.filter(
      (tool) =>
        tool.isFeatured &&
        this.matchesSearch(tool, query) &&
        this.matchesCategory(tool),
    );

    this.regularTools = this.curatedTools.filter(
      (tool) =>
        !tool.isFeatured &&
        this.matchesSearch(tool, query) &&
        this.matchesCategory(tool),
    );

    this.cdr.markForCheck();
  }

  private matchesSearch(tool: CuratedTool, query: string): boolean {
    if (!query) return true;
    return (
      tool.label.toLowerCase().includes(query) ||
      tool.description.toLowerCase().includes(query)
    );
  }

  private matchesCategory(tool: CuratedTool): boolean {
    if (this.selectedCategory === 'All') return true;
    if (this.selectedCategory === 'Featured') return tool.isFeatured;
    return tool.tags.includes(this.selectedCategory);
  }

  trackByTool(index: number, tool: CuratedTool): string {
    return tool.key;
  }

  onLaunchTool(tool: CuratedTool) {
    if (tool.route) {
      this.router.navigate([tool.route, this.sessionId], {
        queryParams: this.route.snapshot.queryParams,
      });
    } else if (tool.url) {
      const sanitizedUrl = trySanitizeUrl(tool.url);
      if (sanitizedUrl) {
        windowOpen(window, sanitizedUrl, '_blank');
      } else {
        console.error('Blocked navigation to untrusted URL: ', tool.url);
        this.snackBar.open(
          'Navigation blocked: unsafe link detected.',
          'Close',
          {
            duration: 4000,
          },
        );
      }
    } else {
      this.snackBar.open(
        `${tool.label} is an upcoming curated experiment. Check back soon!`,
        'Close',
        {duration: 4000},
      );
    }
  }
}
