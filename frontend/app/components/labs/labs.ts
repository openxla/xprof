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
  readonly key: string;
  readonly label: string;
  readonly status: 'Alpha' | 'Beta';
  readonly statusText?: string;
  readonly description: string;
  readonly icon: string;
  readonly url?: string;
  readonly tags: readonly string[];
  readonly isFeatured: boolean;
  readonly owner?: string;
  readonly buganizer?: string;
  readonly lifecycle?: string;
  readonly mau?: number;
  readonly colorClass?: string;
}

function matchesSearch(tool: CuratedTool, query: string): boolean {
  if (!query) return true;
  return (
    tool.label.toLowerCase().includes(query) ||
    tool.description.toLowerCase().includes(query)
  );
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
  activeTab: 'curated' | 'my-tools' = 'curated';
  displayMode: 'grid' | 'list' = 'grid';

  private readonly favoriteToolKeys = new Set<string>();

  featuredTools: CuratedTool[] = [];
  regularTools: CuratedTool[] = [];
  filteredCuratedTools: CuratedTool[] = [];
  myTools: CuratedTool[] = [];

  playgroundCodeContent = '';
  playgroundAiPrompt = '';
  isAiGenerating = false;
  isPlaygroundRunning = false;
  playgroundStatusText = '';
  hasPlaygroundChart = false;

  setActiveView(view: 'experiments' | 'playground'): void {
    this.activeView = view;
    this.cdr.markForCheck();
  }

  setActiveTab(tab: 'curated' | 'my-tools'): void {
    this.activeTab = tab;
    this.cdr.markForCheck();
  }

  setDisplayMode(mode: 'grid' | 'list'): void {
    this.displayMode = mode;
    this.cdr.markForCheck();
  }

  readonly categories = [
    'All',
    'Favorite',
    'Featured',
    'Memory',
    'Compiler',
    'Network',
  ];

  private readonly curatedTools: CuratedTool[] = [
    {
      key: 'memory_analysis',
      label: 'Memory Analysis Tool',
      status: 'Beta',
      statusText: 'Active',
      description:
        'Parses HLO memory profiles, models parameter weights, and exposes memory consumption breakdowns and peak-memory bottlenecks.',
      icon: 'memory',
      tags: ['Memory'],
      isFeatured: true,
      owner: 'hinsu@',
      buganizer: 'b/314159265',
      lifecycle: 'Curated',
      mau: 420,
      colorClass: 'blue',
    },
    {
      key: 'tpu_viz',
      label: 'TPU-viz (Pod Topology)',
      status: 'Alpha',
      statusText: 'Active',
      description:
        'Beautiful, 3D pod-level visualization showing physical network topologies, optical switches, and message routing paths.',
      icon: 'grid_view',
      tags: ['Network'],
      isFeatured: true,
      owner: 'benalbrecht@',
      buganizer: 'b/271828182',
      lifecycle: 'Curated',
      mau: 512,
      colorClass: 'purple',
    },
    {
      key: 'lunc_explorer',
      label: 'Lunc Compiler Explorer',
      status: 'Alpha',
      statusText: 'Experimental',
      description:
        'Interactive visualizer showing compiler HLO graph pass transformations and TPU instruction schedules.',
      icon: 'schema',
      tags: ['Compiler'],
      isFeatured: false,
      owner: 'compiler-ops@',
      buganizer: 'b/161803398',
      lifecycle: 'Incubation',
      mau: 98,
      colorClass: 'amber',
    },
    {
      key: 'megascale_plugin',
      label: 'Megascale GRPC Plugin',
      status: 'Alpha',
      statusText: 'Verified',
      description:
        'Helps diagnose network latency bottlenecks and collective communication overheads across thousands of nodes.',
      icon: 'lan',
      tags: ['Network'],
      isFeatured: false,
      owner: 'tpu-systems@',
      buganizer: 'b/987654321',
      lifecycle: 'Community',
      mau: 245,
      colorClass: 'teal',
    },
    {
      key: 'youtube_profiler',
      label: 'YouTube Custom Profiler',
      status: 'Alpha',
      statusText: 'Verified',
      description:
        'Bespoke tool to profile video decoding threads, buffering loops, and rendering pipeline pipelines.',
      icon: 'play_circle_outline',
      tags: ['Memory'],
      isFeatured: false,
      owner: 'yt-perf-team@',
      buganizer: 'b/555121212',
      lifecycle: 'Community',
      mau: 132,
      colorClass: 'red',
    },
  ];

  ngOnInit(): void {
    this.sessionId = this.route.snapshot.paramMap.get('sessionId') ?? '';

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
      this.router.navigate(['/overview_page', this.sessionId], {
        queryParams,
        replaceUrl: true,
      });
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

  onSearchChange(event: Event): void {
    // Type assertion necessary because unit tests pass mock event object literals rather than actual DOM HTMLInputElement instances.
    const input = event.target as HTMLInputElement;
    this.searchQuery = input.value;
    this.updateFilteredTools();
  }

  onCategoryChange(category: string): void {
    this.selectedCategory = category;
    this.updateFilteredTools();
  }

  clearSearch(): void {
    this.searchQuery = '';
    this.updateFilteredTools();
  }

  toggleFavorite(tool: CuratedTool): void {
    if (this.favoriteToolKeys.has(tool.key)) {
      this.favoriteToolKeys.delete(tool.key);
    } else {
      this.favoriteToolKeys.add(tool.key);
    }
    this.updateFilteredTools();
  }

  isFavorite(tool: CuratedTool): boolean {
    return this.favoriteToolKeys.has(tool.key);
  }

  private updateFilteredTools(): void {
    const query = this.searchQuery.trim().toLowerCase();

    this.filteredCuratedTools = this.curatedTools.filter(
      (tool) => matchesSearch(tool, query) && this.matchesCategory(tool),
    );

    this.featuredTools = this.filteredCuratedTools.filter(
      (tool) => tool.isFeatured,
    );
    this.regularTools = this.filteredCuratedTools.filter(
      (tool) => !tool.isFeatured,
    );

    this.cdr.markForCheck();
  }

  private matchesCategory(tool: CuratedTool): boolean {
    if (this.selectedCategory === 'All') return true;
    if (this.selectedCategory === 'Featured') return tool.isFeatured;
    if (this.selectedCategory === 'Favorite') {
      return this.favoriteToolKeys.has(tool.key);
    }
    return tool.tags.includes(this.selectedCategory);
  }

  trackByTool(index: number, tool: CuratedTool): string {
    return tool.key;
  }

  onLaunchTool(tool: CuratedTool): void {
    if (!tool.url) {
      this.snackBar.open(
        `${tool.label} is an upcoming curated experiment. Check back soon!`,
        'Close',
        {duration: 4000},
      );
      return;
    }

    const sanitizedUrl = trySanitizeUrl(tool.url);
    if (!sanitizedUrl) {
      console.error('Blocked navigation to untrusted URL: ', tool.url);
      this.snackBar.open('Navigation blocked: unsafe link detected.', 'Close', {
        duration: 4000,
      });
      return;
    }

    windowOpen(window, sanitizedUrl, '_blank');
  }

  runPlayground(): void {
    if (this.isPlaygroundRunning) return;
    this.isPlaygroundRunning = true;
    this.playgroundStatusText = 'Running...';
    this.cdr.markForCheck();
    setTimeout(() => {
      this.isPlaygroundRunning = false;
      this.playgroundStatusText = 'Completed (0.42s)';
      this.hasPlaygroundChart = true;
      this.cdr.markForCheck();
    }, 1000);
  }

  askAi(): void {
    if (this.isAiGenerating || !this.playgroundAiPrompt.trim()) return;
    this.isAiGenerating = true;
    this.cdr.markForCheck();
    setTimeout(() => {
      this.isAiGenerating = false;
      this.playgroundCodeContent = `# AI Generated Visualization\nimport xprof\nimport matplotlib.pyplot as plt\n\n# Fetch profile session data\nsession = xprof.get_session('${this.sessionId}')\ntpu_metrics = session.get_tpu_compilation_metrics()\n\nplt.figure(figsize=(10, 6))\nplt.plot(tpu_metrics['timestamps'], tpu_metrics['overhead'], color='#0b57d0')\nplt.title('TPU 0 Instruction Compilation Overhead')\nplt.xlabel('Time (s)')\nplt.ylabel('Overhead (ms)')\nplt.grid(True)\nplt.show()`;
      this.playgroundAiPrompt = '';
      this.cdr.markForCheck();
    }, 1500);
  }

  onCodeChange(event: Event): void {
    // Type assertion necessary to support mock event object literals in testing environments.
    const textarea = event.target as HTMLTextAreaElement;
    this.playgroundCodeContent = textarea.value;
  }

  onAiPromptChange(event: Event): void {
    // Type assertion necessary to support mock event object literals in testing environments.
    const input = event.target as HTMLInputElement;
    this.playgroundAiPrompt = input.value;
  }
}
