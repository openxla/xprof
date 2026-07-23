import {
  ChangeDetectionStrategy,
  Component,
  computed,
  DestroyRef,
  inject,
  OnInit,
  signal,
} from '@angular/core';
import {takeUntilDestroyed} from '@angular/core/rxjs-interop';
import {MatSnackBar} from '@angular/material/snack-bar';
import {ActivatedRoute, Router} from '@angular/router';
import {Store} from '@ngrx/store';
import {
  setCurrentToolStateAction,
  setLoadingStateAction,
} from 'org_xprof/frontend/app/store/actions';
import {trySanitizeUrl} from 'safevalues';
import {windowOpen} from 'safevalues/dom';
import {CuratedTool} from './curated_tool';
import {CuratedToolsService} from './curated_tools.service';

function matchesSearch(tool: CuratedTool, query: string): boolean {
  if (!query) return true;
  return (
    tool.label.toLowerCase().includes(query) ||
    tool.description.toLowerCase().includes(query)
  );
}

const CATEGORIES = [
  'All',
  'Favorite',
  'Featured',
  'Memory',
  'Compiler',
  'Network',
] as const;

type Category = (typeof CATEGORIES)[number];

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
  private readonly store = inject(Store);
  private readonly curatedToolsService = inject(CuratedToolsService);
  private readonly destroyRef = inject(DestroyRef);

  private sessionId = '';
  readonly searchQuery = signal('');
  readonly selectedCategory = signal<Category>('All');
  readonly isLabsEnabled = signal(false);
  readonly activeView = signal<'experiments' | 'playground'>('experiments');
  readonly activeTab = signal<'curated' | 'my-tools'>('curated');
  readonly displayMode = signal<'grid' | 'list'>('grid');

  private readonly favoriteToolKeys = signal(new Set<string>());
  private readonly curatedTools = signal<readonly CuratedTool[]>([]);

  readonly filteredCuratedTools = computed<readonly CuratedTool[]>(() => {
    const query = this.searchQuery().trim().toLowerCase();
    const category = this.selectedCategory();
    const isAll = category === 'All';
    const isFeatured = category === 'Featured';
    const isFavorite = category === 'Favorite';
    const favorites = this.favoriteToolKeys();

    return this.curatedTools()
      .filter((tool) => {
        if (!matchesSearch(tool, query)) return false;
        if (isAll) return true;
        if (isFeatured) return tool.isFeatured;
        if (isFavorite) return favorites.has(tool.key);
        return tool.tags.includes(category);
      })
      .map((tool) => ({
        ...tool,
        isFavorite: favorites.has(tool.key),
        lifecycleClass: tool.lifecycle?.toLowerCase() ?? '',
        statusTextClass: tool.statusText?.toLowerCase() ?? '',
      }));
  });

  readonly myTools = computed<readonly CuratedTool[]>(() => []);

  readonly searchResultsCount = computed<number>(() => {
    return this.activeTab() === 'curated'
      ? this.filteredCuratedTools().length
      : this.myTools().length;
  });

  readonly isPlaygroundRunDisabled = computed<boolean>(() => {
    return !this.playgroundCodeContent().trim() || this.isPlaygroundRunning();
  });

  readonly isAskAiDisabled = computed<boolean>(() => {
    return this.isAiGenerating() || !this.playgroundAiPrompt().trim();
  });

  readonly playgroundCodeContent = signal('');
  readonly playgroundAiPrompt = signal('');
  readonly isAiGenerating = signal(false);
  readonly isPlaygroundRunning = signal(false);
  readonly playgroundStatusText = signal('');
  readonly hasPlaygroundChart = signal(false);

  setActiveView(view: 'experiments' | 'playground'): void {
    this.activeView.set(view);
  }

  setActiveTab(tab: 'curated' | 'my-tools'): void {
    this.activeTab.set(tab);
  }

  setDisplayMode(mode: 'grid' | 'list'): void {
    this.displayMode.set(mode);
  }

  readonly categories = CATEGORIES;

  ngOnInit(): void {
    this.sessionId = this.route.snapshot.paramMap.get('sessionId') ?? '';

    const labsQueryParam = this.route.snapshot.queryParamMap.get('labs');
    const isEnabled =
      labsQueryParam !== null
        ? labsQueryParam === 'true' || labsQueryParam === '1'
        : window.localStorage.getItem('xprof_labs_enabled') === 'true';

    if (labsQueryParam !== null) {
      if (labsQueryParam === 'true' || labsQueryParam === '1') {
        window.localStorage.setItem('xprof_labs_enabled', 'true');
      } else if (labsQueryParam === 'false' || labsQueryParam === '0') {
        window.localStorage.removeItem('xprof_labs_enabled');
      }
    }

    this.isLabsEnabled.set(isEnabled);

    if (!this.isLabsEnabled()) {
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

    this.curatedToolsService
      .getCuratedTools()
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe((tools) => {
        this.curatedTools.set(tools);
      });
  }

  onSearchChange(event: Event): void {
    if (event.target instanceof HTMLInputElement) {
      this.searchQuery.set(event.target.value);
    }
  }

  onCategoryChange(category: Category): void {
    this.selectedCategory.set(category);
  }

  clearSearch(): void {
    this.searchQuery.set('');
  }

  toggleFavorite(tool: CuratedTool): void {
    this.favoriteToolKeys.update((keys) => {
      const newKeys = new Set(keys);
      if (newKeys.has(tool.key)) {
        newKeys.delete(tool.key);
      } else {
        newKeys.add(tool.key);
      }
      return newKeys;
    });
  }

  onLaunchTool(tool: CuratedTool): void {
    if (tool.route) {
      this.router.navigate([tool.route, this.sessionId], {
        queryParams: this.route.snapshot.queryParams,
      });
      return;
    }

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
    if (this.isPlaygroundRunning()) return;
    this.isPlaygroundRunning.set(true);
    this.playgroundStatusText.set('Running...');
    setTimeout(() => {
      this.isPlaygroundRunning.set(false);
      this.playgroundStatusText.set('Completed (0.42s)');
      this.hasPlaygroundChart.set(true);
    }, 1000);
  }

  askAi(): void {
    if (this.isAiGenerating() || !this.playgroundAiPrompt().trim()) return;
    this.isAiGenerating.set(true);
    setTimeout(() => {
      this.isAiGenerating.set(false);
      this.playgroundCodeContent.set(
        `# AI Generated Visualization\nimport xprof\nimport matplotlib.pyplot as plt\n\n# Fetch profile session data\nsession = xprof.get_session('${this.sessionId}')\ntpu_metrics = session.get_tpu_compilation_metrics()\n\nplt.figure(figsize=(10, 6))\nplt.plot(tpu_metrics['timestamps'], tpu_metrics['overhead'], color='#0b57d0')\nplt.title('TPU 0 Instruction Compilation Overhead')\nplt.xlabel('Time (s)')\nplt.ylabel('Overhead (ms)')\nplt.grid(True)\nplt.show()`,
      );
      this.playgroundAiPrompt.set('');
    }, 1500);
  }

  onCodeChange(event: Event): void {
    if (event.target instanceof HTMLTextAreaElement) {
      this.playgroundCodeContent.set(event.target.value);
    }
  }

  onAiPromptChange(event: Event): void {
    if (event.target instanceof HTMLInputElement) {
      this.playgroundAiPrompt.set(event.target.value);
    }
  }
}
