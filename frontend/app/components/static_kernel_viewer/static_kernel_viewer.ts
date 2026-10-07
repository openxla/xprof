import {Location} from '@angular/common';
import {
  AfterViewInit,
  ChangeDetectionStrategy,
  ChangeDetectorRef,
  Component,
  ElementRef,
  inject,
  OnDestroy,
  OnInit,
  ViewChild,
} from '@angular/core';
import {ActivatedRoute, Params} from '@angular/router';
import {Store} from '@ngrx/store';
import {DEFAULT_HOST} from 'org_xprof/frontend/app/common/constants/constants';
import {
  shutdownTraceViewerV2,
  traceViewerV2Main,
  TraceViewerV2Module,
} from 'org_xprof/frontend/app/components/trace_viewer_v2/main';
import {
  DATA_SERVICE_INTERFACE_TOKEN,
  DataServiceV2Interface,
} from 'org_xprof/frontend/app/services/data_service_v2/data_service_v2_interface';
import {setCurrentToolStateAction} from 'org_xprof/frontend/app/store/actions';
import {combineLatest, firstValueFrom, ReplaySubject} from 'rxjs';
import {takeUntil} from 'rxjs/operators';

/** Response structure from the Kernel Viewer backend list request. */
declare interface KernelListResponse {
  // tslint:disable-next-line:enforce-name-casing
  readonly hlo_modules?: readonly string[];
  // tslint:disable-next-line:enforce-name-casing
  readonly module_kernels?: Record<string, readonly string[]>;
  readonly kernels?: readonly string[];
}

/** A text or keyboard shortcut token in the footer of a details card. */
export interface HintPart {
  readonly text: string;
  readonly isKey: boolean;
}

/** A kernel that can be opened in the viewer. */
export interface KernelEntry {
  /** HLO module of the kernel, or '' if the backend reports no modules. */
  readonly module: string;
  /** Kernel name, as reported by the backend. */
  readonly kernel: string;
  /**
   * Kernel name split after each `_` and `-`, where the details card may wrap
   * it, e.g. `['flash_', 'attention_', 'bwd.3']`.
   */
  readonly nameParts: readonly string[];
  /** Zero-based position in the kernel list. */
  readonly index: number;
  /** Short label shown while the kernel rail is collapsed. */
  readonly monogram: string;
  /** Color of the kernel's module group. */
  readonly color: string;
  /** Module name, shown below the kernel name in its tab. */
  readonly subtitle: string;
  /** Accessible description of the kernel. */
  readonly description: string;
}

/** The kernels of one HLO module, shown like a browser tab group. */
export interface KernelGroup {
  readonly module: string;
  readonly color: string;
  readonly entries: readonly KernelEntry[];
}

/** A kernel opened in a tab of the kernel bar. */
export interface KernelTab {
  /** Identifies the tab while its kernel changes. */
  readonly id: number;
  /** Kernel shown in the tab. Stepping through kernels replaces it. */
  entry: KernelEntry;
  /** When the tab was last active, to close the least recently used tab. */
  lastUsed: number;
}

/** Details card of a hovered kernel row or tab. */
export interface KernelCard {
  readonly entry: KernelEntry;
  /** Whether the card belongs to a row of the kernel rail or to a tab. */
  readonly source: 'rail' | 'tab';
  /** Footer, e.g. the shortcut that switches to the kernel's tab. */
  readonly hint: string;
  /** `hint`, split into plain text and keyboard shortcut tokens. */
  readonly hintParts: readonly HintPart[];
  /** Position relative to the component, in pixels. */
  readonly left: number;
  readonly top?: number;
  readonly bottom?: number;
}

/** Colors assigned to module groups, in order (browser tab group colors). */
const GROUP_COLORS = [
  '#1a73e8',
  '#d93025',
  '#e37400',
  '#188038',
  '#d01884',
  '#9334e6',
  '#007b83',
  '#e8710a',
  '#5f6368',
];

/** Delay before the collapsed kernel rail expands on hover. */
const RAIL_EXPAND_DELAY_MS = 150;

/** Delay before the hover-expanded kernel rail collapses again. */
const RAIL_COLLAPSE_DELAY_MS = 250;

/** Duration of the kernel rail width transition, see the stylesheet. */
const RAIL_TRANSITION_MS = 200;

/** How long the copy link button shows its confirmation. */
const LINK_COPIED_FEEDBACK_MS = 1500;

/** Local storage key of the kernel rail pin preference. */
const RAIL_PINNED_STORAGE_KEY = 'xprof_static_kernel_viewer_rail_pinned';

/** Maximum number of tabs. Opening another closes the least recently used. */
const MAX_OPEN_TABS = 8;

/** How long a tab opened in the background stays highlighted. */
const TAB_HIGHLIGHT_MS = 900;

/** Delay before the details card of a hovered kernel appears. */
const CARD_SHOW_DELAY_MS = 400;

/** Delay before the details card hides, so it can move to a neighbor. */
const CARD_HIDE_DELAY_MS = 100;

/** Width of the details card, see the stylesheet. */
const CARD_WIDTH = 320;

/** Distance between the details card and its kernel. */
const CARD_GAP = 8;

/** Keys that only modify other keys. */
const MODIFIER_KEYS = new Set(['Alt', 'Control', 'Meta', 'Shift']);

const IS_APPLE_PLATFORM = /Mac|iPhone|iPad/.test(navigator.userAgent);

/** Label of the modifier key that opens kernels in background tabs. */
const BACKGROUND_CLICK_KEY = IS_APPLE_PLATFORM ? '⌘' : 'Ctrl';

/**
 * Returns a two-letter label for a kernel, built from its first two words,
 * e.g. `add_add_fusion.114` -> `Aa` and `MLA-bd-bq_1` -> `Mb`.
 */
export function kernelMonogram(kernel: string): string {
  const [first = '', second = ''] =
    kernel.match(/[A-Z]?[a-z]+|[A-Z]+(?![a-z])/g) ?? [];
  if (!first) {
    return kernel.slice(0, 2).toUpperCase() || '?';
  }
  const secondLetter = (second || first.slice(1)).charAt(0);
  return first.charAt(0).toUpperCase() + secondLetter.toLowerCase();
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function buildCardHint(parts: readonly HintPart[]): {
  hint: string;
  hintParts: readonly HintPart[];
} {
  return {
    hint: parts.map((part) => part.text).join(''),
    hintParts: parts,
  };
}

function createEntry(
  module: string,
  kernel: string,
  index: number,
  color: string,
): KernelEntry {
  return {
    module,
    kernel,
    nameParts: kernel.match(/[^-_]+[-_]*|[-_]+/g) ?? [kernel],
    index,
    monogram: kernelMonogram(kernel),
    color,
    subtitle: module,
    description: [kernel, module].filter(Boolean).join(', '),
  };
}

/** Groups the kernels of a list response by HLO module, in response order. */
function buildKernelGroups(response: unknown): KernelGroup[] {
  const list = (isRecord(response) ? response : {}) as KernelListResponse;
  const moduleKernels = list.module_kernels ?? {};
  const modules = new Set([
    ...(list.hlo_modules ?? []),
    ...Object.keys(moduleKernels),
  ]);
  const groups: KernelGroup[] = [];
  const groupedKernels = new Set<string>();
  let index = 0;
  const addGroup = (module: string, kernels: readonly string[]) => {
    if (kernels.length === 0) {
      return;
    }
    const color = GROUP_COLORS[groups.length % GROUP_COLORS.length];
    const entries = kernels.map((kernel) =>
      createEntry(module, kernel, index++, color),
    );
    groups.push({module, color, entries});
  };
  for (const module of modules) {
    const kernels = moduleKernels[module] ?? [];
    for (const kernel of kernels) {
      groupedKernels.add(kernel);
    }
    addGroup(module, kernels);
  }
  // Older backends only report a flat list of kernels.
  addGroup(
    '',
    (list.kernels ?? []).filter((kernel) => !groupedKernels.has(kernel)),
  );
  return groups;
}

function isEditableTarget(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) {
    return false;
  }
  return (
    target.isContentEditable ||
    target.tagName === 'INPUT' ||
    target.tagName === 'TEXTAREA' ||
    target.tagName === 'SELECT'
  );
}

/** Scrolls `container` horizontally just enough to show all of `child`. */
function revealHorizontally(container: HTMLElement, child: HTMLElement) {
  const start = child.offsetLeft;
  const end = start + child.offsetWidth;
  if (start < container.scrollLeft) {
    container.scrollLeft = start;
  } else if (end > container.scrollLeft + container.clientWidth) {
    container.scrollLeft = end - container.clientWidth;
  }
}

function readRailPinned(): boolean {
  try {
    return window.localStorage.getItem(RAIL_PINNED_STORAGE_KEY) === 'true';
  } catch {
    // Storage can be unavailable, e.g. in sandboxed iframes.
    return false;
  }
}

function writeRailPinned(pinned: boolean) {
  try {
    window.localStorage.setItem(RAIL_PINNED_STORAGE_KEY, String(pinned));
  } catch {
    // Storage can be unavailable, e.g. in sandboxed iframes.
  }
}

/**
 * Component for the Static Kernel Viewer tool page.
 *
 * Kernels are listed in a rail on the left, grouped by HLO module like
 * vertical browser tabs. The rail stays collapsed to a narrow strip so the
 * timeline gets most of the width; it expands over the timeline on hover and
 * can be pinned open. Kernels open in tabs above the timeline.
 */
@Component({
  changeDetection: ChangeDetectionStrategy.Default,
  standalone: false,
  selector: 'static-kernel-viewer',
  templateUrl: './static_kernel_viewer.ng.html',
  styleUrls: ['./static_kernel_viewer.scss'],
})
export class StaticKernelViewer implements OnInit, AfterViewInit, OnDestroy {
  readonly tool = 'kernel_viewer';
  sessionId = '';
  host = DEFAULT_HOST;

  /** Kernels grouped by HLO module, in display order. */
  groups: KernelGroup[] = [];
  /** All kernels in display order. */
  entries: KernelEntry[] = [];
  /** `groups`, narrowed down to the kernels matching `filterQuery`. */
  visibleGroups: KernelGroup[] = [];
  /** Number of kernels in `visibleGroups`. */
  visibleCount = 0;
  /** Open kernel tabs, in display order. */
  tabs: KernelTab[] = [];
  /** The tab whose kernel the timeline shows. */
  activeTab: KernelTab | undefined = undefined;
  /** A tab that was just opened in the background, or switched to. */
  highlightedTab: KernelTab | undefined = undefined;
  /** Details card of the hovered kernel row or tab. */
  hoverCard: KernelCard | undefined = undefined;
  url = '';
  loadingKernels = false;
  loadFailed = false;
  filterQuery = '';

  /** Whether the kernel rail is docked open next to the timeline. */
  isRailPinned = readRailPinned();
  /** Whether the unpinned kernel rail is expanded over the timeline. */
  isRailPeeking = false;
  /** The Clipboard API is only available in secure contexts. */
  readonly canCopyLink = navigator.clipboard !== undefined;
  linkCopied = false;

  traceViewerModule: TraceViewerV2Module | null = null;

  @ViewChild('searchInput') searchInput?: ElementRef<HTMLInputElement>;

  private readonly collapsedModules = new Set<string>();
  private listKey = '';
  private listRequestId = 0;
  private requestedKernel = '';
  private requestedModule = '';
  private isPointerInRail = false;
  private isFocusInRail = false;
  /** Whether the last focused element was focused with the keyboard. */
  private isKeyboardFocus = false;
  private nextTabId = 0;
  private tabClock = 0;
  /** Number of tabs opened in the background since the last tab switch. */
  private backgroundTabCount = 0;
  private railTimer: ReturnType<typeof setTimeout> | undefined = undefined;
  private cardTimer: ReturnType<typeof setTimeout> | undefined = undefined;
  private highlightTimer: ReturnType<typeof setTimeout> | undefined = undefined;
  private linkCopiedTimer: ReturnType<typeof setTimeout> | undefined =
    undefined;
  private isInitializing = false;
  private isDestroyed = false;

  private readonly dataService: DataServiceV2Interface = inject(
    DATA_SERVICE_INTERFACE_TOKEN,
  );
  private readonly location = inject(Location, {optional: true});
  private readonly changeDetectorRef = inject(ChangeDetectorRef);
  private readonly elementRef = inject<ElementRef<HTMLElement>>(ElementRef);
  private readonly route = inject(ActivatedRoute);
  private readonly store = inject<Store<{}>>(Store);
  private readonly destroyed = new ReplaySubject<void>(1);
  private readonly keydownListener = (event: KeyboardEvent) => {
    this.onWindowKeydown(event);
  };

  get isRailExpanded(): boolean {
    return this.isRailPinned || this.isRailPeeking;
  }

  /** Whether the search narrows down the kernel rail. */
  get isFiltering(): boolean {
    return this.filterQuery.trim() !== '';
  }

  /** The kernel that the timeline shows. */
  get selectedEntry(): KernelEntry | undefined {
    return this.activeTab?.entry;
  }

  /** One-based position of the selected kernel, or 0 if none is selected. */
  get selectedPosition(): number {
    return this.selectedEntry ? this.selectedEntry.index + 1 : 0;
  }

  ngOnInit(): void {
    this.store.dispatch(setCurrentToolStateAction({currentTool: this.tool}));

    combineLatest([this.route.params, this.route.queryParams])
      .pipe(takeUntil(this.destroyed))
      .subscribe(([params, queryParams]: [Params, Params]) => {
        this.onRouteChange(params, queryParams);
      });
  }

  ngAfterViewInit(): void {
    // The trace viewer container registers its own keydown listener in its
    // ngAfterViewInit, which runs before this one. Registering afterwards lets
    // the shortcuts below skip keys the container already handled.
    window.addEventListener('keydown', this.keydownListener);
  }

  ngOnDestroy(): void {
    this.isDestroyed = true;
    window.removeEventListener('keydown', this.keydownListener);
    clearTimeout(this.railTimer);
    clearTimeout(this.cardTimer);
    clearTimeout(this.highlightTimer);
    clearTimeout(this.linkCopiedTimer);
    this.destroyed.next();
    this.destroyed.complete();
    if (this.traceViewerModule !== null) {
      shutdownTraceViewerV2();
      this.traceViewerModule = null;
    }
  }

  async onInitializeWasm(): Promise<void> {
    if (this.isInitializing || this.traceViewerModule !== null) {
      return;
    }
    this.isInitializing = true;
    try {
      this.traceViewerModule = await traceViewerV2Main();
      // The kernel event tooltip replaces the built-in one, which formats
      // bundle counts as durations.
      this.traceViewerModule?.SetEventTooltipEnabled?.(false);
      if (this.isDestroyed) {
        if (this.traceViewerModule !== null) {
          shutdownTraceViewerV2();
          this.traceViewerModule = null;
        }
      } else {
        // The kernel backend ignores time ranges and always returns the whole
        // kernel, so incremental refetching would only re-download it.
        this.traceViewerModule?.SetIncrementalFetchEnabled?.(false);
        if (this.url && this.traceViewerModule?.loadTraceData) {
          void this.traceViewerModule.loadTraceData(this.url);
        }
      }
    } catch (error) {
      console.error('Failed to initialize Trace Viewer V2 WASM module:', error);
    } finally {
      this.isInitializing = false;
    }
  }

  async loadKernelList(): Promise<void> {
    if (!this.sessionId) {
      return;
    }
    const requestId = ++this.listRequestId;
    this.loadingKernels = true;
    this.loadFailed = false;
    let groups: KernelGroup[] = [];
    let failed = false;
    try {
      const response: unknown = await firstValueFrom(
        this.dataService.getData(
          this.sessionId,
          this.tool,
          this.host,
          new Map<string, string>([['request_type', 'list']]),
        ),
      );
      groups = buildKernelGroups(response);
    } catch (error) {
      console.error('Failed to fetch kernel list:', error);
      failed = true;
    }
    if (this.isDestroyed || requestId !== this.listRequestId) {
      return;
    }
    this.loadingKernels = false;
    this.loadFailed = failed;
    this.setKernelGroups(groups);
    this.changeDetectorRef.markForCheck();
  }

  /**
   * Shows `entry` in the active tab, or switches to the tab that already shows
   * it. Opens the first tab if there is none.
   */
  selectEntry(entry: KernelEntry | undefined): void {
    if (!entry) {
      return;
    }
    const tab = this.findTab(entry);
    if (tab || !this.activeTab) {
      this.openEntry(entry);
      return;
    }
    this.activeTab.entry = entry;
    this.activeTab.lastUsed = ++this.tabClock;
    this.showEntry(entry);
  }

  /**
   * Opens `entry` in a new tab next to the active tab, or switches to the tab
   * that already shows it. A `background` tab opens without switching to it.
   */
  openEntry(entry: KernelEntry | undefined, background = false): void {
    if (!entry) {
      return;
    }
    let tab = this.findTab(entry);
    if (!tab) {
      if (this.tabs.length >= MAX_OPEN_TABS) {
        this.closeLeastRecentlyUsedTab();
      }
      tab = {id: this.nextTabId++, entry, lastUsed: ++this.tabClock};
      // Like links opened in the background, consecutive tabs open in order
      // after the active tab.
      const activeIndex = this.activeTab
        ? this.tabs.indexOf(this.activeTab)
        : -1;
      const index = Math.min(
        activeIndex + 1 + this.backgroundTabCount,
        this.tabs.length,
      );
      this.tabs.splice(index, 0, tab);
      if (background) {
        this.backgroundTabCount++;
      }
    }
    if (background) {
      this.highlightTab(tab);
    } else {
      this.activateTab(tab);
    }
  }

  /** Shows the kernel of `tab` in the timeline. */
  activateTab(tab: KernelTab | undefined): void {
    if (!tab || tab === this.activeTab) {
      return;
    }
    this.activeTab = tab;
    this.backgroundTabCount = 0;
    tab.lastUsed = ++this.tabClock;
    this.showEntry(tab.entry);
  }

  /**
   * Closes `tab`, and switches to its right neighbor if it was active. The
   * last tab stays open.
   */
  closeTab(tab: KernelTab): void {
    const index = this.tabs.indexOf(tab);
    if (index < 0 || this.tabs.length < 2) {
      return;
    }
    this.hideCard();
    this.tabs.splice(index, 1);
    if (tab === this.activeTab) {
      this.activeTab = undefined;
      this.activateTab(this.tabs[index] ?? this.tabs[index - 1]);
    }
  }

  /** Whether a tab shows `entry`. */
  isEntryOpen(entry: KernelEntry): boolean {
    return this.findTab(entry) !== undefined;
  }

  /** Opens the next (`delta` = 1) or previous (`delta` = -1) kernel. */
  stepKernel(delta: number): void {
    const count = this.entries.length;
    if (count === 0) {
      return;
    }
    const current = this.selectedEntry?.index;
    const next =
      current === undefined
        ? delta > 0
          ? 0
          : count - 1
        : (((current + delta) % count) + count) % count;
    this.selectEntry(this.entries[next]);
  }

  onKernelRowClick(entry: KernelEntry, event: MouseEvent): void {
    this.hideCard();
    this.openEntry(entry, event.ctrlKey || event.metaKey);
  }

  /** Opens kernels in background tabs on middle click, like links. */
  onKernelRowAuxClick(entry: KernelEntry, event: MouseEvent): void {
    if (event.button === 1) {
      event.preventDefault();
      this.hideCard();
      this.openEntry(entry, true);
    }
  }

  onTabClick(tab: KernelTab): void {
    this.hideCard();
    this.activateTab(tab);
  }

  /** Closes tabs on middle click, like browser tabs. */
  onTabAuxClick(tab: KernelTab, event: MouseEvent): void {
    if (event.button === 1) {
      event.preventDefault();
      this.closeTab(tab);
    }
  }

  onCloseTabClick(tab: KernelTab): void {
    const restoreFocus = this.isKeyboardFocus;
    this.closeTab(tab);
    if (restoreFocus) {
      // The focused close button is gone, so focus the tab that replaced it.
      setTimeout(() => {
        this.elementRef.nativeElement
          .querySelector<HTMLElement>('.kernel-tab.active .tab-button')
          ?.focus();
      });
    }
  }

  /** Keeps middle clicks on kernels and tabs from starting to autoscroll. */
  onMouseDown(event: MouseEvent): void {
    if (event.button === 1) {
      event.preventDefault();
    }
  }

  /** Shows the details card of a hovered or keyboard-focused kernel row. */
  onKernelRowHover(entry: KernelEntry, event: Event): void {
    const row = event.currentTarget;
    if (!(row instanceof HTMLElement) || !this.isHoverOrKeyboardFocus(event)) {
      return;
    }
    // Rows only show cards once the rail has expanded to show their names.
    this.scheduleCard(() =>
      this.isRailExpanded ? this.cardBesideRow(entry, row) : undefined,
    );
  }

  /** Shows the details card of a hovered or keyboard-focused tab. */
  onTabHover(tab: KernelTab, event: Event): void {
    const target = event.currentTarget;
    const tabElement =
      target instanceof HTMLElement ? target.closest('.kernel-tab') : null;
    if (!tabElement || !this.isHoverOrKeyboardFocus(event)) {
      return;
    }
    this.scheduleCard(() => this.cardBelowTab(tab, tabElement));
  }

  /** Hides the details card. */
  hideCard(): void {
    clearTimeout(this.cardTimer);
    this.cardTimer = undefined;
    this.hoverCard = undefined;
  }

  /** Hides the details card soon, unless the pointer moves to a neighbor. */
  hideCardSoon(): void {
    clearTimeout(this.cardTimer);
    this.cardTimer = setTimeout(() => {
      this.hideCard();
      this.changeDetectorRef.markForCheck();
    }, CARD_HIDE_DELAY_MS);
  }

  onRailScroll(): void {
    this.hideCardFrom('rail');
  }

  onTabListScroll(): void {
    this.hideCardFrom('tab');
  }

  onFilterChange(query: string): void {
    this.filterQuery = query;
    this.updateVisibleGroups();
  }

  clearFilter(): void {
    this.onFilterChange('');
  }

  onSearchKeydown(event: KeyboardEvent): void {
    if (event.key === 'Enter') {
      event.preventDefault();
      if (this.isFiltering) {
        this.openEntry(
          this.visibleGroups[0]?.entries[0],
          event.ctrlKey || event.metaKey,
        );
      }
      this.dismissRail();
    } else if (event.key === 'Escape') {
      event.preventDefault();
      if (this.filterQuery) {
        this.clearFilter();
      } else {
        this.dismissRail();
      }
    }
  }

  /** Whether the kernels of `group` are listed; searching expands all groups. */
  isGroupExpanded(group: KernelGroup): boolean {
    return this.isFiltering || !this.collapsedModules.has(group.module);
  }

  toggleGroup(group: KernelGroup): void {
    if (this.collapsedModules.has(group.module)) {
      this.collapsedModules.delete(group.module);
    } else {
      this.collapsedModules.add(group.module);
    }
  }

  togglePin(): void {
    this.isRailPinned = !this.isRailPinned;
    // Collapse right away when unpinning, like a browser's vertical tabs: the
    // rail only expands again once the pointer re-enters it.
    this.isRailPeeking = false;
    clearTimeout(this.railTimer);
    if (!this.isRailPinned) {
      this.clearFilter();
      this.hideCardFrom('rail');
    }
    writeRailPinned(this.isRailPinned);
    this.notifyResize();
  }

  onRailMouseEnter(): void {
    this.isPointerInRail = true;
    this.setRailPeeking(true, RAIL_EXPAND_DELAY_MS);
  }

  onRailMouseLeave(): void {
    this.isPointerInRail = false;
    if (!this.isFocusInRail) {
      this.setRailPeeking(false, RAIL_COLLAPSE_DELAY_MS);
    }
  }

  onRailFocusIn(event: FocusEvent): void {
    // Keep the rail open while typing a search or navigating with the
    // keyboard, but not after a mouse click on a kernel.
    const target = event.target;
    this.isFocusInRail =
      target instanceof HTMLElement &&
      (target.tagName === 'INPUT' || target.matches(':focus-visible'));
    if (this.isFocusInRail) {
      this.setRailPeeking(true, 0);
    }
  }

  onRailFocusOut(event: FocusEvent): void {
    const rail = event.currentTarget;
    const next = event.relatedTarget;
    if (rail instanceof Node && next instanceof Node && rail.contains(next)) {
      return;
    }
    this.isFocusInRail = false;
    if (!this.isPointerInRail) {
      this.setRailPeeking(false, RAIL_COLLAPSE_DELAY_MS);
    }
  }

  onLayoutFocusIn(event: FocusEvent): void {
    const target = event.target;
    this.isKeyboardFocus =
      target instanceof HTMLElement &&
      (isEditableTarget(target) || target.matches(':focus-visible'));
  }

  /** Expands the kernel rail and focuses its search box. */
  focusSearch(): void {
    this.hideCard();
    this.setRailPeeking(true, 0);
    this.searchInput?.nativeElement.focus();
    this.searchInput?.nativeElement.select();
  }

  /** Copies a link that opens the viewer at the selected kernel. */
  copyKernelLink(): void {
    const entry = this.selectedEntry;
    if (!entry || !this.canCopyLink) {
      return;
    }
    const link = new URL(window.location.href);
    link.searchParams.delete('kernel');
    link.searchParams.set('kernel_name', entry.kernel);
    if (entry.module) {
      link.searchParams.set('hlo_module', entry.module);
    } else {
      link.searchParams.delete('hlo_module');
    }
    navigator.clipboard.writeText(link.toString()).then(
      () => {
        if (this.isDestroyed) {
          return;
        }
        this.linkCopied = true;
        clearTimeout(this.linkCopiedTimer);
        this.linkCopiedTimer = setTimeout(() => {
          this.linkCopied = false;
          this.changeDetectorRef.markForCheck();
        }, LINK_COPIED_FEEDBACK_MS);
        this.changeDetectorRef.markForCheck();
      },
      () => {
        // Clipboard access can be denied; there is nothing else to do.
      },
    );
  }

  trackByModule(index: number, group: KernelGroup): string {
    return group.module;
  }

  trackByEntry(index: number, entry: KernelEntry): string {
    return `${entry.module}/${entry.kernel}`;
  }

  trackByTab(index: number, tab: KernelTab): number {
    return tab.id;
  }

  private onRouteChange(params: Params, queryParams: Params): void {
    const searchParams = this.dataService.getSearchParams?.();
    const getParam = (key: string): string | undefined => {
      const fromRoute = queryParams[key];
      if (typeof fromRoute === 'string' && fromRoute) {
        return fromRoute;
      }
      const fromSearch = searchParams?.get(key);
      return fromSearch ? fromSearch : undefined;
    };

    this.sessionId =
      params['sessionId'] ??
      getParam('run') ??
      getParam('sessionId') ??
      this.sessionId;
    this.host = getParam('host') ?? DEFAULT_HOST;
    const hasRouteKernelOrModule =
      queryParams['kernel_name'] !== undefined ||
      queryParams['kernel'] !== undefined ||
      queryParams['hlo_module'] !== undefined;
    this.requestedKernel = hasRouteKernelOrModule
      ? (queryParams['kernel_name'] ?? queryParams['kernel'] ?? '')
      : (getParam('kernel_name') ?? getParam('kernel') ?? '');
    this.requestedModule = hasRouteKernelOrModule
      ? (queryParams['hlo_module'] ?? '')
      : (getParam('hlo_module') ?? '');

    // Only refetch the kernel list when the profile changes. Other query
    // parameter changes at most switch the open kernel.
    const listKey = `${this.sessionId}|${this.host}`;
    if (listKey !== this.listKey) {
      this.listKey = listKey;
      void this.loadKernelList();
    } else {
      this.selectEntry(this.findRequestedEntry());
    }
  }

  private setKernelGroups(groups: KernelGroup[]): void {
    this.groups = groups;
    this.entries = groups.flatMap((group) => [...group.entries]);
    this.collapsedModules.clear();
    this.updateVisibleGroups();
    this.tabs = [];
    this.activeTab = undefined;
    this.highlightedTab = undefined;
    this.backgroundTabCount = 0;
    this.hideCard();
    this.url = '';
    const initialEntry = this.findRequestedEntry() ?? this.entries[0];
    if (initialEntry) {
      this.selectEntry(initialEntry);
    } else {
      this.updateUrlQueryParams();
    }
  }

  private findRequestedEntry(): KernelEntry | undefined {
    if (!this.requestedKernel) {
      return undefined;
    }
    const matches = this.entries.filter(
      (entry) => entry.kernel === this.requestedKernel,
    );
    return (
      matches.find((entry) => entry.module === this.requestedModule) ??
      matches[0]
    );
  }

  private findTab(entry: KernelEntry): KernelTab | undefined {
    return this.tabs.find((tab) => tab.entry === entry);
  }

  /** Loads the trace of `entry` into the timeline. */
  private showEntry(entry: KernelEntry): void {
    this.collapsedModules.delete(entry.module);
    this.updateUrlQueryParams();
    const queryParamsMap = new Map<string, string>([
      ['request_type', 'trace'],
      ['kernel_name', entry.kernel],
    ]);
    // Kernel names are only unique within a module.
    if (entry.module) {
      queryParamsMap.set('hlo_module', entry.module);
    }
    this.url = this.dataService.getDataUrl(
      this.sessionId,
      this.tool,
      this.host,
      queryParamsMap,
    );
    if (this.traceViewerModule?.loadTraceData) {
      this.traceViewerModule.application?.instance?.()?.dataProvider?.();
      this.traceViewerModule.processTraceEvents?.({traceEvents: []}, undefined);
      void this.traceViewerModule.loadTraceData(this.url);
    }
    this.scrollIntoView('.kernel-row.active', '.kernel-tab.active');
  }

  private updateUrlQueryParams(): void {
    const entry = this.selectedEntry;
    const applyParams = (params: URLSearchParams) => {
      if (entry?.module) {
        params.set('hlo_module', entry.module);
      } else {
        params.delete('hlo_module');
      }

      if (entry?.kernel) {
        params.set('kernel_name', entry.kernel);
      } else {
        params.delete('kernel_name');
      }
      params.delete('kernel');
    };

    if (this.dataService.getSearchParams && this.dataService.setSearchParams) {
      const searchParams = this.dataService.getSearchParams();
      applyParams(searchParams);
      this.dataService.setSearchParams(searchParams);
    }

    if (this.location) {
      const url = new URL(this.location.path(), window.location.origin);
      const searchParams = new URLSearchParams(url.search);
      applyParams(searchParams);
      const newSearch = searchParams.toString();
      const currentSearch = url.search.startsWith('?')
        ? url.search.slice(1)
        : url.search;
      if (currentSearch !== newSearch) {
        this.location.replaceState(url.pathname, decodeURIComponent(newSearch));
      }
    }
  }

  private closeLeastRecentlyUsedTab(): void {
    let leastRecent: KernelTab | undefined = undefined;
    for (const tab of this.tabs) {
      if (
        tab !== this.activeTab &&
        (!leastRecent || tab.lastUsed < leastRecent.lastUsed)
      ) {
        leastRecent = tab;
      }
    }
    if (leastRecent) {
      this.closeTab(leastRecent);
    }
  }

  /** Briefly highlights `tab`, e.g. after opening it in the background. */
  private highlightTab(tab: KernelTab): void {
    this.highlightedTab = tab;
    clearTimeout(this.highlightTimer);
    this.highlightTimer = setTimeout(() => {
      this.highlightedTab = undefined;
      this.changeDetectorRef.markForCheck();
    }, TAB_HIGHLIGHT_MS);
    this.scrollIntoView(undefined, '.kernel-tab.highlighted');
  }

  private updateVisibleGroups(): void {
    const query = this.filterQuery.trim().toLowerCase();
    if (!query) {
      this.visibleGroups = this.groups;
    } else {
      // A module name match keeps the whole group.
      this.visibleGroups = this.groups
        .map((group) =>
          group.module.toLowerCase().includes(query)
            ? group
            : {
                ...group,
                entries: group.entries.filter((entry) =>
                  entry.kernel.toLowerCase().includes(query),
                ),
              },
        )
        .filter((group) => group.entries.length > 0);
    }
    this.visibleCount = this.visibleGroups.reduce(
      (count, group) => count + group.entries.length,
      0,
    );
  }

  private setRailPeeking(peeking: boolean, delayMs: number): void {
    clearTimeout(this.railTimer);
    this.railTimer = undefined;
    if (this.isRailPinned || this.isRailPeeking === peeking) {
      return;
    }
    const apply = () => {
      this.isRailPeeking = peeking;
      if (!peeking) {
        // The search only narrows down the rail while it is open.
        this.clearFilter();
        this.hideCardFrom('rail');
        this.releasePointerFocus('.kernel-rail');
      }
      this.changeDetectorRef.markForCheck();
    };
    if (delayMs > 0) {
      this.railTimer = setTimeout(apply, delayMs);
    } else {
      apply();
    }
  }

  private dismissRail(): void {
    this.isFocusInRail = false;
    this.searchInput?.nativeElement.blur();
    this.clearFilter();
    clearTimeout(this.railTimer);
    this.isRailPeeking = false;
    this.hideCardFrom('rail');
  }

  /** Whether `event` is a pointer hover, or focus from the keyboard. */
  private isHoverOrKeyboardFocus(event: Event): boolean {
    const target = event.target;
    return (
      event.type !== 'focus' ||
      (target instanceof HTMLElement && target.matches(':focus-visible'))
    );
  }

  /** Shows a details card, right away if another card is showing. */
  private scheduleCard(build: () => KernelCard | undefined): void {
    clearTimeout(this.cardTimer);
    this.cardTimer = undefined;
    const show = () => {
      this.cardTimer = undefined;
      this.hoverCard = build();
      this.changeDetectorRef.markForCheck();
    };
    if (this.hoverCard) {
      show();
    } else {
      this.cardTimer = setTimeout(show, CARD_SHOW_DELAY_MS);
    }
  }

  private hideCardFrom(source: KernelCard['source']): void {
    if (this.hoverCard?.source === source) {
      this.hideCard();
    }
  }

  private cardBesideRow(entry: KernelEntry, row: HTMLElement): KernelCard {
    const host = this.elementRef.nativeElement.getBoundingClientRect();
    const rowRect = row.getBoundingClientRect();
    const railRight =
      row.closest('.kernel-rail')?.getBoundingClientRect().right ??
      rowRect.right;
    const left = railRight - host.left + CARD_GAP;
    const top = rowRect.top - host.top;
    const bottom = host.bottom - rowRect.bottom;
    const tab = this.findTab(entry);
    let hintParts: readonly HintPart[] = [
      {text: BACKGROUND_CLICK_KEY, isKey: true},
      {text: '+click to open in a background tab', isKey: false},
    ];
    if (tab === this.activeTab) {
      hintParts = [];
    } else if (tab) {
      hintParts = [
        {
          text: `Already open in tab ${this.tabs.indexOf(tab) + 1}`,
          isKey: false,
        },
      ];
    }
    // Grow the card away from the closer edge, so that it fits.
    return top < bottom
      ? {entry, source: 'rail', ...buildCardHint(hintParts), left, top}
      : {entry, source: 'rail', ...buildCardHint(hintParts), left, bottom};
  }

  private cardBelowTab(tab: KernelTab, tabElement: Element): KernelCard {
    const host = this.elementRef.nativeElement.getBoundingClientRect();
    const rect = tabElement.getBoundingClientRect();
    const maxLeft = Math.max(CARD_GAP, host.width - CARD_WIDTH - CARD_GAP);
    const hintParts: readonly HintPart[] =
      this.tabs.length > 1
        ? [
            {text: '[', isKey: true},
            {text: ' and ', isKey: false},
            {text: ']', isKey: true},
            {text: ' switch tabs', isKey: false},
          ]
        : [];
    return {
      entry: tab.entry,
      source: 'tab',
      ...buildCardHint(hintParts),
      left: Math.min(Math.max(rect.left - host.left, CARD_GAP), maxLeft),
      top: rect.bottom - host.top + CARD_GAP,
    };
  }

  /**
   * Blurs a kernel row or tab that was focused by a click, so that its focus
   * ring does not appear once keys are pressed. Keyboard focus stays.
   */
  private releasePointerFocus(scope: string): void {
    const focused = document.activeElement;
    if (
      this.isKeyboardFocus ||
      !(focused instanceof HTMLElement) ||
      isEditableTarget(focused) ||
      !focused.closest(scope) ||
      !this.elementRef.nativeElement.contains(focused)
    ) {
      return;
    }
    focused.blur();
  }

  private onWindowKeydown(event: KeyboardEvent): void {
    if (!MODIFIER_KEYS.has(event.key)) {
      this.hideCard();
      if (event.key !== 'Tab') {
        this.releasePointerFocus('.kernel-rail, .kernel-bar');
      }
      this.changeDetectorRef.markForCheck();
    }
    if (event.defaultPrevented || isEditableTarget(event.target)) {
      return;
    }
    if (event.ctrlKey || event.metaKey || event.altKey) {
      return;
    }
    if (event.key === '[' || event.key === ']') {
      event.preventDefault();
      // Holding the key down would queue up a trace load per repeat.
      if (!event.repeat) {
        this.stepTab(event.key === ']' ? 1 : -1);
        this.changeDetectorRef.markForCheck();
      }
    } else if (event.key === 'k' || event.key === 'K') {
      event.preventDefault();
      this.focusSearch();
      this.changeDetectorRef.markForCheck();
    }
  }

  /** Switches to the next (`delta` = 1) or previous (`delta` = -1) open tab. */
  private stepTab(delta: number): void {
    const count = this.tabs.length;
    if (!this.activeTab || count < 2) {
      return;
    }
    const index = this.tabs.indexOf(this.activeTab);
    this.activateTab(this.tabs[(((index + delta) % count) + count) % count]);
  }

  /** Scrolls the matching kernel row and tab into view once rendered. */
  private scrollIntoView(rowSelector?: string, tabSelector?: string): void {
    setTimeout(() => {
      const host = this.elementRef.nativeElement;
      if (rowSelector) {
        host.querySelector(rowSelector)?.scrollIntoView({block: 'nearest'});
      }
      const tabList = host.querySelector<HTMLElement>('.tab-list');
      const tab = tabSelector
        ? tabList?.querySelector<HTMLElement>(tabSelector)
        : undefined;
      if (tabList && tab) {
        revealHorizontally(tabList, tab);
      }
    });
  }

  /**
   * Dispatches resize events immediately and after the rail width transition
   * to ensure the canvas recalculates its layout.
   */
  private notifyResize(): void {
    window.dispatchEvent(new Event('resize'));
    setTimeout(() => {
      window.dispatchEvent(new Event('resize'));
    }, RAIL_TRANSITION_MS + 10);
  }
}
