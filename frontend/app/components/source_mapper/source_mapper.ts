import {
  ChangeDetectionStrategy,
  Component,
  computed,
  inject,
  Input,
  OnChanges,
  OnDestroy,
  signal,
  SimpleChanges,
} from '@angular/core';
import {Store} from '@ngrx/store';
import {
  GRAPH_TYPE_DEFAULT,
  GRAPH_TYPE_ORIGINAL_HLO,
} from 'org_xprof/frontend/app/common/constants/constants';
import {FileExtensionType} from 'org_xprof/frontend/app/common/constants/enums';
import {ProfilerConfig} from 'org_xprof/frontend/app/common/interfaces/capture_profile';
import {
  DATA_SERVICE_INTERFACE_TOKEN,
  DataServiceV2Interface,
} from 'org_xprof/frontend/app/services/data_service_v2/data_service_v2_interface';
import {
  Address,
  Content,
} from 'org_xprof/frontend/app/services/source_code_service/source_code_service_interface';
import {
  getProfilerConfig,
  getTagsState,
} from 'org_xprof/frontend/app/store/selectors';
import {ReplaySubject} from 'rxjs';
import {takeUntil} from 'rxjs/operators';

import {CompilerPass, IrText} from './ir_text';

export {CompilerPass, IrText};

const CUSTOM_CALL_CATEGORY = 'custom-call';

/**
 * A source mapper component.
 *
 * We use this component to map TPU operations to python source code.
 * TPU operations can be HLO or LLO.
 */
@Component({
  changeDetection: ChangeDetectionStrategy.Default,
  standalone: false,
  selector: 'source-mapper',
  templateUrl: './source_mapper.ng.html',
  styleUrls: ['./source_mapper.scss'],
})
export class SourceMapper implements OnDestroy, OnChanges {
  private readonly destroyed = new ReplaySubject<void>(1);
  private readonly dataService: DataServiceV2Interface = inject(
    DATA_SERVICE_INTERFACE_TOKEN,
  );
  private readonly sourceContextWindowSignal = signal(40);
  private readonly programIdSignal = signal('');
  private readonly opNameSignal = signal('');
  private readonly selectedCompilerPassSignal = signal(
    CompilerPass.HLO_OPTIMIZED,
  );
  private readonly irCacheVersion = signal(0);
  private readonly irTextFrameComputed = computed<Content | undefined>(() => {
    this.irCacheVersion();
    const opName = this.opName;
    const contextWindow = this.sourceContextWindow;
    return this.currentIrText?.getDisplayFrame(opName, contextWindow);
  });

  /**
   * The source file and line number of the HLO op.
   * This is used to find the source code snippet.
   * Processed from xla source info.
   */
  @Input() sourceFileAndLineNumber: string | undefined = undefined;
  /**
   * The stack trace of the HLO op.
   * Processed from xla stack frame info.
   */
  @Input() stackTrace: string | undefined = undefined;
  // The number of lines to show around the stack frame.
  @Input()
  get sourceContextWindow(): number {
    return this.sourceContextWindowSignal();
  }
  set sourceContextWindow(value: number) {
    this.sourceContextWindowSignal.set(value);
  }
  @Input() sessionId = '';
  // The program id of the HLO op.
  @Input()
  get programId(): string {
    return this.programIdSignal();
  }
  set programId(value: string) {
    this.programIdSignal.set(value);
  }
  // The name of the HLO op.
  @Input()
  get opName(): string {
    return this.opNameSignal();
  }
  set opName(value: string) {
    this.opNameSignal.set(value);
  }
  // The category of the HLO op.
  @Input() opCategory = '';

  sourceCodeSnippetAddresses: readonly Address[] = [];
  hloTextCache = new Map<string, IrText>();
  mosaicTextByKernelName = new Map<string, IrText>();
  mosaicSourceFileAndLineNumberByKernelName = new Map<string, string>();
  get selectedCompilerPass(): CompilerPass {
    return this.selectedCompilerPassSignal();
  }
  set selectedCompilerPass(value: CompilerPass) {
    this.selectedCompilerPassSignal.set(value);
  }
  sourceFileNames: string[] = [];
  selectedSourceFileName = '';
  tags: string[] = [];
  // The prefix of the source file path if specified by users.
  srcPathPrefix = '';

  constructor(private readonly store: Store<{}>) {
    this.store
      .select(getTagsState)
      .pipe(takeUntil(this.destroyed))
      .subscribe((tags: string[]) => {
        if (!tags || tags.length === 0) return;
        this.tags = tags;
      });
    this.store
      .select(getProfilerConfig)
      .pipe(takeUntil(this.destroyed))
      .subscribe((config: ProfilerConfig) => {
        if (!config) return;
        this.srcPathPrefix = config.srcPathPrefix;
      });
  }

  ngOnChanges(changes: SimpleChanges) {
    if (
      changes['sessionId'] &&
      changes['sessionId'].currentValue !== changes['sessionId'].previousValue
    ) {
      this.hloTextCache.clear();
      this.mosaicTextByKernelName.clear();
      this.mosaicSourceFileAndLineNumberByKernelName.clear();
      this.irCacheVersion.update((v) => v + 1);
    }
    if (changes['programId']) {
      this.update();
    }
    if (
      changes['opName'] &&
      changes['opName'].currentValue !== changes['opName'].previousValue
    ) {
      if (!this.compilerPasses.includes(this.selectedCompilerPass)) {
        this.selectedCompilerPass = this.compilerPasses[0];
      }
      this.update();
    }
    if (changes['sourceFileAndLineNumber'] || changes['stackTrace']) {
      this.parseSourceFileNames();
    }
  }

  update() {
    this.maybeUpdateHloTextCache();

    this.maybeUpdateMosaicTextCache();
    this.maybeUpdateMosaicSourceFileAndLineNumberCache();
  }

  get adaptedStackTrace(): string {
    switch (this.selectedCompilerPass) {
      case CompilerPass.HLO_OPTIMIZED:
      case CompilerPass.HLO_UNOPTIMIZED:
        return this.stackTrace || '';
      case CompilerPass.MOSAIC_ORIGINAL:
        return this.getPallasKernelStackTrace();
      default:
        return '';
    }
  }

  get adaptedSourceFileAndLineNumber(): string {
    switch (this.selectedCompilerPass) {
      case CompilerPass.HLO_OPTIMIZED:
      case CompilerPass.HLO_UNOPTIMIZED:
        return this.sourceFileAndLineNumber || '';
      case CompilerPass.MOSAIC_ORIGINAL:
        return this.pallasKernelSourceFileAndLineNumber;
      default:
        return '';
    }
  }

  get compilerPasses(): CompilerPass[] {
    const basePasses = [CompilerPass.HLO_OPTIMIZED];
    if (this.hasUnoptimizedHloTag) {
      basePasses.push(CompilerPass.HLO_UNOPTIMIZED);
    }
    // llo_debug tag is identifier for llo debug proto captured.
    // the proto contains llo bundles for kernels, and source info for mosaic
    // passes.
    if (this.hasLloDebugTag) {
      // CustomCall op category is identifier for a pallas kernel.
      if (this.isCustomCall) {
        basePasses.push(CompilerPass.MOSAIC_ORIGINAL);
      }
    }
    return basePasses;
  }

  get hasLloDebugTag() {
    return this.tags.includes('llo_debug');
  }

  get hasUnoptimizedHloTag() {
    return this.tags.includes('has_original_hlo_proto');
  }

  get irFileName(): string {
    switch (this.selectedCompilerPass) {
      case CompilerPass.HLO_OPTIMIZED:
      case CompilerPass.HLO_UNOPTIMIZED:
        return 'source.hlo';
      case CompilerPass.MOSAIC_ORIGINAL:
        return 'source.mlir';
      default:
        return 'source.txt';
    }
  }

  get irTextLink(): string {
    return '';
  }

  get irTextLinkTooltip(): string {
    return 'The IR text is truncated, click to view entire text in a new tab.';
  }

  get currentIrText(): IrText | undefined {
    switch (this.selectedCompilerPass) {
      case CompilerPass.HLO_OPTIMIZED:
      case CompilerPass.HLO_UNOPTIMIZED:
        return this.hloTextCache.get(
          this.getHloCacheKey(this.programId, this.selectedCompilerPass),
        );
      case CompilerPass.MOSAIC_ORIGINAL:
        return this.mosaicTextByKernelName.get(this.opName);
      default:
        return undefined;
    }
  }

  getHloCacheKey(
    programId: string,
    compilerPass: CompilerPass = CompilerPass.HLO_OPTIMIZED,
  ): string {
    return `${programId}_${compilerPass}`;
  }

  get irText(): string {
    return this.currentIrText?.text || '';
  }

  get irTextLines(): string[] {
    return this.currentIrText?.lines || [''];
  }

  get isCustomCall() {
    return this.opCategory.includes(CUSTOM_CALL_CATEGORY);
  }

  get pallasKernelSourceFileAndLineNumber() {
    return (
      this.mosaicSourceFileAndLineNumberByKernelName.get(this.opName) || ''
    );
  }

  // Not implemented yet.
  getPallasKernelStackTrace() {
    return '';
  }

  isFocusLine(line: string): boolean {
    switch (this.selectedCompilerPass) {
      case CompilerPass.HLO_OPTIMIZED:
        return line.includes(`${this.opName} =`);
      default:
        return false;
    }
  }

  get irTextFocusLineIndex(): number {
    return this.currentIrText?.getFocusLineIndex(this.opName) ?? 0;
  }

  get irTextLinesForDisplay(): string[] {
    return (
      this.currentIrText?.getLinesForDisplay(
        this.opName,
        this.sourceContextWindow,
      ) || ['']
    );
  }

  get irTextFrame(): Content | undefined {
    return this.irTextFrameComputed();
  }

  get focusLineIndexForDisplay(): number | undefined {
    return this.currentIrText?.getFocusLineNumberForDisplay(
      this.opName,
      this.sourceContextWindow,
    );
  }

  trackByIndex(index: number, item: string): number {
    return index;
  }

  parseSourceFileNames() {
    const sourceFileName = this.sourceFileAndLineNumber?.split(':')[0] || '';
    this.sourceFileNames = [sourceFileName];
    if (this.sourceFileNames.length > 0) {
      this.selectedSourceFileName = this.sourceFileNames[0];
    }
  }

  get loaded() {
    return this.irText !== '';
  }

  maybeUpdateHloTextCache() {
    if (!this.programId || this.programId === '0' || this.sessionId === '') {
      return;
    }
    const programId = this.programId;
    const compilerPass = this.selectedCompilerPass;
    let hloGraphType = GRAPH_TYPE_DEFAULT;
    switch (compilerPass) {
      case CompilerPass.HLO_OPTIMIZED:
        hloGraphType = GRAPH_TYPE_DEFAULT;
        break;
      case CompilerPass.HLO_UNOPTIMIZED:
        hloGraphType = GRAPH_TYPE_ORIGINAL_HLO;
        break;
      default:
        return;
    }
    const cacheKey = this.getHloCacheKey(programId, compilerPass);
    // Cache hit, early return.
    if (this.hloTextCache.has(cacheKey)) {
      return;
    }
    this.dataService
      .downloadHloProto(
        this.sessionId,
        hloGraphType,
        '',
        FileExtensionType.LONG_TEXT,
        false,
        programId,
      )
      .pipe(takeUntil(this.destroyed))
      .subscribe((data) => {
        if (typeof data === 'string' && data) {
          this.hloTextCache.set(cacheKey, new IrText(data, compilerPass));
          this.irCacheVersion.update((v) => v + 1);
        }
      });
  }

  maybeUpdateMosaicTextCache() {
    if (
      !this.opName ||
      this.sessionId === '' ||
      this.selectedCompilerPass !== CompilerPass.MOSAIC_ORIGINAL
    ) {
      return;
    }
    const text = this.mosaicTextByKernelName.get(this.opName);
    if (text) {
      return;
    }
    this.dataService
      .getCustomCallText(this.sessionId, '', this.opName, this.programId)
      .pipe(takeUntil(this.destroyed))
      .subscribe((data: string) => {
        if (data) {
          this.mosaicTextByKernelName.set(
            this.opName,
            new IrText(data, CompilerPass.MOSAIC_ORIGINAL),
          );
          this.irCacheVersion.update((v) => v + 1);
        }
      });
  }

  maybeUpdateMosaicSourceFileAndLineNumberCache() {
    if (
      !this.opName ||
      this.sessionId === '' ||
      this.selectedCompilerPass !== CompilerPass.MOSAIC_ORIGINAL
    ) {
      return;
    }
    const sourceFileAndLineNumber =
      this.mosaicSourceFileAndLineNumberByKernelName.get(this.opName);
    if (sourceFileAndLineNumber) {
      return;
    }
    this.dataService
      .getLloSourceInfo(this.sessionId, this.opName)
      .pipe(takeUntil(this.destroyed))
      .subscribe((sourceInfo) => {
        if (sourceInfo) {
          this.mosaicSourceFileAndLineNumberByKernelName.set(
            this.opName,
            sourceInfo,
          );
        }
      });
  }

  onCompilerPassChange(newCompilerPass: CompilerPass) {
    this.selectedCompilerPass = newCompilerPass;
    switch (this.selectedCompilerPass) {
      case CompilerPass.HLO_OPTIMIZED:
      case CompilerPass.HLO_UNOPTIMIZED:
        this.maybeUpdateHloTextCache();
        break;
      case CompilerPass.MOSAIC_ORIGINAL:
        this.maybeUpdateMosaicTextCache();
        this.maybeUpdateMosaicSourceFileAndLineNumberCache();
        break;
      default:
        break;
    }
  }

  ngOnDestroy(): void {
    this.destroyed.next();
    this.destroyed.complete();
  }
}
