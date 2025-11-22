import {Component, inject, Input, OnChanges, OnDestroy, SimpleChanges} from '@angular/core';
import {Store} from '@ngrx/store';
import {GRAPH_TYPE_DEFAULT, GRAPH_TYPE_ORIGINAL_HLO} from 'org_xprof/frontend/app/common/constants/constants';
import {FileExtensionType} from 'org_xprof/frontend/app/common/constants/enums';
import {DATA_SERVICE_INTERFACE_TOKEN, DataServiceV2Interface} from 'org_xprof/frontend/app/services/data_service_v2/data_service_v2_interface';
import {Address} from 'org_xprof/frontend/app/services/source_code_service/source_code_service_interface';
import {getTagsState} from 'org_xprof/frontend/app/store/selectors';
import {ReplaySubject} from 'rxjs';
import {takeUntil} from 'rxjs/operators';

const CUSTOM_CALL_CATEGORY = 'custom-call';

enum CompilerPass {
  HLO_OPTIMIZED = 'HLO(optimized)',
  HLO_UNOPTIMIZED = 'HLO(original)',
  MOSAIC_ORIGINAL = 'Mosaic(original)',
}

/**
 * A source mapper component.
 *
 * We use this component to map TPU operations to python source code.
 * TPU operations can be HLO or LLO.
 */
@Component({
  standalone: false,
  selector: 'source-mapper',
  templateUrl: './source_mapper.ng.html',
  styleUrls: ['./source_mapper.css'],
})
export class SourceMapper implements OnDestroy, OnChanges {
  private readonly destroyed = new ReplaySubject<void>(1);
  private readonly dataService: DataServiceV2Interface =
      inject(DATA_SERVICE_INTERFACE_TOKEN);

  /**
   * The source file and line number of the HLO op.
   * This is used to find the source code snippet.
   * Processed from xla source info.
   */
  @Input() sourceFileAndLineNumber: string|undefined = undefined;
  /**
   * The stack trace of the HLO op.
   * Processed from xla stack frame info.
   */
  @Input() stackTrace: string|undefined = undefined;
  // The number of lines to show around the stack frame.
  @Input() sourceContextWindow = 40;
  @Input() sessionId = '';
  // The program id of the HLO op.
  @Input() programId = '';
  // The name of the HLO op.
  @Input() opName = '';
  // The category of the HLO op.
  @Input() opCategory = '';

  sourceCodeSnippetAddresses: readonly Address[] = [];
  hloTextByProgramId = new Map<string, string>();
  hloUnoptimizedTextByProgramId = new Map<string, string>();
  mosaicTextByKernelName = new Map<string, string>();
  mosaicSourceFileAndLineNumberByKernelName = new Map<string, string>();
  selectedCompilerPass = CompilerPass.HLO_OPTIMIZED;
  sourceFileNames: string[] = [];
  selectedSourceFileName = '';
  tags: string[] = [];

  constructor(private readonly store: Store<{}>) {
    this.store.select(getTagsState)
        .pipe(takeUntil(this.destroyed))
        .subscribe((tags: string[]) => {
          if (!tags || tags.length === 0) return;
          this.tags = tags;
        });
  }

  ngOnChanges(changes: SimpleChanges) {
    if (changes['sessionId'] &&
        changes['sessionId'].currentValue !==
            changes['sessionId'].previousValue) {
      this.hloTextByProgramId.clear();
      this.hloUnoptimizedTextByProgramId.clear();
      this.mosaicTextByKernelName.clear();
      this.mosaicSourceFileAndLineNumberByKernelName.clear();
    }
    if (changes['programId']) {
      this.update();
    }
    if (changes['opName'] &&
        changes['opName'].currentValue !== changes['opName'].previousValue) {
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

  get irTextLink(): string {
    return '';
  }

  get irTextLinkTooltip(): string {
    return 'The IR text is truncated, click to view entire text in a new tab.';
  }

  get irText() {
    switch (this.selectedCompilerPass) {
      case CompilerPass.HLO_OPTIMIZED:
        return this.hloTextByProgramId.get(this.programId) || '';
      case CompilerPass.HLO_UNOPTIMIZED:
        return this.hloUnoptimizedTextByProgramId.get(this.programId) || '';
      case CompilerPass.MOSAIC_ORIGINAL:
        return this.mosaicTextByKernelName.get(this.opName) || '';
      default:
        return '';
    }
  }

  get irTextLines() {
    return this.irText.split('\n');
  }

  get isCustomCall() {
    return this.opCategory.includes(CUSTOM_CALL_CATEGORY);
  }

  get pallasKernelSourceFileAndLineNumber() {
    return this.mosaicSourceFileAndLineNumberByKernelName.get(this.opName) ||
        '';
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
    switch (this.selectedCompilerPass) {
      case CompilerPass.HLO_OPTIMIZED:
        return this.irTextLines.findIndex(
                   (line: string) => line.includes(`${this.opName} =`)) ||
            0;
      case CompilerPass.HLO_UNOPTIMIZED:
        // Currently not able to map from HLO op to its original text line.
        return 0;
      case CompilerPass.MOSAIC_ORIGINAL:
        // Assumptions: the MLIR text contains the key word kernel in the kernel
        // definition line.
        return this.irTextLines.findIndex(
                   (line: string) => line.includes('kernel')) ||
            0;
      default:
        return 0;
    }
  }

  get irTextLinesForDisplay(): string[] {
    // Return the entire HLO text for HLO unoptimized pass, as the mapping from
    // the selected HLO op to its original text line is not implemented yet.
    if (this.selectedCompilerPass === CompilerPass.HLO_UNOPTIMIZED) {
      return this.irTextLines;
    }
    const minLineIndex =
        Math.max(0, this.irTextFocusLineIndex - this.sourceContextWindow / 2);
    const maxLineIndex = Math.min(
        this.irTextLines.length - 1,
        this.irTextFocusLineIndex + this.sourceContextWindow / 2,
    );
    return this.irTextLines.slice(minLineIndex, maxLineIndex);
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
    let hloTextCache = null;
    let hloGraphType = GRAPH_TYPE_DEFAULT;
    switch (this.selectedCompilerPass) {
      case CompilerPass.HLO_OPTIMIZED:
        hloTextCache = this.hloTextByProgramId;
        hloGraphType = GRAPH_TYPE_DEFAULT;
        break;
      case CompilerPass.HLO_UNOPTIMIZED:
        hloTextCache = this.hloUnoptimizedTextByProgramId;
        hloGraphType = GRAPH_TYPE_ORIGINAL_HLO;
        break;
      default:
        return;
    }
    // Selected compiler pass is not for hlo.
    if (!hloTextCache) {
      return;
    }
    // Cache hit, early return.
    const hloText = hloTextCache.get(this.programId);
    if (hloText) {
      return;
    }
    this.dataService
        .downloadHloProto(
            this.sessionId,
            hloGraphType,
            '',
            FileExtensionType.LONG_TEXT,
            false,
            this.programId,
            )
        .pipe(takeUntil(this.destroyed))
        .subscribe((data) => {
          if (data) {
            hloTextCache.set(this.programId, data as string);
          }
        });
  }

  maybeUpdateMosaicTextCache() {
    if (!this.opName || this.sessionId === '' ||
        this.selectedCompilerPass !== CompilerPass.MOSAIC_ORIGINAL) {
      return;
    }
    const text = this.mosaicTextByKernelName.get(this.opName);
    if (text) {
      return;
    }
    this.dataService
        .getCustomCallText(
            this.sessionId,
            '',
            this.opName,
            this.programId,
            )
        .pipe(takeUntil(this.destroyed))
        .subscribe((data: string) => {
          if (data) {
            this.mosaicTextByKernelName.set(this.opName, data);
          }
        });
  }

  maybeUpdateMosaicSourceFileAndLineNumberCache() {
    if (!this.opName || this.sessionId === '' ||
        this.selectedCompilerPass !== CompilerPass.MOSAIC_ORIGINAL) {
      return;
    }
    const sourceFileAndLineNumber =
        this.mosaicSourceFileAndLineNumberByKernelName.get(this.opName);
    if (sourceFileAndLineNumber) {
      return;
    }
    this.dataService.getLloSourceInfo(this.sessionId, this.opName)
        .pipe(takeUntil(this.destroyed))
        .subscribe((sourceInfo) => {
          if (sourceInfo) {
            this.mosaicSourceFileAndLineNumberByKernelName.set(
                this.opName, sourceInfo);
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
