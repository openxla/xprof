import {Component, inject, Input, OnChanges, OnDestroy, SimpleChanges} from '@angular/core';
import {FileExtensionType} from 'org_xprof/frontend/app/common/constants/enums';
import {DATA_SERVICE_INTERFACE_TOKEN, DataServiceV2Interface} from 'org_xprof/frontend/app/services/data_service_v2/data_service_v2_interface';
import {Address} from 'org_xprof/frontend/app/services/source_code_service/source_code_service_interface';
import {ReplaySubject} from 'rxjs';
import {takeUntil} from 'rxjs/operators';

enum CompilerPass {
  HLO_OPTIMIZED = 'HLO(optimized)',
  MOSAIC_LLO = 'Mosaic llo',
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

  @Input() sourceFileAndLineNumber: string|undefined = undefined;
  @Input() stackTrace: string|undefined = undefined;
  /**
   * The number of lines to show around the stack frame.
   */
  @Input() sourceContextWindow = 40;
  @Input() sessionId = '';
  @Input() programId = '';
  @Input() opName = '';

  sourceCodeSnippetAddresses: readonly Address[] = [];
  hloTextByProgramId = new Map<string, string>();
  // TODO(yinzz): add CompilerPass.MOSAIC_LLO
  compilerPasses = [CompilerPass.HLO_OPTIMIZED];
  selectedCompilerPass = CompilerPass.HLO_OPTIMIZED;
  sourceFileNames: string[] = [];
  selectedSourceFileName = '';

  ngOnChanges(changes: SimpleChanges) {
    if (changes['sessionId'] &&
        changes['sessionId'].currentValue !== this.sessionId) {
      this.hloTextByProgramId.clear();
    }
    if (changes['programId']) {
      this.maybeUpdateHloTextCache();
    }
    if (changes['sourceFileAndLineNumber'] || changes['stackTrace']) {
      this.parseSourceFileNames();
    }
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
      default:
        return '';
    }
  }

  get irTextLines() {
    return this.irText.split('\n');
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
                   (line) => line.includes(`${this.opName} =`)) ||
            0;
      default:
        return 0;
    }
  }

  get irTextLinesForDisplay(): string[] {
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
    if (!this.programId || this.programId === '0' || this.sessionId === '' ||
        this.selectedCompilerPass !== CompilerPass.HLO_OPTIMIZED) {
      return;
    }
    const hloText = this.hloTextByProgramId.get(this.programId);
    if (hloText) {
      return;
    }
    this.dataService
        .downloadHloProto(
            this.sessionId,
            '',
            FileExtensionType.LONG_TEXT,
            false,
            this.programId,
            )
        .pipe(takeUntil(this.destroyed))
        .subscribe((data) => {
          if (data) {
            this.hloTextByProgramId.set(this.programId, data as string);
          }
        });
  }

  ngOnDestroy(): void {
    this.destroyed.next();
    this.destroyed.complete();
  }
}
