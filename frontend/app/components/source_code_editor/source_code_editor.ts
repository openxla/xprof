import {
  AfterViewChecked,
  ChangeDetectionStrategy,
  Component,
  ElementRef,
  inject,
  InjectionToken,
  Input,
  OnChanges,
  OnDestroy,
  SimpleChanges,
  ViewChild,
} from '@angular/core';
import {loadMonaco} from 'google3/third_party/javascript/monaco_editor/v0_52_0/gstatic_loader';
import {Metric} from 'org_xprof/frontend/app/common/interfaces/source_stats';
import * as utils from 'org_xprof/frontend/app/common/utils/utils';
import {Content} from 'org_xprof/frontend/app/services/source_code_service/source_code_service_interface';

/** Injection token for the Monaco editor loader. */
export const MONACO_LOADER_TOKEN = new InjectionToken<
  () => Promise<typeof monaco>
>('MONACO_LOADER_TOKEN', {
  factory: () => async () => {
    await loadMonaco();
    return (window as unknown as {monaco: typeof monaco}).monaco;
  },
  providedIn: 'root',
});

/** A component to display and interact with a Monaco code editor for source code. */
@Component({
  changeDetection: ChangeDetectionStrategy.OnPush,
  standalone: false,
  selector: 'source-code-editor',
  templateUrl: './source_code_editor.ng.html',
  styleUrls: ['./source_code_editor.scss'],
})
export class SourceCodeEditor
  implements OnChanges, OnDestroy, AfterViewChecked
{
  @Input() frame: Content | undefined = undefined;
  @Input() isLoading = false;
  @Input() firstLineNumber = 1;
  @Input() targetLineNumber: number | undefined = undefined;
  @Input() targetStartColumn: number | undefined = undefined;
  @Input() targetEndColumn: number | undefined = undefined;
  @Input() fileName: string | undefined = undefined;

  @ViewChild('editorContainer', {static: false})
  editorContainer?: ElementRef<HTMLElement>;

  private readonly monacoLoader = inject(MONACO_LOADER_TOKEN);

  static readonly THEME_STORAGE_KEY = 'xprof_monaco_editor_theme';
  currentTheme = 'vs';

  private editor?: monaco.editor.IStandaloneCodeEditor;
  private contentWidgets: monaco.editor.IContentWidget[] = [];
  private decorationsCollection?: monaco.editor.IEditorDecorationsCollection;
  private isInitializingEditor = false;
  private updatingHeight = false;
  private lineNumberToMetricMap: Map<number, Metric> | undefined = undefined;

  constructor() {
    this.currentTheme = this.getSavedTheme();
  }

  getSavedTheme(): string {
    try {
      return (
        window.localStorage?.getItem(SourceCodeEditor.THEME_STORAGE_KEY) || 'vs'
      );
    } catch {
      return 'vs';
    }
  }

  onThemeSelection(theme: string): void {
    if (theme) {
      this.setTheme(theme);
    }
  }

  setTheme(theme: string): void {
    this.currentTheme = theme;
    try {
      window.localStorage?.setItem(SourceCodeEditor.THEME_STORAGE_KEY, theme);
    } catch (e) {
      console.error('Failed to save Monaco theme to localStorage:', e);
      // Ignore localStorage errors.
    }
    this.applyTheme(theme);
  }

  private async applyTheme(theme: string): Promise<void> {
    const monacoObj = await this.monacoLoader();
    monacoObj?.editor?.setTheme(theme);
  }

  ngAfterViewChecked(): void {
    if (this.frame && this.editorContainer && !this.editor) {
      this.initEditor();
    }
  }

  ngOnDestroy(): void {
    this.disposeEditor();
  }

  ngOnChanges(changes: SimpleChanges) {
    let shouldReinit = false;
    let shouldRerenderDecorations = false;

    // Check if the underlying code payload has changed
    if (changes['frame']) {
      if (this.frame) {
        // Fast O(1) lookup map so we don't have to iterate the metrics array for every line
        this.lineNumberToMetricMap = new Map(
          this.frame.metrics.map((lineMetric) => [
            lineMetric.lineNumber,
            lineMetric.metric,
          ]),
        );
      } else {
        this.lineNumberToMetricMap = undefined;
      }
      // If the code contents change, we have to completely destroy and re-create the Monaco editor
      shouldReinit = true;
    }
    // Changing the file name affects syntax highlighting, requiring a re-init
    if (changes['fileName']) {
      shouldReinit = true;
    }

    // These inputs only affect visual overlays (highlights and metric widgets), so we can skip tearing down the editor
    if (
      changes['targetLineNumber'] ||
      changes['targetStartColumn'] ||
      changes['targetEndColumn'] ||
      changes['firstLineNumber']
    ) {
      shouldRerenderDecorations = true;
    }

    // Execute state updates
    if (shouldReinit) {
      this.reload();
    } else if (shouldRerenderDecorations && this.editor && this.frame) {
      this.monacoLoader().then((monacoObj) => {
        this.renderDecorationsAndWidgets(monacoObj);
      });
    }
  }

  lineMetric(lineNumber: number): Metric | undefined {
    return this.lineNumberToMetricMap?.get(lineNumber);
  }

  private reload() {
    this.disposeEditor();
    if (this.frame) {
      this.initEditor();
    }
  }

  private async initEditor(): Promise<void> {
    if (
      !this.frame ||
      !this.editorContainer ||
      this.editor ||
      this.isInitializingEditor
    ) {
      return;
    }
    this.isInitializingEditor = true;
    try {
      const monacoObj = await this.monacoLoader();
      if (!this.frame || !this.editorContainer || this.editor || !monacoObj) {
        return;
      }

      const codeText = this.frame.lines
        .map((line) => this.stripHtmlTags(line))
        .join('\n');

      // Define additional custom themes before creating the editor
      monacoObj.editor.defineTheme('solarized-light', {
        base: 'vs',
        inherit: true,
        rules: [],
        colors: {
          'editor.background': '#FDF6E3',
        },
      });

      monacoObj.editor.defineTheme('monokai', {
        base: 'vs-dark',
        inherit: true,
        rules: [],
        colors: {
          'editor.background': '#272822',
        },
      });

      this.editor = monacoObj.editor.create(
        this.editorContainer.nativeElement,
        {
          value: codeText,
          theme: this.currentTheme,
          readOnly: true,
          domReadOnly: true,
          language: this.getLanguage(this.fileName),
          lineNumbers: (lineNumber: number) => {
            return String(this.firstLineNumber + lineNumber - 1);
          },
          automaticLayout: true,
          scrollBeyondLastLine: false,
          scrollBeyondLastColumn: 0,
          minimap: {enabled: false},
          wordWrap: 'off',
          fontFamily: '"Roboto Mono", monospace',
          fontSize: 14,
          lineHeight: 24,
          renderLineHighlight: 'none',
          overviewRulerLanes: 0,
          hideCursorInOverviewRuler: true,
          scrollbar: {
            vertical: 'auto',
            horizontal: 'auto',
            alwaysConsumeMouseWheel: false,
          },
        },
      );
      monacoObj.editor.setTheme(this.currentTheme);

      this.editor?.onDidContentSizeChange(() => {
        this.updateEditorHeight();
      });

      this.renderDecorationsAndWidgets(monacoObj);
      this.updateEditorHeight();
    } finally {
      this.isInitializingEditor = false;
    }
  }

  private stripHtmlTags(html: string): string {
    // Legacy xprof source payloads occasionally wrap lines in basic HTML or HTML entities.
    // Monaco natively handles formatting itself, so passing HTML tags will literally print
    // the tags (e.g., <b>) as syntax words. We must sanitize the string into pure plain text.
    return html
      .replace(/<[^>]*>/g, '')
      .replace(/&lt;/g, '<')
      .replace(/&gt;/g, '>')
      .replace(/&amp;/g, '&')
      .replace(/&quot;/g, '"')
      .replace(/&#39;/g, "'")
      .replace(/&nbsp;/g, ' ');
  }

  private updateEditorHeight(): void {
    if (!this.editor || !this.editorContainer || this.updatingHeight) {
      return;
    }
    this.updatingHeight = true;
    try {
      // Synchronize the DOM container height to the actual internal text height computed
      // by the Monaco editor (capping at a minimum of 80px).
      // This allows the editor to flow naturally in the stack trace page without static scrollbars.
      const contentHeight = Math.max(80, this.editor.getContentHeight());
      this.editorContainer.nativeElement.style.height = `${contentHeight}px`;
      this.editor.layout();
    } finally {
      this.updatingHeight = false;
    }
  }

  private getLanguage(fileName?: string): string {
    // Basic file extension to Monaco language ID mapping
    // This allows Monaco to load the correct syntax highlighting grammar dynamically
    if (!fileName) {
      return 'plaintext';
    }
    if (fileName.endsWith('.py')) {
      return 'python';
    }
    if (
      fileName.endsWith('.cc') ||
      fileName.endsWith('.cpp') ||
      fileName.endsWith('.c') ||
      fileName.endsWith('.h') ||
      fileName.endsWith('.hpp')
    ) {
      return 'cpp';
    }
    if (fileName.endsWith('.ts') || fileName.endsWith('.js')) {
      return 'typescript';
    }
    if (fileName.endsWith('.java')) {
      return 'java';
    }
    if (fileName.endsWith('.go')) {
      return 'go';
    }
    if (fileName.endsWith('.rs')) {
      return 'rust';
    }
    if (fileName.endsWith('.sql')) {
      return 'sql';
    }
    if (fileName.endsWith('.sh')) {
      return 'shell';
    }
    return 'plaintext';
  }

  private disposeEditor(): void {
    this.clearWidgets();
    if (this.editor) {
      this.editor.dispose();
      this.editor = undefined;
    }
    this.decorationsCollection = undefined;
    this.isInitializingEditor = false;
  }

  private clearWidgets(): void {
    if (this.editor && this.contentWidgets.length > 0) {
      for (const widget of this.contentWidgets) {
        this.editor.removeContentWidget(widget);
      }
    }
    this.contentWidgets = [];
  }

  private renderDecorationsAndWidgets(monacoObj: typeof monaco): void {
    if (!this.editor || !this.frame) {
      return;
    }

    this.clearWidgets();

    const targetLineNumber = this.targetLineNumber;
    const firstLineNumber = this.firstLineNumber;

    // Highlight the selected line or column range in the stack trace
    if (targetLineNumber !== undefined) {
      const modelLineNumber = targetLineNumber - firstLineNumber + 1;
      if (modelLineNumber >= 1 && modelLineNumber <= this.frame.lines.length) {
        const metric = this.lineMetric(targetLineNumber);

        // Determine the AST Start Column. Try reading from the active metric payload first,
        // then fall back to the component inputs if the metric lacks fine-grained AST information.
        const metricWithCols = metric as Metric & {
          columnNumber?: number;
          endColumnNumber?: number;
        };
        const startCol =
          (metricWithCols?.columnNumber && metricWithCols.columnNumber > 0
            ? metricWithCols.columnNumber
            : undefined) ??
          (this.targetStartColumn && this.targetStartColumn > 0
            ? this.targetStartColumn
            : undefined);

        // Determine the AST End Column using the same fallback pipeline.
        const endCol =
          (metricWithCols?.endColumnNumber && metricWithCols.endColumnNumber > 0
            ? metricWithCols.endColumnNumber
            : undefined) ??
          (this.targetEndColumn && this.targetEndColumn > 0
            ? this.targetEndColumn
            : undefined);

        // Clamp the boundaries so Monaco doesn't crash if AST metadata exceeds the line length.
        const maxCol =
          this.editor.getModel()?.getLineMaxColumn(modelLineNumber) ??
          (this.frame.lines[modelLineNumber - 1]?.length ?? 0) + 1;

        const effectiveStartCol =
          startCol !== undefined ? Math.max(1, Math.min(startCol, maxCol)) : 1;
        const effectiveEndCol =
          endCol !== undefined && endCol > effectiveStartCol
            ? Math.min(endCol, maxCol)
            : startCol !== undefined
              ? maxCol
              : 1;

        // If we don't have AST level column indices, we tell Monaco to highlight the entire block line
        const hasColumnRange = startCol !== undefined;

        this.decorationsCollection = this.editor.createDecorationsCollection([
          {
            range: new monacoObj.Range(
              modelLineNumber,
              effectiveStartCol + 1, // Monaco columns are 1-indexed natively, but these inputs process differently
              modelLineNumber,
              effectiveEndCol + 1,
            ),
            options: {
              isWholeLine: !hasColumnRange,
              className: 'line-selected',
              linesDecorationsClassName: 'line-selected-gutter',
            },
          },
        ]);
      }
    }

    // Add metric content widget for targetLineNumber
    if (this.targetLineNumber !== undefined) {
      const actualLineNumber = this.targetLineNumber;
      const modelLineNumber = actualLineNumber - firstLineNumber + 1;

      if (modelLineNumber >= 1 && modelLineNumber <= this.frame.lines.length) {
        const metric = this.lineMetric(actualLineNumber);
        if (metric) {
          const timeStr = metric.timePs
            ? this.formatDurationPs(metric.timePs)
            : '';
          const flopsStr = metric.flopsUtilization
            ? this.percent(metric.flopsUtilization)
            : '';

          if (timeStr || flopsStr) {
            const maxColumn =
              this.editor.getModel()?.getLineMaxColumn(modelLineNumber) ?? 1;
            const widget = new LineMetricWidget(
              `line-metric-${modelLineNumber}`,
              modelLineNumber,
              maxColumn,
              timeStr,
              flopsStr,
              monacoObj,
            );
            this.contentWidgets.push(widget);
            this.editor.addContentWidget(widget);
          }
        }
      }
    }
  }

  percent = utils.percent;
  formatDurationPs = utils.formatDurationPs;

  getEditorForTesting(): monaco.editor.IStandaloneCodeEditor | undefined {
    return this.editor;
  }

  getContentWidgetsForTesting(): monaco.editor.IContentWidget[] {
    return this.contentWidgets;
  }
}

/** A Monaco content widget that displays a metric badge on a specific line of code. */
export class LineMetricWidget implements monaco.editor.IContentWidget {
  private readonly domNode: HTMLElement;

  static readonly TIME_BADGE_TOOLTIP =
    'The total execution time for all HLO operations generated from this line, including the time spent in any descendant operations.';
  static readonly FLOPS_BADGE_TOOLTIP =
    'The average FLOPS utilization for all HLO operations generated from this line, including the FLOPS of any descendant operations.';

  constructor(
    private readonly widgetId: string,
    private readonly lineNumber: number,
    private readonly columnNumber: number,
    timeStr: string,
    flopsStr: string,
    private readonly monacoObj?: typeof monaco,
  ) {
    this.domNode = document.createElement('span');
    this.domNode.className = 'line-metric-widget';
    this.domNode.style.display = 'inline-flex';
    this.domNode.style.alignItems = 'center';
    this.domNode.style.gap = '8px';
    this.domNode.style.marginLeft = '16px';
    this.domNode.style.userSelect = 'text';
    this.domNode.style.cursor = 'text';

    // Crucial: We must prevent mouse events from bubbling up to the Monaco editor.
    // Because this DOM node is injected directly into Monaco's layout overlay, clicking
    // or selecting text inside the metric badge would otherwise trigger Monaco's Native cursor
    // positioning, line selecting, or scrolling logic, leading to UI jitter/bugs.
    this.domNode.addEventListener('mousedown', (e) => { e.stopPropagation(); });
    this.domNode.addEventListener('mouseup', (e) => { e.stopPropagation(); });
    this.domNode.addEventListener('mousemove', (e) => { e.stopPropagation(); });
    this.domNode.addEventListener('dblclick', (e) => { e.stopPropagation(); });
    this.domNode.addEventListener('contextmenu', (e) => { e.stopPropagation(); });

    if (timeStr) {
      const timeBadge = document.createElement('span');
      timeBadge.className = 'metric-badge time-badge';
      timeBadge.title = LineMetricWidget.TIME_BADGE_TOOLTIP;
      timeBadge.innerText = `Time: ${timeStr}`;
      this.domNode.appendChild(timeBadge);
    }
    if (flopsStr) {
      const flopsBadge = document.createElement('span');
      flopsBadge.className = 'metric-badge flops-badge';
      flopsBadge.title = LineMetricWidget.FLOPS_BADGE_TOOLTIP;
      flopsBadge.innerText = `FLOPS: ${flopsStr}`;
      this.domNode.appendChild(flopsBadge);
    }
  }

  /** Returns the unique identifier for this widget instance. */
  getId(): string {
    return this.widgetId;
  }

  /** Returns the DOM node that physically represents the widget on the page. */
  getDomNode(): HTMLElement {
    return this.domNode;
  }

  /**
   * Returns the positioning preference for the widget relative to the editor text.
   * This dictates exactly which line and column the widget attaches to.
   */
  getPosition(): monaco.editor.IContentWidgetPosition | null {
    const preference =
      this.monacoObj?.editor?.ContentWidgetPositionPreference?.EXACT ?? 0;
    return {
      position: {
        lineNumber: this.lineNumber,
        column: this.columnNumber,
      },
      preference: [preference],
    };
  }
}
