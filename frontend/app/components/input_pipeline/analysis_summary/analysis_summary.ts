import {ChangeDetectionStrategy, Component, effect, input} from '@angular/core';
import {
  DEFAULT_SIMPLE_DATA_TABLE,
  type InputPipelineDeviceAnalysis,
} from 'org_xprof/frontend/app/common/interfaces/data_table';

/** A summary of input pipeline analysis component. */
@Component({
  changeDetection: ChangeDetectionStrategy.OnPush,
  selector: 'analysis-summary',
  templateUrl: './analysis_summary.ng.html',
  styleUrls: ['./analysis_summary.scss'],
})
export class AnalysisSummary {
  /** The input pipeline device analysis data. */
  readonly deviceAnalysis = input<InputPipelineDeviceAnalysis | null>(null);

  inputConclusion = '';
  summaryNextstep = '';
  summaryColor = 'green';

  constructor() {
    effect(() => {
      const analysis = this.deviceAnalysis() || DEFAULT_SIMPLE_DATA_TABLE;
      const p = analysis.p || {};
      this.inputConclusion = p['input_conclusion'] || '';
      this.summaryNextstep = this.replaceSectionName(
        p['summary_nextstep'] || '',
      );
      this.summaryColor = 'green';
      if (this.inputConclusion.includes('HIGHLY')) {
        this.summaryColor = 'red';
      } else if (this.inputConclusion.includes('MODERATE')) {
        this.summaryColor = 'orange';
      }
    });
  }

  private replaceSectionName(summary: string): string {
    return summary
      .replace(/section 2/g, 'Device-side analysis details section')
      .replace(/section 3/g, 'Host-side analysis details section');
  }
}
