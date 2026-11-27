import {ChangeDetectionStrategy, Component, Input} from '@angular/core';
import {
  DEFAULT_SIMPLE_DATA_TABLE,
  type InputPipelineDeviceAnalysis,
} from 'org_xprof/frontend/app/common/interfaces/data_table';

/** A summary of input pipeline analysis component. */
@Component({
  changeDetection: ChangeDetectionStrategy.Default,standalone: false,
  selector: 'analysis-summary',
  templateUrl: './analysis_summary.ng.html',
  styleUrls: ['./analysis_summary.scss'],
})
export class AnalysisSummary {
  /** The input pipeline device anaysis data. */
  @Input()
  set deviceAnalysis(analysis: InputPipelineDeviceAnalysis|null) {
    analysis = analysis || DEFAULT_SIMPLE_DATA_TABLE;
    analysis.p = analysis.p || {};
    this.inputConclusion = analysis.p['input_conclusion'] || '';
    this.summaryNextstep = this.replaceSectionName(
      analysis.p['summary_nextstep'] || '',
    );
    if (
      !this.inputConclusion ||
      this.inputConclusion.includes('No step time measured')
    ) {
      this.summaryColor = '';
    } else if (this.inputConclusion.includes('HIGHLY')) {
      this.summaryColor = 'red';
    } else if (
      this.inputConclusion.includes('MODERATE') ||
      this.inputConclusion.includes('POTENTIALLY') ||
      this.inputConclusion.includes('COULD')
    ) {
      this.summaryColor = 'orange';
    } else {
      this.summaryColor = 'green';
    }
  }

  inputConclusion = '';
  summaryNextstep = '';
  summaryColor = '';

  private replaceSectionName(summary: string): string {
    return summary
      .replace(/section 2/gi, 'Device-side analysis details section')
      .replace(/section 3/gi, 'Host-side analysis details section');
  }
}
