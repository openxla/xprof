import {NgFor} from '@angular/common';
import {ChangeDetectionStrategy, Component, input} from '@angular/core';
import {
  MatExpansionPanel,
  MatExpansionPanelHeader,
  MatExpansionPanelTitle,
} from '@angular/material/expansion';
import {
  DEFAULT_SIMPLE_DATA_TABLE,
  type NormalizedAcceleratorPerformance,
} from 'org_xprof/frontend/app/common/interfaces/data_table';

/** A normalized accelerator performance view component. */
@Component({
  standalone: true,
  changeDetection: ChangeDetectionStrategy.OnPush,
  selector: 'normalized-accelerator-performance-view',
  templateUrl: './normalized_accelerator_performance_view.ng.html',
  styleUrls: ['./normalized_accelerator_performance_view.scss'],
  imports: [
    MatExpansionPanel,
    MatExpansionPanelHeader,
    MatExpansionPanelTitle,
    NgFor,
  ],
})
export class NormalizedAcceleratorPerformanceView {
  /** The run environment data. */
  readonly normalizedAcceleratorPerformance =
    input<NormalizedAcceleratorPerformance | null>(null);

  title = 'GCU/NAP Details';

  private get props(): Record<string, string> {
    const data =
      this.normalizedAcceleratorPerformance() || DEFAULT_SIMPLE_DATA_TABLE;
    return (data.p as Record<string, string>) || {};
  }

  get backgroundInfos(): string[] {
    const p = this.props;
    return [p['background_link_0'] || '', p['background_link_1'] || ''].filter(
      (info) => !!info,
    );
  }

  get totalNapsInfos(): string[] {
    const p = this.props;
    return [
      p['total_naps_line_0'] || '',
      p['total_naps_line_1'] || '',
      p['total_naps_line_2'] || '',
    ].filter((info) => !!info);
  }

  get computeCostInfos(): string[] {
    const p = this.props;
    return [
      p['training_cost_line_0'] || '',
      p['training_cost_line_1'] || '',
    ].filter((info) => !!info);
  }

  get computeProductivityInfos(): string[] {
    const p = this.props;
    return [
      p['training_productivity_line_0'] || '',
      p['training_productivity_line_1'] || '',
    ].filter((info) => !!info);
  }
}
