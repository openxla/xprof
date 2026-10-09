import {NgFor} from '@angular/common';
import {ChangeDetectionStrategy, Component, Input} from '@angular/core';
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
  changeDetection: ChangeDetectionStrategy.OnPush,
  standalone: true,
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
  @Input()
  set normalizedAcceleratorPerformance(
    data: NormalizedAcceleratorPerformance | null,
  ) {
    data = data || DEFAULT_SIMPLE_DATA_TABLE;
    data.p = data.p || {};

    this.backgroundInfos = [
      data.p['background_link_0'] || '',
      data.p['background_link_1'] || '',
    ].filter((info) => !!info);

    this.totalNapsInfos = [
      data.p['total_naps_line_0'] || '',
      data.p['total_naps_line_1'] || '',
      data.p['total_naps_line_2'] || '',
    ].filter((info) => !!info);

    this.computeCostInfos = [
      data.p['training_cost_line_0'] || '',
      data.p['training_cost_line_1'] || '',
    ].filter((info) => !!info);

    this.computeProductivityInfos = [
      data.p['training_productivity_line_0'] || '',
      data.p['training_productivity_line_1'] || '',
    ].filter((info) => !!info);
  }

  title = 'GCU/NAP Details';
  backgroundInfos: string[] = [];
  totalNapsInfos: string[] = [];
  computeCostInfos: string[] = [];
  computeProductivityInfos: string[] = [];
}
