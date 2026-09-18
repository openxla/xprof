import {ChangeDetectionStrategy, Component, effect, input} from '@angular/core';
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
  selector: 'normalized-accelerator-performance-view',
  templateUrl: './normalized_accelerator_performance_view.ng.html',
  styleUrls: ['./normalized_accelerator_performance_view.scss'],
  imports: [MatExpansionPanel, MatExpansionPanelHeader, MatExpansionPanelTitle],
})
export class NormalizedAcceleratorPerformanceView {
  /** The run environment data. */
  readonly normalizedAcceleratorPerformance =
    input<NormalizedAcceleratorPerformance | null>(null);

  title = 'GCU/NAP Details';
  backgroundInfos: string[] = [];
  totalNapsInfos: string[] = [];
  computeCostInfos: string[] = [];
  computeProductivityInfos: string[] = [];

  constructor() {
    effect(() => {
      const data =
        this.normalizedAcceleratorPerformance() || DEFAULT_SIMPLE_DATA_TABLE;
      const p: Record<string, string> =
        (data.p as Record<string, string>) || {};

      const backgroundInfos: string[] = [];
      backgroundInfos.push(p['background_link_0'] || '');
      backgroundInfos.push(p['background_link_1'] || '');
      this.backgroundInfos = backgroundInfos.filter((info) => !!info);

      const totalNapsInfos: string[] = [];
      totalNapsInfos.push(p['total_naps_line_0'] || '');
      totalNapsInfos.push(p['total_naps_line_1'] || '');
      totalNapsInfos.push(p['total_naps_line_2'] || '');
      this.totalNapsInfos = totalNapsInfos.filter((info) => !!info);

      const computeCostInfos: string[] = [];
      computeCostInfos.push(p['training_cost_line_0'] || '');
      computeCostInfos.push(p['training_cost_line_1'] || '');
      this.computeCostInfos = computeCostInfos.filter((info) => !!info);

      const computeProductivityInfos: string[] = [];
      computeProductivityInfos.push(p['training_productivity_line_0'] || '');
      computeProductivityInfos.push(p['training_productivity_line_1'] || '');
      this.computeProductivityInfos = computeProductivityInfos.filter(
        (info) => !!info,
      );
    });
  }
}
