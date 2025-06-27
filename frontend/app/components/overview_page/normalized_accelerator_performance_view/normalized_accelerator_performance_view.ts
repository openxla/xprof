import {Component, Input} from '@angular/core';
import {DEFAULT_SIMPLE_DATA_TABLE, type NormalizedAcceleratorPerformance} from 'org_xprof/frontend/app/common/interfaces/data_table';

/** A normalized accelerator performance view component. */
@Component({
  standalone: false,
  selector: 'normalized-accelerator-performance-view',
  templateUrl: 'normalized_accelerator_performance_view.ng.html',
  styleUrls: ['normalized_accelerator_performance_view.scss']
})
export class NormalizedAcceleratorPerformanceView {
  /** The run environment data. */
  @Input()
  set normalizedAcceleratorPerformance(data: NormalizedAcceleratorPerformance|
                                       null) {
    data = data || DEFAULT_SIMPLE_DATA_TABLE;
    data.p = data.p || {};

    this.backgroundInfos = [];
    this.backgroundInfos.push(data.p['background_link_0'] || '');
    this.backgroundInfos.push(data.p['background_link_1'] || '');
    this.backgroundInfos = this.backgroundInfos.filter(info => !!info);

    this.totalNapsInfos = [];
    this.totalNapsInfos.push(data.p['total_naps_line_0'] || '');
    this.totalNapsInfos.push(data.p['total_naps_line_1'] || '');
    this.totalNapsInfos.push(data.p['total_naps_line_2'] || '');
    this.totalNapsInfos = this.totalNapsInfos.filter(info => !!info);

    this.computeCostInfos = [];
    this.computeCostInfos.push(data.p['training_cost_line_0'] || '');
    this.computeCostInfos.push(data.p['training_cost_line_1'] || '');
    this.computeCostInfos = this.computeCostInfos.filter(info => !!info);

    this.computeProductivityInfos = [];
    this.computeProductivityInfos.push(
        data.p['training_productivity_line_0'] || '');
    this.computeProductivityInfos.push(
        data.p['training_productivity_line_1'] || '');
    this.computeProductivityInfos =
        this.computeProductivityInfos.filter(info => !!info);
  }

  title = 'GCU/NAP Details';
  backgroundInfos: string[] = [];
  totalNapsInfos: string[] = [];
  computeCostInfos: string[] = [];
  computeProductivityInfos: string[] = [];
}
