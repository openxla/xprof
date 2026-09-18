import {
  AfterViewInit,
  ChangeDetectionStrategy,
  Component,
  ElementRef,
  effect,
  input,
  viewChild,
} from '@angular/core';
import {MatDivider} from '@angular/material/divider';
import {type SimpleDataTable} from 'org_xprof/frontend/app/common/interfaces/data_table';

/** A max-infeed-detail view component. */
@Component({
  changeDetection: ChangeDetectionStrategy.OnPush,
  selector: 'max-infeed-detail',
  templateUrl: './max_infeed_detail.ng.html',
  styleUrls: ['./max_infeed_detail.scss'],
  imports: [MatDivider],
})
export class MaxInfeedDetail implements AfterViewInit {
  /** Whether it is a TPU profile. */
  readonly isTpu = input(false);

  /** The table of the core with the maximum infeed at each step. */
  readonly maxInfeedCoreTable = input<SimpleDataTable | null>(null);

  readonly tableRef = viewChild<ElementRef>('table');

  table: google.visualization.Table | null = null;

  constructor() {
    effect(() => {
      this.maxInfeedCoreTable();
      this.drawTable();
    });
  }

  ngAfterViewInit() {
    this.loadGoogleChart();
  }

  drawTable() {
    const maxInfeedCoreTable = this.maxInfeedCoreTable();
    if (!this.table || !maxInfeedCoreTable) {
      return;
    }
    const dataTable = new google.visualization.DataTable(maxInfeedCoreTable);
    if (dataTable.getNumberOfColumns() < 1) {
      return;
    }
    const options = {
      showRowNumber: false,
      cssClassNames: {
        'headerCell': 'max-infeed-detail-table-header-cell',
        'tableCell': 'max-infeed-detail-table-table-cell',
      },
      width: '100%',
    };
    this.table.draw(dataTable, options);
  }

  loadGoogleChart() {
    if (!google || !google.charts) {
      setTimeout(() => {
        this.loadGoogleChart();
      }, 100);
    }

    google.charts.safeLoad({'packages': ['table']});
    google.charts.setOnLoadCallback(() => {
      const tableEl = this.tableRef()?.nativeElement;
      if (!tableEl) return;
      this.table = new google.visualization.Table(tableEl);
      this.drawTable();
    });
  }
}
