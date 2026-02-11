import {
  AfterViewInit,
  Component,
  ElementRef,
  Input,
  OnChanges,
  SimpleChanges,
  ViewChild, ChangeDetectionStrategy,
} from '@angular/core';
import {type SimpleDataTable} from 'org_xprof/frontend/app/common/interfaces/data_table';

/** A max-infeed-detail view component. */
@Component({
  changeDetection: ChangeDetectionStrategy.Eager,standalone: false,
  selector: 'max-infeed-detail',
  templateUrl: './max_infeed_detail.ng.html',
  styleUrls: ['./max_infeed_detail.scss'],
})
export class MaxInfeedDetail implements AfterViewInit, OnChanges {
  /** Whether it is a TPU profile. */
  @Input() isTpu: boolean = false;

  /** The table of the core with the maximum infeed at each step. */
  @Input() maxInfeedCoreTable: SimpleDataTable | null = null;

  @ViewChild('table', {static: false}) tableRef!: ElementRef;

  table: google.visualization.Table | null = null;

  ngAfterViewInit() {
    this.loadGoogleChart();
  }

  ngOnChanges(changes: SimpleChanges) {
    this.drawTable();
  }

  drawTable() {
    if (!this.table || !this.maxInfeedCoreTable) {
      return;
    }
    const dataTable = new google.visualization.DataTable(
      this.maxInfeedCoreTable,
    );
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
      this.table = new google.visualization.Table(this.tableRef.nativeElement);
      this.drawTable();
    });
  }
}
