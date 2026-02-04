import {CommonModule} from '@angular/common';
import {Component, ElementRef, HostListener, Input, OnChanges, OnInit, SimpleChanges, ViewChild} from '@angular/core';
import {MatOptionModule} from '@angular/material/core';
import {MatFormFieldModule} from '@angular/material/form-field';
import {MatSelectModule} from '@angular/material/select';

/** A table view component. */
@Component({
  standalone: true,
  selector: 'table',
  templateUrl: 'table.ng.html',
  styleUrls: ['table.scss'],
  imports: [
    CommonModule,
    MatFormFieldModule,
    MatSelectModule,
    MatOptionModule,
  ],
})
export class Table implements OnChanges, OnInit {
  @Input() dataView?: google.visualization.DataView;
  @Input() showRowNumber = false;
  @Input() page = 'disable';
  @Input() pageSizeOptions: number[] = [];
  @Input() pageSize = 10;

  table?: google.visualization.Table;
  height = '150px';

  @ViewChild('table', {static: false}) tableRef!: ElementRef;

  ngOnInit() {
    this.loadGoogleChart();
    this.populateDefaultPageSize();
  }

  ngOnChanges(changes: SimpleChanges) {
    this.drawTable();
  }

  @HostListener('window:resize')
  onResize() {
    const tableElement = this.tableRef.nativeElement.querySelector('table');
    if (tableElement) {
      this.height = String(Number(tableElement.clientHeight) + 20) + 'px';
    }
  }

  drawTable() {
    if (!this.table || !this.dataView) {
      return;
    }

    const options: google.visualization.TableOptions = {
      allowHtml: true,
      alternatingRowStyle: false,
      showRowNumber: this.showRowNumber,
      page: this.page,
      pageSize: this.pageSize,
      cssClassNames: {
        'headerCell': 'google-chart-table-header-cell',
        'tableCell': 'google-chart-table-table-cell',
      },
    };

    this.table.draw(this.dataView, options);

    this.onResize();
  }

  displayPageSizeSelector() {
    return this.pageSizeOptions.length > 0;
  }

  populateDefaultPageSize() {
    // when passing pageSizeOptions from parent, pageSize will by default be the
    // 1st element in the list
    if (this.pageSizeOptions.length > 0) {
      this.pageSize = this.pageSizeOptions[0];
    }
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
