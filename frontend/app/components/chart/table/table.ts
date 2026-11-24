import {
  ChangeDetectionStrategy,
  Component,
  DestroyRef,
  ElementRef,
  inject,
  Input,
  OnChanges,
  OnInit,
  SimpleChanges,
  ViewChild,
} from '@angular/core';
import {loadGoogleChartPackages} from 'org_xprof/frontend/app/common/utils/utils';

/** A table view component. */
@Component({
  changeDetection: ChangeDetectionStrategy.Default,standalone: false,
  selector: 'table',
  templateUrl: './table.ng.html',
  styleUrls: ['./table.scss']
})
export class Table implements OnChanges, OnInit {
  @Input() dataView?: google.visualization.DataView;
  @Input() showRowNumber = false;
  @Input() page = 'disable';
  @Input() pageSizeOptions: number[] = [];
  @Input() pageSize = 10;

  table?: google.visualization.Table;

  @ViewChild('table', {static: true}) tableRef!: ElementRef;
  private readonly destroyRef = inject(DestroyRef);

  ngOnInit() {
    this.populateDefaultPageSize();
    this.loadGoogleChart();
  }

  ngOnChanges(changes: SimpleChanges) {
    this.drawTable();
  }

  drawTable() {
    if (!this.table) {
      return;
    }
    if (!this.dataView) {
      this.table.clearChart?.();
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
    const cancelLoad = loadGoogleChartPackages(['table'], () => {
      this.table = new google.visualization.Table(this.tableRef.nativeElement);
      this.drawTable();
    });
    this.destroyRef.onDestroy(cancelLoad);
  }
}
