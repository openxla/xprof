import {
  ChangeDetectionStrategy,
  Component,
  OnDestroy,
  inject,
} from '@angular/core';
import {ActivatedRoute, Params} from '@angular/router';
import {Store} from '@ngrx/store';
import {combineLatest, ReplaySubject} from 'rxjs';
import {takeUntil} from 'rxjs/operators';

import {Throbber} from 'org_xprof/frontend/app/common/classes/throbber';
import {SimpleDataTable} from 'org_xprof/frontend/app/common/interfaces/data_table';
import {alignTables} from 'org_xprof/frontend/app/common/utils/diff_utils';
import {setLoadingState} from 'org_xprof/frontend/app/common/utils/utils';
import {Dashboard} from 'org_xprof/frontend/app/components/chart/dashboard/dashboard';
import {BaseDiffService} from 'org_xprof/frontend/app/services/data_service_v2/diff_service';
import {setCurrentToolStateAction} from 'org_xprof/frontend/app/store/actions';

/** A perf counters component. */
@Component({
  changeDetection: ChangeDetectionStrategy.Default,
  standalone: false,
  selector: 'perf-counters',
  templateUrl: './perf_counters.ng.html',
  styleUrls: ['./perf_counters.scss'],
})
export class PerfCounters extends Dashboard implements OnDestroy {
  tool = 'perf_counters';
  host = '';
  /** Handles on-destroy Subject, used to unsubscribe. */
  private readonly destroyed = new ReplaySubject<void>(1);
  readonly pageSizeOptions = [30, 50, 100, 200];
  private readonly throbber = new Throbber(this.tool);
  private readonly diffService = inject(BaseDiffService);

  sessionId = '';
  showZeroValues = false;

  deviceType = '';

  constructor(
    route: ActivatedRoute,
    private readonly store: Store<{}>,
  ) {
    super();
    combineLatest([route.params, route.queryParams])
      .pipe(takeUntil(this.destroyed))
      .subscribe(([params, queryParams]) => {
        this.sessionId = params['sessionId'] || this.sessionId;
        this.processQueryParams(queryParams);
        this.update();
      });
    this.store.dispatch(setCurrentToolStateAction({currentTool: this.tool}));
  }

  processQueryParams(params: Params) {
    this.sessionId = params['run'] || params['sessionId'] || this.sessionId;
    this.tool = params['tag'] || this.tool;
    this.host = params['host'] || this.host;
  }

  update() {
    setLoadingState(true, this.store, 'Loading perf counters data');
    this.throbber.start();

    const params = new Map<string, string>([
      ['show_zeros', this.showZeroValues ? '1' : '0'],
    ]);
    this.diffService
      .getDiffData(this.sessionId, this.tool, {
        parameters: params,
      })
      .pipe(takeUntil(this.destroyed))
      .subscribe(({active, baseline}) => {
        this.throbber.stop();
        setLoadingState(false, this.store);
        this.parseData(
          this.mergeTables(
            active as SimpleDataTable | null,
            baseline as SimpleDataTable | null,
          ),
        );
      });
  }

  mergeTables(
    active: SimpleDataTable | null,
    baseline: SimpleDataTable | null,
  ): SimpleDataTable | null {
    if (!active || !active.cols) return null;
    if (!baseline || !baseline.cols) return active;

    const counterCol = active.cols.findIndex((col) => col.id === 'Counter');
    const descCol = active.cols.findIndex((col) => col.id === 'Description');
    const valueCol = active.cols.findIndex((col) => col.id === 'Value (Hex)');

    if (counterCol === -1 || valueCol === -1) {
      return active;
    }

    const getRowKey = (row: google.visualization.DataObjectRow) => {
      return row.c?.[counterCol]?.v?.toString() ?? '';
    };

    const activeRows = active.rows || [];
    const baselineRows = baseline.rows || [];
    const aligned = alignTables(activeRows, baselineRows, getRowKey);

    const newCols = [...active.cols];
    const baselineValueColIndex = newCols.length;
    newCols.push({
      id: 'baseline_value',
      label: 'Baseline Value',
      type: 'number',
    });

    const newRows: google.visualization.DataObjectRow[] = [];

    for (const compRow of aligned.values()) {
      const activeRow = compRow.active;
      const baselineRow = compRow.baseline;

      const newCells: google.visualization.DataObjectCell[] = [];
      const baseRow = activeRow || baselineRow;
      if (!baseRow) continue;

      if (counterCol !== -1) {
        newCells[counterCol] = {...(baseRow.c?.[counterCol] || {v: ''})};
      }
      if (descCol !== -1) {
        newCells[descCol] = {...(baseRow.c?.[descCol] || {v: ''})};
      }

      const activeVal = Number(activeRow?.c?.[valueCol]?.v ?? 0);
      const baselineVal = Number(baselineRow?.c?.[valueCol]?.v ?? 0);
      const diffVal = activeVal - baselineVal;

      newCells[valueCol] = {
        v: diffVal,
        f: diffVal >= 0 ? `0x${diffVal.toString(16)}` : `-0x${Math.abs(diffVal).toString(16)}`,
      };

      newCells[baselineValueColIndex] = {
        v: baselineVal,
        f: baselineVal >= 0 ? `0x${baselineVal.toString(16)}` : `-0x${Math.abs(baselineVal).toString(16)}`,
      };

      newRows.push({c: newCells});
    }

    return {
      cols: newCols,
      rows: newRows,
      p: active.p,
    };
  }


  override parseData(data: SimpleDataTable | null) {
    if (!data) return;

    const dataTable = new google.visualization.DataTable(data);

    this.deviceType = dataTable.getTableProperty('device_type');

    const counterColumnIndex = dataTable.getColumnIndex('Counter');
    // Show 'Description' values as tooltips over the 'Counter' values.
    const descriptionColumnIndex = dataTable.getColumnIndex('Description');
    if (descriptionColumnIndex !== -1) {
      const pattern = '<div title="{1}">{0}</div>';
      const formatter = new google.visualization.PatternFormat(pattern);
      formatter.format(
        dataTable,
        /* srcColumnIndices= */ [counterColumnIndex, descriptionColumnIndex],
        /* dstColumnIndex= */ counterColumnIndex,
      );
    }

    // Visible columns
    const valueColumnIndex = dataTable.getColumnIndex('Value (Hex)');
    this.columns = [
      counterColumnIndex,
      {sourceColumn: valueColumnIndex, label: 'Value (Dec)'},
      valueColumnIndex,
    ];

    this.dataTable = dataTable;

    this.updateView();
  }

  ngOnDestroy() {
    // Unsubscribes all pending subscriptions.
    setLoadingState(false, this.store);
    this.destroyed.next();
    this.destroyed.complete();
  }
  updateShowZeroValues(showZeroValuesCheckbox: boolean) {
    this.update();
  }

  getAdditionalParams() {
    const params = new Map<string, string>();
    return params;
  }
}
