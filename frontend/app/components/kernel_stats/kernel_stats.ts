import {
  ChangeDetectionStrategy,
  ChangeDetectorRef,
  Component,
  inject,
  Input,
  OnDestroy,
} from '@angular/core';
import {MatDivider} from '@angular/material/divider';
import {Store} from '@ngrx/store';
import {SimpleDataTable} from 'org_xprof/frontend/app/common/interfaces/data_table';
import {getKernelStatsDataState} from 'org_xprof/frontend/app/store/common_data_store/selectors';
import {ReplaySubject} from 'rxjs';
import {takeUntil} from 'rxjs/operators';
import {ExportAsCsv} from '../controls/export_as_csv/export_as_csv';
import {KernelStatsTable} from './kernel_stats_table/kernel_stats_table';

/** A Kernel Stats component. */
@Component({
  changeDetection: ChangeDetectionStrategy.OnPush,
  standalone: true,
  selector: 'kernel-stats',
  templateUrl: './kernel_stats.ng.html',
  styleUrls: ['./kernel_stats.scss'],
  imports: [ExportAsCsv, KernelStatsTable, MatDivider],
})
export class KernelStats implements OnDestroy {
  /** Handles on-destroy Subject, used to unsubscribe. */
  private readonly cdr = inject(ChangeDetectorRef);
  private readonly destroyed = new ReplaySubject<void>(1);
  private readonly store = inject(Store);

  data: SimpleDataTable | null = null;
  hasDataRow = false;
  @Input() sessionId = '';
  @Input() tool = '';
  @Input() host = '';

  constructor() {
    this.store
      .select(getKernelStatsDataState)
      .pipe(takeUntil(this.destroyed))
      .subscribe((data: SimpleDataTable | null) => {
        this.update(data);
        this.cdr.markForCheck();
      });
  }

  update(data: SimpleDataTable | null) {
    this.data = data;
    this.hasDataRow = !!data && !!data.rows && data.rows.length > 0;
  }

  ngOnDestroy() {
    // Unsubscribes all pending subscriptions.
    this.destroyed.next();
    this.destroyed.complete();
  }
}
