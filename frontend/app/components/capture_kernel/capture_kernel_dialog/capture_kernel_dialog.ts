import {CommonModule} from '@angular/common';
import {
  ChangeDetectionStrategy,
  Component,
  EventEmitter,
  Output,
} from '@angular/core';
import {MatButtonModule} from '@angular/material/button';
import {KernelAnalysisComponent} from 'org_xprof/frontend/app/components/kernel_analysis/kernel_analysis.component';

/** A capture kernel dialog component. */
@Component({
  changeDetection: ChangeDetectionStrategy.Default,
  standalone: true,
  selector: 'capture-kernel-dialog',
  templateUrl: './capture_kernel_dialog.ng.html',
  styleUrls: ['./capture_kernel_dialog.scss'],
  imports: [CommonModule, MatButtonModule, KernelAnalysisComponent],
})
export class CaptureKernelDialog {
  @Output() readonly closed = new EventEmitter<void>();

  closeButtonLabel = 'Close';

  close() {
    this.closed.emit();
  }
}
