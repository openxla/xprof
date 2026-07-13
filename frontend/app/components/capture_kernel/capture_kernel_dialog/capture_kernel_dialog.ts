import {CommonModule} from '@angular/common';
import {ChangeDetectionStrategy, Component} from '@angular/core';
import {MatButtonModule} from '@angular/material/button';
import {MatDialogModule, MatDialogRef} from '@angular/material/dialog';
import {KernelAnalysisComponent} from 'org_xprof/frontend/app/components/kernel_analysis/kernel_analysis.component';

/** A capture kernel dialog component. */
@Component({
  changeDetection: ChangeDetectionStrategy.Default,
  standalone: true,
  selector: 'capture-kernel-dialog',
  templateUrl: './capture_kernel_dialog.ng.html',
  styleUrls: ['./capture_kernel_dialog.scss'],
  imports: [
    CommonModule,
    MatButtonModule,
    MatDialogModule,
    KernelAnalysisComponent,
  ],
})
export class CaptureKernelDialog {
  closeButtonLabel = 'Close';

  constructor(private readonly dialogRef: MatDialogRef<CaptureKernelDialog>) {}

  close() {
    this.dialogRef.close();
  }
}
