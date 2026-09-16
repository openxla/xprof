import {CommonModule} from '@angular/common';
import {ChangeDetectionStrategy, Component, inject} from '@angular/core';
import {MatButtonModule} from '@angular/material/button';
import {MatDialogModule, MatDialogRef} from '@angular/material/dialog';
import {KernelAnalysisComponent} from 'org_xprof/frontend/app/components/kernel_analysis/kernel_analysis';

/** A capture kernel dialog component. */
@Component({
  changeDetection: ChangeDetectionStrategy.Default,
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
  private readonly dialogRef =
    inject<MatDialogRef<CaptureKernelDialog>>(MatDialogRef);

  closeButtonLabel = 'Close';

  close() {
    this.dialogRef.close();
  }
}
