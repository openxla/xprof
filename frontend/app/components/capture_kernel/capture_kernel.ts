import {CommonModule} from '@angular/common';
import {ChangeDetectionStrategy, Component, inject} from '@angular/core';
import {MatButtonModule} from '@angular/material/button';
import {MatDialog, MatDialogModule} from '@angular/material/dialog';
import {CaptureKernelDialog} from './capture_kernel_dialog/capture_kernel_dialog';

/** A capture kernel view component. */
@Component({
  changeDetection: ChangeDetectionStrategy.Default,
  selector: 'capture-kernel',
  templateUrl: './capture_kernel.ng.html',
  styleUrls: ['./capture_kernel.scss'],
  imports: [CommonModule, MatButtonModule, MatDialogModule],
})
export class CaptureKernel {
  private readonly dialog = inject(MatDialog);

  readonly captureButtonLabel = 'Capture Kernel';

  openDialog() {
    this.dialog.open(CaptureKernelDialog, {
      width: '90vw',
      maxWidth: '1200px',
      height: '90vh',
      maxHeight: '1000px',
    });
  }
}
