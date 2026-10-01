import '@material/web/dialog/dialog.js';

import {CommonModule} from '@angular/common';
import {
  ChangeDetectionStrategy,
  Component,
  CUSTOM_ELEMENTS_SCHEMA,
  ElementRef,
  inject,
} from '@angular/core';
import {MatButtonModule} from '@angular/material/button';
import {CaptureKernelDialog} from './capture_kernel_dialog/capture_kernel_dialog';

/** A capture kernel view component. */
@Component({
  changeDetection: ChangeDetectionStrategy.Default,
  standalone: true,
  selector: 'capture-kernel',
  templateUrl: './capture_kernel.ng.html',
  styleUrls: ['./capture_kernel.scss'],
  imports: [CommonModule, MatButtonModule, CaptureKernelDialog],
  schemas: [CUSTOM_ELEMENTS_SCHEMA],
})
export class CaptureKernel {
  private readonly elementRef = inject(ElementRef);
  readonly captureButtonLabel = 'Capture Kernel';
  isDialogOpen = false;

  openDialog() {
    this.isDialogOpen = true;
    setTimeout(() => {
      const dialogEl = this.elementRef.nativeElement.querySelector(
        'md-dialog.capture-kernel-dialog-modal',
      );
      if (dialogEl?.shadowRoot) {
        let styleEl = dialogEl.shadowRoot.querySelector(
          '#top-layer-backdrop-style',
        ) as HTMLStyleElement | null;
        if (!styleEl) {
          styleEl = document.createElement('style');
          styleEl.id = 'top-layer-backdrop-style';
          styleEl.textContent = `
            dialog::backdrop {
              background: rgba(0, 0, 0, 0.32);
            }
            .scrim {
              display: none !important;
            }
          `;
          dialogEl.shadowRoot.appendChild(styleEl);
        }
      }
    });
  }

  closeDialog() {
    this.isDialogOpen = false;
  }
}
