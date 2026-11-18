import '@material/web/dialog/dialog.js';

import {CommonModule} from '@angular/common';
import {
  ApplicationRef,
  ChangeDetectionStrategy,
  Component,
  ComponentRef,
  createComponent,
  CUSTOM_ELEMENTS_SCHEMA,
  ElementRef,
  EnvironmentInjector,
  inject,
  Injector,
  OnDestroy,
  Renderer2,
} from '@angular/core';
import {MatButtonModule} from '@angular/material/button';
import {Subscription} from 'rxjs';
import {CaptureKernelDialog} from './capture_kernel_dialog/capture_kernel_dialog';

/** A capture kernel view component. */
@Component({
  changeDetection: ChangeDetectionStrategy.Default,
  standalone: true,
  selector: 'capture-kernel',
  templateUrl: './capture_kernel.ng.html',
  styleUrls: ['./capture_kernel.scss'],
  imports: [CommonModule, MatButtonModule],
  schemas: [CUSTOM_ELEMENTS_SCHEMA],
})
export class CaptureKernel implements OnDestroy {
  private readonly elementRef = inject(ElementRef);
  private readonly appRef = inject(ApplicationRef);
  private readonly environmentInjector = inject(EnvironmentInjector);
  private readonly injector = inject(Injector);
  private readonly renderer = inject(Renderer2);

  readonly captureButtonLabel = 'Capture Kernel';
  isDialogOpen = false;
  private dialogEl: HTMLElement | null = null;
  private dialogComponentRef: ComponentRef<CaptureKernelDialog> | null = null;
  private closedSub: Subscription | null = null;
  private unlistenClosed: (() => void) | null = null;

  openDialog() {
    this.isDialogOpen = true;
    if (!this.dialogEl) {
      const dialogEl = this.renderer.createElement('md-dialog') as HTMLElement;
      this.renderer.addClass(dialogEl, 'capture-kernel-dialog-modal');
      this.renderer.setAttribute(dialogEl, 'quick', '');
      this.renderer.setAttribute(dialogEl, 'open', '');
      (dialogEl as unknown as {open: boolean}).open = true;
      this.unlistenClosed = this.renderer.listen(dialogEl, 'closed', () => {
        this.closeDialog();
      });

      const slotEl = this.renderer.createElement('div') as HTMLElement;
      this.renderer.setAttribute(slotEl, 'slot', 'content');
      this.renderer.addClass(slotEl, 'capture-kernel-dialog-slot');

      const compRef = createComponent(CaptureKernelDialog, {
        environmentInjector: this.environmentInjector,
        elementInjector: this.injector,
      });
      this.closedSub = compRef.instance.closed.subscribe(() => {
        this.closeDialog();
      });
      this.appRef.attachView(compRef.hostView);
      this.renderer.appendChild(slotEl, compRef.location.nativeElement);
      this.renderer.appendChild(dialogEl, slotEl);

      const hostContainer =
        this.elementRef.nativeElement.querySelector('div') ||
        this.elementRef.nativeElement;
      this.renderer.appendChild(hostContainer, dialogEl);
      compRef.changeDetectorRef.detectChanges();

      this.dialogEl = dialogEl;
      this.dialogComponentRef = compRef;
    }
    setTimeout(() => {
      const dialogEl = this.dialogEl;
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
    this.destroyDialog();
  }

  private destroyDialog() {
    if (this.unlistenClosed) {
      this.unlistenClosed();
      this.unlistenClosed = null;
    }
    if (this.closedSub) {
      this.closedSub.unsubscribe();
      this.closedSub = null;
    }
    if (this.dialogComponentRef) {
      this.appRef.detachView(this.dialogComponentRef.hostView);
      this.dialogComponentRef.destroy();
      this.dialogComponentRef = null;
    }
    if (this.dialogEl) {
      this.dialogEl.remove();
      this.dialogEl = null;
    }
  }

  ngOnDestroy() {
    this.destroyDialog();
  }
}
