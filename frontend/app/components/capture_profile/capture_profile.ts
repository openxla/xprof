import '@material/web/dialog/dialog.js';

import {
  ChangeDetectionStrategy,
  Component,
  ElementRef,
  inject,
  OnDestroy,
} from '@angular/core';
import {MatSnackBar} from '@angular/material/snack-bar';
import {Store} from '@ngrx/store';
import {
  CaptureProfileOptions,
  CaptureProfileResponse,
} from 'org_xprof/frontend/app/common/interfaces/capture_profile';
import {DataServiceV2} from 'org_xprof/frontend/app/services/data_service_v2/data_service_v2';
import {setCapturingProfileAction} from 'org_xprof/frontend/app/store/actions';
import {getCapturingProfileState} from 'org_xprof/frontend/app/store/selectors';
import {Observable, ReplaySubject} from 'rxjs';
import {takeUntil} from 'rxjs/operators';

const DELAY_TIME_MS = 1000;

/** A capture profile view component. */
@Component({
  changeDetection: ChangeDetectionStrategy.Default,
  standalone: false,
  selector: 'capture-profile',
  templateUrl: './capture_profile.ng.html',
  styleUrls: ['./capture_profile.scss'],
})
export class CaptureProfile implements OnDestroy {
  private readonly snackBar = inject(MatSnackBar);
  private readonly dataService = inject(DataServiceV2);
  private readonly store: Store<{}> = inject(Store);
  private readonly elementRef = inject(ElementRef);

  readonly captureButtonLabel = 'Capture Profile';
  /** Handles on-destroy Subject, used to unsubscribe. */
  private readonly destroyed = new ReplaySubject<void>(1);

  capturingProfile: Observable<boolean> = this.store.select(
    getCapturingProfileState,
  );
  isDialogOpen = false;

  private openSnackBar(message: string) {
    this.snackBar.open(message, 'Close now!', {duration: 5000});
  }

  openDialog() {
    this.isDialogOpen = true;
    setTimeout(() => {
      const dialogEl = this.elementRef.nativeElement.querySelector(
        'md-dialog.capture-profile-dialog-modal',
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

  onCaptureProfile(options: {[key: string]: string | number | boolean}) {
    this.isDialogOpen = false;
    if (!options) {
      return;
    }

    this.store.dispatch(setCapturingProfileAction({capturingProfile: true}));
    this.dataService
      .captureProfile(options as unknown as CaptureProfileOptions)
      .pipe(takeUntil(this.destroyed))
      .subscribe(
        (response: CaptureProfileResponse) => {
          this.store.dispatch(
            setCapturingProfileAction({capturingProfile: false}),
          );
          if (!response) {
            return;
          }
          if (response.error) {
            this.openSnackBar('Failed to capture profile: ' + response.error);
            return;
          }
          if (response.result) {
            this.openSnackBar(response.result);
            setTimeout(() => {
              document.dispatchEvent(new Event('plugin-reload'));
            }, DELAY_TIME_MS);
          }
        },
        (error) => {
          console.error(error);
          this.store.dispatch(
            setCapturingProfileAction({capturingProfile: false}),
          );
          let errorMessage = '';
          if (error && typeof error === 'object') {
            errorMessage = JSON.stringify(error);
            if (error.error) {
              errorMessage = error.error;
            } else if (error.message) {
              errorMessage = error.message;
            } else if (error.statusText) {
              errorMessage = error.statusText;
            }
          } else if (error) {
            errorMessage = error.toString();
          } else {
            errorMessage = 'Invalid error';
          }
          this.openSnackBar('Failed to capture profile: ' + errorMessage);
        },
      );
  }

  ngOnDestroy() {
    // Unsubscribes all pending subscriptions.
    this.destroyed.next();
    this.destroyed.complete();
  }
}
