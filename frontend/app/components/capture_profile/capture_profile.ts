import {Component, OnDestroy, ChangeDetectionStrategy} from '@angular/core';
import {MatDialog} from '@angular/material/dialog';
import {MatSnackBar} from '@angular/material/snack-bar';
import {Store} from '@ngrx/store';
import {CaptureProfileOptions, CaptureProfileResponse} from 'org_xprof/frontend/app/common/interfaces/capture_profile';
import {DataServiceV2} from 'org_xprof/frontend/app/services/data_service_v2/data_service_v2';
import {setCapturingProfileAction} from 'org_xprof/frontend/app/store/actions';
import {getCapturingProfileState} from 'org_xprof/frontend/app/store/selectors';
import {Observable, ReplaySubject} from 'rxjs';
import {takeUntil} from 'rxjs/operators';

import {CaptureProfileDialog} from './capture_profile_dialog/capture_profile_dialog';

const DELAY_TIME_MS = 1000;

/** A capture profile view component. */
@Component({
  changeDetection: ChangeDetectionStrategy.Default,standalone: false,
  selector: 'capture-profile',
  templateUrl: './capture_profile.ng.html',
  styleUrls: ['./capture_profile.scss']
})
export class CaptureProfile implements OnDestroy {
  readonly captureButtonLabel = 'Capture Profile';
  /** Handles on-destroy Subject, used to unsubscribe. */
  private readonly destroyed = new ReplaySubject<void>(1);

  capturingProfile: Observable<boolean>;

  constructor(
      private readonly dialog: MatDialog,
      private readonly snackBar: MatSnackBar,
      private readonly dataService: DataServiceV2,
      private readonly store: Store<{}>) {
    this.capturingProfile = this.store.select(getCapturingProfileState);
  }

  private openSnackBar(message: string) {
    this.snackBar.open(message, 'Close now!', {duration: 5000});
  }

  private extractErrorMessage(error: unknown): string {
    if (!error) return 'Invalid error';
    if (typeof error === 'string') return error;
    if (typeof error !== 'object') return String(error);

    const err =
        error as {error?: unknown; message?: unknown; statusText?: unknown};
    const nested = err.error && typeof err.error === 'object' ?
        err.error as {error?: unknown; message?: unknown} :
        undefined;
    for (const candidate of [
      err.error,
      nested?.error,
      nested?.message,
      err.message,
      err.statusText,
    ]) {
      if (typeof candidate === 'string' && candidate) {
        return candidate;
      }
    }
    try {
      return JSON.stringify(error);
    } catch {
      return String(error);
    }
  }

  private handleCaptureError(error: unknown) {
    console.error(error);
    this.store.dispatch(setCapturingProfileAction({capturingProfile: false}));
    this.openSnackBar(
        'Failed to capture profile: ' + this.extractErrorMessage(error));
  }

  openDialog() {
    this.dialog.open(CaptureProfileDialog)
        .afterClosed()
        .pipe(takeUntil(this.destroyed))
        .subscribe((options) => {
          if (!options) {
            return;
          }

          this.store.dispatch(
              setCapturingProfileAction({capturingProfile: true}));
          let capture$;
          try {
            capture$ =
                this.dataService.captureProfile(options as CaptureProfileOptions);
          } catch (error: unknown) {
            this.handleCaptureError(error);
            return;
          }
          capture$
              .pipe(takeUntil(this.destroyed))
              .subscribe(
                  (response: CaptureProfileResponse) => {
                    this.store.dispatch(
                        setCapturingProfileAction({capturingProfile: false}));
                    if (!response) {
                      return;
                    }
                    if (response.error) {
                      this.openSnackBar(
                          'Failed to capture profile: ' + response.error);
                      return;
                    }
                    if (response.result) {
                      this.openSnackBar(response.result);
                      setTimeout(() => {
                        document.dispatchEvent(new Event('tensorboard-reload'));
                      }, DELAY_TIME_MS);
                    }
                  },
                  (error: unknown) => {
                    this.handleCaptureError(error);
                  });
        });
  }

  ngOnDestroy() {
    // Unsubscribes all pending subscriptions.
    this.destroyed.next();
    this.destroyed.complete();
  }
}
