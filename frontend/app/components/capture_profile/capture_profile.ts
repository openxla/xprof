import {CommonModule} from '@angular/common';
import {Component, OnDestroy} from '@angular/core';
import {MatButtonModule} from '@angular/material/button';
import {MatDialog} from '@angular/material/dialog';
import {MatProgressSpinnerModule} from '@angular/material/progress-spinner';
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
  standalone: true,
  selector: 'capture-profile',
  templateUrl: 'capture_profile.ng.html',
  styleUrls: ['capture_profile.scss'],
  imports: [
    CommonModule,
    MatButtonModule,
    MatProgressSpinnerModule,
  ],
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
          this.dataService.captureProfile(options as CaptureProfileOptions)
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
                        document.dispatchEvent(new Event('plugin-reload'));
                      }, DELAY_TIME_MS);
                    }
                  },
                  error => {
                    console.error(error);
                    this.store.dispatch(
                        setCapturingProfileAction({capturingProfile: false}));
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
                    this.openSnackBar(
                        'Failed to capture profile: ' + errorMessage);
                  });
        });
  }

  ngOnDestroy() {
    // Unsubscribes all pending subscriptions.
    this.destroyed.next();
    this.destroyed.complete();
  }
}
