import 'org_xprof/frontend/app/common/interfaces/window';

import {NgIf} from '@angular/common';
import {ChangeDetectionStrategy, Component, inject} from '@angular/core';
import {firstValueFrom, ReplaySubject} from 'rxjs';
import {defaultIfEmpty, takeUntil} from 'rxjs/operators';

import {
  DATA_SERVICE_INTERFACE_TOKEN,
  DataServiceV2Interface,
} from 'org_xprof/frontend/app/services/data_service_v2/data_service_v2_interface';

import {CaptureProfile} from '../capture_profile/capture_profile';

/** An empty page component. */
@Component({
  changeDetection: ChangeDetectionStrategy.Default,
  standalone: true,
  selector: 'empty-page',
  templateUrl: './empty_page.ng.html',
  styleUrls: ['./empty_page.scss'],
  imports: [CaptureProfile, NgIf],
})
export class EmptyPage {
  private readonly destroyed = new ReplaySubject<void>(1);

  hideCaptureProfileButton = true;

  private readonly dataService: DataServiceV2Interface = inject(
    DATA_SERVICE_INTERFACE_TOKEN,
  );

  inColab = !!(window.parent.TENSORBOARD_ENV || {}).IN_COLAB;

  ngOnInit() {
    this.fetchProfilerConfig();
  }

  async fetchProfilerConfig() {
    // Deep links show this page until SideNav routes to the tool, which can
    // destroy it before the config arrives.
    const config = await firstValueFrom(
      this.dataService
        .getConfig()
        .pipe(takeUntil(this.destroyed), defaultIfEmpty(null)),
    );
    if (config) {
      this.hideCaptureProfileButton = config.hideCaptureProfileButton;
    }
  }

  ngOnDestroy() {
    this.destroyed.next();
    this.destroyed.complete();
  }
}
