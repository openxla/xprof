import 'org_xprof/frontend/app/common/interfaces/window';

import {Component, inject} from '@angular/core';
import {DATA_SERVICE_INTERFACE_TOKEN, DataServiceV2Interface} from 'org_xprof/frontend/app/services/data_service_v2/data_service_v2_interface';
import {firstValueFrom, ReplaySubject} from 'rxjs';
import {takeUntil} from 'rxjs/operators';

/** An empty page component. */
@Component({
  standalone: false,
  selector: 'empty-page',
  templateUrl: './empty_page.ng.html',
  styleUrls: ['./empty_page.css']
})
export class EmptyPage {
  private readonly destroyed = new ReplaySubject<void>(1);

  hideCaptureProfileButton = true;

  private readonly dataService: DataServiceV2Interface =
      inject(DATA_SERVICE_INTERFACE_TOKEN);

  inColab = !!(window.parent.TENSORBOARD_ENV || {}).IN_COLAB;

   ngOnInit() {
    this.fetchProfilerConfig();
  }

  async fetchProfilerConfig() {
    const config = await firstValueFrom(
        this.dataService.getConfig().pipe(takeUntil(this.destroyed)));
    if (config) {
      this.hideCaptureProfileButton = config.hideCaptureProfileButton;
    }
  }

  ngOnDestroy() {
    this.destroyed.next();
    this.destroyed.complete();
  }
}
