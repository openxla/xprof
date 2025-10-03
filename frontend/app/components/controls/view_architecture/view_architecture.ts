import {Component, inject, Input, OnDestroy, OnInit} from '@angular/core';
import {DATA_SERVICE_INTERFACE_TOKEN, DataServiceV2Interface, } from 'org_xprof/frontend/app/services/data_service_v2/data_service_v2_interface';
import {ReplaySubject} from 'rxjs';
import {takeUntil} from 'rxjs/operators';

/**
 * A 'View Architecture' button component which currently generates a graphviz
 * URL for the device (TPU/GPU) utilization viewer based on the used device
 * architecture in the program code.
 */
@Component({
  standalone: false,
  selector: 'view-architecture',
  templateUrl: './view_architecture.ng.html',
  styleUrls: ['./view_architecture.scss'],
})
export class ViewArchitecture implements OnInit, OnDestroy {
  @Input() sessionId = '';

  private readonly dataService: DataServiceV2Interface =
      inject(DATA_SERVICE_INTERFACE_TOKEN);
  hideViewArchitectureButton = true;
  private readonly destroyed = new ReplaySubject<void>(1);

  ngOnInit() {
    this.dataService.getConfig()
        .pipe(takeUntil(this.destroyed))
        .subscribe((config) => {
          this.hideViewArchitectureButton =
              config?.hideCaptureProfileButton || false;
        });
  }

  ngOnDestroy() {
    this.destroyed.next();
    this.destroyed.complete();
  }

  viewArchitecture() {
    this.dataService.openUtilizationGraphviz(this.sessionId);
  }
}
