import {Component, inject, Input, OnDestroy} from '@angular/core';
import {ActivatedRoute} from '@angular/router';
import {FileExtensionType} from 'org_xprof/frontend/app/common/constants/enums';
import {DATA_SERVICE_INTERFACE_TOKEN, DataServiceV2Interface} from 'org_xprof/frontend/app/services/data_service_v2/data_service_v2_interface';
import {ReplaySubject} from 'rxjs';
import {takeUntil} from 'rxjs/operators';

import {BlobDownloader} from './blob_downloader';

declare interface DownloadMenuItem {
  text: string;
  value: string;
}

const DOWNLOAD_HLO_PROTO_MENU_ITEMS: DownloadMenuItem[] = [
  {text: 'Download as .pb', value: FileExtensionType.PROTO_BINARY},
  {text: 'Download as .pbtxt', value: FileExtensionType.PROTO_TEXT},
  {text: 'Download as .json', value: FileExtensionType.JSON},
  {text: 'Download as short text', value: FileExtensionType.SHORT_TEXT},
  {text: 'Download as long text', value: FileExtensionType.LONG_TEXT},
];

/** A component to download hlo module in proto, text or json formats. */
@Component({
  standalone: false,
  selector: 'download-hlo',
  templateUrl: './download_hlo.ng.html',
  styleUrls: ['./download_hlo.css'],
  providers: [BlobDownloader],
})
export class DownloadHlo implements OnDestroy {
  /** The hlo module name. */
  @Input() moduleName: string = '';
  /** Includes metadata in the proto. */
  @Input() showMetadata: boolean = false;

  readonly downloadMenuItems = DOWNLOAD_HLO_PROTO_MENU_ITEMS;
  private readonly destroyed = new ReplaySubject<void>(1);
  private readonly dataService: DataServiceV2Interface =
      inject(DATA_SERVICE_INTERFACE_TOKEN);
  sessionId = '';

  constructor(
      route: ActivatedRoute,
      private readonly downloader: BlobDownloader,
  ) {
    route.params.pipe(takeUntil(this.destroyed)).subscribe((params) => {
      this.sessionId = (params || {})['sessionId'] || '';
    });
  }

  downloadHloProto(type: string) {
    const fileName = this.moduleName + '.' + type;
    this.dataService
        .downloadHloProto(
            this.sessionId,
            this.moduleName,
            type,
            this.showMetadata,
            )!.pipe(takeUntil(this.destroyed))
        .subscribe((data) => {
          if (type === FileExtensionType.PROTO_BINARY) {
            this.downloader.downloadBlob(data as Blob, fileName);
          } else {
            this.downloader.downloadString(
                data as string,
                fileName,
            );
          }
        });
  }

  ngOnDestroy() {
    this.destroyed.next();
    this.destroyed.complete();
  }
}
