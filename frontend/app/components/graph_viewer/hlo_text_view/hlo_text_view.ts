import {Component, inject, Input, OnDestroy} from '@angular/core';
import {Throbber} from 'org_xprof/frontend/app/common/classes/throbber';
import {FileExtensionType} from 'org_xprof/frontend/app/common/constants/enums';
import {DATA_SERVICE_INTERFACE_TOKEN, DataServiceV2Interface} from 'org_xprof/frontend/app/services/data_service_v2/data_service_v2_interface';
import {ReplaySubject} from 'rxjs';
import {takeUntil} from 'rxjs/operators';

declare interface ToggleButtonItems {
  text: string;
  value: string;
  tooltip: string;
}

const TOGGLE_BUTTON_ITEMS: ToggleButtonItems[] = [
  {
    text: 'SHORT',
    value: FileExtensionType.SHORT_TEXT,
    tooltip: 'Load Short HLO Text',
  },
  {
    text: 'LONG',
    value: FileExtensionType.LONG_TEXT,
    tooltip: 'Load Long HLO Text',
  },
];

/** An Hlo text view component. */
@Component({
  standalone: false,
  selector: 'hlo-text-view',
  templateUrl: './hlo_text_view.ng.html',
  styleUrls: ['./hlo_text_view.scss'],
})
export class HloTextView implements OnDestroy {
  /** The hlo module name. */
  @Input() moduleName = '';
  /** Includes metadata in the proto. */
  @Input() showMetadata = false;
  /** The session id parsed in parent component. */
  @Input() sessionId = '';

  readonly toggleButtonItems = TOGGLE_BUTTON_ITEMS;
  /** Handles on-destroy Subject, used to unsubscribe. */
  private readonly destroyed = new ReplaySubject<void>(1);
  private readonly throbber = new Throbber('hlo_text_view');
  private readonly dataService: DataServiceV2Interface =
      inject(DATA_SERVICE_INTERFACE_TOKEN);

  hloText = '';
  loading = false;
  loadingMessage = '';
  downloadedTextParams = {
    moduleName: '',
    showMetadata: false,
    textType: '', // SHORT_TEXT | LONG_TEXT
  };

  downloadHloText(type: string) {
    this.setLoadingMessage(type, this.showMetadata);
    this.hloText = '';
    this.throbber.start();
    this.dataService
        .downloadHloProto(
            this.sessionId,
            this.moduleName,
            type,
            this.showMetadata,
            )
        .pipe(takeUntil(this.destroyed))
        .subscribe((data) => {
          this.throbber.stop();
          this.loading = false;
          this.hloText = data as string;
          this.downloadedTextParams = {
            moduleName: this.moduleName,
            showMetadata: this.showMetadata,
            textType: type,
          };
        });
  }

  getDownloadStatusMessage() {
    const metadataMessage = this.downloadedTextParams.showMetadata
      ? 'with metadata '
      : '';
    return `(Loaded: hlo ${this.downloadedTextParams.textType} ${metadataMessage}for module ${this.downloadedTextParams.moduleName})`;
  }

  setLoadingMessage(type: string, showMetadata: boolean) {
    this.loading = true;
    this.loadingMessage =
      'Loading hlo ' +
      (type === FileExtensionType.SHORT_TEXT ? 'short text' : 'long text') +
      (showMetadata ? ' (with metadata)' : ' (w/o metadata)');
  }

  ngOnDestroy() {
    // Unsubscribes all pending subscriptions.
    this.destroyed.next();
    this.destroyed.complete();
  }
}
