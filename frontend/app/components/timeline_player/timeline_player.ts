import {CommonModule} from '@angular/common';
import {
  ChangeDetectionStrategy,
  ChangeDetectorRef,
  Component,
  NgZone,
  OnDestroy,
  OnInit,
  model,
  output,
} from '@angular/core';
import {MatButtonModule} from '@angular/material/button';
import {MatIconModule} from '@angular/material/icon';
import {MatMenuModule} from '@angular/material/menu';
import {MatSliderModule} from '@angular/material/slider';
import {MatTooltipModule} from '@angular/material/tooltip';
import {TimeFormatPipe} from './time_format.pipe';

// TODO(b/554087917): Add Karma Scuba screenshot tests for the Timeline Player component.

/** Represents the internal window state exposed by the playback controls */
export interface PlaybackState {
  isPlaying: boolean;
  currentTime: number;
  playbackRate: number;
}

/** Represents the custom event detail from WASM timeline state updates */
export interface SyncEventDetail {
  events?: Array<Record<string, string | number | boolean>>;
  counters?: Array<Record<string, string | number | boolean>>;
  currentTime?: number;
  duration?: number;
  isPlaying?: boolean;
}

/** Component that renders a timeline player with scrub, play/pause controls. */
@Component({
  selector: 'timeline-player',
  standalone: true,
  changeDetection: ChangeDetectionStrategy.OnPush,
  imports: [
    CommonModule,
    MatButtonModule,
    MatIconModule,
    MatMenuModule,
    MatSliderModule,
    MatTooltipModule,
    TimeFormatPipe,
  ],
  templateUrl: 'timeline_player.ng.html',
  styleUrls: ['timeline_player.scss'],
})
export class TimelinePlayer implements OnInit, OnDestroy {
  readonly currentTime = model(0);
  readonly duration = model(100);
  readonly isPlaying = model(false);
  readonly playbackRate = model(1);

  readonly play = output<void>();
  readonly pause = output<void>();
  readonly seek = output<number>();
  readonly speedChange = output<number>();

  activeEvents: Array<Record<string, string | number | boolean>> = [];
  activeCounters: Array<Record<string, string | number | boolean>> = [];

  constructor(
    private readonly cdr: ChangeDetectorRef,
    private readonly ngZone: NgZone,
  ) {}

  ngOnInit() {
    window.addEventListener(
      'timeline-player-sync-backend',
      this.handleBackendSync,
    );
  }

  ngOnDestroy() {
    window.removeEventListener(
      'timeline-player-sync-backend',
      this.handleBackendSync,
    );
  }

  private readonly handleBackendSync = (event: Event) => {
    const customEvent = event as CustomEvent<SyncEventDetail>;
    if (customEvent.detail) {
      let dirty = false;
      if (customEvent.detail.events) {
        this.activeEvents = customEvent.detail.events;
        dirty = true;
      }
      if (customEvent.detail.counters) {
        this.activeCounters = customEvent.detail.counters;
        dirty = true;
      }
      if (customEvent.detail.currentTime !== undefined && !this.isScrubbing) {
        this.currentTime.set(customEvent.detail.currentTime);
        dirty = true;
      }
      if (customEvent.detail.duration !== undefined) {
        this.duration.set(customEvent.detail.duration);
        dirty = true;
      }
      if (customEvent.detail.isPlaying !== undefined) {
        this.isPlaying.set(customEvent.detail.isPlaying);
        dirty = true;
      }

      if (dirty) {
        this.ngZone.run(() => {
          this.cdr.detectChanges();
        });
      }
    }
  };

  togglePlay() {
    this.isPlaying.set(!this.isPlaying());

    if (!this.isPlaying()) {
      this.pause.emit();
    } else {
      this.play.emit();
    }
  }

  onSeek(event: Event | {value: string | number}) {
    let val: string | number = 0;
    if (event instanceof Event && event.target) {
      val = (event.target as HTMLInputElement).value;
    } else {
      val = (event as {value: string | number}).value;
    }
    this.currentTime.set(Number(val));
    this.seek.emit(this.currentTime());
  }

  onSpeedChange(newSpeed: number) {
    this.playbackRate.set(newSpeed);
    this.speedChange.emit(newSpeed);
  }

  private isScrubbing = false;

  onScrubStart() {
    this.isScrubbing = true;
  }

  onScrubEnd(event: Event) {
    this.isScrubbing = false;
    this.onSeek(event);
  }
}
