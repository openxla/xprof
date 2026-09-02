import {CommonModule} from '@angular/common';
import {
  ChangeDetectionStrategy,
  ChangeDetectorRef,
  Component,
  HostListener,
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

  stepAmount = 5; // Configurable step amount in ms or ticks
  loopStart: number | null = null;
  loopEnd: number | null = null;
  loopState: 'INACTIVE' | 'A_SET' | 'ACTIVE' = 'INACTIVE';

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
        let newTime = customEvent.detail.currentTime;
        if (
          this.loopState === 'ACTIVE' &&
          this.loopEnd !== null &&
          newTime >= this.loopEnd
        ) {
          newTime = this.loopStart ?? 0;
          this.currentTime.set(newTime);
          this.seek.emit(newTime);
          // If paused out of bounds, start playing again upon loop reset
          if (!this.isPlaying()) {
            this.isPlaying.set(true);
            this.play.emit();
          }
        } else {
          this.currentTime.set(newTime);
        }
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

  stepBackward() {
    let newTime = this.currentTime() - this.stepAmount;
    if (newTime < 0) {
      newTime = 0;
    }
    this.currentTime.set(newTime);
    this.seek.emit(newTime);
  }

  stepForward() {
    let newTime = this.currentTime() + this.stepAmount;
    if (newTime > this.duration()) {
      newTime = this.duration();
    }
    this.currentTime.set(newTime);
    this.seek.emit(newTime);
  }

  toggleLoop() {
    if (this.loopState === 'INACTIVE') {
      this.loopState = 'A_SET';
      this.loopStart = this.currentTime();
      this.loopEnd = null;
    } else if (this.loopState === 'A_SET') {
      this.loopState = 'ACTIVE';
      this.loopEnd = this.currentTime();
      if (this.loopEnd < this.loopStart!) {
        // Swap bounds if selected in reverse
        const temp = this.loopStart;
        this.loopStart = this.loopEnd;
        this.loopEnd = temp;
      }
    } else {
      this.loopState = 'INACTIVE';
      this.loopStart = null;
      this.loopEnd = null;
    }
  }

  getLoopClass() {
    switch (this.loopState) {
      case 'INACTIVE':
        return '';
      case 'A_SET':
        return 'loop-a-set';
      case 'ACTIVE':
        return 'loop-active';
      default:
        return '';
    }
  }

  getLoopTitle() {
    switch (this.loopState) {
      case 'INACTIVE':
        return 'Set Loop Start (A)';
      case 'A_SET':
        return 'Set Loop End (B)';
      case 'ACTIVE':
        return 'Clear Loop';
      default:
        return '';
    }
  }

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

  isDraggingStartMarker = false;
  isDraggingEndMarker = false;

  @HostListener('window:pointermove', ['$event'])
  onPointerMove(event: PointerEvent) {
    if (!this.isDraggingStartMarker && !this.isDraggingEndMarker) {
      return;
    }
    const container = document.querySelector(
      '.progress-bar-container',
    ) as HTMLElement;
    if (!container) return;
    const rect = container.getBoundingClientRect();
    const x = Math.max(0, Math.min(event.clientX - rect.left, rect.width));
    const percentage = x / rect.width;
    const newTime = percentage * this.duration();

    if (this.isDraggingStartMarker) {
      const maxTime = this.loopEnd !== null ? this.loopEnd : this.duration();
      this.loopStart = Math.min(newTime, maxTime);
      this.seek.emit(this.loopStart);
      this.currentTime.set(this.loopStart);
    } else if (this.isDraggingEndMarker) {
      const minTime = this.loopStart !== null ? this.loopStart : 0;
      this.loopEnd = Math.max(newTime, minTime);
      this.seek.emit(this.loopEnd);
      this.currentTime.set(this.loopEnd);
    }
  }

  @HostListener('window:pointerup', ['$event'])
  onPointerUpMarker(event: PointerEvent) {
    this.isDraggingStartMarker = false;
    this.isDraggingEndMarker = false;
  }

  onStartMarkerPointerDown(event: PointerEvent) {
    event.stopPropagation();
    event.preventDefault();
    this.isDraggingStartMarker = true;
  }

  onEndMarkerPointerDown(event: PointerEvent) {
    event.stopPropagation();
    event.preventDefault();
    this.isDraggingEndMarker = true;
  }
}
