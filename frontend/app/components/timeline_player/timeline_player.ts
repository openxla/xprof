import {CommonModule} from '@angular/common';
import {
  ChangeDetectionStrategy,
  ChangeDetectorRef,
  Component,
  ElementRef,
  HostListener,
  NgZone,
  OnDestroy,
  OnInit,
  ViewChild,
  computed,
  model,
  output,
  signal,
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
  time?: number;
  currentTime?: number;
  visibleStart?: number;
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
  @ViewChild('progressBarContainer')
  progressBarContainer?: ElementRef<HTMLElement>;

  readonly currentTime = model(0);
  readonly duration = model(100);
  readonly visibleStart = signal(0);
  readonly isPlaying = model(false);
  readonly playbackRate = model(1);
  readonly isFinished = computed(
    () => this.duration() > 0 && this.currentTime() >= this.duration(),
  );

  readonly play = output<void>();
  readonly pause = output<void>();
  readonly seek = output<number>();
  readonly speedChange = output<number>();

  activeEvents: Array<Record<string, string | number | boolean>> = [];
  activeCounters: Array<Record<string, string | number | boolean>> = [];

  readonly stepAmount = computed(() => this.duration() * 0.05);
  // Loop bounds are stored as absolute trace timestamps (microseconds).
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

  private getRelativePercent(absTime: number | null): number | null {
    const dur = this.duration();
    return absTime === null || dur <= 0
      ? null
      : ((absTime - this.visibleStart()) / dur) * 100;
  }

  get isLoopStartVisible(): boolean {
    const pct = this.getRelativePercent(this.loopStart);
    return (
      this.loopState !== 'INACTIVE' && pct !== null && pct >= 0 && pct <= 100
    );
  }

  get isLoopEndVisible(): boolean {
    const pct = this.getRelativePercent(this.loopEnd);
    return (
      this.loopState === 'ACTIVE' && pct !== null && pct >= 0 && pct <= 100
    );
  }

  get loopStartPercent(): number {
    const pct = this.getRelativePercent(this.loopStart);
    return pct === null ? 0 : Math.max(0, Math.min(100, pct));
  }

  get loopEndPercent(): number {
    const pct = this.getRelativePercent(this.loopEnd);
    return pct === null ? 0 : Math.max(0, Math.min(100, pct));
  }

  get isLoopRegionVisible(): boolean {
    const dur = this.duration();
    if (
      this.loopState !== 'ACTIVE' ||
      this.loopStart === null ||
      this.loopEnd === null ||
      dur <= 0
    ) {
      return false;
    }
    const visStart = this.visibleStart();
    return this.loopEnd >= visStart && this.loopStart <= visStart + dur;
  }

  get loopRegionLeftPercent(): number {
    return this.loopStartPercent;
  }

  get loopRegionWidthPercent(): number {
    return Math.max(0, this.loopEndPercent - this.loopStartPercent);
  }

  get progressFillPercent(): number {
    const dur = this.duration();
    return dur <= 0
      ? 0
      : Math.max(0, Math.min(100, (this.currentTime() / dur) * 100));
  }

  private readonly handleBackendSync = (event: Event) => {
    const customEvent = event as CustomEvent<SyncEventDetail>;
    if (customEvent.detail) {
      let dirty = false;
      let loopWrapped = false;
      if (customEvent.detail.events) {
        this.activeEvents = customEvent.detail.events;
        dirty = true;
      }
      if (customEvent.detail.counters) {
        this.activeCounters = customEvent.detail.counters;
        dirty = true;
      }
      if (customEvent.detail.visibleStart !== undefined) {
        this.visibleStart.set(customEvent.detail.visibleStart);
        dirty = true;
      }
      if (customEvent.detail.duration !== undefined) {
        this.duration.set(customEvent.detail.duration);
        dirty = true;
      }
      if (customEvent.detail.currentTime !== undefined && !this.isScrubbing) {
        const newTime = customEvent.detail.currentTime;
        const visStart = this.visibleStart();
        const visEnd = visStart + this.duration();
        const absNewTime = visStart + newTime;
        const loopStart = this.loopStart;
        const loopEnd = this.loopEnd;
        const isDraggingMarker =
          this.isDraggingStartMarker || this.isDraggingEndMarker;
        if (
          this.loopState === 'ACTIVE' &&
          this.isPlaying() &&
          !isDraggingMarker &&
          loopStart !== null &&
          loopEnd !== null &&
          loopEnd > loopStart &&
          loopStart >= visStart &&
          loopEnd <= visEnd &&
          absNewTime >= loopEnd
        ) {
          const loopStartRelative = loopStart - visStart;
          this.currentTime.set(loopStartRelative);
          this.seek.emit(loopStartRelative);
          loopWrapped = true;
        } else {
          this.currentTime.set(newTime);
        }
        dirty = true;
      }
      if (customEvent.detail.isPlaying !== undefined && !loopWrapped) {
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
    let newTime = this.currentTime() - this.stepAmount();
    if (newTime < 0) {
      newTime = 0;
    }
    this.currentTime.set(newTime);
    this.seek.emit(newTime);
  }

  stepForward() {
    let newTime = this.currentTime() + this.stepAmount();
    if (newTime > this.duration()) {
      newTime = this.duration();
    }
    this.currentTime.set(newTime);
    this.seek.emit(newTime);
  }

  toggleLoop() {
    const currentAbsTime = this.visibleStart() + this.currentTime();
    if (this.loopState === 'INACTIVE') {
      this.loopState = 'A_SET';
      this.loopStart = currentAbsTime;
      this.loopEnd = null;
    } else if (this.loopState === 'A_SET') {
      this.loopState = 'ACTIVE';
      this.loopEnd = currentAbsTime;
      if (this.loopStart !== null && this.loopEnd < this.loopStart) {
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
    this.cdr.markForCheck();
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
    if (this.isFinished()) {
      this.currentTime.set(0);
      this.seek.emit(0);
      this.isPlaying.set(true);
      this.play.emit();
      return;
    }

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
    const container = this.progressBarContainer?.nativeElement;
    if (!container) return;
    const rect = container.getBoundingClientRect();
    if (rect.width <= 0) {
      return;
    }
    const x = Math.max(0, Math.min(event.clientX - rect.left, rect.width));
    const percentage = x / rect.width;
    const visStart = this.visibleStart();
    const dur = this.duration();
    const newAbsTime = visStart + percentage * dur;

    if (this.isDraggingStartMarker) {
      const maxAbsTime = this.loopEnd !== null ? this.loopEnd : visStart + dur;
      this.loopStart = Math.min(newAbsTime, maxAbsTime);
      const seekRelative = Math.max(
        0,
        Math.min(this.loopStart - visStart, dur),
      );
      this.currentTime.set(seekRelative);
      this.seek.emit(seekRelative);
    } else if (this.isDraggingEndMarker) {
      const minAbsTime = this.loopStart !== null ? this.loopStart : visStart;
      this.loopEnd = Math.max(newAbsTime, minAbsTime);
      const seekRelative = Math.max(0, Math.min(this.loopEnd - visStart, dur));
      this.currentTime.set(seekRelative);
      this.seek.emit(seekRelative);
    }
  }

  @HostListener('window:pointerup')
  @HostListener('window:pointercancel')
  @HostListener('window:blur')
  onInteractionCancel() {
    this.isScrubbing = false;
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
