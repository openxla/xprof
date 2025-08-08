import {CommonModule} from '@angular/common';
import {Component, HostBinding, Input, OnChanges, OnInit, SimpleChanges, inject, OnDestroy} from '@angular/core';
import {MatButtonModule} from '@angular/material/button';
import {MatCardModule} from '@angular/material/card';
import {MatExpansionModule} from '@angular/material/expansion';
import {MatIconModule} from '@angular/material/icon';
import {type SmartSuggestionReport} from 'org_xprof/frontend/app/common/interfaces/smart_suggestion.jsonpb_decls';
import {DATA_SERVICE_INTERFACE_TOKEN, type DataServiceV2Interface} from 'org_xprof/frontend/app/services/data_service_v2/data_service_v2_interface';
import {Subscription} from 'rxjs';

// Declaration for the Google Analytics function.
declare var gtag: Function;

interface ProcessedSuggestion {
  id: string;  // Unique identifier for the suggestion
  ruleName: string;
  htmlContent: string;
}

type FeedbackType = 'up'|'down';

// Structure for storing feedback state for a session.
// Use 'declare' to prevent property renaming by JSCompiler when using localStorage.
declare interface SessionFeedbackState {
  [suggestionKey: string]: FeedbackType;
}

const FEEDBACK_STORAGE_KEY_PREFIX = 'smartSuggestionFeedback';

/** A component for displaying smart suggestions. */
@Component({
  selector: 'smart-suggestion-view',
  templateUrl: './smart_suggestion_view.ng.html',
  styleUrls: ['./smart_suggestion_view.scss'],
  standalone: true,
  imports: [
    CommonModule,
    MatButtonModule,
    MatCardModule,
    MatExpansionModule,
    MatIconModule,
  ],
})
export class SmartSuggestionView implements OnInit, OnChanges, OnDestroy {
  @HostBinding('class.dark-theme') @Input() darkTheme = false;
  @Input() sessionId = 'default_session';

  title = 'Recommendations';
  processedSuggestions: ProcessedSuggestion[] = [];
  feedbackState = new Map<string, FeedbackType>();
  private storageKey = '';
  private subscription: Subscription | null = null;

  private readonly dataService: DataServiceV2Interface = inject(DATA_SERVICE_INTERFACE_TOKEN);

  ngOnInit() {
    this.updateStorageKey();
    this.loadFeedbackState();
    this.fetchSuggestions();
  }

  ngOnChanges(changes: SimpleChanges) {
    if (changes['sessionId']) {
      this.updateStorageKey();
      this.loadFeedbackState();
      this.fetchSuggestions();
    }
  }

  private fetchSuggestions() {
    if (this.subscription) {
      this.subscription.unsubscribe();
    }
    if (!this.sessionId) {
      this.processedSuggestions = [];
      return;
    }

    this.subscription = this.dataService
    .getSmartSuggestions(this.sessionId)
    .subscribe((report: SmartSuggestionReport | null) => {
      if (report && report.suggestions) {
        this.processedSuggestions = report.suggestions.map((suggestion) => {
          const content = suggestion.suggestionText || '';
          return {
            id: this.generateSuggestionKey(
              suggestion.ruleName || 'UnknownRule',
              content,
            ),
            ruleName: suggestion.ruleName || 'UnknownRule',
            htmlContent: content,
          };
        });
        // Reload feedback state in case new suggestions appeared
        this.loadFeedbackState();
      } else {
        this.processedSuggestions = [];
      }
    });
}

  private updateStorageKey() {
    this.storageKey = `${FEEDBACK_STORAGE_KEY_PREFIX}_${this.sessionId}`;
  }

  private hashString(str: string): number {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
      const char = str.charCodeAt(i);
      hash = (hash << 5) - hash + char;
      hash |= 0; // Convert to 32bit integer
    }
    return Math.abs(hash);
  }

  private generateSuggestionKey(ruleName: string, content: string): string {
    const contentHash = this.hashString(content);
    return `${ruleName}-${contentHash}`;
  }

  private loadFeedbackState() {
    if (typeof window === 'undefined' || !window.localStorage) return;
    const storedState = window.localStorage.getItem(this.storageKey);
    if (storedState) {
      try {
        const parsedState = JSON.parse(storedState) as SessionFeedbackState;
        this.feedbackState = new Map(Object.entries(parsedState));
      } catch (e) {
        console.error('Error loading feedback state from localStorage', e);
        this.feedbackState = new Map();
      }
    } else {
      this.feedbackState = new Map();
    }
  }

  private saveFeedbackState() {
    if (typeof window === 'undefined' || !window.localStorage) return;
    try {
      const stateToStore: SessionFeedbackState = {};
      this.feedbackState.forEach((value, key) => {
        stateToStore[key] = value;
      });
      window.localStorage.setItem(this.storageKey, JSON.stringify(stateToStore));
    } catch (e) {
      console.error('Error saving feedback state to localStorage', e);
    }
  }

  toggleFeedback(suggestionId: string, ruleName: string, feedbackType: FeedbackType) {
    const key = suggestionId;
    const currentFeedbackState = this.feedbackState.get(key);

    let oldNumericValue = 0;
    if (currentFeedbackState === 'up') {
      oldNumericValue = 1;
    } else if (currentFeedbackState === 'down') {
      oldNumericValue = -1;
    }

    let newNumericValue = 0;
    let finalFeedbackState: FeedbackType | null = null;

    if (currentFeedbackState === feedbackType) {
      // DESELECT: Clicked the same button
      this.feedbackState.delete(key);
      // newNumericValue remains 0
      finalFeedbackState = null;
    } else {
      // SELECT or SWITCH: Clicked a different button
      this.feedbackState.set(key, feedbackType);
      newNumericValue = (feedbackType === 'up' ? 1 : -1);
      finalFeedbackState = feedbackType;
    }

    // Calculate the change in score
    const gaValue = newNumericValue - oldNumericValue;

    this.saveFeedbackState();

    if (typeof gtag === 'function' && gaValue !== 0) {
      gtag('event', 'recommendation_feedback', {
        'event_category': 'Smart Suggestions',
        'event_label': ruleName, // Key to group by
        'session_id': this.sessionId,
        'event_value': gaValue, // The change in score to be summed by GA
        'feedback_action': this.getFeedbackAction(currentFeedbackState, finalFeedbackState),
      });
    }
  }

  private getFeedbackAction(
    oldState: FeedbackType | undefined,
    newState: FeedbackType | null
  ): string {
    const oldS = oldState || 'none';
    const newS = newState || 'none';
    return `${oldS}_to_${newS}`;
  }

  getFeedbackState(suggestionId: string): FeedbackType | null {
    return this.feedbackState.get(suggestionId) || null;
  }

  ngOnDestroy() {
    if (this.subscription) {
      this.subscription.unsubscribe();
    }
  }
}
