import {CommonModule} from '@angular/common';
import {Component, HostBinding, Input, OnDestroy} from '@angular/core';
import {MatButtonModule} from '@angular/material/button';
import {MatCardModule} from '@angular/material/card';
import {MatExpansionModule} from '@angular/material/expansion';
import {MatIconModule} from '@angular/material/icon';
import {type SmartSuggestionReport} from 'org_xprof/frontend/app/common/interfaces/smart_suggestion.jsonpb_decls';
import {ReplaySubject} from 'rxjs';

// Declaration for the Google Analytics function.
declare var gtag: Function;

interface ProcessedSuggestion {
  id: number;  // Unique identifier for the suggestion
  title: string;
  description: string;
  recommendationsTitle: string;
  recommendations: string[];
  rawText: string;
}

type FeedbackType = 'up'|'down';

/** A component for displaying smart suggestions. */
@Component({
  selector: 'smart-suggestion-view',
  templateUrl: './smart_suggestion_view.ng.html',
  styleUrls: ['./smart_suggestion_view.scss'],
  standalone: true,
  imports: [
    CommonModule,
    MatCardModule,
    MatIconModule,
    MatButtonModule,
    MatExpansionModule
  ],
})
export class SmartSuggestionView implements OnDestroy {
  @HostBinding('class.dark-theme') @Input() darkTheme = false;
  @Input()
  set suggestionReport(value: SmartSuggestionReport|null) {
    if (value) {
      this.processedSuggestions = value.suggestions.map((suggestion, index) => {
        return {
          id: index,
          ...this.processSuggestionText(suggestion.suggestionText),
        };
      });
      this.feedbackState.clear();
    } else {
      this.processedSuggestions = [];
    }
  }

  title = 'Recommendations';
  processedSuggestions: ProcessedSuggestion[] = [];
  feedbackState = new Map<string, FeedbackType>();

  /** Handles on-destroy Subject, used to unsubscribe. */
  private readonly destroyed = new ReplaySubject<void>(1);

  processSuggestionText(text: string): Omit<ProcessedSuggestion, 'id'> {
    const lines = text.split('\n').filter(line => line.trim() !== '');
    const processed: Omit<ProcessedSuggestion, 'id'> = {
      title: '',
      description: '',
      recommendationsTitle: '',
      recommendations: [],
      rawText: text,
    };

    if (lines.length > 0) processed.title = lines[0];
    if (lines.length > 1) processed.description = lines[1];

    const recommendationsIndex =
        lines.findIndex(line => line.startsWith('Recommendations:'));
    if (recommendationsIndex !== -1) {
      processed.recommendationsTitle = lines[recommendationsIndex];
      processed.recommendations = lines.slice(recommendationsIndex + 1)
                                      .filter(line => line.startsWith('- '))
                                      .map(line => line.substring(2));
    }

    return processed;
  }

  getFeedbackKey(suggestionId: number, recommendationIndex: number): string {
    return `${suggestionId}-${recommendationIndex}`;
  }

  splitRecommendation(item: string): {key: string, value: string}|null {
    const colonIndex = item.indexOf(':');
    if (colonIndex === -1) {
      return null;
    }
    return {
      key: item.substring(0, colonIndex),
      value: item.substring(colonIndex + 1)
    };
  }

  toggleFeedback(
      suggestion: ProcessedSuggestion, recommendationIndex: number,
      feedbackType: 'up'|'down') {
    const key = this.getFeedbackKey(suggestion.id, recommendationIndex);
    const currentFeedback = this.feedbackState.get(key);
    let newFeedback: FeedbackType|null = feedbackType;
    let gaValue = 0;

    if (currentFeedback === feedbackType) {
      // User clicked the same button again, so deselect it
      newFeedback = null;
      this.feedbackState.delete(key);
      gaValue =
          feedbackType === 'up' ? -1 : 1;  // Deselect up: -1, Deselect down: +1
    } else {
      // New selection or switching selection
      this.feedbackState.set(key, feedbackType);
      gaValue =
          feedbackType === 'up' ? 1 : -1;  // Select up: +1, Select down: -1
    }

    if (typeof gtag === 'function') {
      gtag('event', 'recommendation_feedback', {
        'event_category': 'Smart Suggestions',
        'event_label': suggestion.recommendations[recommendationIndex],
        'suggestion_title': suggestion.title,
        'value': gaValue,
        'feedback_type': newFeedback,
      });
    }
  }

  getFeedbackState(suggestionId: number, recommendationIndex: number):
      FeedbackType|null {
    const key = this.getFeedbackKey(suggestionId, recommendationIndex);
    return this.feedbackState.get(key) || null;
  }

  ngOnDestroy() {
    // Unsubscribes all pending subscriptions.
    this.destroyed.next();
    this.destroyed.complete();
  }
}
