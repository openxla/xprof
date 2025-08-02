import {CommonModule} from '@angular/common';
import {Component, Input} from '@angular/core';
import {MatCardModule} from '@angular/material/card';
import {type SmartSuggestionReport} from 'org_xprof/frontend/app/common/interfaces/smart_suggestion';
/** A component for displaying smart suggestions. */
@Component({
  selector: 'smart-suggestion-view',
  templateUrl: './smart_suggestion_view.ng.html',
  styleUrls: ['./smart_suggestion_view.scss'],
  standalone: true,
  imports: [CommonModule, MatCardModule],
})
export class SmartSuggestionView {
  @Input() suggestionReport: SmartSuggestionReport|null = null;
  @Input() darkTheme = false;

  title = 'Smart Suggestions';
}
