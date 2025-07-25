/**
 * @fileoverview Interfaces for Smart Suggestion data.
 * These interfaces mirror the structure of the SmartSuggestion and
 * SmartSuggestionReport protos.
 */

/**
 * Interface for a single smart suggestion.
 * Corresponds to tensorflow.profiler.SmartSuggestion
 */
export declare interface SmartSuggestion {
  ruleName: string;
  suggestionText: string;
}

/**
 * Interface for the report containing multiple smart suggestions.
 * Corresponds to tensorflow.profiler.SmartSuggestionReport
 */
export declare interface SmartSuggestionReport {
  suggestions: SmartSuggestion[];
}
