/**
 * Interface representing an individual experiment or curated tool in XProf Labs.
 */
export declare interface CuratedTool {
  readonly key: string;
  readonly label: string;
  readonly status: 'Alpha' | 'Beta';
  readonly statusText?: string;
  readonly description: string;
  readonly icon: string;
  readonly url?: string;
  readonly route?: string;
  readonly tags: readonly string[];
  readonly isFeatured: boolean;
  readonly owner?: string;
  readonly buganizer?: string;
  readonly lifecycle?: string;
  readonly mau?: number;
  readonly colorClass?: string;
  readonly isFavorite?: boolean;
  readonly lifecycleClass?: string;
  readonly statusTextClass?: string;
}
