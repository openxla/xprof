/**
 * @fileoverview Interfaces for HLO Op Profile diffing and comparison.
 */

import {
  Metrics,
  Node,
  Profile,
} from 'org_xprof/frontend/app/common/interfaces/op_profile.jsonpb_decls';

/**
 * Represents a node in an HLO op profile tree that has been merged with a
 * baseline profile node for comparison.
 */
export interface DiffNode extends Node {
  /**
   * Recursive children of the merged diff tree.
   */
  children?: DiffNode[];
  /**
   * The corresponding matched node from the baseline profile, if present.
   */
  baseline?: Node;

  /**
   * Indicates whether this node appears only in the baseline profile (not found
   * in the active profile).
   */
  baselineOnly: boolean;

  /**
   * Indicates whether this node appears only in the active profile (not found
   * in the baseline profile).
   */
  activeOnly: boolean;

  /**
   * The differences between the active node's metrics and the baseline node's
   * metrics (typically `active - baseline`).
   */
  diffMetrics?: Metrics;
}

/**
 * Represents a merged HLO op profile containing diff trees across the various
 * grouping hierarchies (`byCategory`, `byProgram`, `byProvenance`, etc.).
 */
export interface DiffProfile extends Profile {
  /** Merged diff tree grouped by HLO operation category. */
  byCategory?: DiffNode;

  /** Merged diff tree grouped by HLO program/fusion. */
  byProgram?: DiffNode;

  /** Merged diff tree grouped by HLO provenance (source line/code or trace). */
  byProvenance?: DiffNode;

  /** Merged diff tree grouped by category, excluding idle time. */
  byCategoryExcludeIdle?: DiffNode;

  /** Merged diff tree grouped by program, excluding idle time. */
  byProgramExcludeIdle?: DiffNode;

  /** Merged diff tree grouped by provenance, excluding idle time. */
  byProvenanceExcludeIdle?: DiffNode;
}

/** Alias for a merged diff profile comparison. */
export type OpProfileDiff = DiffProfile;
