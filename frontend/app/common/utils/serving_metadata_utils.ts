/**
 * @license
 * Copyright 2026 The OpenXLA Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * @fileoverview Helpers to read and format the serving metadata of Op Profile
 * program nodes.
 */

import {type DiffNode} from 'org_xprof/frontend/app/common/interfaces/op_profile_diff';
import {type Node} from 'org_xprof/frontend/app/common/interfaces/op_profile.jsonpb_decls';

/** A (model, batch size) pair with normalized, non-negative values. */
export interface ServingPair {
  /** Trimmed model id; empty if unknown. */
  modelId: string;
  /** Padded batch size; 0 if unknown. */
  batchSize: number;
  /** Number of batches that ran with this model and batch size. */
  numBatches: number;
}

/**
 * Returns the serving metadata of a node. A baseline-only diff node uses the
 * metadata of its baseline node; a node that exists in the active profile
 * never borrows the baseline's metadata.
 */
export function getServingMetadata(
  node?: Node | DiffNode,
): Node.ServingMetadata | undefined {
  if (!node) {
    return undefined;
  }
  if (node.programServingMetadata) {
    return node.programServingMetadata;
  }
  if ('baselineOnly' in node && node.baselineOnly) {
    return node.baseline?.programServingMetadata;
  }
  return undefined;
}

/**
 * Returns the pairs that have a model id or a positive batch size, with
 * missing or negative values normalized.
 */
export function getServingPairs(
  servingMetadata?: Node.ServingMetadata,
): ServingPair[] {
  const pairs: ServingPair[] = [];
  for (const pair of servingMetadata?.modelBatches ?? []) {
    const modelId = (pair.modelId ?? '').trim();
    const batchSize = Math.max(pair.batchSize ?? 0, 0);
    if (modelId === '' && batchSize === 0) {
      continue;
    }
    pairs.push({
      modelId,
      batchSize,
      numBatches: Math.max(pair.numBatches ?? 0, 0),
    });
  }
  return pairs;
}

/** Formats a batch count, e.g. "1 batch" or "3 batches". */
export function formatBatchCount(numBatches: number): string {
  return numBatches === 1 ? '1 batch' : `${numBatches} batches`;
}

/** Returns the number of batches dropped by the server-side pair cap. */
export function getOmittedServingBatches(
  servingMetadata?: Node.ServingMetadata,
): number {
  return Math.max(servingMetadata?.numOmittedBatches ?? 0, 0);
}

/**
 * Describes the pairs dropped by the server-side cap, e.g.
 * "+2 other pairs (15 batches)". Returns '' if no pair was dropped.
 */
export function formatOmittedServingPairs(
  servingMetadata?: Node.ServingMetadata,
): string {
  const numOmittedPairs = Math.max(
    servingMetadata?.numOmittedModelBatches ?? 0,
    0,
  );
  if (numOmittedPairs === 0) {
    return '';
  }
  const pairWord = numOmittedPairs === 1 ? 'pair' : 'pairs';
  return `+${numOmittedPairs} other ${pairWord} (${formatBatchCount(
    getOmittedServingBatches(servingMetadata),
  )})`;
}
