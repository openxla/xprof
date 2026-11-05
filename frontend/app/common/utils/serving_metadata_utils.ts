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
 * @fileoverview Helpers to read, validate, and format the serving metadata of
 * Op Profile program nodes.
 */

import {type DiffNode} from 'org_xprof/frontend/app/common/interfaces/op_profile_diff';
import {type Node} from 'org_xprof/frontend/app/common/interfaces/op_profile.jsonpb_decls';

/**
 * The `ProgramServingMetadata` fields shown in the UI, after validation:
 * `modelId` is non-empty and trimmed, and `batchSize` is non-negative (`0` for
 * serving sessions without batching).
 */
export type NormalizedServingMetadata = Readonly<
  Required<Pick<Node.ProgramServingMetadata, 'modelId' | 'batchSize'>>
>;

/**
 * Returns the serving metadata to show for a node.
 *
 * - Normal profile: reads the node's own `programServingMetadata`.
 * - Diff view: a node that exists in the active profile also reads its own
 *   metadata and never borrows the baseline's. Only a baseline-only node (a
 *   program removed from the active profile) reads it from `node.baseline`.
 */
export function getServingMetadata(
  node?: Node | DiffNode,
): Node.ProgramServingMetadata | undefined {
  if (!node) {
    return undefined;
  }
  if (node.programServingMetadata) {
    return node.programServingMetadata;
  }
  const diffNode = node as Partial<DiffNode>;
  if (diffNode.baselineOnly) {
    return diffNode.baseline?.programServingMetadata;
  }
  return undefined;
}

/**
 * Shortens a full model id to its last path segment, keeping a trailing numeric
 * version, e.g. "/models/ranking/m:12" -> "m:12" and "/models/m/7" -> "m:7".
 */
export function shortenModelId(modelId: string): string {
  const segments = modelId.trim().split('/').filter(Boolean);
  if (segments.length === 0) {
    return '';
  }
  const last = segments[segments.length - 1];
  if (/^\d+$/.test(last) && segments.length >= 2) {
    return `${segments[segments.length - 2]}:${last}`;
  }
  return last;
}

/**
 * Validates and normalizes `ProgramServingMetadata`. Returns `undefined` unless
 * both a non-empty `modelId` and a non-negative `batchSize` (`>= 0`) are
 * present.
 */
export function normalizeServingMetadata(
  servingMetadata?: Node.ProgramServingMetadata,
): NormalizedServingMetadata | undefined {
  if (!servingMetadata) {
    return undefined;
  }
  const modelId = (servingMetadata.modelId ?? '').trim();
  const rawBatchSize = servingMetadata.batchSize ?? 0;
  if (
    modelId === '' ||
    shortenModelId(modelId) === '' ||
    !(rawBatchSize >= 0) ||
    !Number.isFinite(rawBatchSize)
  ) {
    return undefined;
  }
  const batchSize = rawBatchSize === 0 ? 0 : rawBatchSize;
  return {modelId, batchSize};
}

/**
 * Formats the compact row label for a program node, e.g.
 * "BS 8 • Model: recommendation_model:1786007405", or
 * "Model: recommendation_model:1786007405" when batching is not used.
 */
export function formatServingBadge(
  metadata: NormalizedServingMetadata,
): string {
  const shortModel = shortenModelId(metadata.modelId);
  if (metadata.batchSize > 0) {
    return `BS ${metadata.batchSize} • Model: ${shortModel}`;
  }
  return `Model: ${shortModel}`;
}

/**
 * Formats the single-line hover and screen-reader description for a program
 * node, e.g. "Model: /models/ranking/m:1 · Batch size: 8", or
 * "Model: /models/ranking/m:1" when batching is not used.
 */
export function formatServingTooltip(
  metadata: NormalizedServingMetadata,
): string {
  if (metadata.batchSize > 0) {
    return `Model: ${metadata.modelId} · Batch size: ${metadata.batchSize}`;
  }
  return `Model: ${metadata.modelId}`;
}
