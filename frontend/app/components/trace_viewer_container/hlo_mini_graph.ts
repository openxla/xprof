/**
 * Pure layout model and SVG geometry builder for the one-hop HLO mini graph
 * shown in the Trace Viewer v2 event details panel.
 */

/** Structured metadata for an HLO node returned by the `adj_nodes` endpoint. */
export declare interface AdjacentNodeInfo {
  name: string;
  opcode: string;
  shape: string;
}

/** Response payload for `graph_viewer.json?type=adj_nodes`. */
export declare interface AdjacentNodesResponse {
  operand_names: string[];
  consumer_names: string[];
  center_node?: AdjacentNodeInfo;
  operands?: AdjacentNodeInfo[];
  consumers?: AdjacentNodeInfo[];
}

/** Validated and normalized `adj_nodes` metadata for the one-hop mini graph. */
export interface ValidatedMiniGraphAdjacentNodes {
  centerNode: AdjacentNodeInfo;
  operands: AdjacentNodeInfo[];
  consumers: AdjacentNodeInfo[];
}

/** Role of a node in the one-hop HLO mini graph. */
export type HloMiniGraphNodeRole = 'operand' | 'selected' | 'consumer';

/** A positioned node in the one-hop HLO mini graph. */
export interface HloMiniGraphNode {
  /** Canonical HLO op name without leading '%'. */
  name: string;
  /** Display label with leading '%', truncated to fit the node box. */
  displayName: string;
  /** Secondary line shown under the op name (shape or opcode). */
  subtitle: string;
  /** Truncated subtitle that fits the node box. */
  displaySubtitle: string;
  /** Full tooltip text shown on hover. */
  tooltip: string;
  /** Role determining column position and color styling. */
  role: HloMiniGraphNodeRole;
  /** Whether a matching event exists on the loaded timeline. */
  clickable: boolean;
  /** Left X coordinate in SVG viewBox units. */
  x: number;
  /** Top Y coordinate in SVG viewBox units. */
  y: number;
  /** Width of the node box in SVG viewBox units. */
  width: number;
  /** Height of the node box in SVG viewBox units. */
  height: number;
}

/** A directed edge connecting two nodes in the one-hop HLO mini graph. */
export interface HloMiniGraphEdge {
  id: string;
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  role: 'operand' | 'consumer';
  clickable: boolean;
}

/** Complete layout model for the one-hop HLO mini graph viewer. */
export interface HloMiniGraph {
  width: number;
  height: number;
  nodes: HloMiniGraphNode[];
  edges: HloMiniGraphEdge[];
}

/** Input parameters for building the one-hop HLO mini graph. */
export interface HloMiniGraphInput {
  adjacentNodes: ValidatedMiniGraphAdjacentNodes;
  selectedClickable?: boolean;
  clickableOperandNames?: ReadonlySet<string>;
  clickableConsumerNames?: ReadonlySet<string>;
}

const MINI_GRAPH_NODE_WIDTH = 168;
const MINI_GRAPH_NODE_HEIGHT = 38;
const MINI_GRAPH_COL_GAP = 10;
const MINI_GRAPH_ROW_GAP = 8;
const MINI_GRAPH_PAD_X = 2;
const MINI_GRAPH_PAD_Y = 4;
const MAX_NODE_TITLE_CHARS = 24;
const MAX_NODE_SUBTITLE_CHARS = 27;

function stripLeadingPercent(name: string): string {
  const trimmed = name.trim();
  return trimmed.startsWith('%') ? trimmed.slice(1).trim() : trimmed;
}

/**
 * Parses and normalizes a raw `AdjacentNodeInfo` entry, returning `undefined`
 * if `name` or `opcode` is missing/empty or `shape` is not a string.
 */
export function parseAdjacentNodeInfo(
  value: unknown,
): AdjacentNodeInfo | undefined {
  if (typeof value !== 'object' || value === null) {
    return undefined;
  }
  const record = value as Record<string, unknown>;
  if (
    typeof record['name'] !== 'string' ||
    typeof record['opcode'] !== 'string' ||
    typeof record['shape'] !== 'string'
  ) {
    return undefined;
  }
  const name = stripLeadingPercent(record['name']);
  const opcode = record['opcode'].trim();
  if (!name || !opcode) {
    return undefined;
  }
  return {
    name,
    opcode,
    shape: record['shape'].trim(),
  };
}

function parseAdjacentNodeList(
  values: unknown,
): AdjacentNodeInfo[] | undefined {
  if (!Array.isArray(values)) {
    return undefined;
  }
  const result: AdjacentNodeInfo[] = [];
  const seen = new Set<string>();
  for (const item of values) {
    const parsed = parseAdjacentNodeInfo(item);
    if (!parsed) {
      return undefined;
    }
    if (seen.has(parsed.name)) {
      continue;
    }
    seen.add(parsed.name);
    result.push(parsed);
  }
  return result;
}

/**
 * Validates, normalizes, and deduplicates a raw `adj_nodes` response into
 * `ValidatedMiniGraphAdjacentNodes`, returning `undefined` if `center_node`,
 * `operands`, or `consumers` is missing or malformed.
 */
export function parseMiniGraphAdjacentNodes(
  data: unknown,
): ValidatedMiniGraphAdjacentNodes | undefined {
  if (typeof data !== 'object' || data === null) {
    return undefined;
  }
  const record = data as Record<string, unknown>;
  const centerNode = parseAdjacentNodeInfo(record['center_node']);
  if (!centerNode) {
    return undefined;
  }
  const operands = parseAdjacentNodeList(record['operands']);
  if (!operands) {
    return undefined;
  }
  const consumers = parseAdjacentNodeList(record['consumers']);
  if (!consumers) {
    return undefined;
  }
  return {centerNode, operands, consumers};
}

function formatNodeDisplayName(name: string): string {
  const withPercent = `%${name}`;
  if (withPercent.length <= MAX_NODE_TITLE_CHARS) {
    return withPercent;
  }
  return `${withPercent.slice(0, MAX_NODE_TITLE_CHARS - 1)}…`;
}

function formatNodeSubtitle(subtitle: string): string {
  const clean = subtitle.replace(/\s+/g, ' ').trim();
  if (clean.length <= MAX_NODE_SUBTITLE_CHARS) {
    return clean;
  }
  return `${clean.slice(0, MAX_NODE_SUBTITLE_CHARS - 1)}…`;
}

function createMiniGraphNode(
  entry: AdjacentNodeInfo,
  role: HloMiniGraphNodeRole,
  clickable: boolean,
  x: number,
  y: number,
): HloMiniGraphNode {
  const subtitle = entry.shape || entry.opcode;
  const tooltipDetail = entry.shape
    ? `${entry.opcode}, ${entry.shape}`
    : entry.opcode;
  let tooltip = `%${entry.name} (${tooltipDetail})`;
  if (!clickable) {
    tooltip = `${tooltip} — No matching event in loaded timeline`;
  } else if (role !== 'selected') {
    tooltip = `${tooltip} — Click to focus on ${role} event`;
  }
  return {
    name: entry.name,
    displayName: formatNodeDisplayName(entry.name),
    subtitle,
    displaySubtitle: formatNodeSubtitle(subtitle),
    tooltip,
    role,
    clickable,
    x,
    y,
    width: MINI_GRAPH_NODE_WIDTH,
    height: MINI_GRAPH_NODE_HEIGHT,
  };
}

/**
 * Builds the 3-column (operands -> selected op -> consumers) SVG layout for
 * the one-hop HLO mini graph from validated `adjacentNodes` metadata.
 */
export function buildHloMiniGraph(input: HloMiniGraphInput): HloMiniGraph {
  const {
    centerNode: centerEntry,
    operands: operandEntries,
    consumers: consumerEntries,
  } = input.adjacentNodes;

  const maxRows = Math.max(1, operandEntries.length, consumerEntries.length);
  const contentHeight =
    maxRows * MINI_GRAPH_NODE_HEIGHT + (maxRows - 1) * MINI_GRAPH_ROW_GAP;
  const height = contentHeight + 2 * MINI_GRAPH_PAD_Y;

  const col0X = MINI_GRAPH_PAD_X;
  const col1X = MINI_GRAPH_PAD_X + MINI_GRAPH_NODE_WIDTH + MINI_GRAPH_COL_GAP;
  const col2X =
    MINI_GRAPH_PAD_X + 2 * (MINI_GRAPH_NODE_WIDTH + MINI_GRAPH_COL_GAP);
  const width = col2X + MINI_GRAPH_NODE_WIDTH + MINI_GRAPH_PAD_X;

  const columnStartY = (count: number): number => {
    const colHeight =
      count * MINI_GRAPH_NODE_HEIGHT +
      Math.max(0, count - 1) * MINI_GRAPH_ROW_GAP;
    return MINI_GRAPH_PAD_Y + (contentHeight - colHeight) / 2;
  };

  const clickableOperands = input.clickableOperandNames
    ? new Set(
        Array.from(input.clickableOperandNames, stripLeadingPercent).filter(
          (name) => name.length > 0,
        ),
      )
    : undefined;
  const clickableConsumers = input.clickableConsumerNames
    ? new Set(
        Array.from(input.clickableConsumerNames, stripLeadingPercent).filter(
          (name) => name.length > 0,
        ),
      )
    : undefined;
  const selectedClickable = input.selectedClickable ?? true;

  const nodes: HloMiniGraphNode[] = [];
  const operandNodes: HloMiniGraphNode[] = [];
  const consumerNodes: HloMiniGraphNode[] = [];

  const operandStartY = columnStartY(operandEntries.length);
  for (let i = 0; i < operandEntries.length; i++) {
    const entry = operandEntries[i];
    const clickable = clickableOperands
      ? clickableOperands.has(entry.name)
      : true;
    const node = createMiniGraphNode(
      entry,
      'operand',
      clickable,
      col0X,
      operandStartY + i * (MINI_GRAPH_NODE_HEIGHT + MINI_GRAPH_ROW_GAP),
    );
    nodes.push(node);
    operandNodes.push(node);
  }

  const selectedNode = createMiniGraphNode(
    centerEntry,
    'selected',
    selectedClickable,
    col1X,
    columnStartY(1),
  );
  nodes.push(selectedNode);

  const consumerStartY = columnStartY(consumerEntries.length);
  for (let i = 0; i < consumerEntries.length; i++) {
    const entry = consumerEntries[i];
    const clickable = clickableConsumers
      ? clickableConsumers.has(entry.name)
      : true;
    const node = createMiniGraphNode(
      entry,
      'consumer',
      clickable,
      col2X,
      consumerStartY + i * (MINI_GRAPH_NODE_HEIGHT + MINI_GRAPH_ROW_GAP),
    );
    nodes.push(node);
    consumerNodes.push(node);
  }

  const edges: HloMiniGraphEdge[] = [];
  for (const opNode of operandNodes) {
    edges.push({
      id: `op:${opNode.name}->${selectedNode.name}`,
      x1: opNode.x + opNode.width,
      y1: opNode.y + opNode.height / 2,
      x2: selectedNode.x,
      y2: selectedNode.y + selectedNode.height / 2,
      role: 'operand',
      clickable: opNode.clickable && selectedNode.clickable,
    });
  }
  for (const consNode of consumerNodes) {
    edges.push({
      id: `cons:${selectedNode.name}->${consNode.name}`,
      x1: selectedNode.x + selectedNode.width,
      y1: selectedNode.y + selectedNode.height / 2,
      x2: consNode.x,
      y2: consNode.y + consNode.height / 2,
      role: 'consumer',
      clickable: consNode.clickable && selectedNode.clickable,
    });
  }

  return {width, height, nodes, edges};
}
