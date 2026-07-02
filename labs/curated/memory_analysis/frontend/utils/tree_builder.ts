import type {MemoryAnalysisBuffer} from 'org_xprof/frontend/app/common/interfaces/memory_analysis';
import type {TreeNode} from 'org_xprof/frontend/app/common/interfaces/memory_analysis_view';

/**
 * Builds a hierarchical TreeNode tree from a flat list of buffers.
 * Intermediate scope nodes holding physical memory are automatically split into '[self]' sub-leaves.
 * Duplicate buffers on identical paths accumulate values.
 * @param buffers Flat array of preprocessed memory buffers.
 * @param hierarchyType The structure mode to map ('jax' or 'category').
 * @param metric The size attribute to parse and aggregate ('total' or 'padding').
 * @returns The root TreeNode of the constructed tree.
 */
export function buildTree(
  buffers: MemoryAnalysisBuffer[],
  hierarchyType: 'jax' | 'category',
  metric: 'total' | 'padding',
): TreeNode {
  const root: TreeNode = {
    name: 'root',
    value: 0,
    totalValue: 0,
    paddingValue: 0,
    children: [],
    path: '',
    isLeaf: false,
    depth: 0,
  };

  for (const buffer of buffers) {
    const paths = getPathSegments(buffer, hierarchyType);
    insertBuffer(root, buffer, paths, 0, metric);
  }

  aggregateValues(root, metric);
  sortTree(root);
  return root;
}

function getPathSegments(
  buffer: MemoryAnalysisBuffer,
  hierarchyType: 'jax' | 'category',
): string[] {
  if (hierarchyType === 'jax') {
    const jaxPath = buffer.jaxVariablePath || '';
    if (!jaxPath) {
      return [
        'Others',
        buffer.category || 'Uncategorized',
        buffer.subCategory || 'Others',
        buffer.name,
      ];
    }
    const segments = jaxPath
      .replace(/^\/|\/$/g, '')
      .split('/')
      .filter(Boolean);
    segments.push(buffer.name);
    return segments;
  } else {
    return [
      buffer.category || 'Uncategorized',
      buffer.subCategory || 'Others',
      buffer.group || 'General',
      buffer.name,
    ].filter(Boolean);
  }
}

function insertBuffer(
  parent: TreeNode,
  buffer: MemoryAnalysisBuffer,
  segments: string[],
  depth: number,
  metric: 'total' | 'padding',
) {
  if (depth === segments.length) return;

  const segmentName = segments[depth];
  const isLast = depth === segments.length - 1;
  const currentPath = parent.path
    ? `${parent.path}/${segmentName}`
    : segmentName;

  let child = parent.children.find((c: TreeNode) => c.name === segmentName);
  if (!child) {
    child = {
      name: segmentName,
      value: 0,
      totalValue: 0,
      paddingValue: 0,
      children: [],
      path: currentPath,
      isLeaf: isLast,
      depth: depth + 1,
    };
    if (isLast) {
      child.buffer = buffer;
      child.totalValue = buffer.sizeMib;
      child.paddingValue = buffer.paddingMib;
      child.value = metric === 'total' ? buffer.sizeMib : buffer.paddingMib;
    }
    parent.children.push(child);
  } else {
    if (isLast) {
      if (child.isLeaf) {
        child.totalValue += buffer.sizeMib;
        child.paddingValue += buffer.paddingMib;
        child.value += metric === 'total' ? buffer.sizeMib : buffer.paddingMib;
      } else {
        insertSelfNode(child, buffer, metric);
      }
    } else {
      if (child.isLeaf) {
        child.isLeaf = false;
        if (child.buffer) {
          splitLeafToSelf(child, metric);
          child.buffer = undefined;
        }
      }
    }
  }

  if (!isLast && child) {
    insertBuffer(child, buffer, segments, depth + 1, metric);
  }
}

function splitLeafToSelf(parent: TreeNode, metric: 'total' | 'padding') {
  const selfName = `${parent.name} [self]`;
  const selfPath = `${parent.path}/${selfName}`;
  const selfNode: TreeNode = {
    name: selfName,
    value: parent.value,
    totalValue: parent.totalValue,
    paddingValue: parent.paddingValue,
    children: [],
    buffer: parent.buffer,
    path: selfPath,
    isLeaf: true,
    depth: parent.depth + 1,
  };
  parent.children.push(selfNode);
}

function insertSelfNode(
  parent: TreeNode,
  buffer: MemoryAnalysisBuffer,
  metric: 'total' | 'padding',
) {
  const selfName = `${parent.name} [self]`;
  const selfPath = `${parent.path}/${selfName}`;
  let selfNode = parent.children.find((c: TreeNode) => c.name === selfName);
  if (!selfNode) {
    selfNode = {
      name: selfName,
      value: metric === 'total' ? buffer.sizeMib : buffer.paddingMib,
      totalValue: buffer.sizeMib,
      paddingValue: buffer.paddingMib,
      children: [],
      buffer, // shorthand property resolved (Issue 3.2)
      path: selfPath,
      isLeaf: true,
      depth: parent.depth + 1,
    };
    parent.children.push(selfNode);
  } else {
    selfNode.totalValue += buffer.sizeMib;
    selfNode.paddingValue += buffer.paddingMib;
    selfNode.value += metric === 'total' ? buffer.sizeMib : buffer.paddingMib;
  }
}

function aggregateValues(
  node: TreeNode,
  metric: 'total' | 'padding',
): {total: number; padding: number} {
  let total = node.buffer ? node.totalValue : 0;
  let padding = node.buffer ? node.paddingValue : 0;

  for (const child of node.children) {
    const res = aggregateValues(child, metric);
    total += res.total;
    padding += res.padding;
  }

  node.totalValue = total;
  node.paddingValue = padding;
  node.value = metric === 'total' ? total : padding;
  return {total, padding};
}

function sortTree(node: TreeNode) {
  if (node.children.length === 0) return;
  node.children.sort((a: TreeNode, b: TreeNode) => b.value - a.value);
  for (const child of node.children) {
    sortTree(child);
  }
}
