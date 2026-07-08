import type {MemoryAnalysisBuffer} from 'org_xprof/frontend/app/common/interfaces/memory_analysis';
import type {TreeNode} from 'org_xprof/frontend/app/common/interfaces/memory_analysis_view';

/**
 * Builds a hierarchical TreeNode tree from a flat list of buffers.
 * Intermediate scope nodes holding physical memory are automatically split into '[self]' sub-leaves.
 * Duplicate buffers on identical paths accumulate values.
 */
export function buildTree(options: {
  readonly buffers: Readonly<MemoryAnalysisBuffer[]>;
  readonly hierarchyType: 'jax' | 'category';
  readonly metric: 'total' | 'padding';
}): TreeNode {
  const {buffers, hierarchyType, metric} = options;
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
    const paths = getPathSegments({buffer, hierarchyType});
    insertBuffer({parent: root, buffer, segments: paths, depth: 0, metric});
  }

  aggregateValues({node: root, metric});
  sortTree(root);
  return root;
}

/**
 * Extracts and normalizes path segments for a buffer based on the selected hierarchy type.
 */
export function getPathSegments(options: {
  readonly buffer: Readonly<MemoryAnalysisBuffer>;
  readonly hierarchyType: 'jax' | 'category';
}): string[] {
  const {buffer, hierarchyType} = options;
  if (hierarchyType === 'jax') {
    const jaxPath = buffer.jaxVariablePath ?? '';
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

function insertBuffer(options: {
  readonly parent: TreeNode;
  readonly buffer: Readonly<MemoryAnalysisBuffer>;
  readonly segments: Readonly<string[]>;
  readonly depth: number;
  readonly metric: 'total' | 'padding';
}): void {
  const {parent, buffer, segments, depth, metric} = options;
  if (depth === segments.length) return;

  const segmentName = segments[depth];
  const isLast = depth === segments.length - 1;
  const currentPath = parent.path
    ? `${parent.path}/${segmentName}`
    : segmentName;

  const existingChild = parent.children.find(
    (childNode) => childNode.name === segmentName,
  );
  if (!existingChild) {
    const newChild: TreeNode = {
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
      newChild.buffer = buffer;
      newChild.totalValue = buffer.sizeMib;
      newChild.paddingValue = buffer.paddingMib;
      newChild.value = metric === 'total' ? buffer.sizeMib : buffer.paddingMib;
    }
    parent.children.push(newChild);
    if (!isLast) {
      insertBuffer({
        parent: newChild,
        buffer,
        segments,
        depth: depth + 1,
        metric,
      });
    }
    return;
  }

  if (isLast) {
    if (existingChild.isLeaf) {
      existingChild.totalValue += buffer.sizeMib;
      existingChild.paddingValue += buffer.paddingMib;
      existingChild.value +=
        metric === 'total' ? buffer.sizeMib : buffer.paddingMib;
    } else {
      insertSelfNode({parent: existingChild, buffer, metric});
    }
  } else {
    if (existingChild.isLeaf) {
      existingChild.isLeaf = false;
      if (existingChild.buffer) {
        splitLeafToSelf({parent: existingChild});
        existingChild.buffer = undefined;
      }
    }
    insertBuffer({
      parent: existingChild,
      buffer,
      segments,
      depth: depth + 1,
      metric,
    });
  }
}

function splitLeafToSelf(options: {readonly parent: TreeNode}): void {
  const {parent} = options;
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

/**
 * Inserts a [self] leaf node under a parent scope node to represent its own physical buffer memory.
 */
export function insertSelfNode(options: {
  readonly parent: TreeNode;
  readonly buffer: Readonly<MemoryAnalysisBuffer>;
  readonly metric: 'total' | 'padding';
}): void {
  const {parent, buffer, metric} = options;
  const selfName = `${parent.name} [self]`;
  const selfPath = `${parent.path}/${selfName}`;
  const selfNode = parent.children.find(
    (childNode) => childNode.name === selfName,
  );
  if (selfNode) {
    selfNode.totalValue += buffer.sizeMib;
    selfNode.paddingValue += buffer.paddingMib;
    selfNode.value += metric === 'total' ? buffer.sizeMib : buffer.paddingMib;
  } else {
    const newSelfNode: TreeNode = {
      name: selfName,
      value: metric === 'total' ? buffer.sizeMib : buffer.paddingMib,
      totalValue: buffer.sizeMib,
      paddingValue: buffer.paddingMib,
      children: [],
      buffer,
      path: selfPath,
      isLeaf: true,
      depth: parent.depth + 1,
    };
    parent.children.push(newSelfNode);
  }
}

function aggregateValues(options: {
  readonly node: TreeNode;
  readonly metric: 'total' | 'padding';
}): {readonly total: number; readonly padding: number} {
  const {node, metric} = options;
  let total = node.buffer ? node.totalValue : 0;
  let padding = node.buffer ? node.paddingValue : 0;

  for (const childNode of node.children) {
    const result = aggregateValues({node: childNode, metric});
    total += result.total;
    padding += result.padding;
  }

  node.totalValue = total;
  node.paddingValue = padding;
  node.value = metric === 'total' ? total : padding;
  return {total, padding};
}

/**
 * Recursively sorts tree nodes in descending order of memory value.
 */
export function sortTree(node: TreeNode): void {
  if (node.children.length === 0) return;
  node.children.sort((a, b) => b.value - a.value);
  for (const childNode of node.children) {
    sortTree(childNode);
  }
}
