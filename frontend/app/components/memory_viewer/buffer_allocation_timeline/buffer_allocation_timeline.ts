import {
  AfterViewInit,
  ChangeDetectionStrategy,
  Component,
  ElementRef,
  EventEmitter,
  HostListener,
  Input,
  OnChanges,
  OnDestroy,
  Output,
  SimpleChanges,
  ViewChild,
} from '@angular/core';
import {
  type BufferBlock,
  type BufferBlockProto,
} from 'org_xprof/frontend/app/common/interfaces/data_table';
import * as utils from 'org_xprof/frontend/app/common/utils/utils';

const CONTAINER_COLOR = '#ffffff';
const CONTAINER_BORDER_COLOR = '#d0d0d0';
const LABEL_COLOR = '#000000';
const HOVER_SHADOW_COLOR = 'rgba(0, 0, 0, 0.45)';
const HOVER_SHADOW_BLUR = 10;
const HOVER_SHADOW_OFFSET_Y = 3;
const HOVER_OVERLAY_COLOR = 'rgba(255, 255, 255, 0.45)';

/**
 * The timeline visualizer coordinates map to a virtual bounding square box
 * of size CANVAS_SIZE x CANVAS_SIZE (4096 x 4096).
 */
const CANVAS_SIZE = 4096;

/**
 * Helper to determine a truncated label that fits in the rectangle.
 */
function getFittingLabel(
  label: string,
  width: number,
  height: number,
  fontsize: number,
): string {
  if (!label) return '';
  const kFontHeightScale = 1.2;
  const kFontWidthScale = 0.55;
  const kEllipsis = '...';
  const kGraphMarginPoints = 1.44;

  const usableHeight = height - 2.0 * kGraphMarginPoints;
  const usableWidth = width - 2.0 * kGraphMarginPoints;

  const kHeightThreshold = kFontHeightScale * fontsize;
  const kCharWidth = kFontWidthScale * fontsize;

  if (usableHeight < kHeightThreshold || usableWidth < kCharWidth) {
    return '';
  }

  const maxChars = Math.floor(usableWidth / kCharWidth);

  if (label.length <= maxChars) {
    return label;
  }

  const kMinCharsForTruncation = kEllipsis.length + 1;
  if (maxChars >= kMinCharsForTruncation) {
    return label.substring(0, maxChars - kEllipsis.length) + kEllipsis;
  }

  return '';
}

/**
 * Angular component for rendering decoupled memory viewer buffer allocations timeline using HTML5 Canvas.
 */
@Component({
  changeDetection: ChangeDetectionStrategy.OnPush,
  standalone: false,
  selector: 'buffer-allocation-timeline',
  templateUrl: './buffer_allocation_timeline.ng.html',
  styleUrls: ['./buffer_allocation_timeline.scss'],
})
export class BufferAllocationTimeline
  implements AfterViewInit, OnChanges, OnDestroy
{
  @Input() bufferBlocks: BufferBlockProto[] = [];
  @Input() totalSteps = 0;
  @Input() totalBytes = 0;
  @Output() readonly selected = new EventEmitter<BufferBlock | null>();
  @Output() readonly hovered = new EventEmitter<BufferBlock | null>();

  /**
   * The list of buffer blocks positioned and scaled for rendering.
   * Computed in {@link computeLayout} and used to draw the timeline on canvas
   * and check coordinate selection events.
   */
  private layoutBlocks: BufferBlock[] = [];

  @ViewChild('timelineCanvas', {static: true})
  canvasRef!: ElementRef<HTMLCanvasElement>;

  @ViewChild('fullscreenContainer', {static: true})
  fullscreenContainer!: ElementRef<HTMLDivElement>;

  @Input() isFullscreen = false;
  @Output() readonly isFullscreenChange = new EventEmitter<boolean>();

  private resizeObserver?: ResizeObserver;

  searchQuery = '';
  showSearchInput = false;
  private activeSearchRegex: RegExp | null = null;
  private matchedSearchBlocks: BufferBlock[] = [];
  private currentSearchMatchIndex = -1;

  @ViewChild('searchInput', {static: false})
  searchInputEl?: ElementRef<HTMLInputElement>;

  // Zoom/pan state
  scale = 1.0;
  offsetX = 0;
  offsetY = 0;
  fitScale = 1.0;

  // Selection state
  @Input() selectedBlock: BufferBlock | null = null;

  // Hover state for tooltip
  hoveredBlock: BufferBlock | null = null;
  tooltipLeft = 0;
  tooltipTop = 0;

  // Interactivity state
  isPanning = false;
  dragStartX = 0;
  dragStartY = 0;
  dragStartOffsetX = 0;
  dragStartOffsetY = 0;
  hasDragged = false;

  private canvas!: HTMLCanvasElement;
  private ctx!: CanvasRenderingContext2D;

  ngAfterViewInit() {
    this.canvas = this.canvasRef.nativeElement;
    const context = this.canvas.getContext('2d');
    if (!context) {
      throw new Error('Could not get Canvas 2D context');
    }
    this.ctx = context;

    // Use ResizeObserver to trigger resize when the host container becomes visible and has non-zero size
    this.resizeObserver = new ResizeObserver((entries) => {
      for (const entry of entries) {
        if (entry.contentRect.width > 0 && entry.contentRect.height > 0) {
          this.resizeCanvas();
        }
      }
    });
    this.resizeObserver.observe(
      this.canvasRef.nativeElement.parentElement ||
        this.canvasRef.nativeElement,
    );
  }

  ngOnDestroy() {
    this.resizeObserver?.disconnect();
  }

  computeLayout() {
    if (!this.bufferBlocks || this.totalSteps <= 0 || this.totalBytes <= 0) {
      this.layoutBlocks = [];
      return;
    }

    this.layoutBlocks = [];
    const scaleX = CANVAS_SIZE / this.totalSteps;
    const scaleY = CANVAS_SIZE / this.totalBytes;

    const categoryColorMap = new Map<string, string>();
    let colorIdx = 0;

    for (const proto of this.bufferBlocks) {
      if (
        proto.startStep === undefined ||
        proto.endStep === undefined ||
        proto.offset === undefined ||
        proto.size === undefined
      ) {
        continue;
      }

      const width = (proto.endStep - proto.startStep) * scaleX;
      const height = proto.size * scaleY;

      const centerX =
        (proto.startStep + (proto.endStep - proto.startStep) / 2.0) * scaleX;
      const centerY = (proto.offset + proto.size / 2.0) * scaleY;

      const category = proto.category || 'default';
      let color = proto.color;
      if (!color) {
        if (category.startsWith('Allocation')) {
          color = CONTAINER_COLOR; // White for containers
        } else {
          if (!categoryColorMap.has(category)) {
            categoryColorMap.set(
              category,
              utils.getChartItemColorByIndex(colorIdx++),
            );
          }
          color = categoryColorMap.get(category)!;
        }
      }

      this.layoutBlocks.push({
        id: String(
          proto.logicalBufferId !== undefined && proto.logicalBufferId !== -1
            ? proto.logicalBufferId
            : `${category}_${proto.offset}`,
        ),
        x: centerX,
        y: centerY,
        width,
        height,
        offset: proto.offset || 0,
        tooltip: '',
        color,
        label: proto.name || '',
        fontsize: 10,
        isContainer:
          proto.logicalBufferId === undefined ||
          proto.logicalBufferId === -1 ||
          category.startsWith('Allocation'),
        logicalBufferId:
          proto.logicalBufferId !== undefined && proto.logicalBufferId !== -1
            ? proto.logicalBufferId
            : undefined,
        instructionName: proto.name,
        size: proto.size,
        unpaddedSize: proto.unpaddedSize,
        shapeString: proto.shapeString,
        span: [proto.startStep, proto.endStep],
        tfOpName: proto.tfOpName,
        category: proto.category,
        sourceInfo: proto.sourceInfo,
      });
    }
  }

  ngOnChanges(changes: SimpleChanges) {
    if (
      changes['bufferBlocks'] ||
      changes['totalSteps'] ||
      changes['totalBytes']
    ) {
      this.computeLayout();
      this.updateSearchMatches();
      if (this.ctx) {
        this.resetZoom();
        this.draw();
      }
    }
  }

  @HostListener('window:resize')
  onResize() {
    if (this.canvas) {
      this.resizeCanvas();
    }
  }

  resizeCanvas() {
    const parent = this.canvas.parentElement;
    if (!parent) return;
    const rect = parent.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    this.canvas.width = rect.width * dpr;
    this.canvas.height = rect.height * dpr;

    this.ctx.setTransform(1, 0, 0, 1, 0, 0);
    this.ctx.scale(dpr, dpr);

    this.updateFitScale(rect.width, rect.height);
    this.resetZoom();
    this.draw();
  }

  updateFitScale(viewW: number, viewH: number) {
    // Timeline coordinates map to a CANVAS_SIZE x CANVAS_SIZE virtual bounding box
    this.fitScale = Math.min(viewW / CANVAS_SIZE, viewH / CANVAS_SIZE);
  }

  resetZoom() {
    this.scale = 1.0;
    if (this.canvas) {
      const viewW = this.canvas.width / (window.devicePixelRatio || 1);
      const viewH = this.canvas.height / (window.devicePixelRatio || 1);
      this.offsetX = (viewW - CANVAS_SIZE * this.fitScale) / 2;
      this.offsetY = (viewH - CANVAS_SIZE * this.fitScale) / 2;
    } else {
      this.offsetX = 0;
      this.offsetY = 0;
    }
  }

  // Coordinate mapping helpers (mapping CANVAS_SIZE virtual space to canvas CSS pixels)
  toCanvasX(rawX: number): number {
    return rawX * this.fitScale * this.scale + this.offsetX;
  }

  toCanvasY(rawY: number): number {
    return (CANVAS_SIZE - rawY) * this.fitScale * this.scale + this.offsetY;
  }

  toCanvasLength(len: number): number {
    return len * this.fitScale * this.scale;
  }

  toRawX(canvasX: number): number {
    return (canvasX - this.offsetX) / (this.fitScale * this.scale);
  }

  toRawY(canvasY: number): number {
    return (
      CANVAS_SIZE - (canvasY - this.offsetY) / (this.fitScale * this.scale)
    );
  }

  // Draw loop
  draw() {
    if (!this.canvas || !this.ctx) return;

    this.activeSearchRegex = null;
    if (this.searchQuery) {
      try {
        this.activeSearchRegex = new RegExp(this.searchQuery, 'i');
      } catch (e) {
        // Ignore invalid regex
      }
    }

    const viewW = this.canvas.width / (window.devicePixelRatio || 1);
    const viewH = this.canvas.height / (window.devicePixelRatio || 1);

    // Clear canvas
    this.ctx.clearRect(0, 0, viewW, viewH);

    // Viewport boundaries in raw virtual space for culling
    const minRawX = this.toRawX(0);
    const maxRawX = this.toRawX(viewW);
    const minRawY = this.toRawY(viewH);
    const maxRawY = this.toRawY(0);

    // First draw containers (background allocations), then draw inner logical buffers
    for (const block of this.layoutBlocks) {
      if (!block.isContainer) {
        continue;
      }
      if (this.isBlockVisible(block, minRawX, maxRawX, minRawY, maxRawY)) {
        this.drawBlock(block);
      }
    }

    // Draw all non-selected and non-hovered logical buffers first
    for (const block of this.layoutBlocks) {
      if (block.isContainer) {
        continue;
      }
      const isSelected =
        this.selectedBlock && this.selectedBlock.id === block.id;
      if (isSelected) {
        continue;
      }

      const isHovered = this.hoveredBlock && this.hoveredBlock.id === block.id;
      if (isHovered) {
        continue;
      }

      if (this.isBlockVisible(block, minRawX, maxRawX, minRawY, maxRawY)) {
        this.drawBlock(block);
      }
    }

    // Draw selected block on top of everything
    if (this.selectedBlock && !this.selectedBlock.isContainer) {
      if (
        this.isBlockVisible(
          this.selectedBlock,
          minRawX,
          maxRawX,
          minRawY,
          maxRawY,
        )
      ) {
        this.drawBlock(this.selectedBlock);
      }
    }

    // Draw hovered block on top of normal blocks
    if (this.hoveredBlock && !this.hoveredBlock.isContainer) {
      if (
        this.isBlockVisible(
          this.hoveredBlock,
          minRawX,
          maxRawX,
          minRawY,
          maxRawY,
        )
      ) {
        this.drawBlock(this.hoveredBlock);
      }
    }
  }

  isBlockVisible(
    block: BufferBlock,
    minX: number,
    maxX: number,
    minY: number,
    maxY: number,
  ): boolean {
    const halfW = block.width / 2;
    const halfH = block.height / 2;
    const blockMinX = block.x - halfW;
    const blockMaxX = block.x + halfW;
    const blockMinY = block.y - halfH;
    const blockMaxY = block.y + halfH;

    return (
      blockMaxX >= minX &&
      blockMinX <= maxX &&
      blockMaxY >= minY &&
      blockMinY <= maxY
    );
  }

  drawBlock(block: BufferBlock) {
    const drawX = this.toCanvasX(block.x - block.width / 2);
    const drawY = this.toCanvasY(block.y + block.height / 2);
    const drawW = this.toCanvasLength(block.width);
    const drawH = this.toCanvasLength(block.height);

    const isSelected = !!(
      this.selectedBlock &&
      !block.isContainer &&
      this.selectedBlock.logicalBufferId === block.logicalBufferId
    );

    const isHovered =
      this.hoveredBlock &&
      !block.isContainer &&
      this.hoveredBlock.id === block.id;

    this.ctx.save();

    // Determine opacity: dim if selection or search is active, but block doesn't match either
    let opacity = 1.0;
    const hasSelection = !!this.selectedBlock;
    const hasSearch = !!this.searchQuery;

    if (hasSelection || hasSearch) {
      const matchesSelection = isSelected;

      let matchesSearch = false;
      if (hasSearch && !block.isContainer) {
        const text = block.instructionName || block.label || '';
        if (this.activeSearchRegex) {
          matchesSearch = this.activeSearchRegex.test(text);
        } else {
          matchesSearch = text
            .toLowerCase()
            .includes(this.searchQuery.toLowerCase());
        }
      }

      if (!block.isContainer) {
        const isBright =
          (hasSelection && matchesSelection) || (hasSearch && matchesSearch);
        if (!isBright) {
          opacity = 0.3;
        }
      }
    }

    this.ctx.globalAlpha = opacity;

    // Fill block background
    this.ctx.fillStyle = block.color;

    if (isHovered) {
      this.ctx.shadowColor = HOVER_SHADOW_COLOR;
      this.ctx.shadowBlur = HOVER_SHADOW_BLUR;
      this.ctx.shadowOffsetX = 0;
      this.ctx.shadowOffsetY = HOVER_SHADOW_OFFSET_Y;

      this.ctx.fillRect(drawX, drawY, drawW, drawH);

      // Clear shadow for overlay and border strokes
      this.ctx.shadowColor = 'transparent';
      this.ctx.shadowBlur = 0;
      this.ctx.shadowOffsetX = 0;
      this.ctx.shadowOffsetY = 0;

      this.ctx.fillStyle = HOVER_OVERLAY_COLOR;
      this.ctx.fillRect(drawX, drawY, drawW, drawH);
    } else {
      this.ctx.fillRect(drawX, drawY, drawW, drawH);
    }

    // Stroke border
    if (block.isContainer) {
      this.ctx.strokeStyle = CONTAINER_BORDER_COLOR;
      this.ctx.lineWidth = 1;
      this.ctx.strokeRect(drawX, drawY, drawW, drawH);
    }

    // Render labels dynamically
    if (!block.isContainer && drawH > 4) {
      const nodeFontsize = Math.min(Math.max(drawH * 0.6, 8.0), 14.0);
      const label = getFittingLabel(
        block.instructionName || block.label,
        drawW,
        drawH,
        nodeFontsize,
      );
      if (label) {
        this.ctx.font = `${nodeFontsize}px Arial`;
        this.ctx.fillStyle = LABEL_COLOR;
        this.ctx.textAlign = 'center';
        this.ctx.textBaseline = 'middle';
        this.ctx.fillText(label, drawX + drawW / 2, drawY + drawH / 2);
      }
    }

    this.ctx.restore();
  }

  // Wheel zoom
  onWheel(event: WheelEvent) {
    event.preventDefault();
    const zoomFactor = 1.1;
    const direction = event.deltaY < 0 ? 1 : -1;

    const mouseX = event.offsetX;
    const mouseY = event.offsetY;

    // Zoom relative to mouse cursor position
    const rawXBefore = this.toRawX(mouseX);
    const rawYBefore = this.toRawY(mouseY);

    if (direction > 0) {
      this.scale *= zoomFactor;
    } else {
      this.scale /= zoomFactor;
    }
    // Clamp zoom level
    this.scale = Math.max(1.0, Math.min(this.scale, 500.0));

    this.offsetX = mouseX - rawXBefore * this.fitScale * this.scale;
    this.offsetY =
      mouseY - (CANVAS_SIZE - rawYBefore) * this.fitScale * this.scale;

    this.draw();
    this.updateHoverState(mouseX, mouseY);
  }

  // Mouse pan/select
  onMouseDown(event: MouseEvent) {
    if (event.button === 0) {
      // Left button
      this.isPanning = true;
      this.dragStartX = event.clientX;
      this.dragStartY = event.clientY;
      this.dragStartOffsetX = this.offsetX;
      this.dragStartOffsetY = this.offsetY;
      this.hasDragged = false;
    }
  }

  onMouseMove(event: MouseEvent) {
    if (this.isPanning) {
      const dx = event.clientX - this.dragStartX;
      const dy = event.clientY - this.dragStartY;
      if (Math.abs(dx) > 3 || Math.abs(dy) > 3) {
        this.hasDragged = true;
      }
      this.offsetX = this.dragStartOffsetX + dx;
      this.offsetY = this.dragStartOffsetY + dy;
      this.draw();
      this.hoveredBlock = null;
      return;
    }

    this.updateHoverState(event.offsetX, event.offsetY);
  }

  updateHoverState(mouseX: number, mouseY: number) {
    const rawX = this.toRawX(mouseX);
    const rawY = this.toRawY(mouseY);

    let foundBlock: BufferBlock | null = null;
    let minArea = Infinity;

    for (const block of this.layoutBlocks) {
      if (block.isContainer) {
        continue;
      }
      if (this.isPointInBlock(rawX, rawY, block)) {
        const area = block.width * block.height;
        if (area < minArea) {
          minArea = area;
          foundBlock = block;
        }
      }
    }

    if (foundBlock !== this.hoveredBlock) {
      this.hoveredBlock = foundBlock;
      this.hovered.emit(this.hoveredBlock);
      this.draw();
    }

    if (this.hoveredBlock) {
      const viewW = this.canvas.width / (window.devicePixelRatio || 1);
      const viewH = this.canvas.height / (window.devicePixelRatio || 1);
      const tooltipW = 350;
      const tooltipH = 150;

      if (mouseX + 15 + tooltipW > viewW) {
        this.tooltipLeft = mouseX - tooltipW - 15;
      } else {
        this.tooltipLeft = mouseX + 15;
      }

      if (mouseY + 15 + tooltipH > viewH) {
        this.tooltipTop = mouseY - tooltipH - 15;
      } else {
        this.tooltipTop = mouseY + 15;
      }
    }
  }

  onMouseLeave() {
    this.isPanning = false;
    this.hoveredBlock = null;
    this.hovered.emit(null);
    this.draw();
  }

  isPointInBlock(rawX: number, rawY: number, block: BufferBlock): boolean {
    const halfW = block.width / 2;
    const halfH = block.height / 2;
    return (
      rawX >= block.x - halfW &&
      rawX <= block.x + halfW &&
      rawY >= block.y - halfH &&
      rawY <= block.y + halfH
    );
  }

  onMouseUp(event?: MouseEvent) {
    if (this.isPanning) {
      this.isPanning = false;
      if (!this.hasDragged && event) {
        this.checkSelection(event.offsetX, event.offsetY);
      }
    }
  }
  checkSelection(mouseX: number, mouseY: number) {
    const rawX = this.toRawX(mouseX);
    const rawY = this.toRawY(mouseY);

    let clickedBlock: BufferBlock | null = null;
    let minArea = Infinity;

    for (const block of this.layoutBlocks) {
      if (block.isContainer) {
        continue;
      }
      const left = block.x - block.width / 2;
      const right = block.x + block.width / 2;
      const bottom = block.y - block.height / 2;
      const top = block.y + block.height / 2;

      if (rawX >= left && rawX <= right && rawY >= bottom && rawY <= top) {
        const area = block.width * block.height;
        if (area < minArea) {
          minArea = area;
          clickedBlock = block;
        }
      }
    }

    if (
      clickedBlock &&
      this.selectedBlock &&
      clickedBlock.logicalBufferId === this.selectedBlock.logicalBufferId
    ) {
      clickedBlock = null;
    }

    this.selectedBlock = clickedBlock;
    this.selected.emit(clickedBlock);
    this.draw();
  }

  getHexOffset(offset?: number): string {
    if (offset === undefined || offset === null) {
      return '';
    }
    return '0x' + offset.toString(16);
  }

  get selectedBlockSizeMiB(): string {
    if (!this.selectedBlock || this.selectedBlock.size === undefined) {
      return '';
    }
    return (this.selectedBlock.size / (1024 * 1024)).toFixed(2);
  }

  get selectedBlockUnpaddedSizeMiB(): string {
    if (!this.selectedBlock || this.selectedBlock.unpaddedSize === undefined) {
      return 'N/A';
    }
    return (this.selectedBlock.unpaddedSize / (1024 * 1024)).toFixed(2);
  }

  get selectedBlockPaddingOverheadMiB(): string {
    if (
      !this.selectedBlock ||
      this.selectedBlock.size === undefined ||
      this.selectedBlock.unpaddedSize === undefined
    ) {
      return 'N/A';
    }
    return (
      (this.selectedBlock.size - this.selectedBlock.unpaddedSize) /
      (1024 * 1024)
    ).toFixed(2);
  }

  @HostListener('document:fullscreenchange')
  onFullscreenChange() {
    this.isFullscreen =
      document.fullscreenElement === this.fullscreenContainer.nativeElement;
    this.isFullscreenChange.emit(this.isFullscreen);
    if (this.canvas) {
      setTimeout(() => {
        this.resizeCanvas();
      }, 100);
    }
  }

  toggleFullscreen() {
    const element = this.fullscreenContainer.nativeElement;
    if (!document.fullscreenElement) {
      element
        .requestFullscreen()
        .catch((err) => {
          console.error(
            `Error attempting to enable fullscreen mode: ${err.message}`,
          );
        });
    } else {
      document.exitFullscreen()
        .catch((err) => {
          console.error(
            `Error attempting to exit fullscreen mode: ${err.message}`,
          );
        });
    }
  }

  onSearchInput(event: Event) {
    const target = event.target as HTMLInputElement;
    this.searchQuery = target.value;
    this.updateSearchMatches();
    this.draw();
  }

  updateSearchMatches() {
    this.matchedSearchBlocks = [];
    this.currentSearchMatchIndex = -1;

    if (!this.searchQuery) return;

    let regex: RegExp | null = null;
    try {
      regex = new RegExp(this.searchQuery, 'i');
    } catch (e) {
      // Ignore invalid regex
    }

    for (const block of this.layoutBlocks) {
      if (block.isContainer) continue;
      const text = (block.instructionName || block.label).toLowerCase();
      let isMatch = false;
      if (regex) {
        isMatch = regex.test(text);
      } else {
        isMatch = text.includes(this.searchQuery.toLowerCase());
      }
      if (isMatch) {
        this.matchedSearchBlocks.push(block);
      }
    }
  }

  onSearchEnter() {
    if (this.matchedSearchBlocks.length === 0) return;

    this.currentSearchMatchIndex =
      (this.currentSearchMatchIndex + 1) % this.matchedSearchBlocks.length;
    const match = this.matchedSearchBlocks[this.currentSearchMatchIndex];

    // Select the matched block
    this.selectedBlock = match;
    this.selected.emit(match);

    // Zoom and center on the matched block
    this.scale = Math.max(this.scale, 5.0);
    this.centerOnSelectedBlock();
  }

  centerOnSelectedBlock() {
    if (!this.canvas || !this.selectedBlock) return;
    const viewW = this.canvas.width / (window.devicePixelRatio || 1);
    const viewH = this.canvas.height / (window.devicePixelRatio || 1);
    this.offsetX =
      viewW / 2 - this.selectedBlock.x * this.fitScale * this.scale;
    this.offsetY =
      viewH / 2 -
      (CANVAS_SIZE - this.selectedBlock.y) * this.fitScale * this.scale;
    this.draw();
  }

  toggleSearch() {
    this.showSearchInput = !this.showSearchInput;
    if (!this.showSearchInput) {
      this.searchQuery = '';
      this.matchedSearchBlocks = [];
      this.currentSearchMatchIndex = -1;
      this.draw();
    } else {
      setTimeout(() => {
        this.searchInputEl?.nativeElement.focus();
      }, 50);
    }
  }

  closeSearch() {
    this.showSearchInput = false;
    this.searchQuery = '';
    this.matchedSearchBlocks = [];
    this.currentSearchMatchIndex = -1;
    this.draw();
  }
}
