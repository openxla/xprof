import '@material/web/dialog/dialog';
import '@material/web/iconbutton/icon-button';

import type {MdDialog} from '@material/web/dialog/dialog';
import {css, html, LitElement} from 'lit';
import {customElement, property, state} from 'lit/decorators.js';
import {ShortcutItem, TRACE_VIEWER_SHORTCUTS} from './shortcuts';

/**
 * Represents a keyboard shortcut item flattened with its parent category title.
 */
export interface FlatShortcutItem extends ShortcutItem {
  category: string;
  categoryIcon?: string;
}

/**
 * Filtered group of shortcuts for category section rendering.
 */
export interface FilteredCategory {
  title: string;
  iconName?: string;
  items: FlatShortcutItem[];
}

const selectIcon = html`<svg
  width="15"
  height="15"
  viewBox="0 0 28 28"
  fill="none"
  xmlns="http://www.w3.org/2000/svg"
  style="vertical-align: -2px; margin-right: 6px; display: inline-block;">
  <g clip-path="url(#clip_select)">
    <path
      d="M15.1875 22L12.3125 15.8542L9 20V6L20 15H14.125L17 21.1458L15.1875 22Z"
      fill="#444746" />
  </g>
  <defs>
    <clipPath id="clip_select">
      <rect width="20" height="20" fill="white" transform="translate(4 4)" />
    </clipPath>
  </defs>
</svg>`;

const panIcon = html`<svg
  width="15"
  height="15"
  viewBox="0 0 20 20"
  fill="none"
  xmlns="http://www.w3.org/2000/svg"
  style="vertical-align: -2px; margin-right: 6px; display: inline-block;">
  <g clip-path="url(#clip_pan)">
    <path
      d="M10 18L6.5 14.5L7.5625 13.4375L9.25 15.125V10.75H4.875L6.5625 12.4375L5.5 13.5L2 10L5.5 6.5L6.5625 7.5625L4.875 9.25H9.25V4.875L7.5625 6.5625L6.5 5.5L10 2L13.5 5.5L12.4375 6.5625L10.75 4.875V9.25H15.125L13.4375 7.5625L14.5 6.5L18 10L14.5 13.5L13.4375 12.4375L15.125 10.75H10.75V15.125L12.4375 13.4375L13.5 14.5L10 18Z"
      fill="#444746" />
  </g>
  <defs>
    <clipPath id="clip_pan">
      <rect width="20" height="20" fill="white" />
    </clipPath>
  </defs>
</svg>`;

const zoomIcon = html`<svg
  width="15"
  height="15"
  viewBox="0 0 20 20"
  fill="none"
  xmlns="http://www.w3.org/2000/svg"
  style="vertical-align: -2px; margin-right: 6px; display: inline-block;">
  <g clip-path="url(#clip_zoom)">
    <path
      d="M10 17L7 14L8.0625 12.9375L9.25 14.125V5.875L8.0625 7.0625L7 6L10 3L13 6L11.9375 7.0625L10.75 5.875V14.125L11.9375 12.9375L13 14L10 17Z"
      fill="#444746" />
  </g>
  <defs>
    <clipPath id="clip_zoom">
      <rect width="20" height="20" fill="white" />
    </clipPath>
  </defs>
</svg>`;

const measureIcon = html`<svg
  width="15"
  height="15"
  viewBox="0 0 20 20"
  fill="none"
  xmlns="http://www.w3.org/2000/svg"
  style="vertical-align: -2px; margin-right: 6px; display: inline-block;">
  <g clip-path="url(#clip_measure)">
    <path
      d="M18 16H16.5V4H18V16ZM15 10L12 13L10.9375 11.9375L12.125 10.75H7.875L9.0625 11.9375L8 13L5 10L8 7L9.0625 8.0625L7.875 9.25H12.125L10.9375 8.0625L12 7L15 10ZM3.5 16H2L2 4H3.5L3.5 16Z"
      fill="#444746" />
  </g>
  <defs>
    <clipPath id="clip_measure">
      <rect
        width="20"
        height="20"
        fill="white"
        transform="matrix(0 -1 1 0 0 20)" />
    </clipPath>
  </defs>
</svg>`;

const closeXIcon = html`<svg
  width="20"
  height="20"
  viewBox="0 0 24 24"
  fill="none"
  xmlns="http://www.w3.org/2000/svg">
  <path
    d="M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12 19 6.41Z"
    fill="#5f6368" />
</svg>`;

const searchIcon = html`<svg
  width="16"
  height="16"
  viewBox="0 0 24 24"
  fill="none"
  xmlns="http://www.w3.org/2000/svg"
  style="vertical-align: middle; display: inline-block; flex-shrink: 0;">
  <path
    d="M15.5 14H14.71L14.43 13.73C15.41 12.59 16 11.11 16 9.5C16 5.91 13.09 3 9.5 3C5.91 3 3 5.91 3 9.5C3 13.09 5.91 16 9.5 16C11.11 16 12.59 15.41 13.73 14.43L14 14.71V15.5L19 20.49L20.49 19L15.5 14ZM9.5 14C7.01 14 5 11.99 5 9.5C5 7.01 7.01 5 9.5 5C11.99 5 14 7.01 14 9.5C14 11.99 11.99 14 9.5 14Z"
    fill="#5f6368" />
</svg>`;

const clearIcon = html`<svg
  width="14"
  height="14"
  viewBox="0 0 24 24"
  fill="none"
  xmlns="http://www.w3.org/2000/svg"
  style="vertical-align: middle; display: inline-block;">
  <path
    d="M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12 19 6.41Z"
    fill="#5f6368" />
</svg>`;

// Category Icons (Blue #1a73e8)
const navCategoryIcon = html`<svg
  width="16"
  height="16"
  viewBox="0 0 24 24"
  fill="none"
  xmlns="http://www.w3.org/2000/svg"
  style="flex-shrink: 0;">
  <path
    d="M12 2L4.5 20.29L5.21 21L12 18L18.79 21L19.5 20.29L12 2Z"
    fill="#1a73e8" />
</svg>`;

const modeCategoryIcon = html`<svg
  width="16"
  height="16"
  viewBox="0 0 24 24"
  fill="none"
  xmlns="http://www.w3.org/2000/svg"
  style="flex-shrink: 0;">
  <path
    d="M9 11.24V7.5C9 6.12 10.12 5 11.5 5S14 6.12 14 7.5V11.24C15.82 12.18 17 14.1 17 16.5C17 19.54 14.54 22 11.5 22S6 19.54 6 16.5C6 14.1 7.18 12.18 9 11.24ZM11.5 7C11.22 7 11 7.22 11 7.5V13H12V7.5C12 7.22 11.78 7 11.5 7Z"
    fill="#1a73e8" />
</svg>`;

const selectionCategoryIcon = html`<svg
  width="16"
  height="16"
  viewBox="0 0 24 24"
  fill="none"
  xmlns="http://www.w3.org/2000/svg"
  style="flex-shrink: 0;">
  <path
    d="M3 5H5V3C3.9 3 3 3.9 3 5ZM3 13H5V11H3V13ZM7 21H9V19H7V21ZM3 9H5V7H3V9ZM13 3H11V5H13V3ZM19 3V5H21C21 3.9 20.1 3 19 3ZM5 21V19H3C3 20.1 3.9 21 5 21ZM3 17H5V15H3V17ZM9 3H7V5H9V3ZM11 21H13V19H11V21ZM19 13H21V11H19V13ZM19 21C20.1 21 21 20.1 21 19H19V21ZM19 9H21V7H19V9ZM19 17H21V15H19V17ZM15 21H17V19H15V21ZM15 5H17V3H15V5ZM7 17H17V7H7V17Z"
    fill="#1a73e8" />
</svg>`;

const generalCategoryIcon = html`<svg
  width="16"
  height="16"
  viewBox="0 0 24 24"
  fill="none"
  xmlns="http://www.w3.org/2000/svg"
  style="flex-shrink: 0;">
  <path
    d="M20 5H4C2.9 5 2 5.9 2 7V17C2 18.1 2.9 19 4 19H20C21.1 19 22 18.1 22 17V7C22 5.9 21.1 5 20 5ZM20 17H4V7H20V17ZM5 8H7V10H5V8ZM5 11H7V13H5V11ZM5 14H7V16H5V14ZM8 8H10V10H8V8ZM8 11H10V13H8V11ZM8 14H16V16H8V14ZM11 8H13V10H11V8ZM11 11H13V13H11V11ZM14 8H16V10H14V8ZM14 11H16V13H14V11ZM17 8H19V10H17V8ZM17 11H19V13H17V11ZM17 14H19V16H17V14Z"
    fill="#1a73e8" />
</svg>`;

// Mouse Control Icons (Muted Gray #5f6368)
const mouseScrollIcon = html`<svg
  width="13"
  height="13"
  viewBox="0 0 24 24"
  fill="none"
  stroke="#5f6368"
  stroke-width="2"
  stroke-linecap="round"
  stroke-linejoin="round"
  style="flex-shrink: 0;">
  <rect x="6" y="3" width="12" height="18" rx="6" />
  <line x1="12" y1="7" x2="12" y2="11" />
</svg>`;

const mouseHandIcon = html`<svg
  width="13"
  height="13"
  viewBox="0 0 24 24"
  fill="none"
  stroke="#5f6368"
  stroke-width="2"
  stroke-linecap="round"
  stroke-linejoin="round"
  style="flex-shrink: 0;">
  <path
    d="M18 11V6a2 2 0 0 0-4 0v5M14 10V4a2 2 0 0 0-4 0v7M10 10.5V6a2 2 0 0 0-4 0v8a6 6 0 0 0 12 0v-4a2 2 0 0 0-4 0" />
</svg>`;

const mouseClickIcon = html`<svg
  width="13"
  height="13"
  viewBox="0 0 24 24"
  fill="none"
  stroke="#5f6368"
  stroke-width="2"
  stroke-linecap="round"
  stroke-linejoin="round"
  style="flex-shrink: 0;">
  <circle cx="12" cy="12" r="7" />
  <circle cx="12" cy="12" r="2" />
  <line x1="12" y1="2" x2="12" y2="5" />
  <line x1="12" y1="19" x2="12" y2="22" />
  <line x1="2" y1="12" x2="5" y2="12" />
  <line x1="19" y1="12" x2="22" y2="12" />
</svg>`;

const mouseDoubleClickIcon = html`<svg
  width="13"
  height="13"
  viewBox="0 0 24 24"
  fill="none"
  stroke="#5f6368"
  stroke-width="2"
  stroke-linecap="round"
  stroke-linejoin="round"
  style="flex-shrink: 0;">
  <path d="M4 8V4h4M20 8V4h-4M4 16v4h4M20 16v4h-4" />
</svg>`;

const mouseRefreshIcon = html`<svg
  width="13"
  height="13"
  viewBox="0 0 24 24"
  fill="none"
  stroke="#5f6368"
  stroke-width="2"
  stroke-linecap="round"
  stroke-linejoin="round"
  style="flex-shrink: 0;">
  <path
    d="M21 2v6h-6M3 12a9 9 0 0 1 15-6.7L21 8M3 22v-6h6M21 12a9 9 0 0 1-15 6.7L3 16" />
</svg>`;

const mouseTimerIcon = html`<svg
  width="13"
  height="13"
  viewBox="0 0 24 24"
  fill="none"
  stroke="#5f6368"
  stroke-width="2"
  stroke-linecap="round"
  stroke-linejoin="round"
  style="flex-shrink: 0;">
  <circle cx="12" cy="13" r="8" />
  <path d="M12 9v4l2.5 2.5M10 2h4" />
</svg>`;

const mouseZoomIcon = html`<svg
  width="13"
  height="13"
  viewBox="0 0 24 24"
  fill="none"
  stroke="#5f6368"
  stroke-width="2"
  stroke-linecap="round"
  stroke-linejoin="round"
  style="flex-shrink: 0;">
  <circle cx="11" cy="11" r="7" />
  <line x1="21" y1="21" x2="16.65" y2="16.65" />
  <line x1="11" y1="8" x2="11" y2="14" />
  <line x1="8" y1="11" x2="14" y2="11" />
</svg>`;

const mouseMeasureIcon = html`<svg
  width="13"
  height="13"
  viewBox="0 0 24 24"
  fill="none"
  stroke="#5f6368"
  stroke-width="2"
  stroke-linecap="round"
  stroke-linejoin="round"
  style="flex-shrink: 0;">
  <path d="M2 12h20M7 8v4M12 6v6M17 8v4" />
</svg>`;

/**
 * A Web Component (GM3) dialog displaying keyboard shortcuts and controls
 * for Trace Viewer v2 with unified table layout, categories, and search filtering.
 */
@customElement('trace-viewer-help-dialog')
export class TraceViewerHelpDialog extends LitElement {
  static override styles = css`
    * {
      box-sizing: border-box;
      font-family: Roboto, 'Google Sans', sans-serif;
    }
    :host {
      display: inline-block;
      font-family: Roboto, 'Google Sans', sans-serif;
    }
    md-dialog {
      --md-dialog-container-color: #ffffff;
      --md-sys-color-primary: #1a73e8;
      --md-sys-color-on-surface: #202124;
      --md-sys-color-on-surface-variant: #5f6368;
      --md-dialog-container-shape: 28px;
      border-radius: 28px;
      --md-dialog-container-min-width: 760px;
      --md-dialog-container-max-width: 760px;
      min-width: 760px;
      max-width: 760px;
      height: 540px;
      max-height: 540px;
      --md-dialog-headline-font: Roboto, 'Google Sans', sans-serif;
      --md-dialog-content-font: Roboto, 'Google Sans', sans-serif;
      --md-sys-typescale-headline-small-font: Roboto, 'Google Sans', sans-serif;
      --md-sys-typescale-body-medium-font: Roboto, 'Google Sans', sans-serif;
      font-family: Roboto, 'Google Sans', sans-serif;
    }
    md-icon-button {
      outline: none;
    }
    .dialog-headline {
      display: flex;
      justify-content: space-between;
      align-items: center;
      width: 100%;
      padding: 16px 20px 14px 24px;
      border-bottom: 1px solid #e0e0e0;
      box-sizing: border-box;
      font-family: Roboto, 'Google Sans', sans-serif;
    }
    .headline-left {
      display: flex;
      align-items: center;
      gap: 12px;
    }
    .dialog-title {
      font-size: 20px;
      font-weight: 500;
      color: #202124;
      line-height: 28px;
      font-family: Roboto, 'Google Sans', sans-serif;
    }
    .shortcut-count-badge {
      display: inline-flex;
      align-items: center;
      padding: 3px 10px;
      border-radius: 9999px;
      background-color: #f1f3f4;
      color: #5f6368;
      font-family: Roboto, 'Google Sans', sans-serif;
      font-size: 12px;
      font-weight: 500;
      line-height: 16px;
      white-space: nowrap;
    }
    .headline-right {
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .search-box {
      display: flex;
      align-items: center;
      gap: 8px;
      height: 36px;
      width: 240px;
      padding: 0 12px;
      background-color: #ffffff;
      border: 1px solid #1a73e8;
      border-radius: 9999px;
      box-sizing: border-box;
      transition: box-shadow 0.2s;
    }
    .search-box:focus-within {
      box-shadow: 0 1px 3px rgba(26, 115, 232, 0.3);
    }
    .search-input {
      flex: 1;
      height: 100%;
      border: none;
      outline: none;
      background: transparent;
      font-size: 13px;
      font-weight: 400;
      font-family: Roboto, 'Google Sans', sans-serif;
      color: #202124;
    }
    .search-input::placeholder {
      color: #5f6368;
      font-family: Roboto, 'Google Sans', sans-serif;
    }
    .clear-search-btn {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      border: none;
      background: transparent;
      padding: 2px;
      cursor: pointer;
      border-radius: 50%;
      color: #5f6368;
      line-height: 0;
    }
    .clear-search-btn:hover {
      background-color: #f1f3f4;
    }
    .close-btn {
      color: #5f6368;
    }
    .dialog-content {
      width: 100%;
      height: 444px;
      display: flex;
      flex-direction: column;
      padding: 16px 24px 20px 24px;
      box-sizing: border-box;
      overflow: hidden;
      font-family: Roboto, 'Google Sans', sans-serif;
    }
    .table-card {
      flex: 1;
      display: flex;
      flex-direction: column;
      border: 1px solid #e0e0e0;
      border-radius: 12px;
      overflow: hidden;
      min-height: 0;
      background-color: #ffffff;
      font-family: Roboto, 'Google Sans', sans-serif;
    }
    .table-container {
      flex: 1;
      overflow-y: auto;
      min-height: 0;
      font-family: Roboto, 'Google Sans', sans-serif;
    }
    .shortcuts-table {
      width: 100%;
      border-collapse: collapse;
      table-layout: fixed;
      font-family: Roboto, 'Google Sans', sans-serif;
    }
    .shortcuts-table thead {
      position: sticky;
      top: 0;
      background-color: #ffffff;
      z-index: 2;
    }
    .shortcuts-table thead th,
    th {
      padding: 12px 16px;
      border-bottom: 1px solid #e0e0e0;
      font-family: Roboto, sans-serif;
      font-size: 11px;
      font-weight: 500;
      text-transform: uppercase;
      letter-spacing: 0.8px;
      color: #5f6368;
      text-align: left;
      background-color: #ffffff;
    }
    .col-label {
      width: 38%;
    }
    .col-keyboard {
      width: 32%;
    }
    .col-mouse {
      width: 30%;
    }
    .category-header-row td {
      background-color: #f0f4f9;
      padding: 9px 16px;
      border-bottom: 1px solid #e0e0e0;
      font-family: Roboto, 'Google Sans', sans-serif;
    }
    .category-header-row:not(:first-child) td {
      border-top: 1px solid #e0e0e0;
    }
    .category-header,
    .category-header-content {
      display: flex;
      align-items: center;
      gap: 8px;
      color: #1a73e8;
      font-family: Roboto, 'Google Sans', sans-serif;
      font-size: 13px;
      font-weight: 500;
      line-height: 18px;
    }
    .shortcut-table-row {
      border-bottom: 1px solid #f1f3f4;
      transition: background-color 0.15s ease;
      font-family: Roboto, 'Google Sans', sans-serif;
    }
    .shortcut-table-row:last-child {
      border-bottom: none;
    }
    .shortcut-table-row:hover {
      background-color: #f8f9fa;
    }
    td,
    .action-label,
    .cell-label {
      padding: 9px 16px;
      font-family: Roboto, 'Google Sans', sans-serif;
      font-size: 13px;
      color: #202124;
      font-weight: 400;
      vertical-align: middle;
      white-space: nowrap;
      text-overflow: ellipsis;
      overflow: hidden;
    }
    .label-content {
      display: flex;
      align-items: center;
      gap: 8px;
      font-family: Roboto, 'Google Sans', sans-serif;
      font-size: 13px;
      font-weight: 400;
    }
    .cell-keyboard {
      padding: 9px 16px;
      vertical-align: middle;
      text-align: left;
      white-space: nowrap;
      font-family: Roboto, 'Google Sans', sans-serif;
    }
    .shortcut-keys {
      display: inline-flex;
      align-items: center;
      gap: 4px;
      color: #5f6368;
      font-family: Roboto, 'Google Sans', sans-serif;
      font-size: 13px;
    }
    .key-separator {
      color: #5f6368;
      font-family: Roboto, 'Google Sans', sans-serif;
      font-size: 12px;
      margin: 0 4px;
      font-weight: 400;
    }
    kbd {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      background: linear-gradient(180deg, #ffffff 0%, #f1f3f4 100%);
      border: 1px solid #dadce0;
      border-bottom: 2px solid #9aa0a6;
      border-radius: 6px;
      color: #202124;
      font-family: Roboto, sans-serif;
      font-size: 12px;
      font-weight: 600;
      padding: 2px 8px;
      line-height: 18px;
      min-width: 14px;
      box-shadow:
        0 1px 2px rgba(0, 0, 0, 0.06),
        inset 0 1px 0 #ffffff;
    }
    .cell-mouse {
      padding: 9px 16px;
      vertical-align: middle;
      text-align: left;
      white-space: nowrap;
      font-family: Roboto, 'Google Sans', sans-serif;
    }
    .mouse-badge,
    .mouse-control-pill {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 3px 10px;
      background-color: #f8f9fa;
      border: 1px solid #dadce0;
      border-radius: 6px;
      font-family: Roboto, 'Google Sans', sans-serif;
      font-size: 12px;
      font-weight: 500;
      color: #3c4043;
      white-space: nowrap;
      line-height: 18px;
    }
    .mouse-icon {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      flex-shrink: 0;
    }
    .empty-search-state {
      padding: 48px 24px;
      text-align: center;
      font-family: Roboto, 'Google Sans', sans-serif;
    }
    .empty-search-title {
      font-family: Roboto, 'Google Sans', sans-serif;
      font-size: 16px;
      font-weight: 500;
      color: #202124;
      margin-bottom: 8px;
    }
    .empty-search-subtext {
      font-family: Roboto, 'Google Sans', sans-serif;
      font-size: 13px;
      color: #5f6368;
      max-width: 380px;
      margin: 0 auto;
      line-height: 20px;
    }
  `;

  @property({type: Boolean, reflect: true}) open = false;
  @state() searchQuery = '';

  private readonly handleKeyDown = (e: KeyboardEvent) => {
    if (this.open && e.key === 'Escape') {
      e.preventDefault();
      e.stopPropagation();
      this.closeDialog();
    }
  };

  override connectedCallback() {
    super.connectedCallback();
    window.addEventListener('keydown', this.handleKeyDown);
  }

  override disconnectedCallback() {
    super.disconnectedCallback();
    window.removeEventListener('keydown', this.handleKeyDown);
  }

  async openDialog() {
    this.searchQuery = '';
    this.open = true;
    await this.updateComplete;
    const dialog = this.shadowRoot?.querySelector(
      'md-dialog',
    ) as MdDialog | null;
    await dialog?.show();
    const searchInput = this.shadowRoot?.querySelector(
      '.search-input',
    ) as HTMLInputElement | null;
    searchInput?.focus();
    searchInput?.select();
  }

  async closeDialog() {
    this.open = false;
    this.searchQuery = '';
    const dialog = this.shadowRoot?.querySelector(
      'md-dialog',
    ) as MdDialog | null;
    await dialog?.close();
  }

  private handleSearchInput(e: Event) {
    const target = e.target as HTMLInputElement;
    this.searchQuery = target.value;
  }

  private handleSearchKeyDown(e: KeyboardEvent) {
    if (e.key === 'Escape') {
      e.preventDefault();
      e.stopPropagation();
      if (this.searchQuery) {
        this.clearSearch();
      } else {
        this.closeDialog();
      }
      return;
    }
    // Stop propagation so typing does not bubble to window container shortcuts.
    e.stopPropagation();
  }

  getFocusableElements(): HTMLElement[] {
    if (!this.shadowRoot) return [];
    const selectors = [
      'input:not([disabled])',
      'button:not([disabled])',
      'md-icon-button:not([disabled])',
      '[tabindex]:not([tabindex="-1"])',
    ].join(', ');
    const candidates = Array.from(
      this.shadowRoot.querySelectorAll<HTMLElement>(selectors),
    );
    return candidates.filter((el) => {
      if (el.hidden) return false;
      const style = window.getComputedStyle(el);
      return style.display !== 'none' && style.visibility !== 'hidden';
    });
  }

  handleDialogKeyDown = (e: KeyboardEvent) => {
    if (e.key === 'Tab') {
      const focusable = this.getFocusableElements();
      if (focusable.length === 0) {
        e.preventDefault();
        return;
      }
      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      const currentActive = this.shadowRoot
        ?.activeElement as HTMLElement | null;

      if (e.shiftKey) {
        if (
          currentActive === first ||
          !focusable.includes(currentActive as HTMLElement)
        ) {
          e.preventDefault();
          last.focus();
        }
      } else {
        if (
          currentActive === last ||
          !focusable.includes(currentActive as HTMLElement)
        ) {
          e.preventDefault();
          first.focus();
        }
      }
    }
  };

  clearSearch() {
    this.searchQuery = '';
    const input = this.shadowRoot?.querySelector(
      '.search-input',
    ) as HTMLInputElement | null;
    if (input) {
      input.value = '';
      input.focus();
    }
  }

  get allShortcuts(): FlatShortcutItem[] {
    const result: FlatShortcutItem[] = [];
    for (const section of TRACE_VIEWER_SHORTCUTS) {
      for (const item of section.items) {
        result.push({
          ...item,
          category: section.title,
          categoryIcon: section.iconName,
        });
      }
    }
    return result;
  }

  get filteredCategories(): FilteredCategory[] {
    const query = this.searchQuery.trim().toLowerCase();
    const categories: FilteredCategory[] = [];

    for (const section of TRACE_VIEWER_SHORTCUTS) {
      const categoryExactMatches =
        query.length >= 4 &&
        (section.title.toLowerCase() === query ||
          query.includes(section.title.toLowerCase()) ||
          section.title.toLowerCase().startsWith(query));
      const matchingItems: FlatShortcutItem[] = [];

      for (const item of section.items) {
        if (!query) {
          matchingItems.push({
            ...item,
            category: section.title,
            categoryIcon: section.iconName,
          });
        } else {
          const descMatches = item.description.toLowerCase().includes(query);
          const keysMatch = item.keys.some((k) =>
            k.toLowerCase().includes(query),
          );
          const mouseMatches =
            item.mouseControl?.label.toLowerCase().includes(query) ?? false;
          if (
            descMatches ||
            keysMatch ||
            mouseMatches ||
            categoryExactMatches
          ) {
            matchingItems.push({
              ...item,
              category: section.title,
              categoryIcon: section.iconName,
            });
          }
        }
      }

      if (matchingItems.length > 0) {
        categories.push({
          title: section.title,
          iconName: section.iconName,
          items: matchingItems,
        });
      }
    }

    return categories;
  }

  get filteredShortcuts(): FlatShortcutItem[] {
    return this.filteredCategories.flatMap((category) => category.items);
  }

  get totalShortcutsCount(): number {
    return this.allShortcuts.length;
  }

  get filteredShortcutsCount(): number {
    return this.filteredShortcuts.length;
  }

  private renderCategoryIcon(iconName?: string) {
    switch (iconName) {
      case 'navigation':
        return navCategoryIcon;
      case 'gesture':
        return modeCategoryIcon;
      case 'select_all':
        return selectionCategoryIcon;
      case 'general':
        return generalCategoryIcon;
      default:
        return '';
    }
  }

  private renderIcon(iconName?: string) {
    switch (iconName) {
      case 'select':
        return selectIcon;
      case 'pan':
        return panIcon;
      case 'zoom':
        return zoomIcon;
      case 'measure':
        return measureIcon;
      default:
        return '';
    }
  }

  private renderMouseIcon(iconName?: string) {
    switch (iconName) {
      case 'mouse':
        return mouseScrollIcon;
      case 'pan':
        return mouseHandIcon;
      case 'click':
        return mouseClickIcon;
      case 'double_click':
        return mouseDoubleClickIcon;
      case 'refresh':
        return mouseRefreshIcon;
      case 'timer':
        return mouseTimerIcon;
      case 'zoom':
        return mouseZoomIcon;
      case 'measure':
        return mouseMeasureIcon;
      case 'search':
        return searchIcon;
      default:
        return mouseClickIcon;
    }
  }

  override render() {
    const categories = this.filteredCategories;
    const totalCount = this.allShortcuts.length;
    const currentCount = this.filteredShortcutsCount;

    return html`
      <md-dialog
        ?open=${this.open}
        @closed=${this.closeDialog}
        @cancel=${this.closeDialog}
        @keydown=${this.handleDialogKeyDown}
        aria-label="Keyboard Shortcuts">
        <div slot="headline" class="dialog-headline">
          <div class="headline-left">
            <span class="dialog-title">Keyboard shortcuts</span>
            <span class="shortcut-count-badge"
              >${this.searchQuery.trim()
                ? `${currentCount} ${currentCount === 1 ? 'shortcut' : 'shortcuts'}`
                : `${totalCount} shortcuts`}</span
            >
          </div>
          <div class="headline-right">
            <div class="search-box">
              ${searchIcon}
              <input
                class="search-input"
                type="text"
                placeholder="Search shortcuts..."
                .value=${this.searchQuery}
                @input=${this.handleSearchInput}
                @keydown=${this.handleSearchKeyDown}
                aria-label="Search shortcuts"
                spellcheck="false"
                autocomplete="off"
                autofocus />
              ${this.searchQuery
                ? html`
                    <button
                      type="button"
                      class="clear-search-btn"
                      @click=${this.clearSearch}
                      aria-label="Clear search">
                      ${clearIcon}
                    </button>
                  `
                : ''}
            </div>
            <md-icon-button
              class="close-btn"
              @click=${this.closeDialog}
              aria-label="Close dialog">
              ${closeXIcon}
            </md-icon-button>
          </div>
        </div>
        <div slot="content" class="dialog-content" tabindex="-1">
          ${categories.length === 0
            ? html`
                <div class="empty-search-state">
                  <div class="empty-search-title">
                    No matching shortcuts found
                  </div>
                  <div class="empty-search-subtext">
                    Try searching for other terms like "zoom", "pan", "drag", or
                    specific keys.
                  </div>
                </div>
              `
            : html`
                <div class="table-card">
                  <div class="table-container">
                    <table
                      class="shortcuts-table"
                      aria-label="Keyboard Shortcuts">
                      <thead>
                        <tr>
                          <th class="col-label">Label</th>
                          <th class="col-keyboard">Keyboard Shortcut</th>
                          <th class="col-mouse">Mouse Control</th>
                        </tr>
                      </thead>
                      <tbody>
                        ${categories.map(
                          (category) => html`
                            <tr class="category-header-row">
                              <td colspan="3">
                                <div
                                  class="category-header category-header-content">
                                  ${this.renderCategoryIcon(category.iconName)}
                                  <span>${category.title}</span>
                                </div>
                              </td>
                            </tr>
                            ${category.items.map(
                              (item) => html`
                                <tr class="shortcut-table-row">
                                  <td class="cell-label action-label">
                                    <div class="label-content">
                                      ${this.renderIcon(item.iconName)}
                                      <span>${item.description}</span>
                                    </div>
                                  </td>
                                  <td class="cell-keyboard">
                                    ${item.keys && item.keys.length > 0
                                      ? html`
                                          <span class="shortcut-keys">
                                            ${item.keys.map(
                                              (key, index) => html`
                                                <kbd>${key}</kbd>${index <
                                                item.keys.length - 1
                                                  ? html`<span
                                                      class="key-separator"
                                                      >${item.separator ||
                                                      ' / '}</span
                                                    >`
                                                  : ''}
                                              `,
                                            )}
                                          </span>
                                        `
                                      : ''}
                                  </td>
                                  <td class="cell-mouse">
                                    ${item.mouseControl
                                      ? html`
                                          <span
                                            class="mouse-control-pill mouse-badge">
                                            <span class="mouse-icon">
                                              ${this.renderMouseIcon(
                                                item.mouseControl.icon,
                                              )}
                                            </span>
                                            <span
                                              >${item.mouseControl.label}</span
                                            >
                                          </span>
                                        `
                                      : ''}
                                  </td>
                                </tr>
                              `,
                            )}
                          `,
                        )}
                      </tbody>
                    </table>
                  </div>
                </div>
              `}
        </div>
      </md-dialog>
    `;
  }
}
