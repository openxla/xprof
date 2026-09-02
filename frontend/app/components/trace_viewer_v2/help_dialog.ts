import '@material/web/dialog/dialog';
import '@material/web/iconbutton/icon-button';

import type {MdDialog} from '@material/web/dialog/dialog';
import {css, html, LitElement} from 'lit';
import {customElement, property} from 'lit/decorators.js';

const selectIcon = html`<svg
  width="15"
  height="15"
  viewBox="0 0 28 28"
  fill="none"
  xmlns="http://www.w3.org/2000/svg"
  style="vertical-align: -2px; margin-right: 4px; display: inline-block;">
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
  style="vertical-align: -2px; margin-right: 4px; display: inline-block;">
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
  style="vertical-align: -2px; margin-right: 4px; display: inline-block;">
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
  style="vertical-align: -2px; margin-right: 4px; display: inline-block;">
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
    fill="#444746" />
</svg>`;

/**
 * A Web Component (GM3) dialog displaying keyboard shortcuts and controls
 * for Trace Viewer v2.
 */
@customElement('trace-viewer-help-dialog')
export class TraceViewerHelpDialog extends LitElement {
  static override styles = css`
    :host {
      display: inline-block;
    }
    md-dialog {
      --md-dialog-container-color: #ffffff;
      --md-sys-color-primary: #1a73e8;
      --md-sys-color-on-surface: #1f1f1f;
      --md-sys-color-on-surface-variant: #444746;
      --md-dialog-container-min-width: 860px;
      --md-dialog-container-max-width: 940px;
      min-width: 860px;
      max-width: 940px;
    }
    md-icon-button {
      --md-focus-ring-color: transparent;
      --md-icon-button-state-layer-color: transparent;
      --md-icon-button-hover-state-layer-color: transparent;
      --md-icon-button-pressed-state-layer-color: transparent;
      outline: none;
    }
    .dialog-headline {
      display: flex;
      justify-content: space-between;
      align-items: center;
      width: 100%;
      padding-bottom: 8px;
    }
    .dialog-content {
      width: 100%;
      padding: 0px 24px 20px;
      box-sizing: border-box;
      overflow: visible;
      max-height: none;
      outline: none;
    }
    .grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 36px;
    }
    @media (max-width: 768px) {
      .grid {
        grid-template-columns: 1fr;
      }
    }
    .section {
      display: flex;
      flex-direction: column;
      gap: 2px;
    }
    .section-title {
      font-size: 12px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.5px;
      color: #444746;
      margin-top: 8px;
      margin-bottom: 4px;
      padding-bottom: 3px;
      border-bottom: 1px solid #e0e0e0;
    }
    .section:first-child .section-title {
      margin-top: 0;
    }
    .shortcut-row {
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-size: 14px;
      line-height: 22px;
      color: #1f1f1f;
      white-space: nowrap;
    }
    .shortcut-keys {
      display: flex;
      gap: 3px;
      align-items: center;
      white-space: nowrap;
    }
    kbd {
      background-color: #f1f3f4;
      border: 1px solid #dadce0;
      border-radius: 4px;
      color: #3c4043;
      font-family: Roboto, 'Google Sans', monospace;
      font-size: 12px;
      font-weight: 500;
      padding: 1px 6px;
      min-width: 16px;
      text-align: center;
      box-shadow: none;
    }
    .action-label {
      color: #444746;
      font-size: 14px;
      white-space: nowrap;
      display: inline-flex;
      align-items: center;
    }
    .close-btn {
      margin-top: -8px;
      margin-right: -12px;
    }
  `;

  @property({type: Boolean}) open = false;

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
    this.open = true;
    await this.updateComplete;
    const dialog = this.shadowRoot?.querySelector(
      'md-dialog',
    ) as MdDialog | null;
    await dialog?.show();
    await new Promise((resolve) => {
      setTimeout(resolve, 50);
    });
    if (document.activeElement instanceof HTMLElement) {
      document.activeElement.blur();
    }
  }

  async closeDialog() {
    const dialog = this.shadowRoot?.querySelector(
      'md-dialog',
    ) as MdDialog | null;
    await dialog?.close();
    this.open = false;
  }

  override render() {
    return html`
      <md-dialog
        ?open=${this.open}
        @closed=${this.closeDialog}
        @cancel=${this.closeDialog}
        aria-label="Keyboard Shortcuts">
        <div slot="headline" class="dialog-headline">
          <span>Keyboard Shortcuts</span>
          <md-icon-button
            class="close-btn"
            tabindex="-1"
            @click=${this.closeDialog}
            aria-label="Close dialog">
            ${closeXIcon}
          </md-icon-button>
        </div>
        <div slot="content" class="dialog-content" tabindex="-1">
          <div class="grid">
            <div class="section">
              <div class="section-title">Navigation</div>
              <div class="shortcut-row">
                <span class="action-label">Zoom in / out</span>
                <span class="shortcut-keys"><kbd>W</kbd> / <kbd>S</kbd></span>
              </div>
              <div class="shortcut-row">
                <span class="action-label">Pan left / right</span>
                <span class="shortcut-keys"><kbd>A</kbd> / <kbd>D</kbd></span>
              </div>
              <div class="shortcut-row">
                <span class="action-label">Select prev / next event</span>
                <span class="shortcut-keys"><kbd>←</kbd> / <kbd>→</kbd></span>
              </div>
              <div class="shortcut-row">
                <span class="action-label">Zoom to fit selection</span>
                <span class="shortcut-keys"><kbd>F</kbd></span>
              </div>
              <div class="shortcut-row">
                <span class="action-label">Reset zoom and pan</span>
                <span class="shortcut-keys"><kbd>Z</kbd> / <kbd>0</kbd></span>
              </div>

              <div class="section-title">Mouse Modes</div>
              <div class="shortcut-row">
                <span class="action-label">${selectIcon} Select Mode</span>
                <span class="shortcut-keys"><kbd>1</kbd></span>
              </div>
              <div class="shortcut-row">
                <span class="action-label">${panIcon} Pan Mode</span>
                <span class="shortcut-keys"><kbd>2</kbd></span>
              </div>
              <div class="shortcut-row">
                <span class="action-label">${zoomIcon} Zoom Mode</span>
                <span class="shortcut-keys"><kbd>3</kbd></span>
              </div>
              <div class="shortcut-row">
                <span class="action-label">${measureIcon} Measure Mode</span>
                <span class="shortcut-keys"><kbd>4</kbd></span>
              </div>
            </div>

            <div class="section">
              <div class="section-title">Mouse Controls</div>
              <div class="shortcut-row">
                <span class="action-label">Select event</span>
                <span class="shortcut-keys"><kbd>Click</kbd></span>
              </div>
              <div class="shortcut-row">
                <span class="action-label">Zoom in / out</span>
                <span class="shortcut-keys"><kbd>Scroll Wheel</kbd></span>
              </div>
              <div class="shortcut-row">
                <span class="action-label">Box select events</span>
                <span class="shortcut-keys"><kbd>Drag</kbd> (Mode 1)</span>
              </div>
              <div class="shortcut-row">
                <span class="action-label">Pan timeline</span>
                <span class="shortcut-keys"><kbd>Drag</kbd> (Mode 2)</span>
              </div>
              <div class="shortcut-row">
                <span class="action-label">Vertical zoom</span>
                <span class="shortcut-keys"><kbd>Drag</kbd> (Mode 3)</span>
              </div>
              <div class="shortcut-row">
                <span class="action-label">Measure time range</span>
                <span class="shortcut-keys"><kbd>Drag</kbd> (Mode 4)</span>
              </div>
              <div class="shortcut-row">
                <span class="action-label">Add selection / measure</span>
                <span class="shortcut-keys"
                  ><kbd>Shift</kbd> + <kbd>Click/Drag</kbd></span
                >
              </div>

              <div class="section-title">General</div>
              <div class="shortcut-row">
                <span class="action-label">Search events</span>
                <span class="shortcut-keys"><kbd>/</kbd></span>
              </div>
              <div class="shortcut-row">
                <span class="action-label">Next / prev search result</span>
                <span class="shortcut-keys"
                  ><kbd>Enter</kbd> / <kbd>Shift+Enter</kbd></span
                >
              </div>
              <div class="shortcut-row">
                <span class="action-label">Bookmark selection</span>
                <span class="shortcut-keys"><kbd>M</kbd></span>
              </div>
              <div class="shortcut-row">
                <span class="action-label">Open Settings</span>
                <span class="shortcut-keys"><kbd>;</kbd></span>
              </div>
              <div class="shortcut-row">
                <span class="action-label">Open Help menu</span>
                <span class="shortcut-keys"><kbd>?</kbd></span>
              </div>
            </div>
          </div>
        </div>
      </md-dialog>
    `;
  }
}
