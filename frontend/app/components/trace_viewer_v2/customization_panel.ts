import '@material/web/button/filled-button';
import '@material/web/button/text-button';
import '@material/web/dialog/dialog';
import '@material/web/slider/slider';

import type {MdDialog} from '@material/web/dialog/dialog';
import {css, html, LitElement} from 'lit';
import {customElement, property, state} from 'lit/decorators.js';

import {getFeatureFlags} from './feature_flags';
import {getActiveWasmModule} from './main';

/**
 * A Web Component (GM3) panel for customizing Trace Viewer v2 navigation speeds
 * and preferences, gated by the 'enable_customization' feature flag.
 */
@customElement('trace-viewer-customization-panel')
export class TraceViewerCustomizationPanel extends LitElement {
  static override styles = css`
    :host {
      display: inline-block;
    }
    .slider-container {
      margin-bottom: 24px;
    }
    .slider-container:last-child {
      margin-bottom: 8px;
    }
    .slider-label {
      font-weight: 500;
      font-size: 14px;
      margin-bottom: 6px;
      display: block;
      color: var(--md-sys-color--on-surface, #1f1f1f);
    }
    md-slider {
      width: 100%;
    }
    .dialog-content {
      min-width: 380px;
      padding: 20px 24px;
      box-sizing: border-box;
    }
    .section-title {
      font-size: 16px;
      font-weight: 600;
      color: var(--md-sys-color--primary, #0b57d0);
      margin-bottom: 16px;
      padding-bottom: 4px;
      border-bottom: 1px solid var(--md-sys-color--outline-variant, #cacaca);
    }
    .actions {
      display: flex;
      justify-content: space-between;
      width: 100%;
      gap: 12px;
    }
  `;

  @property({type: Boolean}) enabled = false;

  @state() private panningSpeed = 1000;
  @state() private zoomSpeed = 1.5;
  @state() private mouseWheelZoomSpeed = 0.2;

  private checkInterval?: number;

  override connectedCallback() {
    super.connectedCallback();
    this.enabled = this.checkFeatureFlag();
    this.checkInterval = window.setInterval(() => {
      const newEnabled = this.checkFeatureFlag();
      if (this.enabled !== newEnabled) {
        this.enabled = newEnabled;
      }
    }, 500);
  }

  override disconnectedCallback() {
    super.disconnectedCallback();
    if (this.checkInterval !== undefined) {
      window.clearInterval(this.checkInterval);
    }
  }

  private checkFeatureFlag(): boolean {
    const flagName = 'enable_customization';
    if (window.getFeatureFlag !== undefined) {
      return window.getFeatureFlag(flagName);
    }
    try {
      const value = window.localStorage.getItem(`xprof_ff_${flagName}`);
      if (value !== null) {
        return value === 'true';
      }
    } catch (e) {
      console.warn('Failed to read feature flag from localStorage:', e);
    }
    const flag = getFeatureFlags().find((f) => f.id === flagName);
    return flag?.default ?? false;
  }

  private get traceviewerModule() {
    return getActiveWasmModule();
  }

  onPanningSpeedChange(event: Event) {
    const target = event.target as HTMLInputElement;
    this.panningSpeed = Number(target.value);
    this.traceviewerModule?.SetPanningSpeed?.(this.panningSpeed);
  }

  onZoomSpeedChange(event: Event) {
    const target = event.target as HTMLInputElement;
    this.zoomSpeed = Number(target.value);
    this.traceviewerModule?.SetZoomSpeed?.(this.zoomSpeed);
  }

  onMouseWheelZoomSpeedChange(event: Event) {
    const target = event.target as HTMLInputElement;
    this.mouseWheelZoomSpeed = Number(target.value);
    this.traceviewerModule?.SetMouseWheelZoomSpeed?.(this.mouseWheelZoomSpeed);
  }

  resetToDefault() {
    this.panningSpeed = 1000;
    this.zoomSpeed = 1.5;
    this.mouseWheelZoomSpeed = 0.2;
    this.traceviewerModule?.SetPanningSpeed?.(this.panningSpeed);
    this.traceviewerModule?.SetZoomSpeed?.(this.zoomSpeed);
    this.traceviewerModule?.SetMouseWheelZoomSpeed?.(this.mouseWheelZoomSpeed);
  }

  async openDialog() {
    this.enabled = true;
    await this.updateComplete;
    const dialog = this.shadowRoot?.querySelector(
      'md-dialog',
    ) as MdDialog | null;
    await dialog?.show();
  }

  async closeDialog() {
    const dialog = this.shadowRoot?.querySelector(
      'md-dialog',
    ) as MdDialog | null;
    await dialog?.close();
  }

  override render() {
    if (!this.enabled && !this.checkFeatureFlag()) {
      return html``;
    }
    return html`
      <md-dialog aria-label="Preferences">
        <div slot="headline">Preferences</div>
        <div slot="content" class="dialog-content">
          <div class="section-title">Navigation Speed</div>
          <div class="slider-container">
            <span class="slider-label"
              >Panning Speed: ${this.panningSpeed} px/s</span
            >
            <md-slider
              min="100"
              max="5000"
              step="100"
              value=${this.panningSpeed}
              @input=${this.onPanningSpeedChange}
              @change=${this.onPanningSpeedChange}></md-slider>
          </div>
          <div class="slider-container">
            <span class="slider-label">Zooming Speed: ${this.zoomSpeed}</span>
            <md-slider
              min="0.1"
              max="10.0"
              step="0.1"
              value=${this.zoomSpeed}
              @input=${this.onZoomSpeedChange}
              @change=${this.onZoomSpeedChange}></md-slider>
          </div>
          <div class="slider-container">
            <span class="slider-label"
              >Wheel Zoom Speed: ${this.mouseWheelZoomSpeed}</span
            >
            <md-slider
              min="0.01"
              max="1.0"
              step="0.01"
              value=${this.mouseWheelZoomSpeed}
              @input=${this.onMouseWheelZoomSpeedChange}
              @change=${this.onMouseWheelZoomSpeedChange}></md-slider>
          </div>
        </div>
        <div slot="actions" class="actions">
          <md-text-button @click=${this.resetToDefault}
            >Reset to Default</md-text-button
          >
          <md-filled-button @click=${this.closeDialog}>Close</md-filled-button>
        </div>
      </md-dialog>
    `;
  }
}
