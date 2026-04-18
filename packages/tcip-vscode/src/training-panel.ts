/**
 * Training Panel — WebviewPanel host for the training dashboard.
 * Embeds TensorBoard via iframe and shows live epoch metrics via Chart.js.
 */

import * as vscode from "vscode";
import { WebviewPanelProvider, getUri, getNonce } from "./webview-base";

export class TrainingPanel extends WebviewPanelProvider {
  constructor(extensionUri: vscode.Uri) {
    super(extensionUri, "tcip-training", "TCIP: Training");
  }

  protected getHtmlContent(
    webview: vscode.Webview,
    nonce: string,
    sharedCssUri: vscode.Uri,
    sharedJsUri: vscode.Uri,
  ): string {
    const trainingJsUri = getUri(webview, this.extensionUri, "media", "training.js");
    const chartJsUri = getUri(webview, this.extensionUri, "node_modules", "chart.js", "dist", "chart.umd.js");
    const cspSource = webview.cspSource;

    return /* html */ `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta http-equiv="Content-Security-Policy"
    content="default-src 'none'; frame-src http://localhost:* https://localhost:*; img-src ${cspSource} data:; style-src ${cspSource} 'nonce-${nonce}'; script-src 'nonce-${nonce}';">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <link rel="stylesheet" href="${sharedCssUri}">
  <style nonce="${nonce}">
    .metrics-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 12px;
      padding: 12px;
    }
    .metric-card {
      background: var(--vscode-editorWidget-background);
      border: 1px solid var(--tcip-border);
      border-radius: 6px;
      padding: 12px;
    }
    .metric-card .label {
      font-size: 11px;
      color: var(--vscode-descriptionForeground);
      margin-bottom: 4px;
    }
    .metric-card .value {
      font-size: 24px;
      font-weight: 700;
    }
    .chart-container {
      position: relative;
      height: 250px;
      padding: 12px;
    }
    .chart-block {
      margin-bottom: 8px;
    }
    .tensorboard-frame {
      width: 100%;
      height: 400px;
      border: none;
      border-top: 1px solid var(--tcip-border);
    }
    .section-header {
      font-size: 12px;
      font-weight: 600;
      padding: 8px 12px 4px;
      color: var(--vscode-descriptionForeground);
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }
  </style>
</head>
<body>
  <div class="panel-container">
    <div class="toolbar">
      <span style="font-weight:600">Training Dashboard</span>
      <div style="flex:1"></div>
      <span class="status-text" id="run-id">No active run</span>
      <div class="separator" style="width:1px;height:20px;background:var(--tcip-border)"></div>
      <button id="btn-pause">⏸ Pause</button>
      <button id="btn-stop" class="danger">⏹ Stop</button>
    </div>
    <div class="content" id="training-content">
      <div id="empty-state" class="empty-state">
        <div class="icon">📊</div>
        <div>No training in progress</div>
        <div class="status-text">Use Copilot to start a training run</div>
      </div>
      <div id="live-section" style="display:none">
        <div class="section-header">Live Metrics</div>
        <div class="metrics-grid" id="metrics-grid"></div>
        <div id="charts-container"></div>
      </div>
      <div id="tensorboard-section" style="display:none">
        <div class="section-header">TensorBoard</div>
        <iframe id="tensorboard-frame" class="tensorboard-frame" sandbox="allow-scripts allow-same-origin"></iframe>
      </div>
    </div>
  </div>

  <script nonce="${nonce}" src="${sharedJsUri}"></script>
  <script nonce="${nonce}" src="${chartJsUri}"></script>
  <script nonce="${nonce}" src="${trainingJsUri}"></script>
</body>
</html>`;
  }

  protected handleMessage(msg: { type: string; [key: string]: unknown }): void {
    switch (msg.type) {
      case "pause_training":
      case "stop_training":
        // Forward to agent bridge — handled by extension.ts event routing
        break;
    }
  }
}
