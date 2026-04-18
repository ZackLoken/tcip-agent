/**
 * Shared WebviewPanel provider base class.
 * Handles panel lifecycle, CSP, theme-aware CSS, and bidirectional messaging.
 */

import * as vscode from "vscode";

/** Cryptographic nonce for Content Security Policy. */
export function getNonce(): string {
  const chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
  let nonce = "";
  for (let i = 0; i < 32; i++) {
    nonce += chars.charAt(Math.floor(Math.random() * chars.length));
  }
  return nonce;
}

/** Resolve a local file URI that the webview can load. */
export function getUri(
  webview: vscode.Webview,
  extensionUri: vscode.Uri,
  ...pathSegments: string[]
): vscode.Uri {
  return webview.asWebviewUri(vscode.Uri.joinPath(extensionUri, ...pathSegments));
}

/** Message from webview → extension host. */
export interface WebviewMessage {
  type: string;
  [key: string]: unknown;
}

/** Message from extension host → webview. */
export interface HostMessage {
  type: string;
  [key: string]: unknown;
}

/**
 * Base class for webview panel providers.
 * Subclasses override getHtmlContent() and handleMessage().
 */
export abstract class WebviewPanelProvider {
  protected panel: vscode.WebviewPanel | null = null;
  protected disposables: vscode.Disposable[] = [];

  constructor(
    protected readonly extensionUri: vscode.Uri,
    protected readonly viewType: string,
    protected readonly title: string,
  ) {}

  /** Show the panel (create if needed, reveal if exists). */
  show(column: vscode.ViewColumn = vscode.ViewColumn.Beside): void {
    if (this.panel) {
      this.panel.reveal(column);
      return;
    }

    this.panel = vscode.window.createWebviewPanel(
      this.viewType,
      this.title,
      column,
      {
        enableScripts: true,
        retainContextWhenHidden: true,
        localResourceRoots: [
          vscode.Uri.joinPath(this.extensionUri, "media"),
          vscode.Uri.joinPath(this.extensionUri, "node_modules"),
        ],
      },
    );

    this.panel.webview.html = this.getHtml(this.panel.webview);

    this.panel.webview.onDidReceiveMessage(
      (msg: WebviewMessage) => this.handleMessage(msg),
      null,
      this.disposables,
    );

    this.panel.onDidDispose(
      () => {
        this.panel = null;
        this.disposables.forEach((d) => d.dispose());
        this.disposables = [];
        this.onDispose();
      },
      null,
      this.disposables,
    );
  }

  /** Send a message to the webview. */
  postMessage(msg: HostMessage): void {
    this.panel?.webview.postMessage(msg);
  }

  /** Whether the panel is currently visible. */
  get isVisible(): boolean {
    return this.panel?.visible ?? false;
  }

  /** Dispose the panel programmatically. */
  dispose(): void {
    this.panel?.dispose();
  }

  // ── Subclass hooks ──

  /** Return the full HTML for the webview. Use getHtml() wrapper instead. */
  protected abstract getHtmlContent(
    webview: vscode.Webview,
    nonce: string,
    sharedCssUri: vscode.Uri,
    sharedJsUri: vscode.Uri,
  ): string;

  /** Handle a message from the webview. */
  protected abstract handleMessage(msg: WebviewMessage): void;

  /** Called when the panel is disposed. Override for cleanup. */
  protected onDispose(): void {}

  // ── Private ──

  private getHtml(webview: vscode.Webview): string {
    const nonce = getNonce();
    const sharedCssUri = getUri(webview, this.extensionUri, "media", "shared.css");
    const sharedJsUri = getUri(webview, this.extensionUri, "media", "shared.js");

    return this.getHtmlContent(webview, nonce, sharedCssUri, sharedJsUri);
  }
}
