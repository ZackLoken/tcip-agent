/**
 * Shared WebviewPanel provider base class.
 * Handles panel lifecycle, CSP, theme-aware CSS, and bidirectional messaging.
 */
import * as vscode from "vscode";
/** Cryptographic nonce for Content Security Policy. */
export declare function getNonce(): string;
/** Resolve a local file URI that the webview can load. */
export declare function getUri(webview: vscode.Webview, extensionUri: vscode.Uri, ...pathSegments: string[]): vscode.Uri;
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
export declare abstract class WebviewPanelProvider {
    protected readonly extensionUri: vscode.Uri;
    protected readonly viewType: string;
    protected readonly title: string;
    protected panel: vscode.WebviewPanel | null;
    protected disposables: vscode.Disposable[];
    constructor(extensionUri: vscode.Uri, viewType: string, title: string);
    /** Show the panel (create if needed, reveal if exists). */
    show(column?: vscode.ViewColumn): void;
    /** Send a message to the webview. */
    postMessage(msg: HostMessage): void;
    /** Whether the panel is currently visible. */
    get isVisible(): boolean;
    /** Dispose the panel programmatically. */
    dispose(): void;
    /** Return the full HTML for the webview. Use getHtml() wrapper instead. */
    protected abstract getHtmlContent(webview: vscode.Webview, nonce: string, sharedCssUri: vscode.Uri, sharedJsUri: vscode.Uri): string;
    /** Handle a message from the webview. */
    protected abstract handleMessage(msg: WebviewMessage): void;
    /** Called when the panel is disposed. Override for cleanup. */
    protected onDispose(): void;
    private getHtml;
}
