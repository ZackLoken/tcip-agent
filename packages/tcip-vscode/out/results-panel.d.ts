/**
 * Results Panel — WebviewPanel host for displaying evaluation results.
 * Shows overall + per-class metrics, worst predictions, CSV preview.
 */
import * as vscode from "vscode";
import { WebviewPanelProvider } from "./webview-base";
export declare class ResultsPanel extends WebviewPanelProvider {
    constructor(extensionUri: vscode.Uri);
    protected getHtmlContent(webview: vscode.Webview, nonce: string, sharedCssUri: vscode.Uri, sharedJsUri: vscode.Uri): string;
    protected handleMessage(msg: {
        type: string;
        [key: string]: unknown;
    }): void;
}
