/**
 * Inference Panel — WebviewPanel host for running model inference.
 * Shows model selection, progress, and live result previews.
 */
import * as vscode from "vscode";
import { WebviewPanelProvider } from "./webview-base";
export declare class InferencePanel extends WebviewPanelProvider {
    constructor(extensionUri: vscode.Uri);
    protected getHtmlContent(webview: vscode.Webview, nonce: string, sharedCssUri: vscode.Uri, sharedJsUri: vscode.Uri): string;
    protected handleMessage(msg: {
        type: string;
        [key: string]: unknown;
    }): void;
}
