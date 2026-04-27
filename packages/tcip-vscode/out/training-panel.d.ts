/**
 * Training Panel — WebviewPanel host for the training dashboard.
 * Embeds TensorBoard via iframe and shows live epoch metrics via Chart.js.
 */
import * as vscode from "vscode";
import { WebviewPanelProvider } from "./webview-base";
export declare class TrainingPanel extends WebviewPanelProvider {
    constructor(extensionUri: vscode.Uri);
    protected getHtmlContent(webview: vscode.Webview, nonce: string, sharedCssUri: vscode.Uri, sharedJsUri: vscode.Uri): string;
    protected handleMessage(msg: {
        type: string;
        [key: string]: unknown;
    }): void;
}
