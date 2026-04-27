/**
 * HPO Panel — WebviewPanel host for hyperparameter optimization.
 * Embeds Ray Tune dashboard and shows trials table.
 */
import * as vscode from "vscode";
import { WebviewPanelProvider } from "./webview-base";
export declare class HpoPanel extends WebviewPanelProvider {
    constructor(extensionUri: vscode.Uri);
    protected getHtmlContent(webview: vscode.Webview, nonce: string, sharedCssUri: vscode.Uri, sharedJsUri: vscode.Uri): string;
    protected handleMessage(msg: {
        type: string;
        [key: string]: unknown;
    }): void;
}
