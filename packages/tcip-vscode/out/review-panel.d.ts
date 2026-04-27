/**
 * Review Panel — WebviewPanel host for reviewing predictions against ground truth.
 * Displays TP/FP/FN overlays with IoU/confidence filtering.
 */
import * as vscode from "vscode";
import { WebviewPanelProvider } from "./webview-base";
export declare class ReviewPanel extends WebviewPanelProvider {
    constructor(extensionUri: vscode.Uri);
    protected getHtmlContent(webview: vscode.Webview, nonce: string, sharedCssUri: vscode.Uri, sharedJsUri: vscode.Uri): string;
    /**
     * Show predictions for an image, converting the file path to a webview URI.
     * Sends the image and prediction data to the review webview.
     */
    showPredictions(imagePath: string, predictions: unknown[]): void;
    protected handleMessage(msg: {
        type: string;
        [key: string]: unknown;
    }): void;
    /**
     * Invoke the Python matching engine and return TP/FP/FN detection list to webview.
     * Builds a flat array of {tag, classId, box, confidence, type, polygon?} for the webview.
     */
    private handleRequestMatches;
    /**
     * Save review state (per-image decisions) to review_stats.json next to the images.
     */
    private handleSaveReviewState;
    /**
     * Delete a ground-truth annotation from its label file by index.
     */
    private handleDeleteGt;
}
