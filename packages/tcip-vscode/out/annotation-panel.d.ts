/**
 * Annotation Panel â€” WebviewPanel host for the interactive annotation canvas.
 * Uses Fabric.js for box/polygon drawing, class selection, undo/redo, zoom/pan.
 */
import * as vscode from "vscode";
import { WebviewPanelProvider } from "./webview-base";
export declare class AnnotationPanel extends WebviewPanelProvider {
    constructor(extensionUri: vscode.Uri);
    protected getHtmlContent(webview: vscode.Webview, nonce: string, sharedCssUri: vscode.Uri, sharedJsUri: vscode.Uri): string;
    /** Current image path (used to derive label paths) */
    private currentImagePath;
    /**
     * Load an image into the canvas, converting the file path to a webview URI.
     * Optionally sets class labels and auto-loads existing annotations.
     */
    loadImage(imagePath: string, labels?: string[]): void;
    /** Clear all annotations from the canvas. */
    clearCanvas(): void;
    /** Highlight specific annotation indices on the canvas. */
    highlightAnnotations(indices: number[]): void;
    protected handleMessage(msg: {
        type: string;
        [key: string]: unknown;
    }): void;
    private handleSamRequest;
}
