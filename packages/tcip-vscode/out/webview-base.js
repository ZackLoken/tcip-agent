"use strict";
/**
 * Shared WebviewPanel provider base class.
 * Handles panel lifecycle, CSP, theme-aware CSS, and bidirectional messaging.
 */
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || (function () {
    var ownKeys = function(o) {
        ownKeys = Object.getOwnPropertyNames || function (o) {
            var ar = [];
            for (var k in o) if (Object.prototype.hasOwnProperty.call(o, k)) ar[ar.length] = k;
            return ar;
        };
        return ownKeys(o);
    };
    return function (mod) {
        if (mod && mod.__esModule) return mod;
        var result = {};
        if (mod != null) for (var k = ownKeys(mod), i = 0; i < k.length; i++) if (k[i] !== "default") __createBinding(result, mod, k[i]);
        __setModuleDefault(result, mod);
        return result;
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
exports.WebviewPanelProvider = void 0;
exports.getNonce = getNonce;
exports.getUri = getUri;
const vscode = __importStar(require("vscode"));
/** Cryptographic nonce for Content Security Policy. */
function getNonce() {
    const chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
    let nonce = "";
    for (let i = 0; i < 32; i++) {
        nonce += chars.charAt(Math.floor(Math.random() * chars.length));
    }
    return nonce;
}
/** Resolve a local file URI that the webview can load. */
function getUri(webview, extensionUri, ...pathSegments) {
    return webview.asWebviewUri(vscode.Uri.joinPath(extensionUri, ...pathSegments));
}
/**
 * Base class for webview panel providers.
 * Subclasses override getHtmlContent() and handleMessage().
 */
class WebviewPanelProvider {
    extensionUri;
    viewType;
    title;
    panel = null;
    disposables = [];
    constructor(extensionUri, viewType, title) {
        this.extensionUri = extensionUri;
        this.viewType = viewType;
        this.title = title;
    }
    /** Show the panel (create if needed, reveal if exists). */
    show(column = vscode.ViewColumn.Beside) {
        if (this.panel) {
            this.panel.reveal(column);
            return;
        }
        this.panel = vscode.window.createWebviewPanel(this.viewType, this.title, column, {
            enableScripts: true,
            retainContextWhenHidden: true,
            localResourceRoots: [
                vscode.Uri.joinPath(this.extensionUri, "media"),
                vscode.Uri.joinPath(this.extensionUri, "node_modules"),
            ],
        });
        this.panel.webview.html = this.getHtml(this.panel.webview);
        this.panel.webview.onDidReceiveMessage((msg) => this.handleMessage(msg), null, this.disposables);
        this.panel.onDidDispose(() => {
            this.panel = null;
            this.disposables.forEach((d) => d.dispose());
            this.disposables = [];
            this.onDispose();
        }, null, this.disposables);
    }
    /** Send a message to the webview. */
    postMessage(msg) {
        this.panel?.webview.postMessage(msg);
    }
    /** Whether the panel is currently visible. */
    get isVisible() {
        return this.panel?.visible ?? false;
    }
    /** Dispose the panel programmatically. */
    dispose() {
        this.panel?.dispose();
    }
    /** Called when the panel is disposed. Override for cleanup. */
    onDispose() { }
    // ── Private ──
    getHtml(webview) {
        const nonce = getNonce();
        const sharedCssUri = getUri(webview, this.extensionUri, "media", "shared.css");
        const sharedJsUri = getUri(webview, this.extensionUri, "media", "shared.js");
        return this.getHtmlContent(webview, nonce, sharedCssUri, sharedJsUri);
    }
}
exports.WebviewPanelProvider = WebviewPanelProvider;
//# sourceMappingURL=webview-base.js.map