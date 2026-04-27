/**
 * TCIP VS Code Extension
 *
 * Provides webview panels for annotation, review, training, HPO, inference,
 * and results. Works alongside GitHub Copilot and the tcip-pipeline MCP server.
 * Panel data arrives via FileSystemWatcher on .tcip/events/*.json files,
 * written by the MCP server's push_panel_data tool.
 */
import * as vscode from "vscode";
export declare function activate(context: vscode.ExtensionContext): void;
export declare function deactivate(): void;
