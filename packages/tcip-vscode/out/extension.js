"use strict";
/**
 * TCIP VS Code Extension
 *
 * Provides webview panels for annotation, review, training, HPO, inference,
 * and results. Works alongside GitHub Copilot and the tcip-pipeline MCP server.
 * Panel data arrives via FileSystemWatcher on .tcip/events/*.json files,
 * written by the MCP server's push_panel_data tool.
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
exports.activate = activate;
exports.deactivate = deactivate;
const vscode = __importStar(require("vscode"));
const path = __importStar(require("path"));
const fs = __importStar(require("fs"));
const annotation_panel_1 = require("./annotation-panel");
const review_panel_1 = require("./review-panel");
const training_panel_1 = require("./training-panel");
const hpo_panel_1 = require("./hpo-panel");
const inference_panel_1 = require("./inference-panel");
const results_panel_1 = require("./results-panel");
const dataset_tree_1 = require("./dataset-tree");
let outputChannel;
// Panels
let annotationPanel;
let reviewPanel;
let trainingPanel;
let hpoPanel;
let inferencePanel;
let resultsPanel;
// Status bar
let pipelineStatus;
function activate(context) {
    outputChannel = vscode.window.createOutputChannel("TCIP");
    // -- Status bar --
    pipelineStatus = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Left, 97);
    loadPipelineState();
    pipelineStatus.show();
    // -- Panels --
    annotationPanel = new annotation_panel_1.AnnotationPanel(context.extensionUri);
    reviewPanel = new review_panel_1.ReviewPanel(context.extensionUri);
    trainingPanel = new training_panel_1.TrainingPanel(context.extensionUri);
    hpoPanel = new hpo_panel_1.HpoPanel(context.extensionUri);
    inferencePanel = new inference_panel_1.InferencePanel(context.extensionUri);
    resultsPanel = new results_panel_1.ResultsPanel(context.extensionUri);
    // -- Dataset tree --
    const workspacePath = getWorkspacePath();
    const datasetTree = new dataset_tree_1.DatasetTreeProvider(workspacePath ?? "");
    const treeView = vscode.window.createTreeView("tcip-dataset", {
        treeDataProvider: datasetTree,
        showCollapseAll: true,
    });
    // -- .tcip/events/ watcher: routes push_panel_data JSON to panels --
    const panelMap = {};
    function setupEventsWatcher() {
        if (!workspacePath) {
            return;
        }
        panelMap["annotation"] = annotationPanel;
        panelMap["review"] = reviewPanel;
        panelMap["training"] = trainingPanel;
        panelMap["hpo"] = hpoPanel;
        panelMap["inference"] = inferencePanel;
        panelMap["results"] = resultsPanel;
        const pattern = new vscode.RelativePattern(workspacePath, ".tcip/events/**/*.json");
        const eventsWatcher = vscode.workspace.createFileSystemWatcher(pattern);
        const handleEventFile = async (uri) => {
            try {
                const raw = await vscode.workspace.fs.readFile(uri);
                const json = JSON.parse(Buffer.from(raw).toString("utf-8"));
                const panel = panelMap[json.panel ?? ""];
                if (panel && json.event_type) {
                    panel.show();
                    panel.postMessage({ type: json.event_type, ...json.data });
                }
            }
            catch {
                outputChannel.appendLine(`[events-watcher] failed to process ${uri.fsPath}`);
            }
        };
        eventsWatcher.onDidCreate(handleEventFile);
        eventsWatcher.onDidChange(handleEventFile);
        context.subscriptions.push(eventsWatcher);
    }
    setupEventsWatcher();
    // -- Register commands --
    const commands = [
        ["tcip-agent.openAnnotation", () => annotationPanel.show()],
        ["tcip-agent.openReview", () => reviewPanel.show()],
        ["tcip-agent.openTraining", () => trainingPanel.show()],
        ["tcip-agent.openHpo", () => hpoPanel.show()],
        ["tcip-agent.openInference", () => inferencePanel.show()],
        ["tcip-agent.openResults", () => resultsPanel.show()],
        ["tcip-agent.refreshDataset", () => datasetTree.refresh()],
    ];
    for (const [id, handler] of commands) {
        context.subscriptions.push(vscode.commands.registerCommand(id, handler));
    }
    // Dataset tree item click -> open in annotation/review panel
    context.subscriptions.push(vscode.commands.registerCommand("tcip-agent.openInAnnotation", (filePath) => {
        annotationPanel.show();
        annotationPanel.postMessage({ type: "load_image", path: filePath });
    }), vscode.commands.registerCommand("tcip-agent.openInReview", (filePath) => {
        reviewPanel.show();
        reviewPanel.postMessage({ type: "load_image", path: filePath });
    }));
    // -- Subscriptions cleanup --
    context.subscriptions.push(treeView, pipelineStatus, { dispose: () => { annotationPanel.dispose(); reviewPanel.dispose(); trainingPanel.dispose(); hpoPanel.dispose(); inferencePanel.dispose(); resultsPanel.dispose(); } });
    outputChannel.appendLine("TCIP extension activated");
}
function deactivate() {
    // No agent process to stop - Copilot manages the MCP server lifecycle
}
// -- Pipeline state --
function loadPipelineState() {
    const workspacePath = getWorkspacePath();
    if (!workspacePath) {
        pipelineStatus.text = "$(milestone) No project";
        return;
    }
    const statePath = path.join(workspacePath, ".tcip", "pipeline_state.json");
    try {
        if (fs.existsSync(statePath)) {
            const data = JSON.parse(fs.readFileSync(statePath, "utf-8"));
            const phase = data.phase ?? "setup";
            const label = phase.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
            pipelineStatus.text = `$(milestone) ${label}`;
            pipelineStatus.tooltip = `Pipeline: ${label}` + (data.crop ? ` * ${data.crop}` : "");
        }
        else {
            pipelineStatus.text = "$(milestone) New project";
            pipelineStatus.tooltip = "No pipeline state file found";
        }
    }
    catch {
        pipelineStatus.text = "$(milestone) Unknown";
    }
}
// -- Utilities --
function getWorkspacePath() {
    const folders = vscode.workspace.workspaceFolders;
    if (folders && folders.length > 0) {
        return folders[0].uri.fsPath;
    }
    return null;
}
//# sourceMappingURL=extension.js.map