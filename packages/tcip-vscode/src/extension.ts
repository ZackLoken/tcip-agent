/**
 * TCIP VS Code Extension
 *
 * Provides webview panels for annotation, review, training, HPO, inference,
 * and results. Works alongside GitHub Copilot and the tcip-pipeline MCP server.
 * Panel data arrives via FileSystemWatcher on .tcip/events/*.json files,
 * written by the MCP server's push_panel_data tool.
 */

import * as vscode from "vscode";
import * as path from "path";
import * as fs from "fs";
import { AnnotationPanel } from "./annotation-panel";
import { ReviewPanel } from "./review-panel";
import { TrainingPanel } from "./training-panel";
import { HpoPanel } from "./hpo-panel";
import { InferencePanel } from "./inference-panel";
import { ResultsPanel } from "./results-panel";
import { DatasetTreeProvider } from "./dataset-tree";
import { WebviewPanelProvider } from "./webview-base";

let outputChannel: vscode.OutputChannel;

// Panels
let annotationPanel: AnnotationPanel;
let reviewPanel: ReviewPanel;
let trainingPanel: TrainingPanel;
let hpoPanel: HpoPanel;
let inferencePanel: InferencePanel;
let resultsPanel: ResultsPanel;

// Status bar
let pipelineStatus: vscode.StatusBarItem;

export function activate(context: vscode.ExtensionContext) {
  outputChannel = vscode.window.createOutputChannel("TCIP");

  // -- Status bar --
  pipelineStatus = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Left, 97);
  loadPipelineState();
  pipelineStatus.show();

  // -- Panels --
  annotationPanel = new AnnotationPanel(context.extensionUri);
  reviewPanel = new ReviewPanel(context.extensionUri);
  trainingPanel = new TrainingPanel(context.extensionUri);
  hpoPanel = new HpoPanel(context.extensionUri);
  inferencePanel = new InferencePanel(context.extensionUri);
  resultsPanel = new ResultsPanel(context.extensionUri);

  // -- Dataset tree --
  const workspacePath = getWorkspacePath();
  const datasetTree = new DatasetTreeProvider(workspacePath ?? "");
  const treeView = vscode.window.createTreeView("tcip-dataset", {
    treeDataProvider: datasetTree,
    showCollapseAll: true,
  });

  // -- .tcip/events/ watcher: routes push_panel_data JSON to panels --
  const panelMap: Record<string, WebviewPanelProvider> = {};
  function setupEventsWatcher(): void {
    if (!workspacePath) { return; }
    panelMap["annotation"] = annotationPanel;
    panelMap["review"] = reviewPanel;
    panelMap["training"] = trainingPanel;
    panelMap["hpo"] = hpoPanel;
    panelMap["inference"] = inferencePanel;
    panelMap["results"] = resultsPanel;

    const pattern = new vscode.RelativePattern(workspacePath, ".tcip/events/**/*.json");
    const eventsWatcher = vscode.workspace.createFileSystemWatcher(pattern);

    const handleEventFile = async (uri: vscode.Uri) => {
      try {
        const raw = await vscode.workspace.fs.readFile(uri);
        const json = JSON.parse(Buffer.from(raw).toString("utf-8")) as {
          panel?: string;
          event_type?: string;
          data?: Record<string, unknown>;
        };
        const panel = panelMap[json.panel ?? ""];
        if (panel && json.event_type) {
          panel.show();
          panel.postMessage({ type: json.event_type, ...json.data });
        }
      } catch {
        outputChannel.appendLine(`[events-watcher] failed to process ${uri.fsPath}`);
      }
    };

    eventsWatcher.onDidCreate(handleEventFile);
    eventsWatcher.onDidChange(handleEventFile);

    context.subscriptions.push(eventsWatcher);
  }
  setupEventsWatcher();

  // -- Register commands --
  const commands: [string, () => void][] = [
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
  context.subscriptions.push(
    vscode.commands.registerCommand("tcip-agent.openInAnnotation", (filePath: string) => {
      annotationPanel.show();
      annotationPanel.postMessage({ type: "load_image", path: filePath });
    }),
    vscode.commands.registerCommand("tcip-agent.openInReview", (filePath: string) => {
      reviewPanel.show();
      reviewPanel.postMessage({ type: "load_image", path: filePath });
    }),
  );

  // -- Subscriptions cleanup --
  context.subscriptions.push(
    treeView,
    pipelineStatus,
    { dispose: () => { annotationPanel.dispose(); reviewPanel.dispose(); trainingPanel.dispose(); hpoPanel.dispose(); inferencePanel.dispose(); resultsPanel.dispose(); } },
  );

  outputChannel.appendLine("TCIP extension activated");
}

export function deactivate() {
  // No agent process to stop - Copilot manages the MCP server lifecycle
}

// -- Pipeline state --

function loadPipelineState(): void {
  const workspacePath = getWorkspacePath();
  if (!workspacePath) {
    pipelineStatus.text = "$(milestone) No project";
    return;
  }
  const statePath = path.join(workspacePath, ".tcip", "pipeline_state.json");
  try {
    if (fs.existsSync(statePath)) {
      const data = JSON.parse(fs.readFileSync(statePath, "utf-8"));
      const phase: string = data.phase ?? "setup";
      const label = phase.replace(/_/g, " ").replace(/\b\w/g, (c: string) => c.toUpperCase());
      pipelineStatus.text = `$(milestone) ${label}`;
      pipelineStatus.tooltip = `Pipeline: ${label}` + (data.crop ? ` * ${data.crop}` : "");
    } else {
      pipelineStatus.text = "$(milestone) New project";
      pipelineStatus.tooltip = "No pipeline state file found";
    }
  } catch {
    pipelineStatus.text = "$(milestone) Unknown";
  }
}

// -- Utilities --

function getWorkspacePath(): string | null {
  const folders = vscode.workspace.workspaceFolders;
  if (folders && folders.length > 0) {
    return folders[0].uri.fsPath;
  }
  return null;
}
