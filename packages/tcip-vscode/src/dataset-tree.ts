/**
 * Dataset TreeView provider — scans workspace for images and shows annotation status.
 */

import * as vscode from "vscode";
import * as path from "path";
import * as fs from "fs";

const IMAGE_EXTS = new Set([".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"]);

export class DatasetTreeProvider implements vscode.TreeDataProvider<DatasetItem> {
  private _onDidChangeTreeData = new vscode.EventEmitter<DatasetItem | undefined>();
  readonly onDidChangeTreeData = this._onDidChangeTreeData.event;
  private watcher: vscode.FileSystemWatcher | undefined;

  constructor(private readonly workspacePath: string) {
    if (workspacePath) {
      this.watcher = vscode.workspace.createFileSystemWatcher(
        new vscode.RelativePattern(workspacePath, "data/{images,labels,predictions}/**"),
      );
      this.watcher.onDidCreate(() => this._onDidChangeTreeData.fire(undefined));
      this.watcher.onDidDelete(() => this._onDidChangeTreeData.fire(undefined));
      this.watcher.onDidChange(() => this._onDidChangeTreeData.fire(undefined));
    }
  }

  refresh(): void {
    this._onDidChangeTreeData.fire(undefined);
  }

  getTreeItem(element: DatasetItem): vscode.TreeItem {
    return element;
  }

  getChildren(element?: DatasetItem): DatasetItem[] {
    if (!this.workspacePath) {
      return [];
    }

    // Top level: scan for subdirectories under data/images/
    const imagesDir = path.join(this.workspacePath, "data", "images");
    if (!fs.existsSync(imagesDir)) {
      return [new DatasetItem("No data/images/ directory found", vscode.TreeItemCollapsibleState.None)];
    }

    if (!element) {
      return this.getTopLevel(imagesDir);
    }

    // Children of a directory
    if (element.isDirectory && element.fullPath) {
      return this.getDirectoryContents(element.fullPath);
    }

    return [];
  }

  private getTopLevel(imagesDir: string): DatasetItem[] {
    const entries = this.readDirSafe(imagesDir);
    const dirs: DatasetItem[] = [];
    const files: DatasetItem[] = [];

    for (const entry of entries) {
      const fullPath = path.join(imagesDir, entry.name);
      if (entry.isDirectory()) {
        const count = this.countImages(fullPath);
        const item = new DatasetItem(
          entry.name,
          vscode.TreeItemCollapsibleState.Collapsed,
        );
        item.isDirectory = true;
        item.fullPath = fullPath;
        item.description = `${count} images`;
        item.iconPath = new vscode.ThemeIcon("folder");
        dirs.push(item);
      } else if (IMAGE_EXTS.has(path.extname(entry.name).toLowerCase())) {
        files.push(this.makeImageItem(fullPath, entry.name));
      }
    }

    return [...dirs, ...files];
  }

  private getDirectoryContents(dirPath: string): DatasetItem[] {
    const entries = this.readDirSafe(dirPath);
    const dirs: DatasetItem[] = [];
    const files: DatasetItem[] = [];

    for (const entry of entries) {
      const fullPath = path.join(dirPath, entry.name);
      if (entry.isDirectory()) {
        const count = this.countImages(fullPath);
        const item = new DatasetItem(
          entry.name,
          vscode.TreeItemCollapsibleState.Collapsed,
        );
        item.isDirectory = true;
        item.fullPath = fullPath;
        item.description = `${count} images`;
        item.iconPath = new vscode.ThemeIcon("folder");
        dirs.push(item);
      } else if (IMAGE_EXTS.has(path.extname(entry.name).toLowerCase())) {
        files.push(this.makeImageItem(fullPath, entry.name));
      }
    }

    return [...dirs, ...files];
  }

  private makeImageItem(fullPath: string, name: string): DatasetItem {
    const stem = path.parse(name).name;
    const status = this.getAnnotationStatus(stem);

    const item = new DatasetItem(name, vscode.TreeItemCollapsibleState.None);
    item.fullPath = fullPath;
    item.contextValue = "imageFile";

    // Status badge
    if (status === "annotated") {
      item.description = "✓";
      item.iconPath = new vscode.ThemeIcon("check", new vscode.ThemeColor("testing.iconPassed"));
    } else if (status === "predicted") {
      item.description = "?";
      item.iconPath = new vscode.ThemeIcon("question", new vscode.ThemeColor("editorWarning.foreground"));
    } else {
      item.description = "✗";
      item.iconPath = new vscode.ThemeIcon("circle-outline", new vscode.ThemeColor("disabledForeground"));
    }

    // Click to open in VS Code's built-in image preview
    item.command = {
      command: "vscode.open",
      title: "Open Image",
      arguments: [vscode.Uri.file(fullPath)],
    };

    return item;
  }

  private getAnnotationStatus(stem: string): "annotated" | "predicted" | "none" {
    // Check for ground truth labels
    const detectLabel = path.join(this.workspacePath, "data", "labels", "detect", `${stem}.txt`);
    const segmentLabel = path.join(this.workspacePath, "data", "labels", "segment", `${stem}.txt`);
    if (fs.existsSync(detectLabel) || fs.existsSync(segmentLabel)) {
      return "annotated";
    }

    // Check for predictions
    const detectPred = path.join(this.workspacePath, "data", "predictions", "detect", `${stem}.txt`);
    const segmentPred = path.join(this.workspacePath, "data", "predictions", "segment", `${stem}.txt`);
    if (fs.existsSync(detectPred) || fs.existsSync(segmentPred)) {
      return "predicted";
    }

    return "none";
  }

  private countImages(dirPath: string): number {
    let count = 0;
    const entries = this.readDirSafe(dirPath);
    for (const entry of entries) {
      if (entry.isDirectory()) {
        count += this.countImages(path.join(dirPath, entry.name));
      } else if (IMAGE_EXTS.has(path.extname(entry.name).toLowerCase())) {
        count++;
      }
    }
    return count;
  }

  private readDirSafe(dirPath: string): fs.Dirent[] {
    try {
      return fs.readdirSync(dirPath, { withFileTypes: true });
    } catch {
      return [];
    }
  }

  dispose(): void {
    this.watcher?.dispose();
  }
}

class DatasetItem extends vscode.TreeItem {
  isDirectory = false;
  fullPath?: string;
}
