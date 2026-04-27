"use strict";
/**
 * Dataset TreeView provider — scans workspace for images and shows annotation status.
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
exports.DatasetTreeProvider = void 0;
const vscode = __importStar(require("vscode"));
const path = __importStar(require("path"));
const fs = __importStar(require("fs"));
const IMAGE_EXTS = new Set([".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"]);
class DatasetTreeProvider {
    workspacePath;
    _onDidChangeTreeData = new vscode.EventEmitter();
    onDidChangeTreeData = this._onDidChangeTreeData.event;
    watcher;
    constructor(workspacePath) {
        this.workspacePath = workspacePath;
        if (workspacePath) {
            this.watcher = vscode.workspace.createFileSystemWatcher(new vscode.RelativePattern(workspacePath, "data/{images,labels,predictions}/**"));
            this.watcher.onDidCreate(() => this._onDidChangeTreeData.fire(undefined));
            this.watcher.onDidDelete(() => this._onDidChangeTreeData.fire(undefined));
            this.watcher.onDidChange(() => this._onDidChangeTreeData.fire(undefined));
        }
    }
    refresh() {
        this._onDidChangeTreeData.fire(undefined);
    }
    getTreeItem(element) {
        return element;
    }
    getChildren(element) {
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
    getTopLevel(imagesDir) {
        const entries = this.readDirSafe(imagesDir);
        const dirs = [];
        const files = [];
        for (const entry of entries) {
            const fullPath = path.join(imagesDir, entry.name);
            if (entry.isDirectory()) {
                const count = this.countImages(fullPath);
                const item = new DatasetItem(entry.name, vscode.TreeItemCollapsibleState.Collapsed);
                item.isDirectory = true;
                item.fullPath = fullPath;
                item.description = `${count} images`;
                item.iconPath = new vscode.ThemeIcon("folder");
                dirs.push(item);
            }
            else if (IMAGE_EXTS.has(path.extname(entry.name).toLowerCase())) {
                files.push(this.makeImageItem(fullPath, entry.name));
            }
        }
        return [...dirs, ...files];
    }
    getDirectoryContents(dirPath) {
        const entries = this.readDirSafe(dirPath);
        const dirs = [];
        const files = [];
        for (const entry of entries) {
            const fullPath = path.join(dirPath, entry.name);
            if (entry.isDirectory()) {
                const count = this.countImages(fullPath);
                const item = new DatasetItem(entry.name, vscode.TreeItemCollapsibleState.Collapsed);
                item.isDirectory = true;
                item.fullPath = fullPath;
                item.description = `${count} images`;
                item.iconPath = new vscode.ThemeIcon("folder");
                dirs.push(item);
            }
            else if (IMAGE_EXTS.has(path.extname(entry.name).toLowerCase())) {
                files.push(this.makeImageItem(fullPath, entry.name));
            }
        }
        return [...dirs, ...files];
    }
    makeImageItem(fullPath, name) {
        const stem = path.parse(name).name;
        const status = this.getAnnotationStatus(stem);
        const item = new DatasetItem(name, vscode.TreeItemCollapsibleState.None);
        item.fullPath = fullPath;
        item.contextValue = "imageFile";
        // Status badge
        if (status === "annotated") {
            item.description = "✓";
            item.iconPath = new vscode.ThemeIcon("check", new vscode.ThemeColor("testing.iconPassed"));
        }
        else if (status === "predicted") {
            item.description = "?";
            item.iconPath = new vscode.ThemeIcon("question", new vscode.ThemeColor("editorWarning.foreground"));
        }
        else {
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
    getAnnotationStatus(stem) {
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
    countImages(dirPath) {
        let count = 0;
        const entries = this.readDirSafe(dirPath);
        for (const entry of entries) {
            if (entry.isDirectory()) {
                count += this.countImages(path.join(dirPath, entry.name));
            }
            else if (IMAGE_EXTS.has(path.extname(entry.name).toLowerCase())) {
                count++;
            }
        }
        return count;
    }
    readDirSafe(dirPath) {
        try {
            return fs.readdirSync(dirPath, { withFileTypes: true });
        }
        catch {
            return [];
        }
    }
    dispose() {
        this.watcher?.dispose();
    }
}
exports.DatasetTreeProvider = DatasetTreeProvider;
class DatasetItem extends vscode.TreeItem {
    isDirectory = false;
    fullPath;
}
//# sourceMappingURL=dataset-tree.js.map