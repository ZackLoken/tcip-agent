/**
 * Dataset TreeView provider — scans workspace for images and shows annotation status.
 */
import * as vscode from "vscode";
export declare class DatasetTreeProvider implements vscode.TreeDataProvider<DatasetItem> {
    private readonly workspacePath;
    private _onDidChangeTreeData;
    readonly onDidChangeTreeData: vscode.Event<DatasetItem | undefined>;
    private watcher;
    constructor(workspacePath: string);
    refresh(): void;
    getTreeItem(element: DatasetItem): vscode.TreeItem;
    getChildren(element?: DatasetItem): DatasetItem[];
    private getTopLevel;
    private getDirectoryContents;
    private makeImageItem;
    private getAnnotationStatus;
    private countImages;
    private readDirSafe;
    dispose(): void;
}
declare class DatasetItem extends vscode.TreeItem {
    isDirectory: boolean;
    fullPath?: string;
}
export {};
