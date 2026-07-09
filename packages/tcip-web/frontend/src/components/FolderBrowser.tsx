/**
 * A real folder picker. The backend (running on the same machine as the browser) lists
 * directories on disk via /api/fs/list; the user clicks a folder to open it and "Use this
 * folder" to pick the one they're in. This is how a local web app gets an absolute path —
 * `<input webkitdirectory>` deliberately hides it. Browsing is confined by TCIP_IMAGE_ROOTS
 * when set (see routes/fs.py).
 */

import { useEffect, useState } from "react";

import { api, type FsListing } from "@/api/client";

export function FolderBrowser({
  title,
  initialPath,
  onSelect,
  onClose,
}: {
  title: string;
  initialPath: string;
  onSelect: (path: string) => void;
  onClose: () => void;
}) {
  const [path, setPath] = useState(initialPath);
  const [listing, setListing] = useState<FsListing | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    api.fs
      .list(path || undefined)
      .then((r) => {
        if (!cancelled) setListing(r);
      })
      .catch((e) => {
        if (cancelled) return;
        setListing(null);
        setError(String(e));
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [path]);

  const currentPath = listing?.path ?? "";
  const atTop = !currentPath;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4">
      <div className="tcip-panel rounded-lg w-full max-w-lg h-[70vh] flex flex-col p-4 gap-3">
        <div className="flex items-center gap-2">
          <span className="font-semibold text-[13px]">{title}</span>
          <span className="flex-1" />
          <button className="tcip-btn text-[11px]" onClick={onClose}>
            Cancel
          </button>
        </div>

        {/* Toolbar */}
        <div className="flex items-center gap-2 text-[11px]">
          <button className="tcip-btn" onClick={() => setPath("")} title="Top level / drives">
            Drives
          </button>
          <button
            className="tcip-btn"
            onClick={() => setPath(listing?.parent ?? "")}
            disabled={atTop}
            title="Up one folder"
          >
            ↑&nbsp;&nbsp;Up
          </button>
          <span className="flex-1 font-mono text-tcip-muted truncate" title={currentPath}>
            {currentPath || "This PC"}
          </span>
        </div>

        {/* Listing */}
        <div className="flex-1 overflow-auto border border-tcip-border rounded bg-tcip-bg">
          {loading ? (
            <div className="p-3 text-[11px] text-tcip-muted">Loading…</div>
          ) : error ? (
            <div className="p-3 text-[11px] text-tcip-fp">{error}</div>
          ) : listing && listing.entries.length === 0 ? (
            <div className="p-3 text-[11px] text-tcip-muted">No sub-folders here.</div>
          ) : (
            <ul>
              {listing?.entries.map((e) => (
                <li key={e.path}>
                  <button
                    className="w-full text-left px-3 py-1.5 text-[12px] hover:bg-tcip-hover transition-colors flex items-center gap-2"
                    onClick={() => setPath(e.path)}
                  >
                    <span aria-hidden>📁</span>
                    <span className="flex-1 truncate">{e.name}</span>
                    {e.is_dataset_root && (
                      <span className="tcip-badge bg-tcip-tp/20 text-tcip-tp text-[10px]">
                        dataset
                      </span>
                    )}
                  </button>
                </li>
              ))}
            </ul>
          )}
        </div>

        {/* Footer */}
        <div className="flex items-center gap-2">
          <span className="flex-1 text-[11px] text-tcip-muted truncate">
            {atTop ? (
              "Open a drive or folder"
            ) : (
              <>
                Selected: <span className="font-mono">{currentPath}</span>
                {listing?.is_dataset_root && (
                  <span className="text-tcip-tp"> · looks like a dataset root</span>
                )}
              </>
            )}
          </span>
          <button
            className="tcip-btn-primary text-[11px]"
            onClick={() => onSelect(currentPath)}
            disabled={atTop}
          >
            Use this folder
          </button>
        </div>
      </div>
    </div>
  );
}
