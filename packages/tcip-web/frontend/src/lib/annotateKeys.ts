/** The Annotate tab's key bindings, declared once: the tab binds each by its `keys` and the help
 *  overlay lists each by its `label` and `desc`. */
export const ANNOTATE_KEYS = {
  mode: { keys: "m", label: "m", desc: "Cycle Box / Polygon / Point mode" },
  undo: { keys: "ctrl+z", label: "Ctrl+Z", desc: "Undo" },
  redo: { keys: "ctrl+shift+z", label: "Ctrl+Shift+Z", desc: "Redo" },
  redoAlias: { keys: "ctrl+y", label: "Ctrl+Y", desc: "Redo (alias)" },
  save: { keys: "ctrl+s", label: "Ctrl+S", desc: "Save labels" },
  accept: {
    keys: "a",
    label: "a",
    desc: "Accept the selected proposal: it joins the labels, or confirms the annotation it pairs with",
  },
  reject: { keys: "r", label: "r", desc: "Reject the selected proposal" },
  nextProposal: { keys: "p", label: "p", desc: "Select the next undecided proposal" },
  hideProposals: {
    keys: "h",
    label: "h",
    desc: "Hide or show proposals on this image; a mark made while hidden records it",
  },
  complete: {
    keys: "c",
    label: "c",
    desc: "Mark every instance of the subject on this image annotated, or withdraw the mark",
  },
  stream: {
    keys: "v",
    label: "v",
    desc: "Toggle stream drawing: click starts/pauses laying, double-click closes",
  },
  snap: { keys: "s", label: "s", desc: "Toggle vertex snapping (polygon mode)" },
  cut: {
    keys: "x",
    label: "x",
    desc: "Arm the cut tool: click two points on either side of the selected polygon",
  },
  close: { keys: "enter", label: "Enter", desc: "Close current polygon (or double-click)" },
  delete: { keys: "delete", label: "Delete", desc: "Delete the selected polygon, box or point" },
  cancel: {
    keys: "escape",
    label: "Esc",
    desc: "Cancel in-progress drawing, clear selections and disarm the cut tool",
  },
  previous: { keys: "arrowleft", label: "←", desc: "Previous image" },
  next: { keys: "arrowright", label: "→", desc: "Next image" },
  // Each digit is its own binding: the Nth character picks the Nth registered subject.
  subject: {
    keys: "0123456789",
    label: "0–9",
    desc: "Select the Nth registered subject (0 is the first)",
  },
} as const;
