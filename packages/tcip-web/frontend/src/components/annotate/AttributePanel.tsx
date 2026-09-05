import { useEffect, useMemo, useState } from "react";

import { classesApi, type AttributeDef, type Registry } from "@/api/classes";
import { AttributeEditors } from "@/components/annotate/AttributeEditors";
import { CollapsibleSection } from "@/components/CollapsibleSection";
import { schemaChangeSweepToast } from "@/lib/registrySweep";
import { useStore } from "@/store";

/** Per-instance attribute editing + a geometry-less (image/plant-level) rating entry, plus
 *  authoring new attributes and values onto the active subject. Minimal but functional (the
 *  polished editor is a later slice): the selected shape's attributes, the image-level ratings
 *  that ride in the same label file with no box, and the registry-growing controls a breeder
 *  otherwise has no way to reach without a shell. */
export function AttributePanel({ selectedBoxIdx }: { selectedBoxIdx: number | null }) {
  const activeSubject = useStore((s) => s.gui.active_subject);
  const registry = useStore((s) => s.registry.subjects);
  const registryVersion = useStore((s) => s.registry.version);
  const setRegistry = useStore((s) => s.setRegistry);
  const dataset = useStore((s) => s.gui.dataset);
  const boxes = useStore((s) => s.canvas.boxes);
  const polygons = useStore((s) => s.canvas.polygons);
  const points = useStore((s) => s.canvas.points);
  const selectedPolygonIdx = useStore((s) => s.canvas.selectedPolygonIdx);
  const selectedPointIdx = useStore((s) => s.canvas.selectedPointIdx);
  const imageAnnotations = useStore((s) => s.canvas.imageAnnotations);
  const updateBox = useStore((s) => s.updateBox);
  const updatePolygon = useStore((s) => s.updatePolygon);
  const updatePoint = useStore((s) => s.updatePoint);
  const addImageAnnotation = useStore((s) => s.addImageAnnotation);
  const updateImageAnnotation = useStore((s) => s.updateImageAnnotation);
  const deleteImageAnnotation = useStore((s) => s.deleteImageAnnotation);

  const selected =
    selectedBoxIdx != null && boxes[selectedBoxIdx]
      ? ({ kind: "box", idx: selectedBoxIdx, shape: boxes[selectedBoxIdx] } as const)
      : selectedPolygonIdx != null && polygons[selectedPolygonIdx]
        ? ({
            kind: "polygon",
            idx: selectedPolygonIdx,
            shape: polygons[selectedPolygonIdx],
          } as const)
        : selectedPointIdx != null && points[selectedPointIdx]
          ? ({ kind: "point", idx: selectedPointIdx, shape: points[selectedPointIdx] } as const)
          : null;

  const withAttr = (attrs: Record<string, string>, attr: string, value: string) => {
    const next = { ...attrs };
    if (value) next[attr] = value;
    else delete next[attr];
    return next;
  };

  const setInstanceAttr = (attr: string, value: string) => {
    if (!selected) return;
    if (selected.kind === "box") {
      const b = boxes[selected.idx];
      updateBox(selected.idx, { ...b, attributes: withAttr(b.attributes, attr, value) });
    } else if (selected.kind === "polygon") {
      const p = polygons[selected.idx];
      updatePolygon(selected.idx, { ...p, attributes: withAttr(p.attributes, attr, value) });
    } else {
      const p = points[selected.idx];
      updatePoint(selected.idx, { ...p, attributes: withAttr(p.attributes, attr, value) });
    }
  };

  // Grows the registry through the same door the toolbar's subject add uses, and the same toast.
  async function saveGrownRegistry(next: Registry) {
    setRegistry(next, registryVersion);
    if (!dataset.project_root) return;
    try {
      const saved = await classesApi.save(
        dataset.project_root,
        next,
        dataset.dataset_root,
        dataset.annotations_dir,
        registryVersion,
      );
      setRegistry(next, saved.version);
      const toast = schemaChangeSweepToast(saved.schema_change_sweep);
      if (toast) useStore.getState().pushToast(toast, "info");
    } catch (e) {
      // A refusal means this browser's registry is not trustworthy: reload rather than keep it.
      useStore
        .getState()
        .pushToast(`Could not update attributes: ${e instanceof Error ? e.message : String(e)}`);
      try {
        const fresh = await classesApi.load(
          dataset.project_root,
          dataset.dataset_root,
          dataset.annotations_dir,
        );
        setRegistry(fresh.subjects, fresh.version);
      } catch {
        /* the reload itself failing leaves the optimistic registry in place */
      }
    }
  }

  function addAttribute(
    subject: string,
    name: string,
    type: "categorical" | "ordinal",
    values: string[],
  ) {
    const current = registry[subject] ?? {};
    const attrDef: AttributeDef = { type, values };
    const next: Registry = {
      ...registry,
      [subject]: { ...current, attributes: { ...(current.attributes ?? {}), [name]: attrDef } },
    };
    void saveGrownRegistry(next);
  }

  function addValue(subject: string, attrName: string, value: string) {
    const current = registry[subject];
    const def = current?.attributes?.[attrName];
    if (!def || def.values.includes(value)) return;
    const next: Registry = {
      ...registry,
      [subject]: {
        ...current,
        attributes: {
          ...current.attributes,
          [attrName]: { ...def, values: [...def.values, value] },
        },
      },
    };
    void saveGrownRegistry(next);
  }

  const [addingAttribute, setAddingAttribute] = useState(false);
  const [attrName, setAttrName] = useState("");
  const [attrType, setAttrType] = useState<"categorical" | "ordinal">("categorical");
  const [attrValues, setAttrValues] = useState("");

  function resetAttrDraft() {
    setAddingAttribute(false);
    setAttrName("");
    setAttrType("categorical");
    setAttrValues("");
  }

  function submitAttrDraft() {
    if (!activeSubject) return;
    const name = attrName.trim();
    const values = attrValues
      .split("\n")
      .map((v) => v.trim())
      .filter((v) => v.length > 0);
    if (!name || values.length === 0) return;
    addAttribute(activeSubject, name, attrType, values);
    resetAttrDraft();
  }

  // The training loader excludes an image over an instance with no value for a declared
  // attribute; this reads the same gap over the loaded shapes, one line per declared attribute.
  const activeSubjectShapes = useMemo(() => {
    if (!activeSubject) return [];
    return [
      ...boxes.filter((b) => b.subject === activeSubject),
      ...polygons.filter((p) => p.subject === activeSubject),
      ...points.filter((p) => p.subject === activeSubject),
    ];
  }, [activeSubject, boxes, polygons, points]);
  const activeAttributes = activeSubject ? (registry[activeSubject]?.attributes ?? {}) : {};

  // Open until the user collapses it, and never re-keyed on the active image: for many traits this
  // is the only way to record a rating, so it must not be hidden when an image loads.
  const [ratingsOpen, setRatingsOpen] = useState(true);

  /** Manually dismissible (unlike the legends, this panel holds real inputs, so hover-to-reveal
   *  would fight the user reaching into it). Re-opens on a fresh selection so a stale dismiss
   *  can't hide the one shape you're now trying to edit. */
  const [dismissed, setDismissed] = useState(false);
  useEffect(() => {
    if (selected) setDismissed(false);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selected?.kind, selected?.idx]);

  const hasContent = !!selected || imageAnnotations.length > 0;
  if (dismissed || !hasContent) {
    return (
      <button
        type="button"
        onClick={() => setDismissed(false)}
        className="absolute top-3 right-3 z-20 flex items-center gap-1.5 rounded-full border border-tcip-border bg-tcip-panel/90 px-2.5 py-1 text-[11px] text-tcip-muted backdrop-blur hover:border-tcip-border-hover hover:text-tcip-fg"
      >
        Attributes
      </button>
    );
  }

  return (
    <div className="absolute top-3 right-3 z-20 w-60 rounded-md border border-tcip-border bg-tcip-panel/95 p-3 text-[11px] shadow-lg backdrop-blur">
      <div className="mb-1 flex items-center justify-between">
        <h4 className="text-[11px] font-semibold tracking-wide text-tcip-fg">Attributes</h4>
        <button
          type="button"
          onClick={() => setDismissed(true)}
          aria-label="Close attributes panel"
          title="Close"
          className="text-tcip-muted hover:text-tcip-fg"
        >
          ✕
        </button>
      </div>
      {selected ? (
        <div className="mb-2">
          <div className="mb-1 text-tcip-muted">
            Selected <span className="font-semibold text-tcip-fg">{selected.shape.subject}</span>
          </div>
          <AttributeEditors
            subject={selected.shape.subject}
            attributes={selected.shape.attributes}
            registry={registry}
            onChange={setInstanceAttr}
            onAddValue={(attr, value) => addValue(selected.shape.subject, attr, value)}
          />
        </div>
      ) : (
        <p className="mb-2 text-tcip-muted">Select a shape to set its attributes.</p>
      )}

      {activeSubject && (
        <div className="mb-2 rounded border border-tcip-border bg-tcip-bg/60 p-2">
          <div className="mb-1 flex items-center justify-between">
            <span className="text-tcip-muted">Attributes for {activeSubject}</span>
            <button
              type="button"
              className="text-tcip-accent hover:underline"
              onClick={() => setAddingAttribute((o) => !o)}
            >
              + Attribute
            </button>
          </div>
          {Object.entries(activeAttributes).map(([name]) => {
            const unassessed = activeSubjectShapes.filter((s) => !s.attributes[name]).length;
            return (
              <p key={name} className="text-tcip-muted">
                {unassessed} of {activeSubjectShapes.length} {activeSubject} shapes carry no {name}{" "}
                value.
              </p>
            );
          })}
          {addingAttribute && (
            <div className="mt-1.5 space-y-1.5 border-t border-tcip-border pt-1.5">
              <input
                autoFocus
                className="tcip-input w-full text-[11px]"
                placeholder="attribute name"
                value={attrName}
                onChange={(e) => setAttrName(e.target.value)}
              />
              <select
                className="tcip-select w-full text-[11px]"
                value={attrType}
                onChange={(e) => setAttrType(e.target.value as "categorical" | "ordinal")}
              >
                <option value="categorical">categorical</option>
                <option value="ordinal">ordinal</option>
              </select>
              <textarea
                className="tcip-input w-full text-[11px]"
                placeholder={
                  "one value per line" + (attrType === "ordinal" ? ", lowest first" : "")
                }
                rows={3}
                value={attrValues}
                onChange={(e) => setAttrValues(e.target.value)}
              />
              <div className="flex gap-1.5">
                <button
                  type="button"
                  className="tcip-btn-primary flex-1 text-[11px]"
                  disabled={!attrName.trim() || !attrValues.trim()}
                  onClick={submitAttrDraft}
                >
                  Add
                </button>
                <button
                  type="button"
                  className="tcip-btn flex-1 text-[11px]"
                  onClick={resetAttrDraft}
                >
                  Cancel
                </button>
              </div>
            </div>
          )}
        </div>
      )}

      <CollapsibleSection
        className="mt-2 rounded border border-tcip-border bg-tcip-bg/60 p-2"
        title="Ratings for this whole image"
        caption="Applies to the whole image, not to any shape."
        open={ratingsOpen}
        onToggle={() => setRatingsOpen((o) => !o)}
      >
        {imageAnnotations.length === 0 && (
          <p className="mb-1 text-tcip-muted">None on this image.</p>
        )}
        {imageAnnotations.map((a, i) => (
          <div key={i} className="mb-1.5 rounded border border-tcip-border p-1.5">
            <div className="mb-1 flex items-center gap-1">
              <span className="font-semibold text-tcip-fg">{a.subject}</span>
              <button
                type="button"
                className="ml-auto text-tcip-muted hover:text-tcip-fp"
                title="Remove this rating"
                onClick={() => deleteImageAnnotation(i)}
              >
                ✕
              </button>
            </div>
            <AttributeEditors
              subject={a.subject}
              attributes={a.attributes}
              registry={registry}
              onChange={(attr, value) =>
                updateImageAnnotation(i, { ...a, attributes: withAttr(a.attributes, attr, value) })
              }
              onAddValue={(attr, value) => addValue(a.subject, attr, value)}
            />
          </div>
        ))}
        <button
          type="button"
          className="tcip-btn mt-1 w-full text-[11px]"
          disabled={!activeSubject}
          onClick={() => activeSubject && addImageAnnotation(activeSubject)}
        >
          + Rating for {activeSubject ?? "…"}
        </button>
      </CollapsibleSection>
    </div>
  );
}
