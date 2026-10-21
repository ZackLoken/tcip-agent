import { useMemo, useState } from "react";

import type { AttrType, AttributeDef, Registry } from "@/api/subjects";
import { ATTR_TYPES } from "@/api/types.generated";
import { AttributeEditors } from "@/components/annotate/AttributeEditors";
import { saveRegistry } from "@/lib/registrySave";
import type { ReviewItem } from "@/lib/reviewItems";
import { useStore } from "@/store";
import { TOOL_ARRAY } from "@/store/types";

/** The lower part of the subject menu: the focused shape's attributes, the active subject's
 *  declared attributes with the control that declares a new one, and the image-level ratings
 *  that ride in the label document with no shape. Absent while there is no subject, no focused
 *  shape and no rating to show. */
export function AttributePanel({ focused }: { focused: ReviewItem | null }) {
  const activeSubject = useStore((s) => s.gui.active_subject);
  const registry = useStore((s) => s.registry.subjects);
  const boxes = useStore((s) => s.canvas.boxes);
  const polygons = useStore((s) => s.canvas.polygons);
  const points = useStore((s) => s.canvas.points);
  const imageAnnotations = useStore((s) => s.canvas.imageAnnotations);
  const add = useStore((s) => s.add);
  const update = useStore((s) => s.update);
  const remove = useStore((s) => s.remove);

  // The focused annotation's own shape; a focused proposal has no attributes to set.
  const focusedShape =
    focused?.kind !== "annotation"
      ? undefined
      : { boxes, polygons, points }[TOOL_ARRAY[focused.shape]][focused.ref];
  const selected =
    focused && focusedShape ? { kind: focused.shape, idx: focused.ref, shape: focusedShape } : null;

  const withAttr = (attrs: Record<string, string>, attr: string, value: string) => {
    const next = { ...attrs };
    if (value) next[attr] = value;
    else delete next[attr];
    return next;
  };

  const setInstanceAttr = (attr: string, value: string) => {
    if (!selected) return;
    const { shape } = selected;
    update(TOOL_ARRAY[selected.kind], selected.idx, {
      ...shape,
      attributes: withAttr(shape.attributes, attr, value),
    });
  };

  function saveGrownRegistry(next: Registry) {
    void saveRegistry(next, "Could not update the attributes");
  }

  function addAttribute(subject: string, name: string, type: AttrType, values: string[]) {
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
  const [attrType, setAttrType] = useState<AttrType>(ATTR_TYPES[0]);
  const [attrValues, setAttrValues] = useState("");
  const [attrNameError, setAttrNameError] = useState<string | null>(null);

  function resetAttrDraft() {
    setAddingAttribute(false);
    setAttrName("");
    setAttrType(ATTR_TYPES[0]);
    setAttrValues("");
    setAttrNameError(null);
  }

  function submitAttrDraft() {
    if (!activeSubject) return;
    const name = attrName.trim();
    const values = attrValues
      .split("\n")
      .map((v) => v.trim())
      .filter((v) => v.length > 0);
    if (!name || values.length === 0) return;
    if (activeAttributes[name]) {
      setAttrNameError(`${name} is already declared.`);
      return;
    }
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

  if (!activeSubject && !selected && imageAnnotations.length === 0) return null;

  return (
    <div
      className="mt-1 border-t border-tcip-border px-2 pt-3 pb-1 text-[13px]"
      data-testid="subject-attributes"
    >
      {selected && (
        <div className="mb-5">
          <div className="mb-2 font-semibold">{selected.shape.subject}</div>
          <AttributeEditors
            subject={selected.shape.subject}
            attributes={selected.shape.attributes}
            registry={registry}
            onChange={setInstanceAttr}
            onAddValue={(attr, value) => addValue(selected.shape.subject, attr, value)}
          />
        </div>
      )}

      {activeSubject && (
        <div className="mb-5">
          <div className="mb-2 flex items-center justify-between">
            <span className="tcip-label">{`${activeSubject} attributes`}</span>
            <button
              type="button"
              className="text-tcip-accent hover:underline"
              onClick={() => setAddingAttribute((o) => !o)}
            >
              Add attribute
            </button>
          </div>
          {Object.entries(activeAttributes).map(([name]) => {
            const unassessed = activeSubjectShapes.filter((s) => !s.attributes[name]).length;
            return (
              <p key={name} className="text-tcip-muted">
                {`${name}: ${unassessed} of ${activeSubjectShapes.length} ${activeSubject} shapes unset`}
              </p>
            );
          })}
          {addingAttribute && (
            <div className="mt-2 space-y-2">
              <input
                className="tcip-input w-full"
                placeholder="Attribute"
                aria-label="attribute name"
                value={attrName}
                onChange={(e) => {
                  setAttrName(e.target.value);
                  setAttrNameError(null);
                }}
              />
              {attrNameError && <p className="tcip-problem">{attrNameError}</p>}
              <select
                className="tcip-select w-full"
                aria-label="attribute type"
                value={attrType}
                onChange={(e) => setAttrType(e.target.value as AttrType)}
              >
                {ATTR_TYPES.map((t) => (
                  <option key={t} value={t}>
                    {t}
                  </option>
                ))}
              </select>
              <textarea
                className="tcip-input h-20 w-full py-1"
                placeholder={attrType === "ordinal" ? "Values, lowest first" : "Values"}
                aria-label="attribute values"
                rows={3}
                value={attrValues}
                onChange={(e) => setAttrValues(e.target.value)}
              />
              <div className="flex gap-2">
                <button
                  type="button"
                  className="tcip-btn-primary flex-1"
                  disabled={!attrName.trim() || !attrValues.trim()}
                  onClick={submitAttrDraft}
                >
                  Add
                </button>
                <button type="button" className="tcip-btn flex-1" onClick={resetAttrDraft}>
                  Cancel
                </button>
              </div>
            </div>
          )}
        </div>
      )}

      <div>
        <div className="mb-2 tcip-label">Image ratings</div>
        {imageAnnotations.map((a, i) => (
          <div key={i} className="mb-2 rounded border border-tcip-border p-2">
            <div className="mb-1 flex items-center gap-1">
              <span className="font-semibold">{a.subject}</span>
              <button
                type="button"
                className="ml-auto text-tcip-muted hover:text-tcip-fg"
                title="Remove this rating"
                aria-label="Remove this rating"
                onClick={() => remove("imageAnnotations", i)}
              >
                ✕
              </button>
            </div>
            <AttributeEditors
              subject={a.subject}
              attributes={a.attributes}
              registry={registry}
              onChange={(attr, value) =>
                update("imageAnnotations", i, {
                  ...a,
                  attributes: withAttr(a.attributes, attr, value),
                })
              }
              onAddValue={(attr, value) => addValue(a.subject, attr, value)}
            />
          </div>
        ))}
        <button
          type="button"
          className="tcip-btn w-full"
          disabled={!activeSubject}
          onClick={() =>
            activeSubject &&
            add("imageAnnotations", { subject: activeSubject, attributes: {}, iscrowd: false })
          }
        >
          Add rating
        </button>
      </div>
    </div>
  );
}
