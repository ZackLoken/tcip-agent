/**
 * Breeder-facing labels for every field a statement record carries.
 *
 * One flat map serves every statement kind (operationalization, trait-spec authoring, and any
 * future one): a field named `stated_by` or `rationale` means the same thing regardless of which
 * record carries it, so the label lives here once rather than once per panel. A field without an
 * entry here renders under its own name rather than disappearing, so a server-added field is never
 * silently dropped from the surface.
 */

export const STATEMENT_FIELD_LABELS: Record<string, string> = {
  // Shared across every statement kind: who stated it, when, and who confirmed it.
  stated_by: "Recorded through",
  stated_at: "Recorded at",
  relayed_note: "Relayed note",
  confirmed_by: "Confirmed by",
  confirmed_at: "Confirmed at",
  identity_from_request: "Identity source",
  // The agent harness the statement arrived through, as that software declared itself.
  agent_client_name: "Agent harness",
  agent_client_version: "Harness version",
  agent_session: "Agent session",
  terminal_session: "Terminal session",
  harness_session: "Harness session",
  harness_effort_at_connect: "Harness effort when it connected",

  // Operationalization: what a trait's delivered number means and how the platform decides it.
  statement: "What the number means",
  mechanism: "How the platform decides it",
  measured_subject: "Measured subject",
  delivered_phenotypes: "Delivered phenotypes",
  delivered_value_keys: "Delivered value keys",

  // Trait-spec authoring: the trait's own semantics, as authored, plus the account of why.
  statement_fields: "Authored fields",
  rationale: "Why the agent chose this",
  delivers: "Delivers",
  positive_class_name: "Positive class",
  milestone_fractions: "Milestone fractions",
  milestone_on: "Milestone read on",
  majority_milestone: "Majority milestone",
  majority_provisional: "Majority milestone is provisional",
  phenology_prefix: "Phenology column prefix",
  majority_label: "Majority milestone label",
  count_objective: "Count objective",
  count_bias_tolerance_frac: "Count bias tolerance",
  count_error_tolerance: "Count error tolerance",
  classifier_agreement_floor: "Classifier agreement floor",
  ordinal_agreement_floor: "Ordinal agreement floor",
  regression_skill_floor: "Regression skill floor",
  notes: "Notes",
};

/**
 * One field value, in the breeder's own register: an empty/missing value reads as "none" rather
 * than as a blank cell, an array joins on commas, and a boolean or number renders as itself rather
 * than falling through the string-only check that would otherwise hide it as "none".
 */
export function fieldValueText(value: unknown): string {
  if (Array.isArray(value)) return value.length > 0 ? value.join(", ") : "none";
  if (typeof value === "boolean") return value ? "yes" : "no";
  if (typeof value === "number") return String(value);
  if (typeof value !== "string" || value.trim() === "") return "none";
  return value;
}
