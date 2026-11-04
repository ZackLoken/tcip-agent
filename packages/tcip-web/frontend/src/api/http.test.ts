import { describe, expect, it } from "vitest";

import {
  asJson,
  committedOf,
  decodeRefusal,
  getJson,
  issuesText,
  postJson,
  readRefusalText,
  refusalOf,
  StructuredRefusalError,
} from "@/api/http";
import {
  AUDIT_ENTRY_NOT_WRITTEN,
  type AuditGapRefusal,
  INVALID,
  type SavedExport,
  UNREAD,
  UNREADABLE,
} from "@/api/types.generated";
import { ANNOTATE_ANSWERS } from "@/test/annotate";
import { RESULTS_ANSWERS } from "@/test/deliveryRecords";
import { IMAGE_ANSWERS } from "@/test/images";
import { PICKER } from "@/test/projectRecords";
import { answeredWith, fetchAnswers, thrown } from "@/test/refusals";
import { TRAIT_LISTINGS } from "@/test/traitRecords";

/** A produced refusal whose detail is a sentence: the status route's answer for no such job. */
const UNKNOWN_JOB = IMAGE_ANSWERS.unknown;
const SENTENCE = UNKNOWN_JOB.body.detail;

describe("http helpers", () => {
  it("asJson returns the parsed body on a 2xx", async () => {
    const { completed } = IMAGE_ANSWERS;
    expect(await asJson(answeredWith(200, completed))).toEqual(completed);
  });

  it("asJson throws the backend detail on a non-2xx", async () => {
    await expect(asJson(answeredWith(UNKNOWN_JOB.status, UNKNOWN_JOB.body))).rejects.toThrow(
      SENTENCE,
    );
  });

  it("getJson resolves the body via fetch", async () => {
    const { completed } = IMAGE_ANSWERS;
    fetchAnswers(200, completed);
    expect(await getJson("/x")).toEqual(completed);
  });

  it("postJson throws (does not swallow) on a non-2xx", async () => {
    const refusal = ANNOTATE_ANSWERS.engine_refusal;
    fetchAnswers(refusal.status, refusal.body);
    await expect(postJson("/x", {})).rejects.toThrow(refusal.body.detail);
  });

  it("asJson keeps an object detail parsed instead of stringifying it", async () => {
    const produced = RESULTS_ANSWERS.operationalization_refusal;
    const { detail } = produced.body;
    const raised = await asJson(answeredWith(produced.status, produced.body)).catch(
      (e: unknown) => e,
    );
    expect(raised).toBeInstanceOf(StructuredRefusalError);
    const refusal = raised as StructuredRefusalError;
    expect(refusal.detail).toEqual(detail);
    expect(refusal.status).toBe(produced.status);
    expect(refusal.message).toBe(detail.message);
    expect(refusal.message).not.toContain("[object Object]");
  });

  it("decodeRefusal falls back to the caller's own text when the body carries no detail", async () => {
    // A transport-level failure: a 500 whose body no route wrote.
    const raised = await decodeRefusal(answeredWith(500, {}), "export_csv failed: 500");
    expect(raised).not.toBeInstanceOf(StructuredRefusalError);
    expect(raised.message).toBe("export_csv failed: 500");
  });

  it("a plain string detail still carries its status, so a caller can branch on it", async () => {
    const raised = await asJson(answeredWith(UNKNOWN_JOB.status, UNKNOWN_JOB.body)).catch(
      (e: unknown) => e,
    );
    expect(raised).toBeInstanceOf(StructuredRefusalError);
    const refusal = raised as StructuredRefusalError;
    expect(refusal.status).toBe(UNKNOWN_JOB.status);
    expect(refusal.message).toBe(SENTENCE);
  });

  it("refusalOf reads the refusal by its kind at any status, never just 409", async () => {
    const produced = RESULTS_ANSWERS.mapping_audit_gap;
    const { detail } = produced.body;
    const asAnswered = await thrown(produced);
    // A status no route answers this refusal with, so only its kind can name it.
    const elsewhere = await asJson(answeredWith(500, produced.body)).catch((e: unknown) => e);
    expect(refusalOf<AuditGapRefusal>(asAnswered, AUDIT_ENTRY_NOT_WRITTEN)).toEqual(detail);
    expect(refusalOf<AuditGapRefusal>(elsewhere, AUDIT_ENTRY_NOT_WRITTEN)).toEqual(detail);
  });

  it("refusalOf is null for a refusal of another kind, or a non-error value", async () => {
    const gap = (e: unknown) => refusalOf<AuditGapRefusal>(e, AUDIT_ENTRY_NOT_WRITTEN);
    expect(gap(await thrown(RESULTS_ANSWERS.delivery_refusal))).toBeNull();
    expect(gap(new Error("plain"))).toBeNull();
    expect(gap(null)).toBeNull();
  });

  it("committedOf returns the committed body only for the marker, null otherwise", async () => {
    const produced = RESULTS_ANSWERS.count_export_audit_gap;
    expect(committedOf<SavedExport>(await thrown(produced))).toEqual(
      produced.body.detail.committed,
    );

    expect(committedOf(await thrown(RESULTS_ANSWERS.mapping_audit_gap))).toBeNull();

    expect(committedOf(await thrown(ANNOTATE_ANSWERS.engine_refusal))).toBeNull();
  });
});

describe("a read refusal's text", () => {
  /** The produced trait record its schema refuses, by two issues each placed in the record. */
  const [UNREADABLE_TRAIT] = TRAIT_LISTINGS.unreadable.unreadable;
  /** The produced delivery listing's refusal: one issue, placed at the record as a whole. */
  const UNREADABLE_DELIVERIES = RESULTS_ANSWERS.unreadable_deliveries;
  /** The produced stray directory's problem, a record that will not read for another reason. */
  const STRAY = PICKER.stray;

  it("issuesText joins each issue's dotted place and message, a placeless one by its message", () => {
    const delivery = UNREADABLE_DELIVERIES.body.detail;
    if (delivery.kind !== INVALID) throw new Error("the produced refusal is not a model's");
    expect(issuesText(UNREADABLE_TRAIT.reason.issues)).toBe(
      "revisions.0.entry.localization: Input should be '', 'center_match' or 'iou_match'; " +
        "revisions: Tuple should have at least 1 item after validation, not 0",
    );
    expect(issuesText(delivery.issues)).toBe(
      "Input should be a valid dictionary or instance of DeliveryEventRecord",
    );
  });

  it("readRefusalText reads an invalid record by its issues and an unreadable one by its message", () => {
    if (STRAY.kind !== UNREAD) throw new Error("the produced stray's record read");
    expect(readRefusalText(UNREADABLE_TRAIT.reason)).toBe(
      "revisions.0.entry.localization: Input should be '', 'center_match' or 'iou_match'; " +
        "revisions: Tuple should have at least 1 item after validation, not 0",
    );
    expect(STRAY.problem.kind).toBe(UNREADABLE);
    expect(readRefusalText(STRAY.problem)).toBe(
      "/scratch/workspace/stray is not a project yet: create it with initialize_project",
    );
  });

  it("decodeRefusal reads an invalid detail as its issues' text", async () => {
    const raised = await thrown(UNREADABLE_DELIVERIES);
    expect(raised.message).toBe(
      "Input should be a valid dictionary or instance of DeliveryEventRecord",
    );
  });
});
