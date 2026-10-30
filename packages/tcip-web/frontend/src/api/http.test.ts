import { describe, expect, it } from "vitest";

import {
  asJson,
  committedOf,
  decodeRefusal,
  getJson,
  postJson,
  refusalOf,
  StructuredRefusalError,
} from "@/api/http";
import {
  AUDIT_ENTRY_NOT_WRITTEN,
  type AuditGapRefusal,
  type SavedExport,
} from "@/api/types.generated";
import { ANNOTATE_ANSWERS } from "@/test/annotate";
import { RESULTS_ANSWERS } from "@/test/deliveryRecords";
import { IMAGE_ANSWERS } from "@/test/images";
import { answeredWith, fetchAnswers, thrown } from "@/test/refusals";

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
