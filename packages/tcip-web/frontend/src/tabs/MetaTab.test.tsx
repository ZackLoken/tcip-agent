import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { metaApi } from "@/api/meta";
import { sessionsApi } from "@/api/sessions";
import { useStore } from "@/store";
import { MetaTab } from "@/tabs/MetaTab";
import { META_ANSWERS } from "@/test/meta";
import { thrown } from "@/test/refusals";
import { SESSION_ANSWERS } from "@/test/sessions";
import { openTestProject } from "@/test/store";

const initialStoreState = useStore.getState();

beforeEach(() => {
  useStore.setState(initialStoreState, true);
  openTestProject();
  vi.spyOn(metaApi, "reports").mockResolvedValue({ entries: [], total_available: 0 });
  vi.spyOn(metaApi, "retrospectives").mockResolvedValue({ entries: [], total_available: 0 });
});

afterEach(cleanup);

describe("MetaTab heading", () => {
  it("renders exactly one top-level heading naming the tab", () => {
    render(<MetaTab />);
    const headings = screen.getAllByRole("heading", { level: 1 });
    expect(headings).toHaveLength(1);
    expect(headings[0]).toHaveTextContent("Meta");
  });
});

describe("MetaTab before its loads answer", () => {
  it("shows no empty line before the loads answer", () => {
    vi.mocked(metaApi.reports).mockReturnValue(new Promise(() => {}));
    vi.mocked(metaApi.retrospectives).mockReturnValue(new Promise(() => {}));
    vi.spyOn(sessionsApi, "load").mockReturnValue(new Promise(() => {}));

    render(<MetaTab />);

    expect(screen.queryByText("No reports.")).not.toBeInTheDocument();
    expect(screen.queryByText("No retrospectives.")).not.toBeInTheDocument();
    expect(screen.queryByText("No sessions.")).not.toBeInTheDocument();
  });
});

describe("MetaTab annotation sessions", () => {
  it("shows a rejected sessions load as the tab's error, never as no sessions", async () => {
    const refused = await thrown(SESSION_ANSWERS.unopened);
    vi.spyOn(sessionsApi, "load").mockRejectedValue(refused);

    render(<MetaTab />);

    expect(
      await screen.findByText(
        (text) =>
          text.startsWith("Sessions could not be loaded:") && text.includes(refused.message),
      ),
    ).toBeInTheDocument();
    expect(screen.getByText("not loaded")).toBeInTheDocument();
    expect(screen.queryByText("No sessions.")).not.toBeInTheDocument();
  });

  it("lists a report's context as key and value pairs", async () => {
    const [{ record }] = META_ANSWERS.reports.entries;
    if ("reason" in record) throw new Error(`the produced report will not read: ${record.reason}`);
    vi.spyOn(sessionsApi, "load").mockResolvedValue({ sessions: [] });
    vi.mocked(metaApi.reports).mockResolvedValue(META_ANSWERS.reports);

    render(<MetaTab />);

    expect(await screen.findByText(record.detail)).toBeInTheDocument();
    for (const [key, value] of Object.entries(record.context)) {
      expect(screen.getByText(key)).toBeInTheDocument();
      expect(screen.getByText(String(value))).toBeInTheDocument();
    }
    expect(document.querySelector("pre")).toBeNull();
  });

  it("lists a retrospective's stated sections under their field names", async () => {
    const [{ record }, { record: unread }] = META_ANSWERS.retrospectives.entries;
    if ("reason" in record) throw new Error(`the produced one will not read: ${record.reason}`);
    if (!("reason" in unread)) throw new Error("the produced damaged one reads");
    vi.spyOn(sessionsApi, "load").mockResolvedValue({ sessions: [] });
    vi.mocked(metaApi.retrospectives).mockResolvedValue(META_ANSWERS.retrospectives);

    render(<MetaTab />);

    expect(await screen.findByText(record.project_id)).toBeInTheDocument();
    expect(screen.getByText("did not work")).toBeInTheDocument();
    expect(screen.getByText(record.did_not_work)).toBeInTheDocument();
    expect(record.assumptions_wrong).toBe("");
    expect(screen.queryByText("assumptions wrong")).not.toBeInTheDocument();
    // The reason spans lines, compared with its whitespace collapsed as the matcher reads a node.
    const collapsed = (text: string) => text.replace(/\s+/g, " ").trim();
    expect(screen.getByText(collapsed(unread.reason))).toBeInTheDocument();
  });

  it("lists the sessions a load returns", async () => {
    const { listing } = SESSION_ANSWERS;
    vi.spyOn(sessionsApi, "load").mockResolvedValue(listing);

    render(<MetaTab />);

    expect((await screen.findAllByText(listing.sessions[0].user)).length).toBe(
      listing.sessions.filter((s) => s.user === listing.sessions[0].user).length,
    );
    expect(screen.getByText(`${listing.sessions.length} recorded`)).toBeInTheDocument();
  });
});
