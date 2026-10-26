import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { metaApi } from "@/api/meta";
import { sessionsApi } from "@/api/sessions";
import { useStore } from "@/store";
import { MetaTab } from "@/tabs/MetaTab";
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
    vi.spyOn(sessionsApi, "load").mockRejectedValue(new Error("a label document does not decode"));

    render(<MetaTab />);

    expect(
      await screen.findByText(/Sessions could not be loaded: .*a label document does not decode/),
    ).toBeInTheDocument();
    expect(screen.getByText("not loaded")).toBeInTheDocument();
    expect(screen.queryByText("No sessions.")).not.toBeInTheDocument();
  });

  it("lists a report's context as key and value pairs", async () => {
    vi.spyOn(sessionsApi, "load").mockResolvedValue({ sessions: [] });
    vi.mocked(metaApi.reports).mockResolvedValue({
      entries: [
        {
          id: "r1",
          record: {
            category: "missing_tool",
            detail: "no lister of assessments",
            timestamp: "2026-10-20T10:00:00+00:00",
            context: { route: "/api/results/assessments", attempts: 2 },
            user_disagreement: false,
          },
        },
      ],
      total_available: 1,
    });

    render(<MetaTab />);

    expect(await screen.findByText("no lister of assessments")).toBeInTheDocument();
    expect(screen.getByText("route")).toBeInTheDocument();
    expect(screen.getByText("/api/results/assessments")).toBeInTheDocument();
    expect(screen.getByText("attempts")).toBeInTheDocument();
    expect(screen.getByText("2")).toBeInTheDocument();
    expect(document.querySelector("pre")).toBeNull();
  });

  it("lists a retrospective's stated sections under their field names", async () => {
    vi.spyOn(sessionsApi, "load").mockResolvedValue({ sessions: [] });
    vi.mocked(metaApi.retrospectives).mockResolvedValue({
      entries: [
        {
          id: "r1",
          record: {
            timestamp: "2026-10-20T10:00:00+00:00",
            project_id: "valley-bloom-trial",
            task: "train the bloom counter",
            worked: "tiling held",
            did_not_work: "dense clusters merged",
            assumptions_wrong: "",
            knowledge_for_future: "",
            missing_or_hard_tools: "",
            would_do_differently: "",
          },
        },
        { id: "unreadable", record: { reason: "not a Retrospective" } },
      ],
      total_available: 2,
    });

    render(<MetaTab />);

    expect(await screen.findByText("valley-bloom-trial")).toBeInTheDocument();
    expect(screen.getByText("did not work")).toBeInTheDocument();
    expect(screen.getByText("dense clusters merged")).toBeInTheDocument();
    expect(screen.queryByText("assumptions wrong")).not.toBeInTheDocument();
    expect(screen.getByText("not a Retrospective")).toBeInTheDocument();
  });

  it("lists the sessions a load returns", async () => {
    vi.spyOn(sessionsApi, "load").mockResolvedValue({
      sessions: [
        {
          user: "user:alice",
          started: "2026-02-01T10:00:00+00:00",
          ended: null,
          entries: [
            {
              contribution_id: "c1",
              image_name: "a.jpg",
              seconds: 9,
              annotations_added: 2,
              activity: "new_annotation",
            },
          ],
          images_annotated: 1,
          total_annotations: 2,
          total_time_seconds: 9,
          seconds_by_activity: { new_annotation: 9, review: 0, negative_confirmation: 0 },
          avg_seconds_per_annotation: 4.5,
        },
      ],
    });

    render(<MetaTab />);

    expect(await screen.findByText("user:alice")).toBeInTheDocument();
    expect(screen.getByText("1 recorded")).toBeInTheDocument();
  });
});
