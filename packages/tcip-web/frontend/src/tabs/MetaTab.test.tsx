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
  vi.spyOn(metaApi, "reports").mockResolvedValue({ reports: [], count: 0, total_available: 0 });
  vi.spyOn(metaApi, "retrospectives").mockResolvedValue({
    retrospectives: [],
    count: 0,
    total_available: 0,
  });
});

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

describe("MetaTab heading", () => {
  it("renders exactly one top-level heading naming the tab", () => {
    render(<MetaTab />);
    const headings = screen.getAllByRole("heading", { level: 1 });
    expect(headings).toHaveLength(1);
    expect(headings[0]).toHaveTextContent("Meta");
  });
});

describe("MetaTab annotation sessions", () => {
  it("shows a rejected sessions load as the tab's error, never as no sessions", async () => {
    vi.spyOn(sessionsApi, "load").mockRejectedValue(new Error("a label document does not decode"));

    render(<MetaTab />);

    expect(
      await screen.findByText(
        /Annotation sessions could not be loaded: .*a label document does not decode/,
      ),
    ).toBeInTheDocument();
    expect(screen.getByText("not loaded")).toBeInTheDocument();
    expect(screen.queryByText(/No annotation sessions yet/)).not.toBeInTheDocument();
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
