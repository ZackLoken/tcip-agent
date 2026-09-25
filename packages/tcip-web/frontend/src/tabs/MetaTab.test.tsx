import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { cleanup, render, screen } from "@testing-library/react";

import { metaApi } from "@/api/meta";
import { sessionsApi } from "@/api/sessions";
import { useStore } from "@/store";
import { MetaTab } from "@/tabs/MetaTab";

const initialStoreState = useStore.getState();

beforeEach(() => {
  useStore.setState(initialStoreState, true);
  useStore.setState((s) => ({
    gui: { ...s.gui, dataset: { ...s.gui.dataset, project_root: "/proj" } },
  }));
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
    vi.spyOn(sessionsApi, "load").mockRejectedValue(new Error("image_status does not decode"));

    render(<MetaTab />);

    expect(
      await screen.findByText(
        /Annotation sessions could not be loaded: .*image_status does not decode/,
      ),
    ).toBeInTheDocument();
    expect(screen.getByText("not loaded")).toBeInTheDocument();
    expect(screen.queryByText(/No annotation sessions yet/)).not.toBeInTheDocument();
  });

  it("lists the sessions a load returns", async () => {
    vi.spyOn(sessionsApi, "load").mockResolvedValue({
      sessions: [
        {
          user: "alice",
          started: "01-02-2026 10:00:00",
          ended: "",
          images_annotated: 1,
          total_annotations: 2,
          total_time_seconds: 9,
          avg_seconds_per_annotation: 4.5,
          negative_confirmation_seconds: 0,
          review_seconds: 0,
          new_annotation_seconds: 9,
          images: {
            "a.jpg": {
              session_seconds: 9,
              annotations_added: 2,
              final_annotation_count: 2,
              avg_seconds_per_annotation: 4.5,
            },
          },
        },
      ],
    });

    render(<MetaTab />);

    expect(await screen.findByText("alice")).toBeInTheDocument();
    expect(screen.getByText("1 recorded")).toBeInTheDocument();
  });
});
