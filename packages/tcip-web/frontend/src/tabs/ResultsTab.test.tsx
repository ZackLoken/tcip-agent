import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";

import { api } from "@/api/client";
import { resultsApi } from "@/api/inference";
import { useStore } from "@/store";
import { ResultsTab } from "@/tabs/ResultsTab";

const initialStoreState = useStore.getState();

function setupDataset() {
  useStore.setState((s) => ({
    gui: {
      ...s.gui,
      dataset: {
        ...s.gui.dataset,
        project_root: "C:/proj",
        dataset_root: "C:/data",
      },
    },
  }));
}

beforeEach(() => {
  useStore.setState(initialStoreState, true);
  setupDataset();
});

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

describe("ResultsTab structured predictions-by-date picker (K15 #10)", () => {
  it("defaults each date to its first model with predictions, and skips a date with none", async () => {
    vi.spyOn(api.dataset, "tree").mockResolvedValue({
      dataset_root: "C:/data",
      dates_with_images: ["2026-01-01", "2026-01-08"],
      subjects: ["catkin"],
      model_names: ["baseline", "v2"],
      subjects_by_date: {},
      models_by_date: { "2026-01-01": ["baseline", "v2"], "2026-01-08": [] },
    });
    const curvesSpy = vi.spyOn(resultsApi, "perPlantCurves").mockResolvedValue({
      rows: [],
      n_plants: 0,
      positive_class_id: 1,
      elongation_classified: true,
    });
    vi.spyOn(resultsApi, "onsetDates").mockResolvedValue({ rows: [] });

    render(<ResultsTab />);
    await waitFor(() => expect(api.dataset.tree).toHaveBeenCalledWith("C:/data"));

    // The classified date defaults to "baseline" (first with predictions); the empty date shows
    // its own disabled "no predictions" option, never silently reusing the other date's model.
    await waitFor(() => expect(screen.getByText("2026-01-01")).toBeInTheDocument());
    const selects = screen.getAllByTitle(
      /Model whose predictions to use for this date|No model has predictions for this date/,
    ) as HTMLSelectElement[];
    expect(selects).toHaveLength(2);
    expect(selects[0].value).toBe("baseline");
    expect(selects[1]).toBeDisabled();

    fireEvent.click(screen.getByRole("button", { name: /compute curves/i }));
    await waitFor(() => expect(curvesSpy).toHaveBeenCalled());
    expect(curvesSpy.mock.calls[0][0].predictions_by_date).toEqual({
      "2026-01-01": "C:/data/predictions/baseline/2026-01-01/detect",
    });
  });

  it("dropping a date to '— skip —' excludes it from the computed predictions map", async () => {
    vi.spyOn(api.dataset, "tree").mockResolvedValue({
      dataset_root: "C:/data",
      dates_with_images: ["2026-01-01"],
      subjects: ["catkin"],
      model_names: ["baseline"],
      subjects_by_date: {},
      models_by_date: { "2026-01-01": ["baseline"] },
    });
    const curvesSpy = vi.spyOn(resultsApi, "perPlantCurves").mockResolvedValue({
      rows: [],
      n_plants: 0,
      positive_class_id: 1,
      elongation_classified: true,
    });
    vi.spyOn(resultsApi, "onsetDates").mockResolvedValue({ rows: [] });

    render(<ResultsTab />);
    await waitFor(() => expect(screen.getByText("2026-01-01")).toBeInTheDocument());

    fireEvent.change(screen.getByTitle("Model whose predictions to use for this date"), {
      target: { value: "" },
    });
    fireEvent.click(screen.getByRole("button", { name: /compute curves/i }));
    await waitFor(() => expect(curvesSpy).toHaveBeenCalled());
    expect(curvesSpy.mock.calls[0][0].predictions_by_date).toEqual({});
  });
});

describe("ResultsTab onset table validity marker", () => {
  async function computeWithOnsetRows(
    rows: Awaited<ReturnType<typeof resultsApi.onsetDates>>["rows"],
  ) {
    vi.spyOn(api.dataset, "tree").mockResolvedValue({
      dataset_root: "C:/data",
      dates_with_images: ["2026-01-01"],
      subjects: ["catkin"],
      model_names: ["baseline"],
      subjects_by_date: {},
      models_by_date: { "2026-01-01": ["baseline"] },
    });
    vi.spyOn(resultsApi, "perPlantCurves").mockResolvedValue({
      rows: [
        {
          plant_id: "P1",
          accession: null,
          date: "2026-01-01",
          n_images: 1,
          n_total: 10,
          n_positive: 3,
          n_unclassified: 0,
          n_missing: 0,
          ratio: 0.3,
        },
      ],
      n_plants: 1,
      positive_class_id: 1,
      elongation_classified: true,
    });
    vi.spyOn(resultsApi, "onsetDates").mockResolvedValue({ rows });

    render(<ResultsTab />);
    await waitFor(() => expect(screen.getByText("2026-01-01")).toBeInTheDocument());
    fireEvent.click(screen.getByRole("button", { name: /compute curves/i }));
    await waitFor(() => expect(screen.getByText("P1")).toBeInTheDocument());
  }

  it("marks a fully-classified, fully-observed plant valid", async () => {
    await computeWithOnsetRows([
      {
        plant_id: "P1",
        accession: null,
        n_datapoints: 1,
        n_dates_unclassified: 0,
        n_dates_missing_images: 0,
        n_observed_dates: 1,
        catkin_50per_date: "2026-02-01",
      },
    ]);
    expect(screen.getByText("valid")).toBeInTheDocument();
    expect(screen.queryByText("incomplete")).not.toBeInTheDocument();
    expect(screen.queryByText("no observations")).not.toBeInTheDocument();
  });

  it("marks a plant with an unclassified or missing-image date incomplete, not silently blank", async () => {
    await computeWithOnsetRows([
      {
        plant_id: "P1",
        accession: null,
        n_datapoints: 1,
        n_dates_unclassified: 1,
        n_dates_missing_images: 0,
        n_observed_dates: 0,
        catkin_50per_date: null,
      },
    ]);
    expect(screen.getByText("incomplete")).toBeInTheDocument();
    expect(screen.queryByText("valid")).not.toBeInTheDocument();
    expect(screen.queryByText("no observations")).not.toBeInTheDocument();
  });

  it("marks a fully-classified, fully-observed plant with zero detections as 'no observations', not 'valid' (stage-6 review N6)", async () => {
    await computeWithOnsetRows([
      {
        plant_id: "P1",
        accession: null,
        n_datapoints: 2,
        n_dates_unclassified: 0,
        n_dates_missing_images: 0,
        n_observed_dates: 0,
        catkin_50per_date: null,
      },
    ]);
    expect(screen.getByText("no observations")).toBeInTheDocument();
    expect(screen.queryByText("valid")).not.toBeInTheDocument();
    expect(screen.queryByText("incomplete")).not.toBeInTheDocument();
  });
});
