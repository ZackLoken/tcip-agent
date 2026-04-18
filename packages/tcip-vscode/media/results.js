// @ts-check
// Results Panel — metrics display, per-class table, CSV preview

(function () {
  const emptyState = document.getElementById("empty-state");
  const resultsSection = document.getElementById("results-section");
  const actionBar = document.getElementById("action-bar");

  document.getElementById("btn-accept").addEventListener("click", () => postToHost("accept_model"));
  document.getElementById("btn-retrain").addEventListener("click", () => postToHost("retrain"));
  document.getElementById("btn-export").addEventListener("click", () => postToHost("export_csv"));

  function metricClass(value, thresholds) {
    if (value >= thresholds[0]) return "good";
    if (value >= thresholds[1]) return "warn";
    return "bad";
  }

  function renderOverallMetrics(metrics) {
    const container = document.getElementById("overall-metrics");
    container.innerHTML = "";

    const items = [
      { key: "mAP50", label: "mAP50", thresholds: [0.7, 0.4] },
      { key: "mAP50-95", label: "mAP50-95", thresholds: [0.5, 0.3] },
      { key: "precision", label: "Precision", thresholds: [0.7, 0.4] },
      { key: "recall", label: "Recall", thresholds: [0.7, 0.4] },
      { key: "f1", label: "F1", thresholds: [0.7, 0.4] },
    ];

    for (const item of items) {
      const val = metrics[item.key];
      if (val === undefined) continue;
      const cls = metricClass(val, item.thresholds);
      const card = document.createElement("div");
      card.className = "metric-card";
      card.innerHTML = `<div class="label">${item.label}</div><div class="value ${cls}">${val.toFixed(4)}</div>`;
      container.appendChild(card);
    }
  }

  function renderPerClassTable(perClass) {
    const tbody = document.querySelector("#per-class-table tbody");
    tbody.innerHTML = "";

    const classes = Object.keys(perClass).sort();
    for (const cls of classes) {
      const m = perClass[cls];
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td style="font-weight:600">${cls}</td>
        <td>${(m.precision ?? 0).toFixed(4)}</td>
        <td>${(m.recall ?? 0).toFixed(4)}</td>
        <td>${(m.f1 ?? 0).toFixed(4)}</td>
        <td>${(m.mAP50 ?? m.map50 ?? 0).toFixed(4)}</td>
        <td>${(m["mAP50-95"] ?? m["map50-95"] ?? 0).toFixed(4)}</td>
        <td>${m.count ?? "—"}</td>
      `;
      tbody.appendChild(tr);
    }
  }

  onHostMessage((msg) => {
    switch (msg.type) {
      case "results_show":
        emptyState.style.display = "none";
        resultsSection.style.display = "block";
        actionBar.style.display = "flex";

        if (msg.metrics) renderOverallMetrics(msg.metrics);
        if (msg.perClass) renderPerClassTable(msg.perClass);
        break;
    }
  });
})();
