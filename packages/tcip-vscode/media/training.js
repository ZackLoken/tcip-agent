// Training Dashboard — format-agnostic dynamic metrics visualization

(function () {
  var emptyState = document.getElementById("empty-state");
  var liveSection = document.getElementById("live-section");
  var tensorboardSection = document.getElementById("tensorboard-section");
  var metricsGrid = document.getElementById("metrics-grid");
  var chartsContainer = document.getElementById("charts-container");

  // ── State ──

  /** @type {Object<string, HTMLElement>} */
  var metricCards = {};

  /** @type {Object<string, {canvas: HTMLCanvasElement, chart: Object, labels: string[], data: number[]}>} */
  var metricCharts = {};

  /** @type {string[]} Ordered list of metric keys as they first appeared */
  var metricOrder = [];

  // Chart.js color palette
  var COLORS = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
    "#dcbeff", "#9A6324", "#fffac8", "#800000", "#aaffc3",
  ];

  var chartDefaults = {
    responsive: true,
    maintainAspectRatio: false,
    animation: false,
    plugins: { legend: { display: false } },
    scales: {
      x: { ticks: { color: "#888" }, grid: { color: "#333" } },
      y: { ticks: { color: "#888" }, grid: { color: "#333" } },
    },
  };

  // ── Dynamic metric card ──

  function ensureMetricCard(key) {
    if (metricCards[key]) return metricCards[key];

    var card = document.createElement("div");
    card.className = "metric-card";
    card.innerHTML = '<div class="label">' + escapeHtml(formatLabel(key)) +
      '</div><div class="value" id="mv-' + escapeHtml(key) + '">—</div>';
    metricsGrid.appendChild(card);
    metricCards[key] = card.querySelector(".value");

    if (metricOrder.indexOf(key) === -1) {
      metricOrder.push(key);
    }
    return metricCards[key];
  }

  function updateMetricCard(key, value) {
    var el = ensureMetricCard(key);
    if (typeof value === "number") {
      el.textContent = Math.abs(value) < 0.01 || Math.abs(value) >= 1000
        ? value.toExponential(2)
        : value.toFixed(4);
    } else {
      el.textContent = String(value);
    }
  }

  // ── Dynamic chart ──

  function ensureChart(key) {
    if (metricCharts[key]) return metricCharts[key];
    if (typeof Chart === "undefined") return null;

    var wrapper = document.createElement("div");
    wrapper.className = "chart-block";
    var header = document.createElement("div");
    header.className = "section-header";
    header.textContent = formatLabel(key);
    wrapper.appendChild(header);

    var container = document.createElement("div");
    container.className = "chart-container";
    var canvas = document.createElement("canvas");
    container.appendChild(canvas);
    wrapper.appendChild(container);
    chartsContainer.appendChild(wrapper);

    var colorIdx = metricOrder.indexOf(key) % COLORS.length;
    var data = { labels: [], datasets: [{ label: formatLabel(key), data: [], borderColor: COLORS[colorIdx], tension: 0.3, fill: false }] };
    var chart = new Chart(canvas, { type: "line", data: data, options: chartDefaults });

    metricCharts[key] = { canvas: canvas, chart: chart, labels: data.labels, data: data.datasets[0].data };
    return metricCharts[key];
  }

  function updateChart(key, epoch, value) {
    var entry = ensureChart(key);
    if (!entry) return;
    entry.labels.push(String(epoch));
    entry.data.push(value);
    entry.chart.update("none");
  }

  // ── Helpers ──

  function formatLabel(key) {
    return key.replace(/_/g, " ").replace(/\b\w/g, function (c) { return c.toUpperCase(); });
  }

  function escapeHtml(s) {
    var d = document.createElement("div");
    d.textContent = s;
    return d.innerHTML;
  }

  function resetState() {
    metricCards = {};
    metricCharts = {};
    metricOrder = [];
    metricsGrid.innerHTML = "";
    chartsContainer.innerHTML = "";
  }

  // ── Toolbar ──

  document.getElementById("btn-pause").addEventListener("click", function () { postToHost("pause_training"); });
  document.getElementById("btn-stop").addEventListener("click", function () { postToHost("stop_training"); });

  // ── Host messages ──

  onHostMessage(function (msg) {
    switch (msg.type) {
      case "training_started":
        emptyState.style.display = "none";
        liveSection.style.display = "block";
        document.getElementById("run-id").textContent = "Run: " + (msg.runId || "—");
        resetState();

        if (msg.tensorboardUrl) {
          tensorboardSection.style.display = "block";
          document.getElementById("tensorboard-frame").src = msg.tensorboardUrl;
        }
        break;

      case "metrics_update": {
        var metrics = msg.metrics || {};
        var epoch = msg.epoch || 0;

        // Always show epoch as a card (non-charted)
        updateMetricCard("epoch", epoch);

        var keys = Object.keys(metrics);
        for (var i = 0; i < keys.length; i++) {
          var key = keys[i];
          var val = metrics[key];
          updateMetricCard(key, val);

          // Chart numeric values (skip strings like ETA)
          if (typeof val === "number") {
            updateChart(key, epoch, val);
          }
        }
        break;
      }

      case "training_complete":
        document.getElementById("run-id").textContent =
          "Run: " + (msg.runId || "—") + " — Complete" +
          (msg.bestEpoch !== undefined ? " (best epoch: " + msg.bestEpoch + ")" : "");
        break;
    }
  });
})();
