// HPO Panel — Format-agnostic trial table with auto-generated columns

(function () {
  var emptyState = document.getElementById("empty-state");
  var liveSection = document.getElementById("live-section");
  var raySection = document.getElementById("ray-section");

  /** @type {Object<string, {trialId: string, status: string, params: Object<string, unknown>, metrics: Object<string, number>}>} */
  var trials = {};

  /** @type {string[]} Param column keys in first-seen order */
  var paramCols = [];
  /** @type {string[]} Metric column keys in first-seen order */
  var metricCols = [];

  document.getElementById("btn-stop-hpo").addEventListener("click", function () { postToHost("stop_hpo"); });

  // ── Column discovery ──

  function discoverColumns(params, metrics) {
    var changed = false;
    var pKeys = Object.keys(params || {});
    for (var i = 0; i < pKeys.length; i++) {
      if (paramCols.indexOf(pKeys[i]) === -1) { paramCols.push(pKeys[i]); changed = true; }
    }
    var mKeys = Object.keys(metrics || {});
    for (var j = 0; j < mKeys.length; j++) {
      if (metricCols.indexOf(mKeys[j]) === -1) { metricCols.push(mKeys[j]); changed = true; }
    }
    return changed;
  }

  // ── Table rendering ──

  function renderTrialsTable() {
    var table = document.getElementById("trials-table");

    // Rebuild header
    var thead = table.querySelector("thead");
    var headerRow = "<tr><th>Trial</th><th>Status</th>";
    for (var p = 0; p < paramCols.length; p++) {
      headerRow += "<th>" + escapeHtml(formatLabel(paramCols[p])) + "</th>";
    }
    for (var m = 0; m < metricCols.length; m++) {
      headerRow += "<th>" + escapeHtml(formatLabel(metricCols[m])) + "</th>";
    }
    headerRow += "</tr>";
    thead.innerHTML = headerRow;

    // Rebuild body
    var tbody = table.querySelector("tbody");
    tbody.innerHTML = "";
    var ids = Object.keys(trials);
    for (var i = 0; i < ids.length; i++) {
      var t = trials[ids[i]];
      var statusClass = t.status.toLowerCase().replace(/[^a-z]/g, "");
      var row = "<td>" + escapeHtml(t.trialId) + "</td>";
      row += '<td><span class="trial-status ' + statusClass + '">' + escapeHtml(t.status) + "</span></td>";
      for (var pc = 0; pc < paramCols.length; pc++) {
        var pv = t.params[paramCols[pc]];
        row += "<td>" + formatCell(pv) + "</td>";
      }
      for (var mc = 0; mc < metricCols.length; mc++) {
        var mv = t.metrics[metricCols[mc]];
        row += "<td>" + formatCell(mv) + "</td>";
      }
      var tr = document.createElement("tr");
      if (t._best) tr.className = "best-row";
      tr.innerHTML = row;
      tbody.appendChild(tr);
    }
  }

  function renderSearchSpace(space) {
    var tbody = document.querySelector("#search-space-table tbody");
    tbody.innerHTML = "";
    var keys = Object.keys(space);
    for (var i = 0; i < keys.length; i++) {
      var tr = document.createElement("tr");
      tr.innerHTML = "<td>" + escapeHtml(keys[i]) + "</td><td style='font-size:11px'>" + escapeHtml(JSON.stringify(space[keys[i]])) + "</td>";
      tbody.appendChild(tr);
    }
  }

  function showBestTrial(trialId, params, metrics) {
    var section = document.getElementById("best-trial-section");
    section.style.display = "block";
    var info = document.getElementById("best-trial-info");

    var parts = ["Trial: " + trialId];
    var pKeys = Object.keys(params || {});
    for (var i = 0; i < pKeys.length; i++) {
      parts.push(pKeys[i] + "=" + formatCell(params[pKeys[i]]));
    }
    var mKeys = Object.keys(metrics || {});
    for (var j = 0; j < mKeys.length; j++) {
      parts.push(mKeys[j] + ": " + formatCell(metrics[mKeys[j]]));
    }
    info.textContent = parts.join(" · ");

    // Highlight best row in table
    if (trials[trialId]) trials[trialId]._best = true;
    renderTrialsTable();
  }

  // ── Helpers ──

  function formatCell(v) {
    if (v === undefined || v === null) return "—";
    if (typeof v === "number") return Math.abs(v) < 0.01 || Math.abs(v) >= 1000 ? v.toExponential(2) : v.toFixed(4);
    return escapeHtml(String(v));
  }

  function formatLabel(key) {
    return key.replace(/_/g, " ").replace(/\b\w/g, function (c) { return c.toUpperCase(); });
  }

  function escapeHtml(s) {
    var d = document.createElement("div");
    d.textContent = s;
    return d.innerHTML;
  }

  // ── Host messages ──

  onHostMessage(function (msg) {
    switch (msg.type) {
      case "hpo_started":
        emptyState.style.display = "none";
        liveSection.style.display = "block";
        document.getElementById("experiment-id").textContent = "Experiment: " + (msg.experimentId || "—");
        trials = {};
        paramCols = [];
        metricCols = [];

        if (msg.searchSpace) renderSearchSpace(msg.searchSpace);
        if (msg.rayDashboardUrl) {
          raySection.style.display = "block";
          document.getElementById("ray-frame").src = msg.rayDashboardUrl;
        }
        break;

      case "trial_update":
        discoverColumns(msg.params, msg.metrics);
        trials[msg.trialId] = {
          trialId: msg.trialId,
          status: msg.status || "running",
          params: msg.params || {},
          metrics: msg.metrics || {},
        };
        renderTrialsTable();
        break;

      case "hpo_complete":
        document.getElementById("experiment-id").textContent = "Experiment: " + (msg.experimentId || "—") + " — Complete";
        showBestTrial(msg.bestTrialId, msg.bestParams || {}, msg.bestMetrics || {});
        break;
    }
  });
})();
