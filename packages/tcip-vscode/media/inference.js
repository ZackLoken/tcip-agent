// Inference Panel — progress tracking and result previews

(function () {
  var emptyState = document.getElementById("empty-state");
  var progressSection = document.getElementById("progress-section");
  var completeBanner = document.getElementById("complete-banner");
  var previewSection = document.getElementById("preview-section");
  var previewList = document.getElementById("preview-list");

  var outputDir = "";

  document.getElementById("btn-open-results").addEventListener("click", function () { postToHost("open_results"); });
  document.getElementById("btn-open-dir").addEventListener("click", function () { postToHost("open_output_dir", { path: outputDir }); });

  function addPreviewItem(imagePath, detections) {
    var item = document.createElement("div");
    item.className = "preview-item";
    var filename = imagePath.replace(/.*[/\\]/, "");
    var info = document.createElement("div");
    info.className = "info";
    var fnDiv = document.createElement("div");
    fnDiv.className = "filename";
    fnDiv.textContent = filename;
    var detDiv = document.createElement("div");
    detDiv.className = "det-count";
    detDiv.textContent = detections + " detection" + (detections !== 1 ? "s" : "");
    info.appendChild(fnDiv);
    info.appendChild(detDiv);
    item.appendChild(info);
    previewList.insertBefore(item, previewList.firstChild);
    // Keep max 20 preview items
    while (previewList.children.length > 20) {
      previewList.removeChild(previewList.lastChild);
    }
  }

  onHostMessage(function (msg) {
    switch (msg.type) {
      case "inference_started":
        emptyState.style.display = "none";
        progressSection.style.display = "block";
        completeBanner.style.display = "none";
        previewSection.style.display = "block";
        previewList.innerHTML = "";
        document.getElementById("inference-status").textContent = "Running...";
        document.getElementById("progress-label").textContent = "Model: " + ((msg.modelPath || "").replace(/.*[/\\]/, ""));
        document.getElementById("progress-count").textContent = "0 / " + msg.totalImages;
        document.getElementById("progress-bar").style.width = "0%";
        break;

      case "inference_progress": {
        var pct = msg.total > 0 ? (msg.current / msg.total) * 100 : 0;
        document.getElementById("progress-bar").style.width = pct + "%";
        document.getElementById("progress-count").textContent = msg.current + " / " + msg.total;
        addPreviewItem(msg.imagePath, msg.detections);
        break;
      }

      case "inference_complete":
        outputDir = msg.outputDir || "";
        document.getElementById("inference-status").textContent = "Complete";
        document.getElementById("progress-bar").style.width = "100%";
        completeBanner.style.display = "block";
        document.getElementById("complete-summary").textContent =
          msg.totalImages + " images processed · " + msg.totalDetections + " total detections";
        break;
    }
  });
})();
