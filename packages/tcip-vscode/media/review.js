// Review Panel -- Full-featured TP/FP/FN review with context-aware actions
// Phase 3 rewrite: matching display, detection navigation, GT modification,
// review state persistence, auto-zoom, stipple highlighting, filters

(function () {
  var canvas = /** @type {HTMLCanvasElement} */ (document.getElementById("review-canvas"));
  var wrapper = /** @type {HTMLElement} */ (document.getElementById("review-canvas-wrapper"));
  var emptyState = /** @type {HTMLElement} */ (document.getElementById("empty-state"));
  var ctx = canvas.getContext("2d");

  var COLORS = {tp:"#4ec96e", fp:"#c75050", fn:"#5090c7"};
  var ZOOM_CONTEXT = 3.0; // padding multiplier for auto-zoom

  // -- State --
  /** @type {HTMLImageElement|null} */ var img = null;
  var imageW = 0, imageH = 0;
  var imagePath = "";

  // View transform
  var scl = 1.0, offX = 0, offY = 0;
  var isPanning = false;
  var panSX = 0, panSY = 0, panSOX = 0, panSOY = 0;

  // Detection data
  /** @typedef {{type:string, tag:string, classId:number, conf:number, box:number[], polyPts:number[]|null, gtIdx:number, predIdx:number, decision:string}} Detection */
  /** @type {Detection[]} */ var allDetections = [];
  /** @type {Detection[]} */ var filteredDets = [];
  var currentDetIdx = 0;

  // GT and prediction data (for rendering)
  /** @type {Array<{type:string, classId:number, points:number[]}>} */ var gtAnnotations = [];
  /** @type {Array<{type:string, classId:number, points:number[], conf:number}>} */ var predAnnotations = [];

  // Image queue
  /** @type {Array<{imagePath:string}>} */ var imageQueue = [];
  var currentImageIdx = 0;

  // Filters
  var iouThreshold = 0.5;
  var confThreshold = 0.25;
  var filterType = "all"; // "all", "tp", "fp", "fn"
  var filterClass = -1; // -1 = all

  // Review state persistence
  /** @type {Record<string, Array<{tag:string, classId:number, box:number[], decision:string}>>} */ var reviewState = {};

  // Visibility toggles
  var showGT = true;
  var showPred = true;
  var showHelp = false;

  // Class names
  /** @type {Record<number,string>} */ var classNames = {};

  var CLASS_COLORS = [
    "#e6194b","#3cb44b","#ffe119","#4363d8","#f58231",
    "#911eb4","#42d4f4","#f032e6","#bfef45","#fabed4",
    "#469990","#dcbeff","#9a6324","#fffac8","#800000",
    "#aaffc3","#808000","#ffd8b1","#000075","#a9a9a9",
  ];

  // -- Coordinate transforms --
  /** @param {number} ix @param {number} iy */
  function i2c(ix, iy) { return {x:ix*scl+offX, y:iy*scl+offY}; }
  /** @param {number} cx @param {number} cy */
  function c2i(cx, cy) { return {x:(cx-offX)/scl, y:(cy-offY)/scl}; }

  // -- Halo text --
  /** @param {CanvasRenderingContext2D} c @param {string} txt @param {number} x @param {number} y */
  function haloText(c, txt, x, y) {
    c.save();
    c.font="bold 12px monospace";
    c.strokeStyle="rgba(0,0,0,0.8)"; c.lineWidth=3; c.lineJoin="round";
    c.strokeText(txt,x,y);
    c.fillStyle="#fff"; c.fillText(txt,x,y);
    c.restore();
  }

  // -- Filtering --
  function applyFilters() {
    filteredDets = [];
    for (var i=0; i<allDetections.length; i++) {
      var d = allDetections[i];
      if (d.conf < confThreshold && d.tag !== "fn") continue;
      if (filterType !== "all" && d.tag !== filterType) continue;
      if (filterClass >= 0 && d.classId !== filterClass) continue;
      filteredDets.push(d);
    }
    currentDetIdx = Math.min(currentDetIdx, Math.max(0, filteredDets.length-1));
    updateCounters();
    updateDetCounter();
  }

  function updateCounters() {
    var tp=0, fp=0, fn=0;
    for (var i=0; i<filteredDets.length; i++) {
      if (filteredDets[i].tag==="tp") tp++;
      else if (filteredDets[i].tag==="fp") fp++;
      else if (filteredDets[i].tag==="fn") fn++;
    }
    var tpEl=document.getElementById("tp-count"); if (tpEl) tpEl.textContent="TP: "+tp;
    var fpEl=document.getElementById("fp-count"); if (fpEl) fpEl.textContent="FP: "+fp;
    var fnEl=document.getElementById("fn-count"); if (fnEl) fnEl.textContent="FN: "+fn;
  }

  function updateDetCounter() {
    var el=document.getElementById("det-counter");
    if (el) el.textContent="Det "+(filteredDets.length>0?currentDetIdx+1:0)+" / "+filteredDets.length;
  }

  function updateImgCounter() {
    var el=document.getElementById("img-counter");
    if (el) el.textContent=(imageQueue.length>0?currentImageIdx+1:0)+" / "+imageQueue.length;
  }

  // -- Zoom / Pan --
  function fitToView() {
    if (imageW<=0||imageH<=0) return;
    var cw=canvas.width, ch=canvas.height;
    scl=Math.min(cw/imageW, ch/imageH)*0.95;
    offX=(cw-imageW*scl)/2; offY=(ch-imageH*scl)/2;
    render();
  }

  /** @param {number} detIdx */
  function zoomToDetection(detIdx) {
    if (detIdx<0||detIdx>=filteredDets.length) return;
    var d=filteredDets[detIdx];
    var bx1=d.box[0], by1=d.box[1], bx2=d.box[2], by2=d.box[3];
    var bw=bx2-bx1, bh=by2-by1;
    var cx2=(bx1+bx2)/2, cy2=(by1+by2)/2;
    // expand by context ratio
    var ew=bw*ZOOM_CONTEXT, eh=bh*ZOOM_CONTEXT;
    var cw=canvas.width, ch=canvas.height;
    scl=Math.min(cw/ew, ch/eh);
    offX=cw/2 - cx2*scl; offY=ch/2 - cy2*scl;
    render();
  }

  // -- Navigation --
  /** @param {number} delta */
  function navigateImage(delta) {
    if (imageQueue.length===0) return;
    var ni=currentImageIdx+delta;
    if (ni<0) ni=0; if (ni>=imageQueue.length) ni=imageQueue.length-1;
    if (ni===currentImageIdx) return;
    currentImageIdx=ni; currentDetIdx=0;
    loadCurrentImage();
  }

  /** @param {number} delta */
  function navigateDet(delta) {
    if (filteredDets.length===0) return;
    var ni=currentDetIdx+delta;
    if (ni<0) ni=0; if (ni>=filteredDets.length) ni=filteredDets.length-1;
    currentDetIdx=ni;
    updateDetCounter();
    zoomToDetection(currentDetIdx);
  }

  // -- Context-aware actions --
  /** @param {string} action */
  function reviewAction(action) {
    if (filteredDets.length===0) return;
    var d=filteredDets[currentDetIdx];
    d.decision=action;
    // persist in review state
    var stem=imagePath.replace(/^.*[\\/]/,"").replace(/\.[^.]+$/,"");
    if (!reviewState[stem]) reviewState[stem]=[];
    reviewState[stem].push({tag:d.tag,classId:d.classId,box:d.box.slice(),decision:action});
    // save review state to host
    postToHost("save_review_state", {state:reviewState});

    if (action==="reject" && d.tag!=="fp") {
      // rejecting TP or FN means delete GT — notify host
      postToHost("delete_gt", {gtIdx:d.gtIdx, tag:d.tag});
    }
    if (action==="edit") {
      // switch to annotation panel with prediction reference
      postToHost("open_in_annotation", {path:imagePath});
    }

    updateActionLabels();
    // auto-advance
    if (action!=="edit") navigateDet(1);
  }

  function updateActionLabels() {
    var acceptBtn=document.getElementById("btn-accept");
    var rejectBtn=document.getElementById("btn-reject");
    var editBtn=document.getElementById("btn-edit");
    if (!acceptBtn||!rejectBtn||!editBtn) return;
    if (filteredDets.length===0) return;
    var d=filteredDets[currentDetIdx];
    if (d.tag==="tp") {
      acceptBtn.textContent="\u2713 Confirm TP"; rejectBtn.textContent="\u2717 Remove GT"; editBtn.textContent="\u270E Edit GT";
    } else if (d.tag==="fp") {
      acceptBtn.textContent="\u2713 Dismiss FP"; rejectBtn.textContent="\u2717 Reject FP"; editBtn.textContent="\u270E Add GT";
    } else if (d.tag==="fn") {
      acceptBtn.textContent="\u2713 Keep GT"; rejectBtn.textContent="\u2717 Remove GT"; editBtn.textContent="\u270E Edit GT";
    }
    // show decision badge if already decided
    if (d.decision) {
      var badge=d.decision==="accept"?"\u2713":d.decision==="reject"?"\u2717":"\u270E";
      haloText(ctx, badge+" "+d.decision, 10, 20);
    }
  }

  // -- Load matching results --
  /** @param {Array<{type:string,tag:string,classId:number,conf:number,box:number[],polyPts?:number[],gtIdx:number,predIdx:number}>} matches */
  function loadMatches(matches) {
    allDetections = [];
    for (var i=0; i<matches.length; i++) {
      var m=matches[i];
      allDetections.push({
        type:m.type||"box", tag:m.tag, classId:m.classId, conf:m.conf||0,
        box:m.box, polyPts:m.polyPts||null, gtIdx:m.gtIdx||(-1), predIdx:m.predIdx||(-1),
        decision:""
      });
    }
    // restore previous decisions from review state
    var stem=imagePath.replace(/^.*[\\/]/,"").replace(/\.[^.]+$/,"");
    var prevReviews=reviewState[stem]||[];
    for (var ri=0; ri<prevReviews.length; ri++) {
      var pr=prevReviews[ri];
      // match by tag + box proximity
      for (var di=0; di<allDetections.length; di++) {
        var dd=allDetections[di];
        if (dd.tag===pr.tag && dd.classId===pr.classId && !dd.decision) {
          var dist=Math.abs(dd.box[0]-pr.box[0])+Math.abs(dd.box[1]-pr.box[1])+Math.abs(dd.box[2]-pr.box[2])+Math.abs(dd.box[3]-pr.box[3]);
          if (dist<5) { dd.decision=pr.decision; break; }
        }
      }
    }
    currentDetIdx=0;
    applyFilters();
    if (filteredDets.length>0) zoomToDetection(0);
    else fitToView();
    updateActionLabels();
  }

  // -- Render --
  function render() {
    var cw=canvas.width=canvas.clientWidth, ch=canvas.height=canvas.clientHeight;
    ctx.clearRect(0,0,cw,ch);

    // Draw image
    if (img && img.complete && img.naturalWidth>0) {
      ctx.drawImage(img, offX, offY, imageW*scl, imageH*scl);
    }

    // Draw GT annotations
    if (showGT) {
      for (var gi=0; gi<gtAnnotations.length; gi++) {
        var ga=gtAnnotations[gi];
        var col=CLASS_COLORS[ga.classId % CLASS_COLORS.length];
        ctx.strokeStyle=col; ctx.lineWidth=1.5; ctx.globalAlpha=0.5;
        if (ga.type==="box") {
          var gb1=i2c(ga.points[0],ga.points[1]), gb2=i2c(ga.points[2],ga.points[3]);
          ctx.strokeRect(gb1.x,gb1.y,gb2.x-gb1.x,gb2.y-gb1.y);
        } else if (ga.type==="polygon" && ga.points.length>=4) {
          ctx.beginPath();
          var gf=i2c(ga.points[0],ga.points[1]); ctx.moveTo(gf.x,gf.y);
          for (var gk=2;gk<ga.points.length;gk+=2) {
            var gv=i2c(ga.points[gk],ga.points[gk+1]); ctx.lineTo(gv.x,gv.y);
          }
          ctx.closePath(); ctx.stroke();
        }
        ctx.globalAlpha=1.0;
      }
    }

    // Draw detections
    for (var di=0; di<filteredDets.length; di++) {
      var d=filteredDets[di];
      var dcol=COLORS[d.tag]||"#888";
      var isCur=(di===currentDetIdx);

      ctx.strokeStyle=isCur?"#fff":dcol;
      ctx.lineWidth=isCur?3:1.5;
      ctx.fillStyle=dcol+(isCur?"40":"15");

      var dt1=i2c(d.box[0],d.box[1]), dt2=i2c(d.box[2],d.box[3]);
      var dw=dt2.x-dt1.x, dh=dt2.y-dt1.y;
      ctx.fillRect(dt1.x,dt1.y,dw,dh);
      ctx.strokeRect(dt1.x,dt1.y,dw,dh);

      // Draw polygon outline if available
      if (d.polyPts && d.polyPts.length>=4) {
        ctx.beginPath();
        var pf=i2c(d.polyPts[0],d.polyPts[1]); ctx.moveTo(pf.x,pf.y);
        for (var pk=2;pk<d.polyPts.length;pk+=2) {
          var pv=i2c(d.polyPts[pk],d.polyPts[pk+1]); ctx.lineTo(pv.x,pv.y);
        }
        ctx.closePath(); ctx.stroke();
      }

      // Tag label
      var tagLabel=d.tag.toUpperCase()+" "+(d.conf>0?d.conf.toFixed(2):"") +" c"+d.classId;
      haloText(ctx, tagLabel, dt1.x+2, dt1.y-4);

      // Decision stipple overlay
      if (d.decision) {
        ctx.save();
        if (d.decision==="accept") {
          ctx.strokeStyle="#4ec96e"; ctx.setLineDash([4,4]); ctx.lineWidth=2;
        } else if (d.decision==="reject") {
          ctx.strokeStyle="#c75050"; ctx.setLineDash([2,2]); ctx.lineWidth=2;
        } else {
          ctx.strokeStyle="#ff0"; ctx.setLineDash([6,3]); ctx.lineWidth=2;
        }
        ctx.strokeRect(dt1.x-2,dt1.y-2,dw+4,dh+4);
        ctx.setLineDash([]);
        var decLabel=d.decision==="accept"?"\u2713":d.decision==="reject"?"\u2717":"\u270E";
        haloText(ctx, decLabel, dt2.x+3, dt1.y+12);
        ctx.restore();
      }

      // Current detection highlight ring
      if (isCur) {
        ctx.save();
        ctx.strokeStyle="#fff"; ctx.lineWidth=1; ctx.setLineDash([3,3]);
        ctx.strokeRect(dt1.x-4,dt1.y-4,dw+8,dh+8);
        ctx.setLineDash([]); ctx.restore();
      }
    }

    // Help overlay
    if (showHelp) renderHelp();
  }

  // -- Help --
  function renderHelp() {
    ctx.save();
    ctx.fillStyle="rgba(0,0,0,0.75)";
    ctx.fillRect(20,20,280,280);
    ctx.fillStyle="#fff"; ctx.font="bold 14px monospace";
    var lines=[
      "REVIEW SHORTCUTS",
      "──────────────────",
      "Left/Right  Prev / Next det",
      "Up/Down     Prev / Next image",
      "A           Accept",
      "R           Reject",
      "E           Edit (open annotate)",
      "G           Toggle GT visibility",
      "P           Toggle pred visibility",
      "F           Fit to view",
      "H           Toggle help",
      "1/2/3       Filter: TP/FP/FN",
      "0           Show all types",
    ];
    for (var hi=0; hi<lines.length; hi++) {
      ctx.fillText(lines[hi], 30, 45+hi*19);
    }
    ctx.restore();
  }

  // -- Image loading --
  /** @param {string} uri */
  function loadImageFromUri(uri) {
    var newImg = new Image();
    newImg.onload = function() {
      img = newImg;
      imageW = newImg.naturalWidth;
      imageH = newImg.naturalHeight;
      emptyState.style.display = "none";
      fitToView();
    };
    newImg.src = uri;
  }

  function loadCurrentImage() {
    if (imageQueue.length===0) return;
    var item=imageQueue[currentImageIdx];
    imagePath=item.imagePath;
    allDetections=[]; filteredDets=[]; currentDetIdx=0;
    updateImgCounter();
    updateCounters();
    updateDetCounter();
    // Request image URI and matching data from host
    postToHost("request_image", {path:item.imagePath});
    postToHost("request_matches", {path:item.imagePath, iouThreshold:iouThreshold, confThreshold:confThreshold});
  }

  // -- Slider wiring --
  var iouSlider = /** @type {HTMLInputElement|null} */ (document.getElementById("iou-slider"));
  var confSlider = /** @type {HTMLInputElement|null} */ (document.getElementById("conf-slider"));
  if (iouSlider) {
    iouSlider.addEventListener("input", function() {
      iouThreshold = parseInt(iouSlider.value) / 100;
      var valEl=document.getElementById("iou-value"); if (valEl) valEl.textContent=iouThreshold.toFixed(2);
      applyFilters(); render();
    });
  }
  if (confSlider) {
    confSlider.addEventListener("input", function() {
      confThreshold = parseInt(confSlider.value) / 100;
      var valEl=document.getElementById("conf-value"); if (valEl) valEl.textContent=confThreshold.toFixed(2);
      applyFilters(); render();
    });
  }

  // -- Button wiring --
  var prevImgBtn=document.getElementById("btn-prev-img");
  var nextImgBtn=document.getElementById("btn-next-img");
  var prevDetBtn=document.getElementById("btn-prev-det");
  var nextDetBtn=document.getElementById("btn-next-det");
  var acceptBtn=document.getElementById("btn-accept");
  var editBtn=document.getElementById("btn-edit");
  var rejectBtn=document.getElementById("btn-reject");
  if (prevImgBtn) prevImgBtn.addEventListener("click", function() { navigateImage(-1); });
  if (nextImgBtn) nextImgBtn.addEventListener("click", function() { navigateImage(1); });
  if (prevDetBtn) prevDetBtn.addEventListener("click", function() { navigateDet(-1); });
  if (nextDetBtn) nextDetBtn.addEventListener("click", function() { navigateDet(1); });
  if (acceptBtn) acceptBtn.addEventListener("click", function() { reviewAction("accept"); });
  if (editBtn) editBtn.addEventListener("click", function() { reviewAction("edit"); });
  if (rejectBtn) rejectBtn.addEventListener("click", function() { reviewAction("reject"); });

  // -- Keyboard --
  /** @param {KeyboardEvent} e */
  function onKey(e) {
    if (e.key==="ArrowLeft") { navigateDet(-1); return; }
    if (e.key==="ArrowRight") { navigateDet(1); return; }
    if (e.key==="ArrowUp") { e.preventDefault(); navigateImage(-1); return; }
    if (e.key==="ArrowDown") { e.preventDefault(); navigateImage(1); return; }
    var kl=e.key.toLowerCase();
    if (kl==="a") { reviewAction("accept"); return; }
    if (kl==="r") { reviewAction("reject"); return; }
    if (kl==="e") { reviewAction("edit"); return; }
    if (kl==="g") { showGT=!showGT; render(); return; }
    if (kl==="p") { showPred=!showPred; render(); return; }
    if (kl==="f") { fitToView(); return; }
    if (kl==="h") { showHelp=!showHelp; render(); return; }
    if (e.key==="1") { filterType=filterType==="tp"?"all":"tp"; applyFilters(); render(); return; }
    if (e.key==="2") { filterType=filterType==="fp"?"all":"fp"; applyFilters(); render(); return; }
    if (e.key==="3") { filterType=filterType==="fn"?"all":"fn"; applyFilters(); render(); return; }
    if (e.key==="0") { filterType="all"; applyFilters(); render(); return; }
  }
  document.addEventListener("keydown", onKey);

  // -- Mouse: wheel zoom + middle-click pan --
  /** @param {WheelEvent} e */
  canvas.addEventListener("wheel", function(e) {
    e.preventDefault();
    var r=canvas.getBoundingClientRect(), cx=e.clientX-r.left, cy=e.clientY-r.top;
    if (e.ctrlKey) {
      var factor=e.deltaY<0?1.15:1/1.15;
      var ip=c2i(cx,cy);
      scl*=factor; offX=cx-ip.x*scl; offY=cy-ip.y*scl;
    } else if (e.shiftKey) {
      offX -= e.deltaY;
    } else {
      offY -= e.deltaY;
    }
    render();
  }, {passive:false});

  /** @param {MouseEvent} e */
  canvas.addEventListener("mousedown", function(e) {
    if (e.button===1) { isPanning=true; panSX=e.clientX; panSY=e.clientY; panSOX=offX; panSOY=offY; e.preventDefault(); }
  });
  /** @param {MouseEvent} e */
  canvas.addEventListener("mousemove", function(e) {
    if (isPanning) { offX=panSOX+(e.clientX-panSX); offY=panSOY+(e.clientY-panSY); render(); }
  });
  /** @param {MouseEvent} e */
  canvas.addEventListener("mouseup", function(e) {
    if (e.button===1) isPanning=false;
  });

  // -- Resize --
  new ResizeObserver(function() { render(); }).observe(wrapper);

  // -- Host message handler --
  onHostMessage(function(/** @type {{type:string, [key:string]:any}} */ msg) {
    switch (msg.type) {
      case "image_uri":
        if (msg.uri) loadImageFromUri(msg.uri);
        if (msg.path) imagePath = msg.path;
        break;
      case "load_matches":
        if (Array.isArray(msg.matches)) loadMatches(msg.matches);
        break;
      case "load_gt":
        if (Array.isArray(msg.annotations)) gtAnnotations=msg.annotations;
        render();
        break;
      case "load_predictions":
        if (Array.isArray(msg.annotations)) predAnnotations=msg.annotations;
        render();
        break;
      case "set_image_queue":
        if (Array.isArray(msg.images)) {
          imageQueue=[];
          for (var qi=0; qi<msg.images.length; qi++) {
            imageQueue.push({imagePath:msg.images[qi]});
          }
          currentImageIdx=0;
          if (imageQueue.length>0) { emptyState.style.display="none"; loadCurrentImage(); }
          updateImgCounter();
        }
        break;
      case "review_state_loaded":
        if (msg.state && typeof msg.state==="object") {
          reviewState = /** @type {Record<string, Array<{tag:string, classId:number, box:number[], decision:string}>>} */ (msg.state);
        }
        break;
      case "setClasses":
        if (msg.classes && typeof msg.classes==="object") {
          classNames = /** @type {Record<number,string>} */ (msg.classes);
        }
        break;
    }
  });

  // Initial state
  render();
})();
