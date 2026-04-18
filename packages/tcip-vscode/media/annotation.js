// Annotation Canvas -- Full-featured implementation ported from yolo-annotator
// Phase 1 rewrite: box/polygon drawing, vertex editing, streaming, snapping,
// scale-dependent rendering, halo text, help overlay, prediction reference

/** @typedef {{classId: number, type: 'box'|'polygon', points: number[], id: number}} Annotation */

(function () {
  var canvas = /** @type {HTMLCanvasElement} */ (document.getElementById("annotation-canvas"));
  var wrapper = /** @type {HTMLElement} */ (document.getElementById("canvas-wrapper"));
  var emptyState = /** @type {HTMLElement} */ (document.getElementById("empty-state"));
  var ctx = canvas.getContext("2d");

  var CLASS_COLORS = [
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
    "#911eb4", "#42d4f4", "#f032e6", "#bfef45", "#fabed4",
    "#469990", "#dcbeff", "#9a6324", "#fffac8", "#800000",
    "#aaffc3", "#808000", "#ffd8b1", "#000075", "#a9a9a9",
  ];
  var ZOOM_LEVELS = [
    0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.33, 0.5, 0.67,
    0.75, 0.85, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0,
  ];
  var SNAP_RADIUS = 15;
  var STREAM_MIN_DIST = 6;
  var VERTEX_HIT = 8;
  var EDGE_HIT = 6;
  var HALO_OFFSETS = [
    [-2,-2],[-2,-1],[-2,0],[-2,1],[-2,2],
    [-1,-2],[-1,-1],[-1,0],[-1,1],[-1,2],
    [0,-2],[0,-1],[0,1],[0,2],
    [1,-2],[1,-1],[1,0],[1,1],[1,2],
    [2,-2],[2,-1],[2,0],[2,1],[2,2],
  ];

  // -- State --
  /** @type {HTMLImageElement|null} */ var image = null;
  var imageW = 0, imageH = 0;
  var imagePath = "";
  /** @type {Annotation[]} */ var annotations = [];
  var nextId = 1;
  var mode = "select";
  var selectedId = -1;
  var hoveredId = -1;
  var currentClassId = 0;
  /** @type {Record<number,string>} */ var classNames = {};

  // View transform
  var scl = 1.0;
  var offX = 0, offY = 0;
  var zoomIdx = 11;

  // Undo/redo
  /** @type {string[]} */ var undoStack = [];
  /** @type {string[]} */ var redoStack = [];

  // Box drawing
  var isDrawingBox = false;
  var boxStartX = 0, boxStartY = 0;

  // Polygon drawing -- flat [x,y,x,y,...] coords in image space
  /** @type {number[]} */ var curPoly = [];
  /** @type {number[]} */ var vertRedoStack = [];

  // Vertex editing
  /** @type {{annoIdx:number,vertIdx:number}|null} */ var dragVert = null;

  // Streaming
  var streamMode = false;
  var streamActive = false;
  /** @type {number[]|null} */ var lastStreamXY = null;

  // Toggles
  var snapOn = true;
  var annoVisible = true;
  var showHelp = false;
  var dirty = false;

  // Spatial index
  /** @type {Array<{x1:number,y1:number,x2:number,y2:number}|null>} */ var polyBB = [];
  var polyBBDirty = true;

  // Prediction reference overlay
  /** @type {Annotation[]} */ var predAnnos = [];
  var predVisible = false;

  // SAM-assisted annotation
  var samBusy = false;
  /** @type {Array<{x:number,y:number,label:number}>} */ var samPoints = [];
  /** @type {{x1:number,y1:number,x2:number,y2:number}|null} */ var samBox = null;
  /** @type {number[]|null} */ var samPreviewPoly = null;

  // Mouse tracking
  var mCX = 0, mCY = 0;

  // Pan state
  var isPanning = false;
  var panSX = 0, panSY = 0, panSOX = 0, panSOY = 0;

  // -- Coordinate transforms --
  /** @param {number} cx @param {number} cy */
  function c2i(cx, cy) { return {x:(cx-offX)/scl, y:(cy-offY)/scl}; }
  /** @param {number} ix @param {number} iy */
  function i2c(ix, iy) { return {x:ix*scl+offX, y:iy*scl+offY}; }
  /** @param {number} ix @param {number} iy */
  function clampI(ix, iy) { return {x:Math.max(0,Math.min(imageW,ix)), y:Math.max(0,Math.min(imageH,iy))}; }

  // -- Geometry helpers --
  /** @param {number} px @param {number} py @param {number[]} pts */
  function pip(px, py, pts) {
    var inside = false;
    for (var i=0, j=pts.length-2; i<pts.length; j=i, i+=2) {
      var xi=pts[i],yi=pts[i+1],xj=pts[j],yj=pts[j+1];
      if ((yi>py)!==(yj>py) && px<((xj-xi)*(py-yi))/(yj-yi)+xi) inside=!inside;
    }
    return inside;
  }
  /** @param {number} px @param {number} py @param {number} ax @param {number} ay @param {number} bx @param {number} by */
  function ptSegD(px,py,ax,ay,bx,by) {
    var dx=bx-ax, dy=by-ay, ls=dx*dx+dy*dy;
    if (ls===0) return Math.hypot(px-ax,py-ay);
    var t=Math.max(0,Math.min(1,((px-ax)*dx+(py-ay)*dy)/ls));
    return Math.hypot(px-(ax+t*dx),py-(ay+t*dy));
  }
  /** @param {number} px @param {number} py @param {number} ax @param {number} ay @param {number} bx @param {number} by */
  function projSeg(px,py,ax,ay,bx,by) {
    var dx=bx-ax, dy=by-ay, ls=dx*dx+dy*dy;
    if (ls===0) return {x:ax,y:ay};
    var t=Math.max(0,Math.min(1,((px-ax)*dx+(py-ay)*dy)/ls));
    return {x:ax+t*dx, y:ay+t*dy};
  }

  // -- Spatial index --
  function invalidBB() { polyBBDirty = true; }
  function ensureBB() {
    if (!polyBBDirty) return;
    polyBB = [];
    for (var ai=0; ai<annotations.length; ai++) {
      var a = annotations[ai];
      if (a.type!=="polygon") { polyBB.push(null); continue; }
      var x1=1e9,y1=1e9,x2=-1e9,y2=-1e9;
      for (var k=0; k<a.points.length; k+=2) {
        if (a.points[k]<x1) x1=a.points[k]; if (a.points[k+1]<y1) y1=a.points[k+1];
        if (a.points[k]>x2) x2=a.points[k]; if (a.points[k+1]>y2) y2=a.points[k+1];
      }
      polyBB.push({x1:x1,y1:y1,x2:x2,y2:y2});
    }
    polyBBDirty = false;
  }

  // -- Hit testing --
  /** @param {number} ix @param {number} iy @returns {number} */
  function hitBox(ix, iy) {
    for (var i=annotations.length-1; i>=0; i--) {
      var a=annotations[i]; if (a.type!=="box") continue;
      if (ix>=a.points[0]&&ix<=a.points[2]&&iy>=a.points[1]&&iy<=a.points[3]) return i;
    }
    return -1;
  }
  /** @param {number} ix @param {number} iy @returns {number} */
  function hitPoly(ix, iy) {
    for (var i=annotations.length-1; i>=0; i--) {
      var a=annotations[i]; if (a.type!=="polygon") continue;
      if (pip(ix,iy,a.points)) return i;
    }
    return -1;
  }
  /** @param {number} cx @param {number} cy @param {number} thr @returns {{annoIdx:number,vertIdx:number,dist:number}|null} */
  function findVert(cx, cy, thr) {
    ensureBB();
    var qp=c2i(cx,cy), imgT=thr/(scl||1);
    /** @type {{annoIdx:number,vertIdx:number,dist:number}|null} */ var best=null;
    var bestD=thr;
    for (var i=0; i<annotations.length; i++) {
      var a=annotations[i]; if (a.type!=="polygon") continue;
      var bb=polyBB[i];
      if (bb&&(qp.x+imgT<bb.x1||qp.x-imgT>bb.x2||qp.y+imgT<bb.y1||qp.y-imgT>bb.y2)) continue;
      for (var k=0; k<a.points.length; k+=2) {
        var vc=i2c(a.points[k],a.points[k+1]);
        var d=Math.hypot(cx-vc.x,cy-vc.y);
        if (d<bestD) { bestD=d; best={annoIdx:i,vertIdx:k/2,dist:d}; }
      }
    }
    return best;
  }
  /** @param {number} cx @param {number} cy @param {number} thr @param {number} [restrict] @returns {{annoIdx:number,edgeIdx:number,point:{x:number,y:number}}|null} */
  function findEdge(cx, cy, thr, restrict) {
    ensureBB();
    var qp=c2i(cx,cy), imgT=thr/(scl||1);
    /** @type {{annoIdx:number,edgeIdx:number,point:{x:number,y:number}}|null} */ var best=null;
    var bestD=thr;
    for (var i=0; i<annotations.length; i++) {
      if (restrict!==undefined && i!==restrict) continue;
      var a=annotations[i]; if (a.type!=="polygon") continue;
      var bb=polyBB[i];
      if (bb&&(qp.x+imgT<bb.x1||qp.x-imgT>bb.x2||qp.y+imgT<bb.y1||qp.y-imgT>bb.y2)) continue;
      var n=a.points.length/2;
      for (var ei=0; ei<n; ei++) {
        var ax2=a.points[ei*2],ay2=a.points[ei*2+1];
        var bx2=a.points[((ei+1)%n)*2],by2=a.points[((ei+1)%n)*2+1];
        var ac=i2c(ax2,ay2), bc=i2c(bx2,by2);
        var d=ptSegD(cx,cy,ac.x,ac.y,bc.x,bc.y);
        if (d<bestD) {
          var pr=projSeg(cx,cy,ac.x,ac.y,bc.x,bc.y);
          var ip=c2i(pr.x,pr.y), cp=clampI(ip.x,ip.y);
          bestD=d; best={annoIdx:i,edgeIdx:ei,point:cp};
        }
      }
    }
    return best;
  }

  // -- Snapping --
  /** @param {number} ix @param {number} iy @param {number} snapPx @returns {{x:number,y:number,snapped:boolean}} */
  function maybeSnap(ix, iy, snapPx) {
    var imgR = snapPx / (scl || 1);
    /** @type {{x:number,y:number,snapped:boolean}} */ var best = {x:ix,y:iy,snapped:false};
    var bestD = imgR;
    for (var i=0; i<annotations.length; i++) {
      var a=annotations[i]; if (a.type!=="polygon") continue;
      for (var k=0; k<a.points.length; k+=2) {
        var d=Math.hypot(ix-a.points[k], iy-a.points[k+1]);
        if (d<bestD) { bestD=d; best={x:a.points[k],y:a.points[k+1],snapped:true}; }
      }
    }
    return best;
  }

  // -- Undo / Redo --
  function pushUndo() {
    var snap = JSON.stringify(annotations);
    undoStack.push(snap);
    if (undoStack.length > UNDO_LIMIT) undoStack.shift();
    redoStack.length = 0;
  }
  function undo() {
    if (undoStack.length===0) return;
    redoStack.push(JSON.stringify(annotations));
    var s = undoStack.pop();
    if (s !== undefined) annotations = JSON.parse(s);
    selectedIdx = -1; invalidBB(); render();
  }
  function redo() {
    if (redoStack.length===0) return;
    undoStack.push(JSON.stringify(annotations));
    var s = redoStack.pop();
    if (s !== undefined) annotations = JSON.parse(s);
    selectedIdx = -1; invalidBB(); render();
  }
  function clearDrag() {
    isDragging=false; dragAnnIdx=-1; dragVertIdx=-1; dragOffX=0; dragOffY=0; boxDragCorner=-1;
  }

  // -- Zoom --
  /** @param {number} target @returns {number} */
  function nearZoomIdx(target) {
    var best=0;
    for (var i=1; i<ZOOM_LEVELS.length; i++) {
      if (Math.abs(ZOOM_LEVELS[i]-target)<Math.abs(ZOOM_LEVELS[best]-target)) best=i;
    }
    return best;
  }
  /** @param {number} dir @param {number} cx @param {number} cy */
  function zoomStep(dir, cx, cy) {
    var cur=nearZoomIdx(scl), ni=Math.max(0,Math.min(ZOOM_LEVELS.length-1,cur+dir));
    var ns=ZOOM_LEVELS[ni];
    var ip=c2i(cx,cy);
    scl=ns; offX=cx-ip.x*scl; offY=cy-ip.y*scl;
    render();
  }
  function fitToView() {
    if (imageW<=0||imageH<=0) return;
    var cw=canvas.width, ch=canvas.height;
    scl=Math.min(cw/imageW, ch/imageH);
    offX=(cw-imageW*scl)/2; offY=(ch-imageH*scl)/2;
    render();
  }
  function updateZoomDisp() {
    var el=document.getElementById("zoom-display");
    if (el) el.textContent=Math.round(scl*100)+"%";
  }

  // -- Mode switching --
  /** @param {string} m */
  function setMode(m) {
    mode=m; clearDrag();
    curPoly=[]; curPolyStream=false;
    selectedIdx=-1;
    samPoints=[]; samBox=null; samPreviewPoly=null; samBusy=false;
    updateBtnStates(); updateStatus(); render();
  }
  function updateBtnStates() {
    document.querySelectorAll(".tool-btn").forEach(function(b) {
      if (b instanceof HTMLElement) {
        b.classList.toggle("active", b.dataset.mode===mode);
      }
    });
  }
  /** @param {string} [msg] */
  function updateStatus(msg) {
    var el=document.getElementById("status");
    if (!el) return;
    if (msg) { el.textContent=msg; return; }
    var t=mode==="box"?"Box":mode==="polygon"?"Polygon":mode==="select"?"Select":
          mode==="vertex"?"Vertex":mode==="navigate"?"Navigate":mode==="stream"?"Stream":
          mode==="sam_point"?"SAM Point":mode==="sam_box"?"SAM Box":mode;
    el.textContent=t+" | Class: "+(classes[currentClassId]||currentClassId)+" | Annotations: "+annotations.length;
  }

  // -- Delete --
  function deleteSelected() {
    if (selectedIdx<0||selectedIdx>=annotations.length) return;
    pushUndo();
    annotations.splice(selectedIdx,1);
    selectedIdx=-1; invalidBB(); render();
  }

  // -- Mouse handlers --
  /** @param {MouseEvent} e */
  function onMouseDown(e) {
    var r=canvas.getBoundingClientRect(), cx=e.clientX-r.left, cy=e.clientY-r.top;
    lastCX=cx; lastCY=cy;
    if (e.button===1) { isPanning=true; panStartX=cx; panStartY=cy; panOffX0=offX; panOffY0=offY; e.preventDefault(); return; }
    if (e.button!==0) return;
    var ip=c2i(cx,cy), im=clampI(ip.x,ip.y);

    if (mode==="navigate") { isPanning=true; panStartX=cx; panStartY=cy; panOffX0=offX; panOffY0=offY; return; }

    if (mode==="box") {
      boxStartImg={x:im.x,y:im.y}; boxEndImg=null; isDragging=true; return;
    }

    if (mode==="polygon") {
      if (curPolyStream) { curPoly.push(im.x,im.y); render(); return; }
      if (curPoly.length>=6) {
        var fst=i2c(curPoly[0],curPoly[1]);
        if (Math.hypot(cx-fst.x,cy-fst.y)<CLOSE_DISTANCE) { closePoly(); return; }
      }
      var sn=maybeSnap(im.x,im.y,SNAP_DISTANCE);
      curPoly.push(sn.x,sn.y); snapActive=sn.snapped; render(); return;
    }

    if (mode==="stream") {
      if (curPoly.length===0) { curPoly.push(im.x,im.y); curPolyStream=true; isDragging=true; render(); return; }
    }

    if (mode==="select") {
      // try hit annotations
      var hi=hitBox(im.x,im.y);
      if (hi<0) hi=hitPoly(im.x,im.y);
      if (hi>=0) {
        selectedIdx=hi;
        isDragging=true; dragAnnIdx=hi;
        dragOffX=im.x-annotations[hi].points[0]; dragOffY=im.y-annotations[hi].points[1];
        // check box corners for resize
        if (annotations[hi].type==="box") {
          var b=annotations[hi].points;
          var corners=[{x:b[0],y:b[1]},{x:b[2],y:b[1]},{x:b[2],y:b[3]},{x:b[0],y:b[3]}];
          for (var ci=0;ci<4;ci++) {
            var cc=i2c(corners[ci].x,corners[ci].y);
            if (Math.hypot(cx-cc.x,cy-cc.y)<8) { boxDragCorner=ci; break; }
          }
        }
        render(); return;
      }
      selectedIdx=-1; render(); return;
    }

    if (mode==="vertex") {
      var fv=findVert(cx,cy,SNAP_DISTANCE);
      if (fv) {
        pushUndo();
        isDragging=true; dragAnnIdx=fv.annoIdx; dragVertIdx=fv.vertIdx;
        selectedIdx=fv.annoIdx; render(); return;
      }
      // try insert vertex on edge
      var fe=findEdge(cx,cy,SNAP_DISTANCE);
      if (fe) {
        pushUndo();
        var pa=annotations[fe.annoIdx].points;
        var ins=(fe.edgeIdx+1)*2;
        pa.splice(ins,0,fe.point.x,fe.point.y);
        isDragging=true; dragAnnIdx=fe.annoIdx; dragVertIdx=ins/2;
        selectedIdx=fe.annoIdx; invalidBB(); render(); return;
      }
      selectedIdx=-1; render();
    }

    if (mode==="sam_point" && !samBusy) {
      // Click adds a foreground point; right-click adds background (handled in onRightClick)
      samPoints.push({x:im.x, y:im.y, label:1});
      samBusy = true;
      updateStatus("SAM: predicting...");
      postToHost("sam_request", {
        type: "point",
        points: samPoints.map(function(p){ return {x:p.x, y:p.y, label:p.label}; })
      });
      render(); return;
    }

    if (mode==="sam_box" && !samBusy) {
      boxStartImg={x:im.x, y:im.y}; boxEndImg=null; isDragging=true; return;
    }
  }

  /** @param {MouseEvent} e */
  function onMouseMove(e) {
    var r=canvas.getBoundingClientRect(), cx=e.clientX-r.left, cy=e.clientY-r.top;
    mouseCanvasX=cx; mouseCanvasY=cy;

    if (isPanning) {
      offX=panOffX0+(cx-panStartX); offY=panOffY0+(cy-panStartY); render(); return;
    }

    var ip=c2i(cx,cy), im=clampI(ip.x,ip.y);
    mouseImgX=im.x; mouseImgY=im.y;

    if (mode==="box" && isDragging && boxStartImg) {
      boxEndImg={x:im.x,y:im.y}; render(); return;
    }

    if (mode==="sam_box" && isDragging && boxStartImg) {
      boxEndImg={x:im.x,y:im.y}; render(); return;
    }

    if (mode==="stream" && isDragging && curPolyStream) {
      var last=curPoly.length>=2?{x:curPoly[curPoly.length-2],y:curPoly[curPoly.length-1]}:null;
      if (!last || Math.hypot(im.x-last.x,im.y-last.y)>STREAM_MIN_DIST/(scl||1)) {
        curPoly.push(im.x,im.y); render();
      }
      return;
    }

    if (mode==="select" && isDragging && dragAnnIdx>=0) {
      var a=annotations[dragAnnIdx];
      pushUndo();
      if (a.type==="box" && boxDragCorner>=0) {
        // resize box corner
        var bp=a.points;
        if (boxDragCorner===0) { bp[0]=im.x; bp[1]=im.y; }
        else if (boxDragCorner===1) { bp[2]=im.x; bp[1]=im.y; }
        else if (boxDragCorner===2) { bp[2]=im.x; bp[3]=im.y; }
        else { bp[0]=im.x; bp[3]=im.y; }
      } else {
        // translate whole annotation
        var dx=im.x-dragOffX-a.points[0], dy=im.y-dragOffY-a.points[1];
        for (var k=0;k<a.points.length;k+=2) { a.points[k]+=dx; a.points[k+1]+=dy; }
      }
      invalidBB(); render(); return;
    }

    if (mode==="vertex" && isDragging && dragAnnIdx>=0 && dragVertIdx>=0) {
      var pa2=annotations[dragAnnIdx].points;
      pa2[dragVertIdx*2]=im.x; pa2[dragVertIdx*2+1]=im.y;
      invalidBB(); render(); return;
    }

    // cursor update for polygon — show snap preview
    if (mode==="polygon" && curPoly.length>=2) {
      var sn2=maybeSnap(im.x,im.y,SNAP_DISTANCE);
      snapActive=sn2.snapped;
    }
    render();
  }

  /** @param {MouseEvent} e */
  function onMouseUp(e) {
    if (e.button===1) { isPanning=false; return; }
    if (isPanning && mode==="navigate") { isPanning=false; return; }

    if (mode==="box" && isDragging && boxStartImg && boxEndImg) {
      var x1=Math.min(boxStartImg.x,boxEndImg.x), y1=Math.min(boxStartImg.y,boxEndImg.y);
      var x2=Math.max(boxStartImg.x,boxEndImg.x), y2=Math.max(boxStartImg.y,boxEndImg.y);
      if (Math.abs(x2-x1)>2&&Math.abs(y2-y1)>2) {
        pushUndo();
        annotations.push({type:"box",classId:currentClassId,points:[x1,y1,x2,y2]});
        invalidBB();
      }
      boxStartImg=null; boxEndImg=null; isDragging=false; render(); return;
    }

    if (mode==="stream" && isDragging && curPolyStream) {
      isDragging=false;
      if (curPoly.length>=6) { closePoly(); } else { curPoly=[]; curPolyStream=false; render(); }
      return;
    }

    if (mode==="sam_box" && isDragging && boxStartImg && boxEndImg && !samBusy) {
      var sx1=Math.min(boxStartImg.x,boxEndImg.x), sy1=Math.min(boxStartImg.y,boxEndImg.y);
      var sx2=Math.max(boxStartImg.x,boxEndImg.x), sy2=Math.max(boxStartImg.y,boxEndImg.y);
      if (Math.abs(sx2-sx1)>2 && Math.abs(sy2-sy1)>2) {
        samBusy = true;
        samBox = {x1:sx1, y1:sy1, x2:sx2, y2:sy2};
        updateStatus("SAM: predicting from box...");
        postToHost("sam_request", {type:"box", box:{x1:sx1,y1:sy1,x2:sx2,y2:sy2}});
      }
      boxStartImg=null; boxEndImg=null; isDragging=false; render(); return;
    }

    clearDrag(); render();
  }

  // -- Close polygon --
  function closePoly() {
    if (curPoly.length<6) { curPoly=[]; curPolyStream=false; render(); return; }
    pushUndo();
    annotations.push({type:"polygon",classId:currentClassId,points:curPoly.slice()});
    curPoly=[]; curPolyStream=false; snapActive=false; invalidBB(); render();
  }

  // -- Double-click: close polygon or delete vertex --
  /** @param {MouseEvent} e */
  function onDblClick(e) {
    var r=canvas.getBoundingClientRect(), cx=e.clientX-r.left, cy=e.clientY-r.top;
    if (mode==="polygon" && curPoly.length>=6) { closePoly(); return; }
    if (mode==="vertex") {
      var fv=findVert(cx,cy,SNAP_DISTANCE);
      if (fv && annotations[fv.annoIdx].points.length>6) {
        pushUndo();
        annotations[fv.annoIdx].points.splice(fv.vertIdx*2,2);
        invalidBB(); render();
      }
    }
  }

  // -- Right click: cancel current drawing or add SAM background point --
  /** @param {MouseEvent} e */
  function onRightClick(e) {
    e.preventDefault();
    if (mode==="sam_point" && !samBusy) {
      var r2=canvas.getBoundingClientRect(), cx2=e.clientX-r2.left, cy2=e.clientY-r2.top;
      var ip2=c2i(cx2,cy2), im2=clampI(ip2.x,ip2.y);
      samPoints.push({x:im2.x, y:im2.y, label:0});
      samBusy = true;
      updateStatus("SAM: predicting (with background)...");
      postToHost("sam_request", {
        type: "point",
        points: samPoints.map(function(p){ return {x:p.x, y:p.y, label:p.label}; })
      });
      render(); return;
    }
    if (mode==="polygon" || mode==="stream") {
      curPoly=[]; curPolyStream=false; snapActive=false; isDragging=false; render();
    }
  }

  // -- Wheel: zoom or pan --
  /** @param {WheelEvent} e */
  function onWheel(e) {
    e.preventDefault();
    var r=canvas.getBoundingClientRect(), cx=e.clientX-r.left, cy=e.clientY-r.top;
    if (e.ctrlKey) {
      zoomStep(e.deltaY<0?1:-1, cx, cy);
    } else if (e.shiftKey) {
      offX -= e.deltaY; render();
    } else {
      offY -= e.deltaY; render();
    }
  }

  // -- Keyboard shortcuts --
  /** @param {KeyboardEvent} e */
  function onKey(e) {
    if (e.ctrlKey && e.key==="z") { e.preventDefault(); undo(); return; }
    if (e.ctrlKey && e.key==="y") { e.preventDefault(); redo(); return; }
    if (e.key==="Delete"||e.key==="Backspace") { deleteSelected(); return; }
    if (e.key==="Enter" && samPreviewPoly && samPreviewPoly.length>=6) {
      pushUndo();
      annotations.push({type:"polygon",classId:currentClassId,points:samPreviewPoly.slice()});
      samPreviewPoly=null; samPoints=[]; samBox=null; invalidBB();
      updateStatus("SAM: polygon accepted"); render(); return;
    }
    if (e.key==="Escape") {
      if (samPreviewPoly) { samPreviewPoly=null; samPoints=[]; samBox=null; samBusy=false; updateStatus("SAM: cancelled"); render(); return; }
      if (curPoly.length>0) { curPoly=[]; curPolyStream=false; render(); return; }
      selectedIdx=-1; render(); return;
    }
    var kl=e.key.toLowerCase();
    if (kl==="b") { setMode("box"); return; }
    if (kl==="p") { setMode("polygon"); return; }
    if (kl==="s") { setMode("select"); return; }
    if (kl==="v") { setMode("vertex"); return; }
    if (kl==="n") { setMode("navigate"); return; }
    if (kl==="m") { setMode("stream"); return; }
    if (kl==="a") { setMode("sam_point"); return; }
    if (kl==="q") { setMode("sam_box"); return; }
    if (kl==="t") { showAnnotations=!showAnnotations; render(); return; }
    if (kl==="r") { showPredictions=!showPredictions; render(); return; }
    if (kl==="h") { showHelp=!showHelp; render(); return; }
    if (kl==="f") { fitToView(); return; }
    // class shortcuts 0-9
    var num=parseInt(e.key);
    if (!isNaN(num) && num>=0 && num<=9) { currentClassId=num; updateStatus(); render(); }
    // arrow keys for pan
    var PAN_STEP=50;
    if (e.key==="ArrowLeft") { offX+=PAN_STEP; render(); }
    if (e.key==="ArrowRight") { offX-=PAN_STEP; render(); }
    if (e.key==="ArrowUp") { offY+=PAN_STEP; render(); }
    if (e.key==="ArrowDown") { offY-=PAN_STEP; render(); }
  }

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

  // -- Render --
  function render() {
    updateZoomDisp();
    var cw=canvas.width=canvas.clientWidth, ch=canvas.height=canvas.clientHeight;
    ctx.clearRect(0,0,cw,ch);

    // draw image
    if (img.complete && img.naturalWidth>0) {
      ctx.drawImage(img, offX, offY, imageW*scl, imageH*scl);
    }

    // draw predictions (behind annotations)
    if (showPredictions && predAnnotations.length>0) {
      ctx.save(); ctx.globalAlpha=0.35;
      for (var pi=0; pi<predAnnotations.length; pi++) {
        var pa=predAnnotations[pi];
        var col=CLASS_COLORS[pa.classId % CLASS_COLORS.length];
        ctx.strokeStyle=col; ctx.lineWidth=1;
        if (pa.type==="box") {
          var pc=i2c(pa.points[0],pa.points[1]), pc2=i2c(pa.points[2],pa.points[3]);
          ctx.strokeRect(pc.x,pc.y,pc2.x-pc.x,pc2.y-pc.y);
        } else if (pa.type==="polygon" && pa.points.length>=4) {
          ctx.beginPath();
          var pf=i2c(pa.points[0],pa.points[1]);
          ctx.moveTo(pf.x,pf.y);
          for (var pk=2; pk<pa.points.length; pk+=2) {
            var pp=i2c(pa.points[pk],pa.points[pk+1]);
            ctx.lineTo(pp.x,pp.y);
          }
          ctx.closePath(); ctx.stroke();
        }
      }
      ctx.restore();
    }

    if (!showAnnotations) return;

    // draw annotations
    for (var i=0; i<annotations.length; i++) {
      var a=annotations[i];
      var col2=CLASS_COLORS[a.classId % CLASS_COLORS.length];
      var isSel=(i===selectedIdx);
      ctx.strokeStyle=isSel?"#ff0":col2;
      ctx.lineWidth=isSel?3:2;
      ctx.fillStyle=col2.replace(")",",0.15)").replace("rgb","rgba");

      if (a.type==="box") {
        var b1=i2c(a.points[0],a.points[1]), b2=i2c(a.points[2],a.points[3]);
        var bw=b2.x-b1.x, bh=b2.y-b1.y;
        ctx.fillRect(b1.x,b1.y,bw,bh);
        ctx.strokeRect(b1.x,b1.y,bw,bh);
        if (isSel) {
          // draw corner handles
          var corners2=[b1,{x:b2.x,y:b1.y},b2,{x:b1.x,y:b2.y}];
          for (var ch2=0;ch2<4;ch2++) {
            ctx.fillStyle="#ff0"; ctx.fillRect(corners2[ch2].x-4,corners2[ch2].y-4,8,8);
          }
        }
        haloText(ctx, (classes[a.classId]||""+a.classId), b1.x+2, b1.y-4);
      }

      if (a.type==="polygon" && a.points.length>=4) {
        ctx.beginPath();
        var f2=i2c(a.points[0],a.points[1]);
        ctx.moveTo(f2.x,f2.y);
        for (var k2=2; k2<a.points.length; k2+=2) {
          var v=i2c(a.points[k2],a.points[k2+1]);
          ctx.lineTo(v.x,v.y);
        }
        ctx.closePath(); ctx.fill(); ctx.stroke();

        // draw vertices if selected or in vertex mode
        if (isSel || mode==="vertex") {
          for (var vk=0; vk<a.points.length; vk+=2) {
            var vp=i2c(a.points[vk],a.points[vk+1]);
            ctx.fillStyle=(isSel&&mode==="vertex")?"#ff0":"#fff";
            ctx.fillRect(vp.x-3,vp.y-3,6,6);
            ctx.strokeStyle="#000"; ctx.lineWidth=1;
            ctx.strokeRect(vp.x-3,vp.y-3,6,6);
          }
          ctx.strokeStyle=isSel?"#ff0":col2; ctx.lineWidth=isSel?3:2;
        }
        haloText(ctx, (classes[a.classId]||""+a.classId), f2.x+2, f2.y-4);
      }
    }

    // draw current polygon being drawn
    if (curPoly.length>=2) {
      ctx.strokeStyle="#0f0"; ctx.lineWidth=2;
      ctx.beginPath();
      var cf=i2c(curPoly[0],curPoly[1]);
      ctx.moveTo(cf.x,cf.y);
      for (var ck=2; ck<curPoly.length; ck+=2) {
        var cv=i2c(curPoly[ck],curPoly[ck+1]);
        ctx.lineTo(cv.x,cv.y);
      }
      // rubber band to mouse
      ctx.lineTo(mouseCanvasX,mouseCanvasY);
      ctx.stroke();

      // draw vertices
      for (var cv2=0; cv2<curPoly.length; cv2+=2) {
        var cvp=i2c(curPoly[cv2],curPoly[cv2+1]);
        ctx.fillStyle="#0f0"; ctx.fillRect(cvp.x-3,cvp.y-3,6,6);
      }

      // close indicator
      if (curPoly.length>=6) {
        var clp=i2c(curPoly[0],curPoly[1]);
        if (Math.hypot(mouseCanvasX-clp.x,mouseCanvasY-clp.y)<CLOSE_DISTANCE) {
          ctx.beginPath();
          ctx.arc(clp.x,clp.y,CLOSE_DISTANCE,0,Math.PI*2);
          ctx.strokeStyle="#0f0"; ctx.lineWidth=1; ctx.stroke();
        }
      }
    }

    // draw box preview
    if (mode==="box" && isDragging && boxStartImg && boxEndImg) {
      var bs=i2c(boxStartImg.x,boxStartImg.y), be=i2c(boxEndImg.x,boxEndImg.y);
      ctx.strokeStyle="#0f0"; ctx.lineWidth=2; ctx.setLineDash([5,5]);
      ctx.strokeRect(bs.x,bs.y,be.x-bs.x,be.y-bs.y);
      ctx.setLineDash([]);
    }

    // snap indicator
    if (snapActive && mode==="polygon") {
      ctx.beginPath();
      ctx.arc(mouseCanvasX,mouseCanvasY,8,0,Math.PI*2);
      ctx.strokeStyle="#ff0"; ctx.lineWidth=2; ctx.stroke();
    }

    // SAM preview polygon
    if (samPreviewPoly && samPreviewPoly.length>=4) {
      ctx.strokeStyle="#00ffff"; ctx.lineWidth=2; ctx.setLineDash([6,4]);
      ctx.fillStyle="rgba(0,255,255,0.15)";
      ctx.beginPath();
      var sf=i2c(samPreviewPoly[0],samPreviewPoly[1]);
      ctx.moveTo(sf.x,sf.y);
      for (var sk=2; sk<samPreviewPoly.length; sk+=2) {
        var sv=i2c(samPreviewPoly[sk],samPreviewPoly[sk+1]);
        ctx.lineTo(sv.x,sv.y);
      }
      ctx.closePath(); ctx.fill(); ctx.stroke();
      ctx.setLineDash([]);
      haloText(ctx, "SAM preview (Enter=accept, Esc=cancel)", sf.x, sf.y-8);
    }

    // SAM point markers
    if ((mode==="sam_point"||mode==="sam_box") && samPoints.length>0) {
      for (var spi=0; spi<samPoints.length; spi++) {
        var sp=i2c(samPoints[spi].x, samPoints[spi].y);
        ctx.beginPath();
        ctx.arc(sp.x,sp.y,6,0,Math.PI*2);
        ctx.fillStyle=samPoints[spi].label===1?"#0f0":"#f00";
        ctx.fill();
        ctx.strokeStyle="#fff"; ctx.lineWidth=1.5; ctx.stroke();
      }
    }

    // SAM busy indicator
    if (samBusy) {
      haloText(ctx, "SAM processing...", 20, cw > 0 ? 40 : 40);
    }

    // help overlay
    if (showHelp) renderHelp();

    updateStatus();
  }

  // -- Save: dual detect + segment YOLO format --
  function save() {
    if (imageW<=0||imageH<=0) return;
    /** @type {string[]} */ var detectLines = [];
    /** @type {string[]} */ var segmentLines = [];
    for (var i=0; i<annotations.length; i++) {
      var a=annotations[i];
      if (a.type==="box") {
        var cx2=(a.points[0]+a.points[2])/(2*imageW);
        var cy2=(a.points[1]+a.points[3])/(2*imageH);
        var w=(a.points[2]-a.points[0])/imageW;
        var h=(a.points[3]-a.points[1])/imageH;
        detectLines.push(a.classId+" "+cx2.toFixed(6)+" "+cy2.toFixed(6)+" "+w.toFixed(6)+" "+h.toFixed(6));
      }
      if (a.type==="polygon" && a.points.length>=6) {
        // segment format: classId x1 y1 x2 y2 ...
        var parts=[String(a.classId)];
        for (var k=0; k<a.points.length; k+=2) {
          parts.push((a.points[k]/imageW).toFixed(6));
          parts.push((a.points[k+1]/imageH).toFixed(6));
        }
        segmentLines.push(parts.join(" "));
        // also add bounding box to detect
        var bx1=1e9,by1=1e9,bx2=-1e9,by2=-1e9;
        for (var bk=0; bk<a.points.length; bk+=2) {
          if (a.points[bk]<bx1) bx1=a.points[bk]; if (a.points[bk+1]<by1) by1=a.points[bk+1];
          if (a.points[bk]>bx2) bx2=a.points[bk]; if (a.points[bk+1]>by2) by2=a.points[bk+1];
        }
        var dcx=(bx1+bx2)/(2*imageW), dcy=(by1+by2)/(2*imageH);
        var dw=(bx2-bx1)/imageW, dh=(by2-by1)/imageH;
        detectLines.push(a.classId+" "+dcx.toFixed(6)+" "+dcy.toFixed(6)+" "+dw.toFixed(6)+" "+dh.toFixed(6));
      }
    }
    postToHost("save_detect_annotations", {content:detectLines.join("\n")});
    postToHost("save_segment_annotations", {content:segmentLines.join("\n")});
    updateStatus("Saved "+detectLines.length+" detect + "+segmentLines.length+" segment labels");
  }

  // -- Parse YOLO labels --
  /** @param {string} text @param {string} format @returns {Array<{type:string,classId:number,points:number[]}>} */
  function parseYolo(text, format) {
    /** @type {Array<{type:string,classId:number,points:number[]}>} */ var result = [];
    var lines=text.split("\n").filter(function(l){return l.trim().length>0;});
    for (var li=0; li<lines.length; li++) {
      var vals=lines[li].trim().split(/\s+/).map(Number);
      if (vals.length<5) continue;
      var cid=vals[0];
      if (format==="detect") {
        var cx3=vals[1]*imageW, cy3=vals[2]*imageH, w2=vals[3]*imageW, h2=vals[4]*imageH;
        result.push({type:"box",classId:cid,points:[cx3-w2/2,cy3-h2/2,cx3+w2/2,cy3+h2/2]});
      } else {
        // segment: classId x1 y1 x2 y2 ...
        /** @type {number[]} */ var pts=[];
        for (var vi=1; vi<vals.length; vi+=2) {
          if (vi+1<vals.length) { pts.push(vals[vi]*imageW, vals[vi+1]*imageH); }
        }
        if (pts.length>=6) result.push({type:"polygon",classId:cid,points:pts});
      }
    }
    return result;
  }

  // -- Load image --
  /** @param {string} uri */
  function loadImage(uri) {
    img.onload = function() {
      imageW=img.naturalWidth; imageH=img.naturalHeight;
      fitToView();
    };
    img.src=uri;
  }

  // -- Set classes --
  /** @param {string[]} classList */
  function setClasses(classList) {
    classes = classList;
    currentClassId = 0;
    updateStatus();
  }

  // -- Help overlay --
  function renderHelp() {
    ctx.save();
    ctx.fillStyle="rgba(0,0,0,0.75)";
    ctx.fillRect(20,20,320,380);
    ctx.fillStyle="#fff"; ctx.font="bold 14px monospace";
    var lines=[
      "KEYBOARD SHORTCUTS",
      "─────────────────────",
      "B   Box mode",
      "P   Polygon mode",
      "S   Select mode",
      "V   Vertex mode",
      "N   Navigate (pan)",
      "M   Stream (freehand)",
      "A   SAM point mode",
      "Q   SAM box mode",
      "0-9 Set class ID",
      "T   Toggle annotations",
      "R   Toggle predictions",
      "H   Toggle this help",
      "F   Fit to view",
      "Ctrl+Z  Undo",
      "Ctrl+Y  Redo",
      "Del     Delete selected",
      "Esc     Cancel / deselect",
      "─────────────────────",
      "Scroll: V-pan  Shift: H-pan",
      "Ctrl+Scroll: Zoom",
      "Middle-click: Pan"
    ];
    for (var hi=0; hi<lines.length; hi++) {
      ctx.fillText(lines[hi], 30, 45+hi*17);
    }
    ctx.restore();
  }

  // -- Host message handler --
  onHostMessage(function(/** @type {{type:string, [key:string]:any}} */ msg) {
    switch (msg.type) {
      case "loadImage":
        if (msg.uri) loadImage(msg.uri);
        break;
      case "setClasses":
        if (Array.isArray(msg.classes)) setClasses(msg.classes);
        break;
      case "loadLabels":
        if (typeof msg.text==="string" && typeof msg.format==="string") {
          var parsed=parseYolo(msg.text, msg.format);
          if (msg.format==="detect") {
            // merge boxes
            for (var pi2=0; pi2<parsed.length; pi2++) {
              annotations.push(parsed[pi2]);
            }
          } else {
            for (var pi3=0; pi3<parsed.length; pi3++) {
              annotations.push(parsed[pi3]);
            }
          }
          invalidBB(); render();
        }
        break;
      case "loadPredictions":
        if (typeof msg.text==="string" && typeof msg.format==="string") {
          predAnnotations = parseYolo(msg.text, msg.format);
          render();
        }
        break;
      case "clearAnnotations":
        pushUndo();
        annotations=[]; selectedIdx=-1; invalidBB(); render();
        break;
      case "highlight":
        // Highlight specific annotation indices — flash selection ring
        if (Array.isArray(msg.indices) && msg.indices.length > 0) {
          selectedIdx = msg.indices[0];
          render();
        }
        break;
      case "sam_result":
        samBusy = false;
        if (msg.error) {
          updateStatus("SAM error: " + msg.error);
          break;
        }
        if (Array.isArray(msg.polygon) && msg.polygon.length >= 3) {
          /** @type {number[]} */ var samPts = [];
          for (var si = 0; si < msg.polygon.length; si++) {
            samPts.push(msg.polygon[si].x, msg.polygon[si].y);
          }
          samPreviewPoly = samPts;
          updateStatus("SAM: preview ready — Enter to accept, Esc to reject, click to refine");
        } else {
          updateStatus("SAM: no mask found");
          samPreviewPoly = null;
        }
        render();
        break;
    }
  });

  // -- Toolbar wiring --
  document.querySelectorAll(".tool-btn").forEach(function(btn) {
    if (btn instanceof HTMLElement && btn.dataset.mode) {
      btn.addEventListener("click", function() { setMode(/** @type {string} */(btn.dataset.mode)); });
    }
  });
  var saveBtn=document.getElementById("save-btn");
  if (saveBtn) saveBtn.addEventListener("click", save);
  var fitBtn=document.getElementById("fit-btn");
  if (fitBtn) fitBtn.addEventListener("click", fitToView);
  var helpBtn=document.getElementById("help-btn");
  if (helpBtn) helpBtn.addEventListener("click", function() { showHelp=!showHelp; render(); });
  var undoBtn=document.getElementById("undo-btn");
  if (undoBtn) undoBtn.addEventListener("click", undo);
  var redoBtn=document.getElementById("redo-btn");
  if (redoBtn) redoBtn.addEventListener("click", redo);
  var deleteBtn=document.getElementById("delete-btn");
  if (deleteBtn) deleteBtn.addEventListener("click", deleteSelected);

  // -- Event listeners --
  canvas.addEventListener("mousedown", onMouseDown);
  canvas.addEventListener("mousemove", onMouseMove);
  canvas.addEventListener("mouseup", onMouseUp);
  canvas.addEventListener("dblclick", onDblClick);
  canvas.addEventListener("contextmenu", onRightClick);
  canvas.addEventListener("wheel", onWheel, {passive:false});
  document.addEventListener("keydown", onKey);

  // -- Resize observer --
  new ResizeObserver(function() { render(); }).observe(canvas);

  // -- Initial render --
  updateBtnStates();
  updateStatus();
  render();

})();
