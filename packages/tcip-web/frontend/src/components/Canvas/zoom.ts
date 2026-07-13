// Discrete zoom levels mirror yolo-annotator (5% .. 1000%, 20 stops). The wheel/pinch
// path zooms continuously within [MIN_SCALE, MAX_SCALE]; the ladder remains for
// toolbar/keyboard stepping and for clamping programmatic zooms (zoomToDetection).
export const ZOOM_LEVELS = [
  0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.33, 0.5, 0.67, 0.75, 0.85, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0,
  5.0, 7.0, 10.0,
];

export const MIN_SCALE = ZOOM_LEVELS[0];
export const MAX_SCALE = ZOOM_LEVELS[ZOOM_LEVELS.length - 1];
