/* TCIP VS Code Extension — shared webview messaging utilities */

// @ts-check

/** @type {ReturnType<typeof acquireVsCodeApi>} */
// eslint-disable-next-line no-undef
const vscode = acquireVsCodeApi();

/**
 * Send a message to the extension host.
 * @param {string} type
 * @param {Record<string, unknown>} [data]
 */
function postToHost(type, data) {
  vscode.postMessage({ type, ...data });
}

/**
 * Register a handler for messages from the extension host.
 * @param {(msg: {type: string, [key: string]: unknown}) => void} handler
 */
function onHostMessage(handler) {
  window.addEventListener("message", (event) => {
    handler(event.data);
  });
}

/**
 * Persist webview state so it survives panel hide/show.
 * @param {string} key
 * @param {unknown} value
 */
function saveState(key, value) {
  const state = vscode.getState() || {};
  state[key] = value;
  vscode.setState(state);
}

/**
 * Retrieve persisted state.
 * @param {string} key
 * @param {unknown} [defaultValue]
 * @returns {unknown}
 */
function loadState(key, defaultValue) {
  const state = vscode.getState() || {};
  return state[key] !== undefined ? state[key] : defaultValue;
}

/**
 * Format a number with locale-aware thousands separators.
 * @param {number} n
 * @returns {string}
 */
function formatNumber(n) {
  return n.toLocaleString();
}

/**
 * Clamp a number between min and max.
 * @param {number} v
 * @param {number} min
 * @param {number} max
 * @returns {number}
 */
function clamp(v, min, max) {
  return Math.max(min, Math.min(max, v));
}
