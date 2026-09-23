// Shared color tokens (from the validated reference palette) and small
// helpers used across dashboard.html / alerts.html / models.html.

const IS_DARK = window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;

// Severity is an ORDERED tier (Informational -> Low -> Medium -> High -> Critical),
// so it uses the single-hue ordinal ramp, not the unordered categorical palette.
const SEVERITY_ORDER = ["Informational", "Low", "Medium", "High", "Critical"];
const SEVERITY_COLORS = IS_DARK
  ? { Informational: "#86b6ef", Low: "#5598e7", Medium: "#2a78d6", High: "#1c5cab", Critical: "#184f95" }
  : { Informational: "#86b6ef", Low: "#5598e7", Medium: "#2a78d6", High: "#1c5cab", Critical: "#104281" };

// Categorical palette in fixed order -- assigned by rank, never cycled per-category.
const CATEGORICAL_PALETTE = IS_DARK
  ? ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"]
  : ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"];

const CHART_TEXT = IS_DARK ? "#c3c2b7" : "#52514e";
const CHART_GRID = IS_DARK ? "#2c2c2a" : "#e1e0d9";

function severityBadge(severity) {
  const color = SEVERITY_COLORS[severity] || "#898781";
  return `<span class="badge"><span class="dot" style="background:${color}"></span>${severity}</span>`;
}

// Cap unordered categories at 7 + "Other" so the fixed 8-slot palette never
// has to be cycled or extended.
function foldToOther(counts, maxSlots = 7) {
  const entries = Object.entries(counts).sort((a, b) => b[1] - a[1]);
  const kept = entries.slice(0, maxSlots);
  const rest = entries.slice(maxSlots).reduce((sum, [, v]) => sum + v, 0);
  if (rest > 0) kept.push(["Other", rest]);
  return kept;
}

async function fetchJSON(url, options) {
  const res = await fetch(url, options);
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.error || `Request to ${url} failed (${res.status})`);
  }
  return res.json();
}
