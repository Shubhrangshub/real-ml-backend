// explainableAI.js — Client-side Explainable AI Engine (optimized)

function mean(arr) { let s = 0; for (let i = 0; i < arr.length; i++) s += arr[i]; return arr.length ? s / arr.length : 0; }

function sampleN(arr, n) {
  if (n >= arr.length) return arr.slice();
  const res = new Array(n), used = new Set();
  for (let i = 0; i < n;) { const idx = (Math.random() * arr.length) | 0; if (!used.has(idx)) { used.add(idx); res[i++] = arr[idx]; } }
  return res;
}

function boxMuller() {
  let u = 0, v = 0;
  while (!u) u = Math.random();
  while (!v) v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function lerpHex(c1, c2, t) {
  const p = (s, o) => parseInt(s.slice(o, o + 2), 16);
  const r = Math.round(p(c1, 1) + (p(c2, 1) - p(c1, 1)) * t);
  const g = Math.round(p(c1, 3) + (p(c2, 3) - p(c1, 3)) * t);
  const b = Math.round(p(c1, 5) + (p(c2, 5) - p(c1, 5)) * t);
  return '#' + [r, g, b].map(v => Math.max(0, Math.min(255, v)).toString(16).padStart(2, '0')).join('');
}

export function valueToColor(n) {
  const t = Math.max(0, Math.min(1, n));
  return t <= 0.5 ? lerpHex('#3b82f6', '#8b5cf6', t * 2) : lerpHex('#8b5cf6', '#ec4899', (t - 0.5) * 2);
}

export function importanceColor(rank, total) {
  return lerpHex('#ec4899', '#3b82f6', total > 1 ? rank / (total - 1) : 0);
}

// ── Core SHAP (reusable scratch array, pre-computed bg/basePred) ──

function localSHAPFast(predictFn, instance, bg, basePred) {
  const p = instance.length;
  const instPred = predictFn(instance);
  const sv = new Float64Array(p);
  // Reuse a single scratch array instead of copying per-iteration
  const scratch = new Float64Array(p);
  for (let i = 0; i < p; i++) scratch[i] = instance[i];
  const bgLen = bg.length;
  for (let f = 0; f < p; f++) {
    let total = 0;
    const orig = scratch[f];
    for (let b = 0; b < bgLen; b++) { scratch[f] = bg[b][f]; total += instPred - predictFn(scratch); }
    scratch[f] = orig;
    sv[f] = total / bgLen;
  }
  // Normalize to sum to (instPred - basePred)
  let ss = 0; for (let f = 0; f < p; f++) ss += sv[f];
  const target = instPred - basePred;
  if (Math.abs(ss) > 1e-10) { const sc = target / ss; for (let f = 0; f < p; f++) sv[f] *= sc; }
  return { shapValues: Array.from(sv), basePred, instancePred: instPred };
}

// Public API (backward compat)
export function computeLocalSHAP(predictFn, instance, background, nSamples = 80) {
  const bg = sampleN(background, Math.min(nSamples, background.length));
  const basePred = mean(bg.map(b => predictFn(b)));
  return localSHAPFast(predictFn, instance, bg, basePred);
}

// ── Unified SHAP computation — single pass for all visualizations ──

export function computeAllSHAP(predictFn, data, featureNames, opts = {}) {
  const nInst = Math.min(opts.nInstances || 100, data.length);
  const nBg = Math.min(opts.nBackground || 50, data.length);
  const rowIdx = Math.min(opts.rowIndex || 0, data.length - 1);
  const depFeature = Math.min(opts.dependenceFeature || 0, featureNames.length - 1);

  // Shared background & base prediction — computed ONCE
  const bg = sampleN(data, nBg);
  let basePredSum = 0; for (const b of bg) basePredSum += predictFn(b);
  const basePred = basePredSum / bg.length;

  // Sample instances once for all views
  const instances = sampleN(data, nInst);

  // Feature ranges (loop-based, safe for large arrays)
  const ranges = new Array(featureNames.length);
  for (let i = 0; i < featureNames.length; i++) {
    let lo = Infinity, hi = -Infinity;
    for (let j = 0; j < data.length; j++) { const v = data[j][i]; if (v < lo) lo = v; if (v > hi) hi = v; }
    ranges[i] = { min: lo, max: hi };
  }

  // Compute local SHAP for all sampled instances — ONE pass
  const allShap = new Array(nInst);
  for (let si = 0; si < nInst; si++) {
    allShap[si] = localSHAPFast(predictFn, instances[si], bg, basePred);
  }

  // Global importance: mean |SHAP| per feature
  const absSum = new Float64Array(featureNames.length);
  for (let si = 0; si < nInst; si++) {
    const sv = allShap[si].shapValues;
    for (let i = 0; i < featureNames.length; i++) absSum[i] += Math.abs(sv[i]);
  }
  const importance = featureNames.map((f, i) => ({ feature: f, importance: absSum[i] / nInst }));
  importance.sort((a, b) => b.importance - a.importance);

  // Beeswarm points — derived from the same SHAP values
  const pts = [];
  for (let si = 0; si < nInst; si++) {
    const inst = instances[si], sv = allShap[si].shapValues;
    for (let fi = 0; fi < featureNames.length; fi++) {
      const r = ranges[fi];
      const norm = r.max > r.min ? (inst[fi] - r.min) / (r.max - r.min) : 0.5;
      pts.push({ feature: featureNames[fi], featureIdx: fi, shapValue: sv[fi], featureValue: inst[fi], normalizedValue: norm, color: valueToColor(norm), jitter: (Math.random() - 0.5) * 0.35 });
    }
  }

  // Local SHAP for the selected row
  const localInst = data[rowIdx];
  const local = localSHAPFast(predictFn, localInst, bg, basePred);

  // Dependence data — reuse already-computed SHAP values
  const dependence = new Array(nInst);
  for (let si = 0; si < nInst; si++) {
    dependence[si] = { featureValue: instances[si][depFeature], shapValue: allShap[si].shapValues[depFeature] };
  }

  return {
    global: { importance, basePred },
    beeswarm: { points: pts, ranges },
    local: { ...local, featureNames, instance: localInst },
    dependence,
  };
}

// Keep individual functions for backward compat & standalone use
export function computeGlobalSHAP(predictFn, data, featureNames, nInst = 50, nBg = 50) {
  const bg = sampleN(data, Math.min(nBg, data.length));
  let bpSum = 0; for (const b of bg) bpSum += predictFn(b);
  const basePred = bpSum / bg.length;
  const insts = sampleN(data, Math.min(nInst, data.length));
  const abs = new Float64Array(featureNames.length);
  for (const inst of insts) {
    const { shapValues } = localSHAPFast(predictFn, inst, bg, basePred);
    for (let i = 0; i < featureNames.length; i++) abs[i] += Math.abs(shapValues[i]);
  }
  const imp = featureNames.map((f, i) => ({ feature: f, importance: abs[i] / insts.length }));
  imp.sort((a, b) => b.importance - a.importance);
  return { importance: imp, basePred };
}

export function computeBeeswarmData(predictFn, data, featureNames, nInst = 100, nBg = 50) {
  const bg = sampleN(data, Math.min(nBg, data.length));
  let bpSum = 0; for (const b of bg) bpSum += predictFn(b);
  const basePred = bpSum / bg.length;
  const insts = sampleN(data, Math.min(nInst, data.length));
  const ranges = featureNames.map((_, i) => {
    let lo = Infinity, hi = -Infinity;
    for (let j = 0; j < data.length; j++) { const v = data[j][i]; if (v < lo) lo = v; if (v > hi) hi = v; }
    return { min: lo, max: hi };
  });
  const pts = [];
  for (const inst of insts) {
    const { shapValues } = localSHAPFast(predictFn, inst, bg, basePred);
    for (let fi = 0; fi < featureNames.length; fi++) {
      const r = ranges[fi]; const norm = r.max > r.min ? (inst[fi] - r.min) / (r.max - r.min) : 0.5;
      pts.push({ feature: featureNames[fi], featureIdx: fi, shapValue: shapValues[fi], featureValue: inst[fi], normalizedValue: norm, color: valueToColor(norm), jitter: (Math.random() - 0.5) * 0.35 });
    }
  }
  return { points: pts, ranges };
}

export function computeDependenceData(predictFn, data, featureNames, featureIndex, nInst = 150, nBg = 50) {
  const bg = sampleN(data, Math.min(nBg, data.length));
  let bpSum = 0; for (const b of bg) bpSum += predictFn(b);
  const basePred = bpSum / bg.length;
  const insts = sampleN(data, Math.min(nInst, data.length));
  return insts.map(inst => {
    const { shapValues } = localSHAPFast(predictFn, inst, bg, basePred);
    return { featureValue: inst[featureIndex], shapValue: shapValues[featureIndex] };
  });
}

// ── LIME ────────────────────────────────────────────────────────

function weightedRidge(X, y, w, p, lambda) {
  const n = X.length;
  const A = Array.from({ length: p }, () => new Float64Array(p));
  const b = new Float64Array(p);
  for (let i = 0; i < n; i++) { const wi = w[i], xi = X[i]; for (let j = 0; j < p; j++) { const wixj = wi * xi[j]; for (let k = j; k < p; k++) A[j][k] += wixj * xi[k]; b[j] += wixj * y[i]; } }
  // Mirror symmetric entries
  for (let j = 0; j < p; j++) for (let k = 0; k < j; k++) A[j][k] = A[k][j];
  for (let i = 0; i < p; i++) A[i][i] += lambda;
  for (let i = 0; i < p; i++) {
    let mr = i; for (let k = i + 1; k < p; k++) if (Math.abs(A[k][i]) > Math.abs(A[mr][i])) mr = k;
    if (mr !== i) { [A[i], A[mr]] = [A[mr], A[i]]; [b[i], b[mr]] = [b[mr], b[i]]; }
    if (Math.abs(A[i][i]) < 1e-12) continue;
    for (let k = i + 1; k < p; k++) { const f = A[k][i] / A[i][i]; for (let j = i; j < p; j++) A[k][j] -= f * A[i][j]; b[k] -= f * b[i]; }
  }
  const x = new Float64Array(p);
  for (let i = p - 1; i >= 0; i--) { x[i] = b[i]; for (let j = i + 1; j < p; j++) x[i] -= A[i][j] * x[j]; x[i] = Math.abs(A[i][i]) > 1e-12 ? x[i] / A[i][i] : 0; }
  return Array.from(x);
}

export function computeLIME(predictFn, instance, data, featureNames, nPert = 300) {
  const p = instance.length;
  // Pre-compute stds once from column-extracted data
  const stds = new Float64Array(p);
  for (let i = 0; i < p; i++) {
    let sum = 0, sumSq = 0;
    for (let j = 0; j < data.length; j++) { const v = data[j][i]; sum += v; sumSq += v * v; }
    const m = sum / data.length;
    stds[i] = Math.sqrt(sumSq / data.length - m * m) || 1;
  }
  const kw = Math.sqrt(p) * 0.75;
  const X = [[...instance]], Y = [predictFn(instance)], W = [1.0];
  for (let k = 0; k < nPert; k++) {
    const perturbed = new Array(p);
    let distSq = 0;
    for (let i = 0; i < p; i++) { const d = boxMuller() * stds[i] * 0.5; perturbed[i] = instance[i] + d; distSq += d * d / (stds[i] * stds[i]); }
    const pred = predictFn(perturbed);
    const dist = Math.sqrt(distSq) / Math.sqrt(p);
    X.push(perturbed); Y.push(pred); W.push(Math.exp(-(dist * dist) / (2 * kw * kw)));
  }
  const beta = weightedRidge(X, Y, W, p, 0.01);
  const intercept = Y[0] - instance.reduce((s, v, i) => s + v * beta[i], 0);
  const contributions = featureNames.map((f, i) => ({ feature: f, weight: beta[i], contribution: beta[i] * instance[i] }));
  contributions.sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution));
  return { contributions, intercept, prediction: Y[0] };
}

export function computeClassProbabilities(predictFn, instance, data, nPert = 200) {
  const p = instance.length;
  const stds = new Float64Array(p);
  for (let i = 0; i < p; i++) {
    let sum = 0, sumSq = 0;
    for (let j = 0; j < data.length; j++) { const v = data[j][i]; sum += v; sumSq += v * v; }
    const m = sum / data.length;
    stds[i] = Math.sqrt(sumSq / data.length - m * m) || 1;
  }
  const counts = {};
  const p0 = Math.round(predictFn(instance));
  counts[p0] = 1;
  for (let k = 0; k < nPert; k++) {
    const perturbed = new Array(p);
    for (let i = 0; i < p; i++) perturbed[i] = instance[i] + boxMuller() * stds[i] * 0.3;
    const key = String(Math.round(predictFn(perturbed)));
    counts[key] = (counts[key] || 0) + 1;
  }
  const total = nPert + 1;
  return Object.entries(counts).map(([cls, c]) => ({ class: cls, probability: c / total })).sort((a, b) => b.probability - a.probability);
}

// ── Data builders (unchanged logic) ─────────────────────────────

export function buildWaterfallData(shapValues, featureNames, basePred) {
  const items = featureNames.map((f, i) => ({ feature: f, contribution: shapValues[i] }));
  items.sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution));
  let running = basePred;
  for (const d of items) { d.offset = Math.min(running, running + d.contribution); d.width = Math.abs(d.contribution); running += d.contribution; }
  return items;
}

export function buildForceData(shapValues, featureNames, basePred, instancePred) {
  const features = featureNames.map((f, i) => ({ feature: f, value: shapValues[i] }));
  const positive = features.filter(f => f.value >= 0).sort((a, b) => b.value - a.value);
  const negative = features.filter(f => f.value < 0).sort((a, b) => a.value - b.value);
  const totalPos = positive.reduce((s, f) => s + f.value, 0);
  const totalNeg = negative.reduce((s, f) => s + Math.abs(f.value), 0);
  return { positive, negative, totalPos, totalNeg, basePred, instancePred };
}
