// explainableAI.js — Client-side Explainable AI Engine
// Implements approximate SHAP (marginal contribution) and LIME for in-browser model explanations

function mean(arr) { return arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : 0; }

function sampleN(arr, n) {
  if (n >= arr.length) return [...arr];
  const res = [], used = new Set();
  while (res.length < n) { const i = Math.floor(Math.random() * arr.length); if (!used.has(i)) { used.add(i); res.push(arr[i]); } }
  return res;
}

function boxMuller() {
  let u = 0, v = 0;
  while (!u) u = Math.random();
  while (!v) v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function euclidean(a, b) {
  let s = 0; for (let i = 0; i < a.length; i++) s += (a[i] - b[i]) ** 2; return Math.sqrt(s);
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

// ── SHAP (marginal contribution approximation) ────────────────

export function computeLocalSHAP(predictFn, instance, background, nSamples = 80) {
  const p = instance.length;
  const bg = sampleN(background, Math.min(nSamples, background.length));
  const instPred = predictFn(instance);
  const basePred = mean(bg.map(b => predictFn(b)));
  const sv = new Array(p).fill(0);
  for (let f = 0; f < p; f++) {
    let total = 0;
    for (const bgRow of bg) { const m = [...instance]; m[f] = bgRow[f]; total += instPred - predictFn(m); }
    sv[f] = total / bg.length;
  }
  const ss = sv.reduce((a, b) => a + b, 0), target = instPred - basePred;
  if (Math.abs(ss) > 1e-10) { const sc = target / ss; for (let f = 0; f < p; f++) sv[f] *= sc; }
  return { shapValues: sv, basePred, instancePred: instPred };
}

export function computeGlobalSHAP(predictFn, data, featureNames, nInst = 50, nBg = 50) {
  const insts = sampleN(data, Math.min(nInst, data.length));
  const bg = sampleN(data, Math.min(nBg, data.length));
  const abs = new Array(featureNames.length).fill(0);
  for (const inst of insts) {
    const { shapValues } = computeLocalSHAP(predictFn, inst, bg, nBg);
    for (let i = 0; i < featureNames.length; i++) abs[i] += Math.abs(shapValues[i]);
  }
  const imp = featureNames.map((f, i) => ({ feature: f, importance: abs[i] / insts.length }));
  imp.sort((a, b) => b.importance - a.importance);
  return { importance: imp, basePred: mean(bg.map(b => predictFn(b))) };
}

export function computeBeeswarmData(predictFn, data, featureNames, nInst = 100, nBg = 50) {
  const insts = sampleN(data, Math.min(nInst, data.length));
  const bg = sampleN(data, Math.min(nBg, data.length));
  const ranges = featureNames.map((_, i) => {
    const vals = data.map(d => d[i]); return { min: Math.min(...vals), max: Math.max(...vals) };
  });
  const pts = [];
  for (const inst of insts) {
    const { shapValues } = computeLocalSHAP(predictFn, inst, bg, nBg);
    for (let fi = 0; fi < featureNames.length; fi++) {
      const r = ranges[fi]; const norm = r.max > r.min ? (inst[fi] - r.min) / (r.max - r.min) : 0.5;
      pts.push({ feature: featureNames[fi], featureIdx: fi, shapValue: shapValues[fi], featureValue: inst[fi], normalizedValue: norm, color: valueToColor(norm), jitter: (Math.random() - 0.5) * 0.35 });
    }
  }
  return { points: pts, ranges };
}

export function computeDependenceData(predictFn, data, featureNames, featureIndex, nInst = 150, nBg = 50) {
  const insts = sampleN(data, Math.min(nInst, data.length));
  const bg = sampleN(data, Math.min(nBg, data.length));
  return insts.map(inst => {
    const { shapValues } = computeLocalSHAP(predictFn, inst, bg, nBg);
    return { featureValue: inst[featureIndex], shapValue: shapValues[featureIndex] };
  });
}

// ── LIME ────────────────────────────────────────────────────────

function weightedRidge(X, y, w, p, lambda) {
  const n = X.length;
  const A = Array.from({ length: p }, () => new Float64Array(p));
  const b = new Float64Array(p);
  for (let i = 0; i < n; i++) for (let j = 0; j < p; j++) { for (let k = 0; k < p; k++) A[j][k] += w[i] * X[i][j] * X[i][k]; b[j] += w[i] * X[i][j] * y[i]; }
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
  const stds = featureNames.map((_, i) => { const v = data.map(d => d[i]), m = mean(v); return Math.sqrt(v.reduce((s, x) => s + (x - m) ** 2, 0) / v.length) || 1; });
  const kw = Math.sqrt(p) * 0.75;
  const X = [[...instance]], Y = [predictFn(instance)], W = [1.0];
  for (let k = 0; k < nPert; k++) {
    const perturbed = instance.map((v, i) => v + boxMuller() * stds[i] * 0.5);
    const pred = predictFn(perturbed);
    const dist = euclidean(perturbed, instance) / Math.sqrt(p);
    X.push(perturbed); Y.push(pred); W.push(Math.exp(-(dist ** 2) / (2 * kw ** 2)));
  }
  const beta = weightedRidge(X, Y, W, p, 0.01);
  const intercept = Y[0] - instance.reduce((s, v, i) => s + v * beta[i], 0);
  const contributions = featureNames.map((f, i) => ({ feature: f, weight: beta[i], contribution: beta[i] * instance[i] }));
  contributions.sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution));
  return { contributions, intercept, prediction: Y[0] };
}

export function computeClassProbabilities(predictFn, instance, data, nPert = 200) {
  const stds = instance.map((_, i) => { const v = data.map(d => d[i]), m = mean(v); return Math.sqrt(v.reduce((s, x) => s + (x - m) ** 2, 0) / v.length) || 1; });
  const preds = [predictFn(instance)];
  for (let k = 0; k < nPert; k++) {
    const perturbed = instance.map((v, i) => v + boxMuller() * stds[i] * 0.3);
    preds.push(predictFn(perturbed));
  }
  const counts = {}; preds.forEach(p => { const k = String(Math.round(p)); counts[k] = (counts[k] || 0) + 1; });
  const total = preds.length;
  return Object.entries(counts).map(([cls, c]) => ({ class: cls, probability: c / total })).sort((a, b) => b.probability - a.probability);
}

// ── Waterfall data builder ──────────────────────────────────────

export function buildWaterfallData(shapValues, featureNames, basePred) {
  const items = featureNames.map((f, i) => ({ feature: f, contribution: shapValues[i] }));
  items.sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution));
  let running = basePred;
  items.forEach(d => { d.offset = Math.min(running, running + d.contribution); d.width = Math.abs(d.contribution); running += d.contribution; });
  return items;
}

// ── Force plot data builder ─────────────────────────────────────

export function buildForceData(shapValues, featureNames, basePred, instancePred) {
  const features = featureNames.map((f, i) => ({ feature: f, value: shapValues[i] }));
  const positive = features.filter(f => f.value >= 0).sort((a, b) => b.value - a.value);
  const negative = features.filter(f => f.value < 0).sort((a, b) => a.value - b.value);
  const totalPos = positive.reduce((s, f) => s + f.value, 0);
  const totalNeg = negative.reduce((s, f) => s + Math.abs(f.value), 0);
  return { positive, negative, totalPos, totalNeg, basePred, instancePred };
}
