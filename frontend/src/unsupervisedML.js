// ==================== UNSUPERVISED ML ENGINE ====================
// All algorithms run 100% client-side in the browser

// ==================== UTILITIES ====================

function euclidean(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += (a[i] - b[i]) ** 2;
  return Math.sqrt(s);
}

function euclideanSq(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += (a[i] - b[i]) ** 2;
  return s;
}

// ==================== DATA PREPROCESSING ====================

export function prepareUnsupervisedData(rows, numericCols) {
  const n = rows.length, p = numericCols.length;
  const rawData = rows.map(row =>
    numericCols.map(col => {
      const v = row[col];
      return typeof v === 'number' && !isNaN(v) ? v : null;
    })
  );
  const medians = numericCols.map((_, j) => {
    const vals = rawData.map(r => r[j]).filter(v => v !== null).sort((a, b) => a - b);
    return vals.length > 0 ? vals[Math.floor(vals.length / 2)] : 0;
  });
  let missingFilled = 0;
  const filled = rawData.map(row =>
    row.map((v, j) => { if (v === null) { missingFilled++; return medians[j]; } return v; })
  );
  const means = new Array(p).fill(0);
  const stds = new Array(p).fill(1);
  for (let j = 0; j < p; j++) {
    for (let i = 0; i < n; i++) means[j] += filled[i][j];
    means[j] /= n;
    let s = 0;
    for (let i = 0; i < n; i++) s += (filled[i][j] - means[j]) ** 2;
    stds[j] = Math.sqrt(s / n) || 1;
  }
  const X = filled.map(row => row.map((v, j) => (v - means[j]) / stds[j]));
  return { X, raw: filled, means, stds, n, p, missingFilled, featureNames: numericCols };
}

// ==================== K-MEANS ====================

export function runKMeans(X, k, maxIter = 100) {
  const n = X.length, p = X[0].length;
  // K-means++ init
  const centroids = [X[Math.floor(Math.random() * n)].slice()];
  for (let c = 1; c < k; c++) {
    const dists = X.map(x => Math.min(...centroids.map(cen => euclideanSq(x, cen))));
    const total = dists.reduce((a, b) => a + b, 0);
    let r = Math.random() * total, acc = 0;
    for (let i = 0; i < n; i++) { acc += dists[i]; if (acc >= r) { centroids.push(X[i].slice()); break; } }
    if (centroids.length <= c) centroids.push(X[Math.floor(Math.random() * n)].slice());
  }
  let labels = new Array(n).fill(0);
  for (let iter = 0; iter < maxIter; iter++) {
    const newLabels = X.map(x => {
      let bestD = Infinity, bestC = 0;
      for (let c = 0; c < k; c++) { const d = euclideanSq(x, centroids[c]); if (d < bestD) { bestD = d; bestC = c; } }
      return bestC;
    });
    const counts = new Array(k).fill(0);
    const sums = centroids.map(() => new Array(p).fill(0));
    for (let i = 0; i < n; i++) { counts[newLabels[i]]++; for (let j = 0; j < p; j++) sums[newLabels[i]][j] += X[i][j]; }
    for (let c = 0; c < k; c++) { if (counts[c] > 0) { for (let j = 0; j < p; j++) centroids[c][j] = sums[c][j] / counts[c]; } }
    let changed = false;
    for (let i = 0; i < n; i++) { if (newLabels[i] !== labels[i]) { changed = true; break; } }
    labels = newLabels;
    if (!changed) break;
  }
  let inertiaVal = 0;
  for (let i = 0; i < n; i++) inertiaVal += euclideanSq(X[i], centroids[labels[i]]);
  return { labels, centroids, k, inertia: inertiaVal };
}

// ==================== HIERARCHICAL CLUSTERING ====================

export function runHierarchical(X, k) {
  const n = X.length;
  // Compute distance matrix
  const dist = Array.from({ length: n }, () => new Float64Array(n));
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) { const d = euclidean(X[i], X[j]); dist[i][j] = d; dist[j][i] = d; }
  }
  const clusterMap = Array.from({ length: n }, (_, i) => [i]);
  const active = new Set(Array.from({ length: n }, (_, i) => i));

  while (active.size > k) {
    let bestI = -1, bestJ = -1, bestDist = Infinity;
    const activeArr = [...active];
    for (let ai = 0; ai < activeArr.length; ai++) {
      for (let aj = ai + 1; aj < activeArr.length; aj++) {
        const ci = activeArr[ai], cj = activeArr[aj];
        let totalDist = 0, count = 0;
        for (const pi of clusterMap[ci]) { for (const pj of clusterMap[cj]) { totalDist += dist[pi][pj]; count++; } }
        const avgDist = count > 0 ? totalDist / count : Infinity;
        if (avgDist < bestDist) { bestDist = avgDist; bestI = ci; bestJ = cj; }
      }
    }
    if (bestI === -1) break;
    clusterMap[bestI] = [...clusterMap[bestI], ...clusterMap[bestJ]];
    active.delete(bestJ);
  }

  const labels = new Array(n).fill(0);
  let clusterIdx = 0;
  const centroids = [];
  for (const ci of active) {
    const members = clusterMap[ci];
    const centroid = new Array(X[0].length).fill(0);
    for (const pi of members) { labels[pi] = clusterIdx; for (let j = 0; j < X[0].length; j++) centroid[j] += X[pi][j]; }
    for (let j = 0; j < X[0].length; j++) centroid[j] /= members.length;
    centroids.push(centroid);
    clusterIdx++;
  }
  return { labels, centroids, k: centroids.length };
}

// ==================== DBSCAN ====================

export function runDBSCAN(X, eps, minPts) {
  const n = X.length;
  const labels = new Array(n).fill(-1);
  let clusterId = 0;
  if (eps == null) {
    const kDists = [];
    for (let i = 0; i < n; i++) {
      const dists = [];
      for (let j = 0; j < n; j++) { if (i !== j) dists.push(euclidean(X[i], X[j])); }
      dists.sort((a, b) => a - b);
      kDists.push(dists[Math.min(minPts - 1, dists.length - 1)] || 0);
    }
    kDists.sort((a, b) => a - b);
    eps = kDists[Math.floor(n * 0.85)] || 1;
  }
  function regionQuery(idx) {
    const neighbors = [];
    for (let j = 0; j < n; j++) { if (euclidean(X[idx], X[j]) <= eps) neighbors.push(j); }
    return neighbors;
  }
  const visited = new Set();
  for (let i = 0; i < n; i++) {
    if (visited.has(i)) continue;
    visited.add(i);
    const neighbors = regionQuery(i);
    if (neighbors.length < minPts) { continue; }
    labels[i] = clusterId;
    const queue = [...neighbors.filter(j => j !== i)];
    const inQueue = new Set(queue);
    while (queue.length > 0) {
      const j = queue.shift();
      if (!visited.has(j)) {
        visited.add(j);
        const jNeighbors = regionQuery(j);
        if (jNeighbors.length >= minPts) {
          for (const nn of jNeighbors) { if (!inQueue.has(nn) && labels[nn] === -1) { queue.push(nn); inQueue.add(nn); } }
        }
      }
      if (labels[j] === -1) labels[j] = clusterId;
    }
    clusterId++;
  }
  const numClusters = clusterId;
  const centroids = [];
  for (let c = 0; c < numClusters; c++) {
    const members = labels.reduce((arr, l, i) => { if (l === c) arr.push(i); return arr; }, []);
    const centroid = new Array(X[0].length).fill(0);
    for (const pi of members) { for (let j = 0; j < X[0].length; j++) centroid[j] += X[pi][j]; }
    if (members.length > 0) { for (let j = 0; j < X[0].length; j++) centroid[j] /= members.length; }
    centroids.push(centroid);
  }
  return { labels, centroids, k: numClusters, noiseCount: labels.filter(l => l === -1).length, eps, minPts };
}

// ==================== GAUSSIAN MIXTURE MODEL ====================

export function runGMM(X, k, maxIter = 50) {
  const n = X.length, p = X[0].length;
  const kmResult = runKMeans(X, k);
  const gmMeans = kmResult.centroids.map(c => [...c]);
  const variances = gmMeans.map(() => new Array(p).fill(1));
  const weights = new Array(k).fill(1 / k);
  const resp = Array.from({ length: n }, () => new Array(k).fill(0));

  for (let iter = 0; iter < maxIter; iter++) {
    // E-step
    for (let i = 0; i < n; i++) {
      let total = 0;
      for (let c = 0; c < k; c++) {
        let logP = Math.log(weights[c] + 1e-300);
        for (let j = 0; j < p; j++) {
          const v = Math.max(variances[c][j], 1e-6);
          logP += -0.5 * Math.log(2 * Math.PI * v) - (X[i][j] - gmMeans[c][j]) ** 2 / (2 * v);
        }
        resp[i][c] = Math.exp(Math.min(logP, 500));
        total += resp[i][c];
      }
      if (total > 0) { for (let c = 0; c < k; c++) resp[i][c] /= total; }
      else { for (let c = 0; c < k; c++) resp[i][c] = 1 / k; }
    }
    // M-step
    for (let c = 0; c < k; c++) {
      const Nc = resp.reduce((s, r) => s + r[c], 0);
      if (Nc < 1e-6) continue;
      weights[c] = Nc / n;
      for (let j = 0; j < p; j++) {
        gmMeans[c][j] = 0;
        for (let i = 0; i < n; i++) gmMeans[c][j] += resp[i][c] * X[i][j];
        gmMeans[c][j] /= Nc;
      }
      for (let j = 0; j < p; j++) {
        variances[c][j] = 0;
        for (let i = 0; i < n; i++) variances[c][j] += resp[i][c] * (X[i][j] - gmMeans[c][j]) ** 2;
        variances[c][j] = Math.max(variances[c][j] / Nc, 1e-6);
      }
    }
  }
  const labels = resp.map(r => { let best = 0; for (let c = 1; c < k; c++) { if (r[c] > r[best]) best = c; } return best; });
  return { labels, centroids: gmMeans, k, weights };
}

// ==================== PCA ====================

export function runPCA(X, nComponents = 2) {
  const n = X.length, p = X[0].length;
  const cov = Array.from({ length: p }, () => new Array(p).fill(0));
  for (let i = 0; i < p; i++) {
    for (let j = i; j < p; j++) {
      let s = 0;
      for (let k = 0; k < n; k++) s += X[k][i] * X[k][j];
      cov[i][j] = s / (n - 1 || 1);
      cov[j][i] = cov[i][j];
    }
  }
  const eigenvectors = [];
  const eigenvalues = [];
  const covCopy = cov.map(row => [...row]);
  for (let comp = 0; comp < Math.min(nComponents, p); comp++) {
    let v = Array.from({ length: p }, () => Math.random() - 0.5);
    let norm = Math.sqrt(v.reduce((s, x) => s + x * x, 0));
    v = v.map(x => x / (norm || 1));
    for (let iter = 0; iter < 200; iter++) {
      const newV = new Array(p).fill(0);
      for (let i = 0; i < p; i++) { for (let j = 0; j < p; j++) newV[i] += covCopy[i][j] * v[j]; }
      norm = Math.sqrt(newV.reduce((s, x) => s + x * x, 0));
      if (norm < 1e-10) break;
      v = newV.map(x => x / norm);
    }
    const Av = new Array(p).fill(0);
    for (let i = 0; i < p; i++) { for (let j = 0; j < p; j++) Av[i] += covCopy[i][j] * v[j]; }
    const eigenvalue = v.reduce((s, x, i) => s + x * Av[i], 0);
    eigenvectors.push([...v]);
    eigenvalues.push(eigenvalue);
    for (let i = 0; i < p; i++) { for (let j = 0; j < p; j++) covCopy[i][j] -= eigenvalue * v[i] * v[j]; }
  }
  const projected = X.map(x => eigenvectors.map(ev => ev.reduce((s, v, i) => s + v * x[i], 0)));
  const totalVar = eigenvalues.reduce((s, v) => s + Math.max(0, v), 0);
  const explainedVariance = eigenvalues.map(v => Math.max(0, v) / (totalVar || 1));
  return { projected, eigenvectors, eigenvalues, explainedVariance, nComponents };
}

// ==================== t-SNE ====================

export function runTSNE(X, nComponents = 2, perplexity = 30, maxIter = 250, lr = 100) {
  const n = X.length;
  const perp = Math.min(perplexity, Math.max(2, Math.floor(n / 3)));
  const distSq = Array.from({ length: n }, () => new Float64Array(n));
  for (let i = 0; i < n; i++) { for (let j = i + 1; j < n; j++) { const d = euclideanSq(X[i], X[j]); distSq[i][j] = d; distSq[j][i] = d; } }
  const P = Array.from({ length: n }, () => new Float64Array(n));
  const targetH = Math.log(perp);
  for (let i = 0; i < n; i++) {
    let lo = 1e-10, hi = 1e4;
    for (let iter = 0; iter < 50; iter++) {
      const sigma = (lo + hi) / 2;
      let sumP = 0;
      for (let j = 0; j < n; j++) { if (j !== i) { P[i][j] = Math.exp(-distSq[i][j] / (2 * sigma * sigma)); sumP += P[i][j]; } }
      if (sumP > 0) { for (let j = 0; j < n; j++) { if (j !== i) P[i][j] /= sumP; } }
      let h = 0;
      for (let j = 0; j < n; j++) { if (j !== i && P[i][j] > 1e-10) h -= P[i][j] * Math.log(P[i][j]); }
      if (h > targetH) hi = sigma; else lo = sigma;
      if (Math.abs(h - targetH) < 1e-5) break;
    }
  }
  for (let i = 0; i < n; i++) { for (let j = i + 1; j < n; j++) { const sym = (P[i][j] + P[j][i]) / (2 * n); P[i][j] = sym; P[j][i] = sym; } }
  const Y = Array.from({ length: n }, () => Array.from({ length: nComponents }, () => (Math.random() - 0.5) * 0.01));
  const vel = Array.from({ length: n }, () => new Array(nComponents).fill(0));
  for (let iter = 0; iter < maxIter; iter++) {
    let sumQ = 0;
    const qij = Array.from({ length: n }, () => new Float64Array(n));
    for (let i = 0; i < n; i++) { for (let j = i + 1; j < n; j++) { let d = 0; for (let dd = 0; dd < nComponents; dd++) d += (Y[i][dd] - Y[j][dd]) ** 2; const q = 1 / (1 + d); qij[i][j] = q; qij[j][i] = q; sumQ += 2 * q; } }
    if (sumQ > 0) { for (let i = 0; i < n; i++) { for (let j = 0; j < n; j++) qij[i][j] /= sumQ; } }
    const grad = Array.from({ length: n }, () => new Array(nComponents).fill(0));
    for (let i = 0; i < n; i++) { for (let j = 0; j < n; j++) { if (i === j) continue; const mult = 4 * (P[i][j] - qij[i][j]) * (1 / (1 + euclideanSq(Y[i], Y[j]))); for (let d = 0; d < nComponents; d++) grad[i][d] += mult * (Y[i][d] - Y[j][d]); } }
    const mom = iter < 200 ? 0.5 : 0.8;
    for (let i = 0; i < n; i++) { for (let d = 0; d < nComponents; d++) { vel[i][d] = mom * vel[i][d] - lr * grad[i][d]; Y[i][d] += vel[i][d]; } }
    const center = new Array(nComponents).fill(0);
    for (let i = 0; i < n; i++) { for (let d = 0; d < nComponents; d++) center[d] += Y[i][d]; }
    for (let i = 0; i < n; i++) { for (let d = 0; d < nComponents; d++) Y[i][d] -= center[d] / n; }
  }
  return { projected: Y };
}

// ==================== ISOLATION FOREST ====================

export function runIsolationForest(X, nTrees = 100, sampleSize = 256, contamination = 0.1) {
  const n = X.length, p = X[0].length;
  const ss = Math.min(sampleSize, n);
  const maxD = Math.ceil(Math.log2(ss));
  function buildTree(indices, depth) {
    if (indices.length <= 1 || depth >= maxD) return { leaf: true, size: indices.length };
    const feat = Math.floor(Math.random() * p);
    const vals = indices.map(i => X[i][feat]);
    const mn = Math.min(...vals), mx = Math.max(...vals);
    if (mn === mx) return { leaf: true, size: indices.length };
    const split = mn + Math.random() * (mx - mn);
    return { leaf: false, feat, split, left: buildTree(indices.filter(i => X[i][feat] < split), depth + 1), right: buildTree(indices.filter(i => X[i][feat] >= split), depth + 1) };
  }
  function pathLen(x, tree, d) {
    if (tree.leaf) { const c = tree.size > 1 ? 2 * (Math.log(tree.size - 1) + 0.5772) - 2 * (tree.size - 1) / tree.size : 0; return d + c; }
    return x[tree.feat] < tree.split ? pathLen(x, tree.left, d + 1) : pathLen(x, tree.right, d + 1);
  }
  const trees = [];
  for (let t = 0; t < nTrees; t++) {
    const idx = []; for (let i = 0; i < ss; i++) idx.push(Math.floor(Math.random() * n));
    trees.push(buildTree(idx, 0));
  }
  const cn = ss > 1 ? 2 * (Math.log(ss - 1) + 0.5772) - 2 * (ss - 1) / ss : 1;
  const scores = X.map(x => { const avg = trees.reduce((s, t) => s + pathLen(x, t, 0), 0) / nTrees; return Math.pow(2, -avg / cn); });
  const sorted = [...scores].sort((a, b) => b - a);
  const threshold = sorted[Math.max(0, Math.floor(n * contamination) - 1)] || 0.5;
  const labels = scores.map(s => s >= threshold ? -1 : 1);
  return { scores, labels, threshold, nAnomalies: labels.filter(l => l === -1).length };
}

// ==================== LOCAL OUTLIER FACTOR ====================

export function runLOF(X, k = 20, contamination = 0.1) {
  const n = X.length;
  k = Math.min(k, n - 1);
  const dists = Array.from({ length: n }, (_, i) => {
    const d = [];
    for (let j = 0; j < n; j++) { if (i !== j) d.push({ idx: j, dist: euclidean(X[i], X[j]) }); }
    d.sort((a, b) => a.dist - b.dist);
    return d;
  });
  const kDist = dists.map(d => d[Math.min(k - 1, d.length - 1)]?.dist || 0);
  const lrd = new Array(n).fill(0);
  for (let i = 0; i < n; i++) {
    const neighbors = dists[i].slice(0, k);
    let sum = 0;
    for (const nb of neighbors) sum += Math.max(kDist[nb.idx], euclidean(X[i], X[nb.idx]));
    lrd[i] = sum > 0 ? k / sum : 1;
  }
  const lofScores = new Array(n).fill(0);
  for (let i = 0; i < n; i++) {
    const neighbors = dists[i].slice(0, k);
    let sum = 0;
    for (const nb of neighbors) sum += lrd[nb.idx];
    lofScores[i] = (sum / k) / (lrd[i] || 1e-10);
  }
  const sorted = [...lofScores].sort((a, b) => b - a);
  const threshold = sorted[Math.max(0, Math.floor(n * contamination) - 1)] || 1.5;
  const labels = lofScores.map(s => s >= threshold ? -1 : 1);
  return { scores: lofScores, labels, threshold, nAnomalies: labels.filter(l => l === -1).length };
}

// ==================== EVALUATION METRICS ====================

export function silhouetteScore(X, labels) {
  const n = X.length;
  const clusters = [...new Set(labels.filter(l => l >= 0))];
  if (clusters.length < 2 || clusters.length >= n) return 0;
  let totalSil = 0, validCount = 0;
  for (let i = 0; i < n; i++) {
    if (labels[i] < 0) continue;
    const same = []; for (let j = 0; j < n; j++) { if (j !== i && labels[j] === labels[i]) same.push(j); }
    if (same.length === 0) continue;
    const a = same.reduce((s, j) => s + euclidean(X[i], X[j]), 0) / same.length;
    let b = Infinity;
    for (const c of clusters) {
      if (c === labels[i]) continue;
      const other = []; for (let j = 0; j < n; j++) { if (labels[j] === c) other.push(j); }
      if (other.length === 0) continue;
      const avg = other.reduce((s, j) => s + euclidean(X[i], X[j]), 0) / other.length;
      if (avg < b) b = avg;
    }
    totalSil += b > a ? (b - a) / Math.max(a, b) : 0;
    validCount++;
  }
  return validCount > 0 ? totalSil / validCount : 0;
}

export function daviesBouldinIndex(X, labels) {
  const clusters = [...new Set(labels.filter(l => l >= 0))];
  if (clusters.length < 2) return Infinity;
  const centroids = {}, avgDists = {};
  for (const c of clusters) {
    const members = []; for (let i = 0; i < X.length; i++) { if (labels[i] === c) members.push(i); }
    const cent = new Array(X[0].length).fill(0);
    for (const m of members) { for (let j = 0; j < X[0].length; j++) cent[j] += X[m][j]; }
    for (let j = 0; j < X[0].length; j++) cent[j] /= members.length;
    centroids[c] = cent;
    avgDists[c] = members.reduce((s, m) => s + euclidean(X[m], cent), 0) / members.length;
  }
  let db = 0;
  for (const ci of clusters) {
    let maxR = 0;
    for (const cj of clusters) {
      if (ci === cj) continue;
      const d = euclidean(centroids[ci], centroids[cj]);
      if (d > 0) { const r = (avgDists[ci] + avgDists[cj]) / d; if (r > maxR) maxR = r; }
    }
    db += maxR;
  }
  return db / clusters.length;
}

export function calinskiHarabaszScore(X, labels) {
  const n = X.length, p = X[0].length;
  const clusters = [...new Set(labels.filter(l => l >= 0))];
  const kk = clusters.length;
  if (kk < 2 || kk >= n) return 0;
  const oc = new Array(p).fill(0);
  let vn = 0;
  for (let i = 0; i < n; i++) { if (labels[i] < 0) continue; vn++; for (let j = 0; j < p; j++) oc[j] += X[i][j]; }
  for (let j = 0; j < p; j++) oc[j] /= (vn || 1);
  let bgss = 0, wgss = 0;
  for (const c of clusters) {
    const members = []; for (let i = 0; i < n; i++) { if (labels[i] === c) members.push(i); }
    const cent = new Array(p).fill(0);
    for (const m of members) { for (let j = 0; j < p; j++) cent[j] += X[m][j]; }
    for (let j = 0; j < p; j++) cent[j] /= members.length;
    bgss += members.length * euclideanSq(cent, oc);
    for (const m of members) wgss += euclideanSq(X[m], cent);
  }
  if (wgss === 0) return 0;
  return (bgss / (kk - 1)) / (wgss / (vn - kk));
}

// ==================== OPTIMAL K DETECTION ====================

export function findOptimalK(X, maxK = 10) {
  maxK = Math.min(maxK, X.length - 1, 10);
  const results = [];
  for (let k = 2; k <= maxK; k++) {
    const km = runKMeans(X, k);
    results.push({ k, inertia: km.inertia, silhouette: silhouetteScore(X, km.labels), daviesBouldin: daviesBouldinIndex(X, km.labels), calinskiHarabasz: calinskiHarabaszScore(X, km.labels) });
  }
  let bestK = 2, bestSil = -1;
  for (const r of results) { if (r.silhouette > bestSil) { bestSil = r.silhouette; bestK = r.k; } }
  return { results, bestK };
}

// ==================== CLUSTER INTERPRETATION ====================

export function interpretClusters(rawData, labels, featureNames) {
  const clusters = [...new Set(labels.filter(l => l >= 0))].sort((a, b) => a - b);
  const n = rawData.length, p = featureNames.length;
  const overallAvg = new Array(p).fill(0);
  let vn = 0;
  for (let i = 0; i < n; i++) { if (labels[i] < 0) continue; vn++; for (let j = 0; j < p; j++) overallAvg[j] += rawData[i][j]; }
  for (let j = 0; j < p; j++) overallAvg[j] /= (vn || 1);

  const interpretations = clusters.map(c => {
    const members = []; for (let i = 0; i < n; i++) { if (labels[i] === c) members.push(i); }
    const avg = new Array(p).fill(0);
    for (const m of members) { for (let j = 0; j < p; j++) avg[j] += rawData[m][j]; }
    for (let j = 0; j < p; j++) avg[j] /= (members.length || 1);
    const deviations = featureNames.map((name, j) => ({
      feature: name, clusterAvg: avg[j], overallAvg: overallAvg[j],
      deviation: overallAvg[j] !== 0 ? ((avg[j] - overallAvg[j]) / Math.abs(overallAvg[j])) * 100 : 0,
      direction: avg[j] > overallAvg[j] ? 'higher' : 'lower'
    })).sort((a, b) => Math.abs(b.deviation) - Math.abs(a.deviation));
    const keyFeatures = deviations.slice(0, 3);
    const parts = keyFeatures.filter(f => Math.abs(f.deviation) > 5).map(f =>
      `${f.direction} than average ${f.feature} (${f.clusterAvg.toFixed(1)} vs ${f.overallAvg.toFixed(1)})`
    );
    const interpretation = parts.length > 0
      ? `Cluster ${c} represents ${members.length} data points with ${parts.join(', ')}.`
      : `Cluster ${c} has ${members.length} points with feature values close to the dataset average.`;
    return { clusterId: c, size: members.length, featureAverages: featureNames.map((name, j) => ({ feature: name, value: avg[j] })), keyFeatures, interpretation };
  });
  return { interpretations, overallAvg: featureNames.map((name, j) => ({ feature: name, value: overallAvg[j] })) };
}

// ==================== CLUSTER PREDICTION ====================

export function predictCluster(point, centroids, interpretations) {
  let bestC = 0, bestD = Infinity;
  for (let c = 0; c < centroids.length; c++) {
    const d = euclidean(point, centroids[c]);
    if (d < bestD) { bestD = d; bestC = c; }
  }
  return { cluster: bestC, distance: bestD, centroid: centroids[bestC], interpretation: interpretations?.find(i => i.clusterId === bestC) };
}

// ==================== FULL PIPELINE ====================

export function runUnsupervisedPipeline(rows, numericCols) {
  const t0 = performance.now();
  const prepared = prepareUnsupervisedData(rows, numericCols);
  const { X, raw, n, p, missingFilled, featureNames } = prepared;
  if (n < 4 || p < 1) throw new Error('Need at least 4 rows and 1 numeric feature');

  const optimalResult = findOptimalK(X, Math.min(10, n - 1));
  const bestK = optimalResult.bestK;
  const pcaResult = runPCA(X, 2);
  let tsneResult = null;
  if (n <= 300) { try { tsneResult = runTSNE(X, 2, Math.min(30, Math.max(2, Math.floor(n / 3))), Math.min(250, n * 2)); } catch { tsneResult = null; } }

  const algorithms = [];

  // K-Means
  let t1 = performance.now();
  const km = runKMeans(X, bestK);
  algorithms.push({ name: 'K-Means', key: 'kmeans', labels: km.labels, centroids: km.centroids, k: bestK,
    metrics: { silhouette: silhouetteScore(X, km.labels), daviesBouldin: daviesBouldinIndex(X, km.labels), calinskiHarabasz: calinskiHarabaszScore(X, km.labels), inertia: km.inertia },
    runtime: (performance.now() - t1) / 1000 });

  // Hierarchical
  if (n <= 500) {
    t1 = performance.now();
    try { const h = runHierarchical(X, bestK);
      algorithms.push({ name: 'Hierarchical', key: 'hierarchical', labels: h.labels, centroids: h.centroids, k: h.k,
        metrics: { silhouette: silhouetteScore(X, h.labels), daviesBouldin: daviesBouldinIndex(X, h.labels), calinskiHarabasz: calinskiHarabaszScore(X, h.labels) },
        runtime: (performance.now() - t1) / 1000 }); } catch { /* skip */ }
  }

  // DBSCAN
  t1 = performance.now();
  try { const db = runDBSCAN(X, null, Math.max(3, Math.min(5, Math.floor(p * 1.5))));
    if (db.k > 1) { algorithms.push({ name: 'DBSCAN', key: 'dbscan', labels: db.labels, centroids: db.centroids, k: db.k, noiseCount: db.noiseCount,
      metrics: { silhouette: silhouetteScore(X, db.labels), daviesBouldin: daviesBouldinIndex(X, db.labels), calinskiHarabasz: calinskiHarabaszScore(X, db.labels) },
      runtime: (performance.now() - t1) / 1000 }); }
  } catch { /* skip */ }

  // GMM
  t1 = performance.now();
  try { const gm = runGMM(X, bestK);
    algorithms.push({ name: 'Gaussian Mixture', key: 'gmm', labels: gm.labels, centroids: gm.centroids, k: bestK,
      metrics: { silhouette: silhouetteScore(X, gm.labels), daviesBouldin: daviesBouldinIndex(X, gm.labels), calinskiHarabasz: calinskiHarabaszScore(X, gm.labels) },
      runtime: (performance.now() - t1) / 1000 }); } catch { /* skip */ }

  algorithms.sort((a, b) => (b.metrics.silhouette || 0) - (a.metrics.silhouette || 0));
  const best = algorithms[0];

  let iforest = null, lof = null;
  try { iforest = runIsolationForest(X); } catch { /* skip */ }
  try { lof = runLOF(X, Math.min(20, n - 1)); } catch { /* skip */ }

  const interpretation = best ? interpretClusters(raw, best.labels, featureNames) : null;
  const pcaPoints = pcaResult.projected.map((pt, i) => ({ x: pt[0], y: pt[1] || 0, cluster: best?.labels[i] ?? 0, index: i }));
  const tsnePoints = tsneResult ? tsneResult.projected.map((pt, i) => ({ x: pt[0], y: pt[1] || 0, cluster: best?.labels[i] ?? 0, index: i })) : null;

  // Anomaly overlay points
  const anomalyPoints = iforest ? pcaResult.projected.map((pt, i) => ({
    x: pt[0], y: pt[1] || 0, anomaly: iforest.labels[i] === -1, score: iforest.scores[i], index: i
  })) : null;

  return {
    preprocessing: { n, p: featureNames.length, featureNames, missingFilled, scalingApplied: 'Standardization (z-score)' },
    optimalK: optimalResult, algorithms, bestAlgorithm: best,
    pca: { ...pcaResult, points: pcaPoints },
    tsne: tsnePoints ? { points: tsnePoints } : null,
    anomalyDetection: { isolationForest: iforest, lof, points: anomalyPoints },
    interpretation, totalTime: (performance.now() - t0) / 1000,
    means: prepared.means, stds: prepared.stds
  };
}
